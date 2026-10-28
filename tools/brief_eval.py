#!/usr/bin/env python
"""Measure briefing quality on frozen evidence pools.

Three steps, each resumable:

    # 1. Freeze: one full pipeline run per topic. The manifest is the pool
    #    (records, labels, full text) and its article is the shipped baseline.
    python tools/brief_eval.py freeze --topics tools/brief_eval_topics.json --out POOLS

    # 2. Write: replay only the writer + style + figure gates on each pool.
    #    The pool is identical across variants, so the writer is the variable.
    python tools/brief_eval.py write --pools POOLS --out RUNS/variant --model openai/gpt-5.6-luna

    # 3. Judge: a blind rubric score per briefing, and a pairwise preference
    #    between two variants with the order swapped to cancel position bias.
    python tools/brief_eval.py judge --runs RUNS/variant
    python tools/brief_eval.py compare --a RUNS/base --b RUNS/variant

The judge is `cli:opus` by default, so scoring is drawn from a Claude
subscription rather than the OpenRouter bill the writer runs on. It sees the
question, the briefing as a reader sees it (Markdown), and the abstracts and
full-text excerpts of every cited source, so it can check support as well as
read the prose.

A score is a measurement of one judge on one pool. Treat a difference smaller
than about one point on a single topic as noise; the pairwise preference across
topics is the number to believe.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articlegen.llm import generate_json  # noqa: E402
from articlegen.pipeline import (  # noqa: E402
    Draft, enforce_statistics, enforce_style, generate_draft,
)
from articlegen.render import render_article, render_markdown  # noqa: E402
from articlegen.sources import full_text_excerpts  # noqa: E402
from articlegen.style import errors as style_errors  # noqa: E402
from articlegen.writer import is_briefing, write_article, write_briefing  # noqa: E402

JUDGE_MODEL = "cli:opus"

DIMENSIONS = {
    "answer": "The opening answer states plainly what the evidence shows for the "
              "question as asked — direction, rough size and how sure — in a way a "
              "busy reader could act on or forward. Mush ('may', 'appears', 'remains "
              "uncertain' piled up without a verdict) scores low; a crisp, calibrated "
              "verdict scores high.",
    "synthesis": "Findings integrate the evidence: they say where studies agree, "
                 "where they disagree and why, and weigh stronger designs above weaker "
                 "ones. A list of one-study-per-bullet summaries scores low.",
    "specificity": "The decisive numbers are there and interpretable: effect sizes with "
                   "units or comparators, sample sizes, follow-up, absolute differences "
                   "where available. Vague magnitude where the sources give numbers "
                   "scores low; numbers dumped without meaning also score low.",
    "calibration": "Each claim's confidence matches its evidence (design, size, "
                   "consistency, directness). Overstating a single small study scores "
                   "low; hedging a consistent body of RCTs into vagueness also scores low. "
                   "Certainty is visible to the reader, not buried.",
    "faithfulness": "Every factual claim and figure is supported by the cited source "
                    "material shown below, attributed to the right source. Any claim the "
                    "material does not support, or a figure credited to the wrong source, "
                    "is a serious defect.",
    "coverage": "It uses the most important evidence available in the cited material "
                "(the largest or best-designed studies, systematic reviews) and addresses "
                "the main outcomes the question asks about, including harms where relevant.",
    "gaps": "The open questions are specific, decision-relevant and not restatements "
            "of the findings.",
    "readability": "Clear, concise and scannable for a professional: no repetition, no "
                   "padding, defined jargon, sensible order, professional tone.",
    "usefulness": "Overall value to a clinician, policy-maker or researcher with three "
                  "minutes: would they trust it and know more than before?",
}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "overall": {"type": "integer"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "biggest_weakness": {"type": "string"},
        "biggest_strength": {"type": "string"},
    },
    "required": ["scores", "overall", "unsupported_claims", "biggest_weakness",
                 "biggest_strength"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM = """\
You are a senior editor at an evidence-synthesis unit (think Cochrane, NICE \
evidence summaries, a health-technology assessment agency). You grade one-page \
evidence briefings written by a machine from published papers. You are strict \
and specific, and you check claims against the source material you are given.

Score each dimension from 1 to 10:
  1-3  poor: a reader would be misled or would learn little
  4-5  weak: usable only with heavy editing
  6-7  competent: what a careful junior analyst produces
  8-9  strong: what a good evidence-synthesis unit publishes
  10   exemplary: nothing to change

Use the whole scale. Do not default to 7. `overall` is your holistic 1-10 \
grade, not an average. List every claim or figure you could not find support \
for in the cited source material in `unsupported_claims` (quote it briefly), \
or an empty list."""

_PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "margin": {"type": "string", "enum": ["slight", "clear", "strong"]},
        "dimensions": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": ["A", "B", "tie"]}
                           for k in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": ["preferred", "margin", "dimensions", "reason"],
    "additionalProperties": False,
}

_PAIR_SYSTEM = """\
You are a senior editor at an evidence-synthesis unit. Two machine-written \
evidence briefings answer the same question from the same pool of papers. \
Decide which one you would rather publish and send to a busy clinician, \
policy-maker or researcher. Check claims against the source material. A \
briefing that states something the sources do not support loses on \
faithfulness no matter how well it reads. Ignore length for its own sake: \
longer is not better, and shorter is not better. Judge each dimension \
separately, then give your overall preference and how large the margin is."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _load(path: str) -> Draft:
    with open(path, encoding="utf-8") as f:
        return Draft.from_dict(json.load(f))


def _save(draft: Draft, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    args = (draft.article, draft.papers, draft.topic, draft.curation,
            draft.verification, draft.provenance, draft.style_report)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(draft.to_dict(), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, stem + ".html"), "w", encoding="utf-8") as f:
        f.write(render_article(*args))
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(*args))


# ------------------------------------------------------------------ freeze

def cmd_freeze(args) -> None:
    with open(args.topics, encoding="utf-8") as f:
        topics = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    def one(topic: str) -> str:
        stem = _slug(topic)
        if os.path.exists(os.path.join(args.out, stem + ".json")):
            return f"skip {stem}"
        log_path = os.path.join(args.out, stem + ".log")
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log_f:
            def log(m: str) -> None:
                log_f.write(m + "\n")
                log_f.flush()
            draft = generate_draft(topic, model=args.model, log=log)
        _save(draft, args.out, stem)
        return f"done {stem} in {time.time() - t0:.0f}s: {draft.summary()}"

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, t) for t in topics]):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001 - report and carry on
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)


# ------------------------------------------------------------------- write

def replay_write(pool: Draft, model: str, log=lambda m: None) -> Draft:
    """The writer and both gates on a frozen pool. No search, no curation, no
    full-text fetch: the papers already carry the text the baseline saw."""
    papers = copy.deepcopy(pool.papers)
    curation = pool.curation
    writer_fn = write_briefing if is_briefing(pool.article) else write_article
    article = writer_fn(pool.topic, papers, model=model, style_note="",
                        curation=curation, api_key=None)
    article, style_report = enforce_style(
        article, model=model, log=log, papers=papers, curation=curation)
    article, verification, style_report = enforce_statistics(
        article, papers, model=model, log=log, style_report=style_report,
        direct_sources=((curation or {}).get("counts") or {}).get("direct"))
    provenance = dict(pool.provenance)
    provenance["model"] = model
    return Draft(topic=pool.topic, article=article, papers=papers,
                 curation=curation, verification=verification,
                 provenance=provenance, style_report=style_report,
                 excerpts=full_text_excerpts(papers))


def cmd_write(args) -> None:
    pools = sorted(p for p in os.listdir(args.pools) if p.endswith(".json"))
    os.makedirs(args.out, exist_ok=True)

    def one(name: str) -> str:
        stem = name[:-5]
        if os.path.exists(os.path.join(args.out, stem + ".json")):
            return f"skip {stem}"
        pool = _load(os.path.join(args.pools, name))
        lines: list[str] = []
        t0 = time.time()
        draft = replay_write(pool, args.model, log=lines.append)
        _save(draft, args.out, stem)
        with open(os.path.join(args.out, stem + ".log"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"done {stem} in {time.time() - t0:.0f}s: {draft.summary()}"

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, n) for n in pools]):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)


# ----------------------------------------------------------------- metrics

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_HEDGES = ("may ", "might ", "appears", "suggest", "possibly", "uncertain",
           "unclear", "limited", "could ")


def _finding_texts(article: dict) -> list[str]:
    out = []
    for f in article.get("findings") or []:
        if isinstance(f, dict):
            out.append(" ".join(str(v) for v in f.values() if isinstance(v, str)))
        else:
            out.append(str(f))
    return out


def _cites_in(text: str) -> set[int]:
    nums: set[int] = set()
    for group in _CITE.findall(text):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                a, b = re.split(r"\s*[–-]\s*", part)[:2]
                nums.update(range(int(a), int(b) + 1))
            else:
                nums.add(int(part))
    return nums


def metrics(draft: Draft) -> dict:
    a = draft.article
    findings = _finding_texts(a)
    answer = a.get("answer") or a.get("bottom_line") or ""
    if isinstance(answer, dict):
        answer = " ".join(str(v) for v in answer.values())
    multi = sum(1 for f in findings if len(_cites_in(f)) >= 2)
    nums = sum(len(_NUM.findall(_CITE.sub("", f))) for f in findings)
    words_all = " ".join([answer] + findings + [str(u) for u in a.get("unknowns") or []])
    sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", words_all)))
    hedges = sum(words_all.lower().count(h) for h in _HEDGES)
    full_text_cited = sum(1 for i in draft.cited_refs if draft.papers[i - 1].full_text)
    return {
        "cited": len(draft.cited_refs),
        "full_text_cited": full_text_cited,
        "findings": len(findings),
        "multi_cite_findings": multi,
        "numbers_in_findings": nums,
        "answer_words": len(answer.split()),
        "total_words": len(words_all.split()),
        "hedges_per_sentence": round(hedges / sentences, 2),
        "unverified": len(draft.verification.get("unverified") or []),
        "misattributed": len(draft.verification.get("misattributed") or []),
        "style_errors": len(style_errors(draft.style_report)) if draft.style_report else 0,
    }


# ------------------------------------------------------------------- judge

def _source_material(draft: Draft) -> str:
    """Everything the writer was shown for the sources it cited, numbered the
    way the reader sees them (citation order)."""
    refs = [r for r in draft.article.get("references") or []
            if isinstance(r, int) and 1 <= r <= len(draft.papers)]
    blocks = []
    for n, idx in enumerate(refs, start=1):
        p = draft.papers[idx - 1]
        block = (f"[{n}] {p.title} ({p.year or 'n.d.'}; {p.venue or 'unknown venue'})\n"
                 f"Abstract: {p.abstract}")
        excerpt = (draft.excerpts or {}).get(idx)
        if excerpt:
            block += f"\nFull-text excerpt:\n{excerpt}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _reader_text(draft: Draft) -> str:
    """The briefing body as Markdown, cut before Methods: the judge grades the
    synthesis, not the deterministic boilerplate under it."""
    md = render_markdown(draft.article, draft.papers, draft.topic, draft.curation,
                         draft.verification, draft.provenance, draft.style_report)
    cut = re.search(r"^#+\s*Methods\b", md, flags=re.M)
    body = md[:cut.start()] if cut else md
    refs = re.search(r"^#+\s*References\b.*", md, flags=re.M | re.S)
    if refs:
        body += "\n\n" + refs.group(0)[:6000]
    return body


def judge_one(draft: Draft, model: str = JUDGE_MODEL) -> dict:
    rubric = "\n".join(f"- {k}: {v}" for k, v in DIMENSIONS.items())
    prompt = (
        f"QUESTION THE READER ASKED:\n{draft.topic}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "THE BRIEFING (as the reader sees it; superscripts are citation numbers "
        "matching the reference list):\n\n"
        f"{_reader_text(draft)}\n\n"
        "=====\nCITED SOURCE MATERIAL (numbered as in the briefing):\n\n"
        f"{_source_material(draft)}"
    )
    return generate_json(prompt, _JUDGE_SCHEMA, system=_JUDGE_SYSTEM, model=model)


def cmd_judge(args) -> None:
    names = sorted(n for n in os.listdir(args.runs)
                   if n.endswith(".json") and not n.endswith(".judge.json"))

    def one(name: str) -> tuple[str, dict]:
        stem = name[:-5]
        out = os.path.join(args.runs, stem + ".judge.json")
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                return stem, json.load(f)
        draft = _load(os.path.join(args.runs, name))
        result = {"judge": judge_one(draft, args.model), "metrics": metrics(draft)}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        return stem, result

    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, n) for n in names]):
            try:
                stem, r = fut.result()
                results[stem] = r
                print(f"judged {stem}: overall {r['judge']['overall']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    summarise(args.runs)


def summarise(run_dir: str) -> dict:
    rows = {}
    for n in sorted(os.listdir(run_dir)):
        if n.endswith(".judge.json"):
            with open(os.path.join(run_dir, n), encoding="utf-8") as f:
                rows[n[:-11]] = json.load(f)
    if not rows:
        print("no judged briefings")
        return {}
    dims = list(DIMENSIONS) + ["overall"]
    print(f"\n{run_dir}")
    print("topic".ljust(34) + "".join(d[:6].rjust(7) for d in dims))
    means = {d: 0.0 for d in dims}
    for stem, r in rows.items():
        s = dict(r["judge"]["scores"], overall=r["judge"]["overall"])
        print(stem[:33].ljust(34) + "".join(str(s[d]).rjust(7) for d in dims))
        for d in dims:
            means[d] += s[d] / len(rows)
    print("MEAN".ljust(34) + "".join(f"{means[d]:.1f}".rjust(7) for d in dims))
    mkeys = list(next(iter(rows.values()))["metrics"])
    mmeans = {k: sum(r["metrics"][k] for r in rows.values()) / len(rows) for k in mkeys}
    print("metrics: " + ", ".join(f"{k}={v:.1f}" for k, v in mmeans.items()))
    return {"judge": means, "metrics": mmeans}


# ----------------------------------------------------------------- compare

def compare_pair(a: Draft, b: Draft, model: str = JUDGE_MODEL, seed: int = 0) -> dict:
    """One pairwise judgement in each order; returns the order-corrected votes."""
    rubric = "\n".join(f"- {k}: {v}" for k, v in DIMENSIONS.items())
    material = _source_material(a)
    # Both briefings cite from the same pool; give the judge the union so it can
    # check either one.
    b_material = _source_material(b)
    if b_material != material:
        material += "\n\n----- (sources cited by the other briefing) -----\n\n" + b_material

    def ask(first: Draft, second: Draft) -> dict:
        prompt = (
            f"QUESTION THE READER ASKED:\n{a.topic}\n\nRUBRIC:\n{rubric}\n\n"
            f"===== BRIEFING A =====\n{_reader_text(first)}\n\n"
            f"===== BRIEFING B =====\n{_reader_text(second)}\n\n"
            "===== SOURCE MATERIAL (each briefing numbers its own citations; match "
            "by title) =====\n\n" + material
        )
        return generate_json(prompt, _PAIR_SCHEMA, system=_PAIR_SYSTEM, model=model)

    ab = ask(a, b)
    ba = ask(b, a)
    flip = {"A": "B", "B": "A", "tie": "tie"}
    ba_fixed = dict(ba, preferred=flip[ba["preferred"]],
                    dimensions={k: flip[v] for k, v in ba["dimensions"].items()})
    return {"ab": ab, "ba": ba_fixed}


def cmd_compare(args) -> None:
    names = sorted(n for n in os.listdir(args.b)
                   if n.endswith(".json") and not n.endswith((".judge.json", ".pair.json"))
                   and os.path.exists(os.path.join(args.a, n)))
    tag = args.tag or (os.path.basename(os.path.normpath(args.a)) + "-vs-"
                       + os.path.basename(os.path.normpath(args.b)))

    def one(name: str):
        stem = name[:-5]
        out = os.path.join(args.b, f"{stem}.{tag}.pair.json")
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                return stem, json.load(f)
        res = compare_pair(_load(os.path.join(args.a, name)),
                           _load(os.path.join(args.b, name)), args.model)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        return stem, res

    wins = {"A": 0, "B": 0, "tie": 0}
    dim_wins = {k: {"A": 0, "B": 0, "tie": 0} for k in DIMENSIONS}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, n) for n in names]):
            try:
                stem, res = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            votes = [res["ab"]["preferred"], res["ba"]["preferred"]]
            print(f"{stem[:40].ljust(42)} votes={votes} "
                  f"margins={[res['ab']['margin'], res['ba']['margin']]}")
            for v in votes:
                wins[v] += 1
            for k in DIMENSIONS:
                for side in ("ab", "ba"):
                    dim_wins[k][res[side]["dimensions"][k]] += 1
    print(f"\nA = {args.a}\nB = {args.b}")
    print(f"overall votes: A {wins['A']}  B {wins['B']}  tie {wins['tie']}")
    for k, w in dim_wins.items():
        print(f"  {k.ljust(14)} A {w['A']:>2}  B {w['B']:>2}  tie {w['tie']:>2}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze")
    p.add_argument("--topics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="openai/gpt-5.6-luna")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("write")
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="openai/gpt-5.6-luna")
    p.add_argument("--workers", type=int, default=3)
    p.set_defaults(fn=cmd_write)

    p = sub.add_parser("judge")
    p.add_argument("--runs", required=True)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--workers", type=int, default=3)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--workers", type=int, default=3)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("summary")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=lambda a: [summarise(r) for r in a.runs])

    args = ap.parse_args()
    random.seed(0)
    args.fn(args)


if __name__ == "__main__":
    main()

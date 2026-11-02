"""End-to-end: the current hosted pipeline vs the best Jev combination.

    python tools/jev_eval/run.py e2e write      # writer + style + figure gates, Sonnet 5.5
    python tools/jev_eval/run.py e2e judge      # blind rubric (cli:opus, subscription)
    python tools/jev_eval/run.py e2e compare    # pairwise, both orders
    python tools/jev_eval/run.py e2e figures    # Jev vs Haiku vs Opus 5.5 on pre-revision figure flags
    python tools/jev_eval/run.py e2e score

Arms, both written by Sonnet 5.5 (the hosted writer) on the same frozen pools:

- `base`: as shipped. Luna labels, regex design labels, the full context.
  Run 0 is the existing `evals/runs/final2` briefing; run 1 is fresh.
- `jev`: Jev pre-screen (confident tangentials) then Haiku 5.5 for the rest
  (real subset calls, leave-one-out cut-offs), Jev design labels, Jev source
  trim (leave-one-out threshold) and Jev passage trim (8,000 chars/excerpt).

The judge only sees the cited material, so it cannot see good evidence a
briefing left out. `score` adds pool-level source quality against the gold
labels: how much of the gold-direct evidence was cited, how much of what was
cited is gold-direct, and the gold design mix of the cited set.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import json
import os
import re
import shutil

import articlegen.sources as sources_mod
import articlegen.writer as writer_mod
from articlegen.pipeline import Draft, enforce_statistics, enforce_style
from articlegen.render import render_article, render_markdown
from articlegen.sources import full_text_excerpts
from articlegen.verify import check_statistics
from articlegen.writer import is_briefing, write_article, write_briefing

from common import (HAIKU, METER, OPUS55, OUT, RUNS, SONNET, jev, load_draft,
                    load_json, pool_paths, save_json, short, slug)

E2E = os.path.join(OUT, "e2e")
PASSAGE_BUDGET = 8000
PASSAGE_P = 0.5
JEV_RUNS = (1, 2)
BASE_RUNS = (1,)

# ------------------------------------------------------------ design override

_DESIGN_OVERRIDE: dict[str, str] = {}
_real_classify = sources_mod.classify_design


def _design_key(p) -> str:
    return f"{(p.title or '').strip().lower()}|{p.year}"


def _classify(paper):
    return _DESIGN_OVERRIDE.get(_design_key(paper)) or _real_classify(paper)


def _install_design(stem: str, draft: Draft) -> None:
    d = load_json(os.path.join(OUT, "design.json"))
    labels = d["jev"][stem]["labels"]
    for i, p in enumerate(draft.papers, start=1):
        _DESIGN_OVERRIDE[_design_key(p)] = labels[str(i)]


sources_mod.classify_design = _classify
writer_mod.classify_design = _classify

# ------------------------------------------------------------ arm pools


def _counts(rel: dict) -> dict:
    return {lab: sum(1 for v in rel.values() if v == lab) for lab in ("direct", "related", "tangential")}


def _strip_headings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("## "))


def jev_pool(stem: str, pool: Draft) -> tuple[Draft, dict]:
    rel_doc = load_json(os.path.join(OUT, "relevance.json"))
    trim_doc = load_json(os.path.join(OUT, "trim.json"))
    trim_sc = load_json(os.path.join(OUT, "trim_scores.json"))
    import trim as trim_mod

    labels = {int(k): v for k, v in rel_doc["cascade_real"]["haiku_tangential_loo"][stem]["relevance"].items()}
    counts = _counts(labels)
    mri = pool.curation.get("most_relevant_index")
    if labels.get(mri) != "direct":
        mri = next((i for i in sorted(labels) if labels[i] == "direct"), mri)

    t_ev = trim_sc["source_trim"]["loo"][stem]["t_ev"]
    answers = rel_doc["jev"][stem]["answers"]
    hide = {i for i, lab in labels.items() if lab == "related" and answers[str(i)]["answer_evidence"] < t_ev}
    shown_rel = {i: ("tangential" if i in hide else lab) for i, lab in labels.items()}

    draft = copy.deepcopy(pool)
    draft.curation = {"relevance": shown_rel, "counts": counts, "most_relevant_index": mri}
    items = trim_doc["passages"][stem]["items"]
    for idx in full_text_excerpts(draft.papers):
        kept = trim_mod.kept_text(items, idx, PASSAGE_P, PASSAGE_BUDGET)
        draft.papers[idx - 1].full_text = _strip_headings(kept)
    return draft, {"labels": {str(k): v for k, v in labels.items()}, "hidden": sorted(hide),
                   "t_ev": t_ev}


# ------------------------------------------------------------ write


def replay(pool: Draft, model: str, log) -> tuple[Draft, dict]:
    """brief_eval.replay_write, plus the figure flags before revision."""
    papers = pool.papers
    curation = pool.curation
    writer_fn = write_briefing if is_briefing(pool.article) else write_article
    article = writer_fn(pool.topic, papers, model=model, style_note="", curation=curation, api_key=None)
    article, style_report = enforce_style(article, model=model, log=log, papers=papers, curation=curation)
    pre = check_statistics(article, papers)
    article, verification, style_report = enforce_statistics(
        article, papers, model=model, log=log, style_report=style_report,
        direct_sources=((curation or {}).get("counts") or {}).get("direct"))
    provenance = dict(pool.provenance, model=model)
    d = Draft(topic=pool.topic, article=article, papers=papers, curation=curation,
              verification=verification, provenance=provenance, style_report=style_report,
              excerpts=full_text_excerpts(papers))
    return d, {"pre_revision_flags": pre.get("details") or [], "pre_total": pre.get("total")}


def _save(draft: Draft, out_dir: str, stem: str, meta: dict, log_lines: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(draft.to_dict(), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(draft.article, draft.papers, draft.topic, draft.curation,
                                draft.verification, draft.provenance, draft.style_report))
    with open(os.path.join(out_dir, stem + ".html"), "w", encoding="utf-8") as f:
        f.write(render_article(draft.article, draft.papers, draft.topic, draft.curation,
                               draft.verification, draft.provenance, draft.style_report))
    with open(os.path.join(out_dir, stem + ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    save_json(os.path.join(out_dir, stem + ".meta.json"), meta)


def write_one(arm: str, r: int, stem: str) -> str:
    out_dir = os.path.join(E2E, f"{arm}_r{r}")
    if os.path.exists(os.path.join(out_dir, stem + ".json")):
        return f"skip {arm}_r{r} {short(stem)}"
    pool = load_draft(os.path.join(os.path.dirname(pool_paths()[0]), stem + ".json"))
    extra = {}
    if arm == "jev":
        pool, extra = jev_pool(stem, pool)
        _install_design(stem, pool)
    lines: list[str] = []
    with METER.stage(f"e2e:{arm}:r{r}:{stem}"), METER.capture() as calls:
        draft, flags = replay(pool, SONNET, lines.append)
    _, ctx_chars = None, len(writer_mod._writer_context(pool.topic, pool.papers, "", pool.curation, kind="briefing")[0])
    meta = {"arm": arm, "run": r, "stem": stem, "usd": round(sum(c["usd"] for c in calls), 5),
            "calls": [{k: c[k] for k in ("model", "usd", "in", "out", "seconds")} for c in calls],
            "writer_context_chars": ctx_chars, **flags, **extra}
    _save(draft, out_dir, stem, meta, lines)
    return (f"done {arm}_r{r} {short(stem):11} ${meta['usd']:.3f} calls={len(calls)} "
            f"ctx={ctx_chars} flags_pre={len(flags['pre_revision_flags'])} cited={len(draft.cited_refs)}")


def cmd_write(argv):
    stems = [slug(p) for p in pool_paths()]
    only = os.environ.get("E2E_ONLY")
    jobs = [("jev", r, s) for r in JEV_RUNS for s in stems] + [("base", r, s) for r in BASE_RUNS for s in stems]
    if only:
        jobs = [j for j in jobs if f"{j[0]}_r{j[1]}" in only.split(",") or short(j[2]) in only.split(",")]
    # base run 0 is the shipped final2 briefing
    b0 = os.path.join(E2E, "base_r0")
    os.makedirs(b0, exist_ok=True)
    for s in stems:
        for ext in (".json", ".md", ".html", ".log"):
            src = os.path.join(RUNS, "final2", s + ext)
            if os.path.exists(src) and not os.path.exists(os.path.join(b0, s + ext)):
                shutil.copy(src, os.path.join(b0, s + ext))
    # Design overrides are process-global, so jev and base jobs cannot share a
    # thread pool: run each arm on its own.
    for arm in ("jev", "base"):
        arm_jobs = [j for j in jobs if j[0] == arm]
        if arm == "base":
            _DESIGN_OVERRIDE.clear()
        with cf.ThreadPoolExecutor(max_workers=5) as ex:
            for fut in cf.as_completed([ex.submit(write_one, *j) for j in arm_jobs]):
                try:
                    print(fut.result(), flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"ledger: ${METER.spent:.3f}")


# ------------------------------------------------------------ judge / compare


def _run_dirs():
    return sorted(d for d in os.listdir(E2E) if os.path.isdir(os.path.join(E2E, d)))


def cmd_judge(argv):
    import brief_eval as be
    jobs = []
    for d in _run_dirs():
        for n in sorted(os.listdir(os.path.join(E2E, d))):
            if n.endswith(".json") and not n.endswith((".judge.json", ".meta.json", ".pair.json")):
                if not os.path.exists(os.path.join(E2E, d, n[:-5] + ".judge.json")):
                    jobs.append((d, n[:-5]))

    def one(job):
        d, stem = job
        draft = load_draft(os.path.join(E2E, d, stem + ".json"))
        res = {"judge": be.judge_one(draft), "metrics": be.metrics(draft)}
        save_json(os.path.join(E2E, d, stem + ".judge.json"), res)
        return f"judged {d} {short(stem)} overall {res['judge']['overall']}"

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for fut in cf.as_completed([ex.submit(one, j) for j in jobs]):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)


PAIRS = [("base_r0", "jev_r1"), ("base_r1", "jev_r2"), ("base_r0", "base_r1")]


def cmd_compare(argv):
    import brief_eval as be
    jobs = []
    for a, b in PAIRS:
        for p in pool_paths():
            s = slug(p)
            out = os.path.join(E2E, b, f"{s}.{a}-vs-{b}.pair.json")
            if not os.path.exists(out) and all(os.path.exists(os.path.join(E2E, x, s + ".json")) for x in (a, b)):
                jobs.append((a, b, s, out))

    def one(job):
        a, b, s, out = job
        res = be.compare_pair(load_draft(os.path.join(E2E, a, s + ".json")),
                              load_draft(os.path.join(E2E, b, s + ".json")))
        save_json(out, res)
        return f"{a} vs {b} {short(s)}: {res['ab']['preferred']}/{res['ba']['preferred']}"

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for fut in cf.as_completed([ex.submit(one, j) for j in jobs]):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)


# ------------------------------------------------------------ figure flags (D2)

from typesafe_sdk import Choice  # noqa: E402

FIG_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["stated", "derived", "absent"]}},
        "required": ["id", "verdict"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}
FIG_SYSTEM = (
    "A deterministic checker could not find each figure below, as written, in the "
    "sources its sentence cites. For each, decide:\n"
    '- "stated": the cited material states this value for this claim, in another '
    "format (rounding, a percentage written as a proportion, a range written "
    "differently, words for numbers).\n"
    '- "derived": the value is a simple derivation of stated numbers (a difference, '
    "a sum, a ratio) the material plainly supports.\n"
    '- "absent": the material does not give this value for this claim.\n'
    "Use only the material shown.")

FIG_Q = Choice(
    instructions=("A checker could not find `figure`, as written, in `material` (the "
                  "sources `sentence` cites). Does `material` give this value for the "
                  "claim in `sentence`?"),
    criteria={
        "absent": "The material does not give this value for this claim.",
        "stated": "The material states this value in another format (rounding, proportion vs percent, words, a differently written range).",
        "derived": "The value is a simple derivation (difference, sum, ratio) of numbers the material states for this claim.",
    },
)


def _flag_items():
    items = []
    for d in _run_dirs():
        for n in sorted(os.listdir(os.path.join(E2E, d))):
            if not n.endswith(".meta.json"):
                continue
            meta = load_json(os.path.join(E2E, d, n))
            draft = load_draft(os.path.join(E2E, d, n[:-10] + ".json"))
            shown = full_text_excerpts(draft.papers)
            for j, f in enumerate(meta.get("pre_revision_flags") or []):
                mats = []
                for i in f["cited"]:
                    p = draft.papers[i - 1]
                    mats.append(f"SOURCE [{i}] {p.title}\n{p.abstract or ''}\n{shown.get(i, '')}")
                items.append({"id": f"{d}/{n[:-10]}/{j}", "figure": f["figure"], "kind": f["kind"],
                              "sentence": f["sentence"], "material": "\n\n".join(mats)})
    return items


def cmd_figures(argv):
    from articlegen.llm import generate_json
    path = os.path.join(OUT, "figures.json")
    doc = load_json(path, {})
    items = doc.get("items") or _flag_items()
    doc["items"] = items
    print(f"{len(items)} pre-revision figure flags")
    if not items:
        save_json(path, doc)
        return
    if "jev" not in doc:
        with METER.stage("figures:jev"), METER.capture() as calls:
            def ask(it):
                from common import choice
                lab, conf, probs = choice(jev({"figure": it["figure"], "sentence": it["sentence"],
                                               "material": it["material"][:60000]}, {"v": FIG_Q}), "v")
                return it["id"], {"verdict": lab, "conf": conf, "probs": probs}
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                doc["jev"] = dict(ex.map(METER.staged(ask), items))
        doc["jev_usd"] = sum(c["usd"] for c in calls)
        save_json(path, doc)
    for key, model in (("haiku", HAIKU), ("ref_opus55", OPUS55)):
        if key in doc:
            continue
        verdicts, usd = {}, 0.0
        groups = {}
        for it in items:
            groups.setdefault(it["id"].rsplit("/", 1)[0], []).append(it)
        for g, its in groups.items():
            prompt = "\n\n".join(
                f"=== FLAG {k} ===\nFigure: {it['figure']}\nSentence: {it['sentence']}\nCited material:\n{it['material']}"
                for k, it in enumerate(its, start=1))
            with METER.stage(f"figures:{key}"), METER.capture() as calls:
                res = generate_json(prompt, FIG_SCHEMA, system=FIG_SYSTEM, model=model)
            usd += sum(c["usd"] for c in calls)
            for v in res.get("verdicts") or []:
                if isinstance(v.get("id"), int) and 1 <= v["id"] <= len(its):
                    verdicts[its[v["id"] - 1]["id"]] = {"verdict": v["verdict"]}
        doc[key], doc[f"{key}_usd"] = verdicts, usd
        save_json(path, doc)
    print(f"ledger: ${METER.spent:.3f}")


# ------------------------------------------------------------ score


def _gold_rel(stem, hand):
    import relevance as R
    return R.gold(load_json(os.path.join(OUT, "relevance.json")), stem, hand)


def _gold_design(stem, hand):
    import design as D
    return D.gold(load_json(os.path.join(OUT, "design.json")), stem, hand)


def source_quality(draft: Draft, stem: str) -> dict:
    hand_r = load_json(os.path.join(OUT, "hand_relevance.json"), {})
    hand_d = load_json(os.path.join(OUT, "hand_design.json"), {})
    g = _gold_rel(stem, hand_r)
    gd = _gold_design(stem, hand_d)
    cited = sorted(draft.cited_refs)
    gold_direct = {int(k) for k, v in g.items() if v == "direct"}
    lab = [g.get(str(i)) for i in cited]
    des = [gd.get(str(i)) for i in cited]
    return {
        "cited": len(cited),
        "cited_gold_direct": sum(1 for x in lab if x == "direct"),
        "cited_gold_tangential": sum(1 for x in lab if x == "tangential"),
        "cited_gold_unknown": sum(1 for x in lab if x is None),
        "gold_direct_in_pool": len(gold_direct),
        "gold_direct_coverage": round(len(gold_direct & set(cited)) / len(gold_direct), 3) if gold_direct else None,
        "cited_synthesis_or_trial": sum(1 for x in des if x in ("synthesis", "trial")),
    }


def cmd_score(argv):
    import brief_eval as be
    rows = {}
    for d in _run_dirs():
        for n in sorted(os.listdir(os.path.join(E2E, d))):
            if not n.endswith(".judge.json"):
                continue
            stem = n[:-11]
            j = load_json(os.path.join(E2E, d, n))
            meta = load_json(os.path.join(E2E, d, stem + ".meta.json"), {})
            draft = load_draft(os.path.join(E2E, d, stem + ".json"))
            rows.setdefault(d, {})[stem] = {
                "overall": j["judge"]["overall"], "scores": j["judge"]["scores"],
                "unsupported": len(j["judge"].get("unsupported_claims") or []),
                "usd": meta.get("usd"), "writer_context_chars": meta.get("writer_context_chars"),
                "flags_pre": len(meta.get("pre_revision_flags") or []),
                "flags_after": len((draft.verification or {}).get("unverified") or [])
                + len((draft.verification or {}).get("misattributed") or []),
                **source_quality(draft, stem), "metrics": j["metrics"]}
    pairs = {}
    for a, b in PAIRS:
        votes = {"A": 0, "B": 0, "tie": 0}
        dims = {k: {"A": 0, "B": 0, "tie": 0} for k in be.DIMENSIONS}
        for p in pool_paths():
            res = load_json(os.path.join(E2E, b, f"{slug(p)}.{a}-vs-{b}.pair.json"))
            if not res:
                continue
            for side in ("ab", "ba"):
                votes[res[side]["preferred"]] += 1
                for k in be.DIMENSIONS:
                    dims[k][res[side]["dimensions"][k]] += 1
        pairs[f"{a}-vs-{b}"] = {"votes": votes, "dimensions": dims}
    summary = {}
    for d, per in rows.items():
        vals = list(per.values())

        def mean(k):
            xs = [v[k] for v in vals if v.get(k) is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        summary[d] = {k: mean(k) for k in ("overall", "unsupported", "usd", "writer_context_chars",
                                            "flags_pre", "flags_after", "cited", "cited_gold_direct",
                                            "cited_gold_tangential", "gold_direct_coverage",
                                            "cited_synthesis_or_trial")}
        summary[d]["dims"] = {k: round(sum(v["scores"][k] for v in vals) / len(vals), 2)
                              for k in be.DIMENSIONS}
        summary[d]["n"] = len(vals)
    out = {"rows": rows, "summary": summary, "pairs": pairs}
    save_json(os.path.join(OUT, "e2e_scores.json"), out)
    for d, s in summary.items():
        print(f"{d:8} n={s['n']} overall {s['overall']} unsup {s['unsupported']} ${s['usd']} "
              f"ctx {s['writer_context_chars']} cited {s['cited']} gold-direct-cited {s['cited_gold_direct']} "
              f"tang-cited {s['cited_gold_tangential']} direct-coverage {s['gold_direct_coverage']} "
              f"syn/trial {s['cited_synthesis_or_trial']} flags {s['flags_pre']}->{s['flags_after']}")
    for k, v in pairs.items():
        print(f"{k}: {v['votes']}")


def main(argv):
    cmd = argv[0] if argv else "score"
    {"write": cmd_write, "judge": cmd_judge, "compare": cmd_compare,
     "figures": cmd_figures, "score": cmd_score}[cmd](argv[1:])

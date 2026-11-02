"""Area D1: study-design labels — production regex vs LLMs vs Jev.

    python tools/jev_eval/run.py design reference   # Opus 5.5 + Fable 5.1
    python tools/jev_eval/run.py design llm         # Luna, Haiku
    python tools/jev_eval/run.py design jev
    python tools/jev_eval/run.py design score

`sources.classify_design` reads title, venue and index publication types only.
Its label is printed in Table 1 and handed to the writer as "Design (from index
metadata)", which is where a GRADE starting certainty comes from. The question
here is accuracy, not cost: the regex is free.
"""

from __future__ import annotations

import concurrent.futures as cf
import os

from typesafe_sdk import Choice

from articlegen.llm import generate_json
from articlegen.sources import classify_design

from common import (HAIKU, LUNA, METER, OPUS55, OUT, SONNET, choice, jev, load_draft,
                    load_json, pool_paths, save_json, short, slug, summary)

PATH = os.path.join(OUT, "design.json")
ABSTRACT_CHARS = 1500

DEFS = {
    "synthesis": "Systematic review or meta-analysis: a stated search of the literature and pooled or systematically appraised studies.",
    "trial": "Randomised or non-randomised controlled trial, cluster trial, or a trial's secondary analysis: investigators assigned the intervention.",
    "observational": "Cohort, case-control, cross-sectional, registry, survey or before-after study of routinely occurring exposures; no assigned intervention.",
    "qualitative": "Interviews, focus groups, or thematic or ethnographic analysis.",
    "case": "Case report or case series.",
    "scoping": "Scoping, mapping or rapid review without formal appraisal.",
    "narrative": "Narrative or expert review, commentary, editorial or perspective without a systematic search.",
    "consensus": "Guideline, consensus statement or position paper.",
    "other": "Anything else: protocol, modelling, methods paper, lab or animal study, or not determinable.",
}
LABELS = tuple(DEFS)
# GRADE starts randomised evidence high and observational low; what matters
# downstream is which tier a paper lands in.
TIER = {"synthesis": "synthesis", "trial": "trial", "observational": "observational",
        "qualitative": "weak", "case": "weak", "scoping": "review", "narrative": "review",
        "consensus": "review", "other": "weak"}
TIERS = ("synthesis", "trial", "observational", "review", "weak")

SYSTEM = ("You classify the study design of each paper from its title, venue, index "
          "publication types and abstract. Use exactly one label per paper:\n"
          + "\n".join(f'- "{k}": {v}' for k, v in DEFS.items())
          + "\nJudge what the paper itself did, not the designs it discusses.")

SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"},
                       "design": {"type": "string", "enum": list(LABELS)}},
        "required": ["index", "design"], "additionalProperties": False}}},
    "required": ["labels"], "additionalProperties": False,
}


def _record(i, p) -> dict:
    return {"index": i, "title": p.title or "", "venue": p.venue or "",
            "publication_types": list(p.publication_types or []),
            "abstract": (p.abstract or "")[:ABSTRACT_CHARS]}


def _prompt(papers) -> str:
    blocks = []
    for i, p in enumerate(papers, start=1):
        r = _record(i, p)
        blocks.append(f"PAPER {i}\nTitle: {r['title']}\nVenue: {r['venue']}\n"
                      f"Publication types: {', '.join(r['publication_types']) or 'none'}\n"
                      f"Abstract: {r['abstract']}")
    return "Label the design of every paper below.\n\n" + "\n\n".join(blocks)


def _pools():
    return [(slug(p), load_draft(p)) for p in pool_paths()]


def _run_llm(doc, key, model):
    out = doc.setdefault(key, {})

    def one(item):
        stem, draft = item
        with METER.stage(f"design:{key}:{stem}"), METER.capture() as calls:
            res = generate_json(_prompt(draft.papers), SCHEMA, system=SYSTEM, model=model)
        labels = {str(x["index"]): x["design"] for x in res.get("labels") or []
                  if isinstance(x.get("index"), int) and x.get("design") in LABELS}
        return stem, {"labels": labels, "usd": round(sum(c["usd"] for c in calls), 5),
                      "served_by": sorted({c["model"] for c in calls})}

    todo = [(s, d) for s, d in _pools() if s not in out]
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for stem, res in ex.map(one, todo):
            out[stem] = res
            print(f"  {key:10} {short(stem):11} {len(res['labels'])} ${res['usd']:.4f} {res['served_by']}",
                  flush=True)
            save_json(PATH, doc)


QUESTION = Choice(
    instructions=("What study design did `paper` itself use? Judge from its title, "
                  "venue, index publication types and abstract — what the authors did, "
                  "not designs they discuss."),
    criteria={k: DEFS[k] for k in ("other", "case", "qualitative", "observational",
                                    "trial", "synthesis", "scoping", "narrative", "consensus")},
)


def cmd_jev(doc):
    out = doc.setdefault("jev", {})
    for stem, draft in _pools():
        if stem in out:
            continue
        with METER.stage(f"design:jev:{stem}"), METER.capture() as calls:
            def ask(ip):
                i, p = ip
                lab, conf, _ = choice(jev({"paper": _record(i, p)}, {"design": QUESTION}), "design")
                return str(i), {"label": lab, "conf": conf}
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                got = dict(ex.map(METER.staged(ask), enumerate(draft.papers, start=1)))
        out[stem] = {"labels": {k: v["label"] for k, v in got.items()},
                     "conf": {k: v["conf"] for k, v in got.items()},
                     "usd": round(sum(c["usd"] for c in calls), 6)}
        print(f"  jev        {short(stem):11} ${out[stem]['usd']:.5f}", flush=True)
        save_json(PATH, doc)


def cmd_regex(doc):
    doc["regex"] = {stem: {"labels": {str(i): classify_design(p)
                                      for i, p in enumerate(d.papers, start=1)}, "usd": 0.0}
                    for stem, d in _pools()}
    save_json(PATH, doc)


def gold(doc, stem, hand=None):
    a, b = doc["ref_opus55"][stem]["labels"], doc["ref_sonnet"][stem]["labels"]
    g = {k: a[k] for k in a if b.get(k) == a[k]}
    for k, v in ((hand or {}).get(stem) or {}).items():
        if v in LABELS:
            g[k] = v
    return g


def score(doc, hand=None):
    res = {}
    for arm in ("regex", "luna", "haiku", "jev", "ref_opus55", "ref_sonnet"):
        if arm not in doc:
            continue
        rows = []
        for stem in sorted(doc["regex"]):
            g = gold(doc, stem, hand)
            pred = doc[arm][stem]["labels"]
            rows.append({"stem": stem, "usd": doc[arm][stem]["usd"],
                         "exact": summary(g, pred, LABELS),
                         "tier": summary({k: TIER[v] for k, v in g.items()},
                                         {k: TIER[v] for k, v in pred.items()}, TIERS)})
        n = sum(r["exact"]["n"] for r in rows)
        res[arm] = {
            "n": n,
            "exact_accuracy": round(sum(r["exact"]["accuracy"] * r["exact"]["n"] for r in rows) / n, 4),
            "tier_accuracy": round(sum(r["tier"]["accuracy"] * r["tier"]["n"] for r in rows) / n, 4),
            "usd_per_pool": round(sum(r["usd"] for r in rows) / len(rows), 5),
            "trial_recall": _pooled(rows, "tier", "trial"),
            "synthesis_recall": _pooled(rows, "exact", "synthesis"),
            "observational_recall": _pooled(rows, "tier", "observational"),
            "rows": rows,
        }
    return res


def _pooled(rows, kind, lab):
    held = sum(sum(r[kind]["confusion"][lab].values()) for r in rows)
    tp = sum(r[kind]["confusion"][lab][lab] for r in rows)
    return round(tp / held, 4) if held else None


def cmd_score(doc):
    hand = load_json(os.path.join(OUT, "hand_design.json"), {})
    res = score(doc, hand)
    save_json(os.path.join(OUT, "design_scores.json"), res)
    print(f"{'arm':12} {'exact':>6} {'tier':>6} {'trial-rec':>9} {'syn-rec':>8} {'obs-rec':>8} {'$/pool':>8}")
    for arm, r in res.items():
        print(f"{arm:12} {r['exact_accuracy']:.3f} {r['tier_accuracy']:.3f} {r['trial_recall'] or 0:9.3f} "
              f"{r['synthesis_recall'] or 0:8.3f} {r['observational_recall'] or 0:8.3f} {r['usd_per_pool']:8.5f}")


def main(argv):
    doc = load_json(PATH, {})
    if "regex" not in doc:
        cmd_regex(doc)
    cmd = argv[0] if argv else "score"
    if cmd == "reference":
        _run_llm(doc, "ref_opus55", OPUS55)
        _run_llm(doc, "ref_sonnet", SONNET)  # Fable cut to stay inside the $15 cap
    elif cmd == "llm":
        _run_llm(doc, "luna", LUNA)
        _run_llm(doc, "haiku", HAIKU)
    elif cmd == "jev":
        cmd_jev(doc)
    else:
        cmd_score(doc)
    print(f"ledger: ${METER.spent:.3f} spent")

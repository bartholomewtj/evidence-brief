"""Follow-up: noul compose v2 (no new curation calls) + modifier-scope claims."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from typesafe_sdk import Noul, TypeSafeClient

import jev_bakeoff as jb
from articlegen.pipeline import Draft
from claim_probe import DEFAULT_MANIFEST, pairs, score as score_claims
from jev_questions import (
    QUESTION,
    RESULTS_PATH,
    _gate_summary,
    compose_claim_noul,
    compose_claim_relation,
)

MODIFIER = Noul(
    instructions=(
        "`sentence` may name several outcomes and a modifier (genetic "
        "predisposition, lifestyle, interaction, duration). Does it attach "
        "that modifier to an outcome for which `excerpt` does not actually "
        "test the modifier?"
    ),
    criteria={
        "true": (
            "The sentence hangs the modifier on an outcome the excerpt did "
            "not test it on — typically a relative clause after a second "
            "outcome, or an 'unmodified by X' claim covering more than the "
            "excerpt measured."
        ),
        "false": (
            "There is no such modifier, only one outcome, or the excerpt "
            "tests the modifier for every outcome the sentence names."
        ),
    },
)

NEAREST = Noul(
    instructions=(
        "If `sentence` ends with a relative clause ('which…', 'that…'), "
        "does `excerpt` establish that clause for the outcome immediately "
        "before the comma — not only for an earlier outcome in the sentence?"
    ),
    criteria={
        "true": "The excerpt supports the trailing clause on the nearest outcome.",
        "false": (
            "There is no trailing relative clause, or the excerpt only "
            "establishes that clause for a different, earlier outcome."
        ),
    },
)


def compose_v2(ans, t_shift=0.5, t_bg=0.7, t_same=0.5) -> str:
    """Drop health_outcome: reviews about the briefing question are still direct."""
    if ans["shift_is_exposure"] < t_shift:
        return "tangential"
    if ans["background_only"] >= t_bg and ans["same_question"] < t_same:
        return "tangential"
    if ans["same_question"] >= t_same:
        return "direct"
    return "related"


def main() -> int:
    doc = json.loads(open(RESULTS_PATH, encoding="utf-8").read())
    prior = json.loads(open(os.path.join(_HERE, "jev_bakeoff_results.json"), encoding="utf-8").read())
    opus = {int(k): v for k, v in prior["curation_runs"][0]["llm"].items()}
    raw = doc["curation"]["noul_compose"]["raw"]

    print("Curation compose v2 (reuse noul payloads, no new Jev calls)\n")
    variants = [
        ("v2_0.5/0.7/0.5", dict(t_shift=0.5, t_bg=0.7, t_same=0.5)),
        ("v2_0.5/0.6/0.5", dict(t_shift=0.5, t_bg=0.6, t_same=0.5)),
        ("v2_0.5/0.55/0.5", dict(t_shift=0.5, t_bg=0.55, t_same=0.5)),
        ("v2_0.5/0.7/0.55", dict(t_shift=0.5, t_bg=0.7, t_same=0.55)),
    ]
    sweeps = []
    for name, kw in variants:
        got = {row["i"]: compose_v2(row["payload"], **kw) for row in raw}
        g = _gate_summary(opus, got)
        sweeps.append({"name": name, **{k: g[k] for k in (
            "agree", "n", "direct", "related", "tangential", "critical_ok", "counts", "disagreements"
        )}})
        flag = "PASS" if g["critical_ok"] else "FAIL"
        print(f"  {name:<22} {g['agree']}/{g['n']}  dir {g['direct']}  rel {g['related']}  "
              f"tang {g['tangential']}  {g['counts']}  {flag}")
        for d in g["disagreements"]:
            print(f"      [{d['i']:2}] opus {d['opus']:<11} -> {d['jev']}")
    doc["curation"]["noul_compose_v2"] = {"note": "post-hoc on noul_compose payloads", "sweeps": sweeps}

    # Confidence gate on relation_choice (no new calls)
    rel = doc["claims"]["relation_choice"]["results"]
    labels = jb._load_labels()
    print("\nClaims: relation_choice + confidence floor (no new calls)\n")
    for floor in (None, 0.55, 0.6, 0.7):
        gated = []
        for r in rel:
            v = r["verdict"]
            conf = r["payload"]["relation"]["confidence"]
            if floor is not None and v == "overreach" and conf < floor:
                v = "unclear"
            gated.append({**r, "verdict": v})
        m = jb._claim_metrics(score_claims(labels, gated))
        tag = "none" if floor is None else f">={floor}"
        print(f"  conf floor {tag:<6} exact {m['exact']}/{m['n']}  OR {m['overreach_recall']}  "
              f"FP {m['false_overreach']}")
        for d in m["disagreements"]:
            print(f"      [{d['id']:2}] human {d['human']:<11} -> {d['model']}")
        if floor is not None:
            doc["claims"][f"relation_conf_{floor}"] = {**m, "seconds": None, "note": "post-hoc"}

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY for modifier-scope calls.")
        _save(doc)
        return 2

    draft = jb._load_draft(DEFAULT_MANIFEST)
    recs = pairs(draft)
    client = TypeSafeClient(timeout=120.0)
    print("\nClaims: modifier-scope Nouls (new Jev calls)\n")
    try:
        results = []
        t0 = time.perf_counter()
        tin = tout = 0
        for rec in recs:
            resp = client.system_one(
                state={
                    "question": QUESTION,
                    "sentence": rec["sentence"],
                    "excerpt": rec["excerpt"],
                    "cited": rec["cited"],
                },
                questions={"modifier_mismatch": MODIFIER, "nearest_clause_ok": NEAREST},
                model="jev-latest",
            )
            a_in, a_out = jb._jev_usage(resp)
            tin += a_in
            tout += a_out
            mm = round(resp.nouls["modifier_mismatch"].noul, 3)
            nc = round(resp.nouls["nearest_clause_ok"].noul, 3)
            # Combine with relation_choice: mismatch wins; else keep relation.
            rel_v = next(r["verdict"] for r in rel if r["id"] == rec["id"])
            if mm >= 0.55:
                verdict = "overreach"
            else:
                verdict = rel_v
            results.append({
                "id": rec["id"], "verdict": verdict,
                "payload": {"modifier_mismatch": mm, "nearest_clause_ok": nc, "relation": rel_v},
                "truncated": rec["truncated"], "excerpt_chars": rec["excerpt_chars"],
            })
            print(f"  [{rec['id']:2}] mismatch {mm:.2f}  nearest_ok {nc:.2f}  "
                  f"rel={rel_v} -> {verdict}", flush=True)
        sec = time.perf_counter() - t0
        report = score_claims(labels, results)
        metrics = jb._claim_metrics(report)
        metrics.update(seconds=round(sec, 2), tokens_in=tin, tokens_out=tout,
                       est_usd=round(jb._usage_cost(tin, tout, (0.042, 0.0)), 6),
                       results=results)
        doc["claims"]["relation_plus_modifier"] = metrics
        print(f"\n  combined exact {metrics['exact']}/{metrics['n']}  "
              f"OR {metrics['overreach_recall']}  FP {metrics['false_overreach']}  {sec:.1f}s")
        for d in metrics["disagreements"]:
            print(f"      [{d['id']:2}] human {d['human']:<11} -> {d['model']}")
    finally:
        client.close()

    _save(doc)
    return 0


def _save(doc: dict) -> None:
    doc["updated"] = datetime.now(timezone.utc).isoformat()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())

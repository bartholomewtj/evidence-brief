#!/usr/bin/env python
"""Try better Jev questions on the frozen night-shift pool.

The 10-run bakeoff asked one 3-way Choice over `topic` + abstracts, and never
emitted `tangential`. TypeSafe's own guidance is: split dimensions, put the
judgment in instructions, and let code compose. This script measures that.

    python tools/jev_questions.py

Curation gold: frozen Opus labels. Claims gold: hand labels.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # noqa: E402

from articlegen.pipeline import Draft  # noqa: E402
import jev_bakeoff as jb  # noqa: E402
from claim_probe import DEFAULT_MANIFEST, pairs, score as score_claims  # noqa: E402

RESULTS_PATH = os.path.join(_HERE, "jev_questions_results.json")
PRIOR_PATH = os.path.join(_HERE, "jev_bakeoff_results.json")
LABELS = ("direct", "related", "tangential")
QUESTION = (
    "What health risks are associated with night shift work in working adults, "
    "and how strong is the evidence across sleep, cardiometabolic, cancer and "
    "mental health outcomes?"
)

# ---- curation question sets -------------------------------------------------

CURATE_CHOICE_STRICT = Choice(
    instructions=(
        "The briefing question is `question`. How should SOURCE "
        "(`paper.title` + `paper.abstract`) be labelled for that question?\n"
        "A passing mention of night shifts in the introduction is not enough "
        "for related or direct. If the paper's own research question is about "
        "something else (diet timing, wearables, a different occupation's "
        "general occupational health, a mechanism with no night-shift outcome "
        "data), it is tangential."
    ),
    criteria={
        "direct": (
            "The paper's own question is the health risks of night shift or "
            "shift work in working adults: right exposure, a health outcome, "
            "the working population."
        ),
        "related": (
            "The paper studies night/shift work as the exposure, but a "
            "different question — a mechanism only, a sleep-disorder subtype, "
            "chronotype, or a neighbouring outcome. Not the briefing question."
        ),
        "tangential": (
            "Night/shift work is absent, or only named as background. The "
            "paper is about another intervention, setting, or occupational "
            "group. Do not promote a mention into related."
        ),
    },
)

CURATE_SCORE = Score(
    instructions=(
        "The briefing question is `question`. How directly does SOURCE "
        "(`paper.title` + `paper.abstract`) address it?"
    ),
    criteria=[
        "Tangential: night/shift work is absent or only a passing mention. "
        "The paper studies another exposure, setting, or occupation.",
        "Related: night/shift work is the exposure, but the paper asks a "
        "different question (mechanism, chronotype, a neighbouring outcome).",
        "Direct: the paper studies health risks of night shift work in "
        "working adults — the briefing question itself.",
    ],
)

CURATE_NOULS = {
    "shift_is_exposure": Noul(
        instructions=(
            "Is night shift work or shift work the exposure this paper is "
            "designed to study, as opposed to a passing mention in the "
            "introduction of `paper.abstract`?"
        ),
        criteria={
            "true": "The methods or research question treat shift/night work as the exposure.",
            "false": "Shift/night work is absent, incidental, or one of several lifestyle examples.",
        },
    ),
    "health_outcome": Noul(
        instructions=(
            "Does `paper.abstract` report a health outcome (disease, symptoms, "
            "mortality, validated scale) measured in people who do night or "
            "shift work?"
        ),
        criteria={
            "true": "An empirical health result in shift/night workers is reported.",
            "false": "No such result — mechanism, methods, needs assessment, or another population.",
        },
    ),
    "same_question": Noul(
        instructions=(
            "Does this paper address `question` itself — health risks of night "
            "shift work in working adults — rather than an adjacent question "
            "such as cortisol mechanisms, chronotype, a sleep-disorder subtype, "
            "or occupational-health services?"
        ),
        criteria={
            "true": "Same exposure, working-adult population, and a health-risk outcome of night shift.",
            "false": "A neighbouring question, even if night shifts appear in the setting.",
        },
    ),
    "background_only": Noul(
        instructions=(
            "Would this paper only be useful as background or framing for "
            "`question`, not as evidence that belongs in the findings?"
        ),
        criteria={
            "true": "Background, definition, mechanism-without-outcomes, or a different topic.",
            "false": "It carries evidence a briefing on `question` should consider citing.",
        },
    ),
}


def compose_noul(ans: dict, *, t_shift=0.5, t_bg=0.55, t_same=0.55, t_out=0.5) -> str:
    """Code owns the label. Thresholds are the thing being measured."""
    if ans["shift_is_exposure"] < t_shift or ans["background_only"] >= t_bg:
        return "tangential"
    if ans["same_question"] >= t_same and ans["health_outcome"] >= t_out:
        return "direct"
    return "related"


def compose_score(value: float) -> str:
    if value < 0.5:
        return "tangential"
    if value < 1.5:
        return "related"
    return "direct"


def compose_two_stage(shift_noul: float, choice: str, *, t_shift=0.5) -> str:
    if shift_noul < t_shift:
        return "tangential"
    return choice if choice in LABELS else "related"


# ---- claim question sets ----------------------------------------------------

CLAIM_UNSUPPORTED = {
    "core_in_excerpt": Noul(
        instructions=(
            "Are the specific numbers, study-design facts, and named findings "
            "in `sentence` present in `excerpt` (same values, even if wording differs)?"
        ),
        criteria={
            "true": "The excerpt contains those facts.",
            "false": "A named number or finding in the sentence is not in the excerpt.",
        },
    ),
    "unsupported_clause": Noul(
        instructions=(
            "Does `sentence` assert any clause — an outcome, population, "
            "modifier, causal verb, or 'no study exists' claim — that `excerpt` "
            "does not establish? Extra hedging is not an unsupported clause. "
            "Attaching a modifier tested for outcome A to outcome B is."
        ),
        criteria={
            "true": "At least one clause goes beyond the excerpt.",
            "false": "Every assertion is stated or plainly entailed by the excerpt.",
        },
    ),
}

CLAIM_RELATION = Choice(
    instructions=(
        "How does `excerpt` relate to `sentence`? Judge the whole sentence, "
        "including every outcome and modifier it names."
    ),
    criteria={
        "supports": "The excerpt states or directly implies the whole sentence.",
        "overreaches": (
            "Part of the sentence is in the excerpt, but another part goes "
            "further: a modifier applied to the wrong outcome, a wider "
            "population, a dropped hedge, or a claim of absence the excerpt "
            "never makes."
        ),
        "silent": "The excerpt does not address what the sentence asserts.",
    },
)


def compose_claim_noul(core: float, extra: float) -> str:
    if extra >= 0.55:
        return "overreach"
    if core >= 0.55 and extra < 0.4:
        return "supported"
    return "unclear"


def compose_claim_relation(choice: str) -> str:
    return {
        "supports": "supported",
        "overreaches": "overreach",
        "silent": "unclear",
    }.get(choice, "unclear")


def _paper_state(topic: str, question: str, paper) -> dict:
    return {
        "topic": topic,
        "question": question,
        "paper": {
            "title": paper.title,
            "year": paper.year,
            "abstract": paper.abstract or "",
        },
    }


def _gate_summary(ref: dict[int, str], got: dict[int, str]) -> dict:
    g = jb._gate_row(ref, got)
    counts = jb._counts(got)
    return {
        "agree": g["agree"],
        "n": g["n"],
        "direct": jb._fmt_retained(g["retained"], "direct"),
        "related": jb._fmt_retained(g["retained"], "related"),
        "tangential": jb._fmt_retained(g["retained"], "tangential"),
        "critical_ok": g["critical_ok"],
        "counts": counts,
        "toward_related": g["toward_related"],
        "away_related": g["away_related"],
        "disagreements": [
            {"i": i, "opus": ref[i], "jev": got[i]}
            for i in sorted(set(ref) & set(got))
            if ref[i] != got[i]
        ],
    }


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY.")
        return 2

    draft = jb._load_draft(DEFAULT_MANIFEST)
    papers = draft.papers[:20]
    recs = pairs(draft)
    labels = jb._load_labels()
    prior = json.loads(open(PRIOR_PATH, encoding="utf-8").read())
    opus = {int(k): v for k, v in prior["curation_runs"][0]["llm"].items()}
    old_jev = {int(k): v for k, v in prior["curation_runs"][0]["jev"].items()}
    old_claims = prior["claim_runs"][0]["jev_results"]

    client = TypeSafeClient(timeout=120.0)
    try:
        return _run(client, draft, papers, recs, labels, prior, opus, old_jev, old_claims)
    finally:
        client.close()


def _run(client, draft, papers, recs, labels, prior, opus, old_jev, old_claims) -> int:
    out = {
        "started": datetime.now(timezone.utc).isoformat(),
        "question": QUESTION,
        "n_papers": len(papers),
        "n_claims": len(recs),
        "curation": {},
        "claims": {},
    }

    # ---- curation ----------------------------------------------------------
    print("Curation: 20 papers, several Jev question sets\n", flush=True)

    def run_per_paper(name: str, questions: dict, compose) -> dict[int, str]:
        labels_out: dict[int, str] = {}
        raw = []
        t0 = time.perf_counter()
        tin = tout = 0
        for i, paper in enumerate(papers, start=1):
            try:
                resp = client.system_one(
                    state=_paper_state(draft.topic, QUESTION, paper),
                    questions=questions,
                    model="jev-latest",
                )
            except Exception as exc:
                print(f"      [{i:2}] {name} error: {type(exc).__name__}: {exc}"[:180], flush=True)
                continue
            a_in, a_out = jb._jev_usage(resp)
            tin += a_in
            tout += a_out
            payload = {}
            for key, ans in resp.answers.items():
                if hasattr(ans, "noul"):
                    payload[key] = round(ans.noul, 3)
                elif hasattr(ans, "choice"):
                    payload[key] = ans.choice
                elif hasattr(ans, "score"):
                    payload[key] = round(ans.score, 3)
            labels_out[i] = compose(payload)
            raw.append({"i": i, "payload": payload, "label": labels_out[i]})
        sec = time.perf_counter() - t0
        summary = _gate_summary(opus, labels_out)
        summary.update(seconds=round(sec, 2), tokens_in=tin, tokens_out=tout,
                       est_usd=round(jb._usage_cost(tin, tout, (0.042, 0.0)), 6),
                       raw=raw)
        out["curation"][name] = summary
        flag = "PASS" if summary["critical_ok"] else "FAIL"
        print(
            f"  {name:<28} {summary['agree']}/{summary['n']}  "
            f"dir {summary['direct']}  tang {summary['tangential']}  "
            f"{summary['counts']}  {sec:.1f}s  {flag}",
            flush=True,
        )
        if summary["disagreements"]:
            for d in summary["disagreements"]:
                print(f"      [{d['i']:2}] opus {d['opus']:<11} -> {d['jev']}", flush=True)
        return labels_out

    # Baseline from the bakeoff (no new call)
    base = _gate_summary(opus, old_jev)
    out["curation"]["baseline_choice_short_topic"] = {
        **base, "seconds": None, "note": "from 10-run bakeoff, short topic only",
    }
    print(
        f"  {'baseline_choice_short_topic':<28} {base['agree']}/{base['n']}  "
        f"dir {base['direct']}  tang {base['tangential']}  {base['counts']}  "
        f"(cached)  {'PASS' if base['critical_ok'] else 'FAIL'}",
        flush=True,
    )

    run_per_paper(
        "choice_full_question",
        {"relevance": Choice(
            instructions=(
                "The briefing question is `question`. How directly does SOURCE "
                "(`paper.title` + `paper.abstract`) address that exact question? "
                "Judge the study itself, not how famous it is."
            ),
            criteria=jb.CURATE_CRITERIA,
        )},
        lambda p: p["relevance"],
    )
    run_per_paper("strict_choice", {"relevance": CURATE_CHOICE_STRICT}, lambda p: p["relevance"])
    run_per_paper("score_three_levels", {"relevance": CURATE_SCORE},
                  lambda p: compose_score(p["relevance"]))
    run_per_paper(
        "noul_compose",
        CURATE_NOULS,
        lambda p: compose_noul(p),
    )
    run_per_paper(
        "two_stage",
        {
            "shift_is_exposure": CURATE_NOULS["shift_is_exposure"],
            "relevance": CURATE_CHOICE_STRICT,
        },
        lambda p: compose_two_stage(p["shift_is_exposure"], p["relevance"]),
    )

    # Threshold sweep on the noul payloads we just stored
    raw = out["curation"]["noul_compose"]["raw"]
    print("\n  noul threshold sweep (shift, bg, same, outcome) -> tang kept, agree, PASS?", flush=True)
    sweeps = []
    for t_shift, t_bg, t_same, t_out in [
        (0.5, 0.5, 0.5, 0.5),
        (0.5, 0.55, 0.55, 0.5),
        (0.4, 0.6, 0.6, 0.5),
        (0.6, 0.5, 0.5, 0.5),
        (0.5, 0.7, 0.5, 0.5),
    ]:
        got = {}
        for row in raw:
            got[row["i"]] = compose_noul(
                row["payload"], t_shift=t_shift, t_bg=t_bg, t_same=t_same, t_out=t_out,
            )
        g = _gate_summary(opus, got)
        sweeps.append({
            "thresholds": [t_shift, t_bg, t_same, t_out],
            **{k: g[k] for k in ("agree", "n", "direct", "tangential", "critical_ok", "counts")},
        })
        print(
            f"    {t_shift}/{t_bg}/{t_same}/{t_out}  "
            f"agree {g['agree']}/{g['n']}  tang {g['tangential']}  "
            f"{g['counts']}  {'PASS' if g['critical_ok'] else 'FAIL'}",
            flush=True,
        )
    out["curation"]["noul_compose"]["sweeps"] = sweeps

    # ---- claims ------------------------------------------------------------
    print("\nClaims: 12 sentences\n", flush=True)

    def run_claims(name: str, questions: dict, compose) -> None:
        results = []
        t0 = time.perf_counter()
        tin = tout = 0
        for rec in recs:
            try:
                resp = client.system_one(
                    state={
                        "question": QUESTION,
                        "sentence": rec["sentence"],
                        "excerpt": rec["excerpt"],
                        "cited": rec["cited"],
                    },
                    questions=questions,
                    model="jev-latest",
                )
            except Exception as exc:
                print(f"      claim {rec['id']} {name} error: {type(exc).__name__}: {exc}"[:180], flush=True)
                results.append({
                    "id": rec["id"], "verdict": "error",
                    "truncated": rec["truncated"], "excerpt_chars": rec["excerpt_chars"],
                })
                continue
            a_in, a_out = jb._jev_usage(resp)
            tin += a_in
            tout += a_out
            payload = {}
            for key, ans in resp.answers.items():
                if hasattr(ans, "noul"):
                    payload[key] = round(ans.noul, 3)
                elif hasattr(ans, "choice"):
                    payload[key] = {
                        "choice": ans.choice,
                        "confidence": round(ans.confidence, 3),
                    }
            verdict = compose(payload)
            results.append({
                "id": rec["id"], "verdict": verdict, "payload": payload,
                "truncated": rec["truncated"], "excerpt_chars": rec["excerpt_chars"],
            })
        sec = time.perf_counter() - t0
        report = score_claims(labels, results)
        metrics = jb._claim_metrics(report)
        metrics.update(seconds=round(sec, 2), tokens_in=tin, tokens_out=tout,
                       est_usd=round(jb._usage_cost(tin, tout, (0.042, 0.0)), 6),
                       results=results)
        out["claims"][name] = metrics
        print(
            f"  {name:<28} exact {metrics['exact']}/{metrics['n']}  "
            f"OR {metrics['overreach_recall']}  FP {metrics['false_overreach']}  "
            f"{sec:.1f}s",
            flush=True,
        )
        for d in metrics["disagreements"]:
            print(f"      [{d['id']:2}] human {d['human']:<11} -> {d['model']}", flush=True)

    old_metrics = jb._claim_metrics(score_claims(labels, old_claims))
    out["claims"]["baseline_choice"] = {
        **old_metrics, "seconds": None, "note": "from 10-run bakeoff run 1",
    }
    print(
        f"  {'baseline_choice':<28} exact {old_metrics['exact']}/{old_metrics['n']}  "
        f"OR {old_metrics['overreach_recall']}  FP {old_metrics['false_overreach']}  "
        f"(cached)",
        flush=True,
    )

    run_claims("relation_choice", {"relation": CLAIM_RELATION},
               lambda p: compose_claim_relation(p["relation"]["choice"]))
    run_claims(
        "noul_clause",
        CLAIM_UNSUPPORTED,
        lambda p: compose_claim_noul(p["core_in_excerpt"], p["unsupported_clause"]),
    )

    out["updated"] = datetime.now(timezone.utc).isoformat()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

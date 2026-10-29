#!/usr/bin/env python
"""Bake off Jev against the live curator and the claim probe, over N local runs.

    python tools/jev_bakeoff.py --smoke
    python tools/jev_bakeoff.py --runs 10

The paper pool and the 12 claim sentences are frozen from
`tests/replays/2026-09-05-night-shift-health-risk.json` and
`tools/claim_probe_labels.json`. Nothing is re-searched. Each run calls:

1. `curate_sources` (current LLM curator, default Opus via OpenRouter)
2. Jev Choice, one question per paper, batched in one TypeSafe request
3. `claim_probe.run_probe` (Luna)
4. Jev Choice, one question per cited sentence

Acceptance for (1) is the same rule as `compare_curation.py`: `direct` and
`tangential` must both be stable against the LLM labels. (2) is scored against
the hand labels, the same way `claim_probe.py --score` does.

Writes `tools/jev_bakeoff_results.json` after every run so a stop is not a loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from typesafe_sdk import Choice, TypeSafeClient  # noqa: E402

from articlegen.pipeline import Draft  # noqa: E402
from articlegen.writer import _CURATE_SYSTEM, curate_sources  # noqa: E402
from claim_probe import (  # noqa: E402
    DEFAULT_MANIFEST,
    PROBE_MODEL,
    _PROBE_SYSTEM,
    _tee_tokens,
    pairs,
    run_probe,
    score,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "claim_probe_labels.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "jev_bakeoff_results.json")

LABELS = ("direct", "related", "tangential")
CRITICAL = ("direct", "tangential")
JEV_MODEL = "jev-latest"
JEV_PRICE_IN = 0.042  # $ / 1M input tokens, TypeSafe cookbook 2026-09
OPUS_PRICE = (5.0, 25.0)  # OpenRouter anthropic/claude-opus-5
LUNA_PRICE = (0.20, 1.20)  # OpenRouter openai/gpt-5.6-luna standard route

CURATE_CRITERIA = {
    "direct": (
        "The study is actually about the exact topic: right population, "
        "right intervention or mechanism, right question."
    ),
    "related": (
        "Adjacent — the same intervention in a different population or "
        "disorder, or the right population but a different question. "
        "Useful for context, not direct evidence."
    ),
    "tangential": (
        "Only good for background or framing. A famous review that never "
        "studies the specific topic is tangential, not direct."
    ),
}

CLAIM_CRITERIA = {
    "supported": (
        "The excerpt states the claim or plainly entails it. Different "
        "wording is fine if the meaning holds."
    ),
    "overreach": (
        "The sentence goes further than the excerpt: an association stated "
        "as a cause, a wider population than the excerpt studied, a dropped "
        "qualifier or hedge, a certainty the excerpt does not carry, or an "
        "outcome the excerpt never reports."
    ),
    "unclear": "The excerpt does not settle it either way.",
}


def _load_draft(path: str) -> Draft:
    with open(path, encoding="utf-8") as f:
        return Draft.from_dict(json.load(f))


def _load_labels() -> list[dict]:
    with open(LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


def _usage_cost(tokens_in: int, tokens_out: int, price: tuple[float, float]) -> float:
    return tokens_in / 1e6 * price[0] + tokens_out / 1e6 * price[1]


def _mode(values: list[str]) -> str | None:
    held = [v for v in values if v]
    if not held:
        return None
    return Counter(held).most_common(1)[0][0]


def _stable_fraction(runs: list[dict[int, str]], keys: list[int]) -> float:
    """Share of items whose label was identical on every run that labelled them."""
    if not keys or not runs:
        return 0.0
    stable = 0
    for key in keys:
        labels = [run.get(key) for run in runs if run.get(key)]
        if labels and len(set(labels)) == 1:
            stable += 1
    return stable / len(keys)


def _counts(relevance: dict[int, str]) -> dict[str, int]:
    return {lab: sum(1 for v in relevance.values() if v == lab) for lab in LABELS}


def _gate_row(reference: dict[int, str], other: dict[int, str]) -> dict:
    rows = [(i, reference[i], other[i])
            for i in sorted(set(reference) & set(other))]
    n = len(rows)
    agree = sum(1 for _, a, b in rows if a == b)
    retained = {}
    for label in LABELS:
        held = [(a, b) for _, a, b in rows if a == label]
        retained[label] = {
            "kept": sum(1 for a, b in held if b == label),
            "held": len(held),
        }
    toward = sum(1 for _, a, b in rows if a != b and b == "related")
    away = sum(1 for _, a, b in rows if a != b and a == "related")
    critical_ok = all(
        retained[lab]["kept"] == retained[lab]["held"]
        for lab in CRITICAL
        if retained[lab]["held"]
    )
    return {
        "n": n,
        "agree": agree,
        "retained": retained,
        "toward_related": toward,
        "away_related": away,
        "critical_ok": critical_ok,
    }


def _jev_usage(response) -> tuple[int, int]:
    usage = response.usage
    return (usage.input_tokens or 0), (usage.output_tokens or 0)


def _claim_question(record: dict) -> Choice:
    return Choice(
        instructions=(
            f"{_PROBE_SYSTEM.strip()}\n\n"
            f"The sentence is `sentence`. It cites SOURCE {record['cited']}. "
            "The excerpt is `excerpt`. Use only those two fields."
        ),
        criteria=CLAIM_CRITERIA,
    )


def curate_jev(client: TypeSafeClient, topic: str, papers: list, *,
               batch: int = 20, model: str = JEV_MODEL
               ) -> tuple[dict[int, str], dict[int, float], int, int]:
    """Label papers with Jev. Batches so a 20-question call can fall back."""
    state_papers = [
        {
            "index": i,
            "title": p.title,
            "year": p.year,
            "abstract": p.abstract or "",
        }
        for i, p in enumerate(papers, start=1)
    ]
    relevance: dict[int, str] = {}
    confidence: dict[int, float] = {}
    tokens_in = tokens_out = 0
    n = len(papers)
    start = 1
    while start <= n:
        end = min(start + batch - 1, n)
        chunk = state_papers[start - 1:end]
        questions = _curate_questions_range(start, end)
        try:
            response = client.system_one(
                state={"topic": topic, "papers": chunk},
                questions=questions,
                model=model,
            )
        except Exception:
            if batch <= 1:
                raise
            return curate_jev(client, topic, papers, batch=max(1, batch // 2),
                              model=model)
        tin, tout = _jev_usage(response)
        tokens_in += tin
        tokens_out += tout
        for i in range(start, end + 1):
            answer = response.choices.get(f"p{i}")
            if answer is None or answer.choice not in LABELS:
                continue
            relevance[i] = answer.choice
            confidence[i] = answer.confidence
        start = end + 1
    return relevance, confidence, tokens_in, tokens_out


def _curate_questions_range(start: int, end: int) -> dict:
    questions = {}
    for i in range(start, end + 1):
        questions[f"p{i}"] = Choice(
            instructions=(
                f"{_CURATE_SYSTEM.strip()}\n\n"
                f"How directly does SOURCE {i} (`papers` item whose `index` is {i}) "
                "address `topic`? Judge the study itself, not how famous or "
                "highly cited it is."
            ),
            criteria=CURATE_CRITERIA,
        )
    return questions


def probe_jev(client: TypeSafeClient, records: list[dict], *,
              model: str = JEV_MODEL) -> tuple[list[dict], int, int]:
    """One TypeSafe call per cited sentence, matching run_probe's unit of work."""
    results = []
    tokens_in = tokens_out = 0
    for record in records:
        try:
            response = client.system_one(
                state={
                    "sentence": record["sentence"],
                    "excerpt": record["excerpt"],
                    "cited": record["cited"],
                },
                questions={"verdict": _claim_question(record)},
                model=model,
            )
            tin, tout = _jev_usage(response)
            tokens_in += tin
            tokens_out += tout
            answer = response.choices["verdict"]
            results.append({
                "id": record["id"],
                "verdict": answer.choice,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
                "truncated": record["truncated"],
                "excerpt_chars": record["excerpt_chars"],
            })
        except Exception as exc:
            results.append({
                "id": record["id"],
                "verdict": "error",
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "truncated": record["truncated"],
                "excerpt_chars": record["excerpt_chars"],
            })
    return results, tokens_in, tokens_out


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    rule = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    print(line)
    print(rule)
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")


def _fmt_retained(block: dict, label: str) -> str:
    held = block[label]["held"]
    kept = block[label]["kept"]
    if not held:
        return "—"
    return f"{kept}/{held}"


def _summarise(doc: dict) -> None:
    curation_runs = doc.get("curation_runs") or []
    claim_runs = doc.get("claim_runs") or []
    n_papers = doc.get("n_papers") or 0
    paper_ids = list(range(1, n_papers + 1))

    print("\n" + "=" * 72)
    print("CURATION  (Jev vs live curate_sources, frozen replay pool)")
    print("=" * 72)
    headers = [
        "run", "agree", "direct", "related", "tangential",
        "to-rel", "from-rel", "gate", "llm $", "jev $",
    ]
    rows = []
    for run in curation_runs:
        g = run["gate"]
        rows.append([
            str(run["run"]),
            f"{g['agree']}/{g['n']}",
            _fmt_retained(g["retained"], "direct"),
            _fmt_retained(g["retained"], "related"),
            _fmt_retained(g["retained"], "tangential"),
            str(g["toward_related"]),
            str(g["away_related"]),
            "PASS" if g["critical_ok"] else "FAIL",
            f"{run['llm_est_usd']:.3f}",
            f"{run['jev_est_usd']:.4f}",
        ])
    if rows:
        _print_table(headers, rows)

        llm_maps = [{int(k): v for k, v in run["llm"].items()} for run in curation_runs]
        jev_maps = [{int(k): v for k, v in run["jev"].items()} for run in curation_runs]
        print()
        print(f"LLM self-stable papers (same label all {len(curation_runs)} runs): "
              f"{_stable_fraction(llm_maps, paper_ids):.0%} of {n_papers}")
        print(f"Jev self-stable papers (same label all {len(curation_runs)} runs): "
              f"{_stable_fraction(jev_maps, paper_ids):.0%} of {n_papers}")
        llm_direct = [run["llm_counts"]["direct"] for run in curation_runs]
        jev_direct = [run["jev_counts"]["direct"] for run in curation_runs]
        print(f"LLM direct count: mean {sum(llm_direct)/len(llm_direct):.1f} "
              f"(min {min(llm_direct)}, max {max(llm_direct)})")
        print(f"Jev direct count: mean {sum(jev_direct)/len(jev_direct):.1f} "
              f"(min {min(jev_direct)}, max {max(jev_direct)})")
        gates = sum(1 for run in curation_runs if run["gate"]["critical_ok"])
        print(f"compare_curation gate (direct+tangential retained vs LLM): "
              f"{gates}/{len(curation_runs)} runs PASS")
        llm_cost = sum(run["llm_est_usd"] for run in curation_runs)
        jev_cost = sum(run["jev_est_usd"] for run in curation_runs)
        print(f"Estimated spend over {len(curation_runs)} runs: "
              f"LLM ${llm_cost:.3f}  Jev ${jev_cost:.4f}")

        # Majority vote vs each other
        llm_maj = {i: _mode([m.get(i) for m in llm_maps]) for i in paper_ids}
        jev_maj = {i: _mode([m.get(i) for m in jev_maps]) for i in paper_ids}
        maj_gate = _gate_row(
            {i: v for i, v in llm_maj.items() if v},
            {i: v for i, v in jev_maj.items() if v},
        )
        print(f"Majority-vote agreement: {maj_gate['agree']}/{maj_gate['n']}  "
              f"direct {_fmt_retained(maj_gate['retained'], 'direct')}  "
              f"tangential {_fmt_retained(maj_gate['retained'], 'tangential')}  "
              f"gate {'PASS' if maj_gate['critical_ok'] else 'FAIL'}")

    print("\n" + "=" * 72)
    print("CLAIM PROBE  (Jev vs Luna, scored on hand labels)")
    print("=" * 72)
    headers = [
        "run", "luna exact", "jev exact", "luna coll.", "jev coll.",
        "luna OR rec", "jev OR rec", "luna FP", "jev FP",
        "luna $", "jev $",
    ]
    rows = []
    for run in claim_runs:
        rows.append([
            str(run["run"]),
            f"{run['luna']['exact']}/{run['luna']['n']}",
            f"{run['jev']['exact']}/{run['jev']['n']}",
            f"{run['luna']['collapsed']}/{run['luna']['n']}",
            f"{run['jev']['collapsed']}/{run['jev']['n']}",
            run["luna"]["overreach_recall"],
            run["jev"]["overreach_recall"],
            str(run["luna"]["false_overreach"]),
            str(run["jev"]["false_overreach"]),
            f"{run['luna_est_usd']:.3f}",
            f"{run['jev_est_usd']:.4f}",
        ])
    if rows:
        _print_table(headers, rows)
        n = len(claim_runs)
        def avg(key_path):
            vals = []
            for run in claim_runs:
                node = run
                for k in key_path:
                    node = node[k]
                # "1/2" -> 0.5
                if isinstance(node, str) and "/" in node:
                    a, b = node.split("/", 1)
                    vals.append(int(a) / int(b) if int(b) else 0.0)
                else:
                    vals.append(float(node))
            return sum(vals) / len(vals)

        print()
        print(f"Mean exact agreement vs human:  Luna {avg(['luna','exact_rate']):.0%}   "
              f"Jev {avg(['jev','exact_rate']):.0%}")
        print(f"Mean collapsed (supported vs not): Luna {avg(['luna','collapsed_rate']):.0%}   "
              f"Jev {avg(['jev','collapsed_rate']):.0%}")
        luna_cost = sum(run["luna_est_usd"] for run in claim_runs)
        jev_cost = sum(run["jev_est_usd"] for run in claim_runs)
        print(f"Estimated spend over {n} runs: Luna ${luna_cost:.3f}  Jev ${jev_cost:.4f}")
        print("OR rec = overreach recall against the two hand-labelled overreaches "
              "(pairs 3 and 11). FP = supported→overreach.")


def _claim_metrics(report: dict) -> dict:
    n = report["n"]
    exact = report["exact"]
    collapsed = report["collapsed"]
    confusion = report["confusion"]
    human_or = confusion.get("overreach", {})
    or_held = sum(human_or.values())
    or_caught = human_or.get("overreach", 0)
    false_or = confusion.get("supported", {}).get("overreach", 0)
    return {
        "n": n,
        "exact": exact,
        "collapsed": collapsed,
        "exact_rate": exact / n if n else 0.0,
        "collapsed_rate": collapsed / n if n else 0.0,
        "overreach_recall": f"{or_caught}/{or_held}" if or_held else "—",
        "false_overreach": false_or,
        "confusion": confusion,
        "disagreements": [
            {"id": d["id"], "human": d["human"], "model": d["model"]}
            for d in report["disagreements"]
        ],
    }


def _save(doc: dict) -> None:
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, RESULTS_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-papers", type=int, default=20,
                    help="Frozen pool size (compare_curation default is 20).")
    ap.add_argument("--curation-model", default=None,
                    help="OpenRouter/Anthropic slug for curate_sources. Default: Opus.")
    ap.add_argument("--claim-model", default=PROBE_MODEL)
    ap.add_argument("--jev-model", default=JEV_MODEL)
    ap.add_argument("--smoke", action="store_true",
                    help="1 run, 4 papers, 2 claims. Check the keys before a 10-run.")
    ap.add_argument("--resume", action="store_true",
                    help="Keep completed runs already in the results file.")
    ap.add_argument("--skip-curation", action="store_true")
    ap.add_argument("--skip-claims", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.runs = 1
        args.max_papers = 4

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY.")
        return 2
    if not args.skip_curation and not (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        or args.curation_model
    ):
        print("Set OPENROUTER_API_KEY / ANTHROPIC_API_KEY, or pass --curation-model.")
        return 2
    if not args.skip_claims and not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY — the Luna claim probe uses OpenRouter.")
        return 2

    draft = _load_draft(DEFAULT_MANIFEST)
    papers = draft.papers[:args.max_papers]
    labels = _load_labels()
    recs = pairs(draft)
    if args.smoke:
        recs = recs[:2]
        labels = [e for e in labels if e["id"] in {r["id"] for r in recs}]

    print(f"Pool: {len(papers)} papers from {draft.topic!r}")
    print(f"Claims: {len(recs)} cited sentences, {sum(1 for e in labels if e.get('human'))} hand labels")
    print(f"Runs: {args.runs}   Jev={args.jev_model}   curator={args.curation_model or 'default/Opus'}   "
          f"claim={args.claim_model}")
    print(f"Results -> {RESULTS_PATH}", flush=True)

    doc = {
        "started": datetime.now(timezone.utc).isoformat(),
        "topic": draft.topic,
        "manifest": DEFAULT_MANIFEST,
        "n_papers": len(papers),
        "n_claims": len(recs),
        "runs_requested": args.runs,
        "curation_model": args.curation_model or "default",
        "claim_model": args.claim_model,
        "jev_model": args.jev_model,
        "smoke": args.smoke,
        "paper_titles": [p.title for p in papers],
        "curation_runs": [],
        "claim_runs": [],
    }
    already = 0
    if args.resume and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("n_papers") == len(papers) and prev.get("n_claims") == len(recs):
            doc = prev
            already = max(len(doc.get("curation_runs") or []),
                          len(doc.get("claim_runs") or []))
            print(f"Resuming; {already} run(s) already stored.", flush=True)

    client = TypeSafeClient(timeout=120.0)

    for run_i in range(already + 1, args.runs + 1):
        print(f"\n----- run {run_i}/{args.runs} -----", flush=True)
        curation_row = None
        claim_row = None

        if not args.skip_curation:
            t0 = time.perf_counter()
            llm_result, llm_in, llm_out = _tee_tokens(
                lambda: curate_sources(
                    draft.topic, papers, model=args.curation_model,
                )
            )
            llm_s = time.perf_counter() - t0
            llm_rel = {int(k): v for k, v in (llm_result.get("relevance") or {}).items()}
            print(f"  LLM curate {len(llm_rel)} labels in {llm_s:.1f}s "
                  f"in={llm_in} out={llm_out}  counts={_counts(llm_rel)}", flush=True)

            t0 = time.perf_counter()
            jev_rel, jev_conf, jev_in, jev_out = curate_jev(
                client, draft.topic, papers, model=args.jev_model,
            )
            jev_s = time.perf_counter() - t0
            print(f"  Jev curate {len(jev_rel)} labels in {jev_s:.1f}s "
                  f"in={jev_in} out={jev_out}  counts={_counts(jev_rel)}", flush=True)

            gate = _gate_row(llm_rel, jev_rel)
            print(f"  agree {gate['agree']}/{gate['n']}  "
                  f"direct {_fmt_retained(gate['retained'], 'direct')}  "
                  f"tangential {_fmt_retained(gate['retained'], 'tangential')}  "
                  f"{'PASS' if gate['critical_ok'] else 'FAIL'}", flush=True)

            curation_row = {
                "run": run_i,
                "llm": {str(k): v for k, v in llm_rel.items()},
                "jev": {str(k): v for k, v in jev_rel.items()},
                "jev_confidence": {str(k): v for k, v in jev_conf.items()},
                "llm_counts": _counts(llm_rel),
                "jev_counts": _counts(jev_rel),
                "gate": gate,
                "llm_tokens_in": llm_in,
                "llm_tokens_out": llm_out,
                "jev_tokens_in": jev_in,
                "jev_tokens_out": jev_out,
                "llm_seconds": round(llm_s, 2),
                "jev_seconds": round(jev_s, 2),
                "llm_est_usd": round(_usage_cost(llm_in, llm_out, OPUS_PRICE), 4),
                "jev_est_usd": round(_usage_cost(jev_in, jev_out, (JEV_PRICE_IN, 0.0)), 6),
            }

        if not args.skip_claims:
            t0 = time.perf_counter()
            luna_results, luna_in, luna_out = _tee_tokens(
                lambda: run_probe(recs, args.claim_model, topic=draft.topic)
            )
            luna_s = time.perf_counter() - t0
            luna_report = score(labels, luna_results)
            luna_m = _claim_metrics(luna_report)
            print(f"  Luna claims exact {luna_m['exact']}/{luna_m['n']} "
                  f"collapsed {luna_m['collapsed']}/{luna_m['n']}  "
                  f"OR {luna_m['overreach_recall']}  {luna_s:.1f}s", flush=True)

            t0 = time.perf_counter()
            jev_results, jev_in, jev_out = probe_jev(
                client, recs, model=args.jev_model)
            jev_s = time.perf_counter() - t0
            jev_report = score(labels, jev_results)
            jev_m = _claim_metrics(jev_report)
            print(f"  Jev  claims exact {jev_m['exact']}/{jev_m['n']} "
                  f"collapsed {jev_m['collapsed']}/{jev_m['n']}  "
                  f"OR {jev_m['overreach_recall']}  {jev_s:.1f}s", flush=True)

            claim_row = {
                "run": run_i,
                "luna": luna_m,
                "jev": jev_m,
                "luna_results": luna_results,
                "jev_results": jev_results,
                "luna_tokens_in": luna_in,
                "luna_tokens_out": luna_out,
                "jev_tokens_in": jev_in,
                "jev_tokens_out": jev_out,
                "luna_seconds": round(luna_s, 2),
                "jev_seconds": round(jev_s, 2),
                "luna_est_usd": round(_usage_cost(luna_in, luna_out, LUNA_PRICE), 4),
                "jev_est_usd": round(_usage_cost(jev_in, jev_out, (JEV_PRICE_IN, 0.0)), 6),
            }

        if curation_row:
            doc["curation_runs"].append(curation_row)
        if claim_row:
            doc["claim_runs"].append(claim_row)
        doc["updated"] = datetime.now(timezone.utc).isoformat()
        _save(doc)

    _summarise(doc)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

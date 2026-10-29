#!/usr/bin/env python
"""Time / cost / quality ladder: several LLMs + Jev, same frozen pool.

    python tools/jev_ladder.py --runs 3

Dumb → smart, same 20 papers and 12 hand-labelled claims as jev_bakeoff.py.
Curation is scored against the Opus labels from the 10-run bakeoff (stable).
Claims are scored against the hand labels.

Writes tools/jev_ladder_results.json after every model+run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from typesafe_sdk import TypeSafeClient  # noqa: E402

from articlegen.pipeline import Draft  # noqa: E402
from articlegen.writer import curate_sources  # noqa: E402
import jev_bakeoff as jb  # noqa: E402
from claim_probe import DEFAULT_MANIFEST, _tee_tokens, pairs, run_probe, score  # noqa: E402

RESULTS_PATH = os.path.join(_HERE, "jev_ladder_results.json")
PRIOR_PATH = os.path.join(_HERE, "jev_bakeoff_results.json")

# OpenRouter list prices, $ / 1M tokens. Jev from the TypeSafe cookbook.
PRICES = {
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "jev-latest": (0.042, 0.00),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-fable-5": (10.00, 50.00),
}

# Cheap → expensive. `kind` is how we call it.
LADDER = [
    {"id": "meta-llama/llama-3.3-70b-instruct", "band": "dumber", "kind": "llm"},
    {"id": "openai/gpt-5.6-luna", "band": "cheap", "kind": "llm"},
    {"id": "google/gemini-2.5-flash", "band": "cheap-mid", "kind": "llm"},
    {"id": "jev-latest", "band": "system-one", "kind": "jev"},
    {"id": "anthropic/claude-sonnet-5", "band": "mid", "kind": "llm"},
    {"id": "anthropic/claude-opus-5", "band": "smart (current curator)", "kind": "llm"},
    {"id": "anthropic/claude-fable-5", "band": "smarter", "kind": "llm"},
]


def _price(model: str) -> tuple[float, float]:
    return PRICES.get(model, (0.0, 0.0))


def _load_opus_ref() -> dict[int, str]:
    if not os.path.exists(PRIOR_PATH):
        return {}
    with open(PRIOR_PATH, encoding="utf-8") as f:
        prior = json.load(f)
    runs = prior.get("curation_runs") or []
    if not runs:
        return {}
    return {int(k): v for k, v in runs[0]["llm"].items()}


def _save(doc: dict) -> None:
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, RESULTS_PATH)


def _done_runs(doc: dict, model_id: str) -> int:
    block = (doc.get("models") or {}).get(model_id) or {}
    return min(len(block.get("curation") or []), len(block.get("claims") or []))


def run_curation(model: dict, topic: str, papers: list, client) -> dict:
    slug, kind = model["id"], model["kind"]
    t0 = time.perf_counter()
    error = ""
    if kind == "jev":
        rel, conf, tin, tout = jb.curate_jev(client, topic, papers, model=slug)
        seconds = time.perf_counter() - t0
        return {
            "relevance": {str(k): v for k, v in rel.items()},
            "confidence": {str(k): v for k, v in conf.items()},
            "counts": jb._counts(rel),
            "tokens_in": tin,
            "tokens_out": tout,
            "seconds": round(seconds, 2),
            "est_usd": round(jb._usage_cost(tin, tout, _price(slug)), 6),
            "error": "",
        }
    result, tin, tout = _tee_tokens(
        lambda: curate_sources(topic, papers, model=slug)
    )
    seconds = time.perf_counter() - t0
    rel = {int(k): v for k, v in (result.get("relevance") or {}).items()}
    error = result.get("error") or ""
    return {
        "relevance": {str(k): v for k, v in rel.items()},
        "counts": jb._counts(rel),
        "tokens_in": tin,
        "tokens_out": tout,
        "seconds": round(seconds, 2),
        "est_usd": round(jb._usage_cost(tin, tout, _price(slug)), 6),
        "error": error,
    }


def run_claims(model: dict, recs: list, labels: list, topic: str, client) -> dict:
    slug, kind = model["id"], model["kind"]
    t0 = time.perf_counter()
    if kind == "jev":
        results, tin, tout = jb.probe_jev(client, recs, model=slug)
    else:
        results, tin, tout = _tee_tokens(
            lambda: run_probe(recs, slug, topic=topic)
        )
    seconds = time.perf_counter() - t0
    report = score(labels, results)
    metrics = jb._claim_metrics(report)
    n_err = sum(1 for r in results if r.get("verdict") == "error")
    return {
        **metrics,
        "results": results,
        "tokens_in": tin,
        "tokens_out": tout,
        "seconds": round(seconds, 2),
        "est_usd": round(jb._usage_cost(tin, tout, _price(slug)), 6),
        "errors": n_err,
    }


def _summarise(doc: dict) -> None:
    ref = {int(k): v for k, v in (doc.get("opus_ref") or {}).items()}
    print("\n" + "=" * 78)
    print("CURATION  (vs frozen Opus-5 labels from the 10-run bakeoff)")
    print("=" * 78)
    headers = [
        "model", "band", "agree", "direct", "tang", "D/R/T",
        "stable", "sec", "$/run", "err",
    ]
    rows = []
    for spec in LADDER:
        block = (doc.get("models") or {}).get(spec["id"]) or {}
        cur = block.get("curation") or []
        if not cur:
            continue
        maps = [{int(k): v for k, v in r["relevance"].items()} for r in cur
                if r.get("relevance")]
        if not maps:
            rows.append([spec["id"].split("/")[-1], spec["band"], "—", "—", "—",
                         "—", "—", "—", "—", "all failed"])
            continue
        gates = [jb._gate_row(ref, m) for m in maps] if ref else []
        agree = f"{sum(g['agree'] for g in gates)/len(gates):.0f}/{gates[0]['n']}" if gates else "—"
        direct = jb._fmt_retained(gates[0]["retained"], "direct") if gates else "—"
        tang = jb._fmt_retained(gates[0]["retained"], "tangential") if gates else "—"
        # counts from last successful
        c = cur[-1]["counts"]
        dtr = f"{c.get('direct', 0)}/{c.get('related', 0)}/{c.get('tangential', 0)}"
        keys = list(range(1, doc["n_papers"] + 1))
        stable = f"{jb._stable_fraction(maps, keys):.0%}"
        sec = sum(r["seconds"] for r in cur) / len(cur)
        usd = sum(r["est_usd"] for r in cur) / len(cur)
        errs = sum(1 for r in cur if r.get("error") or not r.get("relevance"))
        rows.append([
            spec["id"].split("/")[-1][:22],
            spec["band"][:18],
            agree, direct, tang, dtr, stable,
            f"{sec:.1f}", f"{usd:.4f}", str(errs),
        ])
    jb._print_table(headers, rows)

    print("\n" + "=" * 78)
    print("CLAIMS  (vs 12 hand labels; OR rec = pairs 3+11)")
    print("=" * 78)
    headers = [
        "model", "band", "exact", "coll.", "OR rec", "FP",
        "sec", "$/run", "err",
    ]
    rows = []
    for spec in LADDER:
        block = (doc.get("models") or {}).get(spec["id"]) or {}
        claims = block.get("claims") or []
        if not claims:
            continue
        n = max(r["n"] for r in claims) or 12
        exact = sum(r["exact"] for r in claims) / len(claims)
        coll = sum(r["collapsed"] for r in claims) / len(claims)
        # OR rec as mean of caught/held
        caught = held = fp = 0
        for r in claims:
            rec = r.get("overreach_recall") or "—"
            if "/" in rec:
                a, b = rec.split("/", 1)
                caught += int(a)
                held += int(b)
            fp += r.get("false_overreach") or 0
        or_s = f"{caught}/{held}" if held else "—"
        sec = sum(r["seconds"] for r in claims) / len(claims)
        usd = sum(r["est_usd"] for r in claims) / len(claims)
        errs = sum(r.get("errors") or 0 for r in claims)
        rows.append([
            spec["id"].split("/")[-1][:22],
            spec["band"][:18],
            f"{exact:.1f}/{n}", f"{coll:.1f}/{n}", or_s,
            f"{fp/len(claims):.1f}", f"{sec:.1f}", f"{usd:.4f}", str(errs),
        ])
    jb._print_table(headers, rows)

    print("\nD/R/T = direct/related/tangential counts (last successful run).")
    print("agree/direct/tang are vs the frozen Opus labels, not vs each other.")
    print("$/run is estimated from billed tokens and OpenRouter list prices.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of model ids (substring match).")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY.")
        return 2
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY.")
        return 2

    draft = jb._load_draft(DEFAULT_MANIFEST)
    papers = draft.papers[:args.max_papers]
    labels = jb._load_labels()
    recs = pairs(draft)
    opus_ref = _load_opus_ref()
    specs = LADDER
    if args.only:
        specs = [s for s in LADDER if any(tok in s["id"] for tok in args.only)]

    print(f"Pool: {len(papers)} papers, {len(recs)} claims, {args.runs} run(s) each")
    print(f"Models: {', '.join(s['id'] for s in specs)}")
    print(f"Opus reference labels: {len(opus_ref)}")
    print(f"Results -> {RESULTS_PATH}", flush=True)

    doc = {
        "started": datetime.now(timezone.utc).isoformat(),
        "topic": draft.topic,
        "n_papers": len(papers),
        "n_claims": len(recs),
        "runs_requested": args.runs,
        "opus_ref": {str(k): v for k, v in opus_ref.items()},
        "paper_titles": [p.title for p in papers],
        "models": {},
    }
    if args.resume and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("n_papers") == len(papers):
            doc = prev
            print("Resuming from existing ladder file.", flush=True)

    client = TypeSafeClient(timeout=120.0)

    for spec in specs:
        slug = spec["id"]
        doc.setdefault("models", {}).setdefault(slug, {"curation": [], "claims": []})
        already = _done_runs(doc, slug)
        if already >= args.runs:
            print(f"\n===== {slug} already has {already} runs, skip =====", flush=True)
            continue
        print(f"\n===== {spec['band']}: {slug}  ({already} done) =====", flush=True)
        for run_i in range(already + 1, args.runs + 1):
            print(f"  -- run {run_i}/{args.runs} --", flush=True)
            try:
                cur = run_curation(spec, draft.topic, papers, client)
            except Exception as exc:
                cur = {
                    "relevance": {}, "counts": {}, "tokens_in": 0, "tokens_out": 0,
                    "seconds": 0, "est_usd": 0, "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            cur["run"] = run_i
            rel = {int(k): v for k, v in cur.get("relevance", {}).items()}
            if opus_ref and rel:
                gate = jb._gate_row(opus_ref, rel)
                cur["gate"] = gate
                print(f"    curate {len(rel)} in {cur['seconds']:.1f}s  "
                      f"${cur['est_usd']:.4f}  {cur['counts']}  "
                      f"agree {gate['agree']}/{gate['n']}  "
                      f"{'PASS' if gate['critical_ok'] else 'FAIL'}"
                      + (f"  ERR {cur['error'][:80]}" if cur.get("error") else ""),
                      flush=True)
            else:
                print(f"    curate {len(rel)} in {cur['seconds']:.1f}s  "
                      f"${cur['est_usd']:.4f}  {cur['counts']}"
                      + (f"  ERR {cur['error'][:80]}" if cur.get("error") else ""),
                      flush=True)

            try:
                claims = run_claims(spec, recs, labels, draft.topic, client)
            except Exception as exc:
                claims = {
                    "n": 0, "exact": 0, "collapsed": 0, "exact_rate": 0,
                    "collapsed_rate": 0, "overreach_recall": "—",
                    "false_overreach": 0, "results": [],
                    "tokens_in": 0, "tokens_out": 0, "seconds": 0,
                    "est_usd": 0, "errors": 12,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            claims["run"] = run_i
            print(f"    claims exact {claims.get('exact')}/{claims.get('n')}  "
                  f"OR {claims.get('overreach_recall')}  "
                  f"FP {claims.get('false_overreach')}  "
                  f"{claims['seconds']:.1f}s  ${claims['est_usd']:.4f}  "
                  f"err={claims.get('errors', 0)}", flush=True)

            doc["models"][slug]["curation"].append(cur)
            doc["models"][slug]["claims"].append(claims)
            doc["updated"] = datetime.now(timezone.utc).isoformat()
            _save(doc)

    _summarise(doc)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""10-run Gemini 3.8 Flash on the frozen night-shift pool.

Same curator + claim probe as the ladder, separate results file so the
existing 7-model ladder is not overwritten.

    python tools/gemini38_runs.py
    python tools/gemini38_runs.py --runs 10 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import jev_bakeoff as jb  # noqa: E402
import jev_ladder as jl  # noqa: E402
from claim_probe import DEFAULT_MANIFEST, pairs  # noqa: E402

MODEL = "google/gemini-3.8-flash"
PRICE = (0.75, 3.75)  # OpenRouter list, $/1M in/out
RESULTS_PATH = os.path.join(_HERE, "gemini38_runs_results.json")
SPEC = {"id": MODEL, "band": "gemini-3.8-flash", "kind": "llm"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY.")
        return 2

    # Patch the ladder price table so est_usd is right.
    jl.PRICES[MODEL] = PRICE

    draft = jb._load_draft(DEFAULT_MANIFEST)
    papers = draft.papers[:20]
    labels = jb._load_labels()
    recs = pairs(draft)
    opus_ref = jl._load_opus_ref()

    doc = {
        "started": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "price_per_m": {"in": PRICE[0], "out": PRICE[1]},
        "topic": draft.topic,
        "n_papers": len(papers),
        "n_claims": len(recs),
        "runs_requested": args.runs,
        "opus_ref": {str(k): v for k, v in opus_ref.items()},
        "curation": [],
        "claims": [],
    }
    if args.resume and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("model") == MODEL:
            doc = prev
            print("Resuming existing Gemini 3.8 Flash file.", flush=True)

    already = min(len(doc.get("curation") or []), len(doc.get("claims") or []))
    print(f"Gemini 3.8 Flash  {already} done / {args.runs}  "
          f"pool={len(papers)} papers, {len(recs)} claims", flush=True)

    for run_i in range(already + 1, args.runs + 1):
        print(f"\n-- run {run_i}/{args.runs} --", flush=True)
        cur = jl.run_curation(SPEC, draft.topic, papers, client=None)
        cur["run"] = run_i
        rel = {int(k): v for k, v in cur.get("relevance", {}).items()}
        if opus_ref and rel:
            gate = jb._gate_row(opus_ref, rel)
            cur["gate"] = gate
            print(
                f"  curate {len(rel)}  {cur['seconds']:.1f}s  ${cur['est_usd']:.4f}  "
                f"{cur['counts']}  agree {gate['agree']}/{gate['n']}  "
                f"{'PASS' if gate['critical_ok'] else 'FAIL'}"
                + (f"  ERR {cur['error'][:80]}" if cur.get("error") else ""),
                flush=True,
            )
            if not gate["critical_ok"]:
                for i in sorted(set(opus_ref) & set(rel)):
                    if opus_ref[i] != rel[i] and opus_ref[i] in ("direct", "tangential"):
                        print(f"      [{i:2}] opus {opus_ref[i]:<11} -> {rel[i]}", flush=True)
        claims = jl.run_claims(SPEC, recs, labels, draft.topic, client=None)
        claims["run"] = run_i
        print(
            f"  claims exact {claims.get('exact')}/{claims.get('n')}  "
            f"OR {claims.get('overreach_recall')}  FP {claims.get('false_overreach')}  "
            f"{claims['seconds']:.1f}s  ${claims['est_usd']:.4f}  "
            f"err={claims.get('errors', 0)}",
            flush=True,
        )
        for d in claims.get("disagreements") or []:
            print(f"      [{d['id']:2}] human {d['human']:<11} -> {d['model']}", flush=True)
        doc["curation"].append(cur)
        doc["claims"].append(claims)
        doc["updated"] = datetime.now(timezone.utc).isoformat()
        tmp = RESULTS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp, RESULTS_PATH)

    _summarise(doc, opus_ref)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


def _summarise(doc: dict, opus_ref: dict[int, str]) -> None:
    cur = [r for r in doc["curation"] if r.get("relevance")]
    claims = doc["claims"]
    maps = [{int(k): v for k, v in r["relevance"].items()} for r in cur]
    print("\n======== Gemini 3.8 Flash  10-run digest ========")
    if maps and opus_ref:
        gates = [jb._gate_row(opus_ref, m) for m in maps]
        n = gates[0]["n"]
        agrees = [g["agree"] for g in gates]
        tang_kept = [g["retained"]["tangential"]["kept"] for g in gates]
        dir_kept = [g["retained"]["direct"]["kept"] for g in gates]
        print(
            f"Curation  agree {min(agrees)}–{max(agrees)}/{n}  "
            f"mean {sum(agrees)/len(agrees):.1f}  "
            f"direct kept {min(dir_kept)}–{max(dir_kept)}/"
            f"{gates[0]['retained']['direct']['held']}  "
            f"tang kept {min(tang_kept)}–{max(tang_kept)}/"
            f"{gates[0]['retained']['tangential']['held']}  "
            f"stable {jb._stable_fraction(maps, list(range(1, n + 1))):.0%}  "
            f"PASS {sum(1 for g in gates if g['critical_ok'])}/{len(gates)}"
        )
        # modal label per paper
        print("  modal label vs Opus (only disagreements):")
        for i in range(1, n + 1):
            modal = Counter(m[i] for m in maps if i in m).most_common(1)[0][0]
            gold = opus_ref.get(i)
            if gold and modal != gold:
                print(f"    [{i:2}] opus {gold:<11} modal {modal}")
    if claims:
        n = max(r["n"] for r in claims) or 12
        exacts = [r["exact"] for r in claims]
        fps = [r.get("false_overreach") or 0 for r in claims]
        caught = held = 0
        pair_hits = {3: 0, 11: 0}
        for r in claims:
            rec = r.get("overreach_recall") or "—"
            if "/" in rec:
                a, b = rec.split("/", 1)
                caught += int(a)
                held += int(b)
            for row in r.get("results") or []:
                if row.get("id") in pair_hits and row.get("verdict") == "overreach":
                    pair_hits[row["id"]] += 1
        print(
            f"Claims    exact {min(exacts)}–{max(exacts)}/{n}  "
            f"mean {sum(exacts)/len(exacts):.1f}  "
            f"OR {caught}/{held}  "
            f"FP mean {sum(fps)/len(fps):.1f}  "
            f"pair3 overreach {pair_hits[3]}/{len(claims)}  "
            f"pair11 overreach {pair_hits[11]}/{len(claims)}"
        )
        sec_c = sum(r["seconds"] for r in cur) / len(cur) if cur else 0
        sec_k = sum(r["seconds"] for r in claims) / len(claims)
        usd_c = sum(r["est_usd"] for r in cur) / len(cur) if cur else 0
        usd_k = sum(r["est_usd"] for r in claims) / len(claims)
        print(
            f"Cost      curate {sec_c:.1f}s ${usd_c:.4f}  "
            f"claims {sec_k:.1f}s ${usd_k:.4f}  "
            f"combined {sec_c + sec_k:.1f}s ${usd_c + usd_k:.4f}/run"
        )


if __name__ == "__main__":
    raise SystemExit(main())

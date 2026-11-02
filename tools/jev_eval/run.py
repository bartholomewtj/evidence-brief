#!/usr/bin/env python
"""Jev vs LLM in evidence-brief, on the five frozen eval pools.

    python tools/jev_eval/run.py relevance reference|llm|jev|cascade|score
    python tools/jev_eval/run.py claims    reference|llm|jev|score
    python tools/jev_eval/run.py design    reference|llm|jev|score
    python tools/jev_eval/run.py trim      jev|score
    python tools/jev_eval/run.py e2e       write|judge|figures|score
    python tools/jev_eval/run.py handcheck build|apply
    python tools/jev_eval/run.py report
    python tools/jev_eval/run.py spend

Needs OPENROUTER_API_KEY, TYPESAFE_API_KEY and `pip install typesafe-sdk`.
Raw results and the cost ledger go to evals/jev/ (gitignored). The budget
guard (JEV_EVAL_BUDGET, default $14.50) stops new paid calls past the cap.
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AREAS = {"relevance": "relevance", "claims": "claims", "design": "design",
         "trim": "trim", "e2e": "e2e", "handcheck": "handcheck", "report": "report"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    area = sys.argv[1]
    if area == "spend":
        from common import METER, stage_costs
        by_prefix = {}
        for stage, row in stage_costs().items():
            key = ":".join(stage.split(":")[:2])
            by_prefix[key] = by_prefix.get(key, 0.0) + row["usd"]
        for k, v in sorted(by_prefix.items()):
            print(f"  {k:28} ${v:.4f}")
        print(f"total ${METER.spent:.3f}")
        return 0
    mod = importlib.import_module(AREAS[area])
    mod.main(sys.argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main())

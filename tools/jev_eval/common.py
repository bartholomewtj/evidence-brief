"""Shared plumbing for the Jev-vs-LLM evaluation.

- `Meter` records the real dollar cost of every OpenRouter call (the `usage.cost`
  field OpenRouter returns) and every Jev call (input tokens x list price), per
  stage, and refuses new paid calls past the budget.
- `jev()` is one TypeSafe System One request with retry and metering.
- Scoring helpers: confusion, Cohen's kappa, label-specific recall, the
  weighted error cost used for tuning, and a leave-one-topic-out tuner.

Nothing here is imported by `articlegen/`.
"""

from __future__ import annotations

import contextlib
import glob
import itertools
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from articlegen.pipeline import Draft  # noqa: E402

POOLS = os.path.join(ROOT, "evals", "pools")
RUNS = os.path.join(ROOT, "evals", "runs")
OUT = os.path.join(ROOT, "evals", "jev")
os.makedirs(OUT, exist_ok=True)

JEV_MODEL = "jev-1.14.0"          # pinned: jev-latest moves
JEV_PRICE_IN = 0.042 / 1e6        # $/input token; output is free (docs.typesafe.ai/models)

LUNA = "openai/gpt-5.6-luna"            # hosted planner/curator today
HAIKU = "anthropic/claude-haiku-5.5"    # cheapest Claude, extra LLM arm
SONNET = "anthropic/claude-sonnet-5.5"  # hosted writer today
OPUS5 = "anthropic/claude-opus-5"       # CLI default today
OPUS55 = "anthropic/claude-opus-5.5"    # reference A
FABLE = "anthropic/claude-fable-5.1"    # reference B

BUDGET_USD = float(os.environ.get("JEV_EVAL_BUDGET", "14.9"))
LEDGER = os.path.join(OUT, "ledger.jsonl")


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    """Process-wide cost ledger. Persisted so a restart still counts spend."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self.spent = 0.0
        self._captures: dict[str, list] = {}
        if os.path.exists(LEDGER):
            with open(LEDGER, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.spent += json.loads(line).get("usd", 0.0)
                    except json.JSONDecodeError:
                        pass

    @property
    def stage_name(self) -> str:
        return getattr(self._local, "stage", "unstaged")

    @contextlib.contextmanager
    def stage(self, name: str):
        prev = getattr(self._local, "stage", "unstaged")
        self._local.stage = name
        try:
            yield
        finally:
            self._local.stage = prev

    def check(self, upcoming: float = 0.0) -> None:
        if self.spent + upcoming > BUDGET_USD:
            raise BudgetExceeded(f"spent ${self.spent:.3f} of ${BUDGET_USD:.2f}")

    def record(self, provider: str, model: str, usd: float, tin: int, tout: int,
               seconds: float) -> None:
        row = {"t": time.time(), "stage": self.stage_name, "provider": provider,
               "model": model, "usd": usd, "in": tin, "out": tout,
               "seconds": round(seconds, 2)}
        with self._lock:
            self.spent += usd
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            for rows in self._captures.get(row["stage"], []):
                rows.append(row)

    @contextlib.contextmanager
    def capture(self, stage: str | None = None):
        """Collect rows recorded under `stage` (default: this thread's stage),
        from any thread, inside the block. Use `staged()` for worker threads."""
        key = stage or self.stage_name
        rows: list = []
        with self._lock:
            self._captures.setdefault(key, []).append(rows)
        try:
            yield rows
        finally:
            with self._lock:
                self._captures[key].remove(rows)

    def staged(self, fn, stage: str | None = None):
        """Wrap `fn` so it runs under `stage` (default: the caller's) in any thread."""
        name = stage or self.stage_name

        def run(*a, **kw):
            with self.stage(name):
                return fn(*a, **kw)
        return run


METER = Meter()


def _install_openrouter_meter() -> None:
    import requests

    real_post = requests.post
    if getattr(real_post, "_jev_metered", False):
        return

    def post(url, *a, **kw):
        if "openrouter.ai" in str(url):
            METER.check()
        t0 = time.time()
        res = real_post(url, *a, **kw)
        if "openrouter.ai" in str(url):
            try:
                data = res.json()
                usage = data.get("usage") or {}
                METER.record("openrouter", data.get("model") or (kw.get("json") or {}).get("model", "?"),
                             float(usage.get("cost") or 0.0),
                             int(usage.get("prompt_tokens") or 0),
                             int(usage.get("completion_tokens") or 0),
                             time.time() - t0)
            except Exception:  # noqa: BLE001 - a meter must never break the call
                pass
        return res

    post._jev_metered = True
    requests.post = post


_install_openrouter_meter()

# --------------------------------------------------------------------- Jev

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def client():
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from typesafe_sdk import TypeSafeClient
            _CLIENT = TypeSafeClient(timeout=180.0)
        return _CLIENT


def jev(state: dict, questions: dict, *, tries: int = 4):
    """One System One request. Returns the response; meters the cost."""
    METER.check()
    last = None
    for attempt in range(tries):
        t0 = time.time()
        try:
            resp = client().system_one(state=state, questions=questions, model=JEV_MODEL)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
            continue
        tin = int(getattr(resp.usage, "input_tokens", 0) or 0)
        METER.record("typesafe", JEV_MODEL, tin * JEV_PRICE_IN, tin, 0, time.time() - t0)
        return resp
    raise RuntimeError(f"Jev failed after {tries} tries: {last}")


def noul(resp, key: str) -> float:
    return float(resp.nouls[key].noul)


def choice(resp, key: str) -> tuple[str, float, dict]:
    a = resp.choices[key]
    return a.choice, float(a.confidence), {k: float(v) for k, v in dict(a.probabilities).items()}


# ------------------------------------------------------------------- pools

def pool_paths() -> list[str]:
    return sorted(glob.glob(os.path.join(POOLS, "*.json")))


def slug(path: str) -> str:
    return os.path.basename(path)[:-5]


SHORT = {
    "dietary": "gout diet",
    "do-school": "phone bans",
    "does-a-four": "4-day week",
    "does-reducing": "seclusion",
    "is-intermittent": "fasting",
}


def short(stem: str) -> str:
    for k, v in SHORT.items():
        if stem.startswith(k):
            return v
    return stem[:16]


def load_draft(path: str) -> Draft:
    with open(path, encoding="utf-8") as f:
        return Draft.from_dict(json.load(f))


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------- scoring

def confusion(gold: dict, pred: dict, labels) -> dict:
    m = {g: {p: 0 for p in labels} for g in labels}
    for k, g in gold.items():
        p = pred.get(k)
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def kappa(gold: dict, pred: dict, labels) -> float:
    keys = [k for k in gold if k in pred]
    n = len(keys)
    if not n:
        return 0.0
    po = sum(1 for k in keys if gold[k] == pred[k]) / n
    gc = Counter(gold[k] for k in keys)
    pc = Counter(pred[k] for k in keys)
    pe = sum(gc[lab] * pc[lab] for lab in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def summary(gold: dict, pred: dict, labels, cost_matrix: dict | None = None) -> dict:
    keys = [k for k in gold if k in pred]
    n = len(keys)
    agree = sum(1 for k in keys if gold[k] == pred[k])
    out = {"n": n, "accuracy": round(agree / n, 4) if n else 0.0,
           "kappa": round(kappa(gold, pred, labels), 4),
           "confusion": confusion(gold, pred, labels)}
    for lab in labels:
        held = [k for k in keys if gold[k] == lab]
        called = [k for k in keys if pred[k] == lab]
        tp = sum(1 for k in held if pred[k] == lab)
        out[f"recall_{lab}"] = round(tp / len(held), 4) if held else None
        out[f"precision_{lab}"] = round(tp / len(called), 4) if called else None
    if cost_matrix:
        out["error_cost"] = round(sum(cost_matrix[gold[k]][pred[k]] for k in keys), 3)
    return out


def loo_tune(topics: list[str], grid: list[dict], evaluate) -> dict:
    """Leave-one-topic-out. `evaluate(params, topics) -> loss` (lower is better).

    For each held-out topic, pick the params with the lowest loss on the other
    topics, then record the params chosen. Returns {held_out: params}.
    Ties break toward the first grid entry, so put defaults first.
    """
    chosen = {}
    for held in topics:
        train = [t for t in topics if t != held]
        best, best_loss = None, None
        for params in grid:
            loss = evaluate(params, train)
            if best_loss is None or loss < best_loss - 1e-9:
                best, best_loss = params, loss
        chosen[held] = best
    return chosen


def grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def stage_costs(prefix: str = "") -> dict:
    """Sum the ledger by stage (optionally only stages starting with prefix)."""
    tot = defaultdict(lambda: {"usd": 0.0, "calls": 0, "in": 0, "out": 0, "seconds": 0.0})
    if not os.path.exists(LEDGER):
        return {}
    with open(LEDGER, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["stage"].startswith(prefix):
                continue
            t = tot[r["stage"]]
            t["usd"] += r["usd"]
            t["calls"] += 1
            t["in"] += r["in"]
            t["out"] += r["out"]
            t["seconds"] += r["seconds"]
    return {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in sorted(tot.items())}

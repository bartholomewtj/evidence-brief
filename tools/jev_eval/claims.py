"""Area B: a claim-support gate (non-numeric overreach) — Jev vs LLMs.

    python tools/jev_eval/run.py claims reference   # Opus 5.5 + Fable 5.1
    python tools/jev_eval/run.py claims llm         # Luna x2, Haiku x2, Sonnet 5.5 x1
    python tools/jev_eval/run.py claims jev
    python tools/jev_eval/run.py claims score

The pipeline has no such gate today: `verify.check_statistics` checks numbers
only. Claims are every cited sentence in the shipped `final2` briefings and the
older `base` briefings on the same five pools, each paired with exactly the
material the writer was shown for the sources it cites.

LLM arms check one briefing per call (sources once, all sentences listed) —
the shape a production gate would take. Jev asks one request per sentence.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import re

from typesafe_sdk import Choice, Noul

from articlegen.llm import generate_json
from articlegen.sources import full_text_excerpts
from articlegen.verify import _article_sentences, _paper_haystack

from common import (FABLE, HAIKU, LUNA, METER, OPUS55, OUT, RUNS, SONNET, choice,
                    grid, jev, load_draft, load_json, loo_tune, noul, save_json,
                    short)

PATH = os.path.join(OUT, "claims.json")
LABELS = ("supported", "overreach", "unsupported")
RUN_SETS = ("final2", "base")
PER_SOURCE_CHARS = 6000  # Jev state: abstract + best-matching passages per source

LLM_ARMS = [("luna", LUNA, 2), ("haiku", HAIKU, 2), ("sonnet", SONNET, 1)]

SYSTEM = """\
You check an evidence briefing sentence by sentence against the source \
material its author was shown. For each numbered sentence, judge it against \
ONLY the sources it cites, and give one label:

- "supported": the cited material states the claim or plainly entails it. \
Different wording is fine if the meaning holds. Hedged sentences that stay \
within the evidence are supported.
- "overreach": the sentence goes further than the cited material: an \
association stated as a cause, a wider or different population than was \
studied, a dropped qualifier or hedge, more certainty than the material \
carries, a modifier attached to an outcome it was not tested on, or a \
comparison the material never makes.
- "unsupported": the cited material does not contain the claim at all, or \
the claim is credited to the wrong source.

Use no outside knowledge. Numbers are checked by a separate deterministic \
pass: judge the claim, not the arithmetic. Synthesis across several cited \
sources is fine when each part is supported by one of them. Give a short \
`problem` for every label other than supported (else an empty string)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(LABELS)},
                    "problem": {"type": "string"},
                },
                "required": ["id", "verdict", "problem"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

# ------------------------------------------------------------- claim set


def _briefings():
    for rs in RUN_SETS:
        for path in sorted(glob.glob(os.path.join(RUNS, rs, "*.json"))):
            if path.endswith((".judge.json", ".pair.json")):
                continue
            yield rs, os.path.basename(path)[:-5], load_draft(path)


def build_claims() -> list[dict]:
    """One record per cited sentence. Pure; ids are stable across runs."""
    out = []
    for rs, stem, draft in _briefings():
        shown = full_text_excerpts(draft.papers)
        n = len(draft.papers)
        k = 0
        for sentence, cites in _article_sentences(draft.article):
            cited = sorted({i for i in cites if 1 <= i <= n})
            if not cited:
                continue
            k += 1
            out.append({
                "id": f"{rs}/{stem}/{k}", "run": rs, "stem": stem, "topic": draft.topic,
                "sentence": sentence, "cited": cited,
                "material": {str(i): _paper_haystack(draft.papers[i - 1], shown, i)
                             for i in cited},
                "titles": {str(i): draft.papers[i - 1].title for i in cited},
            })
    return out


# ------------------------------------------------------------- LLM gate

def _llm_prompt(topic: str, claims: list[dict]) -> str:
    sources = {}
    for c in claims:
        for i in c["cited"]:
            sources[i] = (c["titles"][str(i)], c["material"][str(i)])
    blocks = [f"SOURCE [{i}] {t}\n{m}" for i, (t, m) in sorted(sources.items())]
    lines = [f"{j}. (cites {', '.join(f'[{i}]' for i in c['cited'])}) {c['sentence']}"
             for j, c in enumerate(claims, start=1)]
    return (f"Briefing question: {topic}\n\n=== SOURCES ===\n\n" + "\n\n".join(blocks)
            + "\n\n=== SENTENCES TO CHECK ===\n\n" + "\n".join(lines))


def llm_gate(claims: list[dict], model: str) -> dict:
    """{claim id: {verdict, problem}} for one briefing's claims, one call."""
    res = generate_json(_llm_prompt(claims[0]["topic"], claims), SCHEMA,
                        system=SYSTEM, model=model)
    out = {}
    for v in res.get("verdicts") or []:
        j = v.get("id")
        if isinstance(j, int) and 1 <= j <= len(claims) and v.get("verdict") in LABELS:
            out[claims[j - 1]["id"]] = {"verdict": v["verdict"], "problem": v.get("problem", "")}
    return out


def _by_briefing(claims):
    groups = {}
    for c in claims:
        groups.setdefault((c["run"], c["stem"]), []).append(c)
    return groups


def _run_llm(doc, key, model, runs):
    claims = doc["claims"]
    groups = _by_briefing(claims)
    have = doc.setdefault(key, [])
    for r in range(len(have), runs):
        run = {"verdicts": {}, "usd": 0.0, "seconds": 0.0, "served_by": [], "per_briefing": {}}

        def one(g):
            stage = f"claims:{key}:r{r + 1}:{g[0]}/{g[1]}"
            with METER.stage(stage), METER.capture() as calls:
                try:
                    v = llm_gate(groups[g], model)
                    err = ""
                except Exception as exc:  # noqa: BLE001
                    v, err = {}, f"{type(exc).__name__}: {exc}"[:200]
            return g, v, calls, err

        with cf.ThreadPoolExecutor(max_workers=5) as ex:
            for g, v, calls, err in ex.map(one, list(groups)):
                run["verdicts"].update(v)
                usd = sum(c["usd"] for c in calls)
                run["usd"] += usd
                run["seconds"] += sum(c["seconds"] for c in calls)
                run["served_by"] = sorted(set(run["served_by"]) | {c["model"] for c in calls})
                run["per_briefing"]["/".join(g)] = {"usd": usd, "n": len(groups[g]),
                                                     "got": len(v), "error": err}
                print(f"  {key:10} r{r + 1} {g[0]:6} {short(g[1]):11} {len(v)}/{len(groups[g])} "
                      f"${usd:.4f} {err[:60]}", flush=True)
        have.append(run)
        save_json(PATH, doc)


def cmd_reference(doc):
    _run_llm(doc, "ref_opus55", OPUS55, 1)
    _run_llm(doc, "ref_fable", FABLE, 1)


def cmd_llm(doc):
    for key, model, runs in LLM_ARMS:
        _run_llm(doc, key, model, runs)


# ------------------------------------------------------------- Jev gate

RELATION = Choice(
    instructions=(
        "How does the cited material in `sources` relate to `sentence`? Judge "
        "the whole sentence, including every outcome, population and modifier it "
        "names. Several sources may each support a different part."
    ),
    criteria={
        "supports": "The sources state or directly imply the whole sentence.",
        "overreaches": (
            "Part of the sentence is in the sources, but another part goes "
            "further: a cause where they report an association, a wider "
            "population, a dropped hedge, a modifier applied to the wrong "
            "outcome, or a claim of absence the sources never make."
        ),
        "silent": "The sources do not address what the sentence asserts.",
    },
)

NOULS = {
    "causal": Noul(
        instructions=(
            "Does `sentence` state a causal effect (causes, improves, reduces, "
            "leads to, prevents) where `sources` only report an association or "
            "correlation, or present observational results as an effect?"
        ),
        criteria={"true": "Causal wording the sources do not license.",
                  "false": "No causal claim, or the sources report a trial or effect that licenses it."},
    ),
    "population": Noul(
        instructions=(
            "Does `sentence` apply a finding to a wider or different population, "
            "setting or group than the one `sources` studied?"
        ),
        criteria={"true": "The sentence generalises beyond the studied population or setting.",
                  "false": "The population and setting match what the sources studied."},
    ),
    "certainty": Noul(
        instructions=(
            "Does `sentence` state a finding with more certainty than `sources` "
            "give it — dropping a hedge, saying 'shows' where they say "
            "'suggests', or omitting that a result was not significant?"
        ),
        criteria={"true": "The sentence is more certain than the sources.",
                  "false": "The certainty matches the sources, or the sentence is more cautious."},
    ),
    "modifier": Noul(
        instructions=(
            "`sentence` may name several outcomes and a modifier (a subgroup, "
            "dose, duration, interaction or comparison). Does it attach that "
            "modifier to an outcome for which `sources` do not report it?"
        ),
        criteria={"true": "A modifier is hung on an outcome the sources did not test it on.",
                  "false": "There is no such modifier, or the sources report it for every outcome named."},
    ),
    "absent": Noul(
        instructions=(
            "Does `sentence` report an outcome, comparison or result that none of "
            "`sources` reports at all?"
        ),
        criteria={"true": "At least one asserted result is not in any cited source.",
                  "false": "Every result the sentence asserts appears in some cited source."},
    ),
}
NOUL_KEYS = tuple(NOULS)

_WORD = re.compile(r"[a-z][a-z\-]{2,}")
_STOP = set("the and for with that this were was are from than have has had not but "
            "their they them which who what when where into over under more less among "
            "between after before those these such also only both each other".split())


def _passages(text: str, sentence: str, budget: int) -> str:
    """Keep the abstract head, then the passages that share most words with the
    sentence, in document order, inside `budget` characters (code filters,
    Jev judges: TypeSafe's context-rot guidance)."""
    if len(text) <= budget:
        return text
    want = {w for w in _WORD.findall(sentence.lower()) if w not in _STOP}
    chunks = [c for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
    windows = [" ".join(chunks[i:i + 3]) for i in range(0, len(chunks), 3)]
    head, rest = windows[:2], list(enumerate(windows[2:], start=2))
    scored = sorted(rest, key=lambda iw: -len(want & set(_WORD.findall(iw[1].lower()))))
    keep, used = set(), sum(len(h) for h in head)
    for i, w in scored:
        if used + len(w) > budget:
            continue
        keep.add(i)
        used += len(w)
    return " … ".join(head + [w for i, w in rest if i in keep])


def jev_state(c: dict) -> dict:
    return {
        "briefing_question": c["topic"],
        "sentence": c["sentence"],
        "sources": [{"index": i, "title": c["titles"][str(i)],
                     "text": _passages(c["material"][str(i)], c["sentence"], PER_SOURCE_CHARS)}
                    for i in c["cited"]],
    }


def ask_claim(c: dict) -> dict:
    resp = jev(jev_state(c), {"relation": RELATION, **NOULS})
    lab, conf, probs = choice(resp, "relation")
    return {"relation": lab, "relation_conf": conf, "relation_probs": probs,
            **{k: noul(resp, k) for k in NOUL_KEYS}}


def cmd_jev(doc):
    claims = doc["claims"]
    runs = doc.setdefault("jev", [])
    want = int(os.environ.get("JEV_RUNS", "2"))
    for r in range(len(runs), want):
        with METER.stage(f"claims:jev:r{r + 1}"), METER.capture() as calls:
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                answers = list(ex.map(METER.staged(ask_claim), claims))
        runs.append({"answers": {c["id"]: a for c, a in zip(claims, answers)},
                     "usd": round(sum(x["usd"] for x in calls), 6),
                     "seconds_sum": round(sum(x["seconds"] for x in calls), 2),
                     "tokens_in": sum(x["in"] for x in calls)})
        print(f"  jev r{r + 1}: {len(claims)} claims ${runs[-1]['usd']:.5f}", flush=True)
        save_json(PATH, doc)


# --------------------------------------------------------- composition

DEFAULT = {"t_rel": 0.0, "t_noul": 0.5, "use": ("causal", "population", "certainty", "modifier", "absent")}


def compose(a: dict, t_rel=0.0, t_noul=0.5, use=NOUL_KEYS) -> str:
    if a["relation"] == "silent" and a["relation_conf"] >= t_rel:
        return "unsupported"
    if a["absent"] >= t_noul and "absent" in use:
        return "unsupported"
    if a["relation"] == "overreaches" and a["relation_conf"] >= t_rel:
        return "overreach"
    if any(a[k] >= t_noul for k in use if k != "absent"):
        return "overreach"
    return "supported"


def flag_margin(a: dict, t_rel=0.0, t_noul=0.5, use=NOUL_KEYS) -> float:
    """How far a 'supported' call is from flipping (0 = on the edge)."""
    worst = max((a[k] for k in use), default=0.0)
    rel_ok = a["relation_probs"].get("supports", 0.0)
    return min(t_noul - worst, rel_ok - 0.5)


USE_SETS = [NOUL_KEYS, ("causal", "population", "modifier", "absent"),
            ("modifier", "absent"), ("causal", "modifier", "absent"), ()]
GRID = grid(t_rel=[0.0, 0.2, 0.4], t_noul=[0.5, 0.4, 0.6, 0.7, 0.8], use=USE_SETS)


def binary(lab: str) -> str:
    return "ok" if lab == "supported" else "flag"


def gold(doc, hand=None) -> dict:
    a = doc["ref_opus55"][0]["verdicts"]
    b = doc["ref_fable"][0]["verdicts"]
    g = {}
    for k in a:
        if k in b and binary(a[k]["verdict"]) == binary(b[k]["verdict"]):
            g[k] = binary(a[k]["verdict"])
    for k, v in (hand or {}).items():
        if v in ("ok", "flag") and k in a:   # only claims in this (sub)set
            g[k] = v
    return g


def _topic_of(cid: str) -> str:
    return cid.split("/")[1]


# Missing a real overreach is worse than a false flag (a false flag costs one
# targeted revision; a miss ships an overclaim).
FN_COST, FP_COST = 2.0, 1.0


def _loss(g, pred, stems):
    loss = 0.0
    for k, v in g.items():
        if _topic_of(k) not in stems or k not in pred:
            continue
        p = pred[k]
        if v == "flag" and p == "ok":
            loss += FN_COST
        elif v == "ok" and p == "flag":
            loss += FP_COST
    return loss


def bin_summary(g, pred) -> dict:
    keys = [k for k in g if k in pred]
    tp = sum(1 for k in keys if g[k] == "flag" and pred[k] == "flag")
    fn = sum(1 for k in keys if g[k] == "flag" and pred[k] == "ok")
    fp = sum(1 for k in keys if g[k] == "ok" and pred[k] == "flag")
    tn = sum(1 for k in keys if g[k] == "ok" and pred[k] == "ok")
    n = len(keys)
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "recall_flag": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision_flag": round(tp / (tp + fp), 4) if tp + fp else None,
            "false_flag_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "loss": FN_COST * fn + FP_COST * fp}


def score(doc, hand=None) -> dict:
    g = gold(doc, hand)
    stems = sorted({_topic_of(k) for k in g})
    res = {"gold_n": len(g), "gold_flags": sum(1 for v in g.values() if v == "flag"),
           "per_arm": {}, "loo_params": {}}

    def add(arm, pred, usd):
        row = res["per_arm"].setdefault(arm, {"runs": []})
        row["runs"].append({"usd": usd, **bin_summary(g, pred)})

    for key in ("luna", "haiku", "sonnet", "ref_opus55", "ref_fable"):
        for run in doc.get(key, []):
            add(key, {k: binary(v["verdict"]) for k, v in run["verdicts"].items()}, run["usd"])

    jr = doc["jev"][0]["answers"]
    for r in doc["jev"]:
        add("jev_relation_only", {k: binary({"supports": "supported"}.get(a["relation"], "flag"))
                                  for k, a in r["answers"].items()}, r["usd"])
        add("jev_default", {k: binary(compose(a, **DEFAULT)) for k, a in r["answers"].items()}, r["usd"])

    chosen = loo_tune(stems, GRID, lambda prm, tr: _loss(
        g, {k: binary(compose(a, **prm)) for k, a in jr.items()}, tr))
    res["loo_params"] = {s: {**p, "use": list(p["use"])} for s, p in chosen.items()}
    for r in doc["jev"]:
        pred = {k: binary(compose(a, **chosen[_topic_of(k)])) for k, a in r["answers"].items()}
        add("jev_loo", pred, r["usd"])

    # Cascade: Jev clears claims it is sure are fine; the LLM re-checks the rest.
    claims = {c["id"]: c for c in doc["claims"]}
    for llm in ("haiku", "sonnet", "luna"):
        if not doc.get(llm):
            continue
        lv = {k: binary(v["verdict"]) for k, v in doc[llm][0]["verdicts"].items()}
        full_usd = doc[llm][0]["usd"]
        total_chars = sum(sum(len(m) for m in c["material"].values()) for c in claims.values())
        for tau in (0.0, 0.1, 0.2, 0.3, 0.4):
            pred, esc = {}, []
            for k, a in jr.items():
                prm = chosen[_topic_of(k)]
                p = binary(compose(a, **prm))
                if p == "ok" and flag_margin(a, **prm) >= tau:
                    pred[k] = "ok"
                elif k in lv:
                    pred[k] = lv[k]
                    esc.append(k)
                else:
                    pred[k] = p
            # Subset call cost ~ the material of sources the escalated claims cite.
            groups = {}
            for k in esc:
                c = claims[k]
                groups.setdefault((c["run"], c["stem"]), {}).update(c["material"])
            esc_chars = sum(len(m) for grp in groups.values() for m in grp.values())
            brief_chars = {}
            for c in claims.values():
                brief_chars.setdefault((c["run"], c["stem"]), {}).update(c["material"])
            all_chars = sum(len(m) for grp in brief_chars.values() for m in grp.values())
            usd = full_usd * esc_chars / max(1, all_chars) + doc["jev"][0]["usd"]
            arm = f"cascade_{llm}_t{tau}"
            add(arm, pred, round(usd, 5))
            res["per_arm"][arm]["escalated"] = f"{len(esc)}/{len(jr)}"

    for arm, row in res["per_arm"].items():
        rs = row["runs"]
        for k in ("accuracy", "recall_flag", "precision_flag", "false_flag_rate", "loss"):
            vals = [r[k] for r in rs if r[k] is not None]
            row[k] = round(sum(vals) / len(vals), 4) if vals else None
        usds = [r["usd"] for r in rs if r["usd"] is not None]
        row["usd_all_128"] = round(sum(usds) / len(usds), 5) if usds else None
    return res


def subset(doc, keep) -> dict:
    """Same doc restricted to claim ids where keep(claim) is true."""
    ids = {c["id"] for c in doc["claims"] if keep(c)}
    out = dict(doc, claims=[c for c in doc["claims"] if c["id"] in ids])
    for key in ("ref_opus55", "ref_fable", "luna", "haiku", "sonnet"):
        out[key] = [dict(r, verdicts={k: v for k, v in r["verdicts"].items() if k in ids})
                    for r in doc.get(key, [])]
    out["jev"] = [dict(r, answers={k: v for k, v in r["answers"].items() if k in ids})
                  for r in doc["jev"]]
    return out


def leading_cite(c) -> bool:
    """`verify._article_sentences` splits after the full stop, so a marker
    written after it ('... fat. [6] Next ...') opens the NEXT sentence and is
    credited to it. The base briefings were written that way; final2 never is."""
    return bool(re.match(r"\s*\[\d+\]", c["sentence"]))


def cmd_score(doc):
    hand = load_json(os.path.join(OUT, "hand_claims.json"), {})
    out = {}
    for name, keep in (("final2", lambda c: c["run"] == "final2"),
                       ("misattributed", leading_cite)):
        print(f"\n== {name}")
        out[name] = _print_scores(score(subset(doc, keep), hand))
    save_json(os.path.join(OUT, "claims_scores.json"), out)


def _print_scores(res):
    print(f"gold n={res['gold_n']} flags={res['gold_flags']}")
    print(f"{'arm':26} {'acc':>6} {'recall':>7} {'prec':>6} {'falseflag':>9} {'loss':>6} {'$':>8}")
    for arm, r in sorted(res["per_arm"].items(), key=lambda kv: kv[1]["loss"] or 0):
        print(f"{arm:26} {r['accuracy'] or 0:.3f} {r['recall_flag'] or 0:7.3f} {r['precision_flag'] or 0:6.3f} "
              f"{r['false_flag_rate'] or 0:9.3f} {r['loss'] or 0:6.1f} {r['usd_all_128'] or 0:8.4f} "
              f"{r.get('escalated', '')}")
    return res


def main(argv):
    doc = load_json(PATH, {})
    if "claims" not in doc:
        doc["claims"] = build_claims()
        save_json(PATH, doc)
        print(f"built {len(doc['claims'])} claims")
    cmd = argv[0] if argv else "score"
    {"reference": cmd_reference, "llm": cmd_llm, "jev": cmd_jev, "score": cmd_score}[cmd](doc)
    print(f"ledger: ${METER.spent:.3f} spent")

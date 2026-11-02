"""Area A (relevance labelling) and C1 (Jev pre-screen before the LLM curator).

    python tools/jev_eval/run.py relevance reference   # Opus 5.5 + Fable 5.1, once
    python tools/jev_eval/run.py relevance llm         # Luna x3, Haiku x3, Opus 5 x2
    python tools/jev_eval/run.py relevance jev         # one Jev request per paper
    python tools/jev_eval/run.py relevance cascade     # real subset calls for the LOO pick
    python tools/jev_eval/run.py relevance score

The LLM arms call the production `curate_sources` unchanged, so the prompt is
the one the site runs. Jev sees the same title/abstract plus the planner's
`core_entity`, which the pipeline already has at curation time.
"""

from __future__ import annotations

import concurrent.futures as cf
import os

from typesafe_sdk import Choice, Noul

from articlegen.writer import curate_sources

from common import (FABLE, HAIKU, LUNA, METER, OPUS5, OPUS55, OUT, choice, grid,
                    jev, load_draft, load_json, loo_tune, noul, pool_paths,
                    save_json, short, slug, summary)

LABELS = ("direct", "related", "tangential")
PATH = os.path.join(OUT, "relevance.json")

# Error cost used for tuning and reported beside accuracy. A reference-direct
# paper called tangential never reaches the writer: the worst miss. A
# tangential paper called direct poses adjacent evidence as on-topic.
COST = {
    "direct": {"direct": 0, "related": 1, "tangential": 3},
    "related": {"direct": 1, "related": 0, "tangential": 1},
    "tangential": {"direct": 2, "related": 0.5, "tangential": 0},
}

LLM_ARMS = [
    ("luna", LUNA, 3),     # hosted curator today
    ("haiku", HAIKU, 3),   # cheapest Claude
    ("opus5", OPUS5, 2),   # CLI default today
]

# ------------------------------------------------------------ Jev questions

CHOICE = Choice(
    instructions=(
        "You judge how directly `paper` addresses ONE specific briefing question, "
        "`question` (its core subject is `core_subject`) — not how good, famous or "
        "highly cited the paper is. Judge the study itself from its title and "
        "abstract. A famous review that never studies the specific question is not "
        "direct."
    ),
    criteria={
        "tangential": (
            "Only good for background or framing: the paper studies something "
            "else, or `core_subject` is only a passing mention."
        ),
        "related": (
            "Adjacent: the same intervention or exposure in a different population "
            "or disorder, or the right population but a different question. Useful "
            "context, not direct evidence."
        ),
        "direct": (
            "The study is actually about `question`: right population, right "
            "intervention or exposure, right question."
        ),
    },
)

NOULS = {
    "on_subject": Noul(
        instructions=(
            "Is `core_subject` what `paper` itself studies — its main exposure, "
            "intervention or object of study — rather than a passing mention in "
            "the background of `paper.abstract`?"
        ),
        criteria={
            "true": "The paper's methods or aims are about `core_subject`.",
            "false": "`core_subject` is absent, incidental, or one example among many.",
        },
    ),
    "same_question": Noul(
        instructions=(
            "Does `paper` address `question` itself, rather than an adjacent "
            "question such as a mechanism, a different comparison, a different "
            "population, or a different outcome?"
        ),
        criteria={
            "true": "Same subject, same population (or one the question does not restrict), and an outcome the question asks about.",
            "false": "A neighbouring question, even if the subject appears in the setting.",
        },
    ),
    "population_fit": Noul(
        instructions=(
            "Does `paper` study the population `question` is about? If `question` "
            "names no population, a population it reasonably covers counts."
        ),
        criteria={
            "true": "The studied people or setting are the ones `question` is about.",
            "false": "A different population, species, or setting from the one `question` is about.",
        },
    ),
    "background_only": Noul(
        instructions=(
            "Would `paper` only be useful as background or framing for a briefing "
            "on `question`, not as evidence that belongs in its findings?"
        ),
        criteria={
            "true": "Background, definitions, mechanism without outcomes, or another topic.",
            "false": "It carries evidence a briefing on `question` should consider citing.",
        },
    ),
    "answer_evidence": Noul(
        instructions=(
            "Does `paper.abstract` state results usable in a direct answer to "
            "`question` (an effect, an association, or a clear null finding)?"
        ),
        criteria={
            "true": "There are results that bear on the answer to `question`.",
            "false": "No such results: protocol, commentary, methods, or results on something else.",
        },
    ),
}

NOUL_KEYS = tuple(NOULS)


def paper_state(topic: str, core: str, p) -> dict:
    return {
        "question": topic,
        "core_subject": core or topic,
        "paper": {
            "title": p.title or "",
            "year": p.year,
            "venue": p.venue or "",
            "publication_types": list(p.publication_types or []),
            "abstract": p.abstract or "(no abstract)",
        },
    }


def ask_paper(topic: str, core: str, p) -> dict:
    resp = jev(paper_state(topic, core, p), {"label": CHOICE, **NOULS})
    lab, conf, probs = choice(resp, "label")
    return {"choice": lab, "choice_conf": conf, "choice_probs": probs,
            **{k: noul(resp, k) for k in NOUL_KEYS}}


# ------------------------------------------------------------ composition

DEFAULT = {"t_sub": 0.5, "t_bg": 0.7, "t_same": 0.5, "t_pop": 0.0}


def compose(a: dict, t_sub=0.5, t_bg=0.7, t_same=0.5, t_pop=0.0) -> str:
    """Code owns the label; Jev supplies the judgments."""
    if a["on_subject"] < t_sub:
        return "tangential"
    if a["background_only"] >= t_bg and a["same_question"] < t_same:
        return "tangential"
    if a["same_question"] >= t_same and a["population_fit"] >= t_pop:
        return "direct"
    return "related"


def margin(a: dict, t_sub=0.5, t_bg=0.7, t_same=0.5, t_pop=0.0) -> float:
    """Distance of the deciding judgments from their thresholds (0 = coin-flip)."""
    ms = [abs(a["on_subject"] - t_sub)]
    if a["on_subject"] >= t_sub:
        ms.append(abs(a["same_question"] - t_same))
        if t_pop > 0 and a["same_question"] >= t_same:
            ms.append(abs(a["population_fit"] - t_pop))
    return min(ms)


GRID = grid(t_sub=[0.5, 0.4, 0.6, 0.3], t_bg=[0.7, 0.6, 0.8, 1.01],
            t_same=[0.5, 0.4, 0.6, 0.7], t_pop=[0.0, 0.3, 0.5])

# ------------------------------------------------------------------ runs


def _pools():
    return [(slug(p), load_draft(p)) for p in pool_paths()]


def _curate(stem, draft, model, tag):
    with METER.stage(f"rel:{tag}:{stem}"), METER.capture() as calls:
        res = curate_sources(draft.topic, draft.papers, model=model)
    return {
        "relevance": {str(k): v for k, v in (res.get("relevance") or {}).items()},
        "error": res.get("error", ""),
        "usd": round(sum(c["usd"] for c in calls), 5),
        "seconds": round(sum(c["seconds"] for c in calls), 2),
        "tokens_in": sum(c["in"] for c in calls),
        "tokens_out": sum(c["out"] for c in calls),
        "served_by": sorted({c["model"] for c in calls}),
    }


def _run_llm(doc: dict, key: str, model: str, runs: int) -> None:
    jobs = []
    for stem, draft in _pools():
        have = doc.setdefault(key, {}).setdefault(stem, [])
        for r in range(len([x for x in have if not x.get("error")]), runs):
            jobs.append((stem, draft, r))
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_curate, stem, draft, model, f"{key}:r{r + 1}"): stem
                for stem, draft, r in jobs}
        for fut in cf.as_completed(futs):
            stem = futs[fut]
            res = fut.result()
            doc[key][stem] = [x for x in doc[key][stem] if not x.get("error")] + [res]
            print(f"  {key:8} {short(stem):11} ${res['usd']:.4f} {res['seconds']:.0f}s "
                  f"{res['served_by']} {res['error'][:80]}", flush=True)
            save_json(PATH, doc)


def cmd_reference(doc):
    _run_llm(doc, "ref_opus55", OPUS55, 1)
    _run_llm(doc, "ref_fable", FABLE, 1)


def cmd_llm(doc):
    # The frozen pool's own labels are a free extra Luna run (28 Oct, same prompt).
    for stem, draft in _pools():
        doc.setdefault("luna_frozen", {})[stem] = [{
            "relevance": {str(k): v for k, v in draft.curation["relevance"].items()},
            "usd": None, "seconds": None}]
    for key, model, runs in LLM_ARMS:
        _run_llm(doc, key, model, runs)


def cmd_jev(doc):
    out = doc.setdefault("jev", {})
    for stem, draft in _pools():
        if stem in out and len(out[stem]["answers"]) == len(draft.papers) and out[stem]["usd"]:
            continue
        prior = out.get(stem, {}).get("answers")
        core = (draft.provenance or {}).get("core_entity") or ""
        with METER.stage(f"rel:jev:{stem}"), METER.capture() as calls:
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                answers = list(ex.map(METER.staged(lambda p: ask_paper(draft.topic, core, p)), draft.papers))
        out[stem] = {"answers": {str(i): a for i, a in enumerate(answers, start=1)},
                     "usd": round(sum(c["usd"] for c in calls), 6),
                     "seconds_sum": round(sum(c["seconds"] for c in calls), 2),
                     "tokens_in": sum(c["in"] for c in calls)}
        drift = None
        if prior:
            drift = max(abs(prior[k][q] - out[stem]["answers"][k][q])
                        for k in prior for q in NOUL_KEYS)
        print(f"  jev      {short(stem):11} ${out[stem]['usd']:.5f} "
              f"{len(answers)} papers  max-noul-drift-vs-prior={drift}", flush=True)
        save_json(PATH, doc)


# --------------------------------------------------------------- scoring

def gold(doc: dict, stem: str, hand: dict | None = None) -> dict:
    """Opus 5.5 + Fable 5.1 agreement; Bart's label where he judged an item."""
    a = (doc["ref_opus55"][stem][0])["relevance"]
    b = (doc["ref_fable"][stem][0])["relevance"]
    g = {k: a[k] for k in a if b.get(k) == a[k]}
    for k, v in ((hand or {}).get(stem) or {}).items():
        if v in LABELS:
            g[k] = v
    return g


def jev_labels(doc, stem, params) -> dict:
    return {k: compose(a, **params) for k, a in doc["jev"][stem]["answers"].items()}


def jev_choice_labels(doc, stem) -> dict:
    return {k: a["choice"] for k, a in doc["jev"][stem]["answers"].items()}


def cascade_labels(doc, stem, params, tau, llm_labels, accept=LABELS, use="margin"):
    """Jev keeps its confident calls in `accept`; everything else goes to the LLM."""
    out, escalated = {}, []
    for k, a in doc["jev"][stem]["answers"].items():
        lab = compose(a, **params)
        conf = margin(a, **params) if use == "margin" else a["choice_conf"]
        if lab in accept and conf >= tau and k in llm_labels:
            out[k] = lab
        elif k in llm_labels:
            out[k] = llm_labels[k]
            escalated.append(k)
        else:
            out[k] = lab
    return out, escalated


def _loss(doc, hand, params, stems):
    tot = 0.0
    for s in stems:
        g = gold(doc, s, hand)
        p = jev_labels(doc, s, params)
        tot += sum(COST[g[k]][p[k]] for k in g if k in p)
    return tot


def score(doc: dict, hand: dict | None = None) -> dict:
    stems = sorted(doc["jev"])
    res = {"per_arm": {}, "loo_params": {}, "cascade": {}}

    def add(arm, stem, labels, usd, seconds):
        g = gold(doc, stem, hand)
        row = res["per_arm"].setdefault(arm, {"runs": []})
        row["runs"].append({"stem": stem, "usd": usd, "seconds": seconds,
                            **summary(g, labels, LABELS, COST)})

    for key in ("luna_frozen", "luna", "haiku", "opus5", "ref_opus55", "ref_fable"):
        for stem in stems:
            for run in doc.get(key, {}).get(stem, []):
                add(key, stem, run["relevance"], run.get("usd"), run.get("seconds"))
    for stem in stems:
        j = doc["jev"][stem]
        add("jev_choice", stem, jev_choice_labels(doc, stem), j["usd"], j["seconds_sum"])
        add("jev_default", stem, jev_labels(doc, stem, DEFAULT), j["usd"], j["seconds_sum"])

    chosen = loo_tune(stems, GRID, lambda prm, tr: _loss(doc, hand, prm, tr))
    res["loo_params"] = chosen
    for stem in stems:
        j = doc["jev"][stem]
        add("jev_loo", stem, jev_labels(doc, stem, chosen[stem]), j["usd"], j["seconds_sum"])

    # Cascades, simulated from each LLM's own first run: the labels a subset
    # call would return are those papers' labels from the full call. Cost is
    # scaled by the escalated share of abstract characters (prompt overhead
    # included pro rata); `cascade` validates the pick with real subset calls.
    for llm in ("luna", "haiku", "opus5"):
        for accept_name, accept in (("all", LABELS), ("tangential", ("tangential",))):
            for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01):
                arm = f"cascade_{llm}_{accept_name}_t{tau}"
                for stem in stems:
                    runs = doc.get(llm, {}).get(stem) or []
                    if not runs:
                        continue
                    draft = load_draft(next(p for p in pool_paths() if slug(p) == stem))
                    chars = {str(i): len(p.abstract or "") + len(p.title or "") + 200
                             for i, p in enumerate(draft.papers, start=1)}
                    # Every run of the LLM, not the first: its run-to-run
                    # spread is larger than the cascade effect.
                    first_usd = max((r["usd"] or 0) for r in runs)  # uncached price
                    for run in runs:
                        labels, esc = cascade_labels(doc, stem, chosen[stem], tau,
                                                     run["relevance"], accept)
                        share = sum(chars[k] for k in esc) / max(1, sum(chars.values()))
                        usd = first_usd * share + doc["jev"][stem]["usd"]
                        add(arm, stem, labels, round(usd, 5), None)
                    res["cascade"].setdefault(arm, []).append(
                        {"stem": stem, "escalated": len(esc), "of": len(labels)})

    for key, per in (doc.get("cascade_real") or {}).items():
        for stem in stems:
            if stem in per:
                add(f"real_cascade_{key}", stem, per[stem]["relevance"], per[stem]["usd"],
                    per[stem].get("seconds"))

    # Leave-one-out tau: the held-out topic uses the tau that minimised error
    # cost on the other four (params already LOO). This is the honest cascade row.
    taus = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01)
    for llm in ("luna", "haiku", "opus5"):
        for accept_name in ("all", "tangential"):
            def arm_loss(prm, train, llm=llm, accept_name=accept_name):
                rows = res["per_arm"][f"cascade_{llm}_{accept_name}_t{prm['tau']}"]["runs"]
                return sum(r["error_cost"] for r in rows if r["stem"] in train) / max(1, len(rows))
            picked = loo_tune(stems, [{"tau": t} for t in taus], arm_loss)
            arm = f"cascade_{llm}_{accept_name}_loo"
            for stem in stems:
                src = res["per_arm"][f"cascade_{llm}_{accept_name}_t{picked[stem]['tau']}"]
                for row in (r for r in src["runs"] if r["stem"] == stem):
                    res["per_arm"].setdefault(arm, {"runs": []})["runs"].append(
                        dict(row, tau=picked[stem]["tau"]))
            res["cascade"][arm] = {s: picked[s]["tau"] for s in stems}

    for arm, row in res["per_arm"].items():
        runs = row["runs"]
        n = sum(r["n"] for r in runs)
        row["n"] = n
        row["accuracy"] = round(sum(r["accuracy"] * r["n"] for r in runs) / n, 4) if n else None
        row["error_cost_per_run_topic"] = round(sum(r["error_cost"] for r in runs) / max(1, len(runs)), 3)
        row["usd_per_pool"] = (round(sum(r["usd"] for r in runs if r["usd"] is not None)
                                     / max(1, sum(1 for r in runs if r["usd"] is not None)), 5)
                               if any(r["usd"] is not None for r in runs) else None)
        for lab in LABELS:
            held = sum(sum(r["confusion"][lab].values()) for r in runs)
            tp = sum(r["confusion"][lab][lab] for r in runs)
            called = sum(r["confusion"][g][lab] for r in runs for g in LABELS)
            row[f"recall_{lab}"] = round(tp / held, 4) if held else None
            row[f"precision_{lab}"] = round(tp / called, 4) if called else None
        row["kappa_mean"] = round(sum(r["kappa"] for r in runs) / len(runs), 4)
        row["direct_lost_to_tangential"] = sum(r["confusion"]["direct"]["tangential"] for r in runs)
    return res


def cmd_score(doc):
    hand = load_json(os.path.join(OUT, "hand_relevance.json"), {})
    res = score(doc, hand)
    save_json(os.path.join(OUT, "relevance_scores.json"), res)
    print(f"{'arm':38} {'acc':>6} {'kappa':>6} {'cost':>6} {'d-rec':>6} {'t-rec':>6} {'d->t':>4} {'$/pool':>8}")
    for arm, r in sorted(res["per_arm"].items(), key=lambda kv: kv[1]["error_cost_per_run_topic"]):
        if arm.startswith("cascade") and not arm.endswith("_loo"):
            continue
        print(f"{arm:38} {r['accuracy']:.3f} {r['kappa_mean']:.3f} {r['error_cost_per_run_topic']:6.2f} "
              f"{r['recall_direct'] or 0:.3f} {r['recall_tangential'] or 0:.3f} "
              f"{r['direct_lost_to_tangential']:4d} {r['usd_per_pool'] if r['usd_per_pool'] is not None else '—':>8}")
    return res


def cmd_cascade(doc):
    """Real subset calls for the leave-one-out cascade: each held-out topic uses
    the compose params AND the tau tuned on the other four topics."""
    accept = os.environ.get("JEV_ACCEPT", "tangential")
    llms = os.environ.get("JEV_CASCADE_LLMS", "haiku,luna").split(",")
    models = {"luna": LUNA, "haiku": HAIKU, "opus5": OPUS5}
    hand = load_json(os.path.join(OUT, "hand_relevance.json"), {})
    sc = score(doc, hand)
    chosen = sc["loo_params"]
    out = doc.setdefault("cascade_real", {})
    for llm_key in llms:
        taus = sc["cascade"][f"cascade_{llm_key}_{accept}_loo"]
        key = f"{llm_key}_{accept}_loo"
        for stem, draft in _pools():
            if stem in out.get(key, {}):
                continue
            params, tau = chosen[stem], taus[stem]
            accept_set = LABELS if accept == "all" else ("tangential",)
            esc = [k for k, a in doc["jev"][stem]["answers"].items()
                   if not (compose(a, **params) in accept_set and margin(a, **params) >= tau)]
            idx = [int(k) for k in esc]
            sub = [draft.papers[i - 1] for i in idx]
            with METER.stage(f"rel:cascade:{key}:{stem}"), METER.capture() as calls:
                res = curate_sources(draft.topic, sub, model=models[llm_key]) if sub else {"relevance": {}}
            labels = {k: compose(a, **params) for k, a in doc["jev"][stem]["answers"].items()}
            for j, lab in (res.get("relevance") or {}).items():
                labels[str(idx[j - 1])] = lab
            out.setdefault(key, {})[stem] = {
                "relevance": labels, "escalated": esc, "tau": tau,
                "unlabelled_by_llm": len(esc) - len(res.get("relevance") or {}),
                "usd": round(sum(c["usd"] for c in calls) + doc["jev"][stem]["usd"], 5),
                "llm_usd": round(sum(c["usd"] for c in calls), 5),
                "seconds": round(sum(c["seconds"] for c in calls), 2),
                "error": res.get("error", ""),
            }
            print(f"  {key:22} {short(stem):11} escalated {len(esc)}/{len(draft.papers)} "
                  f"${out[key][stem]['usd']:.4f}", flush=True)
            save_json(PATH, doc)


def main(argv):
    doc = load_json(PATH, {})
    cmd = argv[0] if argv else "score"
    {"reference": cmd_reference, "llm": cmd_llm, "jev": cmd_jev,
     "cascade": cmd_cascade, "score": cmd_score}[cmd](doc)
    print(f"ledger: ${METER.spent:.3f} spent")

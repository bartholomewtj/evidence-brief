"""Area C2: trim the writer's context with Jev.

    python tools/jev_eval/run.py trim jev     # passage judgments for full-text excerpts
    python tools/jev_eval/run.py trim score

Two cuts, both judged offline against what three earlier writer runs on the
same pools (base, final, final2) actually used:

1. Source trim. The writer is shown every non-tangential source (23-38 per
   pool) but may cite at most min(12, direct + 2). Hide `related` sources whose
   Jev `answer_evidence` is below a threshold (direct are always shown).
   Accuracy = how many sources an earlier writer cited would have been hidden.
2. Passage trim. Full-text excerpts are up to 12,000 characters each. Keep the
   passages Jev judges to carry results on the question, inside a smaller
   budget. Accuracy = share of figures earlier briefings took from the full
   text that survive the cut.

Savings are counted in writer-prompt characters, built by the production
`_writer_context`, and converted to dollars at the measured Sonnet 5.5 price.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import glob
import os
import re

from typesafe_sdk import Noul

from articlegen.sources import full_text_excerpts
from articlegen.verify import _FIGURE_RE, _YEAR_RE, _article_sentences
from articlegen.writer import _writer_context

from common import (METER, OUT, RUNS, jev, load_draft, load_json, loo_tune, noul,
                    pool_paths, save_json, short, slug)

PATH = os.path.join(OUT, "trim.json")
REL = os.path.join(OUT, "relevance.json")
SONNET_IN = 2.0 / 1e6     # $/token
CHARS_PER_TOKEN = 3.3     # measured on these prompts: 111,784 chars -> ~33.9k tokens

PASSAGE = Noul(
    instructions=(
        "Does `passage` (from the full text of `paper_title`) report results, "
        "effect sizes, numbers of participants, or conclusions that bear on "
        "`question`?"
    ),
    criteria={
        "true": "It carries results, sample details or conclusions relevant to the question.",
        "false": "Background, methods boilerplate, references to other work, or off-topic detail.",
    },
)


def _passages(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paras:
        if len(p) <= 1200:
            out.append(p)
            continue
        sents = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sents:
            if len(buf) + len(s) > 900 and buf:
                out.append(buf)
                buf = ""
            buf = (buf + " " + s).strip()
        if buf:
            out.append(buf)
    return out


def cmd_jev(doc):
    out = doc.setdefault("passages", {})
    for path in pool_paths():
        stem, draft = slug(path), load_draft(path)
        if stem in out:
            continue
        ex = full_text_excerpts(draft.papers)
        jobs = []
        for idx, text in ex.items():
            for j, passage in enumerate(_passages(text)):
                jobs.append((idx, j, passage))

        def ask(job, draft=draft):
            idx, j, passage = job
            r = jev({"question": draft.topic, "paper_title": draft.papers[idx - 1].title,
                     "passage": passage}, {"results": PASSAGE})
            return idx, j, passage, noul(r, "results")

        with METER.stage(f"trim:jev:{stem}"), METER.capture() as calls:
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                got = list(pool.map(METER.staged(ask), jobs))
        out[stem] = {"items": [{"idx": i, "j": j, "text": t, "p": p} for i, j, t, p in got],
                     "usd": round(sum(c["usd"] for c in calls), 6)}
        print(f"  passages {short(stem):11} {len(got)} ${out[stem]['usd']:.5f}", flush=True)
        save_json(PATH, doc)


# --------------------------------------------------------------- scoring

def _cited_sets(stem: str) -> dict[str, set[int]]:
    out = {}
    for rs in ("base", "final", "final2"):
        p = os.path.join(RUNS, rs, stem + ".json")
        if os.path.exists(p):
            d = load_draft(p)
            out[rs] = set(d.cited_refs)
    return out


def hidden_sources(draft, relevance: dict, answers: dict, t_ev: float) -> set[int]:
    return {int(k) for k, lab in relevance.items()
            if lab == "related" and answers[str(k)]["answer_evidence"] < t_ev}


def writer_curation(curation: dict, hide: set[int]) -> dict:
    """Same counts the reader sees; hidden sources omitted from the prompt."""
    c = copy.deepcopy(curation)
    rel = {int(k): v for k, v in c["relevance"].items()}
    for i in hide:
        rel[i] = "tangential"
    c["relevance"] = rel
    return c


def context_chars(draft, curation) -> int:
    ctx, _ = _writer_context(draft.topic, draft.papers, "", curation, kind="briefing")
    return len(ctx)


def _figures_from_fulltext(stem: str, draft) -> list[tuple[int, str]]:
    """(source, figure) pairs earlier briefings stated that occur in that
    source's excerpt but not in its title/abstract — i.e. taken from full text."""
    ex = full_text_excerpts(draft.papers)
    pairs = set()
    for rs in ("base", "final", "final2"):
        p = os.path.join(RUNS, rs, stem + ".json")
        if not os.path.exists(p):
            continue
        art = load_draft(p).article
        for sentence, cites in _article_sentences(art):
            for m in _FIGURE_RE.finditer(sentence):
                fig = m.group(0).strip()
                if _YEAR_RE.fullmatch(fig) or len(re.sub(r"\D", "", fig)) < 2:
                    continue
                for i in cites:
                    if i in ex:
                        head = (draft.papers[i - 1].title or "") + " " + (draft.papers[i - 1].abstract or "")
                        if fig in ex[i] and fig not in head:
                            pairs.add((i, fig))
    return sorted(pairs)


def kept_text(items, idx, t_p, budget):
    rows = sorted((x for x in items if x["idx"] == idx), key=lambda x: -x["p"])
    keep, used = set(), 0
    for x in rows:
        if x["p"] < t_p:
            break
        if used + len(x["text"]) > budget:
            continue
        keep.add(x["j"])
        used += len(x["text"])
    return "\n\n".join(x["text"] for x in sorted(rows, key=lambda x: x["j"]) if x["j"] in keep)


def score(doc):
    rel = load_json(REL, {})
    res = {"source_trim": {}, "passage_trim": {}}
    stems = [slug(p) for p in pool_paths()]
    drafts = {slug(p): load_draft(p) for p in pool_paths()}

    def src_eval(t_ev, stem):
        d = drafts[stem]
        answers = rel["jev"][stem]["answers"]
        labels = {int(k): v for k, v in d.curation["relevance"].items()}
        hide = hidden_sources(d, labels, answers, t_ev)
        cited = _cited_sets(stem)
        lost = {rs: sorted(hide & c) for rs, c in cited.items()}
        base = context_chars(d, d.curation)
        trimmed = context_chars(d, writer_curation(d.curation, hide))
        return {"hidden": len(hide), "shown_before": sum(1 for v in labels.values() if v != "tangential"),
                "cited_lost": lost, "n_cited_lost": sum(len(v) for v in lost.values()),
                "chars_before": base, "chars_after": trimmed,
                "usd_saved_per_writer_call": round((base - trimmed) / CHARS_PER_TOKEN * SONNET_IN, 4)}

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    table = {t: {s: src_eval(t, s) for s in stems} for t in thresholds}
    # Pick the largest cut that loses nothing an earlier writer cited on the
    # training topics; ties go to the smaller threshold.
    grid = [{"t_ev": t} for t in thresholds]

    def loss(prm, train):
        lost = sum(table[prm["t_ev"]][s]["n_cited_lost"] for s in train)
        saved = sum(table[prm["t_ev"]][s]["chars_before"] - table[prm["t_ev"]][s]["chars_after"] for s in train)
        return lost * 1e6 - saved
    chosen = loo_tune(stems, grid, loss)
    res["source_trim"] = {"by_threshold": {str(t): v for t, v in table.items()},
                          "loo": {s: {"t_ev": chosen[s]["t_ev"], **table[chosen[s]["t_ev"]][s]}
                                  for s in stems}}

    if "passages" in doc:
        for budget in (4000, 6000, 8000):
            for t_p in (0.3, 0.5):
                key = f"b{budget}_p{t_p}"
                tot_figs = kept = 0
                before = after = 0
                for s in stems:
                    d = drafts[s]
                    ex = full_text_excerpts(d.papers)
                    figs = _figures_from_fulltext(s, d)
                    items = doc["passages"][s]["items"]
                    texts = {i: kept_text(items, i, t_p, budget) for i in ex}
                    for i, fig in figs:
                        tot_figs += 1
                        kept += fig in texts[i]
                    before += sum(len(v) for v in ex.values())
                    after += sum(len(v) for v in texts.values())
                res["passage_trim"][key] = {
                    "fulltext_figures": tot_figs, "kept": kept,
                    "recall": round(kept / tot_figs, 4) if tot_figs else None,
                    "chars_before": before, "chars_after": after,
                    "usd_saved_per_writer_call_all5": round((before - after) / CHARS_PER_TOKEN * SONNET_IN, 4)}
    return res


def cmd_score(doc):
    res = score(doc)
    save_json(os.path.join(OUT, "trim_scores.json"), res)
    print("source trim (hide related with answer_evidence < t):")
    for t, rows in res["source_trim"]["by_threshold"].items():
        print(f"  t={t:4} hidden {sum(r['hidden'] for r in rows.values()):3}/"
              f"{sum(r['shown_before'] for r in rows.values())}  cited-lost "
              f"{sum(r['n_cited_lost'] for r in rows.values()):2}  "
              f"chars {sum(r['chars_before'] for r in rows.values())}->{sum(r['chars_after'] for r in rows.values())}  "
              f"$/writer call saved (5 pools) {sum(r['usd_saved_per_writer_call'] for r in rows.values()):.3f}")
    print("  LOO:", {short(s): (v["t_ev"], v["n_cited_lost"]) for s, v in res["source_trim"]["loo"].items()})
    for k, v in res["passage_trim"].items():
        print(f"passage {k}: figure recall {v['kept']}/{v['fulltext_figures']}  chars {v['chars_before']}->{v['chars_after']}")


def main(argv):
    doc = load_json(PATH, {})
    cmd = argv[0] if argv else "score"
    {"jev": cmd_jev, "score": cmd_score}[cmd](doc)
    print(f"ledger: ${METER.spent:.3f} spent")

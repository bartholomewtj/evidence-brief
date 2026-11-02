"""Opt-in TypeSafe Jev judgments. Off unless `ARTICLEGEN_JEV` (or `--jev`) names them.

    ARTICLEGEN_JEV=prescreen,trim
    TYPESAFE_API_KEY=...            # and `pip install typesafe-sdk`

Two features, each measured on the five frozen eval pools against an
Opus 5.5 + Fable 5.1 reference (tools/JEV_REPORT.md):

- `prescreen` — one Jev request per candidate paper (~$0.0024 for 40). A paper
  Jev is confident does not study the question's core subject is labelled
  `tangential` without an LLM call; the LLM curator labels the rest. The
  accepted calls were 97% right, and the curator's prompt shrinks by about
  half — on the CLI's Opus curator that is ~28% of the curation bill. Only the
  main pool is pre-screened; records the named-source and referenced-works
  passes add go to the LLM as before.
- `trim` — `related` sources Jev judges to report no results bearing on the
  question are left out of the writer's prompt (direct sources never are).
  The sources stay in the pool, the counts and the citation numbering; the
  writer just cannot see them. Writer input fell ~36%, the writer call ~24%,
  and blind-judged quality did not move (10-10 pairwise).

- `passages` — each fetched full-text excerpt (up to 12,000 characters) is
  cut to the passages Jev judges to carry results on the question, inside
  8,000 characters, before the full-text rewrite. The writer and the figure
  check both read the cut text, so verification still searches exactly what
  the writer saw. 63 of 65 figures earlier briefings took from full text
  survived the cut; with the label and trim changes, writer input fell ~36%.

Jev never writes text and never removes a direct source. Any Jev failure
degrades to the LLM-only path with a log line, never a failed run.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
from typing import Callable, Iterable

FEATURES = ("prescreen", "trim", "passages")
JEV_MODEL = os.environ.get("ARTICLEGEN_JEV_MODEL", "jev-1.14.0")

# P(`on_subject`) at or below this labels a paper tangential without the LLM.
# The leave-one-topic-out pick on the eval pools was 0.1-0.2; 0.1 is the
# conservative end because a direct paper called tangential never reaches the
# writer.
PRESCREEN_MAX_ON_SUBJECT = 0.1
# Passage trim: keep passages at or above this P(results), best first, inside
# this many characters per excerpt (eval: 63/65 full-text figures kept).
PASSAGE_MIN_RESULTS = 0.5
PASSAGE_BUDGET = 8000
# P(`answer_evidence`) below this hides a `related` source from the writer.
# Leave-one-out pick on four of five pools; lost none of the sources three
# earlier writer runs had cited.
TRIM_MIN_ANSWER_EVIDENCE = 0.1

_ON_SUBJECT = {
    "instructions": (
        "Is `core_subject` what `paper` itself studies — its main exposure, "
        "intervention or object of study — rather than a passing mention in the "
        "background of `paper.abstract`?"
    ),
    "criteria": {
        "true": "The paper's methods or aims are about `core_subject`.",
        "false": "`core_subject` is absent, incidental, or one example among many.",
    },
}
_ANSWER_EVIDENCE = {
    "instructions": (
        "Does `paper.abstract` state results usable in a direct answer to "
        "`question` (an effect, an association, or a clear null finding)?"
    ),
    "criteria": {
        "true": "There are results that bear on the answer to `question`.",
        "false": "No such results: protocol, commentary, methods, or results on something else.",
    },
}


def features(explicit: Iterable[str] | str | None = None) -> frozenset[str]:
    """The enabled features: `explicit` if given, else `ARTICLEGEN_JEV`."""
    raw = explicit if explicit is not None else os.environ.get("ARTICLEGEN_JEV", "")
    if isinstance(raw, str):
        raw = raw.split(",")
    return frozenset(f.strip().lower() for f in raw if f and f.strip().lower() in FEATURES)


def unavailable_reason() -> str:
    """Empty when Jev can be called; otherwise why not."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        return "TYPESAFE_API_KEY is not set"
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        return "typesafe-sdk is not installed (pip install typesafe-sdk)"
    return ""


def _paper_state(topic: str, core: str, paper) -> dict:
    return {
        "question": topic,
        "core_subject": core or topic,
        "paper": {
            "title": paper.title or "",
            "year": paper.year,
            "venue": paper.venue or "",
            "publication_types": list(paper.publication_types or []),
            "abstract": paper.abstract or "(no abstract)",
        },
    }


def judge_papers(topic: str, core_entity: str, papers: list, log: Callable[[str], None],
                 *, client=None, workers: int = 8) -> dict[int, dict] | None:
    """{1-based index: {on_subject, answer_evidence}} or None if Jev failed.

    One request per paper: Jev's context-rot guidance is to send only what a
    question needs, and a 40-abstract state would be most of the 64k window.
    """
    if not papers:
        return {}
    try:
        from typesafe_sdk import Noul, TypeSafeClient
        client = client or TypeSafeClient(timeout=120.0)
        questions = {"on_subject": Noul(**_ON_SUBJECT),
                     "answer_evidence": Noul(**_ANSWER_EVIDENCE)}

        def ask(item):
            i, p = item
            r = client.system_one(state=_paper_state(topic, core_entity, p),
                                  questions=questions, model=JEV_MODEL)
            return i, {k: float(r.nouls[k].noul) for k in questions}

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(ask, enumerate(papers, start=1)))
    except Exception as exc:  # noqa: BLE001 - Jev is an optimisation, never a gate
        log(f"  Jev judgments failed ({type(exc).__name__}: {str(exc)[:120]}); "
            "continuing without them.")
        return None


def prescreened(judgments: dict[int, dict], threshold: float = PRESCREEN_MAX_ON_SUBJECT) -> set[int]:
    return {i for i, j in judgments.items() if j["on_subject"] <= threshold}


def prescreen_curate(papers: list, judgments: dict[int, dict],
                     curate: Callable[[list], dict], log: Callable[[str], None],
                     threshold: float = PRESCREEN_MAX_ON_SUBJECT) -> tuple[dict, int]:
    """Label Jev's confident tangentials, send the rest to `curate`.

    Returns (curation, n_prescreened) with `curate_sources`'s contract: 1-based
    indices into `papers`, `counts`, `most_relevant_index`, `error` on failure.
    If the LLM call fails the error passes through unchanged, so the caller's
    "no usable labels" stop still fires.
    """
    tangential = prescreened(judgments, threshold)
    rest = [i for i in range(1, len(papers) + 1) if i not in tangential]
    if not rest:
        # Everything looked off-topic to Jev. That is a question the LLM should
        # answer, not a shortcut to take.
        return curate(papers), 0
    sub = curate([papers[i - 1] for i in rest])
    if not sub.get("relevance"):
        return sub, 0
    relevance = {i: "tangential" for i in tangential}
    for j, label in sub["relevance"].items():
        relevance[rest[j - 1]] = label
    mri = sub.get("most_relevant_index")
    mri = rest[mri - 1] if isinstance(mri, int) and 1 <= mri <= len(rest) else None
    counts = {lab: sum(1 for v in relevance.values() if v == lab)
              for lab in ("direct", "related", "tangential")}
    log(f"  Jev pre-screen: {len(tangential)} of {len(papers)} records set aside as "
        f"background; the curator labelled the other {len(rest)}.")
    return {"relevance": relevance, "most_relevant_index": mri, "counts": counts}, len(tangential)


def hidden_related(relevance: dict[int, str], judgments: dict[int, dict],
                   threshold: float = TRIM_MIN_ANSWER_EVIDENCE) -> list[int]:
    """`related` sources to keep out of the writer's prompt. Never a direct one."""
    return sorted(i for i, label in relevance.items()
                  if label == "related" and i in judgments
                  and judgments[i]["answer_evidence"] < threshold)


_PASSAGE = {
    "instructions": (
        "Does `passage` (from the full text of `paper_title`) report results, "
        "effect sizes, numbers of participants, or conclusions that bear on "
        "`question`?"
    ),
    "criteria": {
        "true": "It carries results, sample details or conclusions relevant to the question.",
        "false": "Background, methods boilerplate, references to other work, or off-topic detail.",
    },
}


def split_passages(text: str) -> list[str]:
    """Paragraphs, with long ones cut at sentence ends into ~900-char pieces.
    Section headings are dropped: the cut text is no longer in sections."""
    import re
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)
             if p.strip() and not p.strip().startswith("## ")]
    out = []
    for para in paras:
        if len(para) <= 1200:
            out.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(sent) > 900 and buf:
                out.append(buf)
                buf = ""
            buf = (buf + " " + sent).strip()
        if buf:
            out.append(buf)
    return out


def keep_passages(passages: list[str], scores: list[float],
                  min_p: float = PASSAGE_MIN_RESULTS, budget: int = PASSAGE_BUDGET) -> str:
    """Best-scoring passages that fit the budget, back in document order."""
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    keep, used = set(), 0
    for i in order:
        if scores[i] < min_p:
            break
        if used + len(passages[i]) > budget:
            continue
        keep.add(i)
        used += len(passages[i])
    return "\n\n".join(passages[i] for i in sorted(keep))


def trim_excerpts(topic: str, papers: list, excerpts: dict[int, str], log: Callable[[str], None],
                  *, client=None, workers: int = 8) -> list[int]:
    """Replace each fetched paper's full text with its Jev-selected passages.

    `excerpts` is `sources.full_text_excerpts(papers)` — the text the writer
    would otherwise be shown. Returns the indices that were cut. An excerpt
    already inside the budget, or one where Jev keeps nothing, is left whole.
    """
    try:
        from typesafe_sdk import Noul, TypeSafeClient
        client = client or TypeSafeClient(timeout=120.0)
        q = {"results": Noul(**_PASSAGE)}
        cut = []
        for idx, text in sorted(excerpts.items()):
            if len(text) <= PASSAGE_BUDGET:
                continue
            passages = split_passages(text)

            def ask(passage, idx=idx):
                r = client.system_one(state={"question": topic,
                                             "paper_title": papers[idx - 1].title or "",
                                             "passage": passage},
                                      questions=q, model=JEV_MODEL)
                return float(r.nouls["results"].noul)

            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                scores = list(ex.map(ask, passages))
            kept = keep_passages(passages, scores)
            if kept:
                papers[idx - 1].full_text = kept
                cut.append(idx)
        if cut:
            log(f"  Jev passages: cut the full text of {len(cut)} source(s) to the "
                f"passages that report results (at most {PASSAGE_BUDGET:,} characters each).")
        return cut
    except Exception as exc:  # noqa: BLE001
        log(f"  Jev passage trim failed ({type(exc).__name__}: {str(exc)[:120]}); "
            "using the full excerpts.")
        return []

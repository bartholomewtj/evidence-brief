"""The draft pipeline, in one place, for every caller.

`cmd_draft` (CLI) and `/api/draft` (web) used to each run their own version of
this sequence, and the web one had quietly drifted: it skipped the prose-style
gate and never built `provenance`, so articles generated through the web app
came out without the enforced hedging and with a Methods section missing the
search that actually produced them. The pipeline is the product, so there is
now exactly one copy of it.

Callers differ only in what they do with the result: the CLI writes files into
`drafts/`, the server renders and returns them.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import paperfetch
from .llm import resolve_provider
from .sources import (DATABASE_NAMES, DEFAULT_MAX_PAPERS, NAMED_SOURCE_LIMIT,
                      NAMED_SOURCE_PER_QUERY, NAMED_SOURCE_SCAN,
                      REFERENCED_CITED_LIMIT, REFERENCED_ID_LIMIT,
                      REFERENCED_SEEDS, Paper, SearchFailure,
                      _normalize_doi, fetch_full_text, filter_named_matches,
                      full_text_excerpts, full_text_order, gather_evidence,
                      merge_candidates,
                      named_matches, named_references, openalex_citing_works,
                      openalex_records, openalex_work, paper_design,
                      resolve_pmcid)
from .style import (SUBSTANCE_RULES, check_style, errors as style_errors,
                    format_report as format_style, revision_brief)
from .verify import check_statistics, revision_brief as statistics_brief
from .render import _au_date
from .writer import (cite_target, clean_pico, clean_search_terms, curate_sources,
                     is_briefing, plan_queries, revise_prose,
                     revise_statistics, write_article, write_briefing)

# A caller that wants progress reporting passes one of these; the default drops it.
Logger = Callable[[str], None]

# How many full texts a draft aims for, and what it will spend getting them.
#
# The target is 5 because that is what the reader can actually be shown:
# `sources.full_text_excerpts` allows 12,000 characters per paper inside a
# 60,000-character total, so the sixth full text is truncated and the seventh
# gets nothing. Fetching more than the excerpt budget can display is pure cost.
FULLTEXT_TARGET = 5

# Hosted drafts sit behind a long proxy, but a 5s-per-DOI papers batch
# still times out on a cold cache and yields zero texts. Prefer two
# open-access copies over five misses. Local CLI still aims for the target.
HOSTED_FULLTEXT_FLOOR = 2
HOSTED_PAPERS_TIMEOUT_SEC = 20.0

# Every candidate now costs up to two Europe PMC requests (a DOI lookup, then
# the fetch), where before a paper with no pmcid cost nothing. A topic whose
# sources are all paywalled would otherwise spend one request per paper and
# come back with nothing, against the one scholarly API that reliably answers.
MAX_FULLTEXT_REQUESTS = 18

# How many new records one `refresh` will label. The refresh pays for exactly
# one `curate_sources` call, so this bounds its cost; the ranking in
# `gather_evidence` has already put the best candidates first.
REFRESH_LIMIT = 10


def _hosted() -> bool:
    return os.environ.get("ARTICLEGEN_STATELESS", "").strip().lower() in (
        "1", "true", "yes")


def _fulltext_prefetch_limit() -> int:
    """How many DOIs `papers get -` is sent before the apply loop.

    Local CLI prefetches the request cap so a paywalled leading cite
    leaves later OA ones in the batch. Hosted sends the floor (2)
    OA-first DOIs with a longer per-DOI timeout, then Europe PMC
    fills any remaining gaps.
    """
    if _hosted():
        return HOSTED_FULLTEXT_FLOOR
    return MAX_FULLTEXT_REQUESTS


def _papers_get_timeout() -> float:
    if _hosted():
        return HOSTED_PAPERS_TIMEOUT_SEC
    return paperfetch.DEFAULT_TIMEOUT


# Fail before billing the caller, when the sources are the problem.
#
# `plan_queries` is a paid LLM call and it runs before anything touches a
# scholarly API, so on a day when every API is refusing the caller pays for a
# run that was doomed from the start (#96). A one-query probe against the topic
# costs less than the gather it would precede, and on the dead day it costs
# nothing at all after the first one.
#
# The two memos are what keep this from being a tax on the healthy path. A
# server that has heard from a source recently skips the probe entirely, so a
# busy deployment pays for it once every SOURCE_PROBE_TTL rather than per
# request; a server that just saw every source fail reuses that verdict for
# SOURCE_PROBE_FAIL_TTL rather than re-probing per request.
#
# It can only ever refuse on the same condition `generate_draft` already raises
# on after the fact — *every* source returned an error, not merely no results —
# so it cannot block a draft that would have worked, short of the sources
# recovering inside the two-minute failure window.
SOURCE_PROBE_TTL = 900
SOURCE_PROBE_FAIL_TTL = 120

_probe_lock = threading.Lock()
_sources_last_ok = 0.0
_sources_last_fail: tuple[float, str] = (0.0, "")


def _note_sources_answered() -> None:
    global _sources_last_ok
    with _probe_lock:
        _sources_last_ok = time.time()


def _preflight_sources(topic: str, log: Logger) -> None:
    """Raise NoPapersFound before the first paid call if every source is down."""
    global _sources_last_fail
    if os.environ.get("ARTICLEGEN_SOURCE_PROBE", "").strip() == "0":
        return
    now = time.time()
    with _probe_lock:
        if now - _sources_last_ok < SOURCE_PROBE_TTL:
            return
        failed_at, reasons = _sources_last_fail
        if now - failed_at < SOURCE_PROBE_FAIL_TTL:
            raise NoPapersFound(_SOURCES_DOWN + reasons, sources_failed=True)

    outcomes: list[dict] = []
    log("Checking the scholarly APIs are answering...")
    papers = gather_evidence([topic], max_papers=1, per_query=1, topic=topic,
                             log=_silent, outcomes=outcomes, patient=False,
                             stop_when_any=True)
    failures = [o for o in outcomes if o["error"]]
    if outcomes and len(failures) == len(outcomes):
        reasons = "; ".join(sorted({f"{o['source']}: {o['error']}" for o in failures}))
        with _probe_lock:
            _sources_last_fail = (time.time(), reasons)
        raise NoPapersFound(_SOURCES_DOWN + reasons, sources_failed=True)
    if papers or outcomes:
        _note_sources_answered()


_SOURCES_DOWN = (
    "The scholarly APIs are not responding, so no evidence could be gathered. "
    "This is not a problem with the topic, and nothing has been charged for this "
    "attempt. "
)


def _read_subset_skew(papers: list[Paper], fetched: list[int]) -> str:
    """How the deeply-read subset differs from the rest, in one line.

    The open-access skew is *visible* — Table 1's Read column shows it per
    source, and Limitations says the deeply read subset skews open access. It
    has never been *measured*, so "the reader should weigh that" is advice
    nobody can act on (#84). Median year and citation count are the two axes
    the concern is usually about: newer, and lower-impact.

    Deliberately descriptive. Two small samples cannot support a test, and a
    p-value on n=4 would be worse than the raw numbers.
    """
    read = [p for i, p in enumerate(papers, start=1) if i in fetched]
    rest = [p for i, p in enumerate(papers, start=1) if i not in fetched]
    if not read or not rest:
        return "read-subset skew: not comparable (nothing read, or everything was)"

    def median(values: list[int]) -> str:
        values = sorted(v for v in values if v is not None)
        if not values:
            return "?"
        mid = len(values) // 2
        return str(values[mid] if len(values) % 2 else
                   (values[mid - 1] + values[mid]) // 2)

    return (f"read-subset skew: read n={len(read)} "
            f"median year {median([p.year for p in read])}, "
            f"median citations {median([p.citation_count for p in read])}; "
            f"abstract-only n={len(rest)} "
            f"median year {median([p.year for p in rest])}, "
            f"median citations {median([p.citation_count for p in rest])}")


def _silent(message: str) -> None:
    pass


def _cited_indices(article: dict, n_papers: int) -> list[int]:
    """SOURCE indices the writer cited, in citation order, in range."""
    out: list[int] = []
    seen: set[int] = set()
    for raw in article.get("references") or []:
        if isinstance(raw, int) and 1 <= raw <= n_papers and raw not in seen:
            out.append(raw)
            seen.add(raw)
    return out


def _retrieve_full_texts(
    papers: list[Paper],
    order: list[int],
    log: Logger,
    use_cache: bool = True,
) -> tuple[list[int], int, int, int, int, str]:
    """Fetch open-access full text for papers in `order`.

    Returns (fetched 1-based indices, eligible tried, no-open-access,
    fetch-failed, requests spent, stop reason). Same bounds as the pre-cite
    loop this replaced: FULLTEXT_TARGET and MAX_FULLTEXT_REQUESTS.

    When the papers CLI is available, the DOI list is sent once on stdin
    (`papers get -`) up to `_fulltext_prefetch_limit()`, then this loop
    applies results. Hosted stops at HOSTED_FULLTEXT_FLOOR once two texts
    land; local aims for FULLTEXT_TARGET. `papers status` runs first so a
    missing mailto or S2 key is named before any get. Cited papers missing
    a PMCID are looked up even when papers is installed, so Europe PMC can
    fill after a timed-out batch.
    """
    fetched: list[int] = []
    requests_spent = 0
    via_papers = paperfetch.available(log)
    eligible = no_open_access = fetch_failed = 0
    stopped = "ran out of eligible sources"
    if via_papers:
        dois: list[str] = []
        prefetch = _fulltext_prefetch_limit()
        for index in order:
            if len(dois) >= prefetch:
                break
            doi = _normalize_doi(papers[index - 1].doi)
            if doi:
                dois.append(doi)
        if dois:
            paperfetch.preflight(log)
            paperfetch.fetch_many_via_papers(
                dois, timeout=_papers_get_timeout(), log=log)
    papers_timeout = _papers_get_timeout()
    for index in order:
        paper = papers[index - 1]
        if len(fetched) >= FULLTEXT_TARGET:
            stopped = f"target of {FULLTEXT_TARGET} reached"
            break
        if _hosted() and len(fetched) >= HOSTED_FULLTEXT_FLOOR:
            stopped = f"hosted floor of {HOSTED_FULLTEXT_FLOOR} reached"
            break
        if requests_spent >= MAX_FULLTEXT_REQUESTS:
            stopped = f"request cap of {MAX_FULLTEXT_REQUESTS} reached"
            break
        eligible += 1
        has_doi = bool(_normalize_doi(paper.doi))

        def _fetch() -> str:
            try:
                return fetch_full_text(
                    paper, log=log, use_cache=use_cache, timeout=papers_timeout)
            except TypeError:
                return fetch_full_text(paper)

        text = ""
        if via_papers and has_doi:
            requests_spent += 1
            text = _fetch()
        if not text and not paper.pmcid and paper.doi:
            requests_spent += 1
            resolve_pmcid(paper, log=log, use_cache=use_cache)
        if not text and paper.pmcid and paper.is_open_access:
            requests_spent += 1
            text = _fetch()
        if text:
            paper.full_text = text
            fetched.append(index)
        elif getattr(paper, "full_text_not_oa", False):
            no_open_access += 1
        elif via_papers and has_doi or (paper.pmcid and paper.is_open_access):
            fetch_failed += 1
        else:
            no_open_access += 1
    return fetched, eligible, no_open_access, fetch_failed, requests_spent, stopped


class NoPapersFound(RuntimeError):
    """No usable papers came back.

    `sources_failed` distinguishes the two cases that look identical from an
    empty list: the topic genuinely has no indexed literature with abstracts,
    or every API refused to answer. Telling a user their topic is unsearchable
    when the real problem is a rate limit sends them off rewording a query that
    was fine.
    """

    def __init__(self, message: str, sources_failed: bool = False):
        super().__init__(message)
        self.sources_failed = sources_failed


class CurationFailed(NoPapersFound):
    """Source labelling came back empty, so the run stops before the write.

    A subclass of `NoPapersFound` on purpose: every caller already has an
    `except NoPapersFound` branch that prints the message and stops, and this
    failure wants exactly that handling. A sibling class would mean two more
    edits in two more files and a third way for a caller to forget.
    """


# Bumped when `Draft.to_dict` changes shape in a way `from_dict` cannot read.
MANIFEST_VERSION = 1


def _int_keys(mapping: dict | None) -> dict:
    """JSON turns int keys into strings; turn the all-digit ones back."""
    out = {}
    for key, value in (mapping or {}).items():
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        out[key] = value
    return out


@dataclass
class Draft:
    """Everything the renderers need, plus the provenance of how it was made."""

    topic: str
    article: dict
    papers: list[Paper]
    curation: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    style_report: dict = field(default_factory=dict)
    # {1-based paper index: the full-text excerpt the writer and verifier saw},
    # captured by `generate_draft` once the papers are final. Stored so the
    # verification haystack can be rebuilt from the manifest without refetching,
    # and so a later change to the excerpt budget cannot rewrite history.
    excerpts: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The run manifest: plain JSON, nothing that needs pickle."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "topic": self.topic,
            "article": self.article,
            "papers": [p.to_dict() for p in self.papers],
            "curation": self.curation,
            "verification": self.verification,
            "provenance": self.provenance,
            "style_report": self.style_report,
            "excerpts": {str(i): text for i, text in self.excerpts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Draft":
        """Rebuild a Draft from a manifest written by `to_dict`."""
        version = data.get("manifest_version", MANIFEST_VERSION)
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest_version {version} is not supported "
                f"(this build reads version {MANIFEST_VERSION})"
            )
        curation = dict(data.get("curation") or {})
        if curation.get("relevance"):
            curation["relevance"] = _int_keys(curation["relevance"])
        return cls(
            topic=data["topic"],
            article=data["article"],
            papers=[Paper.from_dict(p) for p in data.get("papers") or []],
            curation=curation,
            verification=data.get("verification") or {},
            provenance=data.get("provenance") or {},
            style_report=data.get("style_report") or {},
            excerpts=_int_keys(data.get("excerpts")),
        )

    @property
    def cited_refs(self) -> set[int]:
        return {
            r for r in self.article.get("references", [])
            if isinstance(r, int) and 1 <= r <= len(self.papers)
        }

    def paywalled_cited(self) -> list[Paper]:
        """Cited papers `papers` confirmed have no open-access copy."""
        return [
            self.papers[i - 1]
            for i in sorted(self.cited_refs)
            if self.papers[i - 1].full_text_not_oa
            and _normalize_doi(self.papers[i - 1].doi)
        ]

    def summary(self) -> str:
        """One greppable line: what was cited, and what to distrust.

        The GitHub workflow surfaces this in its comment, so keep it one line.
        """
        relevance = self.curation.get("relevance") or {}
        cited = self.cited_refs
        direct = sum(1 for r in cited if relevance.get(r) == "direct") if relevance else None
        n_unverified = len(self.verification.get("unverified") or [])
        n_misattributed = len(self.verification.get("misattributed") or [])

        parts = f"{len(cited)} of {len(self.papers)} screened sources cited"
        if direct is not None:
            parts += f"; {direct} directly on-topic"
        if n_unverified:
            parts += f"; ⚠ {n_unverified} figure(s) not found in the cited sources"
        if n_misattributed:
            parts += f"; ⚠ {n_misattributed} figure(s) credited to the wrong source"
        if relevance and not direct:
            parts += "; ⚠ no directly on-topic source found"
        # style_report is optional on the dataclass; a report that never ran is
        # not the same as a clean one, but for a one-line summary "clean" is the
        # honest reading — generate_draft always populates it.
        n_style = len(style_errors(self.style_report)) if self.style_report else 0
        parts += "; prose style clean" if not n_style else f"; ⚠ {n_style} prose-style issue(s)"
        return parts


# How many times `enforce_style` may send the prose back to the model. The
# second pass is gated on the first having *worked*: a pass only repeats after
# a revision that was accepted, and acceptance means strictly fewer errors. So
# an error the model cannot fix costs one call, not two, and nothing loops.
# Two, not three: the runs that motivated this ended at 3 -> 1 and 2 -> 1, a
# residual of one, which is what a second pass is sized for (#146).
MAX_STYLE_PASSES = 2


def enforce_style(
    article: dict,
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    papers: list[Paper] | None = None,
    curation: dict | None = None,
) -> tuple[dict, dict]:
    """Check the prose against journal conventions and, if it misses, revise.

    Allows up to MAX_STYLE_PASSES passes, with a second pass only running if the
    first pass was accepted (i.e. it reduced the error count). The revision is
    only accepted if it actually reduces the error count and keeps the draft
    intact — a revision that drops citations or sections is discarded.

    `papers` matters when the draft failed for thinness or repetition: those can
    only be fixed by adding grounded material, and without the sources the model
    has nothing to add but more words about the same points. They are forwarded
    to the revision only in that case — see the note at `needs_sources`.
    """
    # The section floor scales with how much directly on-topic evidence there is;
    # demanding five sections from three usable abstracts invites padding.
    direct_sources = ((curation or {}).get("counts") or {}).get("direct")
    report = check_style(article, direct_sources=direct_sources)
    problems = style_errors(report)
    if not problems:
        log("Prose style: clean.")
        log(format_style(report))
        return article, report

    for attempt in range(1, MAX_STYLE_PASSES + 1):
        log(
            f"Prose style: {len(problems)} issue(s) against journal conventions; "
            f"revising (pass {attempt} of {MAX_STYLE_PASSES})..."
        )
        # Both of these are recomputed from the *current* report: a second pass
        # is fixing whatever the first one left behind, which need not be the
        # kind of failure the first one was fixing.
        #
        # The revision replaces named blocks, which is far cheaper than
        # regenerating the article — but a block can only be replaced if it
        # exists. `too-few-sections` is the one failure whose fix is a section
        # that is not there yet, so it is also the one that still pays for a
        # whole rewrite.
        rewrite_whole = any(i["rule"] == "too-few-sections" for i in problems)

        # The sources go along only when the draft failed for thinness or
        # repetition. `revision_brief` already splits the two cases: a substance
        # failure is told to pull specific findings out of the sources, and a
        # register failure is told to reword and add nothing. Sending 20
        # abstracts and 60,000 characters of full text to fix a contraction is
        # ~30,000 input tokens the model is explicitly forbidden to use — and
        # material it cannot use is material it can still be distracted by.
        needs_sources = any(i["rule"] in SUBSTANCE_RULES for i in problems)
        if papers and not needs_sources:
            log("  (register-only fixes; revising against the draft alone)")

        try:
            revised = revise_prose(
                article, revision_brief(report), model=model, api_key=api_key,
                papers=papers if needs_sources else None,
                curation=curation, rewrite_whole=rewrite_whole,
            )
        except Exception as exc:
            log(f"  revision failed ({exc}); keeping the draft as it stands.")
            break

        # Intactness is measured against the draft in hand, not the draft this
        # function was handed: after an accepted pass, that is the revision.
        if is_briefing(article):
            intact = (
                len(revised.get("references") or []) >= len(article.get("references") or [])
                and len(revised.get("findings") or []) >= len(article.get("findings") or [])
            )
        else:
            intact = (
                len(revised.get("references") or []) >= len(article.get("references") or [])
                and len(revised.get("sections") or []) == len(article.get("sections") or [])
            )
        revised_report = check_style(revised, direct_sources=direct_sources)
        revised_problems = style_errors(revised_report)
        if not intact or len(revised_problems) >= len(problems):
            reason = (
                "revision dropped citations or sections" if not intact
                else "revision did not improve"
            )
            log(f"  {reason}; keeping the draft as it stands.")
            break

        log(f"  revised: {len(problems)} -> {len(revised_problems)} issue(s).")
        article, report, problems = revised, revised_report, revised_problems
        if not problems:
            log("  prose style now clean.")
            break

    log(format_style(report))
    return article, report


# How many times a draft's figures may go back to the model. One, not two: this
# pass removes or rewords, it never researches, so a figure the model could not
# fix on the first attempt it cannot fix on the second either (#189).
MAX_STATISTIC_PASSES = 1


def enforce_statistics(
    article: dict,
    papers: list[Paper],
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    style_report: dict | None = None,
    direct_sources: int | None = None,
) -> tuple[dict, dict, dict]:
    """Check the figures and, if any missed, ask once for them to be removed.

    Returns (article, verification, style_report) -- all three describing the
    article that actually ships.
    """
    verification = check_statistics(article, papers)
    unverified = verification.get("unverified") or []
    misattributed = verification.get("misattributed") or []
    flagged = len(unverified) + len(misattributed)

    if flagged == 0:
        log("Statistics: every figure checked out.")
        return article, verification, style_report or {}

    for attempt in range(1, MAX_STATISTIC_PASSES + 1):
        log(
            f"Statistics: {flagged} figure(s) could not be verified in the cited "
            f"sources; revising (pass {attempt} of {MAX_STATISTIC_PASSES})..."
        )
        try:
            revised = revise_statistics(
                article, statistics_brief(verification), model=model, api_key=api_key
            )
        except Exception as exc:
            log(f"  revision failed ({exc}); keeping the draft as it stands.")
            break

        if is_briefing(article):
            intact = (
                len(revised.get("references") or []) >= len(article.get("references") or [])
                and len(revised.get("findings") or []) >= len(article.get("findings") or [])
            )
        else:
            intact = (
                len(revised.get("references") or []) >= len(article.get("references") or [])
                and len(revised.get("sections") or []) == len(article.get("sections") or [])
            )

        revised_v = check_statistics(revised, papers)
        revised_flagged = len(revised_v.get("unverified") or []) + len(revised_v.get("misattributed") or [])
        fewer_flags = revised_flagged < flagged
        no_new_numbers = (revised_v.get("total") or 0) <= (verification.get("total") or 0)

        if not intact or not fewer_flags or not no_new_numbers:
            if not intact:
                reason = "revision dropped citations or sections"
            elif not fewer_flags:
                reason = "revision did not reduce flagged figures"
            else:
                reason = "revision introduced new numbers"
            log(f"  {reason}; keeping the draft as it stands.")
            break

        log(f"  revised: {flagged} -> {revised_flagged} flagged figure(s).")
        article, verification = revised, revised_v
        style_report = check_style(article, direct_sources=direct_sources)
        flagged = revised_flagged
        if flagged == 0:
            log("  statistics now clean.")
            break

    return article, verification, style_report or {}


def _referenced_source_pass(
    topic: str,
    papers: list[Paper],
    curation: dict,
    exhausted: set[str],
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    core_entity: str = "",
    pico: dict | None = None,
) -> dict:
    """One-hop citation follow-up (issue #243).

    Reads the reference lists of up to REFERENCED_SEEDS direct syntheses, and
    the papers citing the top direct trial, resolves them via OpenAlex, merges
    via merge_candidates without renumbering (capped at NAMED_SOURCE_LIMIT new
    records), and re-curates only the new records. Shares the run's exhausted
    set with gather_evidence and _named_source_pass; never raises.
    """
    empty = {"seeds": [], "cited_seed": "", "added": 0}

    if "openalex" in exhausted:
        log("  citation follow-up skipped: OpenAlex already failed this run")
        return empty

    relevance = curation.get("relevance") or {}
    direct = [i for i in full_text_order(papers, relevance) if relevance.get(i) == "direct"]

    synthesis_indices = [i for i in direct if paper_design(papers[i - 1]) == "synthesis"][:REFERENCED_SEEDS]
    trial_indices = [i for i in direct if paper_design(papers[i - 1]) == "trial"]
    trial_idx = trial_indices[0] if trial_indices else None

    if not synthesis_indices and trial_idx is None:
        log("  citation follow-up skipped: no direct synthesis or trial to follow")
        return empty

    seed_titles = [papers[i - 1].title for i in synthesis_indices]
    trial_title = papers[trial_idx - 1].title if trial_idx is not None else ""
    if seed_titles:
        log(f"  citation follow-up: reading reference lists of {len(seed_titles)} "
            "synthes" + ("es" if len(seed_titles) != 1 else "is") + ": "
            + "; ".join(f"{t!r}" for t in seed_titles))
    if trial_title:
        log(f"  citation follow-up: following citations of {trial_title!r}")

    per_seed_ids: list[list[str]] = []
    try:
        for i in synthesis_indices:
            _, ref_ids = openalex_work(papers[i - 1])
            if ref_ids:
                per_seed_ids.append(ref_ids)
    except SearchFailure as exc:
        exhausted.add("openalex")
        log(f"  citation follow-up: OpenAlex refused ({exc}); continuing")
        per_seed_ids = per_seed_ids or []

    # Round-robin interleave so one long reference list cannot crowd out others.
    ids: list[str] = []
    seen_ids: set[str] = set()
    if "openalex" not in exhausted:
        cursors = [0] * len(per_seed_ids)
        while len(ids) < REFERENCED_ID_LIMIT and any(
            cursors[j] < len(per_seed_ids[j]) for j in range(len(per_seed_ids))
        ):
            for j in range(len(per_seed_ids)):
                if len(ids) >= REFERENCED_ID_LIMIT:
                    break
                if cursors[j] < len(per_seed_ids[j]):
                    candidate = per_seed_ids[j][cursors[j]]
                    cursors[j] += 1
                    if candidate not in seen_ids:
                        seen_ids.add(candidate)
                        ids.append(candidate)

    referenced_records: list[Paper] = []
    if ids and "openalex" not in exhausted:
        try:
            referenced_records = openalex_records(ids)
            referenced_records.sort(key=lambda p: p.citation_count, reverse=True)
        except SearchFailure as exc:
            exhausted.add("openalex")
            log(f"  citation follow-up: OpenAlex refused ({exc}); continuing")

    citing_records: list[Paper] = []
    cited_seed = ""
    if trial_idx is not None and "openalex" not in exhausted:
        try:
            work_id, _ = openalex_work(papers[trial_idx - 1])
            if work_id:
                citing_records = openalex_citing_works(work_id)
                cited_seed = trial_title
        except SearchFailure as exc:
            exhausted.add("openalex")
            log(f"  citation follow-up: OpenAlex refused ({exc}); continuing")

    candidates = referenced_records + citing_records
    if not candidates:
        return {"seeds": seed_titles, "cited_seed": cited_seed, "added": 0}

    old_len = len(papers)
    new_papers = merge_candidates(papers, candidates, limit=NAMED_SOURCE_LIMIT)
    log(f"  citation follow-up: {len(ids)} referenced id(s) resolved to "
        f"{len(referenced_records)} record(s), {len(citing_records)} citing record(s), "
        f"{len(new_papers)} new after dedupe")

    if new_papers:
        new_curation = curate_sources(topic, new_papers, model=model, api_key=api_key, pico=pico)
        new_rel = new_curation.get("relevance") or {}
        if new_rel:
            for local_idx, label in new_rel.items():
                curation.setdefault("relevance", {})[old_len + local_idx] = label
            merged_rel = curation["relevance"]
            curation["counts"] = {
                level: sum(1 for v in merged_rel.values() if v == level)
                for level in ("direct", "related", "tangential")
            }
            new_counts = {
                level: sum(1 for v in new_rel.values() if v == level)
                for level in ("direct", "related", "tangential")
            }
            log(f"  relevance (new records): {new_counts.get('direct', 0)} direct / "
                f"{new_counts.get('related', 0)} related / {new_counts.get('tangential', 0)} tangential")
        else:
            log("  WARNING: curation of referenced sources returned no usable labels. "
                "The new records are unlabelled and will not be fetched in full text.")

    return {"seeds": seed_titles, "cited_seed": cited_seed, "added": len(new_papers)}


def _named_source_pass(
    topic: str,
    papers: list[Paper],
    curation: dict,
    exhausted: set[str],
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    outcomes: list[dict] | None = None,
    core_entity: str = "",
    pico: dict | None = None,
) -> dict:
    """Follow up papers and trials named in the top curated abstracts.

    Scans the top NAMED_SOURCE_SCAN abstracts, extracts DOIs and study names,
    queries the scholarly APIs via gather_evidence with patient=False and the
    shared exhausted set, filters results with named_matches, merges via
    merge_candidates without renumbering, and re-curates only the newly added records.
    """
    relevance = curation.get("relevance") or {}
    mri = curation.get("most_relevant_index")
    candidates_to_scan: list[int] = []
    if isinstance(mri, int) and 1 <= mri <= len(papers) and relevance.get(mri) in ("direct", "related"):
        candidates_to_scan.append(mri)

    # Candidates to scan: MRI first (if direct/related), then full_text_order
    # (which scans reviews and trials first — the abstracts that name landmark trials).
    for idx in full_text_order(papers, relevance):
        if idx not in candidates_to_scan:
            candidates_to_scan.append(idx)

    scanned_indices = candidates_to_scan[:NAMED_SOURCE_SCAN]
    requests: list[str] = []
    for idx in scanned_indices:
        paper = papers[idx - 1]
        # Extraction reads abstracts only. Full text is not yet fetched at this
        # point in the pipeline so that newly added landmark papers can participate
        # in the full-text deep-read pass.
        for ref in named_references(paper.abstract):
            if ref not in requests:
                requests.append(ref)
            if len(requests) >= NAMED_SOURCE_LIMIT:
                break
        if len(requests) >= NAMED_SOURCE_LIMIT:
            break

    if not requests:
        return {"queries": [], "added": 0}

    log(f"Following up {len(requests)} paper(s) named in the top {len(scanned_indices)} abstract(s): "
        + "; ".join(f"{q!r}" for q in requests))

    extra_records = gather_evidence(
        requests,
        max_papers=NAMED_SOURCE_LIMIT * 2,
        per_query=NAMED_SOURCE_PER_QUERY,
        topic=topic,
        core_entity=core_entity,
        log=log,
        outcomes=outcomes,
        exhausted=exhausted,
        patient=False,
    )

    matched, dropped = filter_named_matches(extra_records, requests)
    old_len = len(papers)
    new_papers = merge_candidates(papers, matched, limit=NAMED_SOURCE_LIMIT)
    log(f"  named-source pass: {len(requests)} requested, {len(extra_records)} records returned, "
        f"{len(matched)} matched, {len(new_papers)} new after dedupe")
    if dropped:
        log("    dropped as generic (matched most of what came back): "
            + "; ".join(f"{q!r}" for q in dropped))

    if new_papers:
        new_curation = curate_sources(topic, new_papers, model=model, api_key=api_key, pico=pico)
        new_rel = new_curation.get("relevance") or {}
        if new_rel:
            for local_idx, label in new_rel.items():
                curation.setdefault("relevance", {})[old_len + local_idx] = label
            merged_rel = curation["relevance"]
            curation["counts"] = {
                level: sum(1 for v in merged_rel.values() if v == level)
                for level in ("direct", "related", "tangential")
            }
            new_counts = {
                level: sum(1 for v in new_rel.values() if v == level)
                for level in ("direct", "related", "tangential")
            }
            log(f"  relevance (new records): {new_counts.get('direct', 0)} direct / "
                f"{new_counts.get('related', 0)} related / {new_counts.get('tangential', 0)} tangential")
        else:
            log("  WARNING: curation of named sources returned no usable labels. "
                "The new records are unlabelled and will not be fetched in full text.")

    return {"queries": requests, "added": len(new_papers)}


def generate_draft(
    topic: str,
    *,
    style_note: str = "",
    max_papers: int = DEFAULT_MAX_PAPERS,
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    long: bool = False,
    search_terms: list[str] | None = None,
    pico: dict | None = None,
) -> Draft:
    """Research and write one briefing (or a `--long` Review). Raises NoPapersFound
    if the search comes back empty.

    `api_key` overrides the environment for this call only — the server passes
    the caller's key here and must never route it through `os.environ`.
    `long=True` is the parked Review path; the default artefact is the briefing.
    """
    # Before anything paid. `plan_queries` is an LLM call, and on a day when
    # every scholarly API is refusing, the caller used to pay for it and then be
    # told the run was doomed from the start (#96).
    _preflight_sources(topic, log)

    supplied = clean_search_terms(search_terms)
    if supplied:
        log(f"Using {len(supplied)} search term(s) from the idea card: "
            + "; ".join(supplied) + " (the planner may add one more)")
    card_pico = clean_pico(pico)
    if card_pico:
        log("Idea card names: " + ", ".join(card_pico))
    log(f"Planning search queries for: {topic}")
    queries, core_entity = plan_queries(
        topic, model=model, api_key=api_key, search_terms=supplied,
        pico=card_pico, log=log,
    )
    log("Queries: " + "; ".join(queries) + (f"  (core: {core_entity})" if core_entity else ""))

    log("Fetching journal articles...")
    outcomes: list[dict] = []
    exhausted: set[str] = set()
    # Hosted has no Semantic Scholar key; the 30s patient wait never recovers
    # it from Render's shared IP and spends a third of the proxy window.
    hosted = os.environ.get("ARTICLEGEN_STATELESS", "").strip().lower() in (
        "1", "true", "yes")
    papers = gather_evidence(
        queries, max_papers=max_papers, topic=topic, core_entity=core_entity,
        log=log, outcomes=outcomes, exhausted=exhausted,
        patient=not hosted,
    )
    if not papers:
        failures = [o for o in outcomes if o["error"]]
        if len(failures) == len(outcomes) and outcomes:
            reasons = sorted({f"{o['source']}: {o['error']}" for o in failures})
            raise NoPapersFound(
                "The scholarly APIs did not respond, so no evidence could be gathered. "
                "This is not a problem with the topic. " + "; ".join(reasons),
                sources_failed=True,
            )
        raise NoPapersFound(
            "No papers with abstracts were found for this topic. Try a broader or "
            "differently worded topic — or wait a minute and retry, since the open "
            "APIs throttle under load."
        )
    log(f"Collected {len(papers)} candidate papers.")
    # Feeds the pre-flight memo: a server that has heard from a source this
    # recently has no reason to spend a probe on the next request.
    _note_sources_answered()

    log("Assessing source relevance...")
    curation = curate_sources(topic, papers, model=model, api_key=api_key, pico=card_pico)
    counts = curation.get("counts") or {}
    if counts:
        log(f"  relevance: {counts.get('direct', 0)} direct / "
            f"{counts.get('related', 0)} related / {counts.get('tangential', 0)} tangential")
    # `curate_sources` degrades to an empty result on any failure, so a curation
    # that returned nothing looks exactly like one that labelled every source
    # tangential: "0 direct / 0 related / 0 tangential", logged and passed over.
    # That is the quietest way this pipeline can go wrong — the relevance gate
    # is what stops topic drift, and full-text fetching skips every unlabelled
    # source, so the draft downgrades to abstracts-only for a reason nothing
    # reports. Stop the run here: an unlabelled pool means the gate that prevents
    # topic drift is off, and the failure is invisible on the finished page (#168).
    if papers and not curation.get("relevance"):
        log("  WARNING: curation returned no usable labels for any of the "
            f"{len(papers)} sources. The relevance gate is not protecting this "
            "draft from topic drift, and no full text will be fetched. "
            "Re-run, or draft on a different provider.")
        reason = curation.get("error") or "no reason was reported"
        raise CurationFailed(
            f"Source relevance labelling failed ({reason}), so the run stopped "
            f"before writing. {len(papers)} papers were found, but none could be "
            "labelled, and without labels the relevance gate cannot keep "
            "off-topic evidence out of the briefing and no full text is "
            "fetched. Nothing was charged for the writing step. Try again, or "
            "draft on a different model."
        )

    # One-hop citation follow-up (issue #243): reference lists of the top direct
    # syntheses, and papers citing the top direct trial. Runs before the
    # named-source pass so a paper found here can also be scanned for names, and
    # before the full-text loop so it can be deep-read.
    referenced = _referenced_source_pass(
        topic, papers, curation, exhausted, model=model, api_key=api_key,
        log=log, core_entity=core_entity, pico=card_pico,
    )

    # Named-source pass (issue #165): look up landmark papers/trials named in
    # the top abstracts, merge into the candidate pool, and re-curate only the
    # new records. Runs BEFORE the full-text loop so new landmark papers can be
    # deep-read.
    named = _named_source_pass(
        topic, papers, curation, exhausted, model=model, api_key=api_key, log=log,
        outcomes=outcomes, core_entity=core_entity, pico=card_pico,
    )

    # Full-text grounding used to run *before* the writer chose citations, so
    # FULLTEXT_TARGET filled with uncited OA papers while cited Cochrane
    # reviews stayed abstract-only. The writer now drafts from abstracts, then
    # only cited eligible sources are fetched (still direct/related only,
    # design-weighted among them). If any cited full text lands, the briefing
    # is written once more with those texts. Methods and Table 1 still record
    # what the writer was shown: a failed rewrite drops the fetched text.
    #
    # Two separate bounds, because they limit different things. FULLTEXT_TARGET
    # stops once the excerpt budget is full; MAX_FULLTEXT_REQUESTS stops a topic
    # with no open-access literature from spending a request per paper to learn
    # that. Neither bounds tokens — `sources.full_text_excerpts` does that.
    #
    # A paper reaches the fetch only after `resolve_pmcid` has had a chance to
    # discover it is open access, which is what a paper found via OpenAlex
    # rather than Europe PMC always needed and never got.

    _n_direct = ((curation or {}).get("counts") or {}).get("direct")
    _shown = len(papers) - sum(
        1 for label in (curation.get("relevance") or {}).values() if label == "tangential"
    )
    log(f"  cite ceiling: {cite_target(_n_direct, _shown)} "
        f"({_n_direct if _n_direct is not None else 'unlabelled'} direct of {_shown} shown)")

    compose = write_article if long else write_briefing
    log("Writing the review (this can take a few minutes)..." if long
        else "Writing the briefing (this can take a few minutes)...")
    article = compose(
        topic, papers, model=model, style_note=style_note, curation=curation, api_key=api_key
    )

    relevance = curation.get("relevance") or {}
    cited_set = set(_cited_indices(article, len(papers)))
    # Still direct and related only — tangential sources are background even
    # if the writer cited one. Among the cited eligible set, keep the #166
    # order (relevance tier → design → recency → rank).
    order = [i for i in full_text_order(papers, relevance) if i in cited_set]

    log("Fetching open-access full texts of cited sources...")
    fetched, eligible, no_open_access, fetch_failed, requests_spent, stopped = (
        _retrieve_full_texts(papers, order, log))

    n_papers_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "papers")
    n_epmc_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "europe_pmc")
    breakdown = ""
    if n_papers_via > 0 and n_epmc_via > 0:
        breakdown = f" ({n_papers_via} via papers, {n_epmc_via} via Europe PMC)"
    elif n_papers_via > 0:
        breakdown = f" ({n_papers_via} via papers)"

    log(f"  full text retrieved for {len(fetched)} cited source(s)"
        + (f": {fetched}" if fetched else " (none)")
        + f" in {requests_spent} request(s){breakdown}")
    log(f"  stopped because: {stopped}. Of {eligible} eligible cited source(s), "
        f"{no_open_access} had no open-access copy and {fetch_failed} were "
        "open access but returned no text.")
    if stopped.startswith("request cap"):
        log(f"  NOTE: the cap bound before the target. Raising "
            f"MAX_FULLTEXT_REQUESTS would find more, at more requests against "
            "the shared Europe PMC quota.")
    log("  " + _read_subset_skew(papers, fetched))
    _log_paywalled_cited(papers, cited_set, log)

    if fetched:
        log("Rewriting with cited full text...")
        try:
            article = compose(
                topic, papers, model=model, style_note=style_note,
                curation=curation, api_key=api_key,
            )
        except Exception as exc:
            log(f"  rewrite failed ({exc}); keeping the abstract-only draft")
            for paper in papers:
                paper.full_text = ""
                paper.full_text_via = ""
            fetched = []
            n_papers_via = n_epmc_via = 0

    # The papers are final from here on. This is the excerpt map the writer
    # was shown and the one `check_statistics` searches, so it is captured
    # now, once, rather than recomputed later against a budget that may have
    # changed.
    excerpts = full_text_excerpts(papers)

    article, style_report = enforce_style(
        article, model=model, api_key=api_key, log=log, papers=papers, curation=curation
    )
    article, verification, style_report = enforce_statistics(
        article, papers, model=model, api_key=api_key, log=log,
        style_report=style_report,
        direct_sources=((curation or {}).get("counts") or {}).get("direct"),
    )

    # Feeds the deterministic Methods section — the search actually performed.
    #
    # `databases` names only the sources that actually returned records. It used
    # to be a hardcoded constant in render.py, so every article stated that both
    # Semantic Scholar and OpenAlex had been searched even when one of them had
    # refused every request. Semantic Scholar's keyless tier currently 429s on
    # every call, which made that claim false in every article the pipeline
    # produced. The Methods section is the one place in this project that must
    # not overstate what was done.
    answered = {o["source"] for o in outcomes if o["count"]}
    databases = [name for key, name in DATABASE_NAMES.items() if key in answered]

    provenance = {
        "queries": queries,
        "core_entity": core_entity,
        "databases": databases,
        "model": resolve_provider(model, api_key)[1],
        "date": _au_date(),
        # Which sources (1-based indices) the writer saw full text for. The
        # Methods section is written from this — like `databases`, it records
        # what actually happened, never what was intended.
        "full_text_sources": sorted(fetched),
        "full_text_via": {"papers": n_papers_via, "europe_pmc": n_epmc_via},
    }
    if named.get("queries"):
        provenance["named_sources"] = {"queries": named["queries"], "added": named["added"]}
    if referenced.get("seeds") or referenced.get("cited_seed"):
        provenance["referenced_sources"] = {
            "seeds": referenced["seeds"],
            "cited_seed": referenced["cited_seed"],
            "added": referenced["added"],
        }

    return Draft(
        topic=topic,
        article=article,
        papers=papers,
        curation=curation,
        verification=verification,
        provenance=provenance,
        style_report=style_report,
        excerpts=excerpts,
    )


def _log_paywalled_cited(papers: list[Paper], cited: set[int], log: Logger) -> None:
    """One heading, then each paywalled cited DOI. Zero cost; always logged."""
    rows = [
        papers[i - 1]
        for i in sorted(cited)
        if papers[i - 1].full_text_not_oa and _normalize_doi(papers[i - 1].doi)
    ]
    if not rows:
        return
    log("Paywalled cited sources (one CKN download away):")
    for paper in rows:
        log(f"  {_normalize_doi(paper.doi)}  {paper.title}")


def _clear_fetched_text(papers: list[Paper]) -> None:
    for paper in papers:
        paper.full_text = ""
        paper.full_text_via = ""
        paper.full_text_not_oa = False


def rerun_draft(
    draft: Draft,
    *,
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    long: bool = False,
) -> Draft:
    """Reuse a run's pool and labels; fetch full text again; rewrite.

    Search and curation are skipped. The cited set is the first run's.
    `use_cache=False` so an ingest since that run is visible, not the
    articlegen failure cache from the first pass.
    """
    papers = draft.papers
    curation = draft.curation
    topic = draft.topic
    _clear_fetched_text(papers)

    cited_set = set(_cited_indices(draft.article, len(papers)))
    relevance = curation.get("relevance") or {}
    order = [i for i in full_text_order(papers, relevance) if i in cited_set]

    compose = write_article if long else write_briefing
    log("Fetching open-access full texts of cited sources...")
    fetched, eligible, no_open_access, fetch_failed, requests_spent, stopped = (
        _retrieve_full_texts(papers, order, log, use_cache=False))

    n_papers_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "papers")
    n_epmc_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "europe_pmc")
    breakdown = ""
    if n_papers_via > 0 and n_epmc_via > 0:
        breakdown = f" ({n_papers_via} via papers, {n_epmc_via} via Europe PMC)"
    elif n_papers_via > 0:
        breakdown = f" ({n_papers_via} via papers)"

    log(f"  full text retrieved for {len(fetched)} cited source(s)"
        + (f": {fetched}" if fetched else " (none)")
        + f" in {requests_spent} request(s){breakdown}")
    log(f"  stopped because: {stopped}. Of {eligible} eligible cited source(s), "
        f"{no_open_access} had no open-access copy and {fetch_failed} were "
        "open access but returned no text.")
    log("  " + _read_subset_skew(papers, fetched))
    _log_paywalled_cited(papers, cited_set, log)

    article = draft.article
    if fetched:
        log("Rewriting with cited full text...")
        try:
            article = compose(
                topic, papers, model=model, style_note="",
                curation=curation, api_key=api_key,
            )
        except Exception as exc:
            log(f"  rewrite failed ({exc}); keeping the first-run draft")
            for paper in papers:
                paper.full_text = ""
                paper.full_text_via = ""
            fetched = []
            n_papers_via = n_epmc_via = 0

    excerpts = full_text_excerpts(papers)
    article, style_report = enforce_style(
        article, model=model, api_key=api_key, log=log, papers=papers, curation=curation
    )
    article, verification, style_report = enforce_statistics(
        article, papers, model=model, api_key=api_key, log=log,
        style_report=style_report,
        direct_sources=((curation or {}).get("counts") or {}).get("direct"),
    )

    provenance = dict(draft.provenance)
    provenance["date"] = _au_date()
    provenance["full_text_sources"] = sorted(fetched)
    provenance["full_text_via"] = {"papers": n_papers_via, "europe_pmc": n_epmc_via}
    if model is not None or api_key is not None:
        provenance["model"] = resolve_provider(model, api_key)[1]

    return Draft(
        topic=topic,
        article=article,
        papers=papers,
        curation=curation,
        verification=verification,
        provenance=provenance,
        style_report=style_report,
        excerpts=excerpts,
    )


@dataclass
class RefreshReport:
    """What one `refresh` found. `draft` is None unless a rewrite was asked for
    AND there was something new to rewrite with."""

    topic: str
    date: str                      # the manifest's run date, verbatim
    queries: list[str]
    screened: int                  # new records after dedupe (labelled)
    new_direct: list[Paper] = field(default_factory=list)
    new_related: list[Paper] = field(default_factory=list)
    curation_error: str = ""       # non-empty when labelling came back empty
    draft: Draft | None = None

    def summary(self) -> str:
        """One greppable line, same idea as `Draft.summary()`."""
        return (
            f"{len(self.new_direct)} new direct record(s) since {self.date}; "
            f"{self.screened} new record(s) screened; "
            + ("rewritten" if self.draft is not None else "not rewritten")
        )


def refresh_draft(
    draft: Draft,
    *,
    model: str | None = None,
    api_key: str | None = None,
    log: Logger = _silent,
    rewrite: bool = False,
    long: bool = False,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> RefreshReport:
    """Re-run the manifest's queries, report records new since that run, and
    rewrite only when asked.

    No `plan_queries` — the queries are the manifest's, so a refresh compares
    like with like rather than against a freshly planned search.
    """
    queries = [q for q in (draft.provenance.get("queries") or []) if q]
    if not queries:
        raise NoPapersFound(
            "This manifest records no search queries, so there is nothing to "
            "re-run. It was written before provenance carried them — draft "
            "the topic again instead."
        )

    core_entity = draft.provenance.get("core_entity") or ""
    manifest_date = draft.provenance.get("date") or "an unknown date"
    log(f"Re-running {len(queries)} query(ies) from {manifest_date}: " + "; ".join(queries))

    outcomes: list[dict] = []
    records = gather_evidence(
        queries, max_papers=max_papers, topic=draft.topic,
        core_entity=core_entity, log=log, outcomes=outcomes,
    )
    if not records:
        failures = [o for o in outcomes if o["error"]]
        if outcomes and len(failures) == len(outcomes):
            reasons = sorted({f"{o['source']}: {o['error']}" for o in failures})
            raise NoPapersFound(
                "The scholarly APIs did not respond, so this refresh could not "
                "check for anything new. " + "; ".join(reasons),
                sources_failed=True,
            )

    pool = list(draft.papers)          # copy the list: never append to the manifest's
    new_papers = merge_candidates(pool, records, limit=REFRESH_LIMIT)

    if not new_papers:
        log(f"  nothing new since {manifest_date} ({len(records)} record(s) "
            "returned, all already screened)")
        return RefreshReport(
            topic=draft.topic, date=manifest_date, queries=queries, screened=0,
        )

    new_curation = curate_sources(draft.topic, new_papers, model=model, api_key=api_key)
    new_rel = new_curation.get("relevance") or {}
    if not new_rel:
        reason = new_curation.get("error") or "no reason was reported"
        log(f"  WARNING: curation of the new records failed ({reason}); "
            "reporting nothing as new since labelling could not run.")
        return RefreshReport(
            topic=draft.topic, date=manifest_date, queries=queries,
            screened=len(new_papers), curation_error=reason,
        )

    new_direct = [new_papers[i - 1] for i, label in sorted(new_rel.items()) if label == "direct"]
    new_related = [new_papers[i - 1] for i, label in sorted(new_rel.items()) if label == "related"]
    log(f"  relevance (new records): {len(new_direct)} direct / "
        f"{len(new_related)} related / "
        f"{sum(1 for v in new_rel.values() if v == 'tangential')} tangential")

    if not rewrite or not new_direct:
        if rewrite and not new_direct:
            log("  --rewrite: no new direct record, so the briefing is unchanged")
        return RefreshReport(
            topic=draft.topic, date=manifest_date, queries=queries,
            screened=len(new_papers), new_direct=new_direct, new_related=new_related,
        )

    # Rewrite path: at least one new direct record, and the operator asked for it.
    papers = pool  # old records then new, in that order — existing indices never move
    curation = dict(draft.curation)
    relevance = dict(curation.get("relevance") or {})
    old_len = len(draft.papers)
    for local_idx, label in new_rel.items():
        relevance[old_len + local_idx] = label
    curation["relevance"] = relevance
    curation["counts"] = {
        level: sum(1 for v in relevance.values() if v == level)
        for level in ("direct", "related", "tangential")
    }

    compose = write_article if long else write_briefing
    log("Writing the review (this can take a few minutes)..." if long
        else "Writing the briefing (this can take a few minutes)...")
    article = compose(
        draft.topic, papers, model=model, style_note="", curation=curation, api_key=api_key,
    )

    cited_set = set(_cited_indices(article, len(papers)))
    order = [i for i in full_text_order(papers, relevance) if i in cited_set]

    log("Fetching open-access full texts of cited sources...")
    fetched, eligible, no_open_access, fetch_failed, requests_spent, stopped = (
        _retrieve_full_texts(papers, order, log))

    n_papers_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "papers")
    n_epmc_via = sum(1 for i in fetched if papers[i - 1].full_text_via == "europe_pmc")
    breakdown = ""
    if n_papers_via > 0 and n_epmc_via > 0:
        breakdown = f" ({n_papers_via} via papers, {n_epmc_via} via Europe PMC)"
    elif n_papers_via > 0:
        breakdown = f" ({n_papers_via} via papers)"

    log(f"  full text retrieved for {len(fetched)} cited source(s)"
        + (f": {fetched}" if fetched else " (none)")
        + f" in {requests_spent} request(s){breakdown}")
    log(f"  stopped because: {stopped}. Of {eligible} eligible cited source(s), "
        f"{no_open_access} had no open-access copy and {fetch_failed} were "
        "open access but returned no text.")
    log("  " + _read_subset_skew(papers, fetched))
    _log_paywalled_cited(papers, cited_set, log)

    if fetched:
        log("Rewriting with cited full text...")
        try:
            article = compose(
                draft.topic, papers, model=model, style_note="",
                curation=curation, api_key=api_key,
            )
        except Exception as exc:
            log(f"  rewrite failed ({exc}); keeping the abstract-only draft")
            for paper in papers:
                paper.full_text = ""
                paper.full_text_via = ""
            fetched = []
            n_papers_via = n_epmc_via = 0

    excerpts = full_text_excerpts(papers)
    article, style_report = enforce_style(
        article, model=model, api_key=api_key, log=log, papers=papers, curation=curation
    )
    article, verification, style_report = enforce_statistics(
        article, papers, model=model, api_key=api_key, log=log,
        style_report=style_report,
        direct_sources=((curation or {}).get("counts") or {}).get("direct"),
    )

    provenance = dict(draft.provenance)
    provenance["date"] = _au_date()
    provenance["queries"] = queries
    provenance["full_text_sources"] = sorted(
        i for i, p in enumerate(papers, start=1) if p.full_text)
    provenance["full_text_via"] = {"papers": n_papers_via, "europe_pmc": n_epmc_via}
    provenance["refreshed_from"] = {
        "date": manifest_date, "added": len(new_papers), "direct": len(new_direct)}
    if model is not None or api_key is not None:
        provenance["model"] = resolve_provider(model, api_key)[1]

    new_draft = Draft(
        topic=draft.topic,
        article=article,
        papers=papers,
        curation=curation,
        verification=verification,
        provenance=provenance,
        style_report=style_report,
        excerpts=excerpts,
    )

    return RefreshReport(
        topic=draft.topic, date=manifest_date, queries=queries,
        screened=len(new_papers), new_direct=new_direct, new_related=new_related,
        draft=new_draft,
    )

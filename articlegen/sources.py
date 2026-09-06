"""Fetch candidate evidence (papers with abstracts) from open scholarly APIs.

Four free, keyless sources are queried:

- Semantic Scholar Graph API (an optional API key raises rate limits — but
  keys are no longer granted to free-domain emails or third-party apps, so
  in practice the contested keyless pool is all this source has, and it
  refuses more often than it answers)
- OpenAlex (an optional mailto address gets you into the "polite pool")
- Europe PMC (no key, no mailto; biomedical/life-science coverage, which
  suits the mental-health topics this is mostly used for)
- arXiv (no key, no mailto; preprints in computing, physics, engineering,
  statistics and economics — the disciplines Europe PMC does not index, and
  where a non-clinical topic would otherwise be left to the two general
  sources alone)

All can be flaky under shared rate limits, so each query tolerates failures —
as long as one source returns results the pipeline keeps going.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import requests

from . import paperfetch

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
CROSSREF_URL = "https://api.crossref.org/works/{doi}"

# How each source is named in an article's Methods section. Keyed by the
# `source` value gather_evidence records in its outcomes.
DATABASE_NAMES = {
    "semantic_scholar": "Semantic Scholar Graph API",
    "openalex": "OpenAlex",
    "europe_pmc": "Europe PMC",
    "arxiv": "arXiv",
}

# Identify ourselves rather than sending requests' default
# "python-requests/x.y". OpenAlex sits behind a CDN, and a stock library
# user-agent arriving from a cloud provider's shared egress IP is the exact
# profile that gets throttled first. OpenAlex also documents the User-Agent as
# a way into the polite pool, so the mailto goes here as well as in the query
# string — belt and braces, since only the header reaches Semantic Scholar.
_UA_BASE = "articlegen/0.1.0 (+https://github.com/bartholomewtj/evidence-brief)"

# Longest we'll honour a Retry-After. A draft has a user watching a progress
# bar, so waiting out a 60s cool-off is worse than failing and letting the
# other source carry the run.
#
# Note what this means in combination with the `exhausted` set below: a source
# that asks for a longer cool-off fails *immediately*, without retrying, and is
# then skipped for the rest of the run. One long Retry-After therefore removes
# a source entirely rather than slowing it down. That is deliberate — but it
# means the pipeline routinely runs on fewer sources than it appears to offer,
# so measure live behaviour with `/api/diag` rather than assuming all three
# answered. OpenAlex sends exactly this kind of long cool-off from shared cloud
# egress IPs.
_MAX_BACKOFF = 30.0

# One extra, patient attempt for the FIRST Semantic Scholar call of a run.
# Measured over four runs spanning more than an hour (#148): every first
# keyless Semantic Scholar query returned HTTP 429 after its three tries, the
# `exhausted` set then skipped the source for the rest of that run, and every
# draft that session was written without it.
#
# The wait sits AT _MAX_BACKOFF, never above it: 30s is the most this codebase
# is willing to make a caller with a progress bar wait. Semantic Scholar only,
# once per run, and once it has failed the source is exhausted exactly as
# before. This does not replace SEMANTIC_SCHOLAR_API_KEY — it only buys back
# the case where the limit clears inside half a minute.
_S2_PATIENT_WAIT = _MAX_BACKOFF

# Spent per run. Reset by gather_evidence, which is the run boundary — the same
# pattern as _recency_query_refused.
_s2_patient_round_spent = False

# How long a search result stays usable. The literature for a topic does not
# change hour to hour, but the free tiers of these APIs refuse constantly — so
# the same query re-run an hour later is far more likely to fail than to return
# something new. Caching turns "unreliable" into "unreliable once per topic".
#
# Set ARTICLEGEN_SEARCH_CACHE_TTL=0 to disable (used by the tests, which need to
# observe every call).
_CACHE_TTL = float(os.environ.get("ARTICLEGEN_SEARCH_CACHE_TTL", 24 * 3600))

# A refusal is cached too, but only briefly. Without this, a rate-limited source
# is re-attempted — three tries with backoff, about ten seconds — by every
# request that arrives while the limit is still in force, which both stalls the
# caller and deepens the throttle that caused it.
_CACHE_FAILURE_TTL = 120.0

_CACHE_MAX_ENTRIES = 256

_cache_lock = threading.Lock()
# (source, query, limit) -> (expires_at, papers, error). A non-empty error means
# the entry records a refusal rather than a result.
_search_cache: dict[tuple[str, str, int], tuple[float, list["Paper"], str]] = {}

# The three caches (search, full text, PMCID) also live in one JSON file so a
# CLI run — a fresh process every time — gets the same 24 hours of memory the
# web server always had. The dicts stay the front layer: a read never touches
# the disk after the first load, and a write goes to the dicts and then to the
# file. Default location `~/.articlegen/cache.json`; `ARTICLEGEN_CACHE_DIR`
# moves the folder (the tests point it at a temp dir so they never touch the
# real one). `ARTICLEGEN_SEARCH_CACHE_TTL=0` switches the file off with the
# rest. The file holds search results, full texts and DOI->PMCID lookups —
# never an API key, which the cache keys do not carry either.
_CACHE_FILE_NAME = "cache.json"
_cache_loaded = False


def _cache_file() -> Path:
    folder = os.environ.get("ARTICLEGEN_CACHE_DIR") or Path.home() / ".articlegen"
    return Path(folder) / _CACHE_FILE_NAME


def _paper_to_json(paper: "Paper") -> dict:
    record = asdict(paper)
    # Full text has its own cache and is not part of what a search returned;
    # a Paper the pipeline has since filled in would otherwise be written back
    # under the search entry, and the file would carry every text twice.
    record["full_text"] = ""
    record["full_text_via"] = ""
    record["full_text_not_oa"] = False
    record["publication_types"] = list(paper.publication_types)
    return record


def _paper_from_json(record: dict) -> "Paper":
    record = dict(record)
    record["publication_types"] = tuple(record.get("publication_types") or ())
    return Paper(**record)


def _load_cache_locked() -> None:
    """Fill the dicts from the file once per process. Caller holds the lock."""
    global _cache_loaded
    if _cache_loaded or _CACHE_TTL <= 0:
        return
    _cache_loaded = True
    path = _cache_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        now = time.time()
        for item in raw.get("search", []):
            if item["expires"] > now:
                key = (item["source"], item["query"], int(item["limit"]))
                papers = [_paper_from_json(p) for p in item["papers"]]
                _search_cache[key] = (item["expires"], papers, item.get("error", ""))
        for key, item in raw.get("full_text", {}).items():
            if item["expires"] > now:
                _fulltext_cache[key] = (item["expires"], item["text"], item.get("status", ""))
        for doi, item in raw.get("pmcid", {}).items():
            if item["expires"] > now:
                _pmcid_cache[doi] = (item["expires"], item["pmcid"], bool(item["open_access"]))
        for doi, item in raw.get("status", {}).items():
            if item.get("expires", 0) > now:
                _status_cache[doi] = (
                    item["expires"],
                    bool(item.get("is_retracted")),
                    item.get("correction_note", ""),
                )
    except FileNotFoundError:
        return
    except Exception as exc:  # corrupt, unreadable, or an older shape
        # One line, then start empty: a broken cache must never fail a run.
        _search_cache.clear()
        _fulltext_cache.clear()
        _pmcid_cache.clear()
        _status_cache.clear()
        print(f"articlegen: ignoring unreadable cache {path}: {exc}", file=sys.stderr)


def _save_cache_locked() -> None:
    """Write the dicts to the file. Caller holds the lock.

    Expired entries are dropped on the way out, so the file never holds more
    than a day of activity. Written to a temp file then renamed, so a run
    killed mid-write leaves the previous file rather than half of a new one.
    """
    if _CACHE_TTL <= 0:
        return
    now = time.time()
    payload = {
        "search": [
            {"source": k[0], "query": k[1], "limit": k[2], "expires": exp,
             "error": error, "papers": [_paper_to_json(p) for p in papers]}
            for k, (exp, papers, error) in _search_cache.items() if exp > now
        ],
        "full_text": {
            key: {"expires": entry[0], "text": entry[1],
                  "status": entry[2] if len(entry) > 2 else ""}
            for key, entry in _fulltext_cache.items() if entry[0] > now
        },
        "pmcid": {
            doi: {"expires": exp, "pmcid": pmcid, "open_access": oa}
            for doi, (exp, pmcid, oa) in _pmcid_cache.items() if exp > now
        },
        "status": {
            doi: {"expires": exp, "is_retracted": retracted, "correction_note": note}
            for doi, (exp, retracted, note) in _status_cache.items() if exp > now
        },
    }
    path = _cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # A read-only home or a full disk costs the next process its memory,
        # not this run its result.
        print(f"articlegen: could not write cache {path}: {exc}", file=sys.stderr)


@contextlib.contextmanager
def _cache_open():
    """Hold the cache lock with the file loaded. Every dict access goes here."""
    with _cache_lock:
        _load_cache_locked()
        yield

_SS_FIELDS = "title,abstract,year,authors,venue,citationCount,externalIds,url,publicationTypes"
_OA_FIELDS = (
    "id,title,publication_year,authorships,primary_location,"
    "cited_by_count,abstract_inverted_index,doi,type,is_retracted,referenced_works"
)

# Crossref registers an update relation on the *notice*, not on the paper it
# corrects: `update-to` on a record means "this record updates that one". The
# `relation.is-corrected-by` key is the other direction and is the one that
# actually fires on a corrected paper, so both are read. Either way the reader
# is being told the record sits inside a correction relationship and is worth a
# look — which is the whole point of the note.
#
# Retraction types are deliberately absent. Retraction is handled by OpenAlex's
# `is_retracted`, which is about the paper itself; printing "Retracted" off a
# Crossref notice record would label the wrong document.
_CROSSREF_UPDATE_LABELS = {
    "correction": "Corrected",
    "corrigendum": "Corrected",
    "erratum": "Corrected",
    "addendum": "Corrected",
    "clarification": "Corrected",
    "expression_of_concern": "Expression of concern",
}


# Inline formatting tags the publishers' JATS leaves in a title. Named
# explicitly rather than matched as `<[^>]+>` (what `_strip_markup` does to
# abstracts): a real title like "Outcomes in adults aged <65 versus >80 years"
# contains a substring a generic tag pattern would eat whole, taking the visible
# text with it. Every tag listed here is character-level, so removing it never
# joins two words.
_TITLE_MARKUP_RE = re.compile(
    r"</?(?:scp|sc|i|b|u|em|strong|italic|bold|underline|monospace|sub|sup|span)\b[^>]*>",
    re.IGNORECASE,
)


def _strip_title_markup(title: str) -> str:
    """Remove publisher markup from a title, leaving the visible text intact.

    OpenAlex returns JATS tags inline ("The <scp>N-PACT</scp> team"), which
    reached Table 1, the reference list and the writer's prompt verbatim — and
    defeated dedupe, because `_normalize_title` kept "scp" as a word and the
    tagged and untagged copies of one paper no longer matched (issue #140).

    The tag goes without a replacement space, then whitespace is collapsed and
    the gaps the tags leave beside brackets and punctuation are closed, so
    "( N-PACT ):" reads "(N-PACT):" whichever way the publisher spaced it.
    """
    text = _TITLE_MARKUP_RE.sub("", title)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]:;,.])", r"\1", text)
    return text


# DOI prefixes owned by preprint servers. A prefix belongs to one registrant, so
# matching one is an identity check rather than a guess — which is what makes
# this safe as a fallback for the sources that publish no type metadata.
#
# 10.1101 needs the extra digit. Cold Spring Harbor Laboratory Press registered
# that prefix and uses it for BOTH bioRxiv/medRxiv postings (10.1101/2024.03.01.
# 583912, or an older bare 10.1101/123456) and its peer-reviewed journals
# (10.1101/gr.*, 10.1101/gad.*, 10.1101/cshperspect.*). A bare `10.1101` check
# would print "not peer reviewed" under every Genome Research paper — a false
# flag is a wrong warning printed in the article, which is worse than a miss.
_PREPRINT_DOI_RES = (
    re.compile(r"^10\.21203/"),          # Research Square
    re.compile(r"^10\.1101/\d"),         # bioRxiv / medRxiv (digit, not a journal code)
    re.compile(r"^10\.48550/arxiv\."),   # arXiv's own registered DOIs
)

_PREPRINT_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)


def _looks_like_preprint(doi: str, url: str) -> bool:
    """True when the identifier itself says the record is a preprint posting.

    The fallback for sources that return no usable type metadata (issue #144).
    Identifier-only on purpose: a title or an abstract can say anything, and a
    venue string arrives too inconsistently to key a printed claim on.
    """
    normalised = _normalize_doi(doi)
    if normalised and any(rx.match(normalised) for rx in _PREPRINT_DOI_RES):
        return True
    return bool(url and _PREPRINT_URL_RE.search(url))


def _clean_types(raw) -> tuple[str, ...]:
    """API document-type metadata as a tuple of lowercase strings.

    The three sources disagree on shape: Semantic Scholar sends a list, OpenAlex
    a single string, Europe PMC one string holding a delimited list. Normalising
    here keeps `paper_design` from knowing which API a paper came from.
    """
    if not raw:
        return ()
    items = [raw] if isinstance(raw, str) else list(raw)
    out = []
    for elem in items:
        if not elem or not isinstance(elem, str):
            continue
        for piece in re.split(r"[;,]", elem):
            cleaned = piece.strip().lower()
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return tuple(out)


# ---------------------------------------------------------------- table facts
#
# Three deterministic reads off an abstract or a source's own metadata, never
# an appraisal (#244): sample size, trial registration, and (Europe PMC only)
# whether a funding statement was declared. No model call, no GRADE, no risk
# of bias, no quality score — these are text, read as text.

# Journals group thousands with a comma *and* with a space — a plain space,
# a no-break space (U+00A0), a thin space (U+2009), or a narrow no-break
# space (U+202F) — "28 431 unique participants" is a real abstract in the
# corpus.
_COUNT = r"\d{1,3}(?:[,    ]\d{3})+|\d+"

# People, and the events that stand for them, as reported in ED/clinical
# research. Plural only — that is how a sample size is stated. "records" is
# deliberately absent: a systematic review's "7732 non-duplicate records
# identified through literature search" is a search-hit count, not a sample
# size, and that exact phrasing is in the corpus (#244).
_PARTICIPANT_NOUNS = (
    "participants|patients|outpatients|inpatients|subjects|adults|children|"
    "adolescents|infants|neonates|women|men|individuals|people|persons|"
    "respondents|volunteers|veterans|students|nurses|clinicians|physicians|"
    "caregivers|families|cases|admissions|presentations|attendances|visits|"
    "encounters|episodes"
)

# Candidate A: "n = 123" / "N=3,671". The `n` must not be preceded by a letter
# or digit, so "ES = 1.28" and "HRSD(17)>or=14" cannot match.
_SAMPLE_EQ_RE = re.compile(rf"(?<![A-Za-z0-9])[Nn]\s*=\s*({_COUNT})")

# Candidate B: "123 participants" / "28 431 unique participants". Up to two
# plain word tokens (no digits, no commas) between the count and the noun, so
# "7 longitudinal studies, with 28 431 unique participants" cannot bind the
# `7` to `participants`.
_SAMPLE_NOUN_RE = re.compile(
    rf"({_COUNT})(?:\s+[A-Za-z][A-Za-z-]*){{0,2}}\s+(?:{_PARTICIPANT_NOUNS})\b",
    re.IGNORECASE,
)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)
_DATE_LEAD_RE = re.compile(
    rf"(?:in|during|since|from|until|through|by|after|before|between|and|to|"
    rf"{_MONTHS})\s*$",
    re.IGNORECASE,
)
_RATE_LEAD_RE = re.compile(r"\d+\s+in\s*$", re.IGNORECASE)
_PAGE_LEAD_RE = re.compile(r"(?:pp?\.|pages?)\s*$", re.IGNORECASE)


def _strip_grouping(text: str) -> int:
    return int(re.sub(r"[,    ]", "", text))


def sample_size(abstract: str) -> int | None:
    """The largest number in `abstract` stated as a sample size, or None.

    Candidates come from "n = 123" and "123 participants" (and similar
    nouns); each is rejected by name before being counted (issue #244) —
    never a grade, just a deterministic text read. Rejections, each a real
    string from the corpus:

    1. A `%` follows immediately — "36.8%" is a rate, not a count.
    2. A `.` plus digit follows — the "1" of "1.46" (an effect size).
    3. Preceded by "<number> in " — "1 in 5 people" is a rate.
    4. Sits in a numeric range or page span — "0.91-2.01", "123-130",
       "200-300 mmHg" — a `-`/`–` with a digit on either side.
    5. Preceded by "p." / "pp." / "page(s)" — a page number.
    6. Looks like a year (1500-2100) preceded by a date word or month name —
       "up to August 2018", "from 2015 to 2019". Only applied to the "n = …"
       pattern: a count bound to a participant noun ("1500 participants") is
       already disambiguated by that noun, so a real trial's "up to 1500
       participants" is not rejected as a year (#244, the PENFUP trap).
    7. Value < 1 or > 10,000,000 — outside any real sample size.
    """
    if not abstract:
        return None
    candidates: list[tuple[int, int, str, bool]] = []  # (start, end, raw, noun_bound)
    for rx, noun_bound in ((_SAMPLE_EQ_RE, False), (_SAMPLE_NOUN_RE, True)):
        for m in rx.finditer(abstract):
            candidates.append((m.start(1), m.end(1), m.group(1), noun_bound))

    survivors: list[int] = []
    for start, end, raw, noun_bound in candidates:
        before = abstract[:start]
        after = abstract[end:]
        if after[:1] == "%":
            continue
        if re.match(r"\.\d", after):
            continue
        if _RATE_LEAD_RE.search(before):
            continue
        if re.match(r"\s*[-–]\s*\d", after) or re.search(r"\d\s*[-–]\s*$", before):
            continue
        if _PAGE_LEAD_RE.search(before):
            continue
        try:
            value = _strip_grouping(raw)
        except ValueError:
            continue
        if not noun_bound and 1500 <= value <= 2100 and _DATE_LEAD_RE.search(before):
            continue
        if value < 1 or value > 10_000_000:
            continue
        survivors.append(value)

    return max(survivors) if survivors else None


# Bounded digit counts on purpose: an unbounded `ISRCTN\d+` would swallow a
# run-on number, and a bounded pattern that misses one record is cheaper than
# a printed id that is wrong.
_REGISTRATION_RE = re.compile(r"\b(NCT\d{8}|ACTRN\d{14}|ISRCTN\d{6,8})\b", re.IGNORECASE)


def registration_id(text: str) -> str:
    """The first trial registration id in `text`, uppercased, or ""."""
    if not text:
        return ""
    m = _REGISTRATION_RE.search(text)
    return m.group(1).upper() if m else ""


@dataclass
class Paper:
    title: str
    abstract: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    citation_count: int = 0
    url: str = ""
    doi: str = ""
    source: str = ""  # which API it came from
    # Full-text plumbing (issue: expand grounding beyond abstracts). `pmcid`
    # and `is_open_access` come from the Europe PMC search response; only a
    # paper with both can have its full text fetched. `full_text` is filled by
    # `fetch_full_text` later in the pipeline, never by search.
    pmcid: str = ""
    is_open_access: bool = False
    full_text: str = ""
    # Which fetcher produced `full_text` ("papers" or "europe_pmc"). Set by
    # fetch_full_text; not populated by search or dedupe.
    full_text_via: str = ""
    # True when `papers` reported a not-OA status (`queued_ckn`, or `no_oa`
    # from the public `paperfetch-oa` package, #173). The pipeline tallies
    # that as no-open-access, not as "OA but returned no text" (#191).
    full_text_not_oa: bool = False
    # Resolvers `papers` listed as tried when the status was not-OA. Passed
    # through to `papers queue --tried` so the pickup row is not blank.
    oa_tried: str = ""
    is_preprint: bool = False
    # Document type as the API reported it, lowercased, e.g. ("journal article",
    # "randomized controlled trial"). Fed to `paper_design` for the full-text
    # order and nothing else — it is never printed, so a source that reports
    # nothing costs nothing. Empty for arXiv, which has no such field.
    publication_types: tuple[str, ...] = ()
    # OpenAlex's `is_retracted`, or a DOI lookup for records from another
    # source (#242). A retracted record is kept in the pool and in the screened
    # count — it was screened — but `writer._writer_context` omits it from what
    # the model sees, so it cannot be cited.
    is_retracted: bool = False
    # Display text for a Crossref correction relation ("Corrected",
    # "Expression of concern"), or "" for the usual case. A note on the
    # reference list entry, never a reason to exclude anything.
    correction_note: str = ""
    # Three facts read off the abstract or the source's own metadata, never
    # judged (#244). They are printed in Table 1 as facts, not as a grade —
    # there is no appraisal anywhere in this pipeline and these do not add one.
    # Largest sample size stated in the abstract, or None when it states none.
    sample_size: int | None = None
    # Trial registration id as printed ("NCT02565745"), or "".
    registration: str = ""
    # True only when Europe PMC's record carries a non-empty grantsList.
    # False means "no grant listed there", never "unfunded" — every other
    # source leaves it False because no other source reports funding at all.
    has_funding_statement: bool = False

    def __post_init__(self) -> None:
        # The one place a title is cleaned. Four search functions build Papers
        # and a fifth would be added without remembering to call this, so the
        # cleaning belongs to the object rather than to each parse site.
        self.title = _strip_title_markup(self.title)
        # The identifier fallback lives here for the same reason the title clean
        # does: one choke point, so a fifth search source inherits it. A parser
        # that already knows from the API's type metadata sets the flag True
        # before construction, and this never clears it.
        if not self.is_preprint:
            self.is_preprint = _looks_like_preprint(self.doi, self.url)
        # Same choke point: a fifth search source inherits the sample-size and
        # registration reads without anyone remembering to call them (#244).
        # Only fill what is still empty, so a parser that already knows more
        # (Europe PMC's registration read, below) wins.
        if self.sample_size is None:
            self.sample_size = sample_size(self.abstract)
        if not self.registration:
            self.registration = registration_id(f"{self.title} {self.abstract}")

    @property
    def author_line(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) > 3:
            return f"{self.authors[0]} et al."
        return ", ".join(self.authors)

    @property
    def link(self) -> str:
        if self.doi:
            doi = self.doi.removeprefix("https://doi.org/")
            return f"https://doi.org/{doi}"
        return self.url

    def to_dict(self) -> dict:
        """Plain JSON-ready dict of every field, for the run manifest."""
        out = asdict(self)
        out["publication_types"] = list(self.publication_types)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        """Rebuild a Paper from `to_dict` output.

        Unknown keys are ignored so a manifest written by a later version still
        loads; a missing key falls back to the field default.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["publication_types"] = tuple(kwargs.get("publication_types") or ())
        return cls(**kwargs)


class SearchFailure(Exception):
    """A scholarly API did not answer. Distinct from answering with nothing.

    These used to be indistinguishable: every failure collapsed to an empty
    list, so a rate-limited or blocked API produced "no papers found for this
    topic" — blaming the user's topic for an infrastructure problem, with
    nothing in the message to suggest retrying.
    """


def _user_agent() -> str:
    """Our User-Agent, carrying the contact address when one is configured."""
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        return f"{_UA_BASE} mailto:{mailto}"
    return _UA_BASE


def _retry_delay(resp: requests.Response | None, fallback: float) -> float | None:
    """How long to wait before retrying, or None to stop trying now.

    A server that sends Retry-After is telling us exactly when it will answer;
    backing off less than that just burns an attempt. Backing off *more* than
    _MAX_BACKOFF isn't worth it either — return None and let the caller fail
    while the other source still has time to answer.
    """
    if resp is None:
        return fallback
    header = resp.headers.get("Retry-After", "")
    try:
        wanted = float(header)
    except ValueError:
        # Retry-After may also be an HTTP date. Neither API sends that form,
        # so fall back rather than carry a date parser for it.
        return fallback
    if wanted > _MAX_BACKOFF:
        return None
    return max(wanted, fallback)


def _on_shared_host() -> bool:
    """True on the public deployment (`ARTICLEGEN_STATELESS=1`).

    That path sits behind a ~100s proxy read timeout. A 30s socket wait that
    then retries twice is 90s of one source, which is how a live draft looks
    like the backend is down.
    """
    return os.environ.get("ARTICLEGEN_STATELESS", "").strip().lower() in ("1", "true", "yes")


def _search_timeout() -> float:
    return 10.0 if _on_shared_host() else 30.0


def _get_with_retry(url: str, params: dict, headers: dict, tries: int = 3) -> requests.Response:
    """Return a 200 response, or raise SearchFailure explaining why not."""
    headers = {"User-Agent": _user_agent(), **headers}
    delay = 2.0
    last = ""
    timeout = _search_timeout()
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.Timeout as exc:
            # A timeout already spent the backoff budget (or the hosted 10s
            # slice of it). Two more 30s waits is how arXiv made /api/diag
            # take 85s and /api/draft miss Cloudflare's ~100s window.
            last = f"{type(exc).__name__}: {exc}"
            fail = SearchFailure(f"{last} (no retry; socket wait is the backoff cap)")
            fail.retry_later = False
            raise fail
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
            resp = None
        else:
            if resp.status_code == 200:
                return resp
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 502, 503):
                exc = SearchFailure(last)  # non-retryable
                exc.retry_later = False
                raise exc
        if attempt < tries - 1:
            wait = _retry_delay(resp, delay)
            if wait is None:
                exc = SearchFailure(f"{last}, cool-off longer than {_MAX_BACKOFF:.0f}s")
                exc.retry_later = False
                raise exc
            time.sleep(wait)
            delay *= 2
    exc = SearchFailure(f"{last} after {tries} attempts")
    exc.retry_later = True
    raise exc


def search_semantic_scholar(query: str, limit: int = 15) -> list[Paper]:
    """Query Semantic Scholar Graph API.

    This is the one source with a patient retry round: under shared rate limits
    the first call routinely 429s after three quick tries (#148), so we give it
    one extra attempt after a 30s wait before declaring failure and exhausting
    the source for the run.
    """
    global _s2_patient_round_spent
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    params = {"query": query, "limit": limit, "fields": _SS_FIELDS}
    try:
        resp = _get_with_retry(SEMANTIC_SCHOLAR_URL, params=params, headers=headers)
    except SearchFailure as exc:
        # `retry_later` defaults True: an unlabelled failure costs one wait,
        # which is cheaper than missing the case this exists for.
        if _s2_patient_round_spent or not getattr(exc, "retry_later", True):
            raise
        _s2_patient_round_spent = True   # set BEFORE the wait: spent is spent,
        time.sleep(_S2_PATIENT_WAIT)     # whether or not this round succeeds
        resp = _get_with_retry(SEMANTIC_SCHOLAR_URL, params=params, headers=headers)
    papers = []
    for item in resp.json().get("data") or []:
        if not item.get("abstract"):
            continue
        external_ids = item.get("externalIds") or {}
        # Semantic Scholar's publicationTypes enum has no preprint value, and
        # externalIds["ArXiv"] is present on plenty of published papers, so keying
        # on it would false-flag them. The DOI fallback covers Research Square,
        # bioRxiv and arXiv-registered DOIs coming through this source.
        papers.append(
            Paper(
                title=item.get("title") or "",
                abstract=item["abstract"],
                year=item.get("year"),
                authors=[a.get("name", "") for a in item.get("authors") or []],
                venue=item.get("venue") or "",
                citation_count=item.get("citationCount") or 0,
                url=item.get("url") or "",
                doi=external_ids.get("DOI") or "",
                source="Semantic Scholar",
                publication_types=_clean_types(item.get("publicationTypes")),
            )
        )
    return papers


def _rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in inverted_index.items():
        if not word or not isinstance(indexes, list):
            continue
        for i in indexes:
            if isinstance(i, int) and i >= 0:
                positions[i] = word
    if not positions:
        return ""
    max_pos = max(positions.keys())
    return " ".join(positions.get(i, "") for i in range(max_pos + 1)).strip()


# How far back the recency-biased companion query reaches. Eight years is wide
# enough to include the consolidation work that follows a finding, and narrow
# enough that the query returns something other than what relevance already gave.
RECENCY_WINDOW_YEARS = 8

# Set when the recency companion query refuses, and cleared at the start of each
# gather. Mirrors gather_evidence's `exhausted` set, which exists because
# retrying a dead source on every query cost more than half the gather time.
_recency_query_refused = False


def _oa_paper(item: dict) -> Paper | None:
    """One OpenAlex result as a Paper, or None when it has no usable abstract."""
    abstract = _rebuild_abstract(item.get("abstract_inverted_index"))
    if not abstract:
        return None
    location = item.get("primary_location") or {}
    source_meta = location.get("source") or {}
    return Paper(
        title=item.get("title") or "",
        abstract=abstract,
        year=item.get("publication_year"),
        authors=[
            (a.get("author") or {}).get("display_name", "")
            for a in item.get("authorships") or []
        ],
        venue=source_meta.get("display_name") or "",
        citation_count=item.get("cited_by_count") or 0,
        url=location.get("landing_page_url") or item.get("id") or "",
        doi=item.get("doi") or "",
        source="OpenAlex",
        is_preprint=(item.get("type") == "preprint"),
        publication_types=_clean_types(item.get("type")),
        is_retracted=bool(item.get("is_retracted")),
    )


def _openalex_page(query: str, limit: int, from_year: int | None = None) -> list[Paper]:
    """One OpenAlex request."""
    filters = ["has_abstract:true"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params = {
        "search": query,
        "per-page": limit,
        "filter": ",".join(filters),
        "select": _OA_FIELDS,
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = _get_with_retry(OPENALEX_URL, params=params, headers={})
    papers = []
    for item in resp.json().get("results") or []:
        paper = _oa_paper(item)
        if paper is not None:
            papers.append(paper)
    return papers


def search_openalex(query: str, limit: int = 15) -> list[Paper]:
    """Relevance search, plus a recency-biased companion query over the same terms.

    OpenAlex's relevance ordering returns old, heavily-cited work. Measured over
    three topics, the median year of a 20-paper page was 2006-2016, with 13-18 of
    20 published before 2015 (issue #38). Ranking can only reorder what it is
    given, and asking for a bigger page makes it worse rather than better: one
    page of 50 returned *more* pre-2015 work, because it simply goes deeper into
    the same ordering.

    A second, date-filtered request fixes it without a hard cutoff that would
    exclude foundational work — on the same three topics the merged pool's median
    moved to 2018-2020 and the count of 2020-or-later papers rose three- to
    tenfold, while the number of pre-2015 papers was unchanged.

    The recency query is best-effort. If it refuses, the relevance results still
    stand: a thinner pool beats no pool, and the caller's `exhausted` bookkeeping
    treats OpenAlex as one source either way.

    A refusal also stops it being attempted again for the rest of the run, for
    the same reason `gather_evidence` keeps an `exhausted` set: the limits are
    per-minute and a run takes seconds, so re-attempting a dead query on every
    subsequent query would spend three tries and about ten seconds each time,
    which is what made gathering take 31s before that set existed.
    """
    global _recency_query_refused
    papers = _openalex_page(query, limit)
    if _recency_query_refused:
        return papers
    cutoff = datetime.date.today().year - RECENCY_WINDOW_YEARS
    try:
        recent = _openalex_page(query, limit, from_year=cutoff)
    except SearchFailure:
        _recency_query_refused = True
        return papers

    seen = {(_normalize_doi(p.doi) or _normalize_title(p.title)) for p in papers}
    for paper in recent:
        key = _normalize_doi(paper.doi) or _normalize_title(paper.title)
        if key not in seen:
            seen.add(key)
            papers.append(paper)
    return papers


def _strip_markup(text: str) -> str:
    """Europe PMC abstracts arrive with embedded HTML (<h4>, <i>, <p>...).

    verify.py checks statistics by substring presence in these abstracts, so
    markup left in place would hide a figure that sits next to a tag.
    """
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _europe_pmc_author(author: dict) -> str:
    """One Europe PMC author, in the given-name-first order the renderer expects.

    Europe PMC returns `fullName` **surname-first** ("Kim SH"); OpenAlex's
    `display_name` is the opposite ("Ari Min"). `render._format_author` takes the
    last token as the surname, so passing `fullName` straight through printed
    "SH, K." for "Kim SH" — every Europe PMC reference in an article had its
    surname and initials swapped, while OpenAlex ones were correct.

    Rather than teach the formatter to guess an order it cannot infer, normalise
    here from the structured fields Europe PMC already returns. Initials are
    split apart ("SH" -> "S H") so each survives as its own initial instead of
    being collapsed into one.
    """
    last = (author.get("lastName") or "").strip()
    if last:
        initials = [c for c in (author.get("initials") or "") if c.isalpha()]
        if initials:
            return f"{' '.join(initials)} {last}"
        first = (author.get("firstName") or "").strip()
        return f"{first} {last}" if first else last
    # Consortium authors carry `collectiveName` and no surname. `fullName` is the
    # last resort: wrong-order for a person, but better than dropping the name.
    return (author.get("collectiveName") or author.get("fullName") or "").strip()


def search_europe_pmc(query: str, limit: int = 15) -> list[Paper]:
    params = {
        # HAS_ABSTRACT restricts server-side, like OpenAlex's has_abstract
        # filter — otherwise most of the page is records we would discard.
        "query": f"({query}) AND HAS_ABSTRACT:y",
        "format": "json",
        # "core" is the smallest result type that includes the abstract.
        "resultType": "core",
        "pageSize": limit,
    }
    resp = _get_with_retry(EUROPE_PMC_URL, params=params, headers={})
    papers = []
    for item in (resp.json().get("resultList") or {}).get("result") or []:
        abstract = _strip_markup(item.get("abstractText") or "")
        if not abstract:
            continue
        # pubYear is a string ("2026"); missing or malformed becomes None and
        # ranking treats it as old.
        try:
            year = int(item.get("pubYear") or "")
        except ValueError:
            year = None
        journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title") or ""
        authors = [
            name
            for a in (item.get("authorList") or {}).get("author") or []
            if (name := _europe_pmc_author(a))
        ]
        # Europe PMC's document types live in pubTypeList.pubType (a list).
        # The flat pubType field is documented but null in practice on search
        # responses (measured live, 21 Aug 2026). Keep flat pubType as a fallback.
        raw_types = (item.get("pubTypeList") or {}).get("pubType") or item.get("pubType")
        cleaned_types = _clean_types(raw_types)
        src, ext_id = item.get("source") or "", item.get("id") or ""
        title = _strip_markup(item.get("title") or "")
        # Measured against the live API on 5 September 2026 with
        # resultType=core: grantsList is present, and is a
        # {"grant": [{"grantId": ..., "agency": ...}]} object, only on records
        # that carry a grant — it is absent (not empty) otherwise, so presence
        # is the whole test. There is no registry-id field in the core search
        # response; keywordList is the only extra place an id turns up, which
        # is why the registration read below is title + abstract + keywords
        # rather than a dedicated field.
        keywords_text = " ".join((item.get("keywordList") or {}).get("keyword") or [])
        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                year=year,
                authors=authors,
                venue=journal,
                citation_count=item.get("citedByCount") or 0,
                url=f"https://europepmc.org/article/{src}/{ext_id}" if src and ext_id else "",
                doi=item.get("doi") or "",
                source="Europe PMC",
                pmcid=item.get("pmcid") or "",
                # Both flags are needed: isOpenAccess without inEPMC means the
                # OA copy lives somewhere Europe PMC cannot serve from.
                is_open_access=(item.get("isOpenAccess") == "Y" and item.get("inEPMC") == "Y"),
                is_preprint=(src == "PPR" or "preprint" in " ".join(cleaned_types)),
                publication_types=cleaned_types,
                has_funding_statement=bool((item.get("grantsList") or {}).get("grant")),
                registration=registration_id(" ".join([title, abstract, keywords_text])),
            )
        )
    return papers


# arXiv asks callers to leave three seconds between requests. Nothing enforces
# it per-key (there is no key), so the penalty for ignoring it is a block on the
# egress IP — which on the hosted backend is shared. A draft makes one arXiv
# call per planned query, so honouring it costs at most a few seconds spread
# across a run that already takes a minute.
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_lock = threading.Lock()
_arxiv_last_call = 0.0

# arXiv's query parser reads a space after `all:` as the end of the term, so a
# multi-word topic sent verbatim searches for the first word alone. Terms are
# ANDed instead.
#
# Measured against the live API, ANDing is the only one of the three
# expressions that behaves: "machine learning emergency department triage" as a
# five-term AND returns 35 on-topic papers, the same words ORed return 135,233
# about anything with a grid in it, and as a quoted phrase, 0.
#
# Function words are dropped before the AND. Not for precision — every paper
# contains "the", so ANDing it excludes nothing — but because a query made of
# nothing else would otherwise become `all:the`, which matches the whole
# archive and ranks it by relevance to "the".
_ARXIV_MAX_TERMS = 8
_ARXIV_STOPWORDS = frozenset(
    "a an and are as at be by for from in is of on or the to with".split())

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _arxiv_query(query: str) -> str:
    """A free-text topic as an arXiv `search_query` expression.

    Short words are kept when they are all there is: "AI in ED" becomes
    `all:AI AND all:ED`, which returns 123 papers where the quoted phrase
    returns 0.
    """
    words = [w for w in re.split(r"[^A-Za-z0-9]+", query) if w]
    terms = [w for w in words if w.lower() not in _ARXIV_STOPWORDS] or words
    return " AND ".join(f"all:{t}" for t in terms[:_ARXIV_MAX_TERMS])


def search_arxiv(query: str, limit: int = 15) -> list[Paper]:
    """Preprints from arXiv, in the disciplines Europe PMC does not cover.

    Two things differ from the other three sources and both are deliberate.
    arXiv publishes no citation counts, so every paper arrives at `_rank_score`
    with a citation weight of zero and is ranked on topic overlap and recency
    alone — right for a preprint server, where the newest work is the point.
    And these are preprints: `venue` says so unless the record carries a
    `journal_ref`, because the reference list is the only place a reader learns
    that a source was never peer reviewed.
    """
    import xml.etree.ElementTree as ET

    with _arxiv_lock:
        global _arxiv_last_call
        wait = _ARXIV_MIN_INTERVAL - (time.time() - _arxiv_last_call)
        if wait > 0:
            time.sleep(wait)
        _arxiv_last_call = time.time()

    resp = _get_with_retry(
        ARXIV_URL,
        params={
            "search_query": _arxiv_query(query),
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        },
        headers={},
    )
    # Atom, not JSON — arXiv offers no JSON representation. A malformed body
    # raises ParseError, which `_search_once` records like any other failure.
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        entry_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        # arXiv reports a bad query as a normal-looking entry whose id points at
        # /api/errors. Parsed as a paper it becomes a source titled "Error".
        if "/api/errors" in entry_id:
            continue
        abstract = _collapse(entry.findtext(f"{_ATOM}summary") or "")
        if not abstract:
            continue
        published = (entry.findtext(f"{_ATOM}published") or "")[:4]
        try:
            year = int(published)
        except ValueError:
            year = None
        authors = [
            name
            for a in entry.findall(f"{_ATOM}author")
            if (name := (a.findtext(f"{_ATOM}name") or "").strip())
        ]
        journal_ref = _collapse(entry.findtext(f"{_ARXIV_NS}journal_ref") or "")
        # Unconditionally True: even when the entry carries a journal_ref, the
        # record we link and cite is the arXiv posting, not the journal's
        # version of record.
        papers.append(
            Paper(
                title=_collapse(entry.findtext(f"{_ATOM}title") or ""),
                abstract=abstract,
                year=year,
                authors=authors,
                venue=journal_ref or "arXiv preprint",
                # arXiv publishes no citation counts. Left at 0 rather than
                # guessed; see the docstring.
                citation_count=0,
                url=entry_id,
                doi=(entry.findtext(f"{_ARXIV_NS}doi") or "").strip(),
                source="arXiv",
                is_preprint=True,
            )
        )
    return papers


def _collapse(text: str) -> str:
    """Whitespace-collapse a field. arXiv wraps abstracts and titles at column 80.

    Same reason `_strip_markup` exists: `verify.py` checks a statistic by
    substring presence, so a figure split across a line break would be reported
    unverifiable.
    """
    return re.sub(r"\s+", " ", text).strip()


_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def _normalize_doi(doi: str) -> str:
    """One spelling for a DOI, so two records of one paper share a merge key.

    The four search sources spell the same DOI three ways — OpenAlex returns
    the resolver URL, Semantic Scholar returns mixed case, Europe PMC returns
    it bare — which is how one paper reached the reference list twice (#139).
    DOI prefixes and suffixes are case-insensitive for lookup, so lowercasing
    is safe.

    Anything that is not a DOI returns "" rather than itself: a junk value
    ("n/a", "unknown") repeated across two unrelated records would otherwise
    become a merge key and collapse two real papers into one.
    """
    text = _DOI_PREFIX_RE.sub("", (doi or "").strip()).strip()
    text = text.rstrip(".,;").lower()
    return text if text.startswith("10.") else ""


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _keyword_overlap(paper: Paper, terms: set[str]) -> int:
    if not terms:
        return 0
    hay = f"{paper.title} {paper.abstract}".lower()
    return sum(1 for t in terms if t in hay)


RECENCY_WEIGHT = 3.0      # value of a brand-new paper, comparable to ~1000 citations
RECENCY_HALF_LIFE = 12    # years over which that value decays to nothing


def _rank_score(paper: Paper, terms: set[str], now: int | None = None) -> tuple:
    """Blend topic-keyword overlap, citation weight, and recency.

    Recency used to be `year / 1000`, which spans 0.02 across two decades while
    the citation term spans 0 to 4 — so it was arithmetically a rounding error
    and old, heavily-cited reviews always won. A real search returned a median
    paper year of 2013, with 2 of 20 papers from 2020 or later. That skews an
    article towards broad old reviews, which carry fewer specific findings than
    recent trials and make the prose vaguer.

    Recency is now on the same scale as the citation term and decays linearly
    to zero over RECENCY_HALF_LIFE years. Citations still matter — this ranks
    a well-cited recent paper top, not merely the newest thing indexed.
    """
    import datetime
    import math

    now = now or datetime.date.today().year
    overlap = _keyword_overlap(paper, terms)
    citation_weight = math.log10(paper.citation_count + 1)
    age = max(0, now - paper.year) if paper.year else RECENCY_HALF_LIFE
    recency = RECENCY_WEIGHT * max(0.0, 1.0 - age / RECENCY_HALF_LIFE)
    return (overlap, citation_weight + recency)


def clear_search_cache() -> None:
    """Forget every cached search, in memory and on disk.

    For tests, and for forcing a fresh look. Marks the file as loaded so the
    next access does not read the entries back from disk.
    """
    global _cache_loaded
    with _cache_lock:
        _search_cache.clear()
        _fulltext_cache.clear()
        _pmcid_cache.clear()
        _status_cache.clear()
        _cache_loaded = True
        try:
            _cache_file().unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"articlegen: could not remove cache {_cache_file()}: {exc}",
                  file=sys.stderr)


# ---------------------------------------------------------------- full text

EUROPE_PMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Back-matter sections that cost tokens and ground nothing. Matched against the
# lowercased section title; anything else is kept in document order. Same list
# as paperfetch (PR #37): a volume or page number in a citation must not sit
# in the verification haystack.
_FULLTEXT_SKIP_TITLES = re.compile(
    r"acknowledg|funding|reference|bibliograph|literature cited|supplementar"
    r"|supporting info|abbreviation|author contribution|conflict|competing"
    r"|availability|ethics|consent|orcid|appendix"
)

# Canonical section names. paperfetch writes a "## Results" marker line
# (title case, blank line after) before each one it finds; meta.json lists
# them lowercase under "sections". Unrecognised headings stay in the body
# under the nearest marker. Match that contract even while PR #37 is open.
SECTION_NAMES = (
    "abstract", "introduction", "methods", "results", "discussion", "conclusions",
)
_SECTION_MARKER = re.compile(
    r"^## (" + "|".join(n.title() for n in SECTION_NAMES) + r")\s*$",
    re.MULTILINE,
)

# Excerpt picker spends the per-paper budget in this order. "discussion or
# conclusions" is one slot: discussion if the paper has it, otherwise
# conclusions. References are never a slot.
_EXCERPT_SECTION_GROUPS = (
    ("abstract",),
    ("results",),
    ("discussion", "conclusions"),
    ("methods",),
    ("introduction",),
)

# Heading text -> canonical name. Same patterns as paperfetch, matched
# against the whole heading after numbering ("2.", "III", "2 |") and
# trailing punctuation are stripped.
_HEADING_PREFIX = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[ivx]+)[.):|\s]+)?", re.IGNORECASE)
_CANONICAL = [
    ("abstract", re.compile(r"(structured )?abstract")),
    ("introduction", re.compile(r"introduction|background( and (objectives?|aims?))?")),
    (
        "methods",
        re.compile(
            r"((materials?|patients?|subjects?|participants?|study design|experimental|"
            r"design)( and | & |, ))?(methods?|methodology|procedures?)( and materials?)?"
            r"|experimental (procedures?|section)|study design and methods?"
        ),
    ),
    ("results", re.compile(r"results?( and discussion)?|findings|main results")),
    ("discussion", re.compile(r"discussion")),
    ("conclusions", re.compile(r"(summary and )?conclusions?|concluding remarks")),
    (
        "references",
        re.compile(
            r"references?( and notes| list| cited)?|bibliography|literature cited|works cited"
        ),
    ),
]
_JATS_SEC_TYPES = {
    "intro": "introduction",
    "introduction": "introduction",
    "materials": "methods",
    "methods": "methods",
    "materials|methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
}

# A whole-line references heading, with or without a leftover "## " marker
# or numbering. Used to cut the list out of unmarked PDF dumps.
_REFERENCES_HEADING = re.compile(
    r"(?im)^(?:##\s+)?(?:(?:\d+(?:\.\d+)*|[ivx]+)[.):|\s]+)?"
    r"(?:references?(?: and notes| list| cited)?|bibliography|"
    r"literature cited|works cited)\s*$"
)

# pmcid or "papers:<doi>" -> (expiry, text, status). Same lifecycle as the
# search cache: full text does not change hour to hour, and each fetch is a
# real request against the one scholarly API that reliably answers — do not
# spend it twice. `status` is the papers CLI's outcome and "" on the Europe
# PMC route.
_fulltext_cache: dict[str, tuple[float, str, str]] = {}

# doi -> (expiry, pmcid, is_open_access). See `resolve_pmcid`.
_pmcid_cache: dict[str, tuple[float, str, bool]] = {}

# doi -> (expiry, is_retracted, correction_note). See `check_record_status`.
_status_cache: dict[str, tuple[float, bool, str]] = {}


def _unpaywall_email() -> str:
    """The contact Unpaywall requires. Never a made-up address — it blocks those."""
    return os.environ.get(
        "OPENALEX_MAILTO", "https://github.com/bartholomewtj/evidence-brief")


def probe_unpaywall(doi: str = "10.1371/journal.pone.0000308") -> dict:
    """Can this host reach Unpaywall right now?

    The full-text path grew a fourth keyless external dependency (#103) and it
    fails soft: if Unpaywall rate-limits, blocks the contact address or changes
    its response shape, every article silently drops to abstracts-only. Soft
    failure is right for the reader and useless for the operator, so it has to
    be visible from outside — the same reason `/api/diag` exists at all.

    The default DOI is a PLOS ONE paper that has been open access since 2007;
    an answer that says otherwise means the response shape changed, which is
    exactly one of the failures worth catching.
    """
    result = {"source": "unpaywall", "doi": doi, "email_set":
              bool(os.environ.get("OPENALEX_MAILTO")), "error": ""}
    try:
        resp = _get_with_retry(UNPAYWALL_URL.format(doi=doi),
                               params={"email": _unpaywall_email()}, headers={})
        data = resp.json()
    except SearchFailure as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not isinstance(data, dict) or "is_oa" not in data:
        result["error"] = "unexpected response shape (no is_oa field)"
        return result
    result["is_oa"] = bool(data.get("is_oa"))
    result["has_oa_location"] = bool(data.get("best_oa_location"))
    if not result["is_oa"]:
        result["error"] = "answered, but reports this known open-access DOI as closed"
    return result


def check_record_status(paper: Paper, use_cache: bool = True, log=lambda msg: None) -> None:
    """Fill `is_retracted` and `correction_note` for one paper, by DOI (#242).

    Two lookups, both best-effort and both soft-failing: a source index that is
    down must cost a note, never a draft.

    * OpenAlex, only when the record came from somewhere else. An OpenAlex
      record already carries `is_retracted` from the search, and a flag once set
      is never cleared — a second opinion that says nothing is not a retraction
      being lifted.
    * Crossref, for the correction relation. Crossref is the registry the
      notices are deposited with; OpenAlex does not expose them.

    Coverage is honest rather than complete: this runs where `resolve_pmcid`
    runs, which is the full-text loop, so a record from Semantic Scholar or
    arXiv that is never a full-text candidate is checked only against whatever
    its own search returned. OpenAlex is the source that supplies most of the
    pool and it is covered for free at search time.
    """
    doi = _normalize_doi(paper.doi)
    if not doi:
        return

    now = time.time()
    if use_cache and _CACHE_TTL > 0:
        with _cache_open():
            entry = _status_cache.get(doi)
        if entry and entry[0] > now:
            paper.is_retracted = paper.is_retracted or entry[1]
            paper.correction_note = paper.correction_note or entry[2]
            return

    retracted, note = False, ""

    if paper.source != "OpenAlex" and not paper.is_retracted:
        try:
            params = {"filter": f"doi:{doi}", "select": "id,doi,is_retracted", "per-page": 1}
            mailto = os.environ.get("OPENALEX_MAILTO")
            if mailto:
                params["mailto"] = mailto
            resp = _get_with_retry(OPENALEX_URL, params=params, headers={})
            results = resp.json().get("results") or []
            if results and results[0].get("is_retracted"):
                retracted = True
        except SearchFailure as exc:
            log(f"  openalex retraction lookup failed for {doi}: {exc}")

    if not paper.correction_note:
        try:
            resp = _get_with_retry(CROSSREF_URL.format(doi=doi), params={}, headers={})
            message = (resp.json() or {}).get("message") or {}
            types = [str((u or {}).get("type") or "").strip().lower().replace(" ", "_")
                     for u in (message.get("update-to") or [])]
            if (message.get("relation") or {}).get("is-corrected-by"):
                types.append("correction")
            note = next((_CROSSREF_UPDATE_LABELS[t] for t in types
                         if t in _CROSSREF_UPDATE_LABELS), "")
        except SearchFailure as exc:
            log(f"  crossref update lookup failed for {doi}: {exc}")

    if retracted:
        log(f"  retracted record excluded: {doi}  {paper.title}")

    paper.is_retracted = paper.is_retracted or retracted
    paper.correction_note = paper.correction_note or note

    if _CACHE_TTL > 0:
        ttl = _CACHE_TTL if (retracted or note) else _CACHE_FAILURE_TTL
        with _cache_open():
            _status_cache[doi] = (now + ttl, paper.is_retracted, paper.correction_note)
            _save_cache_locked()


def resolve_pmcid(paper: Paper, use_cache: bool = True, log=lambda msg: None) -> bool:
    """Look a paper's DOI up in Europe PMC to fill in `pmcid` / `is_open_access`.

    Only the Europe PMC *search* returns those two fields, so before this
    existed a paper's full text was reachable only if that search happened to
    be the one that found it. Everything from OpenAlex arrived with an empty
    pmcid and was therefore treated as abstract-only — including wholly
    open-access journals (Frontiers, MDPI, and the like) whose full text
    Europe PMC serves perfectly well. Measured on one real run, that cost 9 of
    11 available full texts: 2 were fetched where 11 could have been.

    If Europe PMC does not yield an open-access PMCID copy, a secondary lookup
    against Unpaywall is performed.

    Returns True when the paper now has a fetchable open-access full text. Up
    to four HTTP calls per unseen DOI now (Europe PMC, Unpaywall, and the
    retraction/correction lookups in `check_record_status`), so callers must
    bound how many they make.
    """
    doi = (paper.doi or "").removeprefix("https://doi.org/").strip()
    # Retraction and correction status, from the same DOI, before the early
    # return: a paper that already has its PMCID still needs checking. The
    # lookups live in their own function because this one's error handling is
    # pinned by test_full_text_dependencies_fail_loudly_enough_to_diagnose.
    if doi:
        check_record_status(paper, use_cache=use_cache, log=log)
    if (paper.pmcid and paper.is_open_access) or not doi:
        return bool(paper.pmcid and paper.is_open_access)

    now = time.time()
    if use_cache and _CACHE_TTL > 0:
        with _cache_open():
            entry = _pmcid_cache.get(doi)
        if entry and entry[0] > now:
            paper.pmcid, paper.is_open_access = entry[1], entry[2]
            return bool(entry[1] and entry[2])

    pmcid, open_access = "", False
    try:
        resp = _get_with_retry(
            EUROPE_PMC_URL,
            # The DOI field is exact-match, so this returns the one record or none.
            params={"query": f'DOI:"{doi}"', "format": "json",
                    "resultType": "core", "pageSize": 1},
            headers={},
        )
        results = (resp.json().get("resultList") or {}).get("result") or []
        if results:
            item = results[0]
            pmcid = item.get("pmcid") or ""
            # Both flags, for the same reason as in `search_europe_pmc`:
            # isOpenAccess without inEPMC means the copy is somewhere else.
            open_access = item.get("isOpenAccess") == "Y" and item.get("inEPMC") == "Y"
    except SearchFailure as exc:
        # A refused lookup is cached briefly, like a refused search, so a
        # throttled Europe PMC is not asked the same question twice a run.
        # Soft-failing is right for the reader; failing *silently* left the
        # operator with halved full-text coverage and nothing to look at (#104).
        log(f"  europe_pmc DOI lookup failed for {doi}: {exc}")

    if not (pmcid and open_access):
        try:
            unpaywall_resp = _get_with_retry(
                UNPAYWALL_URL.format(doi=doi),
                # Unpaywall requires a real contact and blocks addresses that
                # bounce, so never send a made-up one. It accepts a project URL
                # when no address is configured.
                params={"email": _unpaywall_email()},
                headers={},
            )
            data = unpaywall_resp.json()
            if isinstance(data, dict) and data.get("is_oa") is True and data.get("best_oa_location"):
                best_oa = data.get("best_oa_location") or {}
                extracted_pmcid = best_oa.get("pmcid") or data.get("pmcid")
                if not extracted_pmcid:
                    pmc_match = re.search(
                        r"PMC\d+",
                        str(best_oa.get("url") or "")
                        + " "
                        + str(best_oa.get("url_for_pdf") or "")
                        + " "
                        + str(best_oa.get("pmh_id") or ""),
                    )
                    if pmc_match:
                        extracted_pmcid = pmc_match.group(0)
                if extracted_pmcid:
                    extracted_pmcid = str(extracted_pmcid).strip()
                    if extracted_pmcid and not extracted_pmcid.startswith("PMC") and extracted_pmcid.isdigit():
                        extracted_pmcid = f"PMC{extracted_pmcid}"
                    pmcid = extracted_pmcid
                open_access = True
        except SearchFailure as exc:
            # Same reasoning as above, and this is the one the issue was
            # actually about: a blocked contact address here halves full-text
            # coverage across every article and looks like nothing at all.
            log(f"  unpaywall lookup failed for {doi}: {exc}")

    if _CACHE_TTL > 0:
        ttl = _CACHE_TTL if (pmcid or open_access) else _CACHE_FAILURE_TTL
        with _cache_open():
            _pmcid_cache[doi] = (now + ttl, pmcid, open_access)
            _save_cache_locked()

    paper.pmcid, paper.is_open_access = pmcid, open_access
    return bool(pmcid and open_access)


def _jats_text(node) -> str:
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def _jats_body_text(sec, title_node) -> str:
    """Section text without its own title (the marker line carries that)."""
    if title_node is None:
        return _jats_text(sec)
    pieces = [title_node.tail or ""]
    for child in sec:
        if child is title_node:
            continue
        pieces.append(" ".join(child.itertext()))
        pieces.append(child.tail or "")
    text = (sec.text or "") + " ".join(pieces)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_section(title: str) -> str | None:
    """'2. MATERIALS AND METHODS' -> 'methods'; None when not a standard heading."""
    t = _HEADING_PREFIX.sub("", (title or "").strip().lower())
    t = re.sub(r"\s+", " ", t).strip(" .:;-")
    if not t:
        return None
    for name, pattern in _CANONICAL:
        if pattern.fullmatch(t):
            return name
    return None


def _assemble_sections(segments: list[tuple[str | None, str]]) -> str:
    """[(canonical name or None, body text)] -> text with paperfetch marker lines."""
    parts: list[str] = []
    for name, body in segments:
        body = body.strip()
        if name is None:
            if body:
                parts.append(body)
            continue
        parts.append(f"## {name.title()}\n\n{body}" if body else f"## {name.title()}\n")
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


_CITATION_BRACKETS_RE = re.compile(r"\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]")


def _strip_citation_brackets(text: str) -> str:
    """Remove the paper's own bracketed citation numbers.

    They collide with the pipeline's [N] SOURCE-index scheme: a writer reading
    "improves outcomes [ 4 ]" may echo that number as if it were one of ours.
    Every full-text path runs through this, whichever fetcher produced it.
    """
    if not text:
        return ""
    return _CITATION_BRACKETS_RE.sub("", text)


def _parse_fulltext_xml(xml_text: str) -> str:
    """JATS XML -> plain text with paperfetch-style section markers.

    Recognised sections get a ``## Results`` marker line (title case), a blank
    line, then the body without the heading. Back matter and the reference
    list are dropped. Falls back to unmarked body text when the article has
    no recognised <sec> structure. Markup is flattened because verification
    is a substring check over this text.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""

    segments: list[tuple[str | None, str]] = []
    seen: set[str] = set()
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        text = _jats_text(abstract)
        text = re.sub(r"^abstract\s*", "", text, flags=re.IGNORECASE)
        if text:
            segments.append(("abstract", text))
            seen.add("abstract")
    for sec in body.findall("sec"):
        title_node = sec.find("title")
        title = _jats_text(title_node) if title_node is not None else ""
        if title and _FULLTEXT_SKIP_TITLES.search(title.lower()):
            continue
        name = (_canonical_section(title)
                or _JATS_SEC_TYPES.get((sec.get("sec-type") or "").lower()))
        if name == "references":
            continue
        if name and name not in seen:
            seen.add(name)
            text = _jats_body_text(sec, title_node)
        else:
            name = None
            text = _jats_text(sec)
        if text:
            segments.append((name, text))
    if not any(name for name, _ in segments):
        plain = "\n\n".join(body for _, body in segments)
        text = plain if plain else _jats_text(body)
    else:
        text = _assemble_sections(segments)
    # The paper's own bracketed citation numbers ("[ 1 ]", "[2, 5]", "[3-7]")
    # collide with the pipeline's [N] SOURCE-index scheme: a writer reading
    # "improves outcomes [ 4 ]" mid-paragraph may echo that number as if it
    # were one of OUR sources. Strip them; the prose keeps its meaning.
    return _strip_citation_brackets(text)


def _drop_references(text: str) -> str:
    """Cut the reference list from unmarked (or leftover) full text."""
    m = _REFERENCES_HEADING.search(text)
    if not m:
        return text
    return text[:m.start()].rstrip()


def _split_marked_sections(text: str) -> dict[str, str] | None:
    """Parse paperfetch-style ``## Results`` markers. None if none found."""
    matches = list(_SECTION_MARKER.finditer(text))
    if not matches:
        return None
    found: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if name not in found:
            found[name] = body
    return found


def _excerpt_from_full_text(text: str, budget: int) -> str:
    """Spend `budget` characters on the sections a briefing actually needs."""
    if not text or budget <= 0:
        return ""
    text = _drop_references(text)
    sections = _split_marked_sections(text)
    if sections is None:
        return text[:budget]
    parts: list[str] = []
    used = 0
    for group in _EXCERPT_SECTION_GROUPS:
        name = next((n for n in group if sections.get(n)), None)
        if name is None:
            continue
        body = sections[name]
        chunk = f"## {name.title()}\n\n{body}"
        extra = 2 if parts else 0
        room = budget - used - extra
        if room <= 0:
            break
        if len(chunk) > room:
            chunk = chunk[:room]
            if not chunk.strip():
                break
        parts.append(chunk)
        used += extra + len(chunk)
        if used >= budget:
            break
    return "\n\n".join(parts)


# Prompt policy for full text, and simultaneously the verification contract:
# the writer is shown exactly the excerpts this function yields, and verify.py
# checks statistics against exactly the same excerpts. Both sides derive them
# from this one deterministic function so nothing has to be recorded per run —
# and so verification never searches text the writer was not shown, which
# would let a remembered figure pass as grounded.
FULLTEXT_PER_PAPER_CHARS = 12000
FULLTEXT_TOTAL_CHARS = 60000


def full_text_excerpts(papers: list[Paper]) -> dict[int, str]:
    """{1-based index: the full-text excerpt shown to the writer}.

    When the text carries section markers (``## Results``), sections are
    picked in this order: abstract, results, discussion or conclusions,
    methods, introduction, until FULLTEXT_PER_PAPER_CHARS is spent.
    Unmarked text (older paperfetch, no headings) is still a head-slice.
    References never enter. Papers arrive ranked, so when the total budget
    runs out it is the least relevant full texts that are left as
    abstract-only.
    """
    out: dict[int, str] = {}
    used = 0
    for i, p in enumerate(papers, start=1):
        if not p.full_text:
            continue
        room = FULLTEXT_TOTAL_CHARS - used
        if room <= 0:
            break
        excerpt = _excerpt_from_full_text(
            p.full_text, min(FULLTEXT_PER_PAPER_CHARS, room))
        if not excerpt:
            continue
        out[i] = excerpt
        used += len(excerpt)
    return out


# Which study designs earn one of the five deep reads, best first. Not a quality
# score — nothing here appraises a study. It answers a narrower question: which
# paper repays 12,000 characters of reading? A systematic review carries the
# appraised evidence base, a trial carries the primary result, and a cross-
# sectional survey mostly restates its own abstract (#166).
DESIGN_ORDER = ("synthesis", "trial", "other")

DESIGN_DISPLAY_ORDER = (
    "synthesis",
    "trial",
    "observational",
    "qualitative",
    "case",
    "scoping",
    "narrative",
    "consensus",
    "other",
)
DESIGN_LABELS = {
    "synthesis": "Reviews",
    "trial": "Trials",
    "observational": "Observational",
    "qualitative": "Qualitative",
    "case": "Case reports",
    "scoping": "Scoping",
    "narrative": "Narrative",
    "consensus": "Consensus",
    "other": "Other",
}

_DESIGN_EXCLUDE_RE = re.compile(
    r"\b(?:study\s+protocol|trial\s+protocol|protocol\s+for\s+a|statistical\s+analysis\s+plan|rationale\s+and\s+design)\b"
    r"|:\s*a\s+protocol\b",
    re.IGNORECASE,
)
_SCOPING_RE = re.compile(r"\bscoping\s+reviews?\b", re.IGNORECASE)
_NARRATIVE_RE = re.compile(
    r"\b(?:narrative|literature|integrative|critical)\s+reviews?\b"
    r"|\breview\s+of\s+the\s+literature\b",
    re.IGNORECASE,
)
_CONSENSUS_RE = re.compile(
    r"\b(?:consensus\s+(?:statement|document|guidance|recommendations?|"
    r"development\s+conference)|expert\s+consensus|delphi)\b",
    re.IGNORECASE,
)
_CASE_RE = re.compile(r"\bcase[\s-]reports?\b|\bcase[\s-]series\b", re.IGNORECASE)

_SYNTHESIS_RE = re.compile(
    r"\b(?:systematic\s+(?:\w+\s+)?reviews?|meta-?analy\w+|umbrella\s+reviews?|evidence\s+synthesis|cochrane\s+reviews?)\b"
    r"|cochrane\s+database\s+of\s+systematic\s+reviews",
    re.IGNORECASE,
)
_TRIAL_RE = re.compile(
    r"\b(?:randomi[sz]\w*|rcts?|controlled\s+trials?|clinical\s+trials?|stepped[- ]wedge|cluster[- ]random\w*|pragmatic\s+trials?|feasibility\s+trials?)\b",
    re.IGNORECASE,
)
# Mixed-methods work is deliberately left unclassified — calling it qualitative
# would be a claim the metadata does not support.
_QUALITATIVE_RE = re.compile(
    r"\b(?:qualitative(?:\s+\w+)?\s+(?:study|studies|analysis|research|interviews?|evaluation)"
    r"|qualitative\s+study|focus\s+groups?|thematic\s+analysis|grounded\s+theory"
    r"|semi-?structured\s+interviews?|phenomenolog\w+|ethnograph\w+"
    r"|interpretative\s+phenomenological)\b",
    re.IGNORECASE,
)
_OBSERVATIONAL_RE = re.compile(
    r"\b(?:cohort\s+stud\w+|prospective\s+cohort|retrospective\s+cohort"
    r"|case[-\s]control|cross[-\s]sectional|longitudinal\s+stud\w+"
    r"|observational\s+stud\w+|registry\s+stud\w+"
    r"|population[-\s]based\s+stud\w+|national\s+survey)\b",
    re.IGNORECASE,
)


def classify_design(paper: Paper) -> str:
    """Classify a paper's study design into one of DESIGN_DISPLAY_ORDER.

    Reads title, venue, and `publication_types` metadata only — never the
    abstract (an abstract discussing what 'is needed' would misclassify).
    Preprints are untouched by design detection: a preprint of a trial still
    ranks as a trial.
    """
    text = f"{paper.title} {paper.venue}"
    types_text = " ".join(paper.publication_types)
    # Index vocabularies hyphenate ("systematic-review", "case-report"); match both
    # spellings rather than loosening every regex.
    combined = f"{text} {types_text} {types_text.replace('-', ' ')}"

    if _DESIGN_EXCLUDE_RE.search(combined):
        return "other"
    if _SYNTHESIS_RE.search(combined):
        return "synthesis"
    if _SCOPING_RE.search(combined):
        return "scoping"
    if _NARRATIVE_RE.search(combined):
        return "narrative"
    if _CONSENSUS_RE.search(combined):
        return "consensus"
    if _TRIAL_RE.search(combined):
        return "trial"
    if _CASE_RE.search(combined):
        return "case"
    if _QUALITATIVE_RE.search(combined):
        return "qualitative"
    if _OBSERVATIONAL_RE.search(combined):
        return "observational"
    # Bare 'review' publication types (OpenAlex 'type: review', Europe PMC 'Review')
    # are deliberately NOT mapped to narrative: OpenAlex tags systematic reviews that
    # way too, so mapping bare review to narrative would falsely label systematic
    # syntheses whose titles omit the words.
    return "other"


def paper_design(paper: Paper) -> str:
    """Fetch-ordering view of classify_design: synthesis / trial / other.

    Ordering-only: used by `full_text_order` to prioritise papers that repay
    a deep read (systematic reviews carry the appraised evidence base, trials
    carry primary results, whereas surveys and narrative reviews mostly restate
    their abstracts).

    The classification itself now lives in `classify_design`.
    """
    label = classify_design(paper)
    return label if label in DESIGN_ORDER else "other"


# Which eligible sources get a deep read, and in what order. The set is the
# same as it always was (direct and related, never tangential); only the order
# changed. Rank order put citation weight ahead of everything (#143), while
# recency-first inside direct stranded landmark older reviews and trials (#166).
# The sort key is now relevance tier -> design weight -> recency -> search rank.
FULLTEXT_RELEVANCE_ORDER = ("direct", "related")


def full_text_order(papers: list[Paper], relevance: dict[int, str]) -> list[int]:
    """1-based indices to attempt full text for, best candidate first.

    Direct before related; systematic reviews / meta-analyses first, then
    trials, then other designs (DESIGN_ORDER via paper_design); newest first
    inside a design tier; search rank breaks the remaining ties (#166,
    revising #143). A paper with no year sorts as if year 0 — an undated record
    is not evidence of being current. Tangential and unlabelled sources are
    absent from the result: they are never fetched, whether or not the target
    is met.
    """
    tier = {label: n for n, label in enumerate(FULLTEXT_RELEVANCE_ORDER)}
    design = {label: n for n, label in enumerate(DESIGN_ORDER)}
    ranked = []
    for index, paper in enumerate(papers, start=1):
        label = relevance.get(index)
        if label in tier:
            ranked.append((tier[label], design[paper_design(paper)],
                           -(paper.year or 0), index))
    return [index for _, _, _, index in sorted(ranked)]


def fetch_full_text(paper: Paper, use_cache: bool = True, log=lambda msg: None) -> str:
    """The paper's open-access full text as plain text, or "" when unavailable.

    When the `papers` CLI is available and the paper has a DOI, it is tried
    first. Otherwise falls back to Europe PMC via PMCID.
    Failures are cached briefly, like search refusals, so one dead fetch is
    not retried across a run.
    """
    now = time.time()
    doi = _normalize_doi(paper.doi)
    if doi and paperfetch.available(log):
        key = f"papers:{doi}"
        if use_cache and _CACHE_TTL > 0:
            with _cache_open():
                entry = _fulltext_cache.get(key)
                if entry and entry[0] > now:
                    cached_text = entry[1]
                    cached_status = entry[2] if len(entry) > 2 else ""
                    if cached_text:
                        paper.full_text_via = "papers"
                    elif cached_status in paperfetch.NOT_OA_STATUSES:
                        paper.full_text_not_oa = True
                    return cached_text
        text, status = paperfetch.fetch_via_papers_with_status(doi, log=log)
        text = _strip_citation_brackets(text)
        if status in paperfetch.NOT_OA_STATUSES:
            paper.full_text_not_oa = True
            paper.oa_tried = paperfetch.tried_for(doi)
        if _CACHE_TTL > 0:
            ttl = _CACHE_TTL if text else _CACHE_FAILURE_TTL
            with _cache_open():
                _fulltext_cache[key] = (now + ttl, text, status)
                _save_cache_locked()
        if text:
            paper.full_text_via = "papers"
            paper.full_text_not_oa = False
            return text

    if not (paper.pmcid and paper.is_open_access):
        return ""
    if use_cache and _CACHE_TTL > 0:
        with _cache_open():
            entry = _fulltext_cache.get(paper.pmcid)
            if entry and entry[0] > now:
                if entry[1]:
                    paper.full_text_via = "europe_pmc"
                    paper.full_text_not_oa = False
                return entry[1]
    try:
        resp = _get_with_retry(
            EUROPE_PMC_FULLTEXT_URL.format(pmcid=paper.pmcid), params={}, headers={})
        text = _parse_fulltext_xml(resp.text)
    except SearchFailure:
        text = ""
    if _CACHE_TTL > 0:
        ttl = _CACHE_TTL if text else _CACHE_FAILURE_TTL
        with _cache_open():
            _fulltext_cache[paper.pmcid] = (now + ttl, text, "")
            _save_cache_locked()
    if text:
        paper.full_text_via = "europe_pmc"
        paper.full_text_not_oa = False
    return text


def _cache_get(key: tuple[str, str, int]) -> tuple[list[Paper], str] | None:
    """The cached (papers, error) for this key, or None if absent or stale."""
    if _CACHE_TTL <= 0:
        return None
    now = time.time()
    with _cache_open():
        entry = _search_cache.get(key)
        if entry is None:
            return None
        expires_at, papers, error = entry
        if expires_at <= now:
            del _search_cache[key]
            return None
    # Hand back a copy of the list: callers collect into their own lists and a
    # shared one would accumulate across runs.
    return list(papers), error


def _cache_put(key: tuple[str, str, int], papers: list[Paper], error: str) -> None:
    if _CACHE_TTL <= 0:
        return
    ttl = _CACHE_FAILURE_TTL if error else _CACHE_TTL
    with _cache_open():
        _search_cache[key] = (time.time() + ttl, list(papers), error)
        if len(_search_cache) > _CACHE_MAX_ENTRIES:
            now = time.time()
            for stale in [k for k, (exp, _, _) in _search_cache.items() if exp <= now]:
                del _search_cache[stale]
            while len(_search_cache) > _CACHE_MAX_ENTRIES:
                soonest = min(_search_cache, key=lambda k: _search_cache[k][0])
                del _search_cache[soonest]
        _save_cache_locked()


def _search_once(name: str, search, query: str, limit: int) -> tuple[list[Paper], str, bool]:
    """One source, one query, live. Returns (papers, error, cached=False).

    Failures are returned rather than raised so the caller can record them
    alongside successes; a source failing is survivable as long as another
    answers. The result is written to the cache either way.
    """
    key = (name, query, limit)
    try:
        papers, error = search(query, limit=limit), ""
    except SearchFailure as exc:
        papers, error = [], str(exc)
    except Exception as exc:  # malformed payload, etc.
        papers, error = [], f"{type(exc).__name__}: {exc}"

    _cache_put(key, papers, error)
    return papers, error, False


def _merge_duplicate(kept: Paper, dup: Paper) -> None:
    """Fold a duplicate record's metadata into the copy already collected.

    The kept copy's *identity* is never swapped, only enriched. First-seen has
    to win: arXiv is queried last precisely so a preprint loses to the
    published version, and a preprint that happened to carry more metadata
    would otherwise take its place in the reference list.

    The preprint flag is never OR'd across a merge for the same reason: arXiv
    losing to a published version must not relabel that published version as
    a preprint. The flag is only updated if adopting the duplicate's identifier
    reveals the only identifier we now hold is a preprint one.

    Retraction is the opposite rule. Unlike `is_preprint`, `is_retracted` IS
    OR'd across a merge: a preprint and its published version are two
    documents and the flag distinguishes them, but a retraction is a fact
    about the one work both records describe, so whichever copy carries it,
    the kept record is retracted.

    The abstract is filled only when the kept copy has none. `verify.py`
    checks statistics against the abstract shown to the writer, so pulling a
    different source's wording in under a record identified by the first
    source's title and venue would quietly change what a figure is checked
    against. Every parser already drops records without an abstract, so this
    branch is a guard, not a common path.
    """
    if dup.pmcid and not kept.pmcid:
        kept.pmcid, kept.is_open_access = dup.pmcid, dup.is_open_access
    if dup.abstract and not kept.abstract:
        kept.abstract = dup.abstract
    if dup.citation_count > kept.citation_count:
        kept.citation_count = dup.citation_count
    if kept.year is None and dup.year is not None:
        kept.year = dup.year
    if dup.doi and not kept.doi:
        kept.doi = dup.doi
        # Adopting an identifier adopts what it says about the record: if the
        # only DOI we now hold is a preprint-server one, that is what the
        # reference link points at.
        kept.is_preprint = kept.is_preprint or _looks_like_preprint(kept.doi, "")
    if dup.authors and not kept.authors:
        kept.authors = dup.authors
    if dup.venue and not kept.venue:
        kept.venue = dup.venue
    if dup.url and not kept.url:
        kept.url = dup.url
        kept.is_preprint = kept.is_preprint or _looks_like_preprint("", kept.url)
    if dup.publication_types and not kept.publication_types:
        kept.publication_types = dup.publication_types
    kept.is_retracted = kept.is_retracted or dup.is_retracted
    if dup.correction_note and not kept.correction_note:
        kept.correction_note = dup.correction_note
    if dup.sample_size is not None and kept.sample_size is None:
        kept.sample_size = dup.sample_size
    if dup.registration and not kept.registration:
        kept.registration = dup.registration
    # OR'd across the merge, for the same reason `is_retracted` is: a grant is
    # a fact about the one work both records describe, and only the Europe PMC
    # copy can ever carry it.
    kept.has_funding_statement = kept.has_funding_statement or dup.has_funding_statement


# Constants for the named-source pass (issue #165, #190). Read by pipeline.py.
NAMED_SOURCE_SCAN = 3        # abstracts read for names
NAMED_SOURCE_LIMIT = 8       # lookups requested, and new records kept
NAMED_SOURCE_PER_QUERY = 5   # page size for a lookup; we want one exact record
NAMED_MATCH_RATE_MAX = 0.5   # share of returned records one name may match (#190)
NAMED_MATCH_MIN_MATCHES = 5  # below this, a high share means a small return (#190)

# Constants for the one-hop citation pass (issue #243). Read by pipeline.py.
REFERENCED_SEEDS = 3        # direct syntheses whose reference lists are read
REFERENCED_ID_LIMIT = 50    # ids resolved in one batched request (OpenAlex OR-filter cap)
REFERENCED_CITED_LIMIT = 8  # citing records fetched for the top direct trial


def openalex_work(paper: Paper) -> tuple[str, list[str]]:
    """OpenAlex work id and reference-list ids for one paper. ("", []) when unknown.

    One request. Looked up by DOI (`filter=doi:<doi>`), because a Paper carries
    no OpenAlex id. Raises SearchFailure if OpenAlex refuses — the caller
    decides what that means for the run.
    """
    doi = _normalize_doi(paper.doi)
    if not doi:
        return ("", [])
    params = {
        "filter": f"doi:{doi}",
        "select": "id,referenced_works",
        "per-page": 1,
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = _get_with_retry(OPENALEX_URL, params=params, headers={})
    results = resp.json().get("results") or []
    if not results:
        return ("", [])
    return (results[0].get("id") or "", list(results[0].get("referenced_works") or []))


def openalex_records(ids: list[str]) -> list[Paper]:
    """Papers for up to REFERENCED_ID_LIMIT OpenAlex work ids, in one request."""
    seen: set[str] = set()
    deduped: list[str] = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            deduped.append(i)
    deduped = deduped[:REFERENCED_ID_LIMIT]
    if not deduped:
        return []
    params = {
        "filter": f"openalex_id:{'|'.join(deduped)},has_abstract:true",
        "select": _OA_FIELDS,
        "per-page": len(deduped),
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = _get_with_retry(OPENALEX_URL, params=params, headers={})
    papers = []
    for item in resp.json().get("results") or []:
        paper = _oa_paper(item)
        if paper is not None:
            papers.append(paper)
    return papers


def openalex_citing_works(work_id: str, limit: int = REFERENCED_CITED_LIMIT) -> list[Paper]:
    """The most-cited papers that cite `work_id`. One request."""
    if not work_id:
        return []
    params = {
        "filter": f"cites:{work_id},has_abstract:true",
        "select": _OA_FIELDS,
        "per-page": limit,
        "sort": "cited_by_count:desc",
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    resp = _get_with_retry(OPENALEX_URL, params=params, headers={})
    papers = []
    for item in resp.json().get("results") or []:
        paper = _oa_paper(item)
        if paper is not None:
            papers.append(paper)
    return papers


# Stoplist of capitalised non-names and apparatus acronyms. The negative
# controls in test_named_references_reads_names_not_noise and
# test_generic_named_lookups_are_skipped are the specification.
_NAMED_STOPLIST = frozenset(
    "this the our a an recent previous current prior one two we "
    "rct prisma consort grade prospero who nice nhs nih usa uk covid pico medline embase cinahl "
    "background objective objectives aim aims method methods result results findings "
    "conclusion conclusions discussion introduction design setting participants outcomes "
    "limitations registration funding data evidence eligibility "
    "zero three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand "
    "several many most some few both all other another further additional remaining various "
    "multiple numerous each every no included eligible "
    "us eu ed er icu gp".split()
)

_GENERIC_NAME_SUFFIXES = frozenset(
    "language based related led level wide term arm site centre center "
    "country year month week day group controlled blind blinded specific "
    "adjusted matched only".split()
)

# Capitalised words that look like a study name and are not. Live Luna runs
# looked up "RCTs trial" and "International Clinical trial" from Cochrane
# abstracts; both are how reviews describe the literature, not a paper.
_GENERIC_NAME_WORDS = frozenset(
    "international clinical national european american australian canadian british "
    "randomised randomized controlled observational prospective retrospective "
    "systematic narrative scoping qualitative quantitative "
    "cochrane group groups collaboration register registry "
    "placebo".split()
)

_DOI_EXTRACT_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;)\]]+")

_STUDY_ADJS_RE = (
    r"(?:(?:cluster|randomi[sz]ed|controlled|stepped-wedge|multicentre|multicenter|"
    r"pragmatic|pilot|feasibility|open-label|double-blind|blinded|cross-?over|"
    r"longitudinal|prospective|retrospective|observational|community|phase\s+[IVX\d]+)\s+)*"
)
_STUDY_NOUNS_PATTERN = r"(?:trials?|stud(?:y|ies)|programmes?|programs?|cohorts?|rcts?|interventions?)"

_NAME_THEN_NOUN_RE = re.compile(
    rf"\b(?P<name>[A-Z][A-Za-z0-9*'-]*(?:\s+[A-Z][A-Za-z0-9*'-]*){{0,2}})\s+"
    rf"(?P<adjs>(?i:{_STUDY_ADJS_RE}))"
    rf"(?P<noun>(?i:{_STUDY_NOUNS_PATTERN}))\b"
)

_NOUN_THEN_ACRONYM_RE = re.compile(
    rf"\b(?i:{_STUDY_ADJS_RE})"
    rf"(?P<noun>(?i:{_STUDY_NOUNS_PATTERN}))\s*\(\s*(?P<acronym>[A-Za-z0-9*'-]+)\s*\)"
)

_STUDY_NOUN_CANONICAL = {
    "trial": "trial", "trials": "trial",
    "study": "study", "studies": "study",
    "programme": "programme", "programmes": "programme",
    "program": "program", "programs": "program",
    "cohort": "cohort", "cohorts": "cohort",
    "rct": "RCT", "rcts": "RCT",
    "intervention": "intervention", "interventions": "intervention",
}


def _is_sentence_initial(text: str, pos: int) -> bool:
    prefix = text[:pos].rstrip()
    if not prefix:
        return True
    return bool(re.search(r'(?:[\.\!\?]\s*[\"\')\]]*|[\:\n]\s*)$', prefix))


def _name_token_key(token: str) -> str:
    """Lowercased token with a trailing possessive or plural-s stripped when known."""
    t = token.strip(".,;:\"'()").lower()
    t = re.sub(r"'s$", "", t)
    known = _NAMED_STOPLIST | _GENERIC_NAME_WORDS
    if t.endswith("s") and t[:-1] in known:
        return t[:-1]
    return t


def _is_generic_name_token(token: str) -> bool:
    t = token.strip(".,;:\"'()")
    if not t:
        return True
    key = _name_token_key(t)
    if key in _NAMED_STOPLIST or key in _GENERIC_NAME_WORDS:
        return True
    if "-" in t:
        tail = t.split("-")[-1].strip(".,;:\"'()").lower()
        if tail in _GENERIC_NAME_SUFFIXES:
            return True
    if any(c.isalpha() for c in t) and all(c.isupper() for c in t if c.isalpha()) and len(t) < 3:
        return True
    return False


def _is_valid_name_token(token: str, is_sentence_initial: bool) -> bool:
    t = token.strip(".,;:\"'()")
    if not t:
        return False
    if _is_generic_name_token(t):
        return False
    # ALL-CAPS (>= 3 chars, at least one letter, all alpha characters are uppercase)
    if len(t) >= 3 and any(c.isalpha() for c in t) and all(c.isupper() for c in t if c.isalpha()):
        return True
    # Capitalised word (not sentence-initial, starts with capital letter)
    if t[0].isupper() and not is_sentence_initial:
        return True
    return False


def named_references(text: str) -> list[str]:
    """Extract DOIs and study names mentioned in an abstract.

    Deterministic, ordered, deduped, and capped at NAMED_SOURCE_LIMIT.
    DOIs come first because they are exact; named studies follow.
    """
    dois: list[str] = []
    for m in _DOI_EXTRACT_RE.finditer(text):
        raw = m.group(0).rstrip(".,;:)\"'")
        doi = _normalize_doi(raw)
        if doi and doi not in dois:
            dois.append(doi)

    names: list[str] = []
    # Pattern 1: name-then-noun (e.g. "Safewards trial", "RAISE-ETP study")
    for m in _NAME_THEN_NOUN_RE.finditer(text):
        raw_name = m.group("name").strip()
        tokens = raw_name.split()
        match_start = m.start("name")
        while tokens:
            token_offset = text[match_start:].find(tokens[0])
            token_pos = match_start + (token_offset if token_offset >= 0 else 0)
            si = _is_sentence_initial(text, token_pos)
            if _is_valid_name_token(tokens[0], si):
                break
            match_start = token_pos + len(tokens[0])
            tokens.pop(0)

        if not tokens:
            continue
        if any(not _is_valid_name_token(tok, False) for tok in tokens[1:]):
            continue
        clean_name = " ".join(tokens)
        noun_key = m.group("noun").lower()
        canon_noun = _STUDY_NOUN_CANONICAL.get(noun_key, noun_key)
        q = f"{clean_name} {canon_noun}"
        if q not in names and q not in dois:
            names.append(q)

    # Pattern 2: noun-then-parenthesised acronym (e.g. "... trial (SAFEWARDS)")
    for m in _NOUN_THEN_ACRONYM_RE.finditer(text):
        acronym = m.group("acronym").strip()
        if not _is_valid_name_token(acronym, False):
            continue
        noun_key = m.group("noun").lower()
        canon_noun = _STUDY_NOUN_CANONICAL.get(noun_key, noun_key)
        q = f"{acronym} {canon_noun}"
        if q not in names and q not in dois:
            names.append(q)

    return (dois + names)[:NAMED_SOURCE_LIMIT]


_STUDY_NOUNS_RE = re.compile(
    r"\s+(?:trials?|stud(?:y|ies)|programmes?|programs?|cohorts?|rcts?|interventions?)$",
    re.IGNORECASE,
)


def named_matches(paper: Paper, request: str) -> bool:
    """Acceptance rule: True when paper matches what the request asked for.

    A DOI request must match the paper's normalised DOI. A study name request
    must have its name portion appear in the paper's normalised title.
    """
    req_doi = _normalize_doi(request)
    if req_doi:
        return bool(paper.doi and _normalize_doi(paper.doi) == req_doi)

    name_part = _STUDY_NOUNS_RE.sub("", request).strip() or request
    norm_name = _normalize_title(name_part)
    if not norm_name:
        return False
    norm_title = _normalize_title(paper.title)
    return norm_name in norm_title


def filter_named_matches(records: list[Paper], requests: list[str]) -> tuple[list[Paper], list[str]]:
    """Keep the records a request genuinely asked for; drop requests that behave generically.

    Returns (kept records, requests dropped). A DOI request is never dropped —
    it identifies exactly one work. A name request is dropped when it matches
    at least NAMED_MATCH_MIN_MATCHES records AND more than NAMED_MATCH_RATE_MAX
    of everything returned: a name that matches most of the pool is describing
    the literature, not naming a paper.
    """
    if not records or not requests:
        return ([], [])

    dropped_requests: list[str] = []
    surviving_requests: list[str] = []
    total_records = len(records)

    for req in requests:
        if _normalize_doi(req):
            surviving_requests.append(req)
            continue
        matches = sum(1 for p in records if named_matches(p, req))
        if matches >= NAMED_MATCH_MIN_MATCHES and (matches / total_records) > NAMED_MATCH_RATE_MAX:
            dropped_requests.append(req)
        else:
            surviving_requests.append(req)

    kept_records = [
        p for p in records
        if any(named_matches(p, req) for req in surviving_requests)
    ]
    return (kept_records, dropped_requests)


def merge_candidates(pool: list[Paper], extra: list[Paper], limit: int = NAMED_SOURCE_LIMIT) -> list[Paper]:
    """Merge extra papers into pool using DOI/title dedupe, appending up to `limit` new records.

    Existing records in `pool` are preserved in order (never re-sorted) and enriched via
    `_merge_duplicate` if an extra paper matches an existing DOI or title.
    Returns the list of newly appended Paper objects.
    """
    by_title: dict[str, Paper] = {}
    by_doi: dict[str, Paper] = {}
    for p in pool:
        t_key = _normalize_title(p.title)
        d_key = _normalize_doi(p.doi)
        if t_key and t_key not in by_title:
            by_title[t_key] = p
        if d_key and d_key not in by_doi:
            by_doi[d_key] = p

    new_papers: list[Paper] = []
    for paper in extra:
        title_key = _normalize_title(paper.title)
        doi_key = _normalize_doi(paper.doi)
        if not title_key:
            continue
        kept = by_doi.get(doi_key) if doi_key else None
        if kept is None:
            kept = by_title.get(title_key)
        if kept is not None:
            _merge_duplicate(kept, paper)
            by_title.setdefault(title_key, kept)
            if doi_key:
                by_doi.setdefault(doi_key, kept)
            continue
        if len(new_papers) < limit:
            by_title[title_key] = paper
            if doi_key:
                by_doi[doi_key] = paper
            pool.append(paper)
            new_papers.append(paper)

    return new_papers


# The pool the relevance gate gets to work on. At 20 it barely worked: three
# measured runs collected exactly 20 candidates and cited 16-19 of them, and a
# landmark cluster RCT named by a run's own planned query never made the pool
# (#141). Curation cost scales with this number and that is the accepted price
# — the alternative, truncating the abstracts sent to curation, was measured in
# #117 and destabilises the gate, so CURATION_ABSTRACT_CHARS stays None.
DEFAULT_MAX_PAPERS = 40


def gather_evidence(
    queries: list[str],
    max_papers: int = DEFAULT_MAX_PAPERS,
    # Records without a retrievable abstract are discarded, and the yield is
    # poor: a real run of three queries against both APIs returned 10 usable
    # papers, which is thin for a review and leaves each section restating the
    # same few findings. Asking for more candidates costs one larger page per
    # query, not more requests, and the ranking still keeps only max_papers.
    per_query: int = 25,
    topic: str = "",
    core_entity: str = "",
    log=lambda msg: None,
    outcomes: list[dict] | None = None,
    use_cache: bool = True,
    patient: bool = True,
    exhausted: set[str] | None = None,
    stop_when_any: bool = False,
) -> list[Paper]:
    """Run every query against every source, dedupe, and return the best candidates,
    ranked by a blend of topic relevance, citations, and recency.

    One source failing is survivable — another may still answer — so failures
    are recorded rather than raised. `outcomes` collects them so the caller can
    tell "this topic has no literature" from "every API refused us", which look
    identical from the returned list alone. Each outcome carries `cached`, so a
    caller can see whether a result reflects the sources' behaviour just now.

    `use_cache=False` forces a live probe of every source. Only the diagnostic
    endpoint wants this — it exists to answer "can this host reach its sources
    *right now*", which a cached answer cannot. Drafting always leaves it on.

    `patient=False` starts the run with the patient retry round already spent
    (used by pre-flight probes to fail fast without adding 30s).

    `stop_when_any=True` returns as soon as one source has produced papers.
    Pre-flight only needs to know someone is home; waiting out arXiv after
    OpenAlex already answered is how a hosted draft burns the proxy window
    before the first paid call.
    """
    global _recency_query_refused, _s2_patient_round_spent
    _recency_query_refused = False
    # patient=False starts the run with the round already spent. The pre-flight
    # probe uses it: it exists to fail FAST before the caller is billed, so it
    # must never add 30s to a run that is about to work.
    _s2_patient_round_spent = not patient
    by_title: dict[str, Paper] = {}
    by_doi: dict[str, Paper] = {}
    collected: list[Paper] = []
    # A source that has already refused once in this run will almost certainly
    # refuse again — the limits are per-minute and the run takes seconds. Each
    # attempt costs three tries with backoff, about ten seconds, so retrying a
    # dead source across every query wasted a third of the gather time.
    # Semantic Scholar's keyless tier currently 429s on every call.
    exhausted = set() if exhausted is None else exhausted

    for query in queries:
        # Keys are explicit rather than derived from `search.__name__`: the name
        # is not a stable identity (a replaced function reports whatever it is
        # called, and two lambdas are both `<lambda>`), and these keys index
        # DATABASE_NAMES and the outcome records. The tuple is rebuilt each call
        # so the module-level names are looked up fresh.
        #
        # arXiv goes last, and the order is load-bearing: dedupe is first-seen-
        # wins on the DOI, falling back to the normalised title, and a preprint
        # usually shares its title with the published version. Querying arXiv
        # last means the peer-reviewed record is the one kept, with the
        # preprint discarded as its duplicate — the wrong way round would cite
        # preprints for work that has since appeared in a journal.
        for name, search in (("semantic_scholar", search_semantic_scholar),
                             ("openalex", search_openalex),
                             ("europe_pmc", search_europe_pmc),
                             ("arxiv", search_arxiv)):
            # The cache is consulted before the exhausted set, not after: a hit
            # costs nothing and carries no risk of deepening a throttle, so a
            # source that has already refused this run can still contribute
            # results an earlier run stored for a different query.
            hit = _cache_get((name, query, per_query)) if use_cache else None
            if hit is None and name in exhausted:
                if outcomes is not None:
                    outcomes.append({"source": name, "query": query, "count": 0,
                                     "error": "skipped (already failed this run)",
                                     "cached": False})
                log(f"  {name}({query!r}) -> skipped, already failed this run")
                continue

            if hit is not None:
                results, error, cached = hit[0], hit[1], True
            else:
                results, error, cached = _search_once(name, search, query, per_query)
                if error:
                    exhausted.add(name)

            if outcomes is not None:
                outcomes.append({
                    "source": name, "query": query,
                    "count": len(results), "error": error, "cached": cached,
                })
            suffix = " (cached)" if cached else ""
            log(f"  {name}({query!r}) -> "
                + (f"{len(results)} papers{suffix}" if not error
                   else f"FAILED ({error}){suffix}"))

            for paper in results:
                # DOI first, title second. A DOI is a stable identifier; a
                # title is text, and any wording difference between two
                # sources' copies of one record used to produce two candidates
                # — two slots in the capped pool and two entries in a "N
                # sources cited" count that should have said one (#139).
                title_key = _normalize_title(paper.title)
                doi_key = _normalize_doi(paper.doi)
                if not title_key:
                    continue
                kept = by_doi.get(doi_key) if doi_key else None
                if kept is None:
                    kept = by_title.get(title_key)
                if kept is not None:
                    _merge_duplicate(kept, paper)
                    # Register the duplicate's own keys against the kept copy,
                    # so a third record matching either spelling merges too.
                    by_title.setdefault(title_key, kept)
                    if doi_key:
                        by_doi.setdefault(doi_key, kept)
                    continue
                by_title[title_key] = paper
                if doi_key:
                    by_doi[doi_key] = paper
                collected.append(paper)

            if stop_when_any and collected:
                break
        else:
            continue
        break

    # Build a keyword set from the topic + core entity for a relevance signal.
    raw = f"{topic} {core_entity}".lower()
    terms = {w for w in re.split(r"[^a-z0-9]+", raw) if len(w) > 3}
    collected.sort(key=lambda p: _rank_score(p, terms), reverse=True)
    return collected[:max_papers]

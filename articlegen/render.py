"""Render the structured article + its cited sources as a journal-style page.

The layout follows the conventions of a scientific-journal Review article —
article-type label, unstructured abstract, Key points box, numbered display
items (Box 1 / Fig. 1 / Table 1), superscript Vancouver citations, a Methods
statement of the search strategy, and standard back matter. `docs/journal-style.md`
records where each convention comes from and why we copy it.

Three of the display items are built deterministically from the fetched papers
and the relevance curation, so they cannot be hallucinated.

Also builds the Markdown export and the drafts/ review-queue index page.
"""

from __future__ import annotations

import datetime
import glob
import html
import os
import re

from . import sources
from .briefing import (
    CERTAINTY_LABELS, finding_parts, open_these_items, overall_certainty, split_verdict,
)
from .sources import Paper
from .style import SUBSTANCE_RULES
from .writer import is_briefing

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# The same marker with any space in front of it, so dropping an ungrounded
# citation does not leave that space stranded before the punctuation.
_CITATION_WITH_LEAD_RE = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")
# A citation marker sitting *before* the sentence punctuation, e.g. "…rest [1]."
_MARKER_BEFORE_PUNCT_RE = re.compile(r"[ \t]*(\[\d+(?:\s*,\s*\d+)*\])[ \t]*([.,;:?!])")
# A marker floated off the word it belongs to — the superscript should hug it.
_SPACE_BEFORE_MARKER_RE = re.compile(r"[ \t]+(\[\d+(?:\s*,\s*\d+)*\])")
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)

RELEVANCE_LABELS = {"direct": "Direct", "related": "Related", "tangential": "Background"}


def _au_date(d: datetime.date | None = None) -> str:
    """Day month year, no leading zero: '24 August 2026'."""
    return (d or datetime.date.today()).strftime("%d %B %Y").lstrip("0")


def _au_datetime(dt: datetime.datetime) -> str:
    """Day month year with 24-hour time: '24 Aug 2026 18:02'."""
    return dt.strftime("%d %b %Y %H:%M").lstrip("0")


# --------------------------------------------------------------------------
# citation plumbing
# --------------------------------------------------------------------------

def _citation_map(article: dict, papers: list[Paper]) -> tuple[list[Paper], dict[int, int]]:
    """Resolve `references` (raw SOURCE indices, in first-citation order) into the
    ordered list of cited papers plus a map {raw SOURCE index -> display number 1..N}.

    The writer cites papers by their fed "SOURCE N" label; we renumber those to a
    clean 1..N sequence so the inline markers match the reference list exactly.
    """
    ordered: list[int] = []
    for raw in article.get("references", []):
        if isinstance(raw, int) and 1 <= raw <= len(papers) and raw not in ordered:
            ordered.append(raw)
    cited = [papers[raw - 1] for raw in ordered]
    mapping = {raw: position for position, raw in enumerate(ordered, start=1)}
    return cited, mapping


def _remap_citations(text: str, mapping: dict[int, int]) -> str:
    """Rewrite raw SOURCE-index markers to display numbers; drop any with no source.

    A dropped marker takes the space in front of it with it. The model does cite
    source numbers it was never given — the marker is then removed, and leaving
    the space behind stranded the sentence's full stop: "…the market ." appeared
    in a shipped article. The leading whitespace is captured rather than matched
    globally so a kept marker's spacing is untouched.
    """

    def replace(match: re.Match) -> str:
        lead, body = match.group(1), match.group(2)
        mapped = [str(mapping[int(n)]) for n in body.split(",")
                  if int(n.strip()) in mapping]
        return f"{lead}[{', '.join(mapped)}]" if mapped else ""

    return _CITATION_WITH_LEAD_RE.sub(replace, text)


def _cited_papers(article: dict, papers: list[Paper]) -> list[Paper]:
    return _citation_map(article, papers)[0]


def _shift_markers_after_punctuation(text: str) -> str:
    """Move `[1].` to `.[1]` — journals set the superscript after the punctuation —
    and close up any space left between a marker and the word it attaches to."""
    return _SPACE_BEFORE_MARKER_RE.sub(r"\1", _MARKER_BEFORE_PUNCT_RE.sub(r"\2\1", text))


def _citation_runs(numbers: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted number list into (start, end) runs; runs of 3+ become ranges."""
    runs: list[tuple[int, int]] = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def _marker_parts(numbers: list[int], render_number) -> list[str]:
    """Format sorted citation numbers as runs — 1,3–5 — via `render_number(n)`."""
    parts = []
    for start, end in _citation_runs(numbers):
        if end - start >= 2:  # only a run of 3+ earns a range
            parts.append(f"{render_number(start)}–{render_number(end)}")
        elif end != start:
            parts.append(f"{render_number(start)},{render_number(end)}")
        else:
            parts.append(render_number(start))
    return parts


def _link_citations(escaped_text: str, valid_numbers: set[int]) -> str:
    """Turn [1] / [2, 3] markers into Nature-style superscript runs: ^1,3–5."""

    def replace(match: re.Match) -> str:
        raw = [n.strip() for n in match.group(1).split(",")]
        if not all(n.isdigit() and int(n) in valid_numbers for n in raw):
            return match.group(0)
        numbers = sorted({int(n) for n in raw})
        parts = _marker_parts(numbers, lambda n: f'<a href="#ref-{n}">{n}</a>')
        return f'<sup class="cite">{",".join(parts)}</sup>'

    return _CITATION_RE.sub(replace, escaped_text)


def _plain_citations(text: str) -> str:
    """Same run-collapsing for Markdown, where superscripts aren't available: [1,3–5]."""

    def replace(match: re.Match) -> str:
        raw = [n.strip() for n in match.group(1).split(",")]
        if not all(n.isdigit() for n in raw):
            return match.group(0)
        numbers = sorted({int(n) for n in raw})
        return "[" + ",".join(_marker_parts(numbers, str)) + "]"

    return _CITATION_RE.sub(replace, text)


# Figure flags. `verify` knows which numbers it could not ground, but that used
# to reach the reader only as a clause in the Limitations paragraph, thousands
# of words below the number it retracts and with no way to tell which figure it
# meant. Meanwhile the number carried a superscript citation — the strongest
# "this came from that paper" signal on the page. Someone copies the Key points
# into an email; the numbers travel and every qualification stays behind (#92).
#
# So the flag travels with the figure. A dagger is the journal convention for a
# footnote mark and needs no legend to read as "there is a caveat here".
UNVERIFIED_MARK = "†"
MISATTRIBUTED_MARK = "‡"

_MARK_TITLES = {
    UNVERIFIED_MARK: "This figure could not be located in the sources cited for it.",
    MISATTRIBUTED_MARK: "This figure was found only in a source other than the one cited.",
}


def _figure_flags(verification: dict | None) -> dict[str, str]:
    """{figure string -> mark}. Unverified wins if a figure is somehow in both."""
    v = verification or {}
    flags = {f: MISATTRIBUTED_MARK for f in (v.get("misattributed") or [])}
    flags.update({f: UNVERIFIED_MARK for f in (v.get("unverified") or [])})
    return {f: m for f, m in flags.items() if f}


def _flag_pattern(flags: dict[str, str]) -> re.Pattern | None:
    """Match a flagged figure, but never one inside a citation marker.

    The alternation is longest-first so "7,000 lux" is tried before "7,000",
    and the boundaries stop "12%" matching inside "112%" or "0.66" inside
    "0.665". The `cite` branch exists because at the point this runs the
    citation markers are still literal `[12]`, and a flagged figure of "12"
    would otherwise be found inside one.
    """
    if not flags:
        return None
    alts = "|".join(re.escape(f) for f in sorted(flags, key=len, reverse=True))
    return re.compile(
        r"(?P<cite>\[[\d,\s]+\])"
        r"|(?P<fig>(?<![\w.,])(?:" + alts + r")(?![\d,]))"
    )


def _apply_flags(text: str, flags: dict[str, str], render_mark) -> str:
    """Append each flagged figure's mark where the figure occurs."""
    pattern = _flag_pattern(flags)
    if pattern is None:
        return text

    def replace(match: re.Match) -> str:
        if match.group("cite"):
            return match.group(0)
        found = match.group("fig")
        return found + render_mark(flags[found])

    return pattern.sub(replace, text)


def _mark_html(escaped: str, flags: dict[str, str]) -> str:
    return _apply_flags(
        escaped,
        {html.escape(f): m for f, m in flags.items()},
        lambda mark: f'<sup class="flag" title="{_MARK_TITLES[mark]}">'
                     f'<a href="#limitations">{mark}</a></sup>',
    )


def _mark_plain(text: str, flags: dict[str, str]) -> str:
    return _apply_flags(text, flags, lambda mark: mark)


def _prose(text: str, cite_map: dict[int, int], valid_numbers: set[int],
           flags: dict[str, str] | None = None) -> str:
    """Full text pipeline: renumber -> punctuation-first -> escape -> flag -> superscripts.

    Flagging runs *after* escaping (so the inserted markup survives) and
    *before* citation linking (so the only brackets in the string are still the
    literal `[N]` markers `_flag_pattern` knows to skip).
    """
    remapped = _shift_markers_after_punctuation(_remap_citations(text, cite_map))
    escaped = _mark_html(html.escape(remapped), flags or {})
    return _link_citations(escaped, valid_numbers)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

_CLINICAL_HINTS = (
    "clinician", "patient", "clinical", "therapy", "treatment", "diagnos",
    "schizophreni", "depress", "disorder", "disease", "dose", "drug",
    "symptom", "psychiat", "medic", "health",
)


def _summary_text(article: dict) -> str:
    """The article's lead paragraph — `answer` on a briefing, `standfirst` on legacy drafts."""
    return (
        article.get("answer")
        or article.get("abstract")
        or article.get("standfirst")
        or ""
    )


def _key_points(article: dict) -> list[str]:
    return article.get("key_points") or article.get("key_takeaways") or []


def _is_clinical(topic: str, article: dict) -> bool:
    blob = (topic + " " + article.get("title", "") + " " + _summary_text(article)).lower()
    return any(h in blob for h in _CLINICAL_HINTS)


def _year_range(papers: list[Paper]) -> str:
    years = [p.year for p in papers if p.year]
    if not years:
        return ""
    lo, hi = min(years), max(years)
    return str(lo) if lo == hi else f"{lo}–{hi}"


# Surname particles stay attached to the surname: "Hans Van Dongen" -> "Van Dongen, H."
_NAME_PARTICLES = {
    "van", "von", "de", "del", "della", "der", "den", "di", "da", "dos", "du",
    "la", "le", "los", "ten", "ter", "bin", "ibn",
}

# Words that mark an author as an organisation, not a person — "GBD 2019
# Collaborators" must not become "Collaborators, G." (issue #62). Consortium
# authors are common in exactly the literature this reads: guideline bodies,
# collaborations, working groups.
_CORPORATE_WORDS = {
    "collaborators", "collaboration", "group", "committee", "consortium",
    "network", "initiative", "trialists", "investigators", "society",
    "association", "organization", "organisation",
}


def _is_corporate_author(name: str) -> bool:
    """A digit anywhere, or any organisational word, marks a non-personal name."""
    if any(ch.isdigit() for ch in name):
        return True
    tokens = {t.strip(".,()").lower() for t in name.split()}
    return bool(tokens & _CORPORATE_WORDS)


def _format_author(name: str) -> str:
    """`Susanne Diekelmann` -> `Diekelmann, S.` (journal reference form)."""
    name = " ".join(name.split())
    if not name:
        return ""
    # An organisational name passes through whole: inventing initials from it is
    # always wrong, and passing it unchanged is always safe.
    if _is_corporate_author(name):
        return name
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0]
    split = len(parts) - 1
    while split > 1 and parts[split - 1].lower().strip(".") in _NAME_PARTICLES:
        split -= 1
    surname = " ".join(parts[split:])
    initials = " ".join(f"{p[0].upper()}." for p in parts[:split] if p[:1].isalpha())
    return f"{surname}, {initials}" if initials else surname


def _reference_authors(paper: Paper) -> str:
    """Vancouver/Nature author line: up to three names joined with `&`, then et al."""
    authors = [a.strip() for a in paper.authors if a and a.strip()]
    truncated = bool(authors) and authors[-1].lower().rstrip(".") == "et al"
    if truncated:
        authors = authors[:-1]
    if not authors:
        return "Unknown authors."
    formatted = [_format_author(a) for a in authors]
    if truncated or len(formatted) > 3:
        line = f"{formatted[0]} et al."
    elif len(formatted) == 1:
        line = formatted[0]
    else:
        line = ", ".join(formatted[:-1]) + " & " + formatted[-1]
    # The author line always closes with a stop, so a mononym or a corporate name
    # doesn't run straight into the title.
    return line if line.endswith(".") else line + "."


def _short_author(paper: Paper) -> str:
    """`Diekelmann & Born` / `Xie et al.` — for the table and box citation lines."""
    authors = [a.strip() for a in paper.authors if a and a.strip()]
    truncated = bool(authors) and authors[-1].lower().rstrip(".") == "et al"
    if truncated:
        authors = authors[:-1]
    if not authors:
        return "Unknown authors"
    surnames = [_format_author(a).split(",")[0] for a in authors]
    if truncated or len(surnames) > 2:
        return f"{surnames[0]} et al."
    return " & ".join(surnames)


def _titled(title: str) -> str:
    """Reference titles end in a full stop, per Nature's reference style."""
    title = title.strip()
    return title if title.endswith((".", "?", "!")) else title + "."


def _searched_phrase(full_text_read: bool) -> str:
    """Name what the statistical check actually searched.

    `verify` searches the abstracts *plus* whatever full-text excerpts the
    writer was shown. Saying "the abstracts" on a draft that read full texts
    describes a weaker check than the one that ran (issue #101), so this is
    derived from the draft like every other provenance statement.
    """
    if full_text_read:
        return (
            "the material read for the cited sources — their abstracts, and the "
            "open-access full text where one was retrieved"
        )
    return "the abstracts of the cited sources"


def _unverified_sentence(unverified: list[str], full_text_read: bool = False) -> str:
    """The 'figures we could not find in the sources' limitation, in journal register."""
    shown = ", ".join(unverified[:6])
    more = "" if len(unverified) <= 6 else f" and {len(unverified) - 6} other(s)"
    where = _searched_phrase(full_text_read)
    if len(unverified) == 1:
        return (
            f"One numerical value reported above ({shown}), marked {UNVERIFIED_MARK} where "
            f"it appears, could not be located in {where}. It may originate in a part of a "
            "source that was not read, or may be unreliable, and should be verified before "
            "being quoted."
        )
    return (
        f"{len(unverified)} numerical values reported above ({shown}{more}), each marked "
        f"{UNVERIFIED_MARK} where it appears, could not be located in {where}. They may "
        "originate in a part of a source that was not read, or may be unreliable, and "
        "should be verified before being quoted."
    )


def _misattributed_sentence(misattributed: list[str]) -> str:
    """Figures that are real but credited to the wrong source.

    A separate limitation from the unverified ones because it is a different
    defect: the number is in the evidence base, but not in the source the
    sentence cites for it. Source numbers are deliberately not named — `verify`
    works in SOURCE indices while this section prints display numbers, and a
    wrong number here would be worse than none.
    """
    shown = ", ".join(misattributed[:6])
    more = "" if len(misattributed) <= 6 else f" and {len(misattributed) - 6} other(s)"
    if len(misattributed) == 1:
        return (
            f"One numerical value reported above ({shown}), marked {MISATTRIBUTED_MARK} "
            "where it appears, is found in a cited source other than the one its sentence "
            "credits. The figure is present in the evidence base, but the attribution does "
            "not hold and should be checked before it is quoted."
        )
    return (
        f"{len(misattributed)} numerical values reported above ({shown}{more}), each marked "
        f"{MISATTRIBUTED_MARK} where it appears, are found in cited sources other than the "
        "ones their sentences credit. The figures are present in the evidence base, but the "
        "attributions do not hold and should be checked before they are quoted."
    )


def _grounding_note(verification: dict | None, full_text_read: bool) -> str:
    """One line under the Key points, so no quotable unit travels uncaveated.

    The Key points are the part that gets copied into an email, and until this
    existed the numbers travelled while every qualification stayed behind in a
    Limitations paragraph ninety lines down (#92).

    Derived, never assumed, like every other provenance statement: no
    verification means no claim that anything was checked, and an article with
    no figures in it gets no line at all.
    """
    v = verification or {}
    if not v or not v.get("total"):
        return ""
    n_unverified = len(v.get("unverified") or [])
    n_misattributed = len(v.get("misattributed") or [])
    where = _searched_phrase(full_text_read)
    if not n_unverified and not n_misattributed:
        return f"Every figure in this review was located in {where}."
    clauses = []
    if n_unverified:
        clauses.append(f"{UNVERIFIED_MARK} marks a figure that could not be found there")
    if n_misattributed:
        clauses.append(
            f"{MISATTRIBUTED_MARK} marks one found only in a source other than the one cited")
    return (f"Every figure in this review was checked against {where}. "
            + "; ".join(clauses) + ". See Limitations.")


def _display_relevance(cite_map: dict[int, int], curation: dict | None) -> dict[int, str]:
    """{display number -> relevance label} for the cited papers only."""
    relevance = (curation or {}).get("relevance") or {}
    return {
        display: relevance[raw]
        for raw, display in cite_map.items()
        if relevance.get(raw) in RELEVANCE_LABELS
    }


def _relevance_counts(cite_map: dict[int, int], curation: dict | None) -> dict:
    """Tally direct/related/tangential over the CITED sources only (not all fetched)."""
    labels = _display_relevance(cite_map, curation)
    if not labels:
        return {}
    return {level: sum(1 for v in labels.values() if v == level)
            for level in ("direct", "related", "tangential")}


def _source_link_html(paper: Paper) -> str:
    """A ` · doi:10.1038/nrn2762` (or ` · Source`) trailer for a citation line."""
    if not paper.link:
        return ""
    label = f"doi:{paper.doi.removeprefix('https://doi.org/')}" if paper.doi else "Source"
    return f' · <a href="{html.escape(paper.link, quote=True)}">{html.escape(label)}</a>'


# --------------------------------------------------------------------------
# display items
# --------------------------------------------------------------------------

# --- display-item content -------------------------------------------------
#
# The three display items are the part a model cannot fabricate, so each is
# built from the fetched records. Both renderers need the same *selection* —
# which paper, which fields, which relevance label, which year buckets — and
# differ only in markup. These functions own the selection; the `_html` and
# `_markdown` pairs below own the formatting and nothing else (issue #46).


def _box_parts(article: dict, papers: list[Paper], cite_map: dict[int, int]) -> dict | None:
    """The featured study's content for Box 1, or None if the index is unusable."""
    fs = article.get("featured_study") or {}
    idx = fs.get("source_index")
    if not (isinstance(idx, int) and 1 <= idx <= len(papers)):
        return None
    paper = papers[idx - 1]
    return {
        "paper": paper,
        "reference": cite_map.get(idx),
        # `why` (the old schema's editorial justification) is deliberately not
        # read: the box states what the study did and found, and lets that
        # speak. `limitations` completes the method -> results -> caveat arc.
        "limitations": fs.get("limitations") or "",
        "method": fs.get("method") or "",
        "results": fs.get("results") or "",
    }


def _table_rows(cited: list[Paper], labels: dict[int, str]) -> list[dict]:
    """One row per cited source: the values Table 1 prints, before markup."""
    rows = []
    for n, paper in enumerate(cited, start=1):
        label = labels.get(n)
        # A preprint with no journal shows a blank Source cell, which was the
        # only tell the reader got (#144). Say it instead.
        venue = paper.venue or "—"
        if paper.is_preprint:
            venue = f"{paper.venue} (preprint)" if paper.venue else "Preprint"
        # Should never fire: a retracted record is omitted from what the writer
        # sees. But `_citation_map` accepts any index in range, so a model that
        # invents a source number can still land on one — and a cited retraction
        # printed silently is the failure this issue is about (#242).
        if paper.is_retracted:
            venue = f"Retracted — {paper.venue}" if paper.venue else "Retracted"
        d_label = sources.classify_design(paper)
        design_str = sources.DESIGN_LABELS[d_label]
        rows.append({
            "n": n,
            "paper": paper,
            "study": _short_author(paper),
            "year": paper.year or "n.d.",
            "venue": venue,
            "preprint": bool(paper.is_preprint),
            "retracted": bool(paper.is_retracted),
            "design": design_str,
            "sample": f"{paper.sample_size:,}" if paper.sample_size else "—",
            "registered": paper.registration or "—",
            "funding": "Grant listed" if paper.has_funding_statement else "—",
            "label": label if label in RELEVANCE_LABELS else "",
            "relevance": RELEVANCE_LABELS.get(label, "—") if label else "—",
            "read": "Full text" if paper.full_text else "Abstract",
        })
    return rows


# Fig. 1 counts study designs only when the metadata actually supports it.
# `classify_design` reads titles, venues and index metadata, never an abstract,
# so a pool of terse or badly indexed titles collapses into "other" — and a bar
# chart that is 80% "Other" says less than the year histogram it replaced.
DESIGN_FIGURE_MIN_SHARE = 0.5
FIGURE_WIDE_BUCKETS = 6


def _figure_series(cited: list[Paper], labels: dict[int, str]) -> dict | None:
    """Fig. 1's data: study design (or year fallback) buckets and relevance tallies.

    Returns None when there is too little to plot. Segmenting is all-or-nothing:
    with any source unlabelled the stack would silently drop it, so the figure
    falls back to a single undifferentiated series.
    """
    if len(cited) < 2:
        return None

    designs = {n: sources.classify_design(p) for n, p in enumerate(cited, start=1)}
    labelled_share = sum(1 for d in designs.values() if d != "other") / len(cited)
    distinct_categories = len({d for d in designs.values() if d})

    if labelled_share > DESIGN_FIGURE_MIN_SHARE and distinct_categories >= 2:
        mode = "design"
        buckets = [
            (sources.DESIGN_LABELS[cat], {n for n, d in designs.items() if d == cat})
            for cat in sources.DESIGN_DISPLAY_ORDER
            if any(d == cat for d in designs.values())
        ]
        segmented = all(labels.get(n) in RELEVANCE_LABELS for n in range(1, len(cited) + 1))
        series = (
            [lvl for lvl in ("direct", "related", "tangential")
             if any(labels.get(n) == lvl for n in range(1, len(cited) + 1))]
            if segmented else ["cited"]
        )
        counts = []
        for _, members in buckets:
            tally = {lvl: 0 for lvl in series}
            for n in members:
                tally[labels[n] if segmented else "cited"] += 1
            counts.append(tally)
        return {"buckets": buckets, "series": series, "counts": counts, "segmented": segmented, "mode": mode}

    # Year mode fallback
    dated = [(n, p) for n, p in enumerate(cited, start=1) if p.year]
    if len(dated) < 2:
        return None
    year_buckets = _year_buckets([p.year for _, p in dated])
    if not year_buckets:
        return None

    buckets = [
        (label, {n for n, p in dated if p.year in y_set})
        for label, y_set in year_buckets
    ]
    segmented = all(labels.get(n) in RELEVANCE_LABELS for n, _ in dated)
    series = (
        [lvl for lvl in ("direct", "related", "tangential")
         if any(labels.get(n) == lvl for n in range(1, len(cited) + 1))]
        if segmented else ["cited"]
    )
    counts = []
    for _, members in buckets:
        tally = {lvl: 0 for lvl in series}
        for n in members:
            tally[labels[n] if segmented else "cited"] += 1
        counts.append(tally)
    return {"buckets": buckets, "series": series, "counts": counts, "segmented": segmented, "mode": "year"}


# The box used to be captioned "Key study", which claims an appraisal nobody in
# this pipeline performs. `curate_sources` picks `most_relevant_index` on topic
# fit alone — there is no quality assessment anywhere — so one draft boxed a
# scoping review as the "Key study" while an adequately powered RCT sat in the
# body. Table 1's "Cited by" column was the only quality-looking number on the
# page (removed in #171) and a busy reader reads it as authority. The caption now
# says what the selection actually is, and the note says what it is not (#102).
_BOX_SELECTION_NOTE = (
    "Selected for closeness to the review question, not for study quality; "
    "no quality appraisal was performed."
)


def _box_html(article: dict, papers: list[Paper], cite_map: dict[int, int],
              flags: dict[str, str] | None = None) -> str:
    """Box 1 — the most relevant source, as a self-contained boxed display item."""
    parts = _box_parts(article, papers, cite_map)
    if not parts:
        return ""
    paper, fs = parts["paper"], parts
    display = parts["reference"]
    ref = f'<sup class="cite"><a href="#ref-{display}">{display}</a></sup>' if display else ""

    # Box 1 is a quotable unit too, and `verify` checks the featured study's
    # method and results along with the rest of the article.
    def field(name: str) -> str:
        return _mark_html(html.escape(fs[name]), flags or {})

    rows = []
    if fs.get("method"):
        rows.append(f'<p><span class="run-in">Method.</span> {field("method")}</p>')
    if fs.get("results"):
        rows.append(f'<p><span class="run-in">Results.</span> {field("results")}</p>')
    if fs.get("limitations"):
        rows.append(
            f'<p><span class="run-in">Limitations.</span> {field("limitations")}</p>')
    return (
        '<aside class="display box" aria-labelledby="box-1">\n'
        f'<p class="di-caption" id="box-1"><span class="di-label">Box 1 |</span> '
        f'Most relevant source: {html.escape(paper.title)}</p>\n'
        f'<p class="box-cite">{html.escape(_short_author(paper))} ({paper.year or "n.d."})'
        f'{(" · " + html.escape(paper.venue)) if paper.venue else ""}{ref}'
        f'{_source_link_html(paper)}</p>\n'
        + "\n".join(rows)
        + f'\n<p class="box-note">{html.escape(_BOX_SELECTION_NOTE)}</p>'
        + "\n</aside>"
    )


def _table_html(cited: list[Paper], labels: dict[int, str]) -> str:
    """Table 1 — characteristics of the cited evidence, one row per source."""
    if not cited:
        return ""
    # Rule: Always render the Read column to explicitly indicate whether
    # full text or abstract was accessed for every cited paper.
    show_read = True
    rows = []
    for row in _table_rows(cited, labels):
        relevance = (
            f'<span class="swatch swatch-{row["label"]}"></span>{row["relevance"]}'
            if row["label"] else "—"
        )
        rows.append(
            f'<tr><td class="t-num"><a href="#ref-{row["n"]}">{row["n"]}</a></td>'
            f"<td>{html.escape(row['study'])}</td>"
            f'<td class="t-num">{row["year"]}</td>'
            f'<td class="t-venue">{html.escape(row["venue"])}</td>'
            f"<td>{html.escape(row['design'])}</td>"
            f'<td class="t-num">{html.escape(row["sample"])}</td>'
            f'<td class="t-num">{html.escape(row["registered"])}</td>'
            f"<td>{html.escape(row['funding'])}</td>"
            f"<td>{relevance}</td>"
            f'<td>{row["read"]}</td></tr>'
        )
    caption = (
        "Characteristics of the cited evidence, read as text from each record; "
        "no quality appraisal was performed. Design is inferred from title, journal "
        "and index metadata (Other where none could be inferred). n is the largest "
        "sample size the abstract states. Registered is a trial registration number "
        "in the abstract or Europe PMC record. Funding reads Grant listed where "
        "Europe PMC lists a grant; a dash means none listed, not unfunded. Relevance is the "
        "curation label; Read is whether the model saw open-access full text or the "
        "abstract only."
    )
    return (
        '<div class="display table-wrap" aria-labelledby="table-1">\n'
        '<p class="di-caption" id="table-1"><span class="di-label">Table 1 |</span> '
        f"{caption}</p>\n"
        '<div class="table-scroll"><table>\n'
        "<thead><tr><th>Ref.</th><th>Study</th><th>Year</th><th>Source</th>"
        "<th>Design</th><th>n</th><th>Registered</th><th>Funding</th>"
        "<th>Relevance</th><th>Read</th></tr></thead>\n"
        "<tbody>" + "".join(rows) + "</tbody>\n</table></div>\n</div>"
    )


def _year_buckets(years: list[int]) -> list[tuple[str, set[int]]]:
    """One bucket per year, or 5-year bins once there are more than ten distinct years."""
    distinct = sorted(set(years))
    if not distinct:
        return []
    if len(distinct) <= 10:
        return [(str(y), {y}) for y in distinct]
    start = (min(distinct) // 5) * 5
    buckets = []
    while start <= max(distinct):
        members = {y for y in distinct if start <= y < start + 5}
        if members:
            buckets.append((f"{start}–{str(start + 4)[-2:]}", members))
        start += 5
    return buckets


def _rounded_top_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    r = max(0.0, min(r, w / 2, h))
    return (
        f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y + h:.1f} Z"
    )


def _figure_html(cited: list[Paper], labels: dict[int, str]) -> str:
    """Fig. 1 — cited sources by study design (or publication year), segmented by relevance.

    An ordinal one-hue ramp (direct darkest -> background lightest) driven by CSS
    variables, so the figure re-steps itself for the dark theme.
    """
    data = _figure_series(cited, labels)
    if not data:
        return ""
    buckets, series, counts = data["buckets"], data["series"], data["counts"]
    segmented = data["segmented"]
    mode = data.get("mode", "year")

    totals = [sum(t.values()) for t in counts]
    y_max = max(totals)
    step = max(1, -(-y_max // 4))          # at most four ticks
    y_top = step * -(-y_max // step)

    width = 660.0 if len(buckets) <= FIGURE_WIDE_BUCKETS else 860.0
    height = 250.0
    pad_l, pad_r, pad_t, pad_b = 40.0, 12.0, 24.0, 44.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    slot = plot_w / len(buckets)
    bar_w = min(46.0, slot * 0.6)

    def y_of(value: float) -> float:
        return pad_t + plot_h - (value / y_top) * plot_h

    if mode == "design":
        aria_label = (
            "Cited sources by study design, segmented by relevance"
            if segmented else "Cited sources by study design"
        )
        x_axis_title = "Study design"
    else:
        aria_label = (
            "Cited sources by publication year, segmented by relevance"
            if segmented else "Cited sources by publication year"
        )
        x_axis_title = "Year of publication"

    parts = [
        f'<svg class="fig-svg" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{aria_label}">'
    ]
    for tick in range(0, y_top + 1, step):
        y = y_of(tick)
        parts.append(
            f'<line class="fig-grid" x1="{pad_l:.1f}" x2="{width - pad_r:.1f}" '
            f'y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<text class="fig-tick" x="{pad_l - 8:.1f}" y="{y + 3.5:.1f}" '
            f'text-anchor="end">{tick}</text>'
        )

    for i, ((label, _), tally) in enumerate(zip(buckets, counts)):
        x = pad_l + slot * i + (slot - bar_w) / 2
        cursor = pad_t + plot_h
        topmost = next((lvl for lvl in reversed(series) if tally.get(lvl)), None)
        for lvl in series:
            value = tally.get(lvl, 0)
            if not value:
                continue
            seg_h = (value / y_top) * plot_h
            top = cursor - seg_h
            drawn = max(seg_h - 2, 1.0)     # 2px surface gap between segments
            name = RELEVANCE_LABELS.get(lvl, "Cited sources")
            title = f"<title>{label}: {value} {name.lower()}</title>"
            if lvl == topmost:
                parts.append(
                    f'<path class="fig-seg seg-{lvl}" '
                    f'd="{_rounded_top_path(x, top, bar_w, drawn)}">{title}</path>'
                )
            else:
                parts.append(
                    f'<rect class="fig-seg seg-{lvl}" x="{x:.1f}" y="{top:.1f}" '
                    f'width="{bar_w:.1f}" height="{drawn:.1f}">{title}</rect>'
                )
            cursor = top
        total = sum(tally.values())
        if total:
            parts.append(
                f'<text class="fig-value" x="{x + bar_w / 2:.1f}" '
                f'y="{cursor - 6:.1f}" text-anchor="middle">{total}</text>'
            )
        parts.append(
            f'<text class="fig-tick" x="{x + bar_w / 2:.1f}" '
            f'y="{pad_t + plot_h + 18:.1f}" text-anchor="middle">{html.escape(label)}</text>'
        )

    parts.append(
        f'<line class="fig-axis" x1="{pad_l:.1f}" x2="{width - pad_r:.1f}" '
        f'y1="{pad_t + plot_h:.1f}" y2="{pad_t + plot_h:.1f}"></line>'
    )
    parts.append(
        f'<text class="fig-axis-title" x="{width / 2:.1f}" y="{height - 8:.1f}" '
        f'text-anchor="middle">{x_axis_title}</text>'
    )
    parts.append(
        f'<text class="fig-axis-title" x="{-(pad_t + plot_h / 2):.1f}" y="12" '
        'text-anchor="middle" transform="rotate(-90)">Cited sources</text>'
    )
    parts.append("</svg>")

    legend = ""
    if segmented:
        chips = "".join(
            f'<li><span class="swatch swatch-{lvl}"></span>{RELEVANCE_LABELS[lvl]}</li>'
            for lvl in series
        )
        legend = f'<ul class="fig-legend">{chips}</ul>'

    if mode == "design":
        caption = "Composition of the evidence base. Cited sources by study design"
        caption += (
            ", segmented by how directly each addresses the review question. "
            if segmented else ". "
        )
        caption += (
            "Design is inferred from each record's title, journal and index metadata; "
            "it is not a quality appraisal. Bar labels give the number of sources in "
            "each category; the underlying records are listed in Table 1."
        )
    else:
        caption = "Composition of the evidence base. Cited sources by year of publication"
        caption += (
            ", segmented by how directly each addresses the review question. "
            if segmented else ". "
        )
        caption += (
            "Bar labels give the number of sources in each period; the underlying records "
            "are listed in Table 1."
        )
    return (
        '<figure class="display figure">\n'
        + "".join(parts)
        + f"\n{legend}\n"
        '<figcaption class="di-caption"><span class="di-label">Fig. 1 |</span> '
        f"{caption}</figcaption>\n</figure>"
    )


# --------------------------------------------------------------------------
# article furniture
# --------------------------------------------------------------------------

def _join_list(items: list[str]) -> str:
    """'A', 'A and B', 'A, B and C' — plain English, not 'A and B and C'.

    A two-source join read correctly by accident while there were only ever two
    databases to name; adding a third made the Methods sentence ungrammatical.
    """
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _methods_paragraphs(
    provenance: dict | None, screened: int, n_cited: int, topic: str, esc=lambda s: s,
    n_full_cited: int | None = None, n_retracted: int = 0,
) -> dict[str, str]:
    """The three Methods paragraphs as sentences, before any markup.

    Both renderers build this section from the same text and differ only in how
    they label and wrap the paragraphs. `esc` escapes the values that come from
    outside the module — the topic, the queries, the model and database names;
    the HTML renderer passes `html.escape`, Markdown passes them through.

    Methods is the one section whose whole purpose is stating what was actually
    done, so nothing here may be guessed. In particular, the databases are named
    only when provenance records which ones answered — this used to fall back to
    a hardcoded pair, which made every article claim a search it had not run.
    """
    provenance = provenance or {}
    queries = [q for q in (provenance.get("queries") or []) if q]
    databases = [d for d in (provenance.get("databases") or []) if d]
    date = provenance.get("date") or _au_date()
    model = provenance.get("model")

    search = f"Candidate records were retrieved on {esc(date)}"
    if databases:
        search += " from " + _join_list([esc(d) for d in databases])
    if queries:
        search += ", using the search strings " + "; ".join(f"‘{esc(q)}’" for q in queries)
    search += ". "
    if not databases:
        search += "The databases searched were not recorded for this draft. "
    if screened:
        search += (
            f"Records without a retrievable abstract were discarded, leaving {screened} "
            "for screening. "
        )
    if n_retracted:
        search += (
            f"{n_retracted} of those record{'s' if n_retracted != 1 else ''} "
            f"{'are' if n_retracted != 1 else 'is'} recorded as retracted in the "
            "source index and "
            f"{'were' if n_retracted != 1 else 'was'} withheld from the drafting "
            "model, so none can be cited here; they remain in the screened count "
            "above. "
        )
    search += (
        f"Each remaining record was assessed for how directly it addresses the question ‘{esc(topic.strip().rstrip(chr(46)))}’ "
        f"and labelled direct, related or background; {n_cited} were cited here and are "
        "listed in Table 1."
    )
    named = (provenance.get("named_sources") or {}) if provenance else {}
    named_queries = [q for q in (named.get("queries") or []) if q]
    if named_queries:
        added = int(named.get("added") or 0)
        search += (
            f" A second, targeted search then looked up {len(named_queries)} "
            f"work{'s' if len(named_queries) != 1 else ''} named in the most "
            f"relevant abstracts ("
            + "; ".join(f"‘{esc(q)}’" for q in named_queries) + "), which added "
            + (f"{added} further record{'s' if added != 1 else ''} to the pool."
               if added else "no further records to the pool.")
        )

    referenced = (provenance.get("referenced_sources") or {}) if provenance else {}
    ref_seeds = [s for s in (referenced.get("seeds") or []) if s]
    cited_seed = referenced.get("cited_seed") or ""
    if ref_seeds or cited_seed:
        ref_added = int(referenced.get("added") or 0)
        clauses = []
        if ref_seeds:
            clauses.append(
                f"the reference lists of {len(ref_seeds)} "
                f"synthes{'es' if len(ref_seeds) != 1 else 'is'} ("
                + "; ".join(f"‘{esc(s)}’" for s in ref_seeds) + ")")
        if cited_seed:
            clauses.append(f"the papers citing ‘{esc(cited_seed)}’")
        search += (
            " Citations were then followed one step through OpenAlex, through "
            + _join_list(clauses) + ", which added "
            + (f"{ref_added} further record{'s' if ref_added != 1 else ''} to the pool."
               if ref_added else "no further records to the pool.")
        )

    # `full_text_sources` records which sources the model was actually shown
    # full text for. Like `databases` above, it is never guessed: absent or
    # empty means this draft was abstracts-only and Methods says exactly that.
    n_full = len(provenance.get("full_text_sources") or []) if provenance else 0
    if n_full:
        via = (provenance.get("full_text_via") or {}) if provenance else {}
        retrieved_from = ("retrieved from Europe PMC" if not via.get("papers")
                          else "retrieved from their open-access copies")
        marked = (
            "(marked in Table 1)"
            if n_full_cited is None or n_full_cited == n_full
            else f"({n_full_cited} of which are cited here and marked in Table 1)"
        )
        handling = (
            f"Titles and abstracts were read for every record, and the open-access "
            f"full text{'s' if n_full != 1 else ''} of {n_full} "
            f"source{'s were' if n_full != 1 else ' was'} {retrieved_from} "
            f"and read alongside them {marked}; claims resting on the "
            "remaining sources draw on no data beyond an abstract. Decimals, "
            "percentages, effect estimates and quantities carrying a clinical unit "
            "were then checked automatically, each against the sources its own "
            "sentence cites, within exactly the material shown to the model — the "
            "abstracts plus those full-text excerpts. A figure that could not be "
            "located, or that appears only in a source other than the one cited, is "
            "flagged under Limitations."
        )
    else:
        handling = (
            "Only titles and abstracts were read; full texts were not retrieved, so no claim "
            "in this review rests on data reported beyond an abstract. Decimals, percentages, "
            "effect estimates and quantities carrying a clinical unit were checked "
            "automatically, each against the abstracts of the sources its own sentence cites. "
            "A figure that could not be located, or that appears only in a source other than "
            "the one cited, is flagged under Limitations."
        )

    planner = provenance.get("planning_model")
    if model and planner:
        lead = (f"Search planning and source curation were performed by a large language "
                f"model ({esc(planner)}), and drafting by another ({esc(model)}).")
    else:
        lead = ("Search planning, source curation and drafting were performed by a large "
                "language " + (f"model ({esc(model)})." if model else "model."))
    generation = (
        lead
        + " Source retrieval, relevance tabulation, citation numbering, Table 1, Fig. 1 and "
        "the statistical check are deterministic and were not model-generated."
    )
    return {"search": search, "handling": handling, "generation": generation}


def _methods_html(
    provenance: dict | None, screened: int, n_cited: int, topic: str,
    n_full_cited: int | None = None, n_retracted: int = 0,
) -> str:
    parts = _methods_paragraphs(
        provenance, screened, n_cited, topic, html.escape,
        n_full_cited=n_full_cited, n_retracted=n_retracted,
    )
    return (
        '<section class="back-section">\n<h2>Methods</h2>\n'
        f'<p><span class="run-in">Search strategy.</span> {parts["search"]}</p>\n'
        f'<p><span class="run-in">Evidence handling.</span> {parts["handling"]}</p>\n'
        f'<p><span class="run-in">Generation.</span> {parts["generation"]}</p>\n</section>'
    )


def _full_text_count(cited: list[Paper]) -> int:
    """How many cited sources the model read in full.

    Derived from the papers themselves, exactly as Table 1's Read column is, so
    the two cannot disagree. Methods counts what was *fetched* (from
    provenance); every statement about this synthesis's evidence base has to
    count what was fetched *and* cited, which is this. Absent full text this is
    0 and the abstracts-only wording is the truthful one — never assume either
    way, the same no-fallback rule as `databases`.
    """
    return sum(1 for p in cited if getattr(p, "full_text", None))


def _retracted_count(papers: list[Paper]) -> int:
    """How many screened records were withheld from the writer as retracted."""
    return sum(1 for p in papers if getattr(p, "is_retracted", False))


def _assessment_paragraphs(
    cited: list[Paper],
    counts: dict,
    verification: dict | None,
    style_report: dict | None = None,
    esc=lambda s: s,
) -> dict:
    """The Evidence assessment opening and its Limitations, before any markup.

    Shared by both renderers. The model used to add an `evidence_note` here; it
    is no longer part of the schema, because every tally it wrote could — and
    did — contradict the counts computed two lines above it. One article claimed
    "13 out of 20 sources" beside a line reading 9; another claimed "5 sources",
    "the majority is related" and "some are tangential" against a computed 3
    cited, all direct, none related, none background. Nothing countable in this
    section is model-written now (issue #54).
    """
    n = len(cited)
    if counts:
        direct, related, background = (
            counts.get("direct", 0), counts.get("related", 0), counts.get("tangential", 0)
        )
        # Agreement matters here: a one-source review printed "1 address the
        # review question directly" in the section that exists to be precise.
        opening = (
            f"Of the {n} source{'' if n == 1 else 's'} cited, "
            f"{direct} {'addresses' if direct == 1 else 'address'} the review question "
            f"directly, {related} {'is' if related == 1 else 'are'} related and "
            f"{background} {'provides' if background == 1 else 'provide'} background only"
        )
    else:
        opening = f"This review cites {n} source{'' if n == 1 else 's'}"
    span = _year_range(cited)
    if span:
        opening += f"; they were published in {span}"
    opening += ". The composition of the evidence base is shown in Fig. 1."

    # Abstracts-only used to be asserted here unconditionally, so an article
    # that had read seven full texts still told the reader none were available —
    # contradicting its own Methods section (issue #75). Both wordings are
    # derived, never assumed.
    n_full = _full_text_count(cited)
    n_abstract = n - n_full
    if n_full:
        limit = (
            f"This synthesis was prepared from the open-access full text"
            f"{'s' if n_full != 1 else ''} of {n_full} cited source"
            f"{'s' if n_full != 1 else ''}"
        )
        if n_abstract:
            limit += (
                f" and the abstract{'s' if n_abstract != 1 else ''} of the remaining "
                f"{n_abstract}. Where only an abstract was available, effect estimates, "
                "methodological detail, and the limitations that authors report only in "
                "a full text were unavailable, so the strength of those studies could "
                "not be appraised here."
            )
        else:
            limit += (
                ". The deeply read subset skews open-access, which is a selection the "
                "reader should weigh alongside the findings."
            )
        limitations = [limit]
    else:
        limitations = [
            "This synthesis was prepared from abstracts alone. Effect estimates, "
            "methodological detail, and the limitations that authors report only in a full "
            "text were unavailable, so the strength of individual studies could not be "
            "appraised here."
        ]
    if counts and not counts.get("direct"):
        limitations.append(
            "No cited source studies the review question directly. The claims made above "
            "are extrapolated from adjacent populations or mechanisms and should be "
            "treated as provisional until directly relevant work is identified."
        )
    unverified = (verification or {}).get("unverified") or []
    if unverified:
        limitations.append(_unverified_sentence([esc(u) for u in unverified], bool(n_full)))
    misattributed = (verification or {}).get("misattributed") or []
    if misattributed:
        limitations.append(_misattributed_sentence([esc(m) for m in misattributed]))
    surviving = _style_failure_sentence(style_report, esc)
    if surviving:
        limitations.append(surviving)
    verdict = _working_draft_sentence(style_report, verification)
    if verdict:
        limitations.append(verdict)
    return {"opening": opening, "limitations": limitations}


# Style errors that change whether this page can be sent, as against prose nits
# a reader would never notice. Only these reach the article; every rule still
# fires, still buys a revision pass, and still prints in the CLI log and in
# `style_report` (#169).
#
# Derived from SUBSTANCE_RULES with the two exemptions named, so a *new*
# substance rule brands the page by default — the safe direction — while the
# two that were branding four of five shipped drafts are listed where the
# reason for exempting them is readable. `under-length` is severity 'warning',
# so it never reaches this code; it is named anyway so that a future promotion
# to error does not silently start branding pages on a length count.
SENDABLE_BLOCKING_RULES = frozenset({"clinical-directive"}) | (
    SUBSTANCE_RULES - {"recycled-phrasing", "repeated-opener", "under-length"}
)


# What each rule means to a reader, who has not read style.py. Phrased as the
# defect in the prose, not as the name of a check.
_STYLE_FAILURE_WORDING = {
    "clinical-directive": "instructs the reader on treatment rather than "
                          "reporting what the studies found",
    "echoed-abstract": "repeats the abstract rather than adding to it",
    "bundled-citations": "cites sources in groups without describing them individually",
    "recycled-phrasing": "reuses phrasing between sections",
    "repeated-opener": "opens successive sentences the same way",
    "hedge-monotony": "leans on a single hedging phrase",
    "under-hedged": "states claims more firmly than the sources support",
    "too-few-sections": "covers the topic in fewer sections than the evidence allows",
    "first-person": "uses an author voice this synthesis has no claim to",
    "second-person": "addresses the reader directly",
    "booster": "uses intensifiers the sources do not carry",
    "overclaim": "claims proof the evidence cannot give",
    "contraction": "uses contractions",
    "rhetorical-question": "poses rhetorical questions",
    "exclamation": "uses exclamations",
}


def _style_failure_sentence(style_report: dict | None, esc=lambda s: s) -> str:
    """State, in the article, that the prose check still objected — when it matters.

    A draft that failed the style gate used to be indistinguishable from one that
    passed (issue #53). It then went too far the other way: four of five shipped
    drafts carried this line over a repeated sentence opener or a recycled
    six-word phrase (#169). Only SENDABLE_BLOCKING_RULES reach the reader now.
    """
    failures = [
        i for i in (style_report or {}).get("issues", [])
        if i.get("severity") == "error" and i.get("rule") in SENDABLE_BLOCKING_RULES
    ]
    if not failures:
        return ""
    # Deduplicate by rule: the same rule firing on two sections is one defect to
    # a reader, and the wording above is per-rule.
    seen, described = set(), []
    for issue in failures:
        rule = issue.get("rule", "")
        if rule in seen:
            continue
        seen.add(rule)
        described.append(_STYLE_FAILURE_WORDING.get(rule, rule.replace("-", " ")))
    if len(described) == 1:
        faults = described[0]
    elif len(described) == 2:
        faults = " and ".join(described)
    else:
        faults = ", ".join(described[:-1]) + ", and " + described[-1]
    return (
        "An automated check of the writing against journal prose conventions was "
        f"not satisfied by this draft: it {esc(faults)}. A revision was attempted "
        "and did not resolve this."
    )


def _working_draft_sentence(style_report: dict | None, verification: dict | None) -> str:
    """The one sentence that tells a reader not to send this page as it stands.

    Printed for defects that change whether the page can be sent: a surviving
    clinical directive, a substance failure a revision did not clear, or a
    figure the statistics check could not place. Never for a prose nit (#169).
    It always follows a sentence that has just named the defect, hence
    "therefore".
    """
    blocking = any(
        i.get("severity") == "error" and i.get("rule") in SENDABLE_BLOCKING_RULES
        for i in (style_report or {}).get("issues", [])
    )
    figures = bool((verification or {}).get("unverified")
                   or (verification or {}).get("misattributed"))
    if not (blocking or figures):
        return ""
    return (
        "The text below should therefore be read as a working draft rather than "
        "a finished review."
    )


def _assessment_html(
    cited: list[Paper],
    counts: dict,
    verification: dict | None,
    style_report: dict | None = None,
) -> str:
    """Evidence assessment + Limitations — the journal-register replacement for the
    warning boxes: the same facts, stated in prose, in the back matter.

    Entirely deterministic. A legacy draft carrying `evidence_note` renders
    without it rather than reprinting a tally that may contradict the counts.
    """
    parts = _assessment_paragraphs(cited, counts, verification, style_report, html.escape)
    body = [
        "<p>" + parts["opening"] + "</p>",
        # The id is the target of every inline figure flag, so a reader who taps
        # a dagger lands on the paragraph that explains it.
        '<p id="limitations"><span class="run-in">Limitations.</span> '
        + " ".join(parts["limitations"]) + "</p>",
    ]
    return '<section class="back-section">\n<h2>Evidence assessment</h2>\n' + "\n".join(body) + "\n</section>"


def _glossary_html(article: dict) -> str:
    entries = [
        e for e in (article.get("glossary") or [])
        if isinstance(e, dict) and e.get("term") and e.get("definition")
    ]
    if not entries:
        return ""
    items = "".join(
        f'<div class="gloss-row"><dt>{html.escape(e["term"])}</dt>'
        f'<dd>{html.escape(e["definition"])}</dd></div>'
        for e in entries
    )
    return (
        '<section class="back-section glossary">\n<h2>Glossary</h2>\n'
        f"<dl>{items}</dl>\n</section>"
    )


def _back_matter_html(cited: list[Paper], topic: str, article: dict, provenance: dict | None) -> str:
    linked = sum(1 for p in cited if p.link)
    model = (provenance or {}).get("model")
    contributions = (
        "Search planning, source curation and drafting were carried out by an automated "
        "pipeline (articlegen"
        + (f", {html.escape(model)}" if model else "")
        + "). No human author wrote or verified the text before publication of this draft."
    )
    return (
        '<section class="back-section">\n<h2>Additional information</h2>\n'
        f'<p><span class="run-in">Data availability.</span> No new data were generated. '
        f"All evidence cited is published and openly indexed; {linked} of the "
        f"{len(cited)} cited records resolve through the links in the reference list.</p>\n"
        f'<p><span class="run-in">Author contributions.</span> {contributions}</p>\n'
        '<p><span class="run-in">Competing interests.</span> None declared.</p>\n'
        f'<p><span class="run-in">Peer review.</span> This article has not been peer '
        "reviewed and is not a publication of record.</p>\n</section>"
    )


def _disclaimer_html(topic: str, article: dict, cited: list[Paper]) -> str:
    # "not the full texts" was stated unconditionally here, which understated
    # the grounding of any draft that had read some (issue #75).
    read = _read_phrase(cited)
    base = (
        f"Machine-generated evidence review of the journal articles listed in the "
        f"references, {read}. Follow the source links before relying on any "
        "specific claim."
    )
    if _is_clinical(topic, article):
        base = (
            "<strong>Not medical or clinical advice.</strong> This is a machine-generated "
            f"summary of the cited journal articles, {read}, for background only. "
            "It is not a substitute for professional judgement, primary sources, or "
            "clinical guidelines. Verify every claim, figure, and dose against the cited "
            "papers before acting on it."
        )
    return base


def _synthesis_label(cited: list[Paper]) -> str:
    """The masthead's one-line statement of what this was built from."""
    n_full = _full_text_count(cited)
    if not n_full:
        return "Abstract-derived synthesis"
    if n_full == len(cited):
        return "Full-text synthesis"
    return f"Full text read for {n_full} of {len(cited)} sources"


def _source_depth_phrase(cited: list[Paper]) -> str:
    """How deeply the cited sources were read, as a clause for the banner.

    Same count as `_full_text_count`, for the same reason every other grounding
    statement uses it: the masthead banner and Table 1's Read column must never
    be able to disagree (issue #75). Never hardcode either wording here.
    """
    n_full = _full_text_count(cited)
    n = len(cited)
    if not n_full:
        return "from their abstracts alone"
    if n_full == n:
        return "from their full texts"
    texts = "full text" if n_full == 1 else "full texts"
    return f"from {n_full} {texts} and the abstracts of the other {n - n_full}"


def _disclosure_banner(cited: list[Paper]) -> str:
    """The masthead's plain-language statement that no human wrote this.

    Issue #91: "Not peer reviewed" in small grey type reads as *preprint* to a
    non-academic — real scholarship awaiting review, a far higher trust
    category than what this actually is. This says the quiet part at full
    contrast, directly under the title, before any of the prose.
    """
    return (
        "Written by a language model from the cited research, "
        f"{_source_depth_phrase(cited)}. No human author wrote or checked this "
        "text. Not peer reviewed."
    )


def _read_phrase(cited: list[Paper]) -> str:
    """A trailing clause describing how deeply the cited articles were read."""
    n_full = _full_text_count(cited)
    n = len(cited)
    if not n_full:
        return "written from their abstracts, not their full texts"
    if n_full == n:
        return "each read in full"
    return f"{n_full} of {n} read in full, the rest from their abstracts"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _review_body_html(
    article: dict,
    papers: list[Paper],
    cite_map: dict[int, int],
    valid_numbers: set[int],
    flags: dict[str, str] | None,
    cited: list[Paper],
    labels: dict[int, str],
    verification: dict | None,
) -> str:
    """Introduction → figure → thematic sections → box → key points → conclusions."""
    sections_html = []
    for section in article.get("sections", []):
        paragraphs = "\n".join(
            f"<p>{_prose(para, cite_map, valid_numbers, flags)}</p>"
            for para in section.get("paragraphs", [])
        )
        sections_html.append(
            f'<section class="body-section">\n<h2>{html.escape(section["heading"])}</h2>\n'
            f"{paragraphs}\n</section>"
        )

    box = _box_html(article, papers, cite_map, flags)
    figure = _figure_html(cited, labels)

    key_points = "\n".join(
        f"<li>{_prose(point, cite_map, valid_numbers, flags)}</li>"
        for point in _key_points(article)
    )
    note = _grounding_note(verification, bool(_full_text_count(cited)))
    note_html = f'<p class="key-points-note">{html.escape(note)}</p>\n' if note else ""
    key_points_html = (
        '<aside class="key-points">\n<h2>Key points</h2>\n'
        f'{note_html}<ul>\n'
        f"{key_points}\n</ul>\n</aside>" if key_points else ""
    )

    body: list[str] = []
    pending = [item for item in (figure, box) if item]
    positions = {1: figure, 2: box}
    for i, section_html in enumerate(sections_html, start=1):
        if key_points_html and i == len(sections_html) and i > 1:
            body.append(key_points_html)
        body.append(section_html)
        item = positions.get(i)
        if item and item in pending:
            body.append(item)
            pending.remove(item)
    body.extend(pending)
    if key_points_html and len(sections_html) <= 1:
        body.append(key_points_html)
    return "\n\n".join(body)


def _briefing_body_html(
    article: dict,
    papers: list[Paper],
    cite_map: dict[int, int],
    valid_numbers: set[int],
    flags: dict[str, str] | None,
    verification: dict | None,
    cited: list[Paper],
) -> str:
    """Findings, unknowns, papers to open. No journal display items.

    A structured finding leads with its claim and carries its own certainty
    grade; a legacy string finding renders as it always did, as plain text.
    """
    note = _grounding_note(verification, bool(_full_text_count(cited)))
    note_html = f'<p class="key-points-note">{html.escape(note)}</p>\n' if note else ""

    items_html = []
    for finding in article.get("findings") or []:
        claim, evidence, level = finding_parts(finding)
        if not (claim or evidence):
            continue
        if claim:
            pill = _certainty_pill(level)
            items_html.append(
                '<li class="finding">'
                f'<p class="finding-claim">{_prose(claim, cite_map, valid_numbers, flags)}'
                f'{pill}</p>'
                + (f'<p class="finding-evidence">'
                   f'{_prose(evidence, cite_map, valid_numbers, flags)}</p>' if evidence else "")
                + "</li>"
            )
        else:
            items_html.append(f"<li>{_prose(evidence, cite_map, valid_numbers, flags)}</li>")
    structured = any(isinstance(f, dict) for f in article.get("findings") or [])
    list_tag = "ol" if structured else "ul"
    findings_html = (
        f'<section class="key-points findings{" structured" if structured else ""}">\n'
        '<h2>What the evidence shows</h2>\n'
        f'{note_html}<{list_tag}>\n' + "\n".join(items_html) + f'\n</{list_tag}>\n</section>'
        if items_html else ""
    )

    unknowns = "\n".join(
        f"<li>{_prose(item, cite_map, valid_numbers, flags)}</li>"
        for item in (article.get("unknowns") or [])
    )
    unknowns_html = (
        '<section class="key-points unknowns">\n<h2>What remains open</h2>\n'
        f'<ul>\n{unknowns}\n</ul>\n</section>' if unknowns else ""
    )

    items = []
    for raw, why in open_these_items(article):
        if not 1 <= raw <= len(papers):
            continue
        paper = papers[raw - 1]
        n = cite_map.get(raw)
        num = f'<a class="paper-num" href="#ref-{n}">{n}</a>' if n else ""
        title = html.escape(_titled(paper.title))
        title_html = (
            f'<a href="{html.escape(paper.link, quote=True)}">{title}</a>'
            if paper.link else title
        )
        venue = f" · {html.escape(paper.venue)}" if paper.venue else ""
        why_html = f'<p class="paper-why">{html.escape(why)}</p>' if why else ""
        items.append(
            f'<li>{num}<div><p class="paper-title">{title_html}</p>'
            f'<p class="paper-meta">{html.escape(_short_author(paper))} '
            f'({paper.year or "n.d."}){venue}</p>{why_html}</div></li>'
        )
        if len(items) == 3:
            break
    open_html = (
        '<section class="key-points open-these">\n<h2>Papers to open first</h2>\n'
        f'<ol class="paper-list">\n{"".join(items)}\n</ol>\n</section>' if items else ""
    )
    return "\n\n".join(p for p in (findings_html, unknowns_html, open_html) if p)


def _same_words(topic: str, *others: str) -> bool:
    """True when `topic` says nothing that one of `others` does not."""
    def words(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "a", "an", "of", "in", "on", "for", "and", "to", "does", "do", "is", "are"}
    have = words(topic)
    return any(have and have <= words(o) for o in others if o)


def _certainty_pill(level: str, prefix: str = "") -> str:
    if not level:
        return ""
    slug = level.replace(" ", "-")
    return (f' <span class="certainty c-{slug}" title="GRADE-style certainty, '
            f'assessed by the model">{html.escape(prefix + CERTAINTY_LABELS[level])}</span>')


def _briefing_lead_html(
    article: dict, topic: str, cite_map: dict[int, int], valid_numbers: set[int],
    flags: dict[str, str] | None,
) -> str:
    """Question, then the answer with its verdict sentence set in bold and the
    overall certainty under it."""
    parts = []
    if article.get("question"):
        parts.append(
            '<p class="question"><span class="run-in-head">Question</span>'
            f'{_prose(article["question"], cite_map, valid_numbers, flags)}</p>'
        )
    verdict, rest = split_verdict(_summary_text(article))
    level, reason = overall_certainty(article)
    body = f'<strong class="verdict">{_prose(verdict, cite_map, valid_numbers, flags)}</strong>'
    if rest:
        body += " " + _prose(rest, cite_map, valid_numbers, flags)
    certainty = ""
    if level:
        certainty = (
            '<p class="certainty-row">'
            f'{_certainty_pill(level).strip()}'
            + (f' <span class="certainty-reason">{html.escape(reason)}</span>' if reason else "")
            + "</p>"
        )
    parts.append(
        '<section class="answer">\n<p class="answer-label">Answer</p>\n'
        f'<p class="answer-text">{body}</p>\n{certainty}\n</section>'
    )
    return "\n".join(parts)


def render_article(
    article: dict,
    papers: list[Paper],
    topic: str,
    curation: dict | None = None,
    verification: dict | None = None,
    provenance: dict | None = None,
    style_report: dict | None = None,
    standalone: bool = True,
) -> str:
    """Render the article as one self-contained HTML page.

    `standalone=False` drops the three scripted parts — the theme-restoring
    head script, the Share/Copy/Theme toolbar, and the script that backs it.
    Two reasons, and they point the same way.

    *Security.* The web app shows the article in an iframe, and until this
    existed that iframe ran same-origin with the visitor's API key in
    localStorage. It is fed model-written HTML, and worse, `#read=` and `#p=`
    links let anyone hand a visitor a page of their choosing. A `sandbox`
    attribute fixes that, but a sandbox without `allow-scripts` would leave the
    toolbar's buttons present and dead. Removing them is what makes the sandbox
    free (issue #100).

    *Duplication.* The app already has a sticky action bar with Share, Copy link
    and a theme toggle, so in that context the in-article toolbar was a second
    copy of controls the page around it already provides.

    A file written to `drafts/` is opened directly in a browser with no app
    around it, so it keeps all three.
    """
    cited, cite_map = _citation_map(article, papers)
    valid_numbers = set(range(1, len(cited) + 1))
    labels = _display_relevance(cite_map, curation)
    counts = _relevance_counts(cite_map, curation)
    flags = _figure_flags(verification)
    table = _table_html(cited, labels)
    briefing = is_briefing(article)

    if briefing:
        body_html = _briefing_body_html(
            article, papers, cite_map, valid_numbers, flags, verification, cited)
        article_type = "Evidence briefing"
        summary_label = "Answer"
    else:
        body_html = _review_body_html(
            article, papers, cite_map, valid_numbers, flags, cited, labels,
            verification)
        article_type = "Evidence Review"
        summary_label = "Abstract"

    question_html = ""
    if article.get("question"):
        question_html = (
            '<p class="question"><span class="run-in-head">Question</span>'
            f'{_prose(article["question"], cite_map, valid_numbers, flags)}</p>'
        )

    keywords = [k for k in (article.get("keywords") or []) if k]
    keywords_html = (
        '<p class="keywords"><span class="meta-label">Keywords</span> '
        + "; ".join(html.escape(k) for k in keywords) + "</p>"
        if keywords else ""
    )

    if briefing:
        # Keywords are index terms, not reading: they sit under the body rather
        # than between the answer and the evidence.
        lead_html = _briefing_lead_html(article, topic, cite_map, valid_numbers, flags)
        if keywords_html:
            body_html += "\n\n" + keywords_html.replace(
                'class="keywords"', 'class="keywords keywords-end"', 1)
        # Fig. 1 is referenced by the evidence assessment on both shapes; the
        # briefing used to cite it without ever drawing it.
        table = "\n\n".join(p for p in (_figure_html(cited, labels), table) if p)
    else:
        lead_html = (
            f"{question_html}\n"
            '<div class="abstract">\n'
            f'<p><span class="run-in-head">{html.escape(summary_label)}</span>'
            f"{_prose(_summary_text(article), cite_map, valid_numbers, flags)}</p>\n"
            "</div>\n" + keywords_html
        )

    # The subject is what the reader typed. A briefing's question restates it,
    # usually word for word, so the line only earns its place when they differ.
    subject_html = (
        f'<p class="subject-line"><span class="meta-label">Subject</span>{html.escape(topic)}</p>'
    )
    if briefing and _same_words(topic, article.get("question") or "", article.get("title") or ""):
        subject_html = ""

    refs_html = []
    for n, paper in enumerate(cited, start=1):
        title = html.escape(_titled(paper.title))
        title_html = (
            f'<a href="{html.escape(paper.link, quote=True)}">{title}</a>'
            if paper.link else title
        )
        venue = f' <em>{html.escape(paper.venue)}</em>' if paper.venue else ""
        cites = (
            f' <span class="ref-cites">Cited by {paper.citation_count:,}</span>'
            if paper.citation_count else ""
        )
        preprint = " (preprint, not peer reviewed)" if paper.is_preprint else ""
        correction = f" ({paper.correction_note})" if paper.correction_note else ""
        refs_html.append(
            f'<li id="ref-{n}"><span class="ref-authors">{html.escape(_reference_authors(paper))}</span> '
            f'{title_html}{venue} ({paper.year or "n.d."}).{preprint}{correction}{cites}</li>'
        )

    span = _year_range(cited)
    # The run's date, when the manifest carries one, so a page rendered again
    # later still says when it was generated; Methods uses the same rule.
    meta_bits = [
        f"Generated {(provenance or {}).get('date') or _au_date()}",
        f"{len(cited)} sources cited" + (f", {span}" if span else ""),
        _synthesis_label(cited),
    ]
    # The disclosure banner directly above already says it at full contrast;
    # the briefing does not need it a second time in grey.
    if not briefing:
        meta_bits.append("Not peer reviewed")

    # Format first, THEN inject the stylesheet: the CSS is full of braces that
    # str.format would try to read as fields.
    return (
        _TEMPLATE.format(
            page_title=html.escape(article["title"]),
            title=html.escape(article["title"]),
            subject_line=subject_html,
            meta_line=" &middot; ".join(html.escape(b) for b in meta_bits),
            article_type=html.escape(article_type),
            lead=lead_html,
            body=body_html,
            methods=_methods_html(provenance, len(papers), len(cited), topic,
                                  n_full_cited=_full_text_count(cited),
                                  n_retracted=_retracted_count(papers)),
            assessment=_assessment_html(cited, counts, verification, style_report),
            table=table,
            glossary=_glossary_html(article),
            references="\n".join(refs_html),
            additional=_back_matter_html(cited, topic, article, provenance),
            disclaimer=_disclaimer_html(topic, article, cited),
            disclosure_banner=html.escape(_disclosure_banner(cited)),
            head_script=_HEAD_SCRIPT if standalone else "",
            toolbar=_TOOLBAR if standalone else "",
            foot_script=_FOOT_SCRIPT if standalone else "",
        ).replace("__STYLE__", _CSS)
    )


_CSS = """
  :root, html[data-theme="light"] {
    --bg: #ffffff;
    --surface: #f7f7f5;
    --ink: #14171a;
    --ink-2: #3c4249;
    --muted: #6a7178;
    --accent: #17457f;
    --link: #1b5aa8;
    --rule: #dfdfda;
    --rule-strong: #b9bab5;
    --seg-direct: #184f95;
    --seg-related: #2a78d6;
    --seg-tangential: #86b6ef;
    --c-high: #1f6b45; --c-high-bg: #e3f1e8;
    --c-moderate: #1d5a96; --c-moderate-bg: #e3edf8;
    --c-low: #8a5a00; --c-low-bg: #f8eed9;
    --c-very-low: #9b2c1f; --c-very-low-bg: #f8e4e0;
    color-scheme: light;
  }
  html[data-theme="dark"] {
    --bg: #14150f;
    --surface: #1e1f1c;
    --ink: #f2f1ea;
    --ink-2: #cfcec5;
    --muted: #9d9d94;
    --accent: #9dc4f5;
    --link: #86b6ef;
    --rule: #33342f;
    --rule-strong: #4c4d47;
    --seg-direct: #86b6ef;
    --seg-related: #3987e5;
    --seg-tangential: #184f95;
    --c-high: #8fd6ad; --c-high-bg: #1d3328;
    --c-moderate: #9dc4f5; --c-moderate-bg: #1d2b3d;
    --c-low: #f0c46e; --c-low-bg: #3a3020;
    --c-very-low: #f2a193; --c-very-low-bg: #3d2420;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Iowan Old Style", "Charter", Georgia, "Times New Roman", serif;
    font-size: 1.02rem;
    line-height: 1.62;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
  }
  /* Phone: padded viewport. Desktop: the page uses the window, not a boxed
     column — side padding only. */
  main { max-width: none; width: 100%; margin: 0; padding: 2rem 1.25rem 4rem; }
  /* Full window, readable measure: wide screens centre the article at about
     a broadsheet page rather than running 170-character lines. */
  article { max-width: 68rem; margin: 0 auto; }
  @media (min-width: 1100px) {
    main { padding: 2.5rem 4.5rem 4rem; }
  }
  .sans, .meta-line, .subject-line, .keywords, .di-caption, .fig-legend, .key-points,
  .refs ol, .toolbar, .colophon, table, .meta-label {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }

  /* ---- masthead ---------------------------------------------------- */
  header.masthead { border-top: 3px solid var(--ink); padding-top: 0.9rem; margin-bottom: 2rem; }
  .article-type {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem;
  }
  h1 {
    font-size: clamp(1.7rem, 4.2vw, 2.3rem);
    line-height: 1.22; font-weight: 700; letter-spacing: -0.005em;
    margin: 0 0 0.9rem;
  }
  /* Full contrast, body-text size, above the fold. The point of issue #91 is
     that a disclosure in muted 0.78rem type is not a disclosure. */
  .ai-disclosure {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.92rem; font-weight: 600; line-height: 1.45; color: var(--ink);
    margin: 0 0 0.9rem; padding: 0.6rem 0.8rem;
    background: var(--surface); border-left: 3px solid var(--ink);
  }
  .meta-line {
    font-size: 0.78rem; color: var(--muted); margin: 0 0 0.35rem;
    padding-bottom: 0.9rem; border-bottom: 1px solid var(--rule);
  }
  .subject-line { font-size: 0.78rem; color: var(--muted); margin: 0 0 0.9rem; }
  .meta-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); margin-right: 0.4rem;
  }

  /* ---- abstract & key points --------------------------------------- */
  .abstract { margin: 0 0 1.2rem; }
  .abstract p { margin: 0; font-size: 1.06rem; line-height: 1.6; color: var(--ink-2); }
  .abstract .run-in-head, .question .run-in-head {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink); margin-right: 0.5rem;
  }
  .question { margin: 0 0 1rem; font-size: 1.06rem; line-height: 1.6; color: var(--ink-2); }
  .keywords { font-size: 0.8rem; color: var(--muted); margin: 0 0 2rem; }
  aside.key-points {
    background: var(--surface); border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    padding: 1.1rem 1.3rem 1.2rem; margin: 0 0 2.2rem; border-radius: 3px;
  }
  aside.key-points h2 {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); margin: 0 0 0.7rem;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  aside.key-points ul { margin: 0; padding-left: 1.1rem; }
  .key-points-note { color: var(--muted); font-size: 0.78rem !important;
                     font-style: italic; margin: 0 0 0.7rem 0 !important; }
  sup.flag { font-size: 0.72em; line-height: 0; vertical-align: super; }
  sup.flag a { color: var(--muted); text-decoration: none; font-weight: 700; }
  aside.key-points li { margin-bottom: 0.5rem; font-size: 0.95rem; line-height: 1.55; }
  aside.key-points li:last-child { margin-bottom: 0; }

  /* ---- briefing: answer, findings, papers --------------------------- */
  section.answer {
    margin: 0 0 1.4rem; padding: 1.15rem 1.3rem 1.1rem;
    border: 1px solid var(--rule-strong); border-radius: 6px; background: var(--surface);
  }
  .answer-label {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); margin: 0 0 0.45rem;
  }
  .answer-text { margin: 0; font-size: 1.08rem; line-height: 1.62; color: var(--ink-2); }
  .answer-text .verdict { color: var(--ink); font-weight: 700; }
  .certainty-row {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.3rem 0.6rem;
    margin: 0.85rem 0 0; padding-top: 0.75rem; border-top: 1px solid var(--rule);
    font-size: 0.84rem; color: var(--muted);
  }
  span.certainty {
    display: inline-block; white-space: nowrap; vertical-align: 0.08em;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.03em; line-height: 1.5;
    padding: 0.05rem 0.5rem; border-radius: 999px; margin-left: 0.35rem;
  }
  .certainty-row span.certainty { margin-left: 0; font-size: 0.74rem; }
  .c-high { color: var(--c-high); background: var(--c-high-bg); }
  .c-moderate { color: var(--c-moderate); background: var(--c-moderate-bg); }
  .c-low { color: var(--c-low); background: var(--c-low-bg); }
  .c-very-low { color: var(--c-very-low); background: var(--c-very-low-bg); }
  section.key-points {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0 0 2.2rem;
  }
  section.key-points h2 {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent);
    margin: 0 0 0.8rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--ink);
  }
  section.key-points ul { margin: 0; padding-left: 1.1rem; }
  section.key-points li { margin-bottom: 0.55rem; font-size: 0.95rem; line-height: 1.55; }
  section.findings.structured ol { list-style: none; counter-reset: finding; margin: 0; padding: 0; }
  li.finding {
    counter-increment: finding; position: relative;
    padding: 0.85rem 0 0.9rem 2.2rem; margin: 0 !important;
    border-bottom: 1px solid var(--rule);
  }
  li.finding:last-child { border-bottom: none; }
  li.finding::before {
    content: counter(finding); position: absolute; left: 0; top: 0.9rem;
    width: 1.45rem; height: 1.45rem; border-radius: 50%;
    background: var(--ink); color: var(--bg);
    font-size: 0.75rem; font-weight: 700; line-height: 1.45rem; text-align: center;
  }
  .finding-claim { margin: 0 0 0.3rem; font-size: 1rem; font-weight: 650; line-height: 1.45; color: var(--ink); }
  .finding-evidence { margin: 0; font-size: 0.92rem; line-height: 1.6; color: var(--ink-2); }
  ol.paper-list { list-style: none; margin: 0; padding: 0; }
  ol.paper-list li {
    display: flex; gap: 0.8rem; align-items: flex-start;
    padding: 0.7rem 0; border-bottom: 1px solid var(--rule); margin: 0;
  }
  ol.paper-list li:last-child { border-bottom: none; }
  .paper-num {
    flex: 0 0 auto; min-width: 1.6rem; text-align: center; text-decoration: none;
    font-size: 0.78rem; font-weight: 700; color: var(--link);
    border: 1px solid var(--rule-strong); border-radius: 4px; padding: 0.05rem 0.3rem;
  }
  .paper-title { margin: 0; font-size: 0.95rem; font-weight: 600; line-height: 1.4; }
  .paper-title a { color: var(--ink); text-decoration: none; }
  .paper-title a:hover { color: var(--link); text-decoration: underline; }
  .paper-meta { margin: 0.1rem 0 0; font-size: 0.8rem; color: var(--muted); }
  .paper-why { margin: 0.3rem 0 0; font-size: 0.88rem; color: var(--ink-2); }

  /* ---- body -------------------------------------------------------- */
  h2 {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 1.02rem; font-weight: 700; letter-spacing: 0.01em;
    margin: 2.2rem 0 0.7rem; color: var(--ink);
  }
  p { margin: 0 0 1rem; }
  .body-section p + p { text-indent: 1.2em; }
  .run-in { font-weight: 700; font-style: italic; }
  a { color: var(--link); }
  sup.cite { font-size: 0.68em; line-height: 0; font-weight: 600; letter-spacing: 0.02em; }
  sup.cite a { color: var(--link); text-decoration: none; padding: 0 0.02em; }
  sup.cite a:hover { text-decoration: underline; }

  /* ---- display items ----------------------------------------------- */
  .display { margin: 1.8rem 0 2rem; }
  .di-caption {
    font-size: 0.8rem; line-height: 1.5; color: var(--muted); margin: 0.7rem 0 0;
  }
  .di-label { font-weight: 700; color: var(--ink); }
  aside.box {
    background: var(--surface); border: 1px solid var(--rule-strong);
    padding: 1.1rem 1.3rem 1.2rem; border-radius: 3px;
  }
  aside.box .di-caption { margin: 0 0 0.7rem; font-size: 0.92rem; color: var(--ink); }
  aside.box p { font-size: 0.95rem; line-height: 1.55; margin: 0 0 0.6rem; }
  aside.box p:last-child { margin-bottom: 0; }
  .box-cite { color: var(--muted); font-size: 0.85rem !important; }
  .box-note { color: var(--muted); font-size: 0.78rem !important; font-style: italic;
              margin-top: 0.75rem !important; }
  figure.figure { margin: 1.8rem 0 2rem; padding: 1rem 0 0; border-top: 1px solid var(--rule); }
  .fig-svg { width: 100%; max-width: 44rem; height: auto; display: block; }
  .keywords-end { margin-top: -1rem; }
  .fig-grid { stroke: var(--rule); stroke-width: 1; }
  .fig-axis { stroke: var(--rule-strong); stroke-width: 1; }
  .fig-tick { fill: var(--muted); font-size: 11px;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .fig-value { fill: var(--ink-2); font-size: 11px; font-weight: 600;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .fig-axis-title { fill: var(--muted); font-size: 11px; letter-spacing: 0.04em;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .seg-direct { fill: var(--seg-direct); }
  .seg-related { fill: var(--seg-related); }
  .seg-tangential { fill: var(--seg-tangential); }
  .seg-cited { fill: var(--seg-related); }
  .fig-legend {
    display: flex; flex-wrap: wrap; gap: 0.3rem 1.1rem; list-style: none;
    margin: 0.6rem 0 0; padding: 0; font-size: 0.78rem; color: var(--muted);
  }
  .fig-legend li { display: flex; align-items: center; gap: 0.4rem; }
  .swatch {
    display: inline-block; width: 0.68rem; height: 0.68rem; border-radius: 2px;
    margin-right: 0.35rem; vertical-align: -0.05rem;
  }
  .swatch-direct { background: var(--seg-direct); }
  .swatch-related { background: var(--seg-related); }
  .swatch-tangential { background: var(--seg-tangential); }
  .table-scroll { overflow-x: auto; }
  table { width: 100%; min-width: 32rem; border-collapse: collapse; font-size: 0.82rem; }
  tbody td:nth-child(5) { white-space: nowrap; }
  thead th {
    text-align: left; font-weight: 700; color: var(--ink);
    border-top: 1px solid var(--rule-strong); border-bottom: 1px solid var(--rule-strong);
    padding: 0.45rem 0.5rem; white-space: nowrap;
  }
  tbody td { padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--rule); color: var(--ink-2); }
  tbody tr:last-child td { border-bottom: 1px solid var(--rule-strong); }
  .t-num { white-space: nowrap; font-variant-numeric: tabular-nums; }
  .t-venue { color: var(--muted); }

  /* ---- back matter -------------------------------------------------- */
  .back-section { margin-top: 2.2rem; }
  .back-section > h2:first-child, .refs h2 { margin-top: 0; }
  .back-section p { font-size: 0.92rem; line-height: 1.58; color: var(--ink-2); }
  .glossary dl { margin: 0; }
  .gloss-row { display: flex; gap: 0.75rem; padding: 0.4rem 0; border-bottom: 1px solid var(--rule); }
  .gloss-row dt { flex: 0 0 9.5rem; font-weight: 700; font-size: 0.92rem; }
  .gloss-row dd { margin: 0; font-size: 0.92rem; color: var(--ink-2); }
  .gloss-row:last-child { border-bottom: none; }
  .refs { margin-top: 2.4rem; border-top: 2px solid var(--ink); padding-top: 1.1rem; }
  .refs ol { margin: 0; padding-left: 1.5rem; font-size: 0.82rem; line-height: 1.5; }
  .refs li { margin-bottom: 0.55rem; color: var(--ink-2); }
  .refs li:target { background: var(--surface); border-radius: 2px; }
  .refs a { color: var(--link); text-decoration: none; }
  .refs a:hover { text-decoration: underline; }
  .ref-authors { color: var(--ink); }
  .ref-cites { color: var(--muted); }
  footer.colophon {
    margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid var(--rule);
    font-size: 0.78rem; line-height: 1.55; color: var(--muted);
  }

  /* ---- toolbar ------------------------------------------------------ */
  .toolbar { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 2rem; }
  .tool-btn {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: transparent; color: var(--ink-2);
    border: 1px solid var(--rule-strong); border-radius: 2px;
    padding: 0.35rem 0.7rem; font-size: 0.78rem; font-weight: 600;
    cursor: pointer; font-family: inherit;
  }
  .tool-btn:hover { border-color: var(--accent); color: var(--accent); }
  .share-toast { font-size: 0.78rem; color: var(--accent); align-self: center; display: none; }

  @media (max-width: 34rem) {
    /* Phones: the number floats beside the claim instead of indenting the
       whole finding, which cost a sixth of a narrow column. */
    li.finding { padding-left: 0; }
    li.finding::before {
      position: static; float: left; margin: 0.12rem 0.55rem 0 0;
    }
    section.answer { padding: 1rem 1rem 0.95rem; }
    .gloss-row { display: block; }
    .gloss-row dt { margin-bottom: 0.15rem; }
    .body-section p + p { text-indent: 0; }
  }
  @media print {
    body { font-size: 10.5pt; background: #fff; color: #000; }
    .toolbar { display: none; }
    a, sup.cite a { color: #000; }
  }
"""


# The three scripted parts of the page, kept out of _TEMPLATE so `standalone`
# can drop them. They are substituted *into* the format call as values, never
# formatted themselves, so their braces are ordinary single braces.
_HEAD_SCRIPT = """<script>
  (function() {
    var saved = localStorage.getItem('articlegen_theme') || 'light';
    var dark = saved === 'dark' || (saved === 'system' && window.matchMedia
      && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  })();
</script>"""

_TOOLBAR = """<div class="toolbar">
      <button class="tool-btn" onclick="shareArticle()">Share</button>
      <button class="tool-btn" onclick="copyArticleLink()">Copy link</button>
      <button class="tool-btn" onclick="toggleArticleTheme()"><span id="themeIcon">&#9789;</span> Theme</button>
      <span id="shareToast" class="share-toast">Link copied</span>
    </div>"""

_FOOT_SCRIPT = """<script>
function shareArticle() {
  if (navigator.share) {
    navigator.share({ title: document.title, url: window.location.href }).catch(function() {});
  } else {
    copyArticleLink();
  }
}
function copyArticleLink() {
  navigator.clipboard.writeText(window.location.href).then(function() {
    var t = document.getElementById('shareToast');
    if (t) {
      t.style.display = 'inline';
      setTimeout(function() { t.style.display = 'none'; }, 2500);
    }
  }).catch(function() {});
}
function toggleArticleTheme() {
  var cur = document.documentElement.getAttribute('data-theme');
  var effective = cur || 'light';
  var next = effective === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('articlegen_theme', next);
  updateArticleThemeIcon();
}
function updateArticleThemeIcon() {
  var icon = document.getElementById('themeIcon');
  if (!icon) return;
  var cur = document.documentElement.getAttribute('data-theme');
  var effective = cur || 'light';
  icon.innerHTML = effective === 'dark' ? '&#9789;' : '&#9788;';
}
document.addEventListener('DOMContentLoaded', updateArticleThemeIcon);
</script>"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
{head_script}
<style>__STYLE__</style>
</head>
<body>
<main>
  <article>
  <header class="masthead">
    <p class="article-type">{article_type}</p>
    <h1>{title}</h1>
    <p class="ai-disclosure">{disclosure_banner}</p>
    <p class="meta-line">{meta_line}</p>
    {subject_line}

    {toolbar}

    {lead}
  </header>

  {body}

  {methods}

  {assessment}

  {table}

  {glossary}

  <section class="refs">
    <h2>References</h2>
    <ol>
      {references}
    </ol>
  </section>

  {additional}

  <footer class="colophon">
    {disclaimer}
  </footer>
  </article>
</main>
{foot_script}
</body>
</html>
"""


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def render_markdown(
    article: dict,
    papers: list[Paper],
    topic: str,
    curation: dict | None = None,
    verification: dict | None = None,
    provenance: dict | None = None,
    style_report: dict | None = None,
) -> str:
    """Plain-Markdown version of the same journal-format article."""
    cited, cite_map = _citation_map(article, papers)
    labels = _display_relevance(cite_map, curation)
    counts = _relevance_counts(cite_map, curation)
    today = (provenance or {}).get("date") or _au_date()
    span = _year_range(cited)

    flags = _figure_flags(verification)

    def prose(text: str) -> str:
        # Flagging runs before citation collapsing, while the only brackets in
        # the string are still the literal [N] markers the pattern skips.
        return _plain_citations(
            _mark_plain(
                _shift_markers_after_punctuation(_remap_citations(text, cite_map)), flags)
        )

    briefing = is_briefing(article)
    lines = [
        "**EVIDENCE BRIEFING**" if briefing else "**EVIDENCE REVIEW**",
        "",
        f"# {article['title']}",
        "",
        f"**{_disclosure_banner(cited)}**",
        "",
        f"*Generated {today} · {len(cited)} sources cited"
        + (f", {span}" if span else "")
        + f" · {_synthesis_label(cited)}" + ("" if briefing else " · Not peer reviewed") + "*",
        "",
    ]
    if not (briefing and _same_words(topic, article.get("question") or "", article.get("title") or "")):
        lines += [f"**Subject:** {topic}", ""]
    if article.get("question"):
        lines += [f"**Question.** {prose(article['question'])}", ""]
    if briefing:
        verdict, rest = split_verdict(_summary_text(article))
        lines += [f"**Answer.** **{prose(verdict)}**" + (f" {prose(rest)}" if rest else ""), ""]
        level, reason = overall_certainty(article)
        if level:
            lines += [f"*{CERTAINTY_LABELS[level]}*" + (f" — {reason}" if reason else ""), ""]
    else:
        lines += [f"**Abstract.** {prose(_summary_text(article))}", ""]

    keywords = [k for k in (article.get("keywords") or []) if k]
    if keywords:
        lines += [f"**Keywords:** {'; '.join(keywords)}", ""]

    table_md = _table_markdown(cited, labels)
    note = _grounding_note(verification, bool(_full_text_count(cited)))

    if briefing:
        findings = article.get("findings") or []
        if findings:
            lines += ["## What the evidence shows", ""]
            if note:
                lines += [f"*{note}*", ""]
            for n, finding in enumerate(findings, start=1):
                claim, evidence, level = finding_parts(finding)
                if claim:
                    grade = f" *({CERTAINTY_LABELS[level]})*" if level else ""
                    lines.append(f"{n}. **{prose(claim)}**{grade} {prose(evidence)}".rstrip())
                elif evidence:
                    lines.append(f"- {prose(evidence)}")
            lines.append("")
        unknowns = article.get("unknowns") or []
        if unknowns:
            lines += ["## What remains open", ""]
            lines += [f"- {prose(p)}" for p in unknowns]
            lines.append("")
        opened = []
        for raw, why in open_these_items(article):
            if not 1 <= raw <= len(papers):
                continue
            paper = papers[raw - 1]
            n = cite_map.get(raw)
            num = f"[{n}] " if n else ""
            link = f" <{paper.link}>" if paper.link else ""
            opened.append(
                f"- {num}{_short_author(paper)} ({paper.year or 'n.d.'}). "
                f"{_titled(paper.title)}{link}" + (f" — {why}" if why else "")
            )
            if len(opened) == 3:
                break
        if opened:
            lines += ["## Papers to open first", ""] + opened + [""]
    else:
        # Same layout as the HTML: figure after the introduction, box after the
        # first thematic section, key points directly before the conclusions.
        points = _key_points(article)
        key_points_md = (
            "\n".join(["## Key points", ""]
                      + ([f"*{note}*", ""] if note else [])
                      + [f"- {prose(p)}" for p in points])
            if points else ""
        )
        sections = article.get("sections", [])
        box_md = _box_markdown(article, papers, cite_map, flags)
        figure_md = _figure_markdown(cited, labels)
        positions = {1: figure_md, 2: box_md}
        pending = [item for item in (figure_md, box_md) if item]

        for i, section in enumerate(sections, start=1):
            if key_points_md and i == len(sections) and i > 1:
                lines += [key_points_md, ""]
            lines += [f"## {section['heading']}", ""]
            for para in section.get("paragraphs", []):
                lines += [prose(para), ""]
            item = positions.get(i)
            if item and item in pending:
                lines += [item, ""]
                pending.remove(item)
        for item in pending:
            lines += [item, ""]
        if key_points_md and len(sections) <= 1:
            lines += [key_points_md, ""]

    lines += _methods_markdown(provenance, len(papers), len(cited), topic,
                               n_full_cited=_full_text_count(cited),
                               n_retracted=_retracted_count(papers))
    lines += _assessment_markdown(cited, counts, verification, style_report)
    if briefing:
        figure_md = _figure_markdown(cited, labels)
        if figure_md:
            lines += [figure_md, ""]
    if table_md:
        lines += [table_md, ""]

    glossary = [
        e for e in (article.get("glossary") or [])
        if isinstance(e, dict) and e.get("term") and e.get("definition")
    ]
    if glossary:
        lines += ["## Glossary", ""]
        lines += [f"- **{e['term']}** — {e['definition']}" for e in glossary]
        lines.append("")

    lines += ["## References", ""]
    for n, paper in enumerate(cited, start=1):
        venue = f" *{paper.venue}*" if paper.venue else ""
        link = f" <{paper.link}>" if paper.link else ""
        cites = f" Cited by {paper.citation_count:,}." if paper.citation_count else ""
        preprint = " (preprint, not peer reviewed)" if paper.is_preprint else ""
        correction = f" ({paper.correction_note})" if paper.correction_note else ""
        lines.append(
            f"{n}. {_reference_authors(paper)} {_titled(paper.title)}{venue} "
            f"({paper.year or 'n.d.'}).{preprint}{correction}{cites}{link}"
        )
    lines.append("")

    lines += ["## Additional information", ""]
    linked = sum(1 for p in cited if p.link)
    model = (provenance or {}).get("model")
    lines += [
        f"- **Data availability.** No new data were generated; {linked} of {len(cited)} "
        "cited records resolve through the reference links.",
        "- **Author contributions.** Drafted by an automated pipeline (articlegen"
        + (f", {model}" if model else "") + "); no human author verified the text.",
        "- **Competing interests.** None declared.",
        "- **Peer review.** Not peer reviewed; not a publication of record.",
        "",
        "---",
        "",
        _markdown_disclaimer(topic, article, cited),
        "",
    ]
    return "\n".join(lines)


def _box_markdown(article: dict, papers: list[Paper], cite_map: dict[int, int],
                  flags: dict[str, str] | None = None) -> str:
    parts = _box_parts(article, papers, cite_map)
    if not parts:
        return ""
    paper = parts["paper"]
    ref = f" [{parts['reference']}]" if parts["reference"] else ""
    out = [f"> **Box 1 | Most relevant source: {paper.title}**", ">",
           f"> {_short_author(paper)} ({paper.year or 'n.d.'})"
           + (f", {paper.venue}" if paper.venue else "") + ref
           + (f" <{paper.link}>" if paper.link else "")]
    for label in ("method", "results", "limitations"):
        if parts[label]:
            out.append(f"> **{label.capitalize()}.** "
                       + _mark_plain(parts[label], flags or {}))
    out.append(">")
    out.append(f"> _{_BOX_SELECTION_NOTE}_")
    return "\n".join(out)


def _table_markdown(cited: list[Paper], labels: dict[int, str]) -> str:
    if not cited:
        return ""
    rows = [
        "**Table 1 | Characteristics of the cited evidence.**",
        "",
        "| Ref. | Study | Year | Source | Design | n | Registered | Funding | Relevance | Read |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _table_rows(cited, labels):
        rows.append(
            f"| {row['n']} | {row['study']} | {row['year']} | "
            f"{row['venue']} | {row['design']} | {row['sample']} | {row['registered']} | "
            f"{row['funding']} | {row['relevance']} | {row['read']} |"
        )
    return "\n".join(rows)


def _figure_markdown(cited: list[Paper], labels: dict[int, str]) -> str:
    data = _figure_series(cited, labels)
    if not data:
        return ""
    mode = data.get("mode", "year")
    segmented = data["segmented"]
    axis = "study design" if mode == "design" else "year of publication"
    lines = [
        f"**Fig. 1 | Composition of the evidence base.** Cited sources by {axis}"
        + (", segmented by how directly each addresses the review question." if segmented else "."),
        "",
    ]
    for (label, _), tally in zip(data["buckets"], data["counts"]):
        total = sum(tally.values())
        detail = ", ".join(
            f"{v} {RELEVANCE_LABELS.get(k, k).lower()}" for k, v in tally.items() if v
        )
        lines.append(f"- {label}: {total} ({detail})")
    return "\n".join(lines)


def _methods_markdown(
    provenance: dict | None, screened: int, n_cited: int, topic: str,
    n_full_cited: int | None = None, n_retracted: int = 0,
) -> list[str]:
    parts = _methods_paragraphs(
        provenance, screened, n_cited, topic,
        n_full_cited=n_full_cited, n_retracted=n_retracted,
    )
    return [
        "## Methods",
        "",
        f"**Search strategy.** {parts['search']}",
        "",
        f"**Evidence handling.** {parts['handling']}",
        "",
        f"**Generation.** {parts['generation']}",
        "",
    ]


def _assessment_markdown(cited, counts, verification, style_report=None) -> list[str]:
    parts = _assessment_paragraphs(cited, counts, verification, style_report)
    lines = ["## Evidence assessment", "", parts["opening"], ""]
    lines += ["**Limitations.** " + " ".join(parts["limitations"]), ""]
    return lines


def _markdown_disclaimer(topic: str, article: dict, cited: list[Paper]) -> str:
    read = _read_phrase(cited)
    if _is_clinical(topic, article):
        return (
            "**Not medical or clinical advice.** Machine-generated summary of the cited "
            f"journal articles, {read}, for background only — not a substitute for "
            "professional judgement, primary sources, or clinical guidelines. Verify "
            "every claim, figure, and dose against the cited papers."
        )
    return (
        f"Machine-generated from the cited articles, {read}. "
        "Follow the source links before relying on any specific claim."
    )


# --------------------------------------------------------------------------
# published reviews index
# --------------------------------------------------------------------------

_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence briefings</title>
<script>
  (function() {{
    var saved = localStorage.getItem('articlegen_theme') || 'light';
    var dark = saved === 'dark' || (saved === 'system' && window.matchMedia
      && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }})();
</script>
<style>
  :root, html[data-theme="light"] {{ --bg:#fff; --ink:#14171a; --muted:#6a7178; --accent:#17457f; --link:#1b5aa8; --rule:#dfdfda; --card:#f7f7f5; color-scheme: light; }}
  html[data-theme="dark"] {{ --bg:#14150f; --ink:#f2f1ea; --muted:#9d9d94; --accent:#9dc4f5; --link:#86b6ef; --rule:#33342f; --card:#1e1f1c; color-scheme: dark; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; line-height:1.5; }}
  main {{ max-width:clamp(44rem, 72vw, 58rem); margin:0 auto; padding:2.5rem 1.25rem 5rem; }}
  .header-row {{ display:flex; align-items:baseline; justify-content:space-between;
    border-top:3px solid var(--ink); padding-top:0.8rem; margin-bottom:0.3rem; }}
  h1 {{ font-size:1.3rem; margin:0; letter-spacing:-0.01em; }}
  .theme-toggle-btn {{ background:transparent; border:1px solid var(--rule); color:var(--ink);
    border-radius:2px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.78rem; }}
  .sub {{ color:var(--muted); margin:0 0 2rem; font-size:0.8rem; }}
  ul {{ list-style:none; margin:0; padding:0; }}
  li {{ border-bottom:1px solid var(--rule); padding:0.9rem 0; }}
  li a {{ color:var(--ink); text-decoration:none; font-size:1.02rem; font-weight:600;
    font-family:"Iowan Old Style","Charter",Georgia,serif; }}
  li a:hover {{ color:var(--link); }}
  .meta {{ color:var(--muted); font-size:0.78rem; margin-top:0.3rem; }}
  .meta a {{ color:var(--link); text-decoration:none; }}
  .empty {{ color:var(--muted); }}
  .idx-disclosure {{ font-size:0.92rem; font-weight:600; color:var(--ink);
    background:var(--card); border-left:3px solid var(--ink);
    padding:0.6rem 0.8rem; margin:0 0 1rem; line-height:1.45; }}
</style>
</head>
<body>
<main>
  <div class="header-row">
    <h1>Evidence briefings</h1>
    <button class="theme-toggle-btn" onclick="toggleIndexTheme()"><span id="idxThemeIcon">&#9789;</span> Theme</button>
  </div>
  <p class="idx-disclosure">Every briefing listed here was written by a language
  model from published research. No human author wrote or checked any of them,
  and none has been peer reviewed.</p>
  <p class="sub">{count} briefing(s) · newest first</p>
  <ul>
    {items}
  </ul>
</main>
<script>
function toggleIndexTheme() {{
  var cur = document.documentElement.getAttribute('data-theme');
  var effective = cur || 'light';
  var next = effective === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('articlegen_theme', next);
  updateIndexThemeIcon();
}}
function updateIndexThemeIcon() {{
  var icon = document.getElementById('idxThemeIcon');
  if (!icon) return;
  var cur = document.documentElement.getAttribute('data-theme');
  var effective = cur || 'light';
  icon.innerHTML = effective === 'dark' ? '&#9789;' : '&#9788;';
}}
document.addEventListener('DOMContentLoaded', updateIndexThemeIcon);
</script>
</body>
</html>
"""


def _draft_title(html_path: str) -> str:
    try:
        with open(html_path, encoding="utf-8", errors="replace") as f:
            match = _TITLE_RE.search(f.read())
        if match:
            return html.unescape(match.group(1).strip())
    except Exception:
        pass
    return os.path.basename(html_path)


def build_index(drafts_dir: str) -> str:
    """Scan a drafts folder and write drafts/index.html listing every draft, newest first."""
    html_files = [
        p for p in glob.glob(os.path.join(drafts_dir, "*.html"))
        if os.path.basename(p) != "index.html"
    ]
    html_files.sort(key=os.path.getmtime, reverse=True)

    items = []
    for path in html_files:
        name = os.path.basename(path)
        title = html.escape(_draft_title(path))
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        md_name = name[:-5] + ".md"
        md_link = (
            f' · <a href="{html.escape(md_name, quote=True)}">markdown</a>'
            if os.path.exists(os.path.join(drafts_dir, md_name))
            else ""
        )
        items.append(
            f'<li><a href="{html.escape(name, quote=True)}">{title}</a>'
            f'<div class="meta">{_au_datetime(mtime)} · '
            f'<a href="{html.escape(name, quote=True)}">open</a>{md_link}</div></li>'
        )

    body = "\n    ".join(items) if items else '<li class="empty">No drafts yet.</li>'
    index_html = _INDEX_TEMPLATE.format(count=len(html_files), items=body)
    index_path = os.path.join(drafts_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    return index_path

"""Deterministic grounding check for the statistics in a generated article.

The writer sees titles, abstracts, plus — for open-access sources — the same
full-text excerpts that `sources.full_text_excerpts` yields. It is prone to
stating precise figures (effect sizes, confidence intervals, risk ratios) it was
never shown. This pass extracts the figures the article asserts and checks
whether each appears in that same material. Anything missing is surfaced to the
reader as "verify against the full text" rather than trusted.

The haystack is exactly what the writer was shown (title, abstract, and shown
full-text excerpt), not the whole retrieved paper: searching text the writer
never saw would let a figure recalled from training pass as grounded.
`full_text_excerpts` is deterministic over the papers, so both sides derive the
same excerpts without anything recorded.

**A cited sentence is checked against its own sources only.** Searching every
source and passing on any hit is what this used to do, and it hid the failure
that most changes clinical meaning: a real number lifted from paper 12 and
credited in the prose to paper 3 passed silently as verified. That case is now
reported separately as `misattributed` — the figure is real, the attribution is
not. A sentence citing nothing has no attribution to break, so it is still
checked against everything.

It is intentionally deterministic — an LLM verifier can hallucinate agreement;
substring presence in the real source text cannot. The corollary is that the
check is deliberately generous about *form*: it asks whether the number is
there at all, not whether the surrounding words match. A missed figure is a
warning the reader never sees; a false flag is a wrong warning printed in the
article.
"""

from __future__ import annotations

import re

from .briefing import finding_texts
from .sources import Paper, full_text_excerpts

# Units that make a bare integer a clinical quantity rather than a year, a
# citation marker or ordinary prose. An integer without one is not extracted:
# checking every loose number produces noise, not grounding. Longer forms come
# first so the alternation does not clip them (`ng` before `ng/mL` would).
# Single letters (g, L, h) are left out — under IGNORECASE they match far too
# much for what they would buy.
_CLINICAL_UNITS = (
    r"ng/m[lL]|[mµn]mol/[lL]|mcg|µg|mg|kg|ng|mL|IU|lux|"
    r"minute|min|hour|day|week|month|year|"
    r"participant|patient|subject|children|child|adult|case|women|men|"
    r"session|episode|admission|arm"
)

# Decimals (0.90, -0.38, 4.91), ranges (4.4-5.2, -0.38--0.12), percentages (12%, 3.5%),
# sample sizes (n=60), effect metrics (d=-1.70, CI: 1.2-3.4), and integers carrying a
# clinical unit (7,000 lux, 15 minutes, 50 ng/mL, 441 participants, 25-mg). Plain years
# (2020-2026) are excluded.
#
# A hyphenated range is ONE quantity. Scanning "4.4-5.2" with the plain decimal
# alternative alone matched "4.4" and then "-5.2" -- a negative number the
# article never asserted and no source contains, so a real confidence interval
# printed a flag on its own second half (#189). This alternative is first in the
# alternation so it wins at the position where the range starts; `_found` then
# requires BOTH endpoints, which keeps the check strict about presence while
# staying generous about form (a source writing "4.4 to 5.2" still verifies).
_RANGE_RE_SRC = r"(?P<range>-?\d+\.\d+%?\s*[-–—]\s*-?\d+\.\d+%?)"

_FIGURE_RE = re.compile(
    _RANGE_RE_SRC
    + r"|-?\d+\.\d+%?"
    + r"|\b\d{1,3}%"
    + r"|\bn\s*=\s*\d+\b"
    + r"|\b(?:CI|p|d|OR|RR|HR)\s*=\s*-?\d+(?:\.\d+)?\b"
    + r"|\b(?P<quantity>\d[\d,]*\s*-?\s*(?:" + _CLINICAL_UNITS + r")s?)\b",
    re.IGNORECASE,
)

_RANGE_PARTS_RE = re.compile(r"^(?P<lo>-?\d+\.\d+%?)\s*[-–—]\s*(?P<hi>-?\d+\.\d+%?)$")

# Common publication years to skip during standalone integer extraction
_YEAR_RE = re.compile(r"\b(?:19\d\d|20[0-2]\d)\b")

# The number a quantity leads with, which is the part that has to be real.
_LEADING_NUMBER_RE = re.compile(r"^\d[\d,]*")


def _normalize(token: str) -> str:
    return re.sub(r"^[+\-\s]+|[+\-\s%]+$", "", token)


def _variants(text: str) -> tuple[str, str, str]:
    """(raw, spaces stripped, commas stripped) forms of a haystack.

    One quantity gets written all three ways: `n = 60` against `n=60`, `7,000`
    against `7000`. Stripping on the haystack side is cheaper than enumerating
    the spellings on the figure side.
    """
    return text, text.replace(" ", ""), text.replace(",", "")


def _found(figure: str, haystack: tuple[str, str, str], quantity: bool, is_range: bool = False) -> bool:
    """Is this figure present in this haystack?"""
    if is_range:
        m = _RANGE_PARTS_RE.match(figure.strip())
        if not m:
            return False
        lo = m.group("lo")
        hi = m.group("hi")
        return _found(lo, haystack, quantity=False, is_range=False) and _found(hi, haystack, quantity=False, is_range=False)

    raw, nospace, nocomma = haystack
    norm = _normalize(figure)
    if not norm:
        return False
    if norm in raw or f"{norm}%" in raw or norm.replace(" ", "") in nospace:
        return True
    if not quantity:
        return False
    # A quantity verifies on its number alone. The source may write the same
    # amount with a different unit form ("50 ng/mL" against "50 ng per mL", "15
    # minutes" against "15-min"), and demanding the words match would flag
    # figures that are in fact grounded. Digit boundaries stop 50 matching
    # inside 1950 or 0.50.
    number = _LEADING_NUMBER_RE.match(norm)
    if not number:
        return False
    digits = number.group(0).replace(",", "")
    return re.search(rf"(?<![\d.,]){re.escape(digits)}(?![\d,])", nocomma) is not None


def _article_sentences(article: dict) -> list[tuple[str, list[int]]]:
    """Yields (sentence_text, list_of_cited_source_indices)."""
    text = _article_text(article)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    for s in sentences:
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", s)]
        result.append((s, cites))
    return result


def _paper_haystack(paper: Paper, full_texts: dict[int, str], idx: int) -> str:
    # The title is part of what the writer was shown -- `writer._format_sources`
    # prints it above every abstract -- and it is where a headline effect size
    # often lives ("...reduces seclusion by 37%: a cluster RCT"). Leaving it out
    # printed a dagger on a figure the source states in its own title (#189).
    parts = [paper.title or "", paper.abstract or ""]
    if idx in full_texts:
        parts.append(full_texts[idx])
    return " ".join(parts)


def _article_text(article: dict) -> str:
    parts = [
        article.get("question", ""),
        article.get("answer", "") or article.get("abstract", "") or article.get("standfirst", ""),
        article.get("evidence_note", ""),
    ]
    fs = article.get("featured_study") or {}
    parts += [fs.get("method", ""), fs.get("results", "")]
    parts.extend(finding_texts(article))
    parts.extend(article.get("unknowns") or [])
    for section in article.get("sections", []):
        parts.extend(section.get("paragraphs", []))
        if section.get("pull_quote"):
            parts.append(section["pull_quote"])
    parts.extend(article.get("key_points") or article.get("key_takeaways") or [])
    return "\n".join(parts)


def check_statistics(article: dict, papers: list[Paper]) -> dict:
    """Check every figure the article asserts against the sources it credits.

    Returns:

    - `unverified` — figures found in none of the material the writer was shown.
    - `misattributed` — figures that are real, but appear only in a source other
      than the one their sentence cites.
    - `total` — figures checked. A figure credited to two different sources is
      two claims and counts twice.
    - `details` — structured list of flagged figure records with sentence
      context, for the statistics revision brief (#189).
    """
    shown = full_text_excerpts(papers)
    paper_map = {i: _paper_haystack(p, shown, i) for i, p in enumerate(papers, start=1)}
    everything = _variants(" ".join(paper_map.values()))

    # Sentences citing the same sources share a haystack; most articles cite the
    # same handful of papers repeatedly.
    local_cache: dict[tuple[int, ...], tuple[str, str, str]] = {}

    seen: set[tuple[str, tuple[int, ...]]] = set()
    unverified: list[str] = []
    misattributed: list[str] = []
    details: list[dict] = []
    total = 0

    for sentence, cites in _article_sentences(article):
        cited = tuple(sorted({i for i in cites if i in paper_map}))
        if cited:
            if cited not in local_cache:
                local_cache[cited] = _variants(" ".join(paper_map[i] for i in cited))
            local = local_cache[cited]
        else:
            # Nothing cited — no attribution to break, so the whole evidence
            # base is the right haystack.
            local = everything

        for match in _FIGURE_RE.finditer(sentence):
            text = match.group(0)
            if _YEAR_RE.fullmatch(text.strip()):
                continue
            norm = _normalize(text)
            if not norm:
                continue
            # Keyed on the attribution as well as the figure: the same number
            # credited to two different sources is two claims, and the second
            # one is exactly the case worth catching.
            if (norm, cited) in seen:
                continue
            seen.add((norm, cited))
            total += 1

            is_range = match.group("range") is not None
            quantity = match.group("quantity") is not None
            if _found(text, local, quantity, is_range):
                continue
            if cited and _found(text, everything, quantity, is_range):
                misattributed.append(text)
                details.append({
                    "figure": text,
                    "kind": "misattributed",
                    "sentence": sentence,
                    "cited": list(cited),
                })
            else:
                unverified.append(text)
                details.append({
                    "figure": text,
                    "kind": "unverified",
                    "sentence": sentence,
                    "cited": list(cited),
                })

    return {
        "unverified": unverified,
        "misattributed": misattributed,
        "total": total,
        "details": details,
    }


def revision_brief(verification: dict) -> str:
    """What to send back to the writer when figures did not check out.

    Deterministic string building, no LLM. Three fixes are allowed and they are
    named explicitly, because the fourth one -- inventing a source or a number
    that would make the sentence true -- is the failure this pass could
    otherwise cause.
    """
    details = list(verification.get("details") or [])
    if not details:
        for fig in verification.get("unverified") or []:
            details.append({"figure": fig, "kind": "unverified", "sentence": "", "cited": []})
        for fig in verification.get("misattributed") or []:
            details.append({"figure": fig, "kind": "misattributed", "sentence": "", "cited": []})

    lines = [
        "A deterministic check could not verify the following figures against the material "
        "the draft was written from. Fix each flagged sentence:",
        "",
    ]
    for d in details:
        fig = d["figure"]
        kind = d["kind"]
        sent = d.get("sentence", "")
        if kind == "misattributed":
            desc = f"Figure {fig!r} was found in the sources, but NOT in the source(s) cited by this sentence."
        else:
            desc = f"Figure {fig!r} was NOT found in any abstract or retrieved full text."
        lines.append(f"- [{kind}] {desc}")
        if sent:
            lines.append(f"  Sentence: {sent}")

    lines.extend([
        "",
        "For each flagged sentence, apply EXACTLY one of these three permitted fixes:",
        "1. Delete the figure and keep the claim in words.",
        "2. Restate the quantity qualitatively (e.g., 'a substantial reduction', 'a minority of participants').",
        "3. For a misattributed figure, move the citation to the source that actually reports it, or delete the figure if uncertain.",
        "",
        "PROHIBITIONS:",
        "- You MUST NOT introduce any new number that is not already in the draft.",
        "- You MUST NOT add any new source.",
        "- You MUST NOT 'correct' a figure to a value you believe is right.",
        "- Leave every block the brief does not name out of your reply entirely; anything you omit is kept exactly as it is.",
        "- Every other '[N]' citation marker must survive exactly as it is and stay attached to the same claim.",
    ])
    return "\n".join(lines)

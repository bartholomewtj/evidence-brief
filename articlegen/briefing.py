"""Shape helpers for the briefing's structured blocks.

A briefing finding used to be one string. It is now an object — a one-sentence
`claim`, the `evidence` behind it with citations, and a `certainty` grade — so
the page can lead with the takeaway and show how sure it is. Manifests written
before that change still hold strings, and the style and figure gates read
plain text either way, so every reader of `findings` goes through here rather
than indexing the dicts itself.

`open_these` went the same way: an int list became objects carrying a `why`.
"""

from __future__ import annotations

import json
import re

# GRADE's four levels, in order. The writer is asked for one of these per
# finding and once overall; anything else is dropped rather than displayed.
CERTAINTY_LEVELS = ("high", "moderate", "low", "very low")

CERTAINTY_LABELS = {
    "high": "High certainty",
    "moderate": "Moderate certainty",
    "low": "Low certainty",
    "very low": "Very low certainty",
}


def certainty_level(value) -> str:
    """A GRADE level from whatever the model returned, or ''."""
    level = re.sub(r"[\s_-]+", " ", str(value or "")).strip().lower()
    level = level.replace(" certainty", "")
    return level if level in CERTAINTY_LEVELS else ""


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


# Writers like to close the evidence with the grade they were also asked for as
# a field ("... Certainty: moderate."). The page prints the grade as a badge, so
# the sentence is a duplicate; it is cut wherever a finding is read.
_TRAILING_GRADE = re.compile(
    r"\s*\(?(?:overall\s+)?certainty(?:\s+of\s+(?:the\s+)?evidence)?\s*(?:is|:|—|-)?\s*"
    r"(?:high|moderate|low|very\s+low)\s*\)?\s*\.?\s*$",
    re.I,
)


def _evidence(text) -> str:
    return _TRAILING_GRADE.sub("", " ".join(str(text or "").split())).strip()


def finding_parts(finding) -> tuple[str, str, str]:
    """(claim, evidence, certainty) for a finding of either shape.

    A legacy string finding has no claim and no grade: all of it is evidence.
    """
    if isinstance(finding, dict):
        return (
            _sentence(finding.get("claim")),
            _evidence(finding.get("evidence")),
            certainty_level(finding.get("certainty")),
        )
    return "", " ".join(str(finding or "").split()), ""


def normalize_briefing(article: dict) -> dict:
    """Tidy a freshly written briefing in place: findings with neither claim nor
    evidence are dropped, a restated grade is cut from the evidence, and grades
    are spelled the one way the page knows."""
    findings = []
    for f in article.get("findings") or []:
        if isinstance(f, dict):
            claim, evidence, level = finding_parts(f)
            if claim or evidence:
                findings.append({"claim": claim, "evidence": evidence, "certainty": level})
        elif str(f or "").strip():
            findings.append(f)
    article["findings"] = findings
    raw = article.get("certainty")
    if isinstance(raw, dict):
        article["certainty"] = {"level": certainty_level(raw.get("level")),
                                "reason": " ".join(str(raw.get("reason") or "").split())}
    return article


def finding_text(finding) -> str:
    """One finding as plain prose, claim first — what the gates read."""
    claim, evidence, _ = finding_parts(finding)
    return " ".join(p for p in (claim, evidence) if p)


def finding_texts(article: dict) -> list[str]:
    return [t for t in (finding_text(f) for f in (article or {}).get("findings") or []) if t]


def overall_certainty(article: dict) -> tuple[str, str]:
    """(level, reason) for the briefing as a whole, or ('', '')."""
    raw = (article or {}).get("certainty")
    if isinstance(raw, dict):
        level = certainty_level(raw.get("level"))
        return level, (" ".join(str(raw.get("reason") or "").split()) if level else "")
    return certainty_level(raw), ""


def open_these_items(article: dict) -> list[tuple[int, str]]:
    """(source index, why) for each recommended paper; `why` is '' on old drafts."""
    items: list[tuple[int, str]] = []
    for raw in (article or {}).get("open_these") or []:
        if isinstance(raw, dict):
            idx, why = raw.get("source_index"), " ".join(str(raw.get("why") or "").split())
        else:
            idx, why = raw, ""
        if isinstance(idx, int) and not isinstance(idx, bool):
            items.append((idx, why))
    return items


# The split the renderer bolds: the answer's first sentence is the verdict.
# Abbreviations that end in a full stop must not end it.
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|vs|approx|ca|cf|al|Dr|Fig|No)\.", re.I)
_FIRST_SENTENCE = re.compile(r"^(.+?[.!?])(?:\s+(?=[A-Z0-9\"“‘(]))(.*)$", re.S)


def split_verdict(answer: str) -> tuple[str, str]:
    """(first sentence, the rest) of the answer paragraph."""
    text = " ".join(str(answer or "").split())
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    m = _FIRST_SENTENCE.match(protected)
    if not m:
        return text, ""
    return m.group(1).replace("\x00", "."), m.group(2).replace("\x00", ".")


def _json_object(text) -> dict | None:
    text = str(text or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def revised_findings(old: list, replacement: list[str]) -> list:
    """Merge a text-only revision of `findings` back into structured findings.

    The patch schema returns strings. For a structured finding the revision is
    asked to write `CLAIM || EVIDENCE`; a string without the separator keeps the
    old claim when it still opens with it, and is otherwise all evidence. The
    certainty grade is not editable by a prose revision, so it carries over by
    position.
    """
    if not any(isinstance(f, dict) for f in old or []):
        return list(replacement)
    out = []
    for i, text in enumerate(replacement):
        prior = old[i] if i < len(old) and isinstance(old[i], dict) else {}
        # Shown the draft as JSON, a model will often hand a finding back the
        # same way — as an object, or as that object serialised into the
        # string. Measured: two of five figure revisions did, and the raw JSON
        # landed on the page as the finding's evidence.
        obj = text if isinstance(text, dict) else _json_object(text)
        if obj is not None:
            out.append({
                "claim": _sentence(obj.get("claim")) or prior.get("claim", ""),
                "evidence": " ".join(str(obj.get("evidence") or "").split())
                or prior.get("evidence", ""),
                "certainty": prior.get("certainty", "") or certainty_level(obj.get("certainty")),
            })
            continue
        text = " ".join(str(text).split())
        if "||" in text:
            claim, evidence = (p.strip() for p in text.split("||", 1))
        else:
            old_claim = _sentence(prior.get("claim"))
            first, rest = split_verdict(text)
            if old_claim and text.startswith(old_claim):
                claim, evidence = old_claim, text[len(old_claim):].strip()
            elif rest and "[" not in first:
                # The claim was reworded: an uncited opening sentence is it.
                claim, evidence = first, rest
            else:
                claim, evidence = prior.get("claim", ""), text
        out.append({"claim": claim, "evidence": evidence,
                    "certainty": prior.get("certainty", "")})
    return out

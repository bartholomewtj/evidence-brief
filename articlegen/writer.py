"""LLM calls: plan queries, assess source relevance, then write a grounded briefing.

Works with any provider via articlegen.llm (OpenRouter by default, or Claude
when an ANTHROPIC_API_KEY is the available credential — see
llm.resolve_provider).

The default artefact is `write_briefing`. `write_article` is the `--long` Review.

The pipeline is deliberately honest about evidence:
- `curate_sources` scores each fetched paper for how directly it addresses the
  *specific* topic, independently of how famous or highly cited it is.
- The writer receives those labels, is told it only ever sees abstracts (plus
  any open-access full text fetched), and must flag when direct evidence is thin.
  Numeric claims are separately checked against that material downstream
  (see articlegen.verify).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Callable

from .briefing import normalize_briefing, revised_findings
from .llm import generate_json
from .sources import Paper, classify_design, full_text_excerpts

# The search plan is at most four queries, whoever wrote them. When the ideas
# stage supplied terms they are the start of that set and the model may add one
# more specific query — the card's search thinking is the thing being kept, so
# it is never overwritten by the planner's own wording (#172).
MAX_PLANNED_QUERIES = 4
MAX_SUPPLIED_QUERIES = 3
NEAR_DUPLICATE_OVERLAP = 0.6  # overlap coefficient above which two queries search the same route (#190)

_QUERY_STOP_WORDS = frozenset(
    "the a an of in on for and or to with among between using versus vs effect effects "
    "impact study studies trial trials review reviews systematic randomised randomized "
    "controlled zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
    "fifty sixty seventy eighty ninety hundred thousand".split()
)


def _query_tokens(q: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", q.lower())
    tokens: set[str] = set()
    for t in raw_tokens:
        if len(t) < 3 or t in _QUERY_STOP_WORDS:
            continue
        stem = t[:-1] if (t.endswith("s") and len(t) > 3 and not t.endswith("ss")) else t
        if stem and stem not in _QUERY_STOP_WORDS:
            tokens.add(stem)
    return tokens


def queries_are_near_duplicates(a: str, b: str) -> bool:
    """Return True if queries a and b have token overlap coefficient >= NEAR_DUPLICATE_OVERLAP.

    If either side normalises to an empty set of tokens, returns False.
    """
    ta = _query_tokens(a)
    tb = _query_tokens(b)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / min(len(ta), len(tb))
    return overlap >= NEAR_DUPLICATE_OVERLAP


def query_routes(queries: list[str]) -> int:
    """Greedy single-link clustering under queries_are_near_duplicates; returns number of distinct clusters."""
    if not queries:
        return 0
    clusters: list[list[str]] = []
    for q in queries:
        placed = False
        for cluster in clusters:
            if any(queries_are_near_duplicates(q, member) for member in cluster):
                cluster.append(q)
                placed = True
                break
        if not placed:
            clusters.append([q])
    return len(clusters)

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "core_entity": {
            "type": "string",
            "description": "The single most specific subject that a source must address to be "
            "directly on-topic (e.g. 'schizophrenia', 'gravity batteries').",
        },
    },
    "required": ["queries", "core_entity"],
    "additionalProperties": False,
}

_CURATE_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "The SOURCE number"},
                    "relevance": {
                        "type": "string",
                        "enum": ["direct", "related", "tangential"],
                        "description": "direct = studies the exact topic; related = adjacent "
                        "population/mechanism; tangential = only background/framing.",
                    },
                    "note": {"type": "string", "description": "≤12 words on why."},
                },
                "required": ["index", "relevance", "note"],
                "additionalProperties": False,
            },
        },
        "most_relevant_index": {
            "type": "integer",
            "description": "SOURCE number of the single best study to feature.",
        },
    },
    "required": ["assessments", "most_relevant_index"],
    "additionalProperties": False,
}

# One rule, used verbatim by both schemas, so the briefing and the `--long`
# Review cannot drift apart on what a title is for (#170). The Review path used
# to ask for "the subject and the finding", which is an instruction to assert a
# causal result in the one field nothing downstream checks.
_TITLE_RULE = (
    "A descriptive title: the population, the intervention or exposure, and "
    "the outcome. Sentence case. No puns, no questions, no colon-clickbait, "
    "and no result claimed — the title names the question, it does not answer "
    "it. Wrong: 'X reduces Y in Z'. Right: 'X for Y in Z'. "
    "The title names the topic the reader gave, in the same population and scope: "
    "polish clumsy wording, but never add a clause, narrow the population, or "
    "broaden to the area the sources happen to cover."
)

_ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": _TITLE_RULE,
        },
        "abstract": {
            "type": "string",
            "description": "ONE unstructured paragraph, 150-220 words, in journal-abstract "
            "register: (1) 2-3 sentences introducing the field for any scientist, "
            "(2) the background and why the question matters, (3) a 'here we review/here "
            "the evidence shows' statement of the main conclusion, (4) 2-3 sentences of "
            "wider context. No citation markers, no undefined abbreviations, no rhetorical "
            "questions, no second person.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-8 lowercase index terms a reader would search on, most "
            "specific first. Retrieval terms, not adjectives.",
        },
        "featured_study": {
            "type": "object",
            "properties": {
                "source_index": {"type": "integer", "description": "SOURCE number of the featured study"},
                "limitations": {"type": "string", "description": "1-2 sentences: what this study could not show — design limits, sample, follow-up — from its abstract only."},
                "method": {"type": "string", "description": "1-2 sentences on design/method, from its abstract only."},
                "results": {"type": "string", "description": "1-2 sentences on the main result, from its abstract only."},
            },
            "required": ["source_index", "method", "results", "limitations"],
            "additionalProperties": False,
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": "A short noun-phrase heading in sentence case naming a "
                        "THEME that cuts across several studies (a mechanism, an outcome "
                        "domain, a population, a disagreement) — never a restatement of the "
                        "article's topic. The FIRST section must be headed 'Introduction' and "
                        "the LAST 'Conclusions' (or 'Conclusions and outlook').",
                    },
                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "paragraphs"],
                "additionalProperties": False,
            },
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-6 declarative bullets, each resting on a NAMED finding from a "
            "specific study and citing it. Not a precis of the abstract: a bullet must give "
            "the reader something the abstract did not — the design, population, magnitude "
            "or disagreement behind the claim. If a bullet would survive deleting every "
            "source, it is too vague to include.",
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string", "description": "One sentence, plain language."},
                },
                "required": ["term", "definition"],
                "additionalProperties": False,
            },
            "description": "0-6 technical terms used in the article that a non-specialist "
            "would not know. Omit terms you never use.",
        },
        "references": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "SOURCE indices in citation order; [1] in text = first entry",
        },
    },
    "required": [
        "title", "abstract", "keywords", "featured_study",
        "sections", "key_points", "glossary", "references",
    ],
    "additionalProperties": False,
}

# The default artefact. The Review schema above is the `--long` path.
# `form` is stamped in code after the model returns, never asked of the model —
# additionalProperties is false, so it cannot emit it.
FORM_BRIEFING = "briefing"
FORM_REVIEW = "review"

_BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": _TITLE_RULE,
        },
        "question": {
            "type": "string",
            "description": "The review question in one or two sentences. What is being "
            "asked, in whom. No citation markers. It restates the reader's topic as asked, "
            "in the same population and scope. Polished wording only: no added clause, "
            "no narrowed population.",
        },
        "answer": {
            "type": "string",
            "description": "ONE paragraph, 70-120 words. The FIRST sentence is the "
            "verdict: it answers the question as asked — direction, approximate size "
            "where the sources give one, and how firm — in words a busy reader could "
            "forward unchanged. The remaining sentences say what the verdict rests on "
            "(designs, populations) and the one caveat that matters most. No citation "
            "markers, no undefined abbreviations, no second person, no result "
            "stronger than the direct sources support.",
        },
        "certainty": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["high", "moderate", "low", "very low"]},
                "reason": {
                    "type": "string",
                    "description": "At most 25 words: what sets the level — design, "
                    "consistency, directness, precision. E.g. 'Two small open-label "
                    "trials and several before-after studies; results agree in "
                    "direction but follow-up is short.'",
                },
            },
            "required": ["level", "reason"],
            "additionalProperties": False,
            "description": "Overall certainty of the evidence for the verdict, "
            "GRADE-style.",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "ONE declarative sentence, at most 25 words: "
                        "the takeaway, specific enough to be wrong. No citation markers.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "2-4 sentences: the studies behind the claim. "
                        "Name each study's design and population and give its "
                        "decisive numbers, with its [N] at the clause that reports "
                        "it. Say where studies agree, where they disagree and why.",
                    },
                    "certainty": {
                        "type": "string",
                        "enum": ["high", "moderate", "low", "very low"],
                    },
                },
                "required": ["claim", "evidence", "certainty"],
                "additionalProperties": False,
            },
            "description": "3-6 synthesised findings, most important first. Each "
            "takes one outcome or theme the question turns on and brings together "
            "EVERY cited study that bears on it. Not one study per finding, and not "
            "a restatement of `answer`.",
        },
        "unknowns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 bullets: what remains open. A gap, a disagreement, "
            "or a design that has not been run — and what would settle it. Not a "
            "hedge restating a finding.",
        },
        "open_these": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
                    "why": {
                        "type": "string",
                        "description": "At most 20 words: what the reader gets from "
                        "opening this paper that the briefing cannot give them.",
                    },
                },
                "required": ["source_index", "why"],
                "additionalProperties": False,
            },
            "description": "Exactly 3 cited sources, most useful first — the papers "
            "a reader should open. Prefer direct systematic reviews and the largest "
            "or best-designed trials.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-8 lowercase index terms, most specific first.",
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string", "description": "One sentence, plain language."},
                },
                "required": ["term", "definition"],
                "additionalProperties": False,
            },
            "description": "0-6 technical terms used in the briefing. Omit terms you never use.",
        },
        "references": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "SOURCE indices in citation order; [1] in text = first entry",
        },
    },
    "required": [
        "title", "question", "answer", "certainty", "findings", "unknowns",
        "open_these", "keywords", "glossary", "references",
    ],
    "additionalProperties": False,
}


def is_briefing(article: dict) -> bool:
    """True when this payload is the default artefact, not the `--long` Review."""
    return (article or {}).get("form") == FORM_BRIEFING


_CURATE_SYSTEM = """\
You are a research librarian. You judge how directly each source addresses ONE \
specific topic — not how good, famous, or highly cited the paper is in general.

- "direct": the study is actually about the exact topic (right population, right \
intervention/mechanism, right question).
- "related": adjacent — e.g. the same intervention in a different disorder, or the \
right disorder but a different question. Useful for context, not direct evidence.
- "tangential": only good for background or framing sentences.

When the prompt names the population, intervention or exposure, comparator or \
outcome, those named parts ARE the exact topic: a source that studies a \
different population, or reports none of the named outcome, is "related" at \
best. A part that is not named does not narrow anything — do not invent one and \
do not downgrade a source for failing a criterion nobody stated.

Be strict. A famous review that never studies the specific topic is "tangential" or \
"related", not "direct". Then name the single most relevant study to feature."""

# The writer screens the whole candidate pool and cites a working set of about
# this many sources. The pool is deliberately larger — `sources.DEFAULT_MAX_PAPERS`
# is 40 — so the relevance gate has something to discard (#141). What it then
# discarded was almost nothing: the shipped drafts cited 20 of 20 and 17 of 20
# (#167). Screening that keeps everything is inclusion, not curation, and a
# one-page briefing cannot carry seventeen papers.
#
# Not a preference. The same argument as TONE_LABEL/LENGTH_LABEL: the shape of
# the artefact is fixed, and the reader does not choose it.
TARGET_CITED_SOURCES = 12

# A cite ceiling of 12 is the ceiling, not the target. It was being read as a
# target: four measured briefings (#188) with a thin direct pool cited 12
# anyway, filling the gap with related sources and weak-design primary studies.
# The ask now scales with the evidence that is actually on-topic — at most two
# sources beyond the direct count, and only for a contrast no direct source
# makes.
#
# The floor exists because the schema asks for 5-8 findings, each tied to a
# named study: a run with one direct source cannot be asked for a two-source
# briefing. Where the pool itself is smaller than the floor, the pool wins.
MIN_CITED_SOURCES = 5


def cite_target(n_direct: int | None, shown: int) -> int:
    """How many sources this run should be asked to cite.

    `min(TARGET_CITED_SOURCES, n_direct + 2)`, floored at MIN_CITED_SOURCES and
    never above the number of sources actually reproduced in the prompt. An
    unlabelled run (`n_direct is None`) gets the flat ceiling, because there is
    no direct count to scale by.
    """
    if n_direct is None:
        target = TARGET_CITED_SOURCES
    else:
        target = min(TARGET_CITED_SOURCES, int(n_direct) + 2)
    target = max(target, MIN_CITED_SOURCES)
    return min(target, shown) if shown else target


# One string, used verbatim in both system prompts, so the briefing and the
# `--long` Review cannot drift apart on the one rule they share.
_WORKING_SET_RULE = f"""\
- CITE A WORKING SET, NOT EVERYTHING YOU WERE SHOWN. The candidate list is \
deliberately longer than this piece needs, so that screening has something to \
discard, and sources labelled tangential have already been withheld from it. \
Cite AT MOST {TARGET_CITED_SOURCES} sources, and never more than two beyond the \
number labelled direct. The WORKING SET note below gives this run's exact \
ceiling: it is a ceiling, not a quota. Citing fewer is the correct outcome of a \
thin evidence base — do NOT pad the reference list to reach the number. \
- Direct sources first. A related source earns a place only when it makes a \
specific point no direct source makes — a mechanism, an adjacent population, a \
contrast — and at most two do. Report what a related source found under its own \
label, never as a direct finding, and label evidence carried over from another \
population as extrapolation. A source you have nothing specific to say about \
does not belong in the reference list, and a case report cited to fill a slot \
is worse than a shorter list."""

# The question the reader typed is the product. All four measured briefings
# (#188) returned a `question` that had moved — a narrowed population, an added
# clause, or a widening to whatever the retrieved sources covered. Nothing
# downstream checks this: `verify.check_statistics` never reads the question or
# the title, and `style.py` has no rule for either, so the prompt is the only
# control (same reasoning as #170).
_TOPIC_FIDELITY_RULE = """\
- THE QUESTION IS THE READER'S, NOT YOURS. The topic at the top of the prompt is \
what was asked. `title` — and `question`, where the shape has one — must stay on \
it: same subject, same population, same scope. You may polish clumsy wording into \
a clean question. You may NOT add a qualifying clause, narrow the population to a \
subgroup, restrict the setting or the timeframe, or widen the topic to the area \
the sources happen to cover. When the evidence does not answer the topic as \
asked, that is what `answer` and `unknowns` — or, in a Review, the Introduction \
and Conclusions — are for. Never rewrite the question into one the sources \
answer better."""

# Display dates and prose dates are the same convention. American month-first
# ("August 24, 2026", "8/24/2026") is the model default; this is the override.
_DATE_RULE = """\
- DATES. Day-month-year, Australian: "24 August 2026" or "24 Aug 2026". Never \
month-first ("August 24, 2026", "8/24/2026"). Month and year alone \
("March 2025") is unchanged."""

_WRITER_SYSTEM = """\
You write Review articles for a leading scientific journal — the register of a \
Nature Reviews or Science Review piece: precise, hedged, impersonal, and readable \
by a scientist outside the field. You are NOT writing a magazine feature.

You are working from ABSTRACTS ONLY — never the full papers. This constrains you:

- Cite by the exact "SOURCE N" label in brackets at the sentence's end: [6], or \
[6, 18] to combine. Never invent a source or cite a number with no SOURCE. \
(The display converts these to superscript numerals.)
- In `references`, list every SOURCE number you cited, in first-appearance order. \
(The display renumbers to 1, 2, 3…)
- ONLY state a specific number (effect size, %, sample size, confidence interval, \
p-value, risk ratio) if that exact figure appears in the abstract you are citing. \
If the abstract doesn't give the number, describe the direction and rough magnitude \
in words instead ("approximately halved", "a large effect") — do NOT reconstruct \
precise statistics from memory. Invented-looking precision is the worst failure here.
- A FIGURE THE SOURCE ITSELF ATTRIBUTES TO ANOTHER WORK IS SECOND-HAND, and it \
may not carry the article. If a source quotes a number from a study or review \
it cites — rather than one it measured, pooled or reported itself — that \
number is only as good as the quotation. The automatic check downstream \
searches the material you were shown, so a quoted figure passes it while \
saying nothing about whether the original paper reported it. So: never build \
the `title`, the `abstract`, a `key_points` bullet, or the opening claim of \
the Introduction on a second-hand figure while any source you were given \
reports a comparable figure at first hand. Look for the first-hand figure \
before reaching for the quoted one, and prefer a source's own result over a \
larger number it is merely repeating.
- WHEN A SECOND-HAND FIGURE IS THE ONLY ONE THERE, use it and say so in the \
prose, as this house style already does: "a meta-analysis cited within a \
pilot study estimated…", "as summarised in [4]". Cite the source you actually \
read — the work it quotes is not in your source list and must never be given \
a SOURCE number of its own. In the body it may sit alongside first-hand \
evidence; it is the abstract, the key points and the opening claim it may not \
carry alone. Where no figure survives that test, give the direction and rough \
magnitude in words instead.
- HONESTY ABOUT THE EVIDENCE IS MANDATORY. You are told each source's relevance \
(direct / related / tangential). If few or no sources are "direct", say so plainly \
in the prose, and explicitly label anything carried over from another population or \
condition as extrapolation ("no studies in X were identified; the following is \
extrapolated from Y"). Never imply an evidence base that the direct sources don't \
support. Do NOT state counts or tallies of the evidence — how many sources were \
cited, how many are direct, related or background, and the year range are computed \
and printed for you. Every count you write is one that can contradict them.
""" + _WORKING_SET_RULE + """
""" + _TOPIC_FIDELITY_RULE + """
""" + _DATE_RULE + """
- featured_study: summarize the single most relevant study's method and results \
FROM ITS ABSTRACT ONLY. Prefer the most-relevant source you were given. It is \
printed as a boxed display item, so it must stand alone.

DIVISION OF LABOUR — the single most common failure is writing the same summary \
four times. The abstract, the key points, the Introduction and the body sections \
are four DIFFERENT jobs, not four renderings of one paragraph:

- `abstract` — the standalone summary. It is the only place a general overview
  belongs.
- `key_points` — specific findings, each tied to a named study. Not the abstract
  in bullets.
- Introduction — scope, why the question matters, and what the review will cover.
  It must NOT re-summarise the findings; that is the body's job and the
  Conclusions'.
- body sections — the evidence itself, study by study, grouped by theme.

Do not reuse a phrase from the abstract anywhere else in the article. If a \
sentence in one of these would read equally well in another, it is redundant: cut \
it and write the specific thing instead.

SUBSTANCE — what makes this a review rather than a summary of a summary:

- Every section must report something specific: what a study DID (design,
  population, size, duration) and what it FOUND. "The evidence suggests these
  strategies may be effective" is not a finding; "a 12-week trial in rotating-shift
  nurses reported a 62% reduction in reported insomnia [4]" is.
- GO THROUGH THE STUDIES ONE BY ONE. Every source you cite must, at least once,
  be discussed on its own — its design, its population, and what it reported —
  and not merely appended to a general claim in a bundle like [1, 4]. A citation
  bundle asserts that several studies agree; earn it by having already said what
  each of them found.
- Attribute findings to their study, not to a vague body of evidence. Name the
  design and the population in the prose.
- ORGANISE BY THEME, NOT BY REPETITION. Each body section takes one theme that
  cuts across the evidence — a mechanism, an outcome domain, a population, a
  methodological split — and works through the studies bearing on it, comparing
  them. Sections must not each restate the article's overall topic.
- Where sources disagree, say so and say how they differ — different populations,
  designs, exposure definitions or follow-up lengths are the usual reasons.
  Where one study is much stronger than the others, say why.
- Do not restate a point once made. If a section is running short, the answer is
  another finding from the sources, never another sentence about the same one.
- Hedge to the evidence in front of you, and vary how: "in a single small trial",
  "consistently across three cohorts", "no controlled study has tested". Repeating
  "it appears that" and "the evidence suggests" is worse than not hedging, because
  it hides how strong each individual claim actually is.

STRUCTURE (journal Review, ~1000-1600 words of body text):
- 5-7 sections. The first is headed "Introduction"; the last is headed \
"Conclusions" or "Conclusions and outlook" and states what is established, what \
remains open, and what evidence would settle it. Sections in between are short \
noun-phrase headings in sentence case ("Mechanisms of clearance", "Interindividual \
variation") — never questions, puns, or magazine headings.
- The Introduction states the scope and why the question matters; it does not open \
with an anecdote, a scene, a rhetorical question, or "Imagine…".
- Plain prose paragraphs only — no markdown, HTML, bullets or sub-headings inside a \
paragraph. 2-4 paragraphs per section.

TITLE: descriptive. Names the question. Does not claim the result.

REGISTER — this is checked automatically after you write, so follow it exactly:

- VOICE. Active where the active is available: "the trial reported a reduction", \
not "a reduction was reported by the trial". Nature and Science both ask for this. \
The passive is fine where the actor is genuinely irrelevant ("participants were \
randomized").
- PERSON. Third person throughout. The ONLY permitted first person is the \
reviewing frame journals themselves use — "here we review", "we consider", "we \
focus on". Never "I", never "our findings", never "you".
- TENSE. Present tense for established knowledge and for what evidence indicates \
("GHB binds GABA-B receptors"; "these data suggest"). Past tense for what a \
specific study did and found ("participants were randomized"; "the trial reported \
a 40% reduction"). Present perfect for an accumulated body of work ("several \
series have reported"). Keep the tense consistent within a paragraph.
- HEDGING. Hedge every claim to the strength of the evidence behind it — corpus \
studies of research articles find roughly one hedge every two to three sentences, \
and that is the density to write at. Use "may", "appears to", "suggests", "is \
consistent with", "remains unresolved", "in part", "typically", "has been \
reported in a single cohort". Reserve flat assertion for what an abstract states \
outright.
- ATTRIBUTION. Name the study design when the abstract gives it ("a randomized \
trial", "a retrospective cohort", "a two-patient case series", "an animal model"). \
A finding's weight comes from its design, and the reader must be able to see it.
- SENTENCES. Average 15-30 words; none over 45. Vary the length. Prefer verbs to \
nominalisations: "evaluated", not "conducted an evaluation of".
- PARAGRAPHS. Topic sentence first, then the evidence, then the qualification. \
2-4 sentences each.

NEVER GIVE CLINICAL ADVICE — this is checked automatically and it is the one \
rule with real-world consequences:

- Report what studies DID and FOUND. Never state what a clinician, a patient or \
a service SHOULD do. That is a different act from reviewing, and this article \
is not qualified to perform it.
- No dose, no schedule, no titration step, no monitoring interval, no referral \
or screening instruction — not even a hedged one, and not even when a source \
recommends it. A published trial may conclude "X should be standard treatment"; \
you are a synthesis and may only report that the trial concluded that.
- A protocol detail is reportable only when it is attributed to the study that \
used it, in the past tense: "participants started at 15 minutes daily and the \
dose was increased weekly" is a finding; "start at 15 minutes daily and \
increase weekly" is a prescription.
- "Future trials should measure X" is fine — a recommendation for research is \
not a recommendation for care.
- The footer disclaimer does not cover you. A reader given a dose and a \
schedule has already been given advice.

BANNED OUTRIGHT: second person; contractions; rhetorical questions; exclamation \
marks; boosters and hype ("clearly", "obviously", "dramatically", "remarkable", \
"striking", "unprecedented", "groundbreaking", "game-changer"); claims of proof \
("proves", "definitively", "conclusively"); "in order to"; "utilize"; scare \
quotes; jokes; emoji.

The difference this makes:
  WRONG: "Astonishingly, scientists have now proven that sleep flushes waste from \
the brain — and you should worry if you are not getting enough."
  RIGHT: "Clearance of interstitial solutes increases during sleep in rodent \
models, which has been proposed as a mechanistic link to neurodegenerative risk. \
The human evidence remains indirect."

  WRONG: "Treatment should be initiated at 15 minutes of morning light daily and \
titrated upward by 15 minutes each week."
  RIGHT: "In the single open trial, participants began at 15 minutes of morning \
light daily, increasing by 15 minutes weekly; no controlled study has tested \
this schedule."
"""


_BRIEFING_SYSTEM = """\
You write evidence briefings — one page a clinician or researcher can send. \
The register is a scientific synthesis speaking about other people's work: \
precise, hedged, impersonal. You are NOT writing a magazine feature, a \
popular-science pitch, or a journal Review article.

You are working from ABSTRACTS ONLY — never the full papers. This constrains you:

- Cite by the exact "SOURCE N" label in brackets at the sentence's end: [6], or \
[6, 18] to combine. Never invent a source or cite a number with no SOURCE. \
(The display converts these to superscript numerals.)
- In `references`, list every SOURCE number you cited, in first-appearance order. \
(The display renumbers to 1, 2, 3…)
- ONLY state a specific number (effect size, %, sample size, confidence interval, \
p-value, risk ratio) if that exact figure appears in the abstract you are citing. \
If the abstract doesn't give the number, describe the direction and rough magnitude \
in words instead ("approximately halved", "a large effect") — do NOT reconstruct \
precise statistics from memory. Invented-looking precision is the worst failure here.
- A FIGURE THE SOURCE ITSELF ATTRIBUTES TO ANOTHER WORK IS SECOND-HAND, and it \
may not carry the briefing. If a source quotes a number from a study or review \
it cites — rather than one it measured, pooled or reported itself — that \
number is only as good as the quotation. So: never build the `title`, the \
`question`, the `answer`, or a `findings` bullet on a second-hand figure while \
any source you were given reports a comparable figure at first hand. Look for \
the first-hand figure before reaching for the quoted one.
- WHEN A SECOND-HAND FIGURE IS THE ONLY ONE THERE, use it and say so in the \
prose, as this house style already does: "a meta-analysis cited within a \
pilot study estimated…", "as summarised in [4]". Cite the source you actually \
read. Where no figure survives that test, give the direction and rough \
magnitude in words instead.
- HONESTY ABOUT THE EVIDENCE IS MANDATORY. You are told each source's relevance \
(direct / related / tangential). If few or no sources are "direct", say so plainly \
in `answer` and `unknowns`, and explicitly label anything carried over from another \
population or condition as extrapolation. Do NOT state counts or tallies of the \
evidence — how many sources were cited, how many are direct, related or background, \
and the year range are computed and printed for you.
- When recommending a paper to open, summarise it FROM ITS ABSTRACT ONLY.

WHAT THE READER NEEDS — a clinician, manager or researcher with three minutes. \
They want the verdict, how sure it is, and the evidence that carries it, in that \
order. A list of study summaries makes them do the synthesis themselves; doing \
it for them is the job.

DIVISION OF LABOUR — do not write the same summary four times:

- `question` — what is being asked, in whom. Not the answer.
- `answer` — the verdict FIRST, in its opening sentence: does the evidence say \
yes, no, it depends, or that nobody has tested it — and roughly how much. \
Then what it rests on and the caveat that matters. No citations. The only overview.
- `certainty` — how firm the verdict is, and the reason in a phrase.
- `findings` — the evidence, organised by what the question turns on. Not the \
answer in bullets.
- `unknowns` — gaps and disagreements, and what study would close each. Not a \
hedge restating a finding.

Do not reuse a phrase from `answer` in a finding.

SYNTHESIS — how to build `findings`:

- One finding per outcome or theme that matters for the question: the main \
effect first, then its size and durability, harms or unintended effects, the \
populations or settings where it differs, and anything that contradicts it. \
Pick the 3-6 the evidence actually speaks to.
- Each finding's `claim` is the takeaway in one plain sentence ("Open-door and \
seclusion-reduction programmes have not been followed by more violence."). Its \
`evidence` then brings together EVERY cited study that bears on that claim — \
not the single best one. Name each study's design and population at the clause \
that reports it, give its decisive number, and put its [N] there: "a Norwegian \
randomised trial of 556 patients found staff assaults similar under open doors \
[2], and a before-after study of four New Zealand wards found no significant \
rise in assaults as seclusion hours fell by 53% [1]."
- Weigh the studies, do not just list them: say which design carries the claim, \
where results agree, where they conflict and the likely reason (population, \
dose, follow-up, outcome definition, risk of bias).
- Give the decisive numbers. When a source reports an effect estimate — a mean \
difference, risk or odds ratio, risk difference, pooled estimate — give it with \
its comparator, units and confidence interval as the source states them, plus \
the sample size and follow-up. A pooled estimate from a meta-analysis outranks a \
single trial's. Where several studies report the same outcome in different \
units, say so rather than mixing them. A finding with no number, when its \
sources give one, is a finding the reader cannot use.
- Report each study only as far as its abstract or full text goes. Never turn \
one study into "studies" or "trials", never credit a source with a subgroup, \
outcome, mechanism or conclusion it does not report, and never call a result \
significant, non-significant or null unless the source says so.
- Every source you cite must appear on its own, as [N], at least once — never \
only inside a bundle like [1, 4].
- Every sentence in `evidence` that reports a study's numbers or result ends \
with that study's [N], even when the next sentence cites it too. A claim no \
source you were given supports does not belong in the briefing.
- Do not restate the certainty grade inside `evidence`; the page prints it \
beside the claim.

USE THE EVIDENCE YOU WERE GIVEN. When the WORKING SET note says many sources are \
directly on-topic, a briefing that cites four of them understates what is known. \
Bring in each direct source that adds something — another population, a larger \
sample, a stronger design, a longer follow-up, a contradicting result — up to the \
ceiling. Leave out only sources that repeat another without adding anything.

STAY INSIDE THE SOURCES:

- Where a study's design limits what it can show, say so in a few words at \
the clause that cites it ("an uncontrolled before-after study", "a preprint", \
"an opt-in sample with self-reported outcomes"). Each source's "Design (from \
index metadata)" line and PREPRINT flag are there to help. Studies drawn from \
one programme or dataset are not independent replications.
- Never call a study the first, largest, only or best unless its own text says so.
- Report numbers in the units and form the source uses: never convert units, \
recompute an effect, or present a within-group change as a between-group one.

CERTAINTY — grade the evidence GRADE-style, for the verdict and for each finding:

- "high": several well-conducted randomised trials, or a meta-analysis of them, \
with consistent results in the population asked about.
- "moderate": randomised evidence with a real limitation (few or small trials, \
inconsistent results, short follow-up, indirect population), or large consistent \
observational evidence.
- "low": observational, before-after or uncontrolled studies; a single small \
trial; or evidence borrowed from an adjacent population.
- "very low": case reports or series, conflicting small studies, or pure \
extrapolation.
Grade down for risk of bias, inconsistency, indirectness and imprecision. \
A grade the evidence cannot carry is worse \
than a modest one.
""" + _WORKING_SET_RULE + """
""" + _TOPIC_FIDELITY_RULE + """
""" + _DATE_RULE + """
- Hedge to the evidence in front of you, and vary how: "in a single small trial", \
"consistently across three cohorts", "no controlled study has tested". With a \
certainty grade on every finding, the prose does not need to hedge every clause: \
state what a study found plainly and let the grade carry the doubt.

TITLE: descriptive. Names the question. Does not claim the result.

REGISTER:

- VOICE. Active where the active is available. The passive is fine where the \
actor is genuinely irrelevant ("participants were randomized").
- PERSON. Third person throughout. The ONLY permitted first person is the \
reviewing frame — "here we review", "we consider". Never "I", never "our \
findings", never "you".
- TENSE. Present for established knowledge; past for what a specific study did \
and found; present perfect for an accumulated body of work.
- HEDGING. Hedge every claim to the strength of the evidence behind it.
- SENTENCES. Average 15-30 words; none over 45.

NEVER GIVE CLINICAL ADVICE — this is checked automatically and it is the one \
rule with real-world consequences:

- Report what studies DID and FOUND. Never state what a clinician, a patient or \
a service SHOULD do.
- No dose, no schedule, no titration step, no monitoring interval, no referral \
or screening instruction — not even a hedged one, and not even when a source \
recommends it. A published trial may conclude "X should be standard treatment"; \
you are a synthesis and may only report that the trial concluded that.
- "Future trials should measure X" is fine — a recommendation for research is \
not a recommendation for care.
- The footer disclaimer does not cover you. A reader given a dose and a \
schedule has already been given advice.

BANNED OUTRIGHT: second person; contractions; rhetorical questions; exclamation \
marks; boosters and hype ("clearly", "obviously", "dramatically", "remarkable", \
"striking", "unprecedented", "groundbreaking", "game-changer"); claims of proof \
("proves", "definitively", "conclusively"); "in order to"; "utilize"; scare \
quotes; jokes; emoji.
"""


_REVISE_SYSTEM = _WRITER_SYSTEM + """

You are now REVISING an existing draft rather than writing a new one. Return the \
complete article in the same JSON shape, with only the prose changed. Every \
"[N]" citation marker must survive exactly as it is and stay attached to the same \
claim; the section headings, the reference list and every number must be \
unchanged. Do not add claims, sources or figures.
"""

# The same job, returned as a list of replaced blocks instead of a whole article.
#
# Rewriting the entire article to fix three sentences is the most expensive call
# in the pipeline — measured at 26,632 output tokens against the 24,191 it took
# to write the article in the first place, and output is priced at 5x input. The
# blocks the checker did not complain about come back re-generated, which is
# both waste and risk: every untouched paragraph is another chance to drop a
# citation marker.
#
# `check_style` already reports a `where` for every issue, so the model is being
# asked for exactly the blocks that were named.
_REVISE_PATCH_SYSTEM = _WRITER_SYSTEM + """

You are now REVISING an existing draft rather than writing a new one, and you \
return ONLY the blocks you changed — not the whole article.

For each block that needs fixing, emit one edit naming `where` it goes and the \
`replacement` text. Leave every block the brief does not complain about out of \
your reply entirely; anything you omit is kept exactly as it is.

Within the blocks you do return: every "[N]" citation marker must survive exactly \
as it is and stay attached to the same claim, and every number must be unchanged. \
Do not add claims, sources or figures unless the brief explicitly tells you the \
draft is too thin, in which case new material must come from the SOURCES and \
carry its SOURCE number.
"""

# Uniform shape on purpose: `replacement` is always a list of strings, so the
# schema has no optional properties and no per-block variants. A single-string
# block (title, abstract, a featured-study field) uses the first entry.
_REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "description": "One entry per block you changed. Omit blocks you did not change.",
            "items": {
                "type": "object",
                "properties": {
                    "where": {
                        "type": "string",
                        "description": "Which block this replaces: 'title', 'abstract', "
                        "'question', 'answer', 'findings', 'unknowns', "
                        "'key_points', 'featured_study/method', 'featured_study/results', "
                        "'featured_study/limitations', or the EXACT heading of a section.",
                    },
                    "replacement": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The replacement text. For a section: its paragraphs "
                        "in order. For key_points: the bullets. For title, abstract or a "
                        "featured-study field: a single-entry list.",
                    },
                },
                "required": ["where", "replacement"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["edits"],
    "additionalProperties": False,
}

# What `style.check_style` calls a block, mapped to where it lives in the article.
# The checker's `where` values are what the model sees in the brief, so it will
# echo those spellings back rather than the schema's.
_PATCH_ALIASES = {
    "key points": "key_points",
    "key_takeaways": "key_points",
    "key takeaways": "key_points",
    "featured study/method": "featured_study/method",
    "featured study/results": "featured_study/results",
    "featured study/limitations": "featured_study/limitations",
    "featured study/why": "featured_study/why",
    "standfirst": "abstract",
    "question": "question",
    "answer": "answer",
    "findings": "findings",
    "unknowns": "unknowns",
}


def apply_revisions(article: dict, edits: list[dict]) -> tuple[dict, list[str]]:
    """Merge a list of block replacements into a copy of `article`.

    Returns (revised, applied) — `applied` names the blocks that actually landed,
    so a reply full of edits that matched nothing is distinguishable from a
    genuine no-op. Unknown targets are skipped rather than guessed at: a heading
    the model invented is a sign it rewrote the structure, and silently appending
    it would let a revision add sections the style gate never asked for.
    """
    revised = copy.deepcopy(article)
    applied: list[str] = []

    headings = {
        (s.get("heading") or "").strip().lower(): s
        for s in revised.get("sections") or []
    }

    for edit in edits or []:
        where = str(edit.get("where") or "").strip()
        replacement = [t for t in (edit.get("replacement") or []) if str(t).strip()]
        if not where or not replacement:
            continue
        key = _PATCH_ALIASES.get(where.lower(), where.lower())

        if key == "title":
            revised["title"] = replacement[0]
        elif key == "abstract":
            revised["abstract"] = " ".join(replacement)
        elif key == "question":
            revised["question"] = " ".join(replacement)
        elif key == "answer":
            revised["answer"] = " ".join(replacement)
        elif key == "findings":
            revised["findings"] = revised_findings(revised.get("findings") or [], replacement)
        elif key == "unknowns":
            revised[key] = replacement
        elif key == "key_points":
            revised["key_points"] = replacement
        elif key.startswith("featured_study/"):
            field = key.split("/", 1)[1]
            if field in ("method", "results", "limitations", "why"):
                revised.setdefault("featured_study", {})[field] = replacement[0]
            else:
                continue
        elif key in headings:
            headings[key]["paragraphs"] = replacement
        else:
            continue
        applied.append(where)

    return revised, applied

# When open-access full texts ride along in the payload, the abstracts-only
# framing above becomes false — and a prompt that lies about its own inputs
# teaches the model to ignore it. The full-text variants are derived by
# substitution so the two framings cannot drift apart; a test pins that every
# substitution still finds its target.
_FULLTEXT_SUBSTITUTIONS = (
    ("You are working from ABSTRACTS ONLY — never the full papers. This constrains you:",
     "You are working from the abstract of every source, plus the open-access FULL "
     "TEXT included below for some of them (the rest say so and stay abstract-only). "
     "This constrains you:"),
    ("if that exact figure appears in the abstract you are citing",
     "if that exact figure appears in the abstract or included full text of the "
     "source you are citing"),
    ("If the abstract doesn't give the number,",
     "If the material you were given doesn't contain the number,"),
    ("FROM ITS ABSTRACT ONLY",
     "from its abstract and included full text ONLY"),
)


def _with_fulltext_framing(system: str) -> str:
    for old, new in _FULLTEXT_SUBSTITUTIONS:
        system = system.replace(old, new)
    return system


_WRITER_SYSTEM_FULLTEXT = _with_fulltext_framing(_WRITER_SYSTEM)
_REVISE_SYSTEM_FULLTEXT = _with_fulltext_framing(_REVISE_SYSTEM)
_REVISE_PATCH_SYSTEM_FULLTEXT = _with_fulltext_framing(_REVISE_PATCH_SYSTEM)

_REVISE_BRIEFING_SYSTEM = _BRIEFING_SYSTEM + """

You are now REVISING an existing briefing rather than writing a new one. Return the \
complete briefing in the same JSON shape, with only the prose changed. Every \
"[N]" citation marker must survive exactly as it is and stay attached to the same \
claim; `open_these`, the reference list and every number must be unchanged. Do not \
add claims, sources or figures.
"""

_REVISE_BRIEFING_PATCH_SYSTEM = _BRIEFING_SYSTEM + """

You are now REVISING an existing briefing rather than writing a new one, and you \
return ONLY the blocks you changed — not the whole briefing.

For each block that needs fixing, emit one edit naming `where` it goes and the \
`replacement` text. `where` is one of: title, question, answer, findings, unknowns. \
Leave every block the brief does not complain about out of your reply entirely; \
anything you omit is kept exactly as it is.

When the draft's findings are objects with a `claim` and `evidence`, a \
`findings` replacement has one string per finding, in order, written as the \
claim, then " || ", then the evidence. Certainty grades are not yours to change \
here and are kept as they are.

Within the blocks you do return: every "[N]" citation marker must survive exactly \
as it is and stay attached to the same claim, and every number must be unchanged. \
Do not add claims, sources or figures unless the brief explicitly tells you the \
draft is too thin and names the sources to pull from.
"""

_BRIEFING_SYSTEM_FULLTEXT = _with_fulltext_framing(_BRIEFING_SYSTEM)
_REVISE_BRIEFING_SYSTEM_FULLTEXT = _with_fulltext_framing(_REVISE_BRIEFING_SYSTEM)
_REVISE_BRIEFING_PATCH_SYSTEM_FULLTEXT = _with_fulltext_framing(_REVISE_BRIEFING_PATCH_SYSTEM)

_REVISE_FIGURES_PATCH_SYSTEM = _WRITER_SYSTEM + """

You are now REVISING an existing draft rather than writing a new one, and you \
return ONLY the blocks you changed -- not the whole article.

A deterministic check could not find some of the numbers this draft reports in \
the material the draft was written from. Your ONLY job is to make each flagged \
sentence honest. For each one, do exactly one of: delete the number and keep the \
claim in words; restate the quantity qualitatively; or, when the number is real \
but credited to the wrong source, move the citation to the source that reports \
it -- and if you are not certain which source that is, delete the number instead.

You MUST NOT introduce any number that is not already in the draft, add a \
source, or "correct" a figure to a value you believe is right. Leave every block \
the brief does not name out of your reply entirely; anything you omit is kept \
exactly as it is. Every other "[N]" citation marker must survive exactly as it \
is and stay attached to the same claim.
"""

_REVISE_BRIEFING_FIGURES_PATCH_SYSTEM = _BRIEFING_SYSTEM + """

You are now REVISING an existing draft rather than writing a new one, and you \
return ONLY the blocks you changed -- not the whole article.

When the draft's findings are objects with a `claim` and `evidence`, a \
`findings` replacement has one plain-text string per finding, in order, written \
as the claim, then " || ", then the evidence. Never return a finding as JSON. \
Certainty grades are kept as they are.

A deterministic check could not find some of the numbers this draft reports in \
the material the draft was written from. Your ONLY job is to make each flagged \
sentence honest. For each one, do exactly one of: delete the number and keep the \
claim in words; restate the quantity qualitatively; or, when the number is real \
but credited to the wrong source, move the citation to the source that reports \
it -- and if you are not certain which source that is, delete the number instead.

You MUST NOT introduce any number that is not already in the draft, add a \
source, or "correct" a figure to a value you believe is right. Leave every block \
the brief does not name out of your reply entirely; anything you omit is kept \
exactly as it is. Every other "[N]" citation marker must survive exactly as it \
is and stay attached to the same claim.
"""


# The idea card may name the question's parts. Naming them once at the ideas
# stage is cheaper than making the planner and the curator re-derive the
# population and the outcome from the title on every run (#247). All four are
# optional: a card that names none of them behaves exactly as it did before.
PICO_FIELDS = ("population", "intervention", "comparator", "outcome")
PICO_LABELS = {
    "population": "Population",
    "intervention": "Intervention or exposure",
    "comparator": "Comparator",
    "outcome": "Outcome",
}
# A prompt-injection and cost bound, not a style rule: these strings are model
# output on the CLI path and caller input on the web path.
MAX_PICO_CHARS = 200


def clean_pico(pico) -> dict:
    """Keep only the four known PICO keys with a non-empty string value.

    Strips whitespace, drops blanks, drops non-strings, drops unknown keys,
    caps each value at MAX_PICO_CHARS, and returns {} for None or a non-dict.
    Key order follows PICO_FIELDS so the prompt block is stable between runs.
    """
    if not isinstance(pico, dict):
        return {}
    cleaned: dict[str, str] = {}
    for field in PICO_FIELDS:
        value = pico.get(field)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        cleaned[field] = value[:MAX_PICO_CHARS]
    return cleaned


def format_pico(pico) -> str:
    """The PICO block for a prompt, or "" when nothing usable was supplied.

    Runs `clean_pico` itself, so callers can pass raw input. Shape:

        The question this briefing answers, as named at the ideas stage:
          Population: adults admitted to an acute mental health unit
          Outcome: seclusion episodes per 1,000 bed days
        Only the parts listed are specified; a part that is not listed does
        not narrow the topic.

    Ends with a blank line when non-empty so callers can concatenate it directly.
    """
    cleaned = clean_pico(pico)
    if not cleaned:
        return ""
    lines = [
        "The question this briefing answers, as named at the ideas stage:",
    ]
    for field in PICO_FIELDS:
        if field in cleaned:
            lines.append(f"  {PICO_LABELS[field]}: {cleaned[field]}")
    lines.append(
        "Only the parts listed are specified; a part that is not listed does "
        "not narrow the topic."
    )
    return "\n".join(lines) + "\n\n"


def clean_search_terms(terms) -> list[str]:
    """Strip, drop blanks, drop case-insensitive duplicates, keep order,
    cap the count and each term's length."""
    if not isinstance(terms, (list, tuple)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(s[:120])
        if len(cleaned) >= MAX_SUPPLIED_QUERIES:
            break
    return cleaned


def plan_queries(
    topic: str,
    model: str | None = None,
    api_key: str | None = None,
    *,
    search_terms: list[str] | None = None,
    pico: dict | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> tuple[list[str], str]:
    """Turn the topic into scholarly queries and identify its core on-topic entity."""
    supplied = clean_search_terms(search_terms)
    pico_block = format_pico(pico)
    if not supplied:
        result = generate_json(
            (
                "I want journal articles to support an evidence briefing about: "
                f"{topic!r}\n\n"
                f"{pico_block}"
                "Give 2-4 short keyword queries for scholarly search engines (Semantic "
                "Scholar / OpenAlex). Use researcher terminology. IMPORTANT: make at least "
                "one query specific enough to find work on the exact subject (include the "
                "most specific entity/population by name), not just the general area. Also "
                "return `core_entity`: the specific subject a source must be about to count "
                "as directly on-topic."
            ),
            _QUERY_SCHEMA,
            model=model,
            api_key=api_key,
        )
        queries = (result.get("queries") or [])[:MAX_PLANNED_QUERIES]
        core_entity = (result.get("core_entity") or "").strip()
        if queries and query_routes(queries) <= 1:
            log(f"Planned queries are near-duplicates ({query_routes(queries)} route); requesting one distinct extra query")
            planned_lines = "\n".join(f"  {i}. {q}" for i, q in enumerate(queries, 1))
            follow_prompt = (
                f"I want journal articles to support an evidence briefing about: {topic!r}\n\n"
                f"{pico_block}"
                "These queries were planned for this question:\n"
                f"{planned_lines}\n\n"
                "These terms all search the same route — they re-word one another. Return "
                "`queries`: exactly ONE additional short keyword query that a scholarly "
                "index would match to a different set of records — a different intervention, "
                "population, outcome measure, setting, or the standard indexing term for the "
                "subject. It must not be a re-wording, synonym, reordering or narrowing of "
                "the terms above. An empty list is not an acceptable answer here. Also return "
                "`core_entity`: the specific subject a source must be about to count as "
                "directly on-topic."
            )
            follow_res = generate_json(
                follow_prompt,
                _QUERY_SCHEMA,
                model=model,
                api_key=api_key,
            )
            seen_lower = {q.lower() for q in queries}
            candidate = None
            for cand in follow_res.get("queries") or []:
                cand = (cand or "").strip()
                if cand and cand.lower() not in seen_lower and not any(queries_are_near_duplicates(cand, q) for q in queries):
                    candidate = cand
                    break
            if candidate:
                if len(queries) >= MAX_PLANNED_QUERIES:
                    drop_idx = None
                    for i in range(len(queries) - 1, 0, -1):
                        if any(queries_are_near_duplicates(queries[i], queries[j]) for j in range(i)):
                            drop_idx = i
                            break
                    if drop_idx is not None:
                        queries.pop(drop_idx)
                        queries.append(candidate)
                else:
                    queries.append(candidate)
            else:
                log("Planner follow-up did not return a distinct query; keeping planned queries")
        return queries, core_entity

    terms_lines = "\n".join(f"  {i}. {t}" for i, t in enumerate(supplied, 1))
    routes = query_routes(supplied)
    near_dup = routes <= 1
    if near_dup:
        log(f"Supplied search terms are near-duplicates ({routes} route); requiring one distinct extra query")
        prompt = (
            f"I want journal articles to support an evidence briefing about: {topic!r}\n\n"
            f"{pico_block}"
            "These scholarly search terms were already chosen for this question and will "
            "be searched as they stand:\n"
            f"{terms_lines}\n"
            "Do not rewrite them, reorder them or propose alternatives to them.\n\n"
            "These terms all search the same route — they re-word one another. Return "
            "`queries`: at most ONE additional short keyword query (and here, exactly ONE is required) "
            "that a scholarly index would match to a different set of records — a different "
            "intervention, population, outcome measure, setting, or the standard indexing term "
            "for the subject. It must not be a re-wording, synonym, reordering or narrowing of "
            "the terms above. An empty list is not an acceptable answer here. Also return "
            "`core_entity`: the specific subject a source must be about to count as "
            "directly on-topic."
        )
    else:
        prompt = (
            f"I want journal articles to support an evidence briefing about: {topic!r}\n\n"
            f"{pico_block}"
            "These scholarly search terms were already chosen for this question and will "
            "be searched as they stand:\n"
            f"{terms_lines}\n"
            "Do not rewrite them, reorder them or propose alternatives to them.\n\n"
            "Return `queries`: at most ONE additional short keyword query, and only if it "
            "finds work the terms above would miss — make it specific enough to name the "
            "exact subject (the most specific entity/population by name). Return an empty "
            "list if the terms above already cover the question. Also return "
            "`core_entity`: the specific subject a source must be about to count as "
            "directly on-topic."
        )

    result = generate_json(
        prompt,
        _QUERY_SCHEMA,
        model=model,
        api_key=api_key,
    )
    queries = list(supplied)
    seen_lower = {s.lower() for s in queries}

    if not near_dup:
        for q in result.get("queries") or []:
            q = (q or "").strip()
            if q and q.lower() not in seen_lower:
                queries.append(q)
                break
        return queries[:MAX_PLANNED_QUERIES], (result.get("core_entity") or "").strip()

    candidate = None
    for q in result.get("queries") or []:
        q = (q or "").strip()
        if q and q.lower() not in seen_lower and not any(queries_are_near_duplicates(q, s) for s in queries):
            candidate = q
            break

    if candidate:
        queries.append(candidate)
        return queries[:MAX_PLANNED_QUERIES], (result.get("core_entity") or "").strip()

    # Retry once
    rejected = ""
    for q in result.get("queries") or []:
        q = (q or "").strip()
        if q:
            rejected = q
            break
    if rejected:
        log(f"Planner proposed near-duplicate extra query {rejected!r}; retrying once for a distinct route")
        retry_note = f"\n\n{rejected!r} was a re-wording of the terms above — it searches the same records. Return a different indexing term."
    else:
        log("Planner returned no distinct query; retrying once for a distinct route")
        retry_note = "\n\nThe previous reply was a re-wording or empty — it did not provide a different search route. Return a different indexing term."

    retry_prompt = prompt + retry_note
    retry_result = generate_json(
        retry_prompt,
        _QUERY_SCHEMA,
        model=model,
        api_key=api_key,
    )
    for q in retry_result.get("queries") or []:
        q = (q or "").strip()
        if q and q.lower() not in seen_lower and not any(queries_are_near_duplicates(q, s) for s in queries):
            queries.append(q)
            return queries[:MAX_PLANNED_QUERIES], (result.get("core_entity") or retry_result.get("core_entity") or "").strip()

    log("Planner failed to provide a distinct extra query; using supplied terms unchanged")
    return queries[:MAX_PLANNED_QUERIES], (result.get("core_entity") or retry_result.get("core_entity") or "").strip()


def _truncate_abstract(abstract: str, limit: int | None) -> str:
    """Cut an abstract to `limit` characters at a word boundary, or leave it.

    Only `curate_sources` may pass a limit, and only when explicitly asked
    (#117). Relevance labelling plausibly turns on the topic and the population,
    which live in the first few hundred characters — but that is a hypothesis
    about the gate that stops topic drift, and it has to be measured before it
    is believed. `tools/compare_curation.py` is the measurement.
    """
    if not limit or len(abstract) <= limit:
        return abstract
    cut = abstract[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + " […]"


# `classify_design` labels, as a writer reads them. "other" is left out: a
# design the metadata cannot name is no design at all, not a category.
_PROMPT_DESIGN = {
    "synthesis": "systematic review or meta-analysis",
    "trial": "trial",
    "observational": "observational study",
    "qualitative": "qualitative study",
    "case": "case report or series",
    "scoping": "scoping review",
    "narrative": "narrative review",
    "consensus": "consensus statement or guideline",
}


def _format_sources(
    papers: list[Paper],
    relevance: dict[int, str] | None = None,
    excerpts: dict[int, str] | None = None,
    omit: set[int] | None = None,
    abstract_chars: int | None = None,
    appraisal: bool = False,
) -> str:
    """Render the sources for the prompt.

    `excerpts` (from `sources.full_text_excerpts`) appends each paper's
    open-access full text below its abstract.

    `omit` drops sources by index while leaving every other source's number
    alone, so SOURCE 7 is the seventh paper whether or not SOURCE 6 was
    dropped. The numbering is the citation scheme — `render` maps the writer's
    markers back through it — so it must never be re-packed to close a gap.

    There is no character budget any more. Until Groq was removed this function
    also trimmed abstracts — shortest-cap-first, then dropping the lowest-ranked
    papers — to fit a 12,000 tokens/minute ceiling that counted the reserved
    output as well as the prompt. `sources.full_text_excerpts` already bounds
    the payload (12,000 chars per paper inside a 60,000 total), and no remaining
    provider meters per minute, so the trimming was cost without benefit.
    """

    def render(paper: Paper, index: int, abstract: str) -> str:
        tag = f" [{relevance[index]} to topic]" if relevance and index in relevance else ""
        venue = paper.venue or "unknown"
        design = ""
        if appraisal:
            # What Table 1 will print, so prose and table agree, and the status
            # a reader most needs flagged. Writer prompts only: curation judges
            # topic fit, not design.
            label = _PROMPT_DESIGN.get(classify_design(paper), "")
            if label:
                design = f"Design (from index metadata): {label}\n"
            if paper.is_preprint:
                venue += " — PREPRINT, not peer reviewed"
        block = (
            f"SOURCE {index}{tag}\n"
            f"Title: {paper.title}\n"
            f"Authors: {paper.author_line} ({paper.year or 'n.d.'})\n"
            f"Venue: {venue} | Citations: {paper.citation_count}\n"
            f"{design}"
            f"Abstract: {abstract}"
        )
        if excerpts and index in excerpts:
            block += (
                "\nFull text (open access; may be truncated):\n" + excerpts[index]
            )
        elif excerpts:
            block += "\n(Abstract only — no open-access full text was available.)"
        return block

    return "\n\n".join(
        render(p, i, _truncate_abstract(p.abstract, abstract_chars))
        for i, p in enumerate(papers, start=1)
        if not (omit and i in omit)
    )


# Truncating the abstracts sent to curation saves real tokens and **stays off**
# (#117, measured, closed). Curation is the relevance gate, and two things
# downstream key off its labels — `style._required_sections` scales the section
# floor with the `direct` count, and `write_article` omits `tangential` sources
# from the prompt entirely.
#
# `tools/compare_curation.py --chars 400` over four topics, 80 labels:
#
#   agreement    64/80
#   direct       27/30 retained   <-- gate degraded
#   tangential   12/17 retained   <-- gate degraded
#
# Every topic moved at least one gating label, so it fails the acceptance rule
# outright. The saving it was weighed against is also smaller than the estimate
# that motivated it: measured at ~12,800 input tokens per curation call across
# the four topics, not the ~24,000 assumed.
#
# The way it fails is *not* the cheaper-tier failure. That one collapsed
# everything toward "related". This one moved 8 labels into "related" and 8 out
# — symmetric churn, so truncation does not bias the gate, it destabilises it.
# Cutting to the first 400 characters removes the population and outcome detail
# that separates "direct" from "related", and what replaces it is noise rather
# than a consistent lean. Re-running at a larger `--chars` is the only version
# of this worth trying again, and it needs the same acceptance rule.
CURATION_ABSTRACT_CHARS = None


def curate_sources(
    topic: str, papers: list[Paper], model: str | None = None, api_key: str | None = None,
    abstract_chars: int | None = None,
    *, pico: dict | None = None,
) -> dict:
    """Score each paper's relevance to the exact topic. Returns:
    {relevance: {index: label}, most_relevant_index: int,
     counts: {direct, related, tangential}}. An empty result carries an
    `error` key naming the reason; the caller is expected to treat empty
    as fatal.

    `abstract_chars` truncates each abstract in the prompt. Only the comparison
    harness passes it; the pipeline uses `CURATION_ABSTRACT_CHARS`, which is
    `None` until the measurement says otherwise.

    This is called a second time for the records the named-source pass added.
    The caller shifts the returned indices by the existing pool length, so this
    function must keep returning 1-based indices relative to the list it was handed.
    """
    if not papers:
        return {"relevance": {}, "most_relevant_index": None, "counts": {}}
    limit = abstract_chars if abstract_chars is not None else CURATION_ABSTRACT_CHARS
    try:
        result = generate_json(
            f"Topic: {topic}\n\n{format_pico(pico)}"
            "Rate each source's relevance to that exact topic.\n\n"
            + _format_sources(papers, abstract_chars=limit),
            _CURATE_SCHEMA,
            system=_CURATE_SYSTEM,
            model=model,
            api_key=api_key,
        )
    except Exception as exc:
        return {"relevance": {}, "most_relevant_index": None, "counts": {},
                "error": f"{type(exc).__name__}: {exc}"[:200]}

    relevance: dict[int, str] = {}
    for a in result.get("assessments", []):
        idx = a.get("index")
        label = a.get("relevance")
        if isinstance(idx, int) and 1 <= idx <= len(papers) and label in ("direct", "related", "tangential"):
            relevance[idx] = label
    counts = {
        level: sum(1 for v in relevance.values() if v == level)
        for level in ("direct", "related", "tangential")
    }
    if not relevance:
        return {"relevance": {}, "most_relevant_index": None, "counts": {},
                "error": "the model returned no usable relevance labels"}
    mri = result.get("most_relevant_index")
    if not (isinstance(mri, int) and 1 <= mri <= len(papers)):
        # fall back to a direct source, else the first
        mri = next((i for i, v in relevance.items() if v == "direct"), 1)
    return {"relevance": relevance, "most_relevant_index": mri, "counts": counts}


def _writer_context(
    topic: str,
    papers: list[Paper],
    style_note: str,
    curation: dict,
    *,
    kind: str,
) -> tuple[str, bool]:
    """Shared source payload for the briefing and the `--long` Review.

    `kind` is 'briefing' or 'article' — the prompt must not misdescribe what
    it is asking the model to write. Tangential sources are dropped by number
    rather than filtered out of the list: the SOURCE index IS the citation
    scheme, so the remaining sources keep the numbers `render` will look them
    up by.
    """
    relevance = curation.get("relevance") or {}
    counts = curation.get("counts") or {}
    mri = curation.get("most_relevant_index")

    context = f"Topic: {topic}\n\n"
    if kind == "briefing":
        context += (
            "That line is the reader's question, as typed. `question` and `title` "
            "must stay on it — polish the wording if it is clumsy, but do not add a "
            "clause, narrow the population, or drift to whatever the sources below "
            "happen to cover.\n\n"
        )
    else:
        context += (
            "That line is the reader's question, as typed. The `title` and the "
            "Introduction's statement of scope must stay on it — polish the wording "
            "if it is clumsy, but do not add a clause, narrow the population, or "
            "drift to whatever the sources below happen to cover.\n\n"
        )
    if style_note:
        context += f"Extra guidance from the reader: {style_note}\n\n"
    if counts:
        context += (
            f"Source relevance tally: {counts.get('direct', 0)} directly on-topic, "
            f"{counts.get('related', 0)} related, {counts.get('tangential', 0)} tangential. "
        )
        if not counts.get("direct"):
            context += (
                "NOTE: no sources directly study this exact topic — you MUST say so and "
                "clearly label any evidence borrowed from adjacent populations. "
            )
        context += "\n\n"
    if mri:
        if kind == "briefing":
            context += (
                f"Suggested paper to put first in `open_these` (most relevant): "
                f"SOURCE {mri}.\n\n"
            )
        else:
            context += f"Suggested study to feature (most relevant): SOURCE {mri}.\n\n"
    excerpts = full_text_excerpts(papers)
    use_full_text = bool(excerpts)

    # Retracted records are omitted for a different reason than tangential ones
    # and it is not a relevance judgement: the model must not be able to cite a
    # withdrawn paper (#242). They stay in `papers` — the SOURCE index is the
    # citation scheme and is never re-packed — and stay in the screened count,
    # which is what Methods reports.
    retracted = {i for i, p in enumerate(papers, start=1) if p.is_retracted}
    omit = {i for i, label in relevance.items() if label == "tangential"} | retracted
    context += (
        "Here are the candidate sources with their relevance labels. Choose the ones "
        f"that genuinely support the {kind} and write it.\n\n"
    )
    tangential = omit - retracted
    if tangential:
        context += (
            f"{len(tangential)} tangential source(s) are counted in the tally above but "
            "are NOT reproduced below, because they are background only. "
        )
    if retracted:
        context += (
            f"{len(retracted)} record(s) reported as retracted are also NOT reproduced "
            "below and must not be cited. "
        )
    if omit:
        context += (
            "The numbering below is unchanged and has gaps where they were. Cite only "
            "sources you can actually see.\n\n"
        )
    shown = len(papers) - len(omit)
    target = cite_target(counts.get("direct") if counts else None, shown)
    if shown > target:
        direct_clause = (
            f", of which {counts.get('direct', 0)} are directly on-topic"
            if counts else ""
        )
        context += (
            f"WORKING SET. {len(papers)} records were screened and {shown} are "
            f"reproduced below{direct_clause}. Cite AT MOST {target} of them — the "
            f"ones that carry the {kind}, direct sources first. That is a ceiling "
            "and not a quota: citing fewer is the correct outcome of a thin "
            "evidence base, and a source added to reach the number makes the "
            "piece worse. Leaving a screened source uncited is the expected "
            "outcome of screening, not an omission; the counts the reader sees "
            "are computed from what you cite.\n\n"
        )
        n_direct = (counts or {}).get("direct") or 0
        if kind == "briefing" and n_direct >= target:
            # Measured: with 29 direct sources and a ceiling of 12, the writer
            # cited 5 — the "ceiling, not a quota" wording above read as
            # permission to stop at one study per finding.
            context += (
                f"This is NOT a thin evidence base: {n_direct} sources are directly "
                f"on-topic. A briefing citing far fewer than {target} is leaving direct "
                "evidence out. Build each finding on all the direct studies that "
                "bear on it.\n\n"
            )
    else:
        context += (
            f"WORKING SET. {len(papers)} records were screened and {shown} are "
            f"reproduced below. Cite the ones that carry the {kind} and no more; "
            "a source you have nothing specific to say about does not belong in "
            "the reference list.\n\n"
        )
    read = sorted(i for i in excerpts if i not in omit)
    if read:
        context += (
            "DEEP READS. The open-access full text below was fetched for "
            f"SOURCE {', '.join(str(i) for i in read)} — the pipeline spent a "
            "retrieval on each because they were the most relevant reviews and "
            "trials it could read. Prefer them: a full-text source you leave "
            "uncited while citing an abstract-only case report is the wrong trade. "
            f"Cite one only if the {kind} genuinely has no use for it.\n\n"
        )
    context += _format_sources(papers, relevance, excerpts, omit=omit, appraisal=True)
    return context, use_full_text


def write_briefing(
    topic: str,
    papers: list[Paper],
    model: str | None = None,
    style_note: str = "",
    curation: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """Write the default artefact: a one-page sourced evidence briefing."""
    context, use_full_text = _writer_context(
        topic, papers, style_note, curation or {}, kind="briefing")
    article = generate_json(
        context,
        _BRIEFING_SCHEMA,
        system=_BRIEFING_SYSTEM_FULLTEXT if use_full_text else _BRIEFING_SYSTEM,
        model=model,
        deep=True,
        api_key=api_key,
    )
    article["form"] = FORM_BRIEFING
    return normalize_briefing(article)


def write_article(
    topic: str,
    papers: list[Paper],
    model: str | None = None,
    style_note: str = "",
    curation: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """Write the `--long` Review as structured JSON, grounded in the fetched abstracts."""
    context, use_full_text = _writer_context(
        topic, papers, style_note, curation or {}, kind="article")
    article = generate_json(
        context,
        _ARTICLE_SCHEMA,
        system=_WRITER_SYSTEM_FULLTEXT if use_full_text else _WRITER_SYSTEM,
        model=model,
        deep=True,
        api_key=api_key,
    )
    article["form"] = FORM_REVIEW
    return article


def revise_prose(
    article: dict,
    brief: str,
    model: str | None = None,
    api_key: str | None = None,
    papers: list[Paper] | None = None,
    curation: dict | None = None,
    rewrite_whole: bool = False,
) -> dict:
    """Re-write a draft's prose against a list of style violations from `style.py`.

    The brief names the specific conventions broken and quotes the offending text,
    so this is a targeted edit rather than a second attempt at the whole article.

    By default the model returns only the blocks it changed and they are merged
    in here — see `_REVISE_PATCH_SYSTEM` for why. `rewrite_whole=True` asks for
    the entire article instead, which `enforce_style` uses when the draft needs
    sections it does not yet have: a patch can only replace a block that exists.

    Either way the return value is a complete article, so callers do not need to
    know which path ran.
    """
    draft_json = json.dumps(article, ensure_ascii=False, indent=1)

    context = f"{brief}\n\n"
    if papers:
        # Without these the model can only reshuffle what it already wrote, so a
        # draft that failed for thinness comes back thin. The sources are the
        # only material it may legitimately add.
        relevance = (curation or {}).get("relevance") or {}
        excerpts = full_text_excerpts(papers)
        use_full_text = bool(excerpts)
        context += (
            "SOURCES — the material this article must be grounded in. Anything you "
            "add must come from here and be cited by its SOURCE number:\n\n"
            + _format_sources(papers, relevance, excerpts)
            + "\n\n"
        )
    else:
        use_full_text = False
    context += "Here is the draft to revise, as JSON:\n\n" + draft_json
    briefing = is_briefing(article)

    if rewrite_whole:
        if briefing:
            schema = _BRIEFING_SCHEMA
            system = (_REVISE_BRIEFING_SYSTEM_FULLTEXT if use_full_text
                      else _REVISE_BRIEFING_SYSTEM)
        else:
            schema = _ARTICLE_SCHEMA
            system = _REVISE_SYSTEM_FULLTEXT if use_full_text else _REVISE_SYSTEM
        revised = generate_json(
            context, schema, system=system, model=model, deep=True, api_key=api_key,
        )
        revised["form"] = article.get("form")
        return normalize_briefing(revised) if briefing else revised

    if briefing:
        patch_system = (_REVISE_BRIEFING_PATCH_SYSTEM_FULLTEXT if use_full_text
                        else _REVISE_BRIEFING_PATCH_SYSTEM)
    else:
        patch_system = (_REVISE_PATCH_SYSTEM_FULLTEXT if use_full_text
                        else _REVISE_PATCH_SYSTEM)
    result = generate_json(
        context,
        _REVISION_SCHEMA,
        system=patch_system,
        model=model,
        deep=True,
        api_key=api_key,
    )
    revised, applied = apply_revisions(article, result.get("edits") or [])
    if not applied:
        # Nothing landed: either the model returned no edits, or every `where`
        # it named was a block this article does not have. Both mean the draft
        # is unchanged, and returning it as if it were revised would let
        # `enforce_style` log a revision that never happened.
        raise RuntimeError(
            "the revision returned no edit that matched a block in the draft "
            f"({len(result.get('edits') or [])} edit(s) offered)"
        )
    return revised


def revise_statistics(
    article: dict,
    brief: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Revise a draft's prose to remove or fix unverified/misattributed figures.

    Patch-only: returns only the blocks that changed. No sources are passed in
    the payload because the model is strictly forbidden from adding new numbers
    or claims.
    """
    draft_json = json.dumps(article, ensure_ascii=False, indent=1)
    context = (
        f"{brief}\n\n"
        "Here is the draft to revise, as JSON:\n\n"
        f"{draft_json}"
    )
    briefing = is_briefing(article)
    patch_system = (
        _REVISE_BRIEFING_FIGURES_PATCH_SYSTEM if briefing
        else _REVISE_FIGURES_PATCH_SYSTEM
    )
    result = generate_json(
        context,
        _REVISION_SCHEMA,
        system=patch_system,
        model=model,
        deep=True,
        api_key=api_key,
    )
    revised, applied = apply_revisions(article, result.get("edits") or [])
    if not applied:
        raise RuntimeError(
            "the revision returned no edit that matched a block in the draft "
            f"({len(result.get('edits') or [])} edit(s) offered)"
        )
    return revised

"""Deterministic prose-style check against journal writing conventions.

`verify.py` checks that the *numbers* in a draft are grounded. This checks that
the *prose* reads like a journal article rather than a magazine feature — the
conventions are specific enough to test, so we test them instead of hoping the
model followed the prompt:

- no second person, contractions, rhetorical questions or exclamations
- no booster/hype vocabulary ("clearly", "dramatically", "groundbreaking")
- no absolute claims of proof from observational evidence
- first person only in the reviewing frame journals actually use ("here we review")
- hedging present at roughly the density corpus studies report for research
  articles — about one hedge every two to three sentences
- sentences of a readable length

Everything here is a regex over the draft's own text. Like `verify.py`, it is
deliberately not an LLM pass: a model asked "is this journal style?" will agree
with itself. `docs/journal-style.md` records where each convention comes from.

`cli.cmd_draft` runs this after `write_article` and, when it finds errors, sends
them back to the model once for a targeted revision.
"""

from __future__ import annotations

import collections
import re
import statistics

from .briefing import certainty_level, finding_parts, finding_texts, overall_certainty

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

# Boosters and hype. Journals hedge claims to the evidence; these do the opposite,
# asserting confidence that a cited abstract cannot support.
_BOOSTERS = (
    "clearly", "obviously", "undoubtedly", "unquestionably", "certainly",
    "dramatically", "dramatic", "remarkable", "remarkably", "striking",
    "strikingly", "stunning", "staggering", "astonishing", "incredible",
    "revolutionary", "groundbreaking", "ground-breaking", "game-changing",
    "game-changer", "breakthrough", "unprecedented", "massive", "huge",
    "of course", "needless to say", "it goes without saying",
)

# Claims of proof. Even a randomized trial "shows" or "indicates"; almost nothing
# in an abstract-derived review "proves" anything.
_ABSOLUTES = (
    "proves", "proven", "proved that", "definitively", "conclusively",
    "beyond doubt", "without doubt", "categorically", "irrefutable",
)

# Epistemic hedges — markers that qualify how confident a claim is. These are the
# only ones that count toward the hedging floor.
#
# The frequency and degree adverbs below used to live in this list, which made the
# floor satisfiable by prose that hedges nothing: a paragraph of "approximately
# 40%... typically higher... often persist... generally largest" scored 1.0
# hedges/sentence against a 0.20 floor while qualifying not one claim. That is the
# same loophole the substance rules exist to close, one level down.
#
# "cannot be" and "could not be" were also in here. They assert certainty, so a
# confident claim was raising the hedge score.
_HEDGES = (
    "may", "might", "could", "appears", "appear", "appeared", "seems", "seem",
    "suggests", "suggest", "suggested", "indicates", "indicate", "indicated",
    "likely", "unlikely", "probably", "possibly", "perhaps", "presumably",
    "potentially", "apparently", "consistent with",
    "is thought", "are thought", "reportedly",
    "tends to", "tend to", "in part", "partly", "to some extent", "plausible",
    "plausibly", "possible", "probable", "assumption", "possibility",
    "remains unclear", "remains unresolved", "not established", "uncertain",
    "would be expected",
)

# Frequency and degree adverbs. They soften prose but do not qualify a claim's
# evidential strength, so they are tracked (a draft built entirely from these is
# worth knowing about) but never counted toward the hedging floor.
_SOFTENERS = (
    "relatively", "largely", "broadly", "generally", "typically", "often",
    "commonly", "frequently", "estimated", "approximately",
)

# Clinical directives — the rule that exists because a shipped draft contained a
# titration protocol ("starting with a low-dose exposure of 15 minutes per
# day... titrated upward by 15 minutes each week") for a population the same
# article said had zero studies (issue #102).
#
# Reporting what a trial did is the job. Telling a clinician what to do is a
# different act and the one that carries real-world risk, and the footer
# disclaimer does no work against it: a reader who has been given a dose and a
# schedule has already been given advice.
#
# The line drawn here is grammatical, because that is the part a deterministic
# check can see. "Participants received 10,000 lux for 30 minutes each morning"
# reports. "Exposure should be titrated upward each week" instructs. Past tense
# with a study subject is reporting; a modal or an imperative aimed at a
# clinical act is an instruction.
_CLINICAL_ACTS = (
    r"prescrib\w*", r"administer\w*", r"titrat\w*", r"initiat\w*",
    r"dose[ds]?\b", r"dosing", r"treat\w*", r"monitor\w*", r"screen\w*",
    r"refer\w*", r"offer\w*", r"taper\w*", r"discontinu\w*", r"withdraw\w*",
    r"supplement\w*", r"co-?prescrib\w*", r"augment\w*",
)

# A modal or recommendation frame pointed at one of those acts. The 40-character
# window keeps "should" attached to the verb it governs rather than to something
# two clauses away.
_DIRECTIVE_RE = re.compile(
    r"\b(?:should|must|ought\s+to|needs?\s+to|has\s+to|have\s+to|"
    r"is\s+advised|are\s+advised|is\s+recommended|are\s+recommended|"
    r"is\s+indicated|are\s+indicated|is\s+warranted|are\s+warranted)\b"
    r"[^.;:]{0,40}?\b(?:" + "|".join(_CLINICAL_ACTS) + r")",
    re.IGNORECASE,
)

# "Future trials should monitor adherence" is a research recommendation, which
# is standard in a review's Conclusions and is not clinical advice. The subject
# decides: research nouns in the clause before the modal make it legitimate.
_RESEARCH_SUBJECT = re.compile(
    r"\b(?:research|studies|study|trials?|work|literature|evidence|"
    r"investigations?|investigators?|authors?|analyses|analysis|data|"
    r"reviews?|replications?|guidelines?)\b",
    re.IGNORECASE,
)

# Prescription grammar with no modal in sight. These are multi-word on purpose:
# a bare "titrated" appears in honest reporting of a trial protocol, but
# "titrated upward" and "starting dose" are the vocabulary of an instruction.
# "dose reduction" / "dose escalation" are *not* here: they name trial arms
# ("dose reduction versus maintenance") and a Luna briefing of RADAR was
# branded a working draft for reporting that. Instructional dose changes still
# hit starting/target/recommended dose, titrate-upward, or a modal + clinical act.
_PRESCRIPTION_RE = re.compile(
    r"\btitrat\w+\s+(?:upward|downward|up\b|down\b)"
    r"|\b(?:up|down)-titrat\w+"
    r"|\b(?:starting|initial|loading|maintenance|target|recommended)\s+dos\w+"
    r"|\btitration\s+(?:schedule|protocol|regimen)"
    r"|\bis\s+(?:an?\s+\w+\s+|an?\s+)?prerequisite"
    r"|\bfirst-line\s+(?:treatment|therapy|option|choice)\s+(?:should|must|is\s+recommended)",
    re.IGNORECASE,
)

# A sentence opening with a bare clinical verb is an instruction: "Monitor serum
# levels every four weeks." Deliberately NOT built from _CLINICAL_ACTS — those
# patterns end in \w*, and "Treatment of...", "Screening was...", "Dosing varied
# across trials" all open sentences as nouns. Only bare forms that are not also
# ordinary sentence-initial nouns in this domain are listed; "screen" is left
# out because "Screen time" is a real exposure in mental-health literature.
_IMPERATIVE_RE = re.compile(
    r"^(?:titrate|prescribe|administer|initiate|discontinue|taper|monitor|"
    r"refer|augment|co-prescribe)\b",
    re.IGNORECASE,
)

# Idioms that borrow a clinical verb for something else. "These findings should
# be treated as provisional" and "the syndrome should be referred to as X" are
# ordinary review prose, not advice.
_DIRECTIVE_EXCEPTION = re.compile(
    r"\s*(?:to\s+)?as\b|\s*with\s+(?:caution|care)",
    re.IGNORECASE,
)

_CONTRACTIONS = (
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "cannot've", "won't", "wouldn't", "couldn't", "shouldn't",
    "hasn't", "haven't", "hadn't", "it's", "that's", "there's", "here's",
    "we're", "they're", "you're", "i'm", "let's", "we've", "they've",
    "we'll", "they'll", "you'll", "we'd", "they'd",
)

_WORDY = {
    "utilize": "use",
    "utilise": "use",
    "in order to": "to",
    "due to the fact that": "because",
    "a number of": "several",
    "at this point in time": "now",
    "in the event that": "if",
}

# The first-person frame journals do use — Nature asks for "Here we show" or its
# equivalent. Anything else in the first person implies a human author.
#
# Both halves of this are load-bearing and were wrong once. The adverb slot exists
# because "we also review" and "we further examine" are ordinary journal phrasing
# that an immediate we+verb pattern rejects. The verb list covers what systematic
# reviews actually say — checked against real published abstracts, where the
# earlier list flagged five of eight as if a human had written them.
_ALLOWED_FIRST_PERSON = re.compile(
    r"\b(?:here\s+)?we\s+(?:\w+ly\s+|also\s+|further\s+|then\s+|next\s+){0,2}"
    r"(?:review|reviewed|consider|considered|summarize|summarise|"
    r"summarized|summarised|describe|described|argue|argued|focus|focused|focussed|"
    r"report|reported|show|showed|examine|examined|assess|assessed|discuss|discussed|"
    r"search|searched|aim|aimed|identify|identified|conclude|concluded|compare|compared|"
    r"hypothesise|hypothesize|hypothesised|hypothesized|estimate|estimated|"
    r"analyse|analyze|analysed|analyzed|evaluate|evaluated|investigate|investigated|"
    r"present|presented|propose|proposed|include|included|note|noted|find|found|"
    r"observe|observed|explore|explored|outline|outlined|highlight|highlighted|"
    r"use|used|conduct|conducted|perform|performed|group|grouped|pool|pooled|"
    r"extract|extracted|screen|screened|select|selected|retain|retained|"
    r"restrict|restricted|exclude|excluded|stratify|stratified|calculate|calculated|"
    r"measure|measured|test|tested|apply|applied|suggest|suggested|"
    r"recommend|recommended|seek|sought|develop|developed|derive|derived|"
    r"validate|validated|collect|collected|obtain|obtained|define|defined|"
    r"classify|classified|characterise|characterize|characterised|characterized|"
    r"follow|followed|notice|noticed|adopt|adopted|adapt|adapted|"
    r"searched|term|termed|report|reports|see|saw|seen)\b",
    re.IGNORECASE,
)

# "our understanding of the pathophysiology" is ordinary journal English and does
# not claim authorship, unlike "our findings".
_ALLOWED_OUR = re.compile(
    r"\bour\s+(?:current\s+|present\s+|collective\s+)?"
    r"(?:understanding|knowledge|ability|awareness|view|grasp|appreciation)\b",
    re.IGNORECASE,
)

# Bare "I" is a pronoun only when it is not an enumerator. Clinical writing is
# full of "Axis I", "Type I error", "Phase I trial", "Class I evidence" — all of
# which a naive \bI\b flags as a human author speaking.
_ENUMERATOR_BEFORE_I = re.compile(
    r"\b(?:axis|type|phase|class|grade|stage|group|category|level|factor|"
    r"cluster|tier|part|table|figure|appendix|chapter|study|trial)\s+$",
    re.IGNORECASE,
)
# I² — the heterogeneity statistic, which appears in every meta-analysis this
# tool is likely to review, written variously as "I2", "I 2" or "I²".
_I_SQUARED_AFTER = re.compile(r"\s*[²2]\b")
_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)

# Crude passive detector: "was reported", "are associated". Over-counts slightly,
# which is why it only ever produces a warning at a high ratio.
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|been|being|be)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.IGNORECASE
)
# Nominalisation counting was removed deliberately. The rule counted every word
# ending -tion/-sion/-ment/-ance/-ence/-ity/-ism against an 11% threshold, which
# in clinical writing measures the subject matter rather than the style:
# "Depression and anxiety following treatment: an assessment of intervention
# adherence, medication use, and functional impairment in a clinical population"
# scores 42%, because those nouns ARE the topic ("depression" itself ends -sion).
# It was a warning that fired on essentially every article in this domain, and a
# warning that always fires is one the reader learns to skip.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_ABBREV = re.compile(r"\b(et al|e\.g|i\.e|cf|vs|Fig|Dr|approx|no)\.", re.IGNORECASE)

LONG_SENTENCE_WORDS = 45
MIN_HEDGES_PER_SENTENCE = 0.20   # ~1 hedge per 5 sentences; corpora report 1 per 2-3
MAX_PASSIVE_RATIO = 0.55         # share of sentences containing a passive
MIN_SENTENCES_FOR_DENSITY = 12   # below this, density figures are noise
MIN_WORDS_FOR_DENSITY = 250

# Substance floors.
#
# Every rule above this point is a prohibition — don't under-hedge, don't use
# boosters, don't claim proof. A model optimising only against prohibitions
# writes vague, hedged, contentless prose, because asserting nothing is the
# safest way to break no rule. A real draft came back at 803 words across four
# sections with exactly one number in it, having passed every check, hedging at
# 0.69/sentence — three times the floor — using four stock phrases.
#
# These pull the other way: they fail a draft for saying too little. They were
# calibrated against the curated sample in demo.py, which must keep passing.
MIN_BODY_WORDS = 900             # informational only; see the note at the check
MIN_SECTIONS = 5                 # the writer prompt asks for 5-7
MIN_SECTIONS_FLOOR = 3           # Introduction + one thematic section + Conclusions
MAX_HEDGE_SHARE = 0.40           # no single hedge may be this much of the total
# ...but only once there are enough hedges for variety to be a fair expectation.
# A 40% ceiling implicitly demands three distinct hedges, which is unreasonable
# at low volume and arithmetically brittle: with 7 hedges, 3 of one is 43% and
# fires, so a single extra "suggest" flips a passing draft regardless of whether
# the prose got worse. It did exactly that on a real article that hedged at
# 0.389/sentence with 3 distinct markers — better on both counts than 18 of the
# 20 published abstracts in tests/style_corpus.json, where the median is 0.5
# distinct hedges and half use none at all (issue #56). The rule's actual target
# is a draft that hedges heavily using one stock phrase, which needs volume.
MIN_HEDGES_FOR_MONOTONY = 8
MAX_OPENER_REPEATS = 2           # same three-word sentence opening, at most twice
# A shared verbatim run this long (10+ words) is genuine recycled text. Short 8-word
# clinical terms of art ("randomized double-blind placebo-controlled clinical trial") are
# excluded to prevent false positives on domain vocabulary.
REPEATED_PHRASE_WORDS = 10
MIN_SENTENCES_FOR_VARIETY = 10   # below this, repetition counts are noise

# The abstract, the key points and the Introduction are three different jobs, and
# the commonest failure mode is writing them as three renderings of one paragraph
# — paraphrased closely enough that the 8-word `recycled-phrasing` rule misses it.
# Measured as the share of a block's 6-word runs that also occur in the abstract,
# the separation is wide: the curated sample in demo.py scores 0% (Introduction)
# and 2.4% (key points), while a real draft that read as pure repetition scored
# 39% and 23%. The threshold sits well clear of both.
ECHO_GRAM_WORDS = 6
MAX_ABSTRACT_ECHO = 0.12
MIN_GRAMS_FOR_ECHO = 25          # ~30 words; below this the ratio is noise

# A source cited only ever inside a bundle ("[1, 4]") has had nothing said about
# it individually. A bundle asserts that studies agree; it has to be earned by
# first reporting what each one found.
MAX_NEVER_SOLO_SHARE = 0.34
MIN_SOURCES_FOR_SOLO_CHECK = 4


def _sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = _SENTENCE_SPLIT.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _strip_citations(text: str) -> str:
    return re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)


def prose_blocks(article: dict) -> list[tuple[str, str]]:
    """(where, text) for every piece of model-written prose we hold to the style."""
    blocks: list[tuple[str, str]] = []
    if article.get("question"):
        blocks.append(("question", article["question"]))
    summary = (
        article.get("answer")
        or article.get("abstract")
        or article.get("standfirst")
        or ""
    )
    if summary:
        where = "answer" if article.get("answer") else "abstract"
        blocks.append((where, summary))
    for finding in finding_texts(article):
        blocks.append(("findings", finding))
    for unknown in article.get("unknowns") or []:
        blocks.append(("unknowns", unknown))
    for section in article.get("sections", []):
        heading = section.get("heading", "section")
        for para in section.get("paragraphs", []):
            blocks.append((heading, para))
    for point in article.get("key_points") or article.get("key_takeaways") or []:
        blocks.append(("key points", point))
    if article.get("evidence_note"):
        blocks.append(("evidence note", article["evidence_note"]))
    featured = article.get("featured_study") or {}
    for field in ("why", "method", "results"):
        if featured.get(field):
            blocks.append((f"featured study/{field}", featured[field]))
    return blocks


def _excerpt(text: str, match_start: int, width: int = 60) -> str:
    start = max(0, match_start - width // 2)
    snippet = text[start:start + width].strip()
    return f"…{snippet}…" if start else f"{snippet}…"


def _find(pattern: str, text: str) -> re.Match | None:
    return re.search(pattern, text, re.IGNORECASE)


def _directive_position(text: str, sentences: list[str]) -> int | None:
    """Where this block first instructs rather than reports, or None.

    Three shapes, in the order they are worth catching:

    1. A modal aimed at a clinical act — "exposure should be titrated". Skipped
       when the subject is research ("future trials should monitor adherence"
       belongs in a Conclusions section) or when the verb is idiomatic
       ("treated as provisional", "referred to as").
    2. Prescription grammar with no modal at all, which is how the draft that
       prompted this rule read: "titrated upward by 15 minutes each week".
    3. A sentence opening with a bare clinical verb, which has no subject and so
       can only be an instruction.
    """
    for m in _DIRECTIVE_RE.finditer(text):
        if _DIRECTIVE_EXCEPTION.match(text[m.end():m.end() + 20]):
            continue
        if _RESEARCH_SUBJECT.search(text[max(0, m.start() - 60):m.start()]):
            continue
        return m.start()

    m = _PRESCRIPTION_RE.search(text)
    if m:
        return m.start()

    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and _IMPERATIVE_RE.match(stripped):
            at = text.find(stripped[:30])
            return at if at >= 0 else 0
    return None


def _required_sections(direct_sources: int | None) -> int:
    """How many sections this particular review has the material to support.

    A flat floor of 5 is a thinness rule that can *cause* thinness: told to
    produce five sections from six abstracts, the cheapest way to fill the fifth
    is to restate the fourth, which is exactly the repetition the substance rules
    exist to catch. When the evidence base is known and small, the floor drops.
    """
    if direct_sources is None:
        return MIN_SECTIONS
    return max(MIN_SECTIONS_FLOOR, min(MIN_SECTIONS, direct_sources))


def check_style(article: dict, direct_sources: int | None = None) -> dict:
    """Return {'issues': [...], 'stats': {...}}.

    Each issue is {'rule', 'severity', 'where', 'detail', 'excerpt'}. Severity is
    'error' for things no journal would print, 'warning' for matters of degree.
    """
    issues: list[dict] = []
    all_sentences: list[str] = []
    total_words = 0
    total_hedges = 0
    total_softeners = 0
    passive_sentences = 0

    def add(rule, severity, where, detail, excerpt=""):
        issues.append({"rule": rule, "severity": severity, "where": where,
                       "detail": detail, "excerpt": excerpt})

    for where, raw in prose_blocks(article):
        text = _strip_citations(raw)
        lower = text.lower()
        sentences = _sentences(text)
        all_sentences.extend(sentences)
        words = re.findall(r"[A-Za-z][\w'-]*", text)
        total_words += len(words)

        # The briefing's `question` field is the review question. A terminal
        # '?' there is the artefact doing its job, not magazine rhetoric.
        if "?" in text and where != "question":
            add("rhetorical-question", "error", where,
                "Journal prose states; it does not ask the reader questions.",
                _excerpt(text, text.index("?")))
        if "!" in text:
            add("exclamation", "error", where, "Exclamation marks do not appear in journal prose.",
                _excerpt(text, text.index("!")))

        m = _SECOND_PERSON.search(text)
        if m:
            add("second-person", "error", where,
                f"Addresses the reader directly ({m.group(0)!r}).", _excerpt(text, m.start()))

        for contraction in _CONTRACTIONS:
            if re.search(rf"\b{re.escape(contraction)}\b", lower):
                add("contraction", "error", where,
                    f"Contraction {contraction!r}; journals write it out.",
                    _excerpt(text, lower.index(contraction)))
                break

        for booster in _BOOSTERS:
            m = _find(rf"\b{re.escape(booster)}\b", text)
            if m:
                add("booster", "error", where,
                    f"Hype/booster vocabulary ({booster!r}) asserts more confidence "
                    "than a cited abstract supports.", _excerpt(text, m.start()))
                break

        for absolute in _ABSOLUTES:
            m = _find(rf"\b{re.escape(absolute)}\b", text)
            if m:
                add("overclaim", "error", where,
                    f"Claim of proof ({absolute!r}); evidence indicates or supports, "
                    "it rarely proves.", _excerpt(text, m.start()))
                break

        # Clinical directives. Reporting what a study did is the job; telling a
        # clinician what to do is a different act, and the one that carries
        # real-world risk (#102).
        at = _directive_position(text, sentences)
        if at is not None:
            add("clinical-directive", "error", where,
                "Instructs a clinician rather than reporting what studies found. "
                "State what was done and observed; never give a dose, schedule, "
                "titration or monitoring instruction.", _excerpt(text, at))

        # First person outside the reviewing frame implies a human author.
        for m in _FIRST_PERSON.finditer(text):
            if m.group(0) == "I" and (
                _ENUMERATOR_BEFORE_I.search(text[:m.start()])
                or _I_SQUARED_AFTER.match(text[m.end():])
            ):
                continue  # "Axis I", "Type I error", "I2 = 70%" — not a pronoun
            if m.group(0) == "US":
                continue  # "US adults", "US$16 million" — the country, not "us"
            window = text[max(0, m.start() - 8):m.end() + 30]
            if _ALLOWED_FIRST_PERSON.search(window) or _ALLOWED_OUR.match(text[m.start():]):
                continue
            add("first-person", "error", where,
                f"First person ({m.group(0)!r}) outside the 'here we review' frame.",
                _excerpt(text, m.start()))
            break

        for phrase, better in _WORDY.items():
            if phrase in lower:
                add("wordiness", "warning", where,
                    f"{phrase!r} — prefer {better!r}.", _excerpt(text, lower.index(phrase)))
                break

        for sentence in sentences:
            sentence_words = re.findall(r"[A-Za-z][\w'-]*", sentence)
            if len(sentence_words) > LONG_SENTENCE_WORDS:
                add("long-sentence", "warning", where,
                    f"{len(sentence_words)}-word sentence; split it.", _excerpt(sentence, 0, 70))
            if _PASSIVE.search(sentence):
                passive_sentences += 1

        for hedge in _HEDGES:
            total_hedges += len(re.findall(rf"\b{re.escape(hedge)}\b", lower))
        for softener in _SOFTENERS:
            total_softeners += len(re.findall(rf"\b{re.escape(softener)}\b", lower))

    n_sentences = len(all_sentences)
    lengths = [len(re.findall(r"[A-Za-z][\w'-]*", s)) for s in all_sentences]

    # Which hedges, not just how many. The density rule alone is satisfiable by
    # repeating one stock phrase, and that is exactly what happens.
    body_text = " ".join(_strip_citations(t) for _, t in prose_blocks(article)).lower()
    hedge_counts = {
        h: len(re.findall(rf"\b{re.escape(h)}\b", body_text)) for h in _HEDGES
    }
    hedge_counts = {h: n for h, n in hedge_counts.items() if n}
    top_hedge, top_hedge_n = max(hedge_counts.items(), key=lambda kv: kv[1], default=("", 0))

    # A structured briefing grades every finding and the verdict, GRADE-style.
    # That grade is the hedge, stated once and explicitly, which is what the
    # density rule is a proxy for; counting only verbal hedges would push the
    # writer to re-hedge prose the grade already qualifies.
    graded = sum(
        1 for f in article.get("findings") or []
        if isinstance(f, dict) and certainty_level(f.get("certainty"))
    )
    if graded:
        total_hedges += graded + (1 if overall_certainty(article)[0] else 0)

    stats = {
        "sentences": n_sentences,
        "words": total_words,
        "hedges": total_hedges,
        "distinct_hedges": len(hedge_counts),
        "commonest_hedge": top_hedge,
        "hedges_per_sentence": round(total_hedges / n_sentences, 3) if n_sentences else 0.0,
        "softeners": total_softeners,
        "mean_sentence_words": round(statistics.mean(lengths), 1) if lengths else 0.0,
        "passive_ratio": round(passive_sentences / n_sentences, 3) if n_sentences else 0.0,
    }

    # Density checks need a reasonable amount of real prose before they mean
    # anything — a dozen four-word stubs carry no information about hedging.
    if n_sentences >= MIN_SENTENCES_FOR_DENSITY and total_words >= MIN_WORDS_FOR_DENSITY:
        if stats["hedges_per_sentence"] < MIN_HEDGES_PER_SENTENCE:
            add("under-hedged", "error", "whole article",
                f"{total_hedges} hedges across {n_sentences} sentences "
                f"({stats['hedges_per_sentence']} per sentence). Research articles hedge "
                "roughly once every two to three sentences; claims here are stated with "
                "more certainty than the evidence carries.")
        if stats["passive_ratio"] > MAX_PASSIVE_RATIO:
            add("passive-voice", "warning", "whole article",
                f"{stats['passive_ratio']:.0%} of sentences are passive; Nature and Science "
                "both ask for the active voice where it is available.")

    # ---------------------------------------------------------------- substance
    # These fail a draft for saying too little, rather than for saying it wrongly.

    n_sections = len(article.get("sections") or [])
    required_sections = _required_sections(direct_sources)
    briefing = (article or {}).get("form") == "briefing"
    if n_sections and n_sections < required_sections:
        add("too-few-sections", "error", "whole article",
            f"{n_sections} sections; this review needs at least {required_sections}. Add "
            "sections that develop distinct aspects of the question rather than restating "
            "the same one.")

    # Warning, not error, and deliberately so. Calibrating against the curated
    # sample in demo.py showed length does not discriminate: the good sample runs
    # 773 words and the thin, repetitive draft that prompted these rules ran 803.
    # What separated them was hedge variety (8 distinct hedges vs one phrase at
    # 50%) and verbatim recycling. Failing a draft for brevity would reject the
    # known-good sample while passing the bad one, so length only informs.
    # The default artefact is a briefing: it is supposed to be short, so the
    # Review word-count does not apply to it.
    if (not briefing) and total_words and total_words < MIN_BODY_WORDS:
        add("under-length", "warning", "whole article",
            f"{total_words} words of body prose, against the {MIN_BODY_WORDS}-1600 a Review "
            "typically carries. If this is thin rather than merely concise, the remedy is "
            "more evidence reported in more detail — specific findings, designs and "
            "populations from the sources — not more words about the same points.")

    if (total_hedges >= MIN_HEDGES_FOR_MONOTONY
            and top_hedge_n / total_hedges > MAX_HEDGE_SHARE):
        add("hedge-monotony", "error", "whole article",
            f"'{top_hedge}' accounts for {top_hedge_n} of {total_hedges} hedges "
            f"({top_hedge_n / total_hedges:.0%}). Hedging should track how strong each "
            "piece of evidence actually is — 'in a single small trial', 'consistently "
            "across cohorts' — not repeat one stock phrase to meet a quota.")

    if n_sentences >= MIN_SENTENCES_FOR_VARIETY:
        openers = collections.Counter(
            " ".join(s.split()[:3]).lower().strip(".,;:") for s in all_sentences
        )
        for opener, count in openers.items():
            if count > MAX_OPENER_REPEATS and len(opener.split()) == 3:
                add("repeated-opener", "error", "whole article",
                    f"{count} sentences open '{opener}...'. Vary how sentences begin; "
                    "identical openings read as filler.", opener)
                break

        repeat = _repeated_phrase(all_sentences)
        if repeat:
            add("recycled-phrasing", "error", "whole article",
                f"The phrase '{repeat}' appears more than once verbatim. Each section "
                "should carry information the others do not; if a point has been made, "
                "develop it rather than restate it.", repeat)

    for where, against, share in _abstract_echoes(article):
        add("echoed-abstract", "error", where,
            f"{share:.0%} of this text repeats wording from {against}. The abstract is "
            "the standalone summary; the key points carry specific findings, the "
            "Introduction sets up the question and the Conclusions state what is settled "
            f"and what is not. Say something here that {against} does not.")

    never_solo, n_cited = _never_cited_alone(article)
    if never_solo:
        listed = ", ".join(f"[{n}]" for n in never_solo)
        add("bundled-citations", "error", "whole article",
            f"{len(never_solo)} of {n_cited} cited sources ({listed}) never appear on "
            "their own — only bundled with others behind a general claim, so the review "
            "never says what they individually did or found. Give each its own "
            "treatment: design, population, and result.")

    uncited = _uncited_evidence(article)
    if uncited:
        add("uncited-evidence", "error", "findings",
            f"{len(uncited)} sentence(s) in the findings report a result or a number "
            "but cite no source. End each with the [N] of the study it describes, "
            "taken from the SOURCES. A sentence that no source you were given "
            "supports must be deleted, not cited.",
            " | ".join(u[:140] for u in uncited[:3]))

    return {"issues": issues, "stats": stats}


def _grams(text: str, n: int = ECHO_GRAM_WORDS) -> list[str]:
    words = re.findall(r"[A-Za-z][\w']*", _strip_citations(text).lower())
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def _section_text(article: dict, prefix: str) -> str:
    for section in article.get("sections") or []:
        if section.get("heading", "").strip().lower().startswith(prefix):
            return " ".join(section.get("paragraphs") or [])
    return ""


def _echo_pairs(article: dict) -> list[tuple[str, str, str, str]]:
    """(where, against, text, reference) for every pair worth checking for echo."""
    abstract = (
        article.get("answer")
        or article.get("abstract")
        or article.get("standfirst")
        or ""
    )
    points = " ".join(
        finding_texts(article)
        or article.get("key_points")
        or article.get("key_takeaways")
        or []
    )
    against = "the answer" if article.get("answer") else "the abstract"
    introduction = _section_text(article, "introduction")
    conclusions = _section_text(article, "conclusion")

    pairs = []
    if abstract:
        if points:
            label = "findings" if article.get("findings") else "key points"
            pairs.append((label, against, points, abstract))
        if introduction:
            pairs.append(("Introduction", against, introduction, abstract))
    # The other classic duplication: a Conclusions section that restates the
    # Introduction. Only caught before when it was verbatim.
    if introduction and conclusions:
        pairs.append(("Conclusions", "the Introduction", conclusions, introduction))
    return pairs


def _abstract_echoes(article: dict) -> list[tuple[str, str, float]]:
    """(where, against, overlap) for blocks that mostly re-word an earlier one.

    Catches the close-paraphrase case that `_repeated_phrase` misses: the same
    paragraph rewritten as the key points and again as the Introduction.
    """
    flagged = []
    for where, against, text, reference_text in _echo_pairs(article):
        reference = set(_grams(reference_text))
        grams = _grams(text)
        if not reference or len(grams) < MIN_GRAMS_FOR_ECHO:
            continue
        share = sum(1 for g in grams if g in reference) / len(grams)
        if share > MAX_ABSTRACT_ECHO:
            flagged.append((where, against, share))
    return flagged


def _never_cited_alone(article: dict) -> tuple[list[int], int]:
    """Cited sources that only ever appear inside a multi-source citation bundle.

    A solo citation only counts when it is in the body sections. A bullet in the
    key points saying "roster speed matters [3]" is not a study having been
    discussed; the body is where a design, population and result can actually be
    reported, so that is where the credit has to be earned.
    """
    everywhere = " ".join(raw for _, raw in prose_blocks(article))
    # On a briefing the findings *are* the body — there are no sections.
    body = " ".join(finding_texts(article)) or " ".join(
        para
        for section in article.get("sections") or []
        for para in section.get("paragraphs") or []
    )
    markers = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", everywhere)
    if not markers:
        return [], 0

    cited: set[int] = set()
    for marker in markers:
        cited.update(int(n) for n in marker.split(","))

    solo: set[int] = set()
    for marker in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        numbers = [int(n) for n in marker.split(",")]
        if len(numbers) == 1:
            solo.add(numbers[0])

    if len(cited) < MIN_SOURCES_FOR_SOLO_CHECK:
        return [], len(cited)
    never_solo = sorted(cited - solo)
    if len(never_solo) / len(cited) <= MAX_NEVER_SOLO_SHARE:
        return [], len(cited)
    return never_solo, len(cited)


# A sentence in a finding's evidence that reports a number or a study's result
# names the study it came from. Measured on the Sonnet writer: about one evidence
# sentence in fourteen reported a result with no [N] of its own, and one of them
# described a "conference abstract of the same trial" nothing in the cited
# material supports. Without a citation the figure check falls back to every
# source at once, and a claim with no source is invisible to it entirely.
_RESULT_VERB = re.compile(
    r"\b(?:reported|found|showed|observed|associated|reduced|increased|decreased|"
    r"lowered|raised|improved|worsened|differed|significant(?:ly)?|no difference)\b",
    re.I,
)
_SYNTHESIS_OPENER = re.compile(
    r"^(?:taken together|together|overall|across|in sum|in short|both|these|this|"
    r"the two|the three|all|neither|none|such|so|thus|that)\b",
    re.I,
)
_BARE_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _uncited_evidence(article: dict) -> list[str]:
    """Evidence sentences in structured findings that report a result or a
    number but carry no citation marker. Legacy string findings are exempt."""
    out = []
    for finding in article.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        for sentence in _sentences(finding_parts(finding)[1]):
            if re.search(r"\[\d", sentence):
                continue
            numbered = _BARE_NUMBER.search(_YEAR.sub("", sentence))
            result = _RESULT_VERB.search(sentence) and not _SYNTHESIS_OPENER.search(sentence)
            if numbered or result:
                out.append(sentence)
    return out


def _repeated_phrase(sentences: list[str]) -> str:
    """The longest run of words that appears verbatim more than once, if any.

    Catches a Conclusions section that restates an earlier one word for word —
    which reads as padding and which none of the register rules notice.
    """
    seen: dict[str, int] = {}
    for sentence in sentences:
        words = re.findall(r"[A-Za-z][\w'-]*", sentence.lower())
        for i in range(len(words) - REPEATED_PHRASE_WORDS + 1):
            phrase = " ".join(words[i:i + REPEATED_PHRASE_WORDS])
            seen[phrase] = seen.get(phrase, 0) + 1
    repeated = [p for p, n in seen.items() if n > 1]
    return max(repeated, key=len) if repeated else ""


def errors(report: dict) -> list[dict]:
    return [i for i in report["issues"] if i["severity"] == "error"]


def format_report(report: dict) -> str:
    """Human-readable summary for the CLI log."""
    stats = report["stats"]
    lines = [
        f"  {stats['sentences']} sentences, mean {stats['mean_sentence_words']} words; "
        f"{stats['hedges_per_sentence']} hedges/sentence; "
f"{stats['passive_ratio']:.0%} passive"
    ]
    for issue in report["issues"]:
        mark = "!" if issue["severity"] == "error" else "-"
        lines.append(f"  {mark} [{issue['rule']}] {issue['where']}: {issue['detail']}")
        if issue["excerpt"]:
            lines.append(f"      {issue['excerpt']}")
    return "\n".join(lines)


# Failures that can only be fixed by adding grounded material, not by rewording.
SUBSTANCE_RULES = frozenset({
    "under-length", "too-few-sections", "recycled-phrasing",
    "hedge-monotony", "repeated-opener", "echoed-abstract", "bundled-citations",
})

# Rules whose fix needs the sources in the revision prompt. The substance rules
# need material to add; `uncited-evidence` needs to know which study a sentence
# came from. It is not a substance rule: its brief says cite or cut, not "pull
# in more", so `revision_brief` keeps the reword-only instruction for it.
SOURCED_RULES = SUBSTANCE_RULES | {"uncited-evidence"}

# Warnings that ride along on a revision that is already happening. A warning
# never buys a revision on its own — `enforce_style` returns early when there
# are no errors — but once the model is being paid to rewrite prose anyway,
# a 61-word sentence is a fix nobody else is going to make (#145).
#
# `under-length` is deliberately absent. It is the one warning that is also a
# SUBSTANCE_RULE, and letting it into the brief would flip the brief into "go
# back to the SOURCES below" while `enforce_style` — which keys `needs_sources`
# on the *errors* — sent no sources with it. The brief would then name material
# that is not in the prompt.
RIDE_ALONG_WARNINGS = frozenset({"long-sentence", "wordiness", "passive-voice"})


def revision_brief(report: dict) -> str:
    """The instruction sent back to the model when the prose misses the house style."""
    errs = errors(report)
    problems = errs or report["issues"]
    needs_substance = any(i["rule"] in SUBSTANCE_RULES for i in problems)

    # Warnings ride along only when errors already bought the revision. With no
    # errors there is no revision to attach them to, and appending them here
    # would let a warning-only report produce a brief that reads like a job.
    extras = [
        i for i in report["issues"]
        if i["severity"] != "error" and i["rule"] in RIDE_ALONG_WARNINGS
    ] if errs else []

    lines = ["The draft breaks these journal-prose conventions. Fix each one:"]
    for issue in problems:
        lines.append(f"- [{issue['where']}] {issue['detail']}")
        if issue["excerpt"]:
            lines.append(f"  Offending text: {issue['excerpt']}")

    if extras:
        lines.append(
            "Also fix these while you are in the same blocks, provided doing so "
            "disturbs nothing above:"
        )
        for issue in extras:
            lines.append(f"- [{issue['where']}] {issue['detail']}")
            if issue["excerpt"]:
                lines.append(f"  Offending text: {issue['excerpt']}")

    if needs_substance:
        # A register fix is a rewording; a thinness fix is not. Telling the model
        # "do not introduce new claims or numbers" here would forbid the only
        # thing that can work, so the instruction inverts — with the sources
        # supplied alongside, so the additions stay grounded.
        lines.append(
            "This draft is too thin or too repetitive, which rewording alone cannot fix. "
            "Go back to the SOURCES below and pull in specific material you left out: "
            "what each study actually did, in whom, and what it found. Report figures "
            "that appear in an abstract, attribute findings to their design and "
            "population, and say where studies disagree. Every section must carry "
            "something the others do not. Keep every existing citation marker attached "
            "to the claim it supports, and cite any newly used source by its SOURCE "
            "number. Do NOT state a number that appears in no abstract."
        )
    else:
        lines.append(
            "Rewrite ONLY what is needed to fix these. Keep every citation marker exactly "
            "where it is, keep the section headings, and do not introduce new claims or "
            "numbers."
        )
    return "\n".join(lines)

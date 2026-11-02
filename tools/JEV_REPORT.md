# Jev in evidence-brief: where it cuts cost without losing accuracy

November 2026. TypeSafe **Jev 1.14** (`jev-1.14.0`, $0.042 per million input
tokens, output free) against the LLMs the pipeline uses today, on the five
frozen eval pools (`evals/pools`, 208 papers) and the briefings already
written from them. All dollar figures come from real usage: OpenRouter's
`usage.cost` on every call, Jev's input tokens times list price. Total spend
for the study: **$14.66** of a $15 cap.

Harness: `tools/jev_eval/` (`run.py <area> <step>`). Raw results, the cost
ledger and the hand-check worksheet live in `evals/jev/` (gitignored).

## Short answer

| Area | Best use of Jev | Cost | Accuracy |
|---|---|---|---|
| Relevance labels | **Pre-screen**: Jev sets aside papers it is sure are off-subject; the LLM labels the rest | CLI Opus curator −28% (simulated); hosted ±0 | ≥ the LLM alone in every pairing |
| Writer context | **Trim**: better labels + hide no-result `related` sources + cut full text to result passages | writer call −24%, writer input −36% | blind judge 10–10 vs today; same gold-direct coverage |
| Claim-support gate | **Not for overreach** (0/5 caught). Fine for wrong-source sentences (30/30) | — | Luna is the cheap option for overreach |
| Study design | Jev 87% vs today's regex **40%** — but Haiku/Luna reach 92–94% for <1c | — | use an LLM, not Jev |
| Figure flags | Jev agreed with Opus 5.5 on 34/34 at 1/100 the cost — but **31/34 flags were typography**, fixed in code for free | — | deterministic fix wins |

The largest single accuracy finding is not about Jev: **the hosted curator
(GPT-5.6 Luna) agrees with the reference on only 69% of labels**, and Haiku 5.5
reaches 85% at the same price. Jev's pre-screen lifts Luna to 75%.

Opt-in wiring (off by default): `ARTICLEGEN_JEV=prescreen,trim,passages` or
`articlegen draft ... --jev prescreen,trim,passages`; see `articlegen/jev.py`.

## Method

- **Reference (gold).** Two strong models labelled independently; an item is
  gold where they agree. Relevance: Opus 5.5 + Fable 5.1 (agree on 177/208).
  Claims: Opus 5.5 + Fable 5.1. Design: Opus 5.5 + Sonnet 5.5 (Fable cut for
  budget). Bart hand-judged 15 disputed items (7 claims, 8 designs); his
  answer overrides both models. The 31 disputed relevance labels were left
  out of the gold set rather than judged.
- **Bias to keep in mind.** Both relevance references are Anthropic models;
  Haiku and Opus 5 share the family, Luna does not. On the seven claims Bart
  judged, he cleared one sentence both references had flagged, and on the four
  splits sided with Fable three times and Opus once.
- **No post-hoc thresholds.** Every Jev threshold and cascade cut-off was
  chosen leave-one-topic-out: tuned on four pools, scored on the fifth.
- **Noise.** LLM arms ran 2–3 times (Luna ×3 + the frozen run, Haiku ×3,
  Opus 5 ×2). Haiku's per-pool accuracy ranged 0.64–0.97 across its own
  runs; Jev's probabilities drift by up to ±0.1–0.2 between identical
  requests, which moves a few labels near a threshold.
- **Cascades** were first simulated from the LLM's full-pool labels, then
  rerun for real on the escalated subset (Luna, Haiku). The real Haiku
  cascade scored 0.848 against 0.863 simulated: an LLM labels a subset a
  little differently from a whole pool. Simulated numbers are upper bounds;
  the Opus 5 cascade was not rerun for real (budget).

## A. Relevance labelling (direct / related / tangential)

177 gold papers, accuracy pooled over runs.

| Arm | Accuracy | κ | direct recall | tangential recall | $ / 40-paper pool |
|---|---|---|---|---|---|
| Luna (hosted today) | 0.695 | 0.49 | 0.77 | 0.65 | $0.0048 |
| Haiku 5.5 | 0.845 | 0.72 | 0.94 | 0.82 | $0.0050 |
| Opus 5 (CLI today) | 0.907 | 0.84 | 0.91 | 0.93 | $0.219 |
| Jev alone, composed Nouls | 0.746 | 0.53 | 0.88 | **0.99** | $0.0024 |
| Jev alone, one Choice | 0.655 | 0.42 | 0.80 | 0.64 | $0.0024 |
| Jev pre-screen → Luna (real) | 0.751 | 0.57 | 0.78 | 0.80 | $0.0063 |
| Jev pre-screen → Haiku (real) | 0.848 | 0.73 | 0.88 | 0.91 | $0.0054 |
| Jev pre-screen → Opus 5 (simulated) | 0.912 | 0.84 | 0.91 | 0.96 | $0.166 |
| Jev decides confident calls → Opus 5 (simulated) | **0.921** | 0.86 | 0.92 | 0.96 | **$0.157** |

- Jev alone cannot separate `related` from `direct` (related recall 0.17),
  so it never replaces the curator. What it does reliably is spot papers that
  do not study the subject at all: its confident tangential calls were 61 of
  63 right in the real cascades.
- That is exactly where Luna fails: Luna parks off-topic papers in `related`
  (tangential recall 0.65), and every one of them is then shown to the writer.
- No arm ever turned a gold-direct paper into `tangential`.
- Cost on the hosted path is fractions of a cent either way. On the CLI the
  pre-screen takes about half the abstracts out of the Opus prompt.

## B. Claim-support gate (non-numeric overreach)

The pipeline has no such gate; `verify.py` checks numbers only. 93 cited
sentences from the shipped `final2` briefings, each judged against exactly
the material its writer saw. Gold flags: **5** (after Bart's hand check).
Cost is per briefing.

| Arm | Accuracy | Overreach recall | Precision | False-flag rate | $ / briefing |
|---|---|---|---|---|---|
| Fable 5.1 (reference) | 0.98 | 0.80 | 0.80 | 1.1% | $0.30 |
| Opus 5.5 (reference) | 0.96 | 0.60 | 0.60 | 2.3% | $0.105 |
| Luna | 0.93 | 0.70 | 0.41 | 5.7% | $0.003 |
| Sonnet 5.5 | 0.84 | 0.60 | 0.19 | 14.8% | $0.050 |
| Haiku 5.5 | 0.81 | 0.40 | 0.12 | 17.0% | $0.003 |
| Jev, composed (LOO) | 0.88 | **0.00** | — | 6.8% | $0.001 |

- Today's briefings rarely overreach: about 5 in 93 sentences. With five
  positives every recall figure is coarse.
- Jev caught none. Its documented weak spots (literal reading, multi-hop
  scope such as a modifier hung on the wrong outcome) are what overreach is.
- **Wrong-source sentences are easy.** An older briefing format put the
  citation after the full stop, and `verify._article_sentences` then credits
  it to the next sentence, which yields 30 sentences cited to the wrong
  paper. Jev, Haiku and Sonnet flagged 30/30; Luna 88%. The current briefing
  format never does this (0 of 93), but the splitter would mis-attribute any
  text written that way.

## C. Writer context (pre-screen + trim)

Measured with the production `_writer_context` on all five pools:

| Change | Writer prompt chars | Cut |
|---|---|---|
| Today (Luna labels, everything non-tangential shown) | 551,985 | — |
| Jev pre-screen + Haiku labels | 420,928 | −23.7% |
| + hide `related` sources with no results (answer-evidence < 0.1) | 406,886 | −26.3% |
| + cut full text to result passages (≤ 8,000 chars each) | 349,737 | **−36.6%** |
| Jev pre-screen + Luna labels only | 477,526 | −13.5% |

Accuracy checks before any writing:

- **Source trim** (threshold chosen leave-one-out; 0.1 on four pools) hid
  none of the sources three earlier writer runs had cited. The seclusion
  pool's held-out pick (0.3) hid one.
- **Passage trim** kept 63 of the 65 figures earlier briefings took from full
  text (8,000-char budget; 52/65 at 4,000, 61/65 at 6,000).

## D. End-to-end: today vs the Jev combination

Same frozen pools, Sonnet 5.5 writer and both gates, blind Opus judge
(`tools/brief_eval.py` rubric and pairwise, both orders). The Jev arm used
pre-screen → Haiku labels, Jev design labels, source trim and passage trim.
Today's arm: the shipped `final2` briefing plus one fresh rerun.

| | Today | Jev combination |
|---|---|---|
| Judge overall (1–10) | 7.0 / 7.0 | 7.2 / 7.0 |
| Pairwise votes | — | **10–10** |
| Same pipeline, shipped vs rerun | 10–0 | — |
| Unsupported claims per briefing (judge) | 3.2 / 3.6 | 2.8 / 3.8 |
| Gold-direct papers cited (share of pool's) | 0.81 / 0.80 | 0.80 / 0.80 |
| Gold-tangential papers cited | 0 | 0 |
| Writer call | $0.143 | **$0.108 (−24%)** |
| Whole replay (write + style + figure gates) | $0.230 | $0.236 |

- Quality: indistinguishable. The pipeline disagrees with itself more (10–0
  between two runs of the same code) than with the Jev arm.
- The writer saving was eaten by revisions. A style rule that needs sources
  made `revise_prose` reproduce **the whole pool**, including tangential,
  hidden and **retracted** records, at ~57k tokens, about twice the writer's own
  prompt. That broke the #242 rule (a withdrawn paper must not reach the
  drafting model) on every such revision, and it cancels any trim. It is now
  fixed for all runs: revisions see exactly the writer's sources
  (`writer.writer_omitted`).

## E. Figure flags

The deterministic check (`verify.check_statistics`) raised 34 flags across
the 15 new writes before revision. Opus 5.5, Haiku and Jev each judged all 34
"stated in another format". Jev agreed 34/34 with Opus 5.5 for $0.002
against $0.22.

The cause is in code, so the fix is too: the source printed `26·5%` and
`−0·7` (Lancet middle-dot decimals, Unicode minus), thin spaces, `P = 0.696`
for `p=0.696`. Normalising typography on both sides clears **31 of 34**;
every flag had bought a paid revision that could only make a correct figure
worse. `verify._typeset` now does this, and a genuinely absent figure still
flags (`test_figures_survive_typography`).

## F. Study design (Table 1, the writer's "Design" line, GRADE start)

208 papers; gold = Opus 5.5 + Sonnet 5.5 agreement plus Bart's 8.

| Arm | Exact | GRADE tier | observational recall | $ / pool |
|---|---|---|---|---|
| Regex on title/venue/index types (today) | **0.40** | 0.46 | 0.20 | $0 |
| Jev (one Choice) | 0.87 | 0.88 | 0.79 | $0.0017 |
| Haiku 5.5 | 0.92 | 0.94 | 0.93 | $0.0032 |
| Luna | 0.94 | 0.94 | 0.92 | $0.0052 |

The regex labels most observational studies `other`, so the writer gets no
design for them and Table 1 under-reports. Fixing it is worth doing; an LLM
batch call (one per pool, under a cent) beats Jev on accuracy. Not wired here.

## What to do

1. **Hosted:** switch the curator from Luna to Haiku 5.5 (same price, +15
   points), with `ARTICLEGEN_JEV=prescreen,trim,passages` on top. The two
   deterministic fixes in this PR apply to every run already.
2. **CLI:** `--jev prescreen,trim,passages` with the Opus default: about
   −28% on curation (simulated) and −36% writer input. Separately, Opus 5.5
   lists at $4/$20 against Opus 5's $5/$25.
3. **Design labels:** replace the regex with one cheap LLM call per pool.
4. **Do not** use Jev as the overreach gate. If a gate is wanted, Luna is the
   cheap candidate, but measure it on a larger claim set first: five
   positives is not enough.

## Reproduce

```bash
pip install typesafe-sdk          # plus OPENROUTER_API_KEY, TYPESAFE_API_KEY
python tools/jev_eval/run.py relevance reference && python tools/jev_eval/run.py relevance llm
python tools/jev_eval/run.py relevance jev && python tools/jev_eval/run.py relevance cascade
python tools/jev_eval/run.py claims reference|llm|jev
python tools/jev_eval/run.py design reference|llm|jev
python tools/jev_eval/run.py trim jev
python tools/jev_eval/run.py e2e write && python tools/jev_eval/run.py e2e judge && python tools/jev_eval/run.py e2e compare
python tools/jev_eval/run.py e2e figures
python tools/jev_eval/run.py handcheck build      # then fill evals/jev/hand_answers.txt; handcheck apply
python tools/jev_eval/run.py <area> score
python tools/jev_eval/run.py spend
```

Spend by area: relevance $5.32 (references $2.94), claims $4.68 (references
$4.04), design $0.90, end-to-end writes $3.51, figure flags $0.23, Jev in all
areas $0.06.

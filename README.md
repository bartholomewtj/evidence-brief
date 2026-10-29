# Evidence brief

[![Tests](https://github.com/bartholomewtj/evidence-brief/actions/workflows/tests.yml/badge.svg)](https://github.com/bartholomewtj/evidence-brief/actions/workflows/tests.yml)
[![Deployment health](https://github.com/bartholomewtj/evidence-brief/actions/workflows/health.yml/badge.svg)](https://github.com/bartholomewtj/evidence-brief/actions/workflows/health.yml)

Sourced evidence briefings from a question. Every statistic is checked against the cited papers.

One HTML page you can send: the question, a one-sentence verdict and how sure the evidence is, the findings behind it (each with its own certainty grade), what remains open, and the papers to open first.

**[Open the site](https://bartholomewtj.github.io/evidence-brief/)** — Generating is free on the public site. No key and no account needed. The CLI command is still `articlegen`.

## Run it

On the site: type a theme, pick a question, wait.

On your machine:

```bash
pip install -e .
export OPENROUTER_API_KEY=...           # or ANTHROPIC_API_KEY, or --model cli:opus
export SEMANTIC_SCHOLAR_API_KEY=...     # same User-env name as paperfetch; without it S2 is skipped often

articlegen ideas "seclusion reduction in acute psychiatric wards"
articlegen draft "Does reducing seclusion on acute psychiatric wards increase violence?" --open
```

The site and the CLI run the same function. A draft writes `drafts/<date>-<slug>.html`, `.md`, and `.json`.

`.env.sample` lists every variable the CLI and the server read, with what each one does.

CLI default is Claude Opus 5 via OpenRouter (~50c–$1). The public site plans the search and screens the papers with GPT-5.6 Luna, and writes the briefing with Claude Sonnet 5.5, on the host's key (about 25–30c a briefing). `--model openai/gpt-5.6-luna` is the cheap CLI option.

A hosted briefing takes two to four minutes, longer than the ~100s Cloudflare holds a request open in front of Render, so the site starts it as a job (`POST /api/draft` with `async: true`) and polls `GET /api/draft/<job>`. The poll reports the stage the pipeline is actually in.

## How a briefing is made

You pick the question. Everything between that and the page is `generate_draft()` in `articlegen/pipeline.py` — one copy, used by both the site and the CLI.

[![Evidence brief pipeline](docs/pipeline.png)](https://bartholomewtj.github.io/evidence-brief/docs/pipeline.html)

*[Open the interactive diagram](https://bartholomewtj.github.io/evidence-brief/docs/pipeline.html)* — pan, zoom, and three views: the happy path, how the paper pool grows, and where a run stops.

1. **Pre-flight.** Probe the scholarly APIs. If every source is down, stop before the first paid model call.
2. **Plan queries.** The model turns the question into 2–4 search strings. Terms from the idea card are kept, not replaced. PICO fields (population, intervention, comparator, outcome) reach the planner and the curator.
3. **Search.** Semantic Scholar, OpenAlex, Europe PMC, arXiv. Abstracts only at this stage. Deduped, ranked, capped at about 40. arXiv is queried last so a journal version wins over its preprint.
4. **Label.** Each paper is `direct`, `related`, or `tangential`. Empty labels stop the run — no writing, no full text. Retracted records never reach the writer; Methods names them.
5. **Widen the pool.** One hop through the reference lists of the top direct reviews, plus landmark papers named in those abstracts. New records are labelled too, so they can be read in full later.
6. **Write from abstracts.** A structured briefing: the verdict first, then 3–6 findings that each bring together every cited study bearing on one outcome, graded GRADE-style (high, moderate, low, very low). Cite at most `min(12, direct + 2)` sources, never fewer than 5 unless the pool is smaller.
7. **Read cited full text.** Open-access copies of cited *direct/related* sources only, design-weighted, up to 5 (the excerpt budget). Then rewrite once with those texts. If the rewrite fails, drop the full text so Methods cannot claim a read the prose never saw.
8. **Check the prose.** Deterministic journal-register rules (voice, hedging, no "clearly"/proof claims). Failed drafts go back to the model only while errors drop — at most two passes.
9. **Check the figures.** Every number is searched in exactly the abstracts and excerpts the writer was shown. Flagged figures buy one revision. Leftover misses are marked inline. This is the only gate that flags an article.
10. **Render.** HTML + Markdown + a JSON manifest of everything the run knew. Methods names only databases that actually answered. Table 1 is built from the records, not the model.

Two human gates: pick the question, read the page. Nothing in between asks you.

`--long` still writes the parked journal-style Review. Same pipeline, different shape.

## What the page is

| Block | Who writes it |
|---|---|
| Title (the question), the answer with its verdict first, an overall certainty grade, 3–6 synthesised findings (claim, evidence, certainty), unknowns, three papers to open with a reason each | the model, from the papers it was shown |
| Superscript Vancouver citations, reference list with DOIs | numbered from the records |
| **Methods** — search actually run, how deeply each source was read, which bits were model vs deterministic | derived from the run, never hardcoded |
| **Table 1** — design, n, registration, funding, relevance, read depth | the records |
| **Fig. 1** — study designs (or years) in the cited set | the records |
| Masthead: not peer reviewed; machine-written | fixed |

Treat it as a sourced starting point. Follow the links.

## Measuring briefing quality

`tools/brief_eval.py` freezes one evidence pool per topic (a full pipeline run), replays only the writer and its two gates on those exact papers, and scores the result with a blind Claude judge: a 1–10 rubric (answer, synthesis, specificity, calibration, faithfulness, coverage, gaps, readability, usefulness) and a pairwise preference judged in both orders.

```bash
python tools/brief_eval.py freeze --topics tools/brief_eval_topics.json --out POOLS
python tools/brief_eval.py write --pools POOLS --out RUNS/new
python tools/brief_eval.py judge --runs RUNS/new
python tools/brief_eval.py compare --a RUNS/old --b RUNS/new
```

October 2026, five topics, mean judge score out of 10:

| | Overall | Synthesis | Specificity | Answer |
|---|---|---|---|---|
| One study per bullet (before) | 5.0 | 3.6 | 4.2 | 5.6 |
| Synthesised, graded findings (Luna) | 5.8–6.4 | 5.8–6.0 | 6.2–6.6 | 6.4–7.0 |
| Same prompt, Sonnet 5.5 writer | 6.8 | 6.8 | 7.0 | 8.0 |

Pairwise, the synthesised briefing beat the old one 10–0, and Sonnet beat Luna 10–0 on the same prompt. One sample per topic is noisy: the same prompt rerun lost 8–2 to its first run, so prompt tweaks smaller than that are not measurable with five topics.

## What this will not do

- Invent journals, volumes, DOIs, or affiliations
- Let the writer see full text of papers it did not cite
- Claim a database was searched if it refused
- Quietly substitute adjacent literature when direct evidence is thin — the labels are there so the page can say so
- Instruct a clinician. Clinical topics carry a "not medical advice" line
- Auto-queue paywalled cites for a library download. They are logged, not queued

## After a run

```bash
articlegen render drafts/x.json     # rebuild HTML/Markdown, no search, no model
articlegen rerun drafts/x.json      # same pool and labels; fetch full text again; rewrite
articlegen refresh drafts/x.json    # re-run the same queries; report new direct records
articlegen queue                    # local queue of drafts
articlegen demo                     # sample briefing, no APIs
articlegen web                      # the site, served locally; drafts land in drafts/
```

`refresh --rewrite` only writes a new briefing if it found a new *direct* paper.

## Tests

```bash
python tests/test_offline.py
python tests/test_journal_conformance.py
python tests/test_replays.py
```

Run them as scripts, the way CI does. `tests/replays/` holds real run manifests — do not edit them.

pytest is not a dependency and CI does not use it, but `python -m pytest tests` works: `check()` in `test_offline.py` also raises when `PYTEST_CURRENT_TEST` is set, and `test_journal_conventions()` wraps the conformance suite as one case. A new check helper or suite has to fail under both runners, or pytest reports green while the script prints `FAIL`.

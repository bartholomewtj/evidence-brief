# Evidence brief

[![Tests](https://github.com/bartholomewtj/evidence-brief/actions/workflows/tests.yml/badge.svg)](https://github.com/bartholomewtj/evidence-brief/actions/workflows/tests.yml)
[![Deployment health](https://github.com/bartholomewtj/evidence-brief/actions/workflows/health.yml/badge.svg)](https://github.com/bartholomewtj/evidence-brief/actions/workflows/health.yml)

Sourced evidence briefings from a question. Every statistic is checked against
the cited papers.

The question, what the evidence shows, what remains open, and three papers to
open — as one self-contained HTML page (plus Markdown) you can send as a link.
Three stages: pick a question → research collated automatically → briefing
prepared for your review.

The CLI command is still `articlegen`. `articlegen draft --long` still writes
the journal-style Review; that path is kept for later, not deleted.

## Live site

**[Open the site](https://bartholomewtj.github.io/evidence-brief/)**

Generating is free on the public site (GPT-5.6 Luna, host-paid, rate-limited).
No key and no account needed. A finished briefing is listed on the landing
page under **From other visitors** as soon as it is generated.

---

## Use it from your phone or desktop

1. **Open the site:** [https://bartholomewtj.github.io/evidence-brief/](https://bartholomewtj.github.io/evidence-brief/) (or run `articlegen web --open` locally).
2. **Type a theme:** Enter your topic (e.g. `seclusion reduction in acute psychiatric wards`) and optional audience/style notes. The box is the first thing on the page.
3. **Choose a draft:** Tap any generated idea card to launch the evidence-grounded research pipeline.
4. **Read, save & share:** View the rendered briefing. It is already on the
   public list. **Keep** holds it in your library past the rolling limit.
   **Download** saves it as a single HTML file you can open offline or print
   to PDF. **Shareable link** gives you a short link anyone can open.

Below the topic box the landing page lists **From other visitors**
(briefings generated on the site). Your own saved briefings are under
**Your briefings**. CLI drafts stay in `drafts/` on this machine; they are
not on the public site.

On a window wider than 1100px the page switches to a desktop layout: a left
sidebar instead of the top icon row, and the article fills the remaining
window rather than sitting in a boxed column. The article's actions sit in a
toolbar above it rather than a bar along the bottom of the screen. Narrower
than that, you get the phone layout. Browser Back from an article returns to
the landing page. A home button next to Evidence brief does the same.

The public site writes with **GPT-5.6 Luna**. It is free to you; the host pays, and an hourly cap applies to everyone together. The web app does not take a visitor key. The CLI below also supports Anthropic and a Claude subscription. Local `articlegen web` needs `OPENROUTER_API_KEY` in the environment.

**How it fits together.** The page is a front end only — the pipeline below is
the same Python that the CLI runs, on a small backend the page calls. There is
no second implementation: an article generated from your phone goes through the
same relevance gate, prose-style enforcement and statistic verification as one
generated from the terminal. The hosted backend keeps no articles on disk: it
renders, lists the briefing on the public gallery, returns it, and forgets the
rest. The gallery copy is a public GitHub gist. Public generation uses a
host-held OpenRouter key for Luna only. Your private copies live in your own
browser. See [`render.yaml`](render.yaml) to host the backend yourself.

**Provider key setup (CLI):**

- **OpenRouter** (the default; **roughly 50c–$1 an article**): create a key at
  https://openrouter.ai/keys and set `OPENROUTER_API_KEY`.
- **Claude** (opt-in; **roughly $1–2 an article**): set `ANTHROPIC_API_KEY`.
- **Your Claude subscription** (opt-in; **no key and no per-article cost**): if
  you already pay for Claude and have
  [Claude Code](https://claude.com/claude-code) installed and signed in,
  `--model cli:opus` or `--model cli:sonnet` drafts through it. A Claude.ai
  subscription does not come with an API key, and this is the way to use one
  anyway. Command line only — see the caveats below.

**OpenRouter is used by default** — with an `OPENROUTER_API_KEY` set it runs
automatically, and it takes priority if an Anthropic key is also present. To use
another provider, set `ARTICLEGEN_PROVIDER=anthropic` (or `claude-cli`). Models:
`anthropic/claude-opus-5` on OpenRouter (CLI default), `openai/gpt-5.6-luna` on
the public web app, `claude-fable-5` on Anthropic, `cli:opus` on your
subscription, all overridable with `--model`.

**This block is the single place the defaults are described in prose.** Change a
default in `llm.py` and change it here; nowhere else should restate them. The
CLI default (`OPENROUTER_DEFAULT_MODEL`) and the public web default
(`OPENROUTER_PUBLIC_MODEL`) are two constants on purpose.

**Groq was removed.** It used to be the free default, on a 100,000 tokens/day
tier that allowed 4–7 articles. Its real cost was a 12,000 tokens/**minute**
ceiling that counted the reserved output as well as the prompt, so a Groq draft
could never be shown a single open-access full text — the abstracts had to be
trimmed to fit. Every draft is now grounded in full text where one exists. If
you want the same Llama model, `--model meta-llama/llama-3.3-70b-instruct`
reaches it through OpenRouter for well under a cent an article, untrimmed.

**About `cli:` — what you give up.** It costs nothing beyond the subscription
you already pay for, and it is the only option that needs no key. Two real
trade-offs. It runs on your machine only: the hosted web app cannot use it,
because a shared server has no Claude Code and no subscription to draw on. And
both API providers can *force* the model to return correctly-shaped data, which
this cannot — it can only ask. When a reply comes back as prose instead,
`articlegen` asks once more and then gives up on that article. It is the right
pick for drafting at your own desk, and the wrong one for anything automated.

**The public web app is the exception: generating there is free to the visitor.**
It writes with GPT-5.6 Luna on the host's key, at roughly **2c an article** to
the host, inside the hourly rate limits.

**CLI still costs real money.** OpenRouter's CLI default is
Claude Opus 5, at roughly **50c–$1 an article**. The Anthropic default is Fable
5, at roughly **$1–2**. Those are per article, and a failed run still bills you.
Neither has a daily cap, which is the point — but it also means nothing stops a
bad afternoon costing $20.

Cheaper options on the CLI, if writing quality is not what you're trying to fix:

- `--model openai/gpt-5.6-luna` — the public-site model, a few cents an article.
- `--model meta-llama/llama-3.3-70b-instruct` — well under a cent an article.
- `--model anthropic/claude-sonnet-5` — a cheaper Claude than either default.
- `--model cli:opus` — free, on a Claude subscription you already pay for.

Any OpenRouter catalogue model works with `--model`.

Set a free `SEMANTIC_SCHOLAR_API_KEY`. Without it, Semantic Scholar's
shared keyless limit refuses effectively every call: measured over four
runs spanning an hour, the first query of every run came back HTTP 429 and
the source was then skipped for the rest of that run, so those drafts were
written on the other databases alone. `OPENALEX_MAILTO` is genuinely
optional — it puts OpenAlex requests in its "polite pool".

## The same workflow, locally

```
1. articlegen ideas "<theme>"      # briefing questions, pick one  ◀── you choose
2. articlegen draft "<title>"      # research + briefing, auto     ──▶ drafts/
3. review drafts/index.html                                        ◀── you review
```

Two human gates (choose a question, review the briefing); everything in between
is automatic. `draft --long` writes the parked Review instead. Under the hood,
the `draft` stage:

```
title ──▶ the model plans search queries
      ──▶ Semantic Scholar + OpenAlex + Europe PMC + arXiv return papers
          (with abstracts)
      ──▶ each source is labelled direct / related / background
      ──▶ the model writes the briefing, citing sources inline as [1], [2, 3]…
      ──▶ rendered to drafts/<date>-<slug>.html  +  .md, listed in drafts/index.html
```

- **A briefing you can send.** Article-type label **Evidence briefing**, the
  question, a short answer, 5–8 findings with citations, what remains open,
  three papers to open, superscript Vancouver citations, a **Methods** statement
  of the actual search, and standard back matter. `draft --long` is the parked
  journal-style Review (Box 1, Fig. 1, Key points, Introduction → Conclusions).
- **Journal prose, checked rather than requested.** The house style is the
  register of a Nature Reviews or Science Review piece: active voice, tense that
  carries evidential weight (present for established knowledge, past for what one
  study found), findings attributed to their design, and hedging at the density
  corpus studies find in real research articles. `style.py` checks all of it
  deterministically after drafting — second person, contractions, boosters
  ("clearly", "striking"), claims of proof and under-hedging are errors — and any
  failures go back to the model once for a targeted revision.
- **Evidence-grounded, and honest about it.** The writer sees real abstracts —
  and, for the most relevant sources when an open-access copy can be
  retrieved, the full text too — and cites them; every superscript links to a
  numbered reference with a link back to the paper (DOI when available). Before
  writing, each source is scored for how *directly* it addresses the exact topic
  — so the briefing can say when direct evidence is thin instead of quietly
  substituting adjacent work. A deterministic check flags any statistic absent
  from the material the writer was actually shown, and Table 1 is built from
  the fetched records rather than written by the model. Clinical
  topics get a "not medical advice" disclaimer.
- **No fabricated apparatus.** No invented journal name, volume, DOI, received
  dates or affiliations. The masthead says "Not peer reviewed" and the back
  matter says a machine wrote it.
- **Self-contained.** One HTML file, inlined CSS and SVG, no external requests.
  Works offline, prints cleanly, and adapts to light or dark mode.

## Install

```bash
pip install -r requirements.txt      # or: pip install -e .
```

Set credentials for one provider:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...  # OpenRouter — https://openrouter.ai/keys
# or
export ANTHROPIC_API_KEY=sk-ant-...   # Claude (or: ant auth login)
# or neither — draft on a Claude subscription with: --model cli:opus
```

The scholarly APIs work without keys, but Semantic Scholar's shared tier
refuses nearly every call — setting the [free API key](https://www.semanticscholar.org/product/api)
is the fix worth doing on any machine that runs drafts (a run without it still
works on the remaining databases, and Methods will say so):

```bash
export SEMANTIC_SCHOLAR_API_KEY=...
export OPENALEX_MAILTO=you@example.com  # optional, "polite pool"
```

## Use

```bash
# 1. Generate ideas from a broad theme; skim them and pick one
python -m articlegen ideas "seclusion reduction in acute psychiatric wards"

# 2. Research + draft the idea you picked (opens it when done)
python -m articlegen draft "Does reducing seclusion on acute psychiatric wards increase violence?" --open

# 3. Review everything you've drafted from one page
python -m articlegen queue --open

# Preview the visual design instantly, with no API calls or network
python -m articlegen demo --open
```

Or, after `pip install -e .`, drop the `python -m`: `articlegen ideas "..."`.

Each `draft` run writes four things into `drafts/`:
`<date>-<slug>.html` (the styled article), `<date>-<slug>.md` (same content as
Markdown, for easy editing), `<date>-<slug>.json` (the run manifest, below),
and refreshes `index.html` (your review queue).

### The run manifest and `render`

The manifest is everything a run knew, saved as plain JSON beside the HTML:
the topic, the article, every paper screened (with its abstract, any
open-access full text and how it was fetched), the relevance labels, the
figure-verification result, the provenance that Methods is written from,
the prose-style report, and the exact full-text excerpts the writer and the
verifier were shown. Without it a briefing could not be rebuilt once the
process ended.

Rebuild the HTML and Markdown from a manifest, with no search and no model
call:

```bash
python -m articlegen render drafts/2026-09-05-seclusion.json        # writes the .html and .md beside it
python -m articlegen render drafts/2026-09-05-seclusion.json --open
```

The rebuilt page is the same as the one the run wrote, including the run's
date. `render` does not touch the review queue; run `articlegen queue` if you
want the index refreshed.

Each manifest carries `"manifest_version": 1`; `render` refuses a version it
does not know. The web server never writes one — it stays stateless.

### CKN pickup and `rerun`

A draft never fills the CKN list by itself (`PAPERS_NO_CKN_QUEUE=1` on every
`papers get`). After the full-text pass it logs cited sources that were
paywalled, with their DOIs, under one heading. To add exactly those DOIs to
the pickup list:

```bash
python -m articlegen draft "the question" --queue-ckn
```

Then `papers miss`, download, `papers ingest <doi> path\to\file.pdf`, and
rerun from the manifest. Search and labelling are skipped; full text is
fetched again (so the ingest is visible); the briefing is rewritten; a new
manifest is written next to the first as `<stem>-rerun.json`.

```bash
python -m articlegen rerun drafts/2026-09-05-seclusion.json
```

The hosted app has no queue flag.

### Checking for new evidence (`refresh`)

For a topic you send more than once. `refresh` re-runs a manifest's search
queries, checks the results against the pool that manifest already screened,
and labels only the records that are new. It prints the new `direct` ones and
writes nothing:

```bash
python -m articlegen refresh drafts/2026-09-05-seclusion.json
```

Add `--rewrite` to actually rewrite the briefing with the new records — it
writes `<stem>-refresh.json` (plus HTML/Markdown) next to the manifest, and
only when at least one new record was labelled `direct`. Cost is one labelling
call and the searches; there is no writing call unless `--rewrite` found
something worth it.

### Commands & options

| Command | Key options |
|---------|-------------|
| `ideas <theme>` | `-n` (how many, default 6), `-o` (output .md path) |
| `draft <title>` | `--open`, `--style "<audience/tone>"`, `--max-papers N` (default 40), `--name <stem>`, `--queue-ckn`, `--long` |
| `render <manifest.json>` | `--open` |
| `rerun <manifest.json>` | `--open`, `--queue-ckn`, `--long` |
| `refresh <manifest.json>` | `--rewrite`, `--open`, `--long`, `--queue-ckn` |
| `queue` | `--open` |
| `demo` | `--open`, `-o` |

`--model` is global: `articlegen --model … draft "…"`. Default is auto —
OpenRouter Opus (`anthropic/claude-opus-5`), or `claude-fable-5` when only
an Anthropic key is set.

## Notes & limitations

- Coverage depends on what the open scholarly APIs return; niche or very recent
  topics may surface fewer papers. Under heavy shared rate limits a run can come
  back empty — wait a minute and retry, or add the optional keys above.
- **Four databases are searched, and two of them are specialised.** Europe PMC
  covers biomedicine and arXiv covers computing, physics, engineering and
  statistics, so a given topic usually draws on one or the other rather than
  both. A clinical topic getting nothing from arXiv is normal. Papers found only
  on arXiv are preprints — the reference list labels them "arXiv preprint", and
  a preprint has not been peer reviewed.
- Search results, full texts and DOI lookups are cached for 24 hours in one
  file, `~/.articlegen/cache.json`, so re-running the same topic (in the CLI
  or the web app) makes no search requests and costs nothing against those
  shared limits. This matters more than it sounds: the free tiers refuse often
  enough that a second attempt at the same query is likelier to fail than to
  find anything new. A refusal is remembered for 2 minutes only. To put the
  file somewhere else set `ARTICLEGEN_CACHE_DIR=/some/folder`; to switch
  caching off entirely (memory and file) set `ARTICLEGEN_SEARCH_CACHE_TTL=0`.
  Delete the file to force a fresh search. It holds paper metadata and open
  access text, never an API key.
- The article is AI-written from **abstracts, plus the open-access full texts**
  of the most relevant sources when an open-access copy can be retrieved. The
  Methods section and Table 1's Read column state exactly how deeply each
  source was read. Treat it as a well-sourced starting point: follow the source
  links before relying on any specific claim.

### Full text via the `papers` CLI (optional)

With the `papers` CLI installed (from the separate `paperfetch` project),
full text is retrieved from any open-access copy across Unpaywall, OpenAlex,
Semantic Scholar, and preprint servers — not just Europe PMC — so non-biomedical
topics and arXiv papers stop being abstract-only.

- **Install:** two packages provide the same `papers` console command — the
  private `paperfetch` repository (used locally, and the one this project
  depends on for `queued_ckn`/CKN behaviour) and the public
  [`paperfetch-oa`](https://github.com/bartholomewtj/paperfetch-oa) (used only
  in the hosted Docker image). `pip install -e` whichever one applies to your
  machine, then set `PAPERS_MAILTO=you@example.com` to a real email address
  (scholarly APIs require it and block made-up addresses; defaults to
  `OPENALEX_MAILTO` if unset). **Do not install both on the same machine** —
  `paperfetch-oa` would shadow the private package.
- **Executable path:** If `papers` is not on your PATH, set
  `ARTICLEGEN_PAPERS_CMD="python -m papers"`.
- **CKN miss list:** Local drafts set `PAPERS_NO_CKN_QUEUE=1` on the
  `papers` subprocess, so a paywalled cite is not added to the CKN pickup
  list. `papers get` on its own still queues as usual.
- **One process per run:** Cited DOIs go to `papers get -` on stdin (one
  per line) so Semantic Scholar's 429 skip lives for the whole list. An
  older `papers` that rejects batch/stdin still gets one process per DOI.
  Before the first fetch, `papers status` runs once and the log names any
  missing `mailto_set` or `s2_key_set`.
- **Optional:** Without `papers`, articlegen behaves exactly as before,
  retrieving full text from Europe PMC only.
- **Hosted deployment:** The Render backend installs the public
  [`paperfetch-oa`](https://github.com/bartholomewtj/paperfetch-oa) package in
  its Docker image instead of the private `paperfetch` — same `papers`
  console command, no GitHub credentials needed to build. So the public web
  app also reads full text beyond Europe PMC. Local `articlegen draft` should
  keep using the private `paperfetch`; never install `paperfetch-oa` on a dev
  machine, it would shadow it.

## Layout

```
articlegen/
  cli.py        subcommands (ideas / draft / render / queue / demo / web)
  pipeline.py   the draft pipeline — every caller, CLI and web, runs this one
  web.py        HTTP server + JSON API behind the web front end
  gallery.py    public visitor gallery (local disk or a GitHub gist)
  llm.py        provider layer: OpenRouter or Claude, auto-detected from keys
  ideas.py      LLM call: theme -> shortlist of briefing questions
  writer.py     LLM calls: plan queries, write the briefing (write_article is --long)
  sources.py    Semantic Scholar + OpenAlex + Europe PMC + arXiv fetching,
                dedupe, ranking
  paperfetch.py optional: full text via the separate papers CLI (paperfetch)
  render.py     structured article -> journal-format HTML, Markdown, drafts/ index
  verify.py     deterministic check of every figure against the abstracts
  style.py      deterministic check of the prose against journal writing conventions
  demo.py       built-in sample for `articlegen demo`
tests/
  test_offline.py             pure-logic tests (no keys, no network)
  test_journal_conformance.py the journal conventions, as assertions over
                              rendered fixtures — run both before shipping
  test_replays.py             real run manifests, replayed through the
                              writer-free stages (tests/replays/)
```

## Tests

```bash
python tests/test_offline.py             # provider, citations, render blocks
python tests/test_journal_conformance.py # journal conventions over 5 fixtures
python tests/test_replays.py             # real run manifests replayed offline
```

`tests/replays/` holds run manifests copied verbatim from `drafts/` — a
manifest is the JSON `articlegen draft` also writes: the whole pool of papers
it screened, the relevance labels, the full-text excerpts the writer was
shown, and the article it wrote. `test_replays.py` reruns the deterministic
stages (`verify.check_statistics`, `style.check_style`, `sources.full_text_order`,
`writer.cite_target`, both renderers) against a real manifest and pins the
counts the run had when it was accepted. That catches what a hand-built
fixture cannot: a fixture only ever proves the code agrees with data someone
wrote to make it pass, while a real manifest is data nobody adjusted to be
convenient. Files in `tests/replays/` are the record of a real run — do not
edit, reformat or regenerate them.

Run them as scripts, the way CI does. Both print one `OK`/`FAIL` line per
check, collect every failure, and exit non-zero — so one broken check does not
hide the rest of the run.

pytest is not a dependency and CI does not use it. It works if you have it
(`python -m pytest tests`), but only because both suites now guard against the
two ways it used to lie:

- `check()` in `test_offline.py` records failures instead of raising. pytest saw
  no exception and called every test passed while the suite printed `FAIL`.
  It now raises as well when `PYTEST_CURRENT_TEST` is set.
- `test_journal_conformance.py` had no `test_*` function, so pytest collected
  nothing from it and still printed a green total. `test_journal_conventions()`
  wraps the whole suite as one case.

If you add a check helper or a new suite here, make it fail under both runners.

## Tools

Small opt-in scripts in `tools/`, not part of the pipeline and not run by CI:

- `compare_models.py` — runs (or replays) two models on the same topic/inputs
  and prints their countable metrics side by side.
- `compare_curation.py` — compares curation (relevance labelling) output
  across models on the same paper pool.
- `claim_probe.py` — a probe, not a gate. It shows a cheap second model one
  cited sentence at a time, plus the exact excerpt the deterministic verifier
  already checks it against, and asks whether the sentence overreaches —
  something `verify.py` cannot see, since it only checks numbers. Running it
  costs real OpenRouter credit. It reports agreement with a reading done by
  hand (`python tools/claim_probe.py --score`); see
  `app_docs/46b0fb70_claim-support-probe.md` for the current numbers. It is a
  probe and not a gate — `verify.py` is still the only thing that ever flags
  an article, and nothing here revises one.

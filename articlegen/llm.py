"""Provider layer: one `generate_json()` call, backed by OpenRouter, Anthropic
(Claude), or a Claude subscription via the Claude Code CLI.

Provider resolution, in priority order:
1. The model name, when given: `cli:*` -> the Claude Code CLI, a `vendor/model`
   slug -> OpenRouter, `claude-*` -> Anthropic. `cli:` is checked first because
   it is an explicit instruction and its suffix is a bare alias ("opus") that
   matches nothing else. The slash is checked before the `claude` prefix
   because OpenRouter re-sells the other providers' models under names like
   `anthropic/claude-sonnet-5`, which must not route to Anthropic directly.
2. An explicitly passed `api_key`, by its prefix: `sk-ant-` -> Anthropic,
   `sk-or-` -> OpenRouter.
3. ARTICLEGEN_PROVIDER env var ("openrouter", "anthropic", "claude-cli").
4. Whichever API key is present: OPENROUTER_API_KEY, ANTHROPIC_API_KEY.
5. Fallback: OpenRouter (the default provider).

**`claude-cli` is never reached by steps 2 or 4, only by 1 or 3.** It has no
API key to detect, so there is nothing to auto-detect it *by* — and that
suits it, because it is the one provider that must stay opt-in. It answers as
whoever is signed into the CLI on the machine, and it needs a `claude` binary
that no deployment host has.

Keys are passed **per call**, never through `os.environ`. The server handles
concurrent requests on different threads, and the environment is process-global:
setting a caller's key there lets one request's pipeline pick up another's key
several seconds later. `api_key=None` falls back to the environment, which is
what the CLI wants and what a single-user local run has always done.

OpenRouter is the default: it is what the web app offers, it bills prepaid
credit rather than a daily quota, and any catalogue model is reachable with
`--model`. Anthropic is opt-in — set ARTICLEGEN_PROVIDER=anthropic, pass a
`claude-*` --model, or run with only an Anthropic key.

**Groq was removed** (see the notes on `_format_sources` in `writer.py`). Its
free tier metered 12,000 tokens/minute *including the reserved output*, which
is why this module once carried a `prompt_budget_chars()` and the writer
trimmed abstracts to fit. No remaining provider has a per-minute ceiling, so
the whole trimming path went with it and every draft can now be grounded in
full text. Don't reinstate a char budget without a provider that needs one.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Claude Fable 5 is the top of the Claude 5 family, above Opus in capability.
# `claude-opus-5` and `claude-opus-4-8` still work if you pass them with
# --model. Keep OPENROUTER_PUBLIC_MODEL as the hosted web model:
# web.ALLOWED_MODELS is built from the constants here, and the public path
# always writes with this name so a stale front end cannot pick another.
ANTHROPIC_DEFAULT_MODEL = "claude-fable-5"
# Claude Opus 5, resold by OpenRouter at $5/$25 per million tokens. Fable 5 was
# the default briefly and cost twice as much for a model whose extra capability
# this pipeline never needed. Both run elevated bio/cyber safety classifiers
# that false-positive on clinical and life-sciences topics — this project's
# entire subject matter — which is why the refusal fallback below is not
# optional (issue #79). Pass any catalogue slug with --model:
# `meta-llama/llama-3.3-70b-instruct` when cost matters more than prose,
# `anthropic/claude-sonnet-5` for a cheaper Claude that carries no elevated
# classifiers, `anthropic/claude-fable-5` for the hardest topics.
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-opus-5"
# What the hosted web app writes with. Visitors do not supply a key.
# Measured against Gemini 3.7 Flash on three psychiatry ED topics (lithium
# toxicity, NMS, catatonia): Luna keeps the question on the topic and names
# the shape of the evidence; Flash writes a textbook summary. CLI default
# stays Opus. The public path must not silently inherit that default — a
# crafted POST could otherwise bill the hosted key at $5/$25.
OPENROUTER_PUBLIC_MODEL = "openai/gpt-5.6-luna"

# Where to retry when a safety classifier declines. OpenRouter cannot pass
# Anthropic's server-side `fallbacks` parameter (Claude API only), so this
# client-side retry is the OpenRouter equivalent of what the direct Anthropic
# path gets for free (#45). Sonnet 5 is the substitute precisely because it
# carries no elevated bio/cyber classifiers — falling back from one
# elevated-classifier model to another would reproduce the refusal.
OPENROUTER_REFUSAL_FALLBACK = "anthropic/claude-sonnet-5"
DEFAULT_PROVIDER = "openrouter"

# ---- the Claude Code CLI, i.e. "run this on my Claude subscription" --------
#
# A Claude.ai subscription issues no API key, so none of the three providers
# above can use one. `claude -p` can: it is the same subscription seat, driven
# non-interactively. Opt in with `--model cli:opus` (or `cli:sonnet`), or
# ARTICLEGEN_PROVIDER=claude-cli.
#
# **Local runs only, and deliberately not in web.ALLOWED_MODELS.** The Render
# host has no `claude` binary and no seat to authenticate against, so offering
# this on the web app would advertise a provider that cannot work there.
# Drafting on the subscription is a CLI activity.
CLAUDE_CLI_DEFAULT_MODEL = "opus"
CLAUDE_CLI_PREFIX = "cli:"
# The CLI loads every configured MCP server's tool schemas into the system
# prompt of each invocation. Measured on this machine: 22,944 prompt tokens
# with them, 2,363 without, for the same trivial call — a 10x tax, paid per
# call, roughly eight times an article, for servers the model is not allowed
# to use anyway (`--tools ""`). An empty config with --strict-mcp-config is
# what actually suppresses them; --tools alone does not.
CLAUDE_CLI_BASE_ARGS = (
    "--print",
    "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--mcp-config", '{"mcpServers":{}}',
)
# Subscription time is not billed per token, so there is no reason to buy a
# cheaper answer — unlike the metered paths, where effort is a cost decision.
CLAUDE_CLI_EFFORT = "high"
CLAUDE_CLI_TIMEOUT = 900

# ---- the Antigravity CLI, i.e. "run this on my Gemini subscription" --------
#
# Same shape and the same reasoning as claude-cli above: a Gemini subscription
# issues no API key, and `agy -p` is that seat driven non-interactively. Opt in
# with `--model agy:gemini-3.6-flash-high`, or ARTICLEGEN_PROVIDER=gemini-cli.
# **Local runs only, and deliberately not in web.ALLOWED_MODELS** — it answers
# as whoever is signed in on this machine.
#
# The name after the prefix is passed to `agy --model` verbatim; `agy models`
# lists them. Effort is part of the model name there (-high/-medium/-low), so
# there is no separate effort flag to set.
#
# Two things this provider does that the Claude one does not:
#
# - `--json-schema` actually enforces the schema, and the parsed object comes
#   back in the envelope's `structured_output`. That makes it the one CLI path
#   that is as reliable as the API paths, so the brace-matching recovery below
#   is a fallback rather than the main road.
# - The prompt is handed over as `@<file>`, which the CLI inlines verbatim
#   before the model is called. `agy` does not read stdin — it ignores it — and
#   the article prompt runs to ~95,000 characters against a 32,767-character
#   Windows command line, so a file reference is the only transport that fits.
#   Asking the agent to *open* a file instead is not equivalent: it costs a
#   tool round-trip and, on one test, it opened a different file in the same
#   directory and answered from that.
GEMINI_CLI_DEFAULT_MODEL = "gemini-3.6-flash-high"
GEMINI_CLI_PREFIX = "agy:"
GEMINI_CLI_TIMEOUT = 900

# **Every call runs the model the operator named. Do not step the shallow ones
# down a tier.** Reasoning effort is a suffix on the model id here (`agy models`
# lists `gemini-3.6-flash-high/-medium/-low`), so it is a one-line change to
# make and it looks free: on a measured run 57,687 of 65,383 output tokens were
# thinking, and the two `deep=False` stages were most of the cheap-looking half
# of that. It was tried and measured, and it costs more than it saves.
#
# `curate_sources` at -low: 827 output tokens against 8,684, and 14 of 20
# relevance labels agreeing with -high. The disagreements all ran one way —
# everything collapses toward "related". It called two papers related that were
# squarely on topic (level of care after an overdose *with self-harm intent*;
# alcohol intoxication in *suicidal patients*, in a review of ED length of stay
# for self-harm) and two more related that were background (homelessness;
# injury mechanism in TBI). That is the relevance gate, which is what stops
# topic drift: under-calling `direct` lowers `style._required_sections` so a
# thinner article passes, and under-calling `tangential` puts background
# abstracts back into the writer's prompt.
#
# `plan_queries` at -low and at -medium: both emit a run-on fourth query with
# two queries concatenated into one string, and -medium also returns a whole
# sentence where `core_entity` wants a term. Four well-formed queries from -high
# cost ~1,100 extra output tokens, and they decide which papers the run ever
# sees.
#
# Both cheap tiers report thinking=0 on these prompts, which is the tell: the
# saving is entirely "it stopped thinking", and these two calls are where the
# thinking was doing something.

_PROVIDER_DEFAULT_MODELS = {
    "anthropic": ANTHROPIC_DEFAULT_MODEL,
    "openrouter": OPENROUTER_DEFAULT_MODEL,
    "claude-cli": CLAUDE_CLI_DEFAULT_MODEL,
    "gemini-cli": GEMINI_CLI_DEFAULT_MODEL,
}


def resolve_provider(model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """Return (provider, model). `model` may be empty -> use the provider's default."""
    if model:
        # Checked before everything else, including the slash: `cli:` is an
        # explicit routing instruction from the operator, and the name after it
        # is a CLI alias ("opus", "sonnet") that would otherwise fall through to
        # the default provider under a name it does not recognise.
        if model.startswith(CLAUDE_CLI_PREFIX):
            return "claude-cli", model[len(CLAUDE_CLI_PREFIX):] or CLAUDE_CLI_DEFAULT_MODEL
        if model.startswith(GEMINI_CLI_PREFIX):
            return "gemini-cli", model[len(GEMINI_CLI_PREFIX):] or GEMINI_CLI_DEFAULT_MODEL
        # Checked before the `claude` prefix: OpenRouter re-sells other
        # providers' models as `vendor/model`, and `anthropic/claude-sonnet-5`
        # sent to Anthropic's own SDK is a 404. The slash is what makes a name
        # an OpenRouter slug; no direct provider's model id contains one.
        if "/" in model:
            return "openrouter", model
        if model.startswith("claude"):
            return "anthropic", model
        # A bare Groq-era model name. Without this it falls through to
        # OpenRouter, which has no such slug, and the run dies several seconds
        # later on a 404 that names neither the removed provider nor the fix.
        if model.startswith(("llama", "mixtral", "gemma", "deepseek", "qwen", "groq")):
            raise RuntimeError(
                f"'{model}' is a bare model name with no provider. Groq has been "
                "removed; reach the same model through OpenRouter with a "
                "vendor/model slug, e.g. --model meta-llama/llama-3.3-70b-instruct."
            )

    forced = os.environ.get("ARTICLEGEN_PROVIDER", "").strip().lower()
    if api_key and api_key.startswith("sk-ant-"):
        provider = "anthropic"
    elif api_key and api_key.startswith("sk-or-"):
        provider = "openrouter"
    elif forced in _PROVIDER_DEFAULT_MODELS:
        provider = forced
    elif os.environ.get("OPENROUTER_API_KEY"):
        provider = "openrouter"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    else:
        provider = DEFAULT_PROVIDER  # OpenRouter by default

    return provider, model or _PROVIDER_DEFAULT_MODELS[provider]


def generate_json(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    model: str | None = None,
    deep: bool = False,
    api_key: str | None = None,
) -> dict:
    """Run one structured-output generation and return the parsed JSON object.

    `deep=True` is for the long article call: bigger output budget and, on
    Anthropic, streaming + adaptive thinking at high effort.

    `api_key` overrides the environment for this call only — see the module
    docstring for why the server must never route keys through `os.environ`.
    """
    provider, model = resolve_provider(model, api_key)
    print(f"[articlegen] using provider={provider} model={model}", file=sys.stderr, flush=True)
    if provider == "openrouter":
        return _openrouter_generate(prompt, schema, system, model, deep, api_key)
    if provider == "claude-cli":
        return _claude_cli_generate(prompt, schema, system, model)
    if provider == "gemini-cli":
        return _gemini_cli_generate(prompt, schema, system, model)
    return _anthropic_generate(prompt, schema, system, model, deep, api_key)


# -------------------------------------------------- shared response handling

def _clean_json_text(text: str) -> str:
    """Strip a markdown code fence from a reply that should be a bare object.

    Used by the OpenRouter and CLI paths. Both can come back fenced: OpenRouter
    fronts many providers and the weaker ones treat `response_format` as a hint,
    and the CLI has no `response_format` at all.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


# ---------------------------------------------------------------- claude cli

def _extract_json_object(text: str) -> str:
    """Pull the first complete JSON object out of a reply that also has prose.

    Only the CLI path needs this. The three API providers are given a
    `response_format`, so the whole reply is the object; here the schema is
    only ever a request, and the first thing this provider actually did was
    answer a JSON-schema prompt in YAML.

    Brace-matching rather than a regex because the article payload nests
    several levels deep, and string-aware because abstracts contain braces and
    escaped quotes.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text


def _repair_json_text(text: str) -> str:
    """Single-pass deterministic repair of common JSON syntax slips.

    Deterministic rather than an LLM retry: asking a model to fix its own
    JSON rewrites the prose and risks losing the article (#147).

    Applies three narrow string-aware fixes:
    1. Escapes bare newlines (\\n) and carriage returns (\\r) inside string literals.
    2. Drops trailing commas before '}' or ']'.
    3. Inserts a comma between adjacent value boundaries: closing token
       ('"', '}', ']') followed by an opening token ('"', '{', '[').
    """
    out: list[str] = []
    in_string = False
    escaped = False
    n = len(text)

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
            elif ch == "\\":
                escaped = True
                out.append(ch)
            elif ch == '"':
                in_string = False
                out.append(ch)
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j < n and text[j] in ('"', '{', '['):
                    out.append(",")
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            else:
                out.append(ch)
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ("}", "]"):
                continue
            out.append(ch)
        elif ch in ("}", "]"):
            out.append(ch)
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ('"', '{', '['):
                out.append(",")
        else:
            out.append(ch)

    return "".join(out)


def _repair_json(text: str) -> dict | None:
    """Parse candidate text after deterministic repair, accepting only dicts.

    The dict check is load-bearing: callers of generate_json expect a mapping,
    and repairing non-dict JSON (such as a list '[1, 2,]') would hand the
    pipeline an unusable payload.
    """
    candidate = _repair_json_text(text)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# Appended to the end of the user prompt, not just the system prompt. The
# system prompt already said "JSON only" when Sonnet replied in YAML — with a
# long source payload in between, the instruction nearest the end of the
# context is the one that survives.
_CLI_JSON_DEMAND = (
    "\n\n---\n"
    "OUTPUT FORMAT — this overrides any formatting implied above.\n"
    "Reply with one JSON object and nothing else. The first character of your "
    "reply must be `{` and the last must be `}`. No prose, no preamble, no "
    "explanation, no markdown fences, no YAML."
)

_CLI_JSON_RETRY = (
    "Your previous reply was not valid JSON, so it could not be used. "
    "Send the same content again as a single raw JSON object matching the "
    "schema — starting with `{`, ending with `}`, with nothing before or "
    "after it.\n\n"
)


def _claude_cli_generate(prompt: str, schema: dict, system: str | None, model: str) -> dict:
    """Generate via `claude -p`, so the call is drawn from a Claude subscription.

    No `api_key` parameter, and that is not an oversight: this path has no key
    to pass. It authenticates as whoever is signed into the CLI on this
    machine, which makes it single-tenant by construction — the one provider
    the threaded server must never offer, since every request would be billed
    to, and answered as, the host's own seat.

    `deep` is ignored too. The other three paths use it to size an output
    ceiling; the CLI has no such parameter, and effort is set high regardless
    (see CLAUDE_CLI_EFFORT).
    """
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError(
            "The `claude` CLI is not on PATH, so provider claude-cli cannot run. "
            "Install Claude Code (npm i -g @anthropic-ai/claude-code) and sign in, "
            "or pick a provider that takes an API key."
        )

    # Everything big goes on stdin; only small fixed flags go in `args`.
    #
    # `claude` on Windows is a .cmd shim, so the command line is built by
    # cmd.exe — whose limit is 8,191 characters, not the 32,767 of a native
    # CreateProcess call. _WRITER_SYSTEM (8,168) plus the article schema
    # (3,102) is ~11KB, so passing the system prompt as an argument failed
    # with "The command line is too long" at the article stage, four calls and
    # several minutes into a run. There is no --system-prompt-file to reach
    # for, so the caller's system text is delivered as a preamble on stdin
    # instead, and --system-prompt carries only the short fixed contract below.
    contract = (
        "You are a JSON API. You reply with exactly one JSON object and nothing else: "
        "no prose, no preamble, no explanation, no markdown fences, no YAML. "
        "The first character of every reply is `{` and the last is `}`. "
        "Your instructions and the schema you must satisfy arrive in the message."
    )
    args = [
        exe, *CLAUDE_CLI_BASE_ARGS,
        "--model", model,
        "--effort", CLAUDE_CLI_EFFORT,
        "--system-prompt", contract,
    ]

    # Same belt-and-braces as the OpenRouter path, and here it is the
    # only brace there is: the CLI exposes no response_format, so nothing but
    # the prompt constrains the shape of what comes back.
    preamble = (
        (f"{system}\n\n" if system else "")
        + "You must respond ONLY with a valid JSON object matching this schema. "
        "Every required property must be present, spelled exactly as the schema "
        "spells it.\n"
        f"JSON Schema:\n{json.dumps(schema)}\n\n---\n\n"
    )

    def _run(stdin_text: str) -> tuple[dict, str]:
        # Two reasons for a scratch cwd. The CLI auto-discovers CLAUDE.md from
        # the working directory, and running inside this repo would prepend
        # articlegen's own project memory — several thousand tokens about how
        # the pipeline works — to a call whose job is to write a paragraph
        # about clinical evidence. It also keeps the subprocess pointed away
        # from the repo it is being run from.
        #
        # The prompt goes on stdin, not in `args`: the source payload runs to
        # tens of kilobytes, and Windows caps a command line at 32,767 chars.
        try:
            with tempfile.TemporaryDirectory() as scratch:
                proc = subprocess.run(
                    args, input=stdin_text, cwd=scratch, capture_output=True,
                    text=True, encoding="utf-8", timeout=CLAUDE_CLI_TIMEOUT,
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"`claude -p` did not answer within {CLAUDE_CLI_TIMEOUT}s (model {model}). "
                "Long articles at high effort are slow; retry, or use a metered provider."
            ) from None

        if proc.returncode != 0:
            raise RuntimeError(
                f"`claude -p` exited {proc.returncode} (model {model}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"`claude -p` did not return a JSON envelope (model {model}): {exc}\n"
                f"First 500 chars: {proc.stdout[:500]!r}"
            ) from exc

        # A declined request is a successful invocation carrying a refusal,
        # exactly as on the Anthropic path — surfaced here rather than left to
        # fail later as "invalid JSON" pointing at the model's own apology.
        if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
            raise RuntimeError(
                f"`claude -p` reported an error (model {model}, "
                f"subtype={envelope.get('subtype')}, stop_reason={envelope.get('stop_reason')}): "
                f"{str(envelope.get('result'))[:500]}"
            )

        usage = envelope.get("usage") or {}
        print(
            f"[articlegen] claude-cli in={usage.get('input_tokens', 0)} "
            f"cached={usage.get('cache_read_input_tokens', 0)} "
            f"out={usage.get('output_tokens', 0)} "
            # The other CLI provider, reported the same way. It is the control:
            # both wrap a subscription, only one shows the unexplained input.
            f"sent[{_prompt_size(stdin_text)}] "
            f"({envelope.get('duration_ms', 0) / 1000:.1f}s)",
            file=sys.stderr, flush=True,
        )
        return envelope, _extract_json_object(_clean_json_text(envelope.get("result") or ""))

    # One retry, because format compliance is the failure mode this provider
    # actually has. The API paths get `response_format` and cannot produce
    # prose; here the first real call answered a JSON-schema prompt in YAML,
    # which cost the whole run at the first of eight stages. This is not the
    # refusal retry the OpenRouter path does — a refusal raises above and is
    # not retried, since asking the same model again gets the same answer.
    envelope, cleaned = _run(preamble + prompt + _CLI_JSON_DEMAND)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_json(cleaned)
        if repaired is not None:
            print(
                f"[articlegen] claude-cli {model} returned near-miss JSON; "
                f"repaired deterministically ({exc})",
                file=sys.stderr, flush=True,
            )
            return repaired
        print(
            f"[articlegen] claude-cli {model} replied with non-JSON; retrying once",
            file=sys.stderr, flush=True,
        )

    envelope, cleaned = _run(_CLI_JSON_RETRY + preamble + prompt + _CLI_JSON_DEMAND)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_json(cleaned)
        if repaired is not None:
            print(
                f"[articlegen] claude-cli {model} returned near-miss JSON; "
                f"repaired deterministically ({exc})",
                file=sys.stderr, flush=True,
            )
            return repaired
        raise RuntimeError(
            f"{model} did not return valid JSON via the CLI, twice (stop_reason="
            f"{envelope.get('stop_reason')}): {exc}\n"
            "The CLI cannot enforce a response schema the way the API providers can, "
            "so a conversational reply lands here (repair did not help).\n"
            f"First 500 chars: {cleaned[:500]!r}"
        ) from exc


# ---------------------------------------------------------------- gemini cli

def _gemini_cli_generate(prompt: str, schema: dict, system: str | None, model: str) -> dict:
    """Generate via `agy -p`, so the call is drawn from a Gemini subscription.

    No `api_key` and no `deep`, for the same reasons as `_claude_cli_generate`:
    there is no key to pass, and the CLI has no output ceiling to size. Effort
    is not varied per call either — see the note above `GEMINI_CLI_DEFAULT_MODEL`
    for the measurement that settled it.

    Unlike that path, the schema is enforced rather than requested — `agy
    --json-schema` returns the parsed object in the envelope's
    `structured_output`. The text fallback below is for the case where a future
    version stops populating it, not for the ordinary path.
    """
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("agy")
    if not exe:
        raise RuntimeError(
            "The `agy` CLI is not on PATH, so provider gemini-cli cannot run. "
            "Install the Antigravity CLI and sign in, or pick a provider that "
            "takes an API key."
        )

    preamble = (
        (f"{system}\n\n" if system else "")
        + "Respond with a single JSON object matching the schema you were given. "
        "Every required property must be present, spelled exactly as the schema "
        "spells it.\n\n---\n\n"
    )

    def _run(prompt_text: str) -> tuple[dict, str, dict | None]:
        # A scratch cwd for the same reason as the Claude path: keep the
        # subprocess pointed away from this repo and its project memory.
        with tempfile.TemporaryDirectory() as scratch:
            prompt_path = os.path.join(scratch, "prompt.txt")
            schema_path = os.path.join(scratch, "schema.json")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f)

            args = [
                exe,
                # The whole prompt arrives as an inlined file. See the note at
                # GEMINI_CLI_PREFIX for why neither stdin nor a plain argument
                # can carry it.
                "-p", f"@{prompt_path}",
                "--add-dir", scratch,
                "--model", model,
                "--output-format", "json",
                "--json-schema", schema_path,
                # Sources are arbitrary text and a paragraph can begin with a
                # slash; nothing in a prompt should be read as a command.
                "--disable-slash-commands",
            ]
            try:
                proc = subprocess.run(
                    args, cwd=scratch, capture_output=True, text=True,
                    encoding="utf-8", timeout=GEMINI_CLI_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"`agy -p` did not answer within {GEMINI_CLI_TIMEOUT}s (model {model}). "
                    "Retry, or use a metered provider."
                ) from None

        if proc.returncode != 0:
            raise RuntimeError(
                f"`agy -p` exited {proc.returncode} (model {model}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"`agy -p` did not return a JSON envelope (model {model}): {exc}\n"
                f"First 500 chars: {proc.stdout[:500]!r}"
            ) from exc

        if envelope.get("status") != "SUCCESS":
            raise RuntimeError(
                f"`agy -p` reported status={envelope.get('status')} (model {model}): "
                f"{str(envelope.get('response'))[:500]}"
            )

        usage = envelope.get("usage") or {}
        print(
            f"[articlegen] gemini-cli {model} in={usage.get('input_tokens', 0)} "
            f"cached={usage.get('cache_read_tokens', 0)} "
            f"out={usage.get('output_tokens', 0)} "
            f"thinking={usage.get('thinking_tokens', 0)} "
            # `agy` is an agent, not a completion endpoint: it may take several
            # internal turns, and each one re-sends the context. A call that
            # costs far more than its prompt explains is a multi-turn call, and
            # without this the only symptom is an input count nobody can account
            # for.
            f"turns={envelope.get('num_turns', 1)} "
            # What was actually sent, beside what was charged. This is what
            # answered #116: `in=` tracks `sent` at a slope below 1 on a fixed
            # floor, so the prompt is counted once and the reported input was
            # never unexplained — see the note at `_CHARS_PER_TOKEN`. Keep the
            # pair logged; the question only stayed open as long as it did
            # because the charge was printed and the prompt never was.
            f"sent[{_prompt_size(prompt_text)} "
            f"schema={len(json.dumps(schema))}] "
            f"({envelope.get('duration_seconds', 0):.1f}s)",
            file=sys.stderr, flush=True,
        )
        structured = envelope.get("structured_output")
        text = _extract_json_object(_clean_json_text(envelope.get("response") or ""))
        return envelope, text, structured if isinstance(structured, dict) else None

    envelope, cleaned, structured = _run(preamble + prompt + _CLI_JSON_DEMAND)
    if structured:
        return structured
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(
            f"[articlegen] gemini-cli {model} replied with non-JSON; retrying once",
            file=sys.stderr, flush=True,
        )

    envelope, cleaned, structured = _run(_CLI_JSON_RETRY + preamble + prompt + _CLI_JSON_DEMAND)
    if structured:
        return structured
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{model} did not return valid JSON via `agy`, twice: {exc}\n"
            f"First 500 chars: {cleaned[:500]!r}"
        ) from exc


# ---------------------------------------------------------------- anthropic

# Claude Opus 5 (and the Fable/Mythos tier above it) runs safety classifiers
# that can decline a request outright: a normal HTTP 200 with stop_reason
# "refusal" and no text. The topics here are clinical and user-supplied, so a
# false positive is possible, and it used to cost the whole run (issue #45).
# The server-side fallback re-runs a declined request on Anthropic's
# recommended substitute model, routed by refusal category, inside the same
# call. "default" rather than a pinned model name, so nothing here needs
# migrating when a fallback model is deprecated. Models without these
# classifiers don't take the parameter, so it is attached by model prefix.
_REFUSAL_FALLBACK_MODELS = ("claude-opus-5", "claude-fable", "claude-mythos")
_REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


# How big the thing we sent actually was, so a reported input count can be
# compared against it without re-deriving anything.
#
# This settled #116. Over one full `agy` draft, reported input tracks the
# prompt at slightly under one token per estimated token, on a fixed floor:
#
#   call     sent ~tok        in    against 33,000 + 0.77 x ~tok
#   plan            235    33,858      +677
#   curate       13,095    43,315      +232
#   write        28,343    55,381      +557
#   revise       35,293   101,549   +41,373
#
# Three of the four land within ~700 tokens of that line, so the prompt is
# counted once. The standing hypothesis was that it arrived three ways at once
# — inlined by `-p @file`, exposed by `--add-dir` and sitting in `cwd` — and a
# slope below 1 rules that out. A direct A/B agrees: three runs each with and
# without `--add-dir` gave the same total context to within 0.1% (47,122 vs
# 47,088), and a canary buried at 80% depth of the prompt came back every time
# without it. `--add-dir` is not load-bearing for arrival and does not multiply
# the count.
#
# The "~20,000 characters" this used to be measured against counted only the
# brief and the draft. `revise_prose` also sends the SOURCES block, which
# `sources.FULLTEXT_TOTAL_CHARS` alone allows 60,000 characters of; the real
# revision prompt measured 141,173. There was never a 25x overcount.
#
# What is left: the revision call, and only it, sits ~41,000 above the line. It
# is the largest prompt and carries by far the most thinking (31,735 tokens).
# Not prompt duplication, cause not identified, and small enough to leave.
#
# ~4 characters per token is a rough English average and wrong in the third
# digit. That did not matter here: the question was whether the reported input
# was 1x or 25x what was sent, and a ratio that coarse answers it.
_CHARS_PER_TOKEN = 4


def _prompt_size(*parts: str) -> str:
    chars = sum(len(p) for p in parts if p)
    return f"chars={chars} ~tok={chars // _CHARS_PER_TOKEN}"


def _refusal_fallback_kwargs(model: str) -> dict:
    if not model.startswith(_REFUSAL_FALLBACK_MODELS):
        return {}
    return {"betas": [_REFUSAL_FALLBACK_BETA], "fallbacks": "default"}


def _anthropic_generate(prompt, schema, system, model, deep, api_key=None) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    kwargs.update(_refusal_fallback_kwargs(model))
    if system:
        kwargs["system"] = system

    if deep:
        with client.beta.messages.stream(
            max_tokens=64000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "high",
                "format": {"type": "json_schema", "schema": schema},
            },
            **kwargs,
        ) as stream:
            response = stream.get_final_message()
    else:
        # `max_tokens` caps thinking *and* the reply together, and on Claude
        # Opus 5 adaptive thinking is on whenever the parameter is omitted — so
        # the ceiling that was ample for a bare JSON reply on Opus 4.8 can now
        # truncate one mid-object. The curation call in particular grades the
        # whole candidate pool — forty sources by default.
        response = client.beta.messages.create(
            max_tokens=16000,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **kwargs,
        )

    if response.stop_reason == "refusal":
        # A safety classifier declined. This arrives as a normal 200 with no
        # text block, so without this check the generator below raises
        # StopIteration and the caller reports an empty, baffling failure.
        # On models with the server-side fallback attached, reaching here
        # means the fallback model declined too, not that no retry happened.
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None) or "unspecified"
        raise RuntimeError(
            f"The model declined this request ({category}). Rephrase the topic, "
            "or try --model claude-sonnet-5, which carries no elevated "
            "bio/cyber classifiers."
        )
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "The model hit its output limit before finishing the JSON. Try a "
            "narrower topic, or --max-papers with a smaller number."
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise RuntimeError(f"The model returned no text (stop_reason={response.stop_reason}).")
    return json.loads(text)


# ---------------------------------------------------------------- openrouter

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter bills the tokens actually generated, not the reservation, so a
# generous ceiling is free when it goes unused. It only has to be big enough
# for the reply, and no bigger than the model itself permits.
#
# Reasoning models (claude-fable-5, claude-sonnet-5) think inside `max_tokens`,
# and a truncated reply is invalid JSON, not a short one, so the failure lands
# as a parse error a long way from its cause. At 8000 the revision pass hit the
# ceiling on every reasoning-model run; at 8000 the *shallow* calls broke too
# once Fable became the default (issue #77) — the ideas call came back cut off
# 190 tokens in, having spent the rest of the budget thinking.
#
# The ceilings can't simply be raised for everyone: OpenRouter caps
# `meta-llama/llama-3.3-70b-instruct` at 16,384 completion tokens while the
# Anthropic models allow 128,000. Hence two pairs, chosen by model.
OPENROUTER_DEEP_OUTPUT = 16000
OPENROUTER_OUTPUT = 8000
# Matches what the direct Anthropic path reserves, for the same reason.
OPENROUTER_REASONING_DEEP_OUTPUT = 64000
OPENROUTER_REASONING_OUTPUT = 16000


def _openrouter_provider_routing(model: str) -> dict:
    """Which upstream endpoints OpenRouter may serve this model from.

    `require_parameters` stops a provider that ignores `response_format` from
    answering in prose — a failure that reads like the model being bad at
    instructions rather than a routing choice. It is necessary and it is not
    sufficient: it filters on what a provider *advertises*, and OpenRouter
    lists Azure as supporting structured outputs for the Anthropic models while
    the workspace behind it returns `400 structured_outputs not supported in
    your workspace` (issue #81).

    So Anthropic slugs are pinned to Anthropic's own endpoint, out of the nine
    that re-sell them. That is where structured outputs is GA rather than beta
    or workspace-gated, and where the refusal and thinking semantics match the
    direct Anthropic path — so the handling added for #79 behaves identically
    on both. Everything else keeps open routing, where competing providers are
    the point and the advertised capability is trustworthy enough.
    """
    routing = {"require_parameters": True}
    if model.startswith("anthropic/"):
        routing["only"] = ["anthropic"]
    return routing


def _openrouter_max_tokens(model: str, deep: bool) -> int:
    """How much room to leave for thinking plus the reply.

    Keyed off the vendor prefix rather than a list of model names: OpenRouter
    slugs are `vendor/model`, every Anthropic model it re-sells thinks, and a
    new one should get the headroom without an edit here. Anything else keeps
    the conservative pair, which fits inside Llama 3.3 70B's 16,384 ceiling.
    """
    if model.startswith("anthropic/"):
        return OPENROUTER_REASONING_DEEP_OUTPUT if deep else OPENROUTER_REASONING_OUTPUT
    return OPENROUTER_DEEP_OUTPUT if deep else OPENROUTER_OUTPUT


def _openrouter_generate(
    prompt: str, schema: dict, system: str | None, model: str, deep: bool,
    api_key: str | None = None, allow_fallback: bool = True,
) -> dict:
    import requests

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Create a key at https://openrouter.ai/keys and set OPENROUTER_API_KEY."
        )

    # Belt and braces: `response_format` is the
    # real constraint, but OpenRouter fronts many providers and the weaker ones
    # treat a schema as a hint, so the schema goes in the prompt too.
    system_instruction = (
        (system or "") +
        "\n\nIMPORTANT: You must respond ONLY with a valid JSON object matching this schema. "
        "Do NOT enclose in backticks or markdown fences. Do NOT output any conversational text or commentary.\n"
        f"JSON Schema:\n{json.dumps(schema)}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "articlegen", "strict": True, "schema": schema},
        },
        "provider": _openrouter_provider_routing(model),
        "temperature": 0.2,
        "max_tokens": _openrouter_max_tokens(model, deep),
    }

    def _post(body):
        return requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Attribution headers; optional, and they carry no user content.
                "HTTP-Referer": "https://github.com/bartholomewtj/article-generator",
                "X-Title": "Evidence brief",
            },
            json=body,
            timeout=180,
        )

    res = _post(payload)

    # `require_parameters` demands every sent parameter, not just
    # `response_format` — and reasoning models (e.g. anthropic/claude-sonnet-5)
    # expose no `temperature` at all, so its mere presence 404s with "No
    # endpoints found". Retry without it rather than give up `require_parameters`,
    # which is what actually protects the JSON output.
    if res.status_code == 404 and "temperature" in payload:
        retry = {k: v for k, v in payload.items() if k != "temperature"}
        res = _post(retry)

    if res.status_code == 401:
        raise RuntimeError("OpenRouter rejected the key. Check it at https://openrouter.ai/keys.")
    if res.status_code == 402:
        raise RuntimeError(
            "OpenRouter reports insufficient credit for this request. "
            "Top up at https://openrouter.ai/credits, or use a cheaper --model."
        )
    if res.status_code == 403 and "limit" in res.text.lower():
        # Distinct from 402: the account has credit, but this *key* carries a
        # spending cap that is now spent. The fix is on the key's own page, not
        # the credits page, and nothing about the request will change it.
        raise RuntimeError(
            "This OpenRouter key has hit its own spending limit. Raise or clear "
            "the limit on the key at https://openrouter.ai/keys — topping up "
            "credit will not help, the cap is set per key."
        )
    if res.status_code == 429:
        raise RuntimeError("OpenRouter is rate-limiting this key. Wait a moment and retry.")
    if res.status_code >= 400:
        raise RuntimeError(f"OpenRouter returned HTTP {res.status_code}: {res.text[:300]}")

    data = res.json()
    # A 200 can still carry an error body — an upstream provider refusing, or no
    # provider matching `require_parameters` for this model.
    if isinstance(data.get("error"), dict):
        message = data["error"].get("message") or "unspecified error"
        raise RuntimeError(f"OpenRouter could not complete the request: {message}")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)[:300]}")
    # Truncation is reported inconsistently: OpenRouter normalises to "length",
    # but the upstream provider's own word for it arrives in
    # `native_finish_reason` and sometimes only there — issue #77 was a
    # truncated reply whose normalised reason was not "length", so this check
    # missed it and the caller saw a JSON parse error instead of the real cause.
    finish = choices[0].get("finish_reason")
    native_finish = choices[0].get("native_finish_reason")

    # A safety classifier declined. OpenRouter normalises this to
    # "content_filter" and puts the provider's own word in
    # `native_finish_reason`; Anthropic's is "refusal". It can fire mid-reply,
    # so there may be partial content — which is a fragment, not an answer, and
    # is discarded. Claude Fable 5 runs elevated bio/cyber classifiers that
    # false-positive on clinical and life-sciences topics, and this project
    # writes about little else (issue #79).
    if finish == "content_filter" or native_finish == "refusal":
        if allow_fallback and model != OPENROUTER_REFUSAL_FALLBACK:
            print(
                f"[articlegen] {model} declined this request; retrying on "
                f"{OPENROUTER_REFUSAL_FALLBACK}",
                file=sys.stderr,
                flush=True,
            )
            return _openrouter_generate(
                prompt, schema, system, OPENROUTER_REFUSAL_FALLBACK, deep, api_key,
                allow_fallback=False,
            )
        # Reached only when the fallback itself declined, or when the requested
        # model already was the fallback — either way there is nothing left to try.
        raise RuntimeError(
            f"{model} declined this request on safety grounds. Rephrase the "
            f"topic, or try --model {OPENROUTER_REFUSAL_FALLBACK}, which "
            "carries no elevated bio/cyber classifiers."
        )

    if finish == "length" or native_finish in ("length", "max_tokens", "MAX_TOKENS"):
        raise RuntimeError(
            f"{model} hit its output limit ({_openrouter_max_tokens(model, deep)} "
            "tokens) before finishing the JSON. Reasoning models spend part of "
            "that budget thinking. Try a narrower topic, or --max-papers with a "
            "smaller number."
        )

    # The metered path reports the same two numbers, so "is the input count
    # plausible?" can be answered against a provider whose accounting is
    # independently verifiable — which is what #116 needs and had no baseline
    # for. Same line shape as the two CLI providers, so one grep collects all
    # three.
    usage = data.get("usage") or {}
    print(
        f"[articlegen] openrouter {model} in={usage.get('prompt_tokens', 0)} "
        f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)} "
        f"out={usage.get('completion_tokens', 0)} "
        f"sent[{_prompt_size(system, prompt)}]",
        file=sys.stderr, flush=True,
    )

    text_response = (choices[0].get("message") or {}).get("content") or ""
    cleaned = _clean_json_text(text_response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Name the finish reason: an unterminated string almost always means the
        # reply was cut off, and without this the error points at the JSON
        # rather than at the ceiling that truncated it.
        raise RuntimeError(
            f"OpenRouter returned invalid JSON from {model} "
            f"(finish_reason={finish!r}, native={native_finish!r}, "
            f"max_tokens={_openrouter_max_tokens(model, deep)}): {exc}\n"
            f"Raw response:\n{text_response[:500]}"
        ) from exc

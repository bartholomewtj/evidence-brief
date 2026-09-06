"""Offline smoke tests — no network, no API keys required.

Run with:  python tests/test_offline.py   (exits non-zero on failure)

These cover the pure logic that a fresh session can verify immediately:
provider resolution, citation renumbering,
statistic verification, source ranking, and the render blocks. The LLM calls
and scholarly-API fetches are NOT exercised here (they need keys/network) —
verify those with a live `theme:` issue on GitHub.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Ensure the repo root is importable when run as `python tests/test_offline.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The search cache is written to disk (~/.articlegen/cache.json by default).
# Every test that searches would otherwise leave its fake papers in the real
# file, so the whole run points the cache at a fresh temp folder. Set before
# articlegen is imported, and read by sources at first use, never at import.
import tempfile

os.environ["ARTICLEGEN_CACHE_DIR"] = tempfile.mkdtemp(prefix="articlegen-test-cache-")

FAILURES: list[str] = []


def check(name: str, cond: bool) -> None:
    """Record a failed check. Raises instead when pytest is the runner.

    Collecting rather than raising is deliberate for `python tests/…`: one bad
    check does not hide the rest of the run, and the summary lists everything
    that broke at once.

    pytest is not a dependency of this repo and CI does not use it, but it will
    happily collect these `test_*` functions if someone runs it — and a runner
    that only watches for exceptions called every one of them passing while
    this file printed FAIL and `main()` would have exited 1. That is a green
    local run followed by a red CI run, which is exactly backwards.

    `PYTEST_CURRENT_TEST` is set by pytest for the duration of each test, so
    the runner is detected without importing it. Read at call time, not import
    time: the variable is not reliably set while modules are being collected.
    """
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(name)


def test_provider_resolution() -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER"):
        os.environ.pop(var, None)
    from articlegen.llm import (
        resolve_provider, OPENROUTER_DEFAULT_MODEL, ANTHROPIC_DEFAULT_MODEL,
    )

    check("no keys -> openrouter default",
          resolve_provider() == ("openrouter", OPENROUTER_DEFAULT_MODEL))
    os.environ["ANTHROPIC_API_KEY"] = "x"
    check("only anthropic -> anthropic", resolve_provider() == ("anthropic", ANTHROPIC_DEFAULT_MODEL))
    os.environ["OPENROUTER_API_KEY"] = "y"
    check("both keys -> openrouter wins", resolve_provider()[0] == "openrouter")
    check("claude-* model name forces anthropic", resolve_provider("claude-opus-5")[0] == "anthropic")
    check("superseded claude-* names still route",
          resolve_provider("claude-opus-4-8")[0] == "anthropic")
    # Groq is gone. A bare Groq-era name must fail loudly here rather than
    # reaching OpenRouter as a slug it has never heard of and 404ing seconds
    # later, which names neither the removed provider nor the fix.
    try:
        resolve_provider("llama-3.3-70b-versatile")
        bare_llama_raises = False
    except RuntimeError as exc:
        bare_llama_raises = "meta-llama/llama-3.3-70b-instruct" in str(exc)
    check("a bare Groq-era name errors and names the replacement", bare_llama_raises)
    check("the resold llama slug is the supported route",
          resolve_provider("meta-llama/llama-3.3-70b-instruct")[0] == "openrouter")
    os.environ["ARTICLEGEN_PROVIDER"] = "anthropic"
    check("provider override respected", resolve_provider()[0] == "anthropic")
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER"):
        os.environ.pop(var, None)


def test_per_request_api_key() -> None:
    """Keys must travel as arguments, never through the process environment.

    The web server is threaded; an env-var handoff lets one request's pipeline
    pick up another request's key several seconds later, and bill it.
    """
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER"):
        os.environ.pop(var, None)
    from articlegen.llm import resolve_provider

    check("sk-ant- key -> anthropic", resolve_provider(None, "sk-ant-abc")[0] == "anthropic")
    check("sk-or- key -> openrouter", resolve_provider(None, "sk-or-v1-abc")[0] == "openrouter")
    check(
        "explicit model still beats the key prefix",
        resolve_provider("claude-opus-5", "sk-or-v1-abc")[0] == "anthropic",
    )

    # The whole point: a passed key must not leak into the environment.
    import inspect
    from articlegen import ideas, llm, web, writer

    for fn in (
        llm.generate_json, ideas.generate_ideas, writer.plan_queries,
        writer.curate_sources, writer.write_briefing, writer.write_article,
        writer.revise_prose,
    ):
        check(f"{fn.__name__} accepts api_key", "api_key" in inspect.signature(fn).parameters)

    # The guard used to be a grep for one exact spelling in two modules, on the
    # project's most security-relevant invariant: single quotes, a different
    # key name, `os.environ.update`, or an assignment in writer.py all passed
    # it (issue #98). Both halves below are behavioural, and cover every module
    # that can see a key.
    import re as _re

    from articlegen import cli, gallery, paperfetch, pipeline, sources

    modules = (llm, web, writer, ideas, pipeline, cli, sources, paperfetch, gallery)
    assign = _re.compile(
        r"os\.environ\s*(?:\[[^\]]*\]\s*=(?!=)|\.update\s*\(|\.setdefault\s*\()")
    for module in modules:
        hits = [
            line.strip()
            for line in inspect.getsource(module).splitlines()
            if assign.search(line) and not line.strip().startswith("#")
        ]
        check(f"{module.__name__} never writes into os.environ", not hits)

    # And the real thing: run a call with a fake transport and prove the
    # environment is untouched, whatever the code happens to look like.
    import requests

    snapshot = dict(os.environ)
    real_post = requests.post
    saved_anthropic = sys.modules.get("anthropic")
    try:
        requests.post = lambda *a, **kw: _FakeHTTP()
        llm.generate_json("p", {"type": "object"}, model="anthropic/claude-opus-5",
                          api_key="sk-or-v1-secret")
        _FakeAnthropic.install(_FakeAnthropic._Message(text="{}"))
        llm.generate_json("p", {"type": "object"}, model="claude-fable-5",
                          api_key="sk-ant-secret")
    except Exception:
        pass
    finally:
        requests.post = real_post
        if saved_anthropic is not None:
            sys.modules["anthropic"] = saved_anthropic
        else:
            sys.modules.pop("anthropic", None)

    check("a per-call key never reaches the environment", dict(os.environ) == snapshot)
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        check(f"{var} is not set by a call that was given a key",
              os.environ.get(var) == snapshot.get(var))
    check("openrouter key still falls back to the environment",
          'os.environ.get("OPENROUTER_API_KEY")' in inspect.getsource(llm._openrouter_generate))


class _FakeAnthropic:
    """Enough of the Anthropic SDK to drive `_anthropic_generate` offline.

    `_anthropic_generate` imports `anthropic` inside the function, so putting
    this in `sys.modules` is all it takes. Everything the real client is asked
    for is here: the constructor's `api_key`, `beta.messages.create`, the
    streaming context manager the deep path uses, and the kwargs of each call.
    """

    calls: list[dict] = []
    reply = None          # the response object the next call returns
    keys: list = []       # api_key seen by each constructor

    class _Message:
        def __init__(self, stop_reason="end_turn", text='{"ok": true}', category=None):
            self.stop_reason = stop_reason
            self.content = ([_FakeAnthropic._Block(text)] if text else [])
            if category:
                self.stop_details = type("D", (), {"category": category})()

    class _Block:
        def __init__(self, text):
            self.type, self.text = "text", text

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return _FakeAnthropic.reply

    class _Messages:
        def create(self, **kwargs):
            _FakeAnthropic.calls.append(kwargs)
            return _FakeAnthropic.reply

        def stream(self, **kwargs):
            _FakeAnthropic.calls.append(kwargs)
            return _FakeAnthropic._Stream()

    class Anthropic:
        def __init__(self, api_key=None):
            _FakeAnthropic.keys.append(api_key)
            self.beta = type("B", (), {"messages": _FakeAnthropic._Messages()})()

    @classmethod
    def install(cls, reply):
        cls.calls, cls.keys, cls.reply = [], [], reply
        sys.modules["anthropic"] = cls
        return cls


def test_anthropic_generate_behaviour() -> None:
    """The direct Anthropic path, exercised rather than grepped.

    Every production failure on record happened on the provider seam, and two
    of the three provider functions were covered only by reading their source
    (issue #98). A #77 (truncation) or #79 (refusal) class failure here would
    have passed a green run.
    """
    from articlegen import llm

    saved = sys.modules.get("anthropic")
    try:
        # -- happy path: the parsed object comes back, and the key is passed
        #    as an argument rather than reaching the environment ------------
        before = dict(os.environ)
        _FakeAnthropic.install(_FakeAnthropic._Message(text='{"title": "t"}'))
        out = llm._anthropic_generate("p", {"type": "object"}, "sys",
                                      "claude-fable-5", False, api_key="sk-ant-test")
        check("a good reply is parsed", out == {"title": "t"})
        check("the key goes to the client, not the environment",
              _FakeAnthropic.keys == ["sk-ant-test"] and dict(os.environ) == before)
        check("the schema is pinned as structured output",
              _FakeAnthropic.calls[0]["output_config"]["format"]
              == {"type": "json_schema", "schema": {"type": "object"}})
        check("the system prompt is sent separately from the user prompt",
              _FakeAnthropic.calls[0]["system"] == "sys"
              and _FakeAnthropic.calls[0]["messages"][0]["content"] == "p")

        # -- refusal (#79). This arrives as a normal 200 with no text block,
        #    so without the explicit check it surfaced as a bare StopIteration.
        _FakeAnthropic.install(
            _FakeAnthropic._Message(stop_reason="refusal", text="", category="bio"))
        try:
            llm._anthropic_generate("p", {}, None, "claude-opus-5", False)
            refusal_msg = ""
        except RuntimeError as exc:
            refusal_msg = str(exc)
        check("a refusal raises rather than dying on StopIteration",
              "declined this request" in refusal_msg)
        check("and names the category and a model without the classifier",
              "bio" in refusal_msg and "claude-sonnet-5" in refusal_msg)

        # -- truncation (#77). A truncated reply is invalid JSON, not a short
        #    one, so it must be named here rather than surfacing as a parse
        #    error nowhere near its cause.
        _FakeAnthropic.install(
            _FakeAnthropic._Message(stop_reason="max_tokens", text='{"title": "unfinis'))
        try:
            llm._anthropic_generate("p", {}, None, "claude-fable-5", False)
            truncation_msg = ""
        except RuntimeError as exc:
            truncation_msg = str(exc)
        check("truncation is reported as truncation, not as bad JSON",
              "output limit" in truncation_msg)

        # -- an empty reply with an unremarkable stop_reason ----------------
        _FakeAnthropic.install(_FakeAnthropic._Message(text=""))
        try:
            llm._anthropic_generate("p", {}, None, "claude-fable-5", False)
            empty_msg = ""
        except RuntimeError as exc:
            empty_msg = str(exc)
        check("an empty reply names the stop reason", "stop_reason=end_turn" in empty_msg)

        # -- malformed JSON from a model that claims to have finished -------
        _FakeAnthropic.install(_FakeAnthropic._Message(text="Here is your article!"))
        try:
            llm._anthropic_generate("p", {}, None, "claude-fable-5", False)
            malformed = False
        except json.JSONDecodeError:
            malformed = True
        except RuntimeError:
            malformed = True
        check("prose where JSON was demanded fails loudly", malformed)

        # -- the deep path uses the streaming API, and reserves more room ---
        _FakeAnthropic.install(_FakeAnthropic._Message(text="{}"))
        llm._anthropic_generate("p", {}, None, "claude-opus-5", True)
        deep_kwargs = _FakeAnthropic.calls[0]
        check("the deep call reserves more than the shallow one",
              deep_kwargs["max_tokens"] > 16000)
        check("and asks for adaptive thinking",
              deep_kwargs["thinking"] == {"type": "adaptive"})
        # Opus/Fable opt into the server-side refusal fallback by model prefix;
        # older models reject the beta outright.
        check("a fallback-eligible model carries the beta",
              deep_kwargs.get("fallbacks") == "default")
        _FakeAnthropic.install(_FakeAnthropic._Message(text="{}"))
        llm._anthropic_generate("p", {}, None, "claude-3-haiku", False)
        check("an older model does not", "fallbacks" not in _FakeAnthropic.calls[0])
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)


def _capture_openrouter(monkey_response, model="anthropic/claude-opus-5", deep=False):
    """Run `_openrouter_generate` against a fake transport; return sent payloads."""
    import requests

    from articlegen import llm

    sent: list[dict] = []
    real_post = requests.post

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return monkey_response(len(sent), json)

    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test"
    try:
        requests.post = fake_post
        try:
            llm._openrouter_generate("p", {"type": "object"}, "sys", model, deep)
        except Exception:
            pass          # the caller asserts on what was sent, not on the result
    finally:
        requests.post = real_post
        os.environ.pop("OPENROUTER_API_KEY", None)
    return sent


class _FakeHTTP:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {
            "choices": [{"finish_reason": "stop",
                         "message": {"content": '{"title": "t"}'}}]}
        self.text = text or "{}"

    def json(self):
        return self._body


def test_openrouter_request_is_asserted_on_the_payload() -> None:
    """Assert on what goes over the wire, not on the text of the function.

    `'"type": "json_schema"' in src` passes if the string sits in a comment or
    a dead branch, and breaks on a harmless refactor. These were the guards on
    the seam where #77, #79 and #81 all happened (issue #98).
    """
    from articlegen import llm

    sent = _capture_openrouter(lambda n, payload: _FakeHTTP())
    check("exactly one request when it succeeds", len(sent) == 1)
    payload = sent[0]
    check("structured output is demanded",
          payload["response_format"]["type"] == "json_schema")
    check("in strict mode", payload["response_format"]["json_schema"]["strict"] is True)
    check("and the schema travels with it",
          payload["response_format"]["json_schema"]["schema"] == {"type": "object"})
    # `require_parameters` filters on what a provider *advertises*, and the
    # advertisement can be wrong — OpenRouter listed an Azure endpoint as
    # supporting structured outputs and it returned 400 (#81).
    check("providers that ignore response_format are filtered out",
          payload["provider"]["require_parameters"] is True)
    check("and Anthropic slugs are pinned to Anthropic's own endpoint",
          payload["provider"]["only"] == ["anthropic"])

    llama = _capture_openrouter(lambda n, payload: _FakeHTTP(),
                                model="meta-llama/llama-3.3-70b-instruct")
    check("other vendors keep open routing", "only" not in llama[0]["provider"])

    # A reasoning model exposes no `temperature`, and `require_parameters`
    # demands every sent parameter — so sending it 404s with "No endpoints
    # found". The retry must drop temperature and keep the routing: giving up
    # `provider` instead would reach an endpoint that answers in prose.
    def temperature_404(n, payload):
        if n == 1:
            return _FakeHTTP(status=404, text="No endpoints found matching temperature")
        return _FakeHTTP()

    retried = _capture_openrouter(temperature_404)
    check("a temperature 404 is retried", len(retried) == 2)
    check("the retry drops temperature", "temperature" not in retried[1])
    check("and keeps the provider routing",
          retried[1]["provider"] == retried[0]["provider"])
    check("and keeps the schema", "response_format" in retried[1])

    # A refusal can arrive mid-reply with partial content. That fragment is
    # discarded, or it resurfaces as "invalid JSON" pointing at the model's own
    # half-written sentence (#79).
    def refusal_then_ok(n, payload):
        if n == 1:
            return _FakeHTTP(body={"choices": [{
                "finish_reason": "content_filter",
                "native_finish_reason": "refusal",
                "message": {"content": '{"title": "I cannot'}}]})
        return _FakeHTTP()

    fell_back = _capture_openrouter(refusal_then_ok)
    check("a refusal is retried on the fallback model", len(fell_back) == 2)
    check("and the fallback is a model without elevated classifiers",
          fell_back[1]["model"] == llm.OPENROUTER_REFUSAL_FALLBACK)

    # The failure messages, taken from the failures rather than from the
    # source. Telling them apart is the whole value: topping up credit does
    # nothing about a per-key spending cap, and they are different pages.
    def raised(response_factory) -> str:
        import requests

        real_post = requests.post
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test"
        try:
            requests.post = lambda *a, **kw: response_factory()
            try:
                llm._openrouter_generate("p", {"type": "object"}, None,
                                         "anthropic/claude-opus-5", False)
            except Exception as exc:
                return str(exc)
            return ""
        finally:
            requests.post = real_post
            os.environ.pop("OPENROUTER_API_KEY", None)

    credit = raised(lambda: _FakeHTTP(status=402, text="insufficient credit"))
    check("running out of credit is named",
          "402" in credit or "credit" in credit.lower())
    cap = raised(lambda: _FakeHTTP(status=403, text="key spending limit exceeded"))
    check("a per-key spending cap is named separately from spent credit",
          "limit" in cap.lower())
    check("and the two messages differ", credit != cap and cap != "")
    # A 200 can still carry an error body, which is why the status code is not
    # the only thing checked.
    body_error = raised(lambda: _FakeHTTP(body={"error": {"message": "upstream exploded"}}))
    check("an error body behind a 200 is still an error", body_error != "")
    truncated = raised(lambda: _FakeHTTP(body={"choices": [
        {"finish_reason": "length", "message": {"content": '{"title": "unfinis'}}]}))
    check("a truncated reply names the ceiling, not the JSON",
          "output limit" in truncated.lower() or "max_tokens" in truncated.lower())


def test_every_provider_reports_what_it_sent() -> None:
    """Reported input has to be comparable against what was actually sent.

    This pairing is what answered #116. The revision call's input looked
    unexplainable for as long as the log printed what was charged and never
    what was sent, so the comparison meant re-deriving the prompt by hand — and
    the figure it got re-derived against left out the SOURCES block entirely.
    Once both were on the same line, reported input turned out to be a fixed
    floor plus the prompt counted once (`docs/decisions.md`).

    Keep the pair logged. The measurement is only repeatable while it is there.
    """
    import inspect
    import re as _re

    from articlegen import llm

    check("a prompt's size is reported in both units",
          llm._prompt_size("x" * 4000) == "chars=4000 ~tok=1000")

    for fn in (llm._gemini_cli_generate, llm._claude_cli_generate,
               llm._openrouter_generate):
        src = inspect.getsource(fn)
        check(f"{fn.__name__} logs what it sent beside what it was charged",
              "sent[" in src and "_prompt_size(" in src)
        # One shape for all three, so a single grep collects a whole run.
        check(f"{fn.__name__} uses the shared log prefix",
              _re.search(r"\[articlegen\] (gemini-cli|claude-cli|openrouter)", src)
              is not None)

    # The metered provider is the baseline: its accounting is independently
    # verifiable, which is exactly what the CLI providers' is not.
    check("the metered path reports its input count too",
          "prompt_tokens" in inspect.getsource(llm._openrouter_generate))
    # The gemini path also reports the other file in its scratch directory. It
    # was logged to test whether prompt and schema were being counted several
    # times over; they are not (#116), but the size stays on the line because
    # the schema is real input and belongs beside the rest of it.
    check("the gemini path reports the schema it also exposes",
          "json.dumps(schema)" in inspect.getsource(llm._gemini_cli_generate))


def test_output_ceilings_follow_the_default_model() -> None:
    """The ceiling must be right for the model actually in use.

    #77 happened because #74 repointed the default without resizing the
    ceiling, and the test pinned named models rather than the default — so it
    stayed green while live requests truncated (issue #98).
    """
    from articlegen import llm

    default = llm.OPENROUTER_DEFAULT_MODEL
    fallback = llm.OPENROUTER_REFUSAL_FALLBACK
    for name, model in (("default", default), ("refusal fallback", fallback)):
        deep = llm._openrouter_max_tokens(model, True)
        shallow = llm._openrouter_max_tokens(model, False)
        # Whatever the default becomes, it is a reasoning model unless someone
        # deliberately changes that, and reasoning is spent from this budget.
        check(f"the {name} model gets the reasoning-sized ceiling",
              deep == llm.OPENROUTER_REASONING_DEEP_OUTPUT
              and shallow == llm.OPENROUTER_REASONING_OUTPUT)
        check(f"and the deep call gets more room than the shallow one ({name})",
              deep > shallow)

    # The published per-vendor maxima. Raising a ceiling past one trades a
    # truncation bug for a rejected request.
    ANTHROPIC_CEILING, LLAMA_CEILING = 128_000, 16_384
    llama = "meta-llama/llama-3.3-70b-instruct"
    for deep in (False, True):
        check(f"the default stays inside its vendor limit (deep={deep})",
              llm._openrouter_max_tokens(default, deep) <= ANTHROPIC_CEILING)
        check(f"Llama stays inside its own limit (deep={deep})",
              llm._openrouter_max_tokens(llama, deep) <= LLAMA_CEILING)
        check(f"a thinking model gets more room than one that doesn't (deep={deep})",
              llm._openrouter_max_tokens(default, deep)
              > llm._openrouter_max_tokens(llama, deep))
    check("the shallow ceiling matches what the direct Anthropic path reserves",
          llm._openrouter_max_tokens(default, False) == 16000)
    check("an unrecognised vendor keeps the conservative pair",
          llm._openrouter_max_tokens("mistralai/mistral-large", False)
          == llm._openrouter_max_tokens(llama, False))


def test_openrouter_routing() -> None:
    """OpenRouter is the default provider — it must not steal the other
    providers' traffic, and must not hand their own model ids back to them
    prefixed.

    The slash is the whole discriminator: OpenRouter re-sells Claude as
    `anthropic/claude-sonnet-5`, which routed to Anthropic's SDK would 404.
    """
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER"):
        os.environ.pop(var, None)
    from articlegen.llm import OPENROUTER_DEFAULT_MODEL, resolve_provider

    check("vendor/model slug -> openrouter",
          resolve_provider("meta-llama/llama-3.3-70b-instruct")[0] == "openrouter")
    check("a resold claude slug stays on openrouter",
          resolve_provider("anthropic/claude-sonnet-5")[0] == "openrouter")
    check("and keeps the slug as given",
          resolve_provider("anthropic/claude-sonnet-5")[1] == "anthropic/claude-sonnet-5")
    check("bare claude name still goes direct to anthropic",
          resolve_provider("claude-opus-5")[0] == "anthropic")

    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-x"
    check("openrouter key alone selects it", resolve_provider()[0] == "openrouter")
    check("and supplies its default model", resolve_provider()[1] == OPENROUTER_DEFAULT_MODEL)
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-x"
    check("openrouter wins when both keys are set", resolve_provider()[0] == "openrouter")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["ARTICLEGEN_PROVIDER"] = "openrouter"
    check("provider override accepts openrouter", resolve_provider()[0] == "openrouter")

    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER"):
        os.environ.pop(var, None)


def test_openrouter_refusal_falls_back() -> None:
    """A safety classifier decline must retry elsewhere, not surface as bad JSON.

    OpenRouter cannot pass Anthropic's server-side `fallbacks` parameter, so the
    protection the direct path gets from #45 has to be re-implemented here. The
    trigger is real: Fable declined a circadian-biology topic mid-reply, and
    because the refusal went unrecognised the caller saw a JSON parse error
    pointing at the fragment it had already written (issue #79).
    """
    import json as _json
    from articlegen import llm

    calls: list[str] = []
    refuse_everything = False

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def _fake_post(url, headers=None, json=None, timeout=None):
        model = json["model"]
        calls.append(model)
        if refuse_everything or model != llm.OPENROUTER_REFUSAL_FALLBACK:
            # Mid-reply decline: partial content, normalised to content_filter.
            return _Resp({"choices": [{
                "finish_reason": "content_filter",
                "native_finish_reason": "refusal",
                "message": {"content": '{"ideas": [{"title": "Circadian'},
            }]})
        return _Resp({"choices": [{
            "finish_reason": "stop",
            "message": {"content": '{"ideas": []}'},
        }]})

    import requests
    original = requests.post
    requests.post = _fake_post
    try:
        out = llm._openrouter_generate(
            "p", {"type": "object"}, None, llm.OPENROUTER_DEFAULT_MODEL, False, "sk-or-v1-x")
        check("a refusal is retried on the fallback model", calls == [
            llm.OPENROUTER_DEFAULT_MODEL, llm.OPENROUTER_REFUSAL_FALLBACK])
        check("the fallback is not itself an elevated-classifier model",
              llm.OPENROUTER_REFUSAL_FALLBACK != llm.OPENROUTER_DEFAULT_MODEL)
        check("and the fallback's answer is what comes back", out == {"ideas": []})

        # When nothing will serve it, the partial fragment must still never reach
        # the caller: it parses as broken JSON, which is the confusing failure
        # this whole check replaces.
        refuse_everything = True
        calls.clear()
        raised = ""
        try:
            llm._openrouter_generate(
                "p", {"type": "object"}, None, llm.OPENROUTER_DEFAULT_MODEL, False,
                "sk-or-v1-x")
        except RuntimeError as exc:
            raised = str(exc)
        check("a refusal by the fallback too is a clear error, not a parse error",
              "declined this request on safety grounds" in raised
              and "invalid JSON" not in raised)
        check("and the retry happens once, never in a loop",
              calls == [llm.OPENROUTER_DEFAULT_MODEL, llm.OPENROUTER_REFUSAL_FALLBACK])
    finally:
        requests.post = original


def test_refusal_fallbacks() -> None:
    """Opus 5's safety classifiers can decline a clinical topic outright.

    The server-side fallback re-runs a declined request on Anthropic's
    recommended substitute model inside the same call, so a false positive
    doesn't cost the whole run (issue #45). Models without those classifiers
    reject the parameter, so it must only be attached where they fire.
    """
    import inspect
    from articlegen import llm

    kwargs = llm._refusal_fallback_kwargs("claude-opus-5")
    check("opus 5 opts into the server-side fallback", kwargs.get("fallbacks") == "default")
    check("and sends the matching beta header", kwargs.get("betas") == [llm._REFUSAL_FALLBACK_BETA])
    check("fable gets the fallback too", "fallbacks" in llm._refusal_fallback_kwargs("claude-fable-5"))
    check("opus 4.8 does not take the parameter", llm._refusal_fallback_kwargs("claude-opus-4-8") == {})
    check("haiku does not either", llm._refusal_fallback_kwargs("claude-haiku-4-5") == {})

    src = inspect.getsource(llm._anthropic_generate)
    check("anthropic calls go through the beta endpoint",
          "client.beta.messages" in src and "client.messages." not in src)
    check("both call paths attach the fallback kwargs", "_refusal_fallback_kwargs(" in src)


def test_pipeline_is_shared() -> None:
    """Both entry points must run the same pipeline.

    The web handler used to have its own copy that skipped the prose-style gate
    and never built provenance, so web-generated articles came out without the
    enforced hedging and with an incomplete Methods section.
    """
    import inspect
    from articlegen import cli, pipeline, web

    cmd_draft_src = inspect.getsource(cli.cmd_draft)
    handler_src = inspect.getsource(web.ArticleGenHandler._handle_draft)

    for name, src in (("cli.cmd_draft", cmd_draft_src), ("web._handle_draft", handler_src)):
        check(f"{name} calls generate_draft", "generate_draft(" in src)
        for stage in ("plan_queries(", "curate_sources(", "write_article(", "write_briefing("):
            check(f"{name} does not re-run {stage[:-1]}", stage not in src)

    src = inspect.getsource(pipeline.generate_draft)
    check("pipeline writes a briefing by default", "write_briefing" in src)
    check("pipeline keeps the Review writer for --long", "write_article" in src)
    check("pipeline enforces style", "enforce_style(" in src)
    check("pipeline builds provenance", '"queries": queries' in inspect.getsource(pipeline.generate_draft))


def test_idea_search_terms_reach_the_draft() -> None:
    """When idea search terms are supplied they start the plan and are never replaced (#172)."""
    import inspect
    from articlegen import cli, pipeline, web, writer
    from articlegen.sources import Paper

    # 1. clean_search_terms normalisation
    cleaned = writer.clean_search_terms(["  a ", "", "A", "b", "c", "d", None, 7])
    check("clean_search_terms strips, drops blanks/dups/non-strings and caps count",
          cleaned == ["a", "b", "c"])
    check("clean_search_terms on None returns empty list", writer.clean_search_terms(None) == [])
    check("clean_search_terms on non-list returns empty list", writer.clean_search_terms("not a list") == [])

    # 2-6. plan_queries with fake generate_json
    captured_prompts: list[str] = []
    saved_generate_json = writer.generate_json
    try:
        def fake_generate(prompt, schema, **kw):
            captured_prompts.append(prompt)
            return {"queries": ["totally different one", "and another"], "core_entity": "x"}

        writer.generate_json = fake_generate

        # 2. Terms supplied -> supplied first in order, at most one addition, core_entity preserved
        queries, core = writer.plan_queries("test topic", search_terms=["term one", "term two"])
        check("supplied terms are preserved first in order with at most one addition",
              queries == ["term one", "term two", "totally different one"])
        check("core_entity is returned from the planner", core == "x")

        # 3. Captured prompt contains supplied terms and instructions
        check("planner prompt contains supplied terms",
              "term one" in captured_prompts[0] and "term two" in captured_prompts[0])
        check("planner prompt instructs against rewriting and asks for at most one addition",
              "Do not rewrite them" in captured_prompts[0]
              and ("at most ONE additional" in captured_prompts[0] or "at most one" in captured_prompts[0].lower()))

        # 4. A duplicate of a supplied term adds nothing
        writer.generate_json = lambda prompt, schema, **kw: {"queries": ["TERM ONE", "other"], "core_entity": "y"}
        q_dup, _ = writer.plan_queries("test topic", search_terms=["term one", "term two"])
        check("duplicate of supplied query adds nothing further",
              q_dup == ["term one", "term two", "other"])

        writer.generate_json = lambda prompt, schema, **kw: {"queries": ["TERM ONE"], "core_entity": "y"}
        q_dup_only, _ = writer.plan_queries("test topic", search_terms=["term one", "term two"])
        check("case-insensitive duplicate only keeps length at 2", len(q_dup_only) == 2)

        # 5. Cap at MAX_PLANNED_QUERIES
        writer.generate_json = lambda prompt, schema, **kw: {
            "queries": ["q1", "q2", "q3", "q4"], "core_entity": "z"
        }
        q_max, _ = writer.plan_queries("test topic", search_terms=["s1", "s2", "s3"])
        check("result is capped at MAX_PLANNED_QUERIES",
              len(q_max) <= writer.MAX_PLANNED_QUERIES and q_max == ["s1", "s2", "s3", "q1"])

        # 6. No terms supplied -> today's behaviour
        captured_prompts.clear()
        writer.generate_json = lambda prompt, schema, **kw: (
            captured_prompts.append(prompt) or {
                "queries": ["q1", "q2", "q3", "q4", "q5", "q6"],
                "core_entity": "core_ent",
            }
        )
        q_none, core_none = writer.plan_queries("test topic", search_terms=None)
        check("no terms supplied returns first MAX_PLANNED_QUERIES",
              q_none == ["q1", "q2", "q3", "q4"] and core_none == "core_ent")
        check("no terms prompt does not contain supplied-terms preamble",
              "were already chosen" not in captured_prompts[0])

    finally:
        writer.generate_json = saved_generate_json

    # 7. pipeline.generate_draft passes search_terms through
    captured_plan_kw: dict = {}
    papers = [Paper(title=f"p{i}", abstract="a", pmcid=f"PMC{i}", is_open_access=True)
              for i in range(1, 5)]
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [], "references": [1]}
    curation = {"relevance": {1: "direct", 2: "tangential", 3: "related", 4: "direct"},
                "most_relevant_index": 1,
                "counts": {"direct": 2, "related": 1, "tangential": 1}}

    saved_pipeline = (
        pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
        pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
        pipeline.enforce_style,
    )
    try:
        pipeline.plan_queries = lambda topic, **kw: (captured_plan_kw.update(kw) or (["q"], "core"))
        def fake_gather(queries, **kw):
            kw.get("outcomes", []).append(
                {"source": "europe_pmc", "query": "q", "count": 4, "error": "", "cached": False})
            return papers
        pipeline.gather_evidence = fake_gather
        pipeline.curate_sources = lambda topic, p, **kw: curation
        pipeline.write_article = lambda topic, p, **kw: dict(article)
        pipeline.write_briefing = pipeline.write_article
        pipeline.fetch_full_text = lambda p, use_cache=True: "body text"
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})

        pipeline.generate_draft("topic", search_terms=["a", "b"])
        check("pipeline passes search_terms through to plan_queries",
              captured_plan_kw.get("search_terms") == ["a", "b"])
    finally:
        (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
            pipeline.enforce_style,
        ) = saved_pipeline

    # 8-9. CLI
    parser = cli.build_parser()
    check("cli parser accepts --queries",
          parser.parse_args(["draft", "t", "--queries", "a, b"]).queries == "a, b")
    check("cmd_draft passes search_terms",
          "search_terms=" in inspect.getsource(cli.cmd_draft))

    # 10. Web
    web_src = inspect.getsource(web.ArticleGenHandler._handle_draft)
    check("web handler reads search_terms from payload",
          'payload.get("search_terms")' in web_src)
    check("web handler forwards search_terms to generate_draft",
          "search_terms=search_terms" in web_src)

    # 11-12. Front end index.html
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    check("index.html sends search_terms in /api/draft body",
          "search_terms: terms" in index_html or "search_terms:" in index_html)
    check("index.html passes terms to selectDraft in renderDraftCards",
          "selectDraft(idea.title, terms, pico)" in index_html)


def test_idea_pico_reaches_the_curator() -> None:
    """A card's population/intervention/comparator/outcome reach the planner and the curator (#247)."""
    import inspect
    from articlegen import ideas, pipeline, web, writer
    from articlegen.sources import Paper

    # 1-5. writer.clean_pico / format_pico
    check("clean_pico strips, drops blanks/non-strings/unknown keys",
          writer.clean_pico({"population": "  adults ", "outcome": "", "intervention": None,
                              "comparator": 7, "junk": "x"}) == {"population": "adults"})
    check("clean_pico(None) is empty", writer.clean_pico(None) == {})
    check("clean_pico on non-dict is empty", writer.clean_pico("not a dict") == {})
    long_value = "x" * (writer.MAX_PICO_CHARS + 50)
    capped = writer.clean_pico({"population": long_value})
    check("clean_pico caps a value at MAX_PICO_CHARS",
          len(capped["population"]) == writer.MAX_PICO_CHARS)
    check("format_pico({}) is empty string", writer.format_pico({}) == "")
    only_pop = writer.format_pico({"population": "adults"})
    check("format_pico contains the named field", "Population: adults" in only_pop)
    check("format_pico omits an unnamed field", "Outcome" not in only_pop)
    out_of_order = writer.clean_pico({"outcome": "o", "population": "p"})
    check("clean_pico key order follows PICO_FIELDS",
          list(out_of_order.keys()) == ["population", "outcome"])

    # 6-9. curate_sources prompt
    saved_generate_json = writer.generate_json
    captured_prompts: list[str] = []
    try:
        def fake_generate(prompt, schema, **kw):
            captured_prompts.append(prompt)
            return {"assessments": [{"index": 1, "relevance": "direct"}], "most_relevant_index": 1}

        writer.generate_json = fake_generate
        paper = Paper(title="p1", abstract="a", pmcid="PMC1", is_open_access=True)

        writer.curate_sources("topic", [paper], pico={"population": "PICO-POP", "outcome": "PICO-OUT"})
        check("curation prompt contains the supplied population",
              "PICO-POP" in captured_prompts[-1])
        check("curation prompt contains the supplied outcome",
              "PICO-OUT" in captured_prompts[-1])
        check("curation prompt labels the population", "Population:" in captured_prompts[-1])
        check("curation prompt labels the outcome", "Outcome:" in captured_prompts[-1])

        writer.curate_sources("topic", [paper])
        check("curation prompt with no pico has no Population label",
              "Population:" not in captured_prompts[-1])
        check("curation prompt with no pico has no pico opening sentence",
              "as named at the ideas stage" not in captured_prompts[-1])

        writer.curate_sources("topic", [paper], pico={"population": "PICO-POP", "outcome": "PICO-OUT"})
        check("curation prompt with population+outcome only has no Comparator label",
              "Comparator" not in captured_prompts[-1])

        check("_CURATE_SYSTEM says an unnamed part does not narrow the topic",
              "does not narrow" in writer._CURATE_SYSTEM)

        # 10-11. planner prompt
        captured_prompts.clear()
        writer.plan_queries("topic", search_terms=["a"], pico={"population": "PICO-POP"})
        check("planner prompt contains the supplied population",
              "PICO-POP" in captured_prompts[-1])

        captured_prompts.clear()
        writer.plan_queries("topic")
        check("planner prompt with no pico has no Population label",
              "Population:" not in captured_prompts[-1])
    finally:
        writer.generate_json = saved_generate_json

    # 12-13. pipeline.generate_draft forwards pico
    captured_plan_kw: dict = {}
    captured_curate_kw: dict = {}
    papers = [Paper(title=f"p{i}", abstract="a", pmcid=f"PMC{i}", is_open_access=True)
              for i in range(1, 5)]
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [], "references": [1]}
    curation = {"relevance": {1: "direct", 2: "tangential", 3: "related", 4: "direct"},
                "most_relevant_index": 1,
                "counts": {"direct": 2, "related": 1, "tangential": 1}}

    saved_pipeline = (
        pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
        pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
        pipeline.enforce_style,
    )
    try:
        pipeline.plan_queries = lambda topic, **kw: (captured_plan_kw.update(kw) or (["q"], "core"))
        def fake_gather(queries, **kw):
            kw.get("outcomes", []).append(
                {"source": "europe_pmc", "query": "q", "count": 4, "error": "", "cached": False})
            return papers
        pipeline.gather_evidence = fake_gather
        def fake_curate(topic, p, **kw):
            captured_curate_kw.update(kw)
            return curation
        pipeline.curate_sources = fake_curate
        pipeline.write_article = lambda topic, p, **kw: dict(article)
        pipeline.write_briefing = pipeline.write_article
        pipeline.fetch_full_text = lambda p, use_cache=True: "body text"
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})

        pipeline.generate_draft("topic", pico={"population": "adults", "junk": "x"})
        check("pipeline passes cleaned pico to curate_sources",
              captured_curate_kw.get("pico") == {"population": "adults"})
        check("pipeline passes the same cleaned pico to plan_queries",
              captured_plan_kw.get("pico") == {"population": "adults"})

        captured_curate_kw.clear()
        pipeline.generate_draft("topic")
        check("draft with no card keeps working: curator called with pico absent or empty",
              not captured_curate_kw.get("pico"))
    finally:
        (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
            pipeline.enforce_style,
        ) = saved_pipeline

    # 14. web
    web_src = inspect.getsource(web.ArticleGenHandler._handle_draft)
    check("web handler reads pico from payload", 'payload.get("pico")' in web_src)
    check("web handler forwards pico to generate_draft", "pico=pico" in web_src)

    # 15-17. front end
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    check("index.html sends pico in /api/draft body", "pico: pico" in index_html)
    check("index.html passes pico from renderDraftCards to selectDraft",
          "selectDraft(idea.title, terms, pico)" in index_html)
    check("index.html renders the PICO labels on the card",
          "Population" in index_html and "draft-pico" in index_html)

    # 18-20. ideas
    item_props = ideas._IDEAS_SCHEMA["properties"]["ideas"]["items"]["properties"]
    item_required = ideas._IDEAS_SCHEMA["properties"]["ideas"]["items"]["required"]
    for field in ("population", "intervention", "comparator", "outcome"):
        check(f"_IDEAS_SCHEMA declares {field}", field in item_props)
        check(f"_IDEAS_SCHEMA requires {field} (strict-schema trap)", field in item_required)

    full_card = {
        "title": "t", "angle": "a", "search_terms": ["x"],
        "population": "adults", "intervention": "drug", "comparator": "",
        "outcome": "seclusion episodes",
    }
    md = ideas.ideas_to_markdown("theme", [full_card])
    check("ideas_to_markdown renders a named PICO field",
          "*Population:* adults" in md and "seclusion episodes" in md)
    check("ideas_to_markdown omits an empty PICO field",
          "*Comparator:*" not in md)

    bare_card = {"title": "t", "angle": "a", "search_terms": ["x"]}
    md_bare = ideas.ideas_to_markdown("theme", [bare_card])
    check("ideas_to_markdown with no PICO keys does not raise and adds no PICO text",
          "Population" not in md_bare)


def test_paraphrase_terms_buy_a_distinct_query() -> None:
    """When supplied or planned terms are near-duplicates, one distinct extra query is required (#190)."""
    from articlegen import writer
    from articlegen.writer import (
        NEAR_DUPLICATE_OVERLAP, queries_are_near_duplicates, query_routes,
    )

    # Unit checks for queries_are_near_duplicates
    check("paraphrase pair is near-duplicate",
          queries_are_near_duplicates("seclusion reduction acute mental health",
                                     "reducing seclusion in acute mental health wards"))
    check("different intervention/setting pair is not near-duplicate",
          not queries_are_near_duplicates("seclusion reduction acute mental health",
                                         "post-incident debriefing psychiatric wards"))
    check("pair where one side has empty normalisation is not near-duplicate",
          not queries_are_near_duplicates("term one", ""))
    check("pair where both sides have empty normalisation is not near-duplicate",
          not queries_are_near_duplicates("", ""))

    # Unit checks for query_routes
    check("paraphrase terms have query_routes == 1",
          query_routes(["seclusion reduction acute mental health",
                        "reducing seclusion in acute mental health wards"]) == 1)
    check("distinct terms have query_routes == 2",
          query_routes(["seclusion reduction", "post-incident debriefing"]) == 2)
    check("empty terms list has query_routes == 0", query_routes([]) == 0)
    check("single term has query_routes == 1", query_routes(["single term"]) == 1)

    captured_prompts: list[str] = []
    call_count = 0
    saved_generate_json = writer.generate_json
    try:
        # Supplied near-duplicate terms -> prompt contains "exactly ONE" demand, "Do not rewrite them", and appends distinct reply
        def fake_distinct(prompt, schema, **kw):
            captured_prompts.append(prompt)
            return {"queries": ["debriefing and sensory rooms"], "core_entity": "seclusion"}

        writer.generate_json = fake_distinct
        q, core = writer.plan_queries(
            "seclusion reduction",
            search_terms=["seclusion reduction acute wards", "reducing seclusion in acute wards"]
        )
        check("supplied near-duplicate terms prompt contains 'exactly ONE' demand",
              "exactly ONE" in captured_prompts[0])
        check("supplied near-duplicate terms prompt contains 'Do not rewrite them'",
              "Do not rewrite them" in captured_prompts[0])
        check("distinct reply is appended after supplied terms in order",
              q == ["seclusion reduction acute wards", "reducing seclusion in acute wards", "debriefing and sensory rooms"])
        check("core_entity is preserved", core == "seclusion")

        # Supplied near-duplicate terms + paraphrase reply -> rejected, exactly 2 generate_json calls (1 retry), retry failure leaves supplied unchanged
        captured_prompts.clear()
        call_count = 0

        def fake_paraphrase_retry(prompt, schema, **kw):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            if call_count == 1:
                return {"queries": ["acute ward seclusion reduction"], "core_entity": "seclusion"}
            # Retry also paraphrases
            return {"queries": ["reducing seclusion in psychiatric wards"], "core_entity": "seclusion"}

        writer.generate_json = fake_paraphrase_retry
        q_retry, _ = writer.plan_queries(
            "seclusion reduction",
            search_terms=["seclusion reduction acute wards", "reducing seclusion in acute wards"]
        )
        check("paraphrase reply causes exactly one retry (2 calls total)", call_count == 2)
        check("retry prompt quotes rejected paraphrase or notes re-wording",
              "acute ward seclusion reduction" in captured_prompts[1] or "re-wording" in captured_prompts[1])
        check("double paraphrase failure returns supplied terms unchanged",
              q_retry == ["seclusion reduction acute wards", "reducing seclusion in acute wards"])

        # Supplied terms that already take distinct routes -> 1 call only, prompt does not carry "exactly ONE", behaviour matches today
        captured_prompts.clear()
        call_count = 0

        def fake_distinct_supplied(prompt, schema, **kw):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            return {"queries": ["sensory modulation"], "core_entity": "seclusion"}

        writer.generate_json = fake_distinct_supplied
        q_dist, _ = writer.plan_queries(
            "seclusion and sensory rooms",
            search_terms=["seclusion reduction", "sensory rooms psychiatric"]
        )
        check("distinct supplied terms makes exactly 1 call", call_count == 1)
        check("distinct supplied prompt does not contain 'These terms all search the same route'",
              "These terms all search the same route" not in captured_prompts[0])
        check("distinct supplied returns supplied plus extra",
              q_dist == ["seclusion reduction", "sensory rooms psychiatric", "sensory modulation"])

        # No-supplied path, planner returns 3 paraphrases -> exactly 1 follow-up call, distinct query appended, core_entity preserved
        captured_prompts.clear()
        call_count = 0

        def fake_nosupplied_paraphrases(prompt, schema, **kw):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            if call_count == 1:
                return {
                    "queries": [
                        "seclusion reduction acute mental health",
                        "reducing seclusion in acute mental health wards",
                        "acute mental health seclusion reduction",
                    ],
                    "core_entity": "initial_core",
                }
            # Follow-up returns distinct query
            return {"queries": ["Safewards intervention restraint"], "core_entity": "ignored_core"}

        writer.generate_json = fake_nosupplied_paraphrases
        q_nosup, core_nosup = writer.plan_queries("seclusion reduction", search_terms=None)
        check("no-supplied paraphrase plan triggers exactly 1 follow-up call (2 calls total)", call_count == 2)
        check("follow-up appends distinct query",
              "Safewards intervention restraint" in q_nosup and len(q_nosup) == 4)
        check("core_entity comes from first call", core_nosup == "initial_core")

        # No-supplied path, planner returns 4 paraphrases (at cap) -> last near-duplicate dropped to make room, result <= MAX_PLANNED_QUERIES
        captured_prompts.clear()
        call_count = 0

        def fake_nosupplied_four_paraphrases(prompt, schema, **kw):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            if call_count == 1:
                return {
                    "queries": [
                        "seclusion reduction acute mental health",
                        "reducing seclusion in acute mental health wards",
                        "acute mental health seclusion reduction",
                        "reducing seclusion in acute mental health units",
                    ],
                    "core_entity": "core_four",
                }
            return {"queries": ["debriefing interventions"], "core_entity": "other"}

        writer.generate_json = fake_nosupplied_four_paraphrases
        q_four, core_four = writer.plan_queries("seclusion reduction", search_terms=None)
        check("four paraphrases plan stays <= MAX_PLANNED_QUERIES after follow-up",
              len(q_four) <= writer.MAX_PLANNED_QUERIES and len(q_four) == 4)
        check("distinct query is added after dropping last duplicate",
              q_four[-1] == "debriefing interventions")
        check("core_entity is core_four", core_four == "core_four")

        # No-supplied path, planner returns distinct queries -> no follow-up call
        captured_prompts.clear()
        call_count = 0

        def fake_nosupplied_distinct(prompt, schema, **kw):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            return {
                "queries": ["seclusion reduction", "post-incident debriefing", "sensory modulation"],
                "core_entity": "core_distinct",
            }

        writer.generate_json = fake_nosupplied_distinct
        q_nodist, _ = writer.plan_queries("seclusion and alternatives", search_terms=None)
        check("no-supplied distinct plan makes no follow-up call (1 call total)", call_count == 1)
        check("queries returned as planned", len(q_nodist) == 3)

    finally:
        writer.generate_json = saved_generate_json


def test_dead_sources_fail_before_the_caller_is_billed() -> None:
    """A doomed run must not spend an LLM call first.

    `plan_queries` is paid and runs before anything touches a scholarly API, so
    on a day when every API refuses, the caller paid for a run that could never
    have worked (issue #96). The pre-flight probe can only refuse on the same
    condition `generate_draft` already raises on afterwards — *every* source
    errored, not merely no results — so it cannot block a draft that would have
    succeeded.
    """
    import inspect
    import time

    from articlegen import pipeline

    src = inspect.getsource(pipeline.generate_draft)
    check("the probe runs before the first paid call",
          src.index("_preflight_sources(") < src.index("plan_queries("))

    def reset(ok=0.0, fail=(0.0, "")):
        pipeline._sources_last_ok = ok
        pipeline._sources_last_fail = fail

    real_gather = pipeline.gather_evidence
    calls = []

    def all_sources_down(queries, **kwargs):
        calls.append(queries)
        outcomes = kwargs.get("outcomes")
        if outcomes is not None:
            for name in ("semantic_scholar", "openalex", "europe_pmc"):
                outcomes.append({"source": name, "query": queries[0], "count": 0,
                                 "error": "429 Too Many Requests", "cached": False})
        return []

    try:
        pipeline.gather_evidence = all_sources_down
        reset()
        raised = None
        try:
            pipeline._preflight_sources("light therapy", lambda m: None)
        except pipeline.NoPapersFound as exc:
            raised = exc
        check("every source failing refuses the run", raised is not None)
        check("and names it as the sources' problem, not the topic's",
              raised is not None and raised.sources_failed
              and "not a problem with the topic" in str(raised))
        check("and says nothing was charged", "nothing has been charged" in str(raised))

        # A second request must not re-probe dead sources — it reuses the verdict.
        before = len(calls)
        try:
            pipeline._preflight_sources("light therapy", lambda m: None)
        except pipeline.NoPapersFound:
            pass
        check("a recent failure is reused rather than re-probed", len(calls) == before)

        # And a server that heard from a source recently skips the probe, so
        # this is not a tax on every request of a healthy deployment.
        reset(ok=time.time())
        before = len(calls)
        pipeline._preflight_sources("light therapy", lambda m: None)
        check("a recently healthy server does not probe at all", len(calls) == before)

        # Sources answering with no results is a topic problem, not an outage:
        # the probe must let that through to the real gather.
        def answered_but_empty(queries, **kwargs):
            calls.append(queries)
            outcomes = kwargs.get("outcomes")
            if outcomes is not None:
                outcomes.append({"source": "europe_pmc", "query": queries[0], "count": 0,
                                 "error": "", "cached": False})
            return []

        pipeline.gather_evidence = answered_but_empty
        reset()
        pipeline._preflight_sources("a topic with no literature", lambda m: None)
        check("a source that answered with nothing does not refuse the run", True)
    finally:
        pipeline.gather_evidence = real_gather
        reset()


def test_draft_summary() -> None:
    from articlegen.pipeline import Draft
    from articlegen.sources import Paper

    papers = [Paper(title=f"P{i}", abstract="a", year=2020) for i in range(1, 6)]
    clean = Draft(
        topic="t",
        article={"references": [1, 2, 3]},
        papers=papers,
        curation={"relevance": {1: "direct", 2: "direct", 3: "related"}},
        verification={"unverified": []},
        style_report={"issues": [], "stats": {}},
    )
    check("counts cited sources", clean.summary().startswith("3 of 5 screened sources cited"))
    check("counts direct sources", "2 directly on-topic" in clean.summary())
    check("clean prose reported", "prose style clean" in clean.summary())

    messy = Draft(
        topic="t",
        article={"references": [1, 2]},
        papers=papers,
        curation={"relevance": {1: "related", 2: "tangential"}},
        verification={"unverified": ["42%"]},
        style_report={
            "issues": [{"severity": "error", "rule": "boosters", "detail": "clearly"}],
            "stats": {},
        },
    )
    summary = messy.summary()
    check("unverified figures flagged", "1 figure(s) not found" in summary)
    check("no direct source flagged", "no directly on-topic source found" in summary)
    check("style issues flagged", "prose-style issue(s)" in summary)

    out_of_range = Draft(topic="t", article={"references": [1, 99, "x"]}, papers=papers)
    check("ignores out-of-range references", out_of_range.summary().startswith("1 of 5 screened sources cited"))


def test_run_manifest_round_trips() -> None:
    """A draft survives the trip to JSON and back, and renders the same page.

    `articlegen draft` used to write HTML and Markdown and nothing else, so
    the paper pool, labels, excerpts and verification were gone when the
    process ended. The manifest keeps them; this pins that nothing is lost on
    the way through JSON (int keys, tuples, full text) and that `render`
    rebuilds the exact page the run wrote.
    """
    import copy
    import inspect
    import tempfile
    from articlegen import cli, demo, pipeline, render, style, verify, web
    from articlegen.pipeline import Draft
    from articlegen.sources import full_text_excerpts

    papers = copy.deepcopy(demo.SAMPLE_PAPERS)
    papers[0].full_text = "Replay was seen in 61% of trials. " * 600
    papers[0].full_text_via = "papers"
    papers[1].publication_types = ("journal article", "review")
    article = copy.deepcopy(demo.SAMPLE_BRIEFING)
    article["findings"] = list(article["findings"]) + ["Replay was seen in 61% of trials, and 23% elsewhere [1]."]
    original = Draft(
        topic="the functions of sleep in the brain",
        article=article,
        papers=papers,
        curation=copy.deepcopy(demo.SAMPLE_CURATION),
        verification=verify.check_statistics(article, papers),
        provenance=dict(demo.SAMPLE_PROVENANCE, date="1 September 2026",
                        full_text_sources=[1], full_text_via={"papers": 1, "europe_pmc": 0}),
        style_report=style.check_style(article, direct_sources=3),
        excerpts=full_text_excerpts(papers),
    )
    check("the fixture has an excerpt to carry", original.excerpts.get(1, "").startswith("Replay"))
    check("the fixture has a flagged figure to carry", "23%" in original.verification["unverified"])

    manifest = original.to_dict()
    check("manifest carries its version", manifest.get("manifest_version") == pipeline.MANIFEST_VERSION)
    check("manifest has the expected top-level keys",
          set(manifest) == {"manifest_version", "topic", "article", "papers", "curation",
                            "verification", "provenance", "style_report", "excerpts"})
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    restored = Draft.from_dict(json.loads(text))
    check("from_dict(to_dict(d)) == d", restored == original)
    check("relevance keys come back as ints",
          all(isinstance(k, int) for k in restored.curation["relevance"]))
    check("excerpt keys come back as ints", list(restored.excerpts) == [1])
    check("publication types come back as a tuple",
          restored.papers[1].publication_types == ("journal article", "review"))
    check("full text and its route survive",
          restored.papers[0].full_text == papers[0].full_text
          and restored.papers[0].full_text_via == "papers")

    def render_args(d):
        return (d.article, d.papers, d.topic, d.curation, d.verification, d.provenance, d.style_report)

    html_a, html_b = render.render_article(*render_args(original)), render.render_article(*render_args(restored))
    md_a, md_b = render.render_markdown(*render_args(original)), render.render_markdown(*render_args(restored))
    check("restored draft renders identical HTML", html_a == html_b)
    check("restored draft renders identical Markdown", md_a == md_b)
    check("the page is stamped with the run's date, not today's", "Generated 1 September 2026" in html_a)
    check("re-verifying against the stored excerpts reproduces the result",
          verify.check_statistics(restored.article, restored.papers) == original.verification)

    # The CLI writes the manifest beside the HTML, and `render` rebuilds the
    # same bytes from it with no pipeline call.
    check("render never runs the pipeline", "generate_draft(" not in inspect.getsource(cli.cmd_render))
    check("the web server never writes a manifest", "to_dict(" not in inspect.getsource(web))
    saved_generate, saved_cwd = cli.generate_draft, os.getcwd()
    try:
        cli.generate_draft = lambda *a, **kw: original
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            check("draft exits 0", cli.main(["draft", original.topic, "--name", "x"]) == 0)
            html_path, json_path = os.path.join("drafts", "x.html"), os.path.join("drafts", "x.json")
            check("draft still writes HTML and Markdown",
                  os.path.isfile(html_path) and os.path.isfile(os.path.join("drafts", "x.md")))
            check("draft writes the manifest beside them", os.path.isfile(json_path))
            with open(json_path, encoding="utf-8") as f:
                on_disk = json.load(f)
            check("the manifest on disk is version 1", on_disk.get("manifest_version") == 1)
            run_html = open(html_path, encoding="utf-8").read()
            run_md = open(os.path.join("drafts", "x.md"), encoding="utf-8").read()
            os.remove(html_path)
            os.remove(os.path.join("drafts", "x.md"))
            check("render exits 0", cli.main(["render", json_path]) == 0)
            check("render rebuilds the same HTML",
                  open(html_path, encoding="utf-8").read() == run_html)
            check("render rebuilds the same Markdown",
                  open(os.path.join("drafts", "x.md"), encoding="utf-8").read() == run_md)

            on_disk["manifest_version"] = 99
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(on_disk, f)
            check("an unsupported manifest version is refused", cli.main(["render", json_path]) == 1)
            os.chdir(saved_cwd)
    finally:
        cli.generate_draft = saved_generate
        os.chdir(saved_cwd)

    # The pipeline captures the excerpt map itself, once the papers are final.
    src = inspect.getsource(pipeline.generate_draft)
    check("the pipeline records the excerpts it showed", "excerpts=excerpts" in src
          and src.index("full_text_excerpts(papers)") < src.index("enforce_style("))


def test_rate_limit() -> None:
    from articlegen import web

    original_max = web.RATE_LIMIT_MAX
    original_total = web.RATE_LIMIT_TOTAL
    original_trust = web.TRUST_PROXY
    web.RATE_LIMIT_MAX = 3
    web.RATE_LIMIT_TOTAL = 10_000
    web._rate_hits.clear()
    web._rate_hits_all.clear()
    try:
        allowed = [web._rate_limited("10.0.0.1") is None for _ in range(4)]
        check("first N requests allowed", allowed[:3] == [True, True, True])
        check("request over the limit is blocked", allowed[3] is False)
        check("a different address is unaffected", web._rate_limited("10.0.0.2") is None)

        # The per-IP limit protects nothing the scholarly APIs care about: they
        # meter against this server's single egress IP, so upstream load scales
        # with visitor count while every individual stays politely under 20
        # (#96). The aggregate ceiling is the one that matches the real quota.
        web._rate_hits.clear()
        web._rate_hits_all.clear()
        web.RATE_LIMIT_TOTAL = 4
        spread = [web._rate_limited(f"10.0.1.{i}") for i in range(6)]
        check("the aggregate ceiling stops a crowd of polite visitors",
              spread[:4] == [None, None, None, None] and spread[4] is not None)
        check("and says it is not the visitor's own limit",
              "not your limit" in spread[4])

        # Behind Render's load balancer client_address[0] is the proxy, so
        # every visitor shared one bucket: one abuser locked out everybody.
        web._rate_hits.clear()
        web._rate_hits_all.clear()
        web.RATE_LIMIT_TOTAL = 10_000
        web.TRUST_PROXY = True
        for _ in range(3):
            web._rate_limited(web._client_ip("10.0.0.9", "203.0.113.5"))
        check("a proxied caller fills their own bucket",
              web._rate_limited(web._client_ip("10.0.0.9", "203.0.113.5")) is not None)
        check("and does not lock out the next visitor behind the same proxy",
              web._rate_limited(web._client_ip("10.0.0.9", "203.0.113.6")) is None)
    finally:
        web.RATE_LIMIT_MAX = original_max
        web.RATE_LIMIT_TOTAL = original_total
        web.TRUST_PROXY = original_trust
        web._rate_hits.clear()
        web._rate_hits_all.clear()

    # X-Forwarded-For is caller-controlled unless a proxy is known to rewrite
    # it, so trusting it by default would let anyone pick their own bucket. And
    # the *rightmost* entry is the one to use: a caller can send their own
    # header and the proxy appends the real peer to whatever arrived, so the
    # leftmost entry is attacker-chosen.
    try:
        web.TRUST_PROXY = False
        check("an untrusted deployment ignores the header",
              web._client_ip("10.0.0.1", "203.0.113.5") == "10.0.0.1")
        web.TRUST_PROXY = True
        check("a trusted deployment takes the rightmost entry",
              web._client_ip("10.0.0.1", "1.2.3.4, 203.0.113.5") == "203.0.113.5")
        check("and falls back to the peer with no header",
              web._client_ip("10.0.0.1", None) == "10.0.0.1")
    finally:
        web.TRUST_PROXY = original_trust

    # /api/diag deliberately bypasses the search cache so it reports what the
    # sources are doing right now, which makes every call spend real quota. It
    # therefore has to be metered like drafting is: unmetered, it was a way for
    # anyone with the URL to exhaust the quota the limiter exists to protect.
    import inspect

    source = inspect.getsource(web.ArticleGenHandler._handle_diag)
    check("/api/diag is charged against the rate limit",
          "_over_rate_limit" in source)
    check("/api/diag probes live rather than serving a cached answer",
          "use_cache=False" in source)


def test_keepalive_connection_reuse() -> None:
    """Several requests must survive on ONE connection.

    The server ran on http.server's default HTTP/1.0 for a while, which closes
    the connection after every response. Browsers and reverse proxies pool
    connections, so they kept reusing sockets the server had already hung up on
    and roughly every other request to the deployed backend failed in ~140ms.
    curl never caught it — each invocation opens a fresh connection, so only a
    pooling client reproduces it.
    """
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    from articlegen.web import ArticleGenHandler

    check("handler speaks HTTP/1.1", ArticleGenHandler.protocol_version == "HTTP/1.1")

    server = ThreadingHTTPServer(("127.0.0.1", 0), ArticleGenHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        statuses, closes = [], []
        for _ in range(3):
            conn.request("GET", "/api/health")
            resp = conn.getresponse()
            resp.read()
            statuses.append(resp.status)
            # will_close is the assertion that matters. Python's http.client
            # transparently reconnects when the server hangs up, so simply
            # issuing three requests passes under HTTP/1.0 too and proves
            # nothing — a browser's connection pool is what breaks. This asks
            # the server directly whether it intends to keep the socket open.
            closes.append(resp.will_close)
        conn.close()
        check("three requests succeed", statuses == [200, 200, 200])
        check("server keeps the connection open", closes == [False, False, False])

        # OPTIONS must not strand a pooling client waiting for a body.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("OPTIONS", "/api/draft")
        resp = conn.getresponse()
        resp.read()
        opt_status, opt_len = resp.status, resp.getheader("Content-Length")
        conn.request("GET", "/api/health")
        follow = conn.getresponse()
        follow.read()
        check("OPTIONS returns 204", opt_status == 204)
        check("OPTIONS declares Content-Length: 0", opt_len == "0")
        check("connection still usable after OPTIONS", follow.status == 200)
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_substance_checks() -> None:
    """Thin, repetitive prose must fail even when the register is perfect.

    Every other rule in style.py is a prohibition. A model optimising only
    against prohibitions writes vague hedged filler, because asserting nothing
    breaks no rule — a real draft passed clean at 803 words with one number in
    it, hedging at 0.69/sentence (three times the floor) using four stock
    phrases. These rules fail a draft for saying too little.
    """
    from articlegen.style import check_style, errors, revision_brief, SUBSTANCE_RULES

    def rules(article):
        return {i["rule"] for i in errors(check_style(article))}

    filler = (
        "The evidence suggests that these effects may be significant, although the "
        "magnitude may vary. It appears that the risk may be higher among some workers. "
        "The evidence suggests that these strategies may be effective in some settings. "
        "It appears that this approach may be beneficial, although the evidence may be limited. "
    )
    thin = {"sections": [{"heading": h, "paragraphs": [filler * 2]}
                         for h in ("Introduction", "Effects", "Conclusions")]}
    found = rules(thin)
    check("too-few-sections flagged", "too-few-sections" in found)
    check("hedge-monotony flagged", "hedge-monotony" in found)
    check("under-length is only a warning, not an error", "under-length" not in found)

    # Verbatim recycling across sections, which the register rules never noticed.
    # Padded past MIN_SENTENCES_FOR_VARIETY, below which repetition counts are noise.
    line = ("a range of strategies can be used to mitigate these effects including "
            "sleep hygiene education and flexible scheduling for affected staff")
    padding = " ".join(f"Investigators in cohort {i} recorded a distinct outcome." for i in range(8))
    recycled = {"sections": [
        {"heading": "Introduction", "paragraphs": [line + ". " + padding]},
        {"heading": "Conclusions", "paragraphs": [line + ". A different closing thought here."]},
    ]}
    check("recycled-phrasing flagged", "recycled-phrasing" in rules(recycled))

    # A varied, specific draft must stay clean. Written out rather than looped:
    # a loop produces near-identical sections, which these rules rightly reject.
    good = {"sections": [
        {"heading": "Introduction", "paragraphs": [
            "Rotating rosters are the dominant scheduling pattern in acute hospital "
            "nursing, and their health consequences have been studied for three "
            "decades. Whether any intervention reliably offsets those consequences "
            "remains unresolved."]},
        {"heading": "Mechanisms", "paragraphs": [
            "Circadian misalignment is the account most often advanced, resting "
            "largely on preclinical work in which light exposure was manipulated "
            "directly. Human evidence for the pathway is indirect."]},
        {"heading": "Interventions", "paragraphs": [
            "A 12-week randomised trial reported a 62% reduction in insomnia "
            "symptoms among rotating-shift nurses. A later cohort study of "
            "fixed-night staff did not reproduce that finding, and investigators "
            "attributed the discrepancy to how rapidly each roster rotated."]},
        {"heading": "Populations", "paragraphs": [
            "Across three cohorts the direction of effect held, though its "
            "magnitude differed considerably and the confidence intervals were "
            "wide. No controlled study has enrolled workers over 55."]},
        {"heading": "Conclusions", "paragraphs": [
            "Roster design plausibly matters more than any individual sleep "
            "intervention, but no trial has compared the two directly. An "
            "adequately powered comparison would settle most of what is uncertain."]},
    ]}
    check("a specific, varied draft passes", not (rules(good) & SUBSTANCE_RULES))

    # The abstract, key points and Introduction written as one paragraph three
    # times. Paraphrased too loosely for the 8-word recycled-phrasing rule, which
    # is exactly how a real shipped draft slipped through: its Introduction
    # repeated 38% of the abstract's 6-word runs and its key points 24%.
    abstract = (
        "Night shift work has been linked to adverse health outcomes, including "
        "metabolic, immune and cardiovascular impairments. The disruption of "
        "circadian rhythms, particularly in shift workers, has been increasingly "
        "associated with these outcomes. This review explores the impact of "
        "artificial light on the health of night shift workers, with a focus on "
        "the circadian system and its health implications."
    )
    echoed = {
        "abstract": abstract,
        "key_points": [
            "The disruption of circadian rhythms, particularly in shift workers, "
            "has been increasingly associated with these outcomes. [1]",
            "Artificial light has been linked to adverse health outcomes, including "
            "metabolic, immune and cardiovascular impairments. [2]",
        ],
        "sections": [{"heading": "Introduction", "paragraphs": [
            "This review explores the impact of artificial light on the health of "
            "night shift workers, with a focus on the circadian system and its "
            "health implications. The disruption of circadian rhythms, particularly "
            "in shift workers, has been increasingly associated with these outcomes."]}],
    }
    check("echoed-abstract flagged", "echoed-abstract" in rules(echoed))
    check("echoed-abstract is a substance rule", "echoed-abstract" in SUBSTANCE_RULES)

    # Sources cited only ever in a bundle have had nothing said about them.
    bundled = dict(good)
    bundled["sections"] = list(good["sections"])
    bundled["sections"][2] = {"heading": "Interventions", "paragraphs": [
        "Several studies report benefit from roster redesign [1, 2, 3, 4]. The "
        "direction of effect was consistent across them [1, 2, 3, 4]. One review "
        "considered the same question [5]."]}
    check("bundled-citations flagged", "bundled-citations" in rules(bundled))

    solo = dict(bundled)
    solo["sections"] = list(bundled["sections"])
    solo["sections"][2] = {"heading": "Interventions", "paragraphs": [
        "A randomised trial in rotating-shift nurses reported fewer insomnia "
        "symptoms [1]. A fixed-night cohort did not reproduce it [2]. A third "
        "cohort found the effect only in workers under 40 [3]. A pooled analysis "
        "reached no conclusion [4]. Together these suggest roster speed matters "
        "[1, 2, 3, 4]."]}
    check("citing sources individually clears bundled-citations",
          "bundled-citations" not in rules(solo))

    # The curated sample is the calibration reference: these rules must never
    # reject it, or they are measuring the wrong thing.
    from articlegen.demo import SAMPLE_ARTICLE
    check("the curated demo sample still passes",
          not (rules(SAMPLE_ARTICLE) & SUBSTANCE_RULES))

    # The brief must invert when the fix requires adding material, not rewording.
    brief = revision_brief(check_style(thin))
    check("substance brief asks for sources", "SOURCES" in brief)
    check("substance brief does not forbid new numbers",
          "do not introduce new claims or numbers" not in brief)

    # Register faults only — enough sections that no substance rule fires, so
    # this tests which brief is chosen rather than how many sections there are.
    register_only = dict(good)
    register_only["sections"] = list(good["sections"])
    register_only["sections"][0] = {
        "heading": "Introduction",
        "paragraphs": ["You should note that this clearly proves the point!"],
    }
    reg_brief = revision_brief(check_style(register_only))
    check("register-only brief still forbids new numbers",
          "do not introduce new claims or numbers" in reg_brief)


def test_source_failures_are_distinguishable() -> None:
    """An API refusing must not be reported as the topic having no literature.

    Every failure used to collapse to an empty list, so a rate-limited API
    produced "no papers found for this topic" — sending the user off to reword
    a query that was fine.
    """
    from articlegen import sources
    from articlegen.pipeline import NoPapersFound

    real = (sources.search_semantic_scholar, sources.search_openalex,
            sources.search_europe_pmc, sources.search_arxiv)
    refuse = lambda msg: lambda q, limit=15: (_ for _ in ()).throw(
        sources.SearchFailure(msg))
    try:
        # Every source refuses. The cache is cleared between phases because both
        # use the same query, and a cached refusal would otherwise answer for
        # the source the second phase is trying to prove works.
        sources.clear_search_cache()
        sources.search_semantic_scholar = refuse("HTTP 429 after 3 attempts")
        sources.search_openalex = refuse("HTTP 403")
        sources.search_europe_pmc = refuse("HTTP 503 after 3 attempts")
        sources.search_arxiv = refuse("HTTP 429 after 3 attempts")
        outcomes: list[dict] = []
        papers = sources.gather_evidence(["x"], outcomes=outcomes)
        check("no papers when all refuse", papers == [])
        check("every failure recorded", len([o for o in outcomes if o["error"]]) == 4)
        check("the reason is kept", any("429" in o["error"] for o in outcomes))

        # Two sources down, one fine — the run must survive.
        sources.clear_search_cache()
        sources.search_openalex = lambda q, limit=15: [
            sources.Paper(title=f"P{i}", abstract="a") for i in range(3)]
        outcomes = []
        papers = sources.gather_evidence(["x"], outcomes=outcomes)
        check("one working source is enough", len(papers) == 3)
        check("the failed sources are still recorded",
              any(o["error"] for o in outcomes) and any(not o["error"] for o in outcomes))
    finally:
        sources.clear_search_cache()
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = real

    check("NoPapersFound carries the distinction",
          NoPapersFound("x", sources_failed=True).sources_failed is True)
    check("and defaults to a topic problem", NoPapersFound("x").sources_failed is False)


def test_front_end_models_match_the_allowlist() -> None:
    """The web app has no model picker; the public path is Luna.

    A Settings dropdown used to send a visitor-chosen model. That UI is gone.
    The remaining invariant is that the hosted model is on the allowlist, and
    the page does not advertise a model the server would drop.
    """
    import re

    from articlegen import llm
    from articlegen.web import ALLOWED_MODELS

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "index.html"), encoding="utf-8") as f:
        page = f.read()

    check("the public model is on the allowlist",
          llm.OPENROUTER_PUBLIC_MODEL in ALLOWED_MODELS)
    check("the page has no provider picker", 'id="providerSelect"' not in page)
    check("the page has no PROVIDERS map", "const PROVIDERS" not in page)
    offered = re.findall(r"^\s*model: '([^']+)'", page, re.MULTILINE)
    for model in offered:
        check(f"index.html names {model}, which the server accepts",
              model in ALLOWED_MODELS)


def test_public_generation_forces_luna_on_the_hosted_key() -> None:
    """A request on the hosted key must write Luna, never Opus.

    The public site pays from ARTICLEGEN_PUBLIC_OPENROUTER_KEY. If `_credentials`
    honoured the payload's model, a crafted POST could select
    `anthropic/claude-opus-5` and bill the host at $5/$25. A visitor-supplied
    key in the payload is ignored for the same reason.
    """
    from articlegen import llm, web

    check("Luna is the public model",
          llm.OPENROUTER_PUBLIC_MODEL == "openai/gpt-5.6-luna")
    check("and the server accepts it",
          llm.OPENROUTER_PUBLIC_MODEL in web.ALLOWED_MODELS)
    check("CLI default is still Opus",
          llm.OPENROUTER_DEFAULT_MODEL == "anthropic/claude-opus-5")
    check("the two defaults are not the same model",
          llm.OPENROUTER_PUBLIC_MODEL != llm.OPENROUTER_DEFAULT_MODEL)

    saved = {k: os.environ.get(k) for k in (
        "ARTICLEGEN_PUBLIC_OPENROUTER_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ARTICLEGEN_PUBLIC_OPENROUTER_KEY"] = "sk-or-v1-hosted"

        check("hosted key enables public generation", web.public_generation_enabled())
        key, model = web._credentials({
            "model": llm.OPENROUTER_DEFAULT_MODEL, "key": ""})
        check("keyless hosted request uses the hosted key",
              key == "sk-or-v1-hosted")
        check("and forces Luna even if the payload asked for Opus",
              model == llm.OPENROUTER_PUBLIC_MODEL)

        key, model = web._credentials({
            "model": llm.OPENROUTER_DEFAULT_MODEL, "key": "sk-or-v1-visitor"})
        check("a visitor key in the payload is ignored",
              key == "sk-or-v1-hosted")
        check("and Luna is still forced",
              model == llm.OPENROUTER_PUBLIC_MODEL)

        os.environ.pop("ARTICLEGEN_PUBLIC_OPENROUTER_KEY", None)
        check("no hosted key means public generation is off",
              not web.public_generation_enabled())
        key, model = web._credentials({"model": llm.OPENROUTER_PUBLIC_MODEL})
        check("keyless with no host key has no api_key", key is None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_search_cache() -> None:
    """A repeated query must not cost a second call against the shared quota.

    The scholarly APIs meter against this server's IP and refuse constantly, so
    the same topic searched twice in a day is far likelier to fail the second
    time than to return anything new. Caching is what makes a demo repeatable.
    """
    from articlegen import sources

    real = (sources.search_semantic_scholar, sources.search_openalex,
            sources.search_europe_pmc, sources.search_arxiv)
    calls = {"n": 0, "fail": 0}
    try:
        sources.clear_search_cache()

        def counting(q, limit=15):
            calls["n"] += 1
            return [sources.Paper(title=f"{q}-paper", abstract="a", year=2024)]

        def failing(q, limit=15):
            calls["fail"] += 1
            raise sources.SearchFailure("HTTP 429 after 3 attempts")

        sources.search_semantic_scholar = counting
        sources.search_openalex = failing
        sources.search_europe_pmc = failing
        sources.search_arxiv = failing

        first: list[dict] = []
        got_first = sources.gather_evidence(["topic a"], outcomes=first)
        second: list[dict] = []
        got_second = sources.gather_evidence(["topic a"], outcomes=second)

        check("the same query is fetched once, not twice", calls["n"] == 1)
        check("the cached run returns the same papers", got_first == got_second)
        check("the first run is not marked cached",
              all(not o["cached"] for o in first))
        check("the second run is served from cache",
              all(o["cached"] for o in second))

        # A refusal is cached too, so a rate-limited source is not re-attempted
        # by every request that arrives while the limit is still in force.
        check("a refusal is cached, not retried", calls["fail"] == 3)
        check("and is still reported as a failure",
              sum(1 for o in second if o["error"]) == 3)

        # A different query is a different key — the cache must not answer for
        # a topic nobody searched.
        sources.gather_evidence(["topic b"], outcomes=[])
        check("a new query still fetches", calls["n"] == 2)

        sources.clear_search_cache()
        sources.gather_evidence(["topic a"], outcomes=[])
        check("clearing the cache forces a fresh fetch", calls["n"] == 3)
    finally:
        sources.clear_search_cache()
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = real


def test_search_cache_survives_a_new_process() -> None:
    """A second CLI run on the same topic makes no search request at all.

    Every `articlegen draft` is a fresh process, so an in-memory cache only
    ever helped the web server and the README's "re-running is instant" was
    false for the CLI. The three caches now round-trip through one JSON file
    under ARTICLEGEN_CACHE_DIR. A new process is simulated the honest way: the
    dicts are emptied and the loaded flag reset, so the only place the second
    run can get its answer from is the file.
    """
    import shutil
    import tempfile
    import time

    from articlegen import sources

    def forget_process() -> None:
        # What a fresh interpreter starts with: empty dicts, nothing loaded.
        sources._search_cache.clear()
        sources._fulltext_cache.clear()
        sources._pmcid_cache.clear()
        sources._cache_loaded = False

    real = (sources.search_semantic_scholar, sources.search_openalex,
            sources.search_europe_pmc, sources.search_arxiv)
    real_ttl = sources._CACHE_TTL
    saved_dir = os.environ.get("ARTICLEGEN_CACHE_DIR")
    folder = tempfile.mkdtemp(prefix="articlegen-cache-test-")
    os.environ["ARTICLEGEN_CACHE_DIR"] = folder
    calls = {"n": 0}
    try:
        sources.clear_search_cache()
        cache_file = os.path.join(folder, "cache.json")
        check("the cache file lives under ARTICLEGEN_CACHE_DIR",
              str(sources._cache_file()) == cache_file)

        def counting(q, limit=15):
            calls["n"] += 1
            return [sources.Paper(
                title=f"{q} paper", abstract="a", year=2024, doi="10.1000/x1",
                authors=["A", "B"], venue="V", citation_count=3, url="https://x",
                source="semantic_scholar", pmcid="PMC1", is_open_access=True,
                publication_types=("journal article", "review"))]

        sources.search_semantic_scholar = counting
        sources.search_openalex = counting
        sources.search_europe_pmc = counting
        sources.search_arxiv = counting

        first: list[dict] = []
        got_first = sources.gather_evidence(["topic a"], outcomes=first)
        check("the first run searches every source", calls["n"] == 4)
        check("the first run writes the file", os.path.isfile(cache_file))
        with open(cache_file, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        check("the file holds the search entries",
              {e["source"] for e in on_disk["search"]}
              == {"semantic_scholar", "openalex", "europe_pmc", "arxiv"})
        check("the file names the query and limit",
              on_disk["search"][0]["query"] == "topic a"
              and isinstance(on_disk["search"][0]["limit"], int))

        # Second "process": nothing in memory, and the file is the only memory.
        forget_process()
        second: list[dict] = []
        got_second = sources.gather_evidence(["topic a"], outcomes=second)
        check("the second process makes no search request", calls["n"] == 4)
        check("every outcome in the second process is cached",
              second and all(o["cached"] for o in second))
        check("the papers round-trip unchanged", got_second == got_first)
        check("tuple fields come back as tuples",
              got_second and got_second[0].publication_types == ("journal article", "review"))

        # The other two caches ride in the same file, with the same shape rule.
        with sources._cache_open():
            sources._fulltext_cache["PMC1"] = (time.time() + 60, "body text", "")
            sources._fulltext_cache["papers:10.1000/x1"] = (time.time() + 60, "", "queued_ckn")
            sources._pmcid_cache["10.1000/x1"] = (time.time() + 60, "PMC1", True)
            sources._save_cache_locked()
        forget_process()
        with sources._cache_open():
            check("full text round-trips",
                  sources._fulltext_cache.get("PMC1", ("", ""))[1] == "body text")
            check("the papers CLI status round-trips",
                  sources._fulltext_cache.get("papers:10.1000/x1", ("", "", ""))[2] == "queued_ckn")
            check("the PMCID lookup round-trips",
                  sources._pmcid_cache.get("10.1000/x1") == sources._pmcid_cache["10.1000/x1"]
                  and sources._pmcid_cache["10.1000/x1"][1:] == ("PMC1", True))

        # An expired entry is not loaded, so yesterday's refusal cannot answer.
        with sources._cache_open():
            sources._search_cache[("openalex", "stale", 15)] = (time.time() - 1, [], "HTTP 429")
            sources._save_cache_locked()
        forget_process()
        with sources._cache_open():
            check("an expired entry is dropped on load",
                  ("openalex", "stale", 15) not in sources._search_cache)

        # The file never carries a key: nothing from the environment goes in.
        with open(cache_file, encoding="utf-8") as fh:
            text = fh.read()
        check("the file carries no API key", "sk-" not in text and "API_KEY" not in text)

        # clear_search_cache() empties the file too, so a fresh process after a
        # clear has to search again.
        sources.clear_search_cache()
        check("clearing removes the file", not os.path.exists(cache_file))
        forget_process()
        sources.gather_evidence(["topic a"], outcomes=[])
        check("after a clear the next process searches again", calls["n"] == 8)

        # A corrupt file is one stderr line and an empty start, never a failure.
        with open(cache_file, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        forget_process()
        sources.gather_evidence(["topic a"], outcomes=[])
        check("a corrupt file is ignored and the search runs", calls["n"] == 12)
        with open(cache_file, encoding="utf-8") as fh:
            check("the corrupt file is replaced by a valid one",
                  isinstance(json.load(fh), dict))

        # TTL 0 disables the file along with everything else.
        sources.clear_search_cache()
        sources._CACHE_TTL = 0
        forget_process()
        sources.gather_evidence(["topic a"], outcomes=[])
        check("TTL 0 writes no file", not os.path.exists(cache_file))
    finally:
        sources._CACHE_TTL = real_ttl
        sources.clear_search_cache()
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = real
        if saved_dir is None:
            os.environ.pop("ARTICLEGEN_CACHE_DIR", None)
        else:
            os.environ["ARTICLEGEN_CACHE_DIR"] = saved_dir
        shutil.rmtree(folder, ignore_errors=True)


def test_first_semantic_scholar_refusal_buys_one_patient_round() -> None:
    """The first Semantic Scholar refusal of a run buys one patient retry (#148).

    Measured over four runs spanning more than an hour: every first keyless
    Semantic Scholar query returned HTTP 429 after its three tries, the
    `exhausted` set then skipped the source for the rest of that run, and every
    draft that session was written without it.

    The patient wait stays at _MAX_BACKOFF (30s) and is scoped to Semantic
    Scholar only, once per run. A second refusal still exhausts the source for
    the run, and non-retryable failures or probe runs (patient=False) get no wait.
    """
    from articlegen import sources

    check("the patient wait stays within the backoff cap",
          sources._S2_PATIENT_WAIT <= sources._MAX_BACKOFF)

    real_get = sources._get_with_retry
    real_sleep = sources.time.sleep
    reals = (sources.search_semantic_scholar, sources.search_openalex,
             sources.search_europe_pmc, sources.search_arxiv)
    try:
        sources.clear_search_cache()

        # 1. First refusal buys a second round; scoped to S2; second refusal exhausts for the run
        url_calls: dict[str, int] = {}
        waits: list[float] = []

        def fake_get_429(url, params=None, headers=None, tries=3):
            url_calls[url] = url_calls.get(url, 0) + 1
            exc = sources.SearchFailure("HTTP 429 after 3 attempts")
            exc.retry_later = True
            raise exc

        def fake_sleep(seconds):
            waits.append(seconds)

        sources._get_with_retry = fake_get_429
        sources.time.sleep = fake_sleep
        sources.search_europe_pmc = lambda q, limit=15: []
        sources.search_arxiv = lambda q, limit=15: []

        outs: list[dict] = []
        sources.gather_evidence(["q one", "q two"], outcomes=outs)

        s2_calls = url_calls.get(sources.SEMANTIC_SCHOLAR_URL, 0)
        oa_calls = url_calls.get(sources.OPENALEX_URL, 0)

        check("first Semantic Scholar refusal buys a second round", s2_calls == 2)
        check("exactly one patient wait was recorded, matching _S2_PATIENT_WAIT",
              waits == [sources._S2_PATIENT_WAIT])
        check("patient round is scoped to Semantic Scholar (OpenAlex called once)",
              oa_calls == 1)
        check("second S2 refusal still exhausts the source for the run",
              any(o["source"] == "semantic_scholar" and o["query"] == "q two"
                  and "skipped (already failed this run)" in o["error"] for o in outs))

        # 2. Non-retryable failure gets no patient wait (maintains _MAX_BACKOFF invariant)
        sources.clear_search_cache()
        url_calls.clear()
        waits.clear()

        def fake_get_non_retryable(url, params=None, headers=None, tries=3):
            url_calls[url] = url_calls.get(url, 0) + 1
            exc = sources.SearchFailure("HTTP 400")
            exc.retry_later = False
            raise exc

        sources._get_with_retry = fake_get_non_retryable
        outs_non_retryable: list[dict] = []
        sources.gather_evidence(["q three"], outcomes=outs_non_retryable)

        check("non-retryable failure gets no patient wait",
              url_calls.get(sources.SEMANTIC_SCHOLAR_URL, 0) == 1 and waits == [])

        # 3. Pre-flight probe (patient=False) does not buy the round
        sources.clear_search_cache()
        url_calls.clear()
        waits.clear()
        sources._get_with_retry = fake_get_429
        outs_probe: list[dict] = []
        sources.gather_evidence(["q four"], outcomes=outs_probe, patient=False)

        check("pre-flight probe (patient=False) buys no patient wait",
              url_calls.get(sources.SEMANTIC_SCHOLAR_URL, 0) == 1 and waits == [])

    finally:
        sources.clear_search_cache()
        sources._get_with_retry = real_get
        sources.time.sleep = real_sleep
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = reals
        sources._s2_patient_round_spent = False


def test_read_timeout_does_not_retry() -> None:
    """A socket timeout already spent the backoff cap; do not retry twice more.

    /api/diag on 5 September 2026 spent 85s, almost all of it arXiv
    ReadTimeout 30s × 3. Cloudflare in front of Render drops the request
    around 100s, so a live draft then looks like the backend is down.
    """
    from articlegen import sources

    calls: list = []
    real_get = sources.requests.get
    real_sleep = sources.time.sleep
    saved_hosted = os.environ.get("ARTICLEGEN_STATELESS")
    try:
        os.environ.pop("ARTICLEGEN_STATELESS", None)

        def boom(*a, **kw):
            calls.append(kw.get("timeout"))
            raise sources.requests.ReadTimeout("read timed out")

        sources.requests.get = boom
        sources.time.sleep = lambda s: calls.append(("sleep", s))
        raised = None
        try:
            sources._get_with_retry("http://example.test", {}, {})
        except sources.SearchFailure as exc:
            raised = exc
        check("a socket timeout is a SearchFailure", raised is not None)
        check("and is not retried", calls == [30.0])
        check("retry_later is False so S2 does not buy a patient wait",
              raised is not None and getattr(raised, "retry_later", True) is False)

        os.environ["ARTICLEGEN_STATELESS"] = "1"
        calls.clear()
        try:
            sources._get_with_retry("http://example.test", {}, {})
        except sources.SearchFailure:
            pass
        check("hosted search timeout is 10s, still not retried", calls == [10.0])
    finally:
        sources.requests.get = real_get
        sources.time.sleep = real_sleep
        if saved_hosted is None:
            os.environ.pop("ARTICLEGEN_STATELESS", None)
        else:
            os.environ["ARTICLEGEN_STATELESS"] = saved_hosted


def test_preflight_stops_when_a_source_answers() -> None:
    """Pre-flight only needs to know someone is home.

    Waiting out arXiv after OpenAlex already answered is how a hosted draft
    burned the proxy window before the first paid call.
    """
    from articlegen import pipeline, sources

    called: list[str] = []
    reals = (sources.search_semantic_scholar, sources.search_openalex,
             sources.search_europe_pmc, sources.search_arxiv)
    saved_ok, saved_fail = pipeline._sources_last_ok, pipeline._sources_last_fail
    try:
        sources.clear_search_cache()

        def s2(q, limit=15):
            called.append("s2")
            raise sources.SearchFailure("HTTP 429")

        def oa(q, limit=15):
            called.append("oa")
            return [sources.Paper(title="P", abstract="enough abstract text")]

        def epmc(q, limit=15):
            called.append("epmc")
            return []

        def arx(q, limit=15):
            called.append("arxiv")
            return []

        sources.search_semantic_scholar = s2
        sources.search_openalex = oa
        sources.search_europe_pmc = epmc
        sources.search_arxiv = arx
        pipeline._sources_last_ok = 0.0
        pipeline._sources_last_fail = (0.0, "")
        pipeline._preflight_sources("topic", lambda m: None)
        check("preflight called S2 then OpenAlex", called == ["s2", "oa"])
        check("and did not wait for Europe PMC or arXiv after a source answered",
              "epmc" not in called and "arxiv" not in called)
    finally:
        sources.clear_search_cache()
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = reals
        pipeline._sources_last_ok = saved_ok
        pipeline._sources_last_fail = saved_fail


def test_polite_pool_identification() -> None:
    """We must identify ourselves, and respect a cool-off the server asks for.

    Requests' default "python-requests/x.y" arriving from a cloud provider's
    shared egress IP is the profile CDNs throttle first, and a fixed 2s/4s
    backoff ignores a server that just told us exactly when to come back.
    """
    from articlegen import sources

    real = os.environ.get("OPENALEX_MAILTO")
    try:
        os.environ.pop("OPENALEX_MAILTO", None)
        ua = sources._user_agent()
        check("user-agent names the project", "articlegen/" in ua)
        check("and is not the requests default", "python-requests" not in ua)
        check("no mailto when unconfigured", "mailto:" not in ua)

        os.environ["OPENALEX_MAILTO"] = "you@example.com"
        check("mailto rides in the header when set",
              "mailto:you@example.com" in sources._user_agent())
    finally:
        os.environ.pop("OPENALEX_MAILTO", None)
        if real is not None:
            os.environ["OPENALEX_MAILTO"] = real

    class FakeResp:
        def __init__(self, retry_after=None):
            self.headers = {} if retry_after is None else {"Retry-After": retry_after}

    check("a short Retry-After wins over the backoff",
          sources._retry_delay(FakeResp("10"), 2.0) == 10.0)
    check("but never shortens it",
          sources._retry_delay(FakeResp("1"), 4.0) == 4.0)
    check("no header falls back to the backoff",
          sources._retry_delay(FakeResp(), 2.0) == 2.0)
    check("an HTTP-date form falls back too",
          sources._retry_delay(FakeResp("Wed, 21 Oct 2026 07:28:00 GMT"), 2.0) == 2.0)
    check("a cool-off past the cap gives up instead of waiting",
          sources._retry_delay(FakeResp("600"), 2.0) is None)
    check("a connection error has no response to read",
          sources._retry_delay(None, 2.0) == 2.0)


def test_europe_pmc_parsing() -> None:
    """Europe PMC records parse into Papers without trusting any field.

    Real records have string years, absent DOIs/journals, and HTML inside the
    abstract. The markup matters beyond cosmetics: verify.py checks statistics
    by substring presence in the abstract, so a figure adjacent to a tag would
    be reported unverifiable if tags were left in.
    """
    from articlegen import render, sources

    payload = {"resultList": {"result": [
        {   # the full-featured record
            "id": "38000001", "source": "MED",
            "title": "A <i>trial</i> of something",
            "abstractText": "<h4>Background</h4>Depression affects 20% of adults.<h4>Results</h4>Improved.",
            "pubYear": "2026", "citedByCount": 7, "doi": "10.1000/xyz",
            "authorString": "Kim SH, Jang G",
            # The real shape: Europe PMC gives surname and initials separately,
            # and `fullName` surname-first — the opposite of OpenAlex.
            "authorList": {"author": [
                {"fullName": "Kim SH", "lastName": "Kim", "initials": "SH", "firstName": "Su Hyun"},
                {"fullName": "Jang G", "lastName": "Jang", "initials": "G", "firstName": "Geunsoo"},
            ]},
            "journalInfo": {"journal": {"title": "J Affect Disord"}},
            "pubType": "research-article; Randomized Controlled Trial",
        },
        {   # pubTypeList list-form (the real EPMC API shape; pubType flat field is null)
            "id": "38000004", "source": "MED",
            "title": "Treatment outcomes in adults with bipolar disorder",
            "abstractText": "Results of synthesis.",
            "pubYear": "2023", "citedByCount": 12, "doi": "10.1001/jamapsychiatry.2023.2225",
            "pubTypeList": {"pubType": ["Meta-Analysis", "research-article", "Systematic Review", "Journal Article"]},
        },
        {   # sparse: book chapter — no journal, no doi, bad year
            "id": "PPR000002", "source": "PPR",
            "title": "Sparse record",
            "abstractText": "An abstract.",
            "pubYear": "n.d.",
        },
        {   # no abstract despite the filter — must be dropped, like the others do
            "id": "38000003", "source": "MED",
            "title": "No abstract", "pubYear": "2025",
        },
    ]}}

    class FakeResp:
        def json(self):
            return payload

    real = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeResp()
        papers = sources.search_europe_pmc("depression", limit=4)
    finally:
        sources._get_with_retry = real

    check("abstract-less record dropped", len(papers) == 3)
    full, meta, sparse = papers
    check("markup stripped from the abstract", "<h4>" not in full.abstract)
    check("statistics survive stripping adjacent tags",
          "Depression affects 20% of adults." in full.abstract)
    check("markup stripped from the title", full.title == "A trial of something")
    check("string year becomes int", full.year == 2026)
    check("publication_types parsed from pubType", "randomized controlled trial" in full.publication_types)
    check("publication_types parsed from pubTypeList.pubType list",
          "meta-analysis" in meta.publication_types and "systematic review" in meta.publication_types)
    check("classify_design identifies synthesis from pubTypeList alone",
          sources.classify_design(meta) == "synthesis")
    # Europe PMC names arrive surname-first; the renderer takes the last token as
    # the surname, so they are normalised to given-name-first at parse time.
    # Passing `fullName` through printed "SH, K." for "Kim SH" in every reference.
    check("authors are normalised to given-name-first", full.authors == ["S H Kim", "G Jang"])
    check("and render as a correct reference line",
          render._reference_authors(full) == "Kim, S. H. & Jang, G.")
    check("and as a correct short form", render._short_author(full) == "Kim & Jang")
    check("a consortium author survives without a surname",
          sources._europe_pmc_author({"collectiveName": "GBD 2019 Collaborators"})
          == "GBD 2019 Collaborators")
    check("a record with only fullName still yields a name",
          sources._europe_pmc_author({"fullName": "Lee K"}) == "Lee K")
    check("dotted initials are split too",
          sources._europe_pmc_author({"lastName": "Kim", "initials": "S.H."}) == "S H Kim")
    check("firstName is used when initials are absent",
          sources._europe_pmc_author({"lastName": "Kim", "firstName": "Su Hyun"}) == "Su Hyun Kim")
    check("journal title found", full.venue == "J Affect Disord")
    check("europepmc url built from source+id", full.url.endswith("/MED/38000001"))
    check("unparseable year becomes None", sparse.year is None)
    check("missing doi/journal tolerated", sparse.doi == "" and sparse.venue == "")
    check("source is named in DATABASE_NAMES", "europe_pmc" in sources.DATABASE_NAMES)

    import inspect
    check("gather_evidence queries europe_pmc",
          '"europe_pmc"' in inspect.getsource(sources.gather_evidence))


def test_table_one_reports_sample_size_registration_and_funding() -> None:
    """Table 1 prints three facts read off metadata and abstracts, not a grade (#244).

    Sample size, trial registration and funding-statement presence are text
    reads, never a model call and never an appraisal — the pipeline still
    grades nothing.
    """
    from articlegen import render, sources
    from articlegen.sources import Paper

    # 1. Derivation lives in __post_init__, so every source inherits it with
    # nothing passed in explicitly.
    p = Paper(title="A trial", abstract="We enrolled n = 240 adults. Registered NCT01234567.")
    check("sample_size derived from the abstract", p.sample_size == 240)
    check("registration derived from the abstract", p.registration == "NCT01234567")

    # 2. Europe PMC parser reads grantsList and (title+abstract+keywords) for a
    # registration id.
    payload = {"resultList": {"result": [
        {
            "id": "39000001", "source": "MED",
            "title": "A funded trial",
            "abstractText": "We randomised 120 patients.",
            "pubYear": "2024",
            "grantsList": {"grant": [{"grantId": "G1", "agency": "NHMRC"}]},
            "keywordList": {"keyword": ["NCT02222222", "depression"]},
        },
        {
            "id": "39000002", "source": "MED",
            "title": "An unfunded record",
            "abstractText": "We randomised 80 patients.",
            "pubYear": "2024",
        },
    ]}}

    class FakeResp:
        def json(self):
            return payload

    real = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeResp()
        funded, unfunded = sources.search_europe_pmc("depression", limit=2)
    finally:
        sources._get_with_retry = real

    check("grantsList presence sets has_funding_statement",
          funded.has_funding_statement is True)
    check("registration id picked up from keywordList",
          funded.registration == "NCT02222222")
    check("no grantsList means no funding statement",
          unfunded.has_funding_statement is False)

    # 3. Round-trip through to_dict/from_dict.
    rt = Paper.from_dict(p.to_dict())
    check("sample_size round-trips", rt.sample_size == p.sample_size)
    check("registration round-trips", rt.registration == p.registration)
    check("has_funding_statement round-trips",
          rt.has_funding_statement == p.has_funding_statement)

    # 4. _merge_duplicate: enrichment, not overwrite.
    kept = Paper(title="Same study", abstract="No numbers here.", sample_size=999)
    dup = Paper(title="Same study", abstract="No numbers here.",
                sample_size=50, registration="NCT03333333", has_funding_statement=True)
    sources._merge_duplicate(kept, dup)
    check("a kept sample_size is never overwritten by a duplicate's", kept.sample_size == 999)
    check("registration fills in from the duplicate when the kept copy has none",
          kept.registration == "NCT03333333")
    check("funding is OR'd across the merge", kept.has_funding_statement is True)

    kept2 = Paper(title="Other study", abstract="No numbers here.")
    dup2 = Paper(title="Other study", abstract="No numbers here.", sample_size=42)
    sources._merge_duplicate(kept2, dup2)
    check("a kept copy with no sample_size takes the duplicate's",
          kept2.sample_size == 42)

    # 5. Rendered output.
    paper = Paper(
        title="A randomised controlled trial of X",
        abstract="We randomised 364 patients. Registered NCT04655638.",
        year=2024, authors=["A Smith"], venue="BMJ",
        has_funding_statement=True,
    )
    html = render._table_html([paper], {})
    check("HTML header gains n", "<th>n</th>" in html)
    check("HTML header gains Registered", "<th>Registered</th>" in html)
    check("HTML header gains Funding", "<th>Funding</th>" in html)
    check("formatted sample size printed", "364" in html)
    check("registration id printed", "NCT04655638" in html)
    check("funding cell printed", "Grant listed" in html)
    check("no empty cell anywhere", "<td></td>" not in html)

    md = render._table_markdown([paper], {})
    header_cols = md.splitlines()[2].count("|")
    sep_cols = md.splitlines()[3].count("|")
    check("markdown separator matches header column count", header_cols == sep_cols)
    check("markdown carries the same three facts",
          "364" in md and "NCT04655638" in md and "Grant listed" in md)

    for banned in ("GRADE", "risk of bias", "quality score"):
        check(f"HTML never says {banned!r}", banned not in html)
        check(f"Markdown never says {banned!r}", banned not in md)


def test_sample_size_never_reads_a_year_or_a_page_number() -> None:
    """A sample size is read as a fact, and a year or page number is never one (#244)."""
    import json

    from articlegen import sources

    for text in (
        "Data were collected from 2015 to 2019 in primary care.",
        "Reported in J Emerg Med 2019;21(4):123-130.",
        "With 1 in 5 people doing shift work...",
        "Scores on the 21-item Hamilton scale improved.",
        "Remission was reached by 36.8% of patients in the first step.",
        "Adults aged 65 years or older were eligible.",
    ):
        check(f"never a false positive: {text!r}", sources.sample_size(text) is None)

    for text, expected in (
        ("n = 1,204", 1204),
        ("28 431 unique participants", 28431),
        ("randomised 364 patients", 364),
        ("Of the 17 subjects treated", 17),
    ):
        check(f"reads the count in {text!r}", sources.sample_size(text) == expected)

    check("NCT id found", sources.registration_id("see NCT04655638") == "NCT04655638")
    check("ACTRN id found",
          sources.registration_id("see ACTRN12613000710729") == "ACTRN12613000710729")
    check("ISRCTN id found", sources.registration_id("see ISRCTN91737921") == "ISRCTN91737921")
    check("short NCT id rejected", sources.registration_id("see NCT123") == "")
    check("bare ISRCTN rejected", sources.registration_id("see ISRCTN alone") == "")
    check("lowercase id comes back uppercased",
          sources.registration_id("see nct04655638") == "NCT04655638")

    path = os.path.join(os.path.dirname(__file__), "real_abstracts.json")
    corpus = json.load(open(path, encoding="utf-8"))
    mismatches = []
    for entry in corpus:
        got_n = sources.sample_size(entry["abstract"])
        if got_n != entry.get("expect_n"):
            mismatches.append((entry["title"][:50], "n", got_n, entry.get("expect_n")))
        got_reg = sources.registration_id(entry["abstract"])
        if got_reg != entry.get("expect_registration", ""):
            mismatches.append(
                (entry["title"][:50], "registration", got_reg, entry.get("expect_registration", "")))
    if mismatches:
        for title, field, got, expected in mismatches:
            print(f"      {title}: {field} got {got!r}, expected {expected!r}")
    check(f"sample size and registration match expectations on {len(corpus)} real abstracts",
          not mismatches)


def test_arxiv_parsing() -> None:
    """arXiv Atom entries parse into Papers, and an error entry is not one.

    arXiv is the only source that returns XML rather than JSON, the only one
    with no citation counts, and the only one that reports a bad query as a
    normal-looking entry — parsed naively that becomes a source titled "Error"
    in the reference list of a real article.
    """
    from articlegen import sources

    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2306.08302v3</id>
        <published>2023-06-14T17:59:59Z</published>
        <title>Gravity batteries
  for grid storage</title>
        <summary>  Storage costs fell by
  38% over the decade.
        </summary>
        <author><name>Ada Lovelace</name></author>
        <author><name>Alan Turing</name></author>
        <arxiv:doi>10.1000/arxiv-example</arxiv:doi>
        <arxiv:journal_ref>Nature Energy 9, 1 (2024)</arxiv:journal_ref>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <published>bad-date</published>
        <title>A preprint that was never published</title>
        <summary>An abstract.</summary>
        <author><name>Grace Hopper</name></author>
      </entry>
      <entry>
        <id>http://arxiv.org/api/errors#incorrect_id_format</id>
        <title>Error</title>
        <summary>incorrect id format for 1234</summary>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>No abstract</title>
        <summary>   </summary>
      </entry>
    </feed>"""

    class FakeResp:
        text = atom

    seen = {}
    real = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: (
            seen.update(params) or FakeResp())
        papers = sources.search_arxiv("gravity battery storage", limit=3)
    finally:
        sources._get_with_retry = real

    check("the error entry and the abstract-less one are dropped", len(papers) == 2)
    published, preprint = papers
    check("the title is unwrapped to one line",
          published.title == "Gravity batteries for grid storage")
    check("statistics survive the line the abstract was wrapped at",
          "fell by 38% over the decade." in published.abstract)
    check("the year comes from the published date", published.year == 2023)
    check("authors keep given-name-first order",
          published.authors == ["Ada Lovelace", "Alan Turing"])
    check("a journal_ref is preferred over the preprint label",
          published.venue == "Nature Energy 9, 1 (2024)")
    check("the arxiv doi is picked up", published.doi == "10.1000/arxiv-example")
    check("no journal_ref says preprint, so the reference list can",
          preprint.venue == "arXiv preprint")
    check("an unparseable date becomes None", preprint.year is None)
    check("arXiv publishes no citation counts", published.citation_count == 0)
    check("the abs url is kept as the link",
          preprint.url == "http://arxiv.org/abs/2401.00001v1")

    # The query expression, not the raw topic: `all:gravity battery storage`
    # searches arXiv for "gravity" alone.
    check("multi-word topics are ANDed, not sent verbatim",
          seen["search_query"] == "all:gravity AND all:battery AND all:storage")
    # `all:the` alone would match the whole archive, which is the only reason
    # function words are dropped — ANDing one excludes nothing.
    check("function words are dropped, short content words are not",
          sources._arxiv_query("AI in the ED") == "all:AI AND all:ED")
    check("a query of nothing but function words still searches for them",
          sources._arxiv_query("in the of") == "all:in AND all:the AND all:of")
    check("the term count is capped",
          sources._arxiv_query(" ".join(f"term{i}" for i in range(20))
                               ).count(" AND ") == sources._ARXIV_MAX_TERMS - 1)

    check("source is named in DATABASE_NAMES", "arxiv" in sources.DATABASE_NAMES)

    import inspect
    gather = inspect.getsource(sources.gather_evidence)
    check("gather_evidence queries arxiv", '"arxiv"' in gather)
    # Dedupe is first-seen-wins on the normalised title, so a preprint sharing a
    # title with the published version must lose. That is purely an ordering
    # property of this tuple.
    check("arXiv is queried last, so the published version wins dedupe",
          gather.index('"arxiv"') > gather.index('"europe_pmc"'))


def test_titles_arrive_without_markup() -> None:
    """Publisher markup never reaches a Paper title (issue #140).

    OpenAlex passes JATS through, so a real cited title arrived as
    "The <scp>Nurse-Police</scp> ... (<scp>N-PACT</scp>): ...". The tags reached
    Table 1, the reference list and the writer's prompt, and they defeated
    dedupe: `_normalize_title` keeps "scp" as a word, so the tagged and untagged
    copies of one paper were cited as two references.
    """
    from articlegen.sources import Paper, _normalize_title, _strip_title_markup
    from articlegen import sources

    clean = "The Nurse-Police Assistance Crisis Team (N-PACT): A new role for nursing"

    # Both spacings the publisher may send: tags tight against the bracket, and
    # tags with the space the markup itself introduced.
    tight = ("The <scp>Nurse-Police</scp> Assistance Crisis Team "
             "(<scp>N-PACT</scp>): A new role for nursing")
    spaced = ("The <scp>Nurse-Police</scp> Assistance Crisis Team "
              "( <scp>N-PACT</scp> ): A new role for nursing")
    check("tags removed and spacing normalised", _strip_title_markup(tight) == clean)
    check("and the spaces the tags left are closed too",
          _strip_title_markup(spaced) == clean)

    # Case and attributes.
    check("uppercase tags go too", _strip_title_markup("A <SCP>MDMA</SCP> trial")
          == "A MDMA trial")
    check("attributes do not save a tag",
          _strip_title_markup('Effects of <i class="genus">Lactobacillus</i> on IBS')
          == "Effects of Lactobacillus on IBS")
    check("sub/sup go too",
          _strip_title_markup("Serum 25(OH)D<sub>3</sub> and CO<sub>2</sub>")
          == "Serum 25(OH)D3 and CO2")

    # Negative control: comparison operators are not markup. A generic
    # `<[^>]+>` sweep eats "<65 versus >" and the title loses its meaning.
    operators = "Outcomes in adults aged <65 versus >80 years"
    check("comparison operators survive", _strip_title_markup(operators) == operators)

    # The choke point: every Paper, whichever API built it.
    check("Paper cleans its own title", Paper(title=tight, abstract="a").title == clean)

    # Which is what restores dedupe.
    check("tagged and untagged copies now dedupe together",
          _normalize_title(Paper(title=tight, abstract="a").title)
          == _normalize_title(Paper(title=clean, abstract="a").title))

    # And through a real OpenAlex parse, since that is where it came from.
    payload = {"results": [{
        "id": "https://openalex.org/W1", "title": tight,
        "publication_year": 2024, "cited_by_count": 3,
        "abstract_inverted_index": {"An": [0], "abstract.": [1]},
        "authorships": [{"author": {"display_name": "Ann Ab"}}],
        "primary_location": {"source": {"display_name": "J Psychiatr Nurs"},
                             "landing_page_url": "https://example.org/1"},
        "doi": "10.1000/npact",
    }]}

    class FakeResp:
        def json(self):
            return payload

    real = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeResp()
        papers = sources._openalex_page("nurse police", limit=1)
    finally:
        sources._get_with_retry = real
    check("an OpenAlex record parses with a clean title", papers[0].title == clean)


def test_candidate_papers_dedupe_by_doi() -> None:
    """Papers are deduped by normalised DOI first, then by title (issue #139).

    Two of three recent runs cited one paper twice because the search sources
    spell the same DOI three ways (resolver URL, mixed case, bare) and the
    titles differed by a subtitle or publisher markup.
    """
    from articlegen.sources import Paper, _normalize_doi
    from articlegen import sources

    # 1. _normalize_doi unit checks
    expected = "10.1001/jamapsychiatry.2025.1317"
    check("bare lowercase DOI normalises",
          _normalize_doi("10.1001/jamapsychiatry.2025.1317") == expected)
    check("mixed-case DOI is lowercased",
          _normalize_doi("10.1001/JAMAPsychiatry.2025.1317") == expected)
    check("https://doi.org/ prefix stripped",
          _normalize_doi("https://doi.org/10.1001/jamapsychiatry.2025.1317") == expected)
    check("http://dx.doi.org/ prefix stripped",
          _normalize_doi("http://dx.doi.org/10.1001/jamapsychiatry.2025.1317") == expected)
    check("doi: prefix stripped and trimmed",
          _normalize_doi("doi: 10.1001/JAMAPsychiatry.2025.1317") == expected)
    check("surrounding whitespace and punctuation stripped",
          _normalize_doi("  https://doi.org/10.1001/jamapsychiatry.2025.1317.  ") == expected)
    check("empty DOI returns empty string", _normalize_doi("") == "")
    # Junk values ("n/a", "unknown") must return "" rather than becoming a shared merge key
    # that would collapse unrelated records into one.
    check("junk non-DOI returns empty string", _normalize_doi("n/a") == "")
    check("unknown non-DOI returns empty string", _normalize_doi("unknown") == "")

    reals = (sources.search_semantic_scholar, sources.search_openalex,
             sources.search_europe_pmc, sources.search_arxiv)
    try:
        sources.search_semantic_scholar = lambda q, limit=15: []
        sources.search_arxiv = lambda q, limit=15: []

        # 2. Different casings/prefixes merge, keeping richer metadata and first-seen identity
        oa_thin = Paper(
            title="Janik Study: A Randomized Trial on Psychiatric Care",
            abstract="Abstract from OpenAlex",
            doi="https://doi.org/10.1001/jamapsychiatry.2025.1317",
            citation_count=0,
            year=None,
            source="openalex",
        )
        epmc_rich = Paper(
            title="Janik Study",
            abstract="Abstract from EPMC",
            doi="10.1001/JAMAPsychiatry.2025.1317",
            pmcid="PMC55",
            is_open_access=True,
            citation_count=12,
            year=2025,
            source="europe_pmc",
        )
        sources.search_openalex = lambda q, limit=15: [oa_thin]
        sources.search_europe_pmc = lambda q, limit=15: [epmc_rich]
        sources.clear_search_cache()
        merged = sources.gather_evidence(["q"], use_cache=False)

        check("same DOI in different formats merges to one record", len(merged) == 1)
        # First-seen title wins: this preserves the invariant that arXiv querying last
        # discards a preprint in favour of the peer-reviewed published version.
        check("first-seen title is kept",
              merged[0].title == "Janik Study: A Randomized Trial on Psychiatric Care")
        check("pmcid is enriched from duplicate",
              merged[0].pmcid == "PMC55" and merged[0].is_open_access)
        check("citation count is enriched from duplicate", merged[0].citation_count == 12)
        check("year is enriched from duplicate", merged[0].year == 2025)

        # 3. No-DOI records still dedupe by title
        no_doi_1 = Paper(title="No DOI Study", abstract="Abs 1", doi="")
        no_doi_2 = Paper(title="No DOI Study", abstract="Abs 2", doi="")
        sources.search_openalex = lambda q, limit=15: [no_doi_1]
        sources.search_europe_pmc = lambda q, limit=15: [no_doi_2]
        sources.clear_search_cache()
        merged_no_doi = sources.gather_evidence(["q"], use_cache=False)
        check("no-DOI records still dedupe by title", len(merged_no_doi) == 1)

        # 4. Distinct DOIs never merge on DOI
        diff_1 = Paper(title="Title A", abstract="Abs A", doi="10.1000/1")
        diff_2 = Paper(title="Title B", abstract="Abs B", doi="10.1000/2")
        sources.search_openalex = lambda q, limit=15: [diff_1]
        sources.search_europe_pmc = lambda q, limit=15: [diff_2]
        sources.clear_search_cache()
        merged_diff = sources.gather_evidence(["q"], use_cache=False)
        check("distinct DOIs and titles never merge", len(merged_diff) == 2)

        # Same title with distinct DOIs still merges via title fallback (the preprint case)
        preprint_pub_1 = Paper(title="Same Title Everywhere", abstract="Pub abs", doi="10.1000/journal.1")
        preprint_pub_2 = Paper(title="Same Title Everywhere", abstract="Preprint abs", doi="10.48550/arxiv.1")
        sources.search_openalex = lambda q, limit=15: [preprint_pub_1]
        sources.search_europe_pmc = lambda q, limit=15: [preprint_pub_2]
        sources.clear_search_cache()
        merged_preprint = sources.gather_evidence(["q"], use_cache=False)
        check("same title with different DOIs merges via title (preprint vs published)",
              len(merged_preprint) == 1)

    finally:
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = reals
        sources.clear_search_cache()


def test_preprints_are_marked_as_preprints() -> None:
    """Preprints are detected and marked in references and Table 1 (issue #144).

    A recent draft cited a Research Square preprint (10.21203/rs.3.rs-9924877/v1)
    beside a Cochrane review with nothing to tell them apart except a blank journal
    cell in Table 1.
    """
    from articlegen.sources import Paper, _looks_like_preprint
    from articlegen import render, sources

    # 1. Identifier detection
    check("Research Square DOI flagged",
          _looks_like_preprint("10.21203/rs.3.rs-9924877/v1", ""))
    check("Research Square resolver URL flagged",
          _looks_like_preprint("https://doi.org/10.21203/rs.3.rs-1/v1", ""))
    check("bioRxiv/medRxiv date-format DOI flagged",
          _looks_like_preprint("10.1101/2024.03.01.583912", ""))
    check("CSH journal Genome Research negative control not flagged",
          not _looks_like_preprint("10.1101/gr.123456", ""))
    check("CSH journal Perspectives negative control not flagged",
          not _looks_like_preprint("10.1101/cshperspect.a012345", ""))
    check("arXiv registered DOI flagged (mixed-case)",
          _looks_like_preprint("10.48550/arXiv.2401.00001", ""))
    check("arXiv URL with no DOI flagged",
          _looks_like_preprint("", "https://arxiv.org/abs/2401.00001"))
    check("ordinary journal DOI not flagged",
          not _looks_like_preprint("10.1001/jamapsychiatry.2025.1317", ""))
    check("empty identifier not flagged",
          not _looks_like_preprint("", ""))

    # 2. The choke point in Paper.__post_init__
    p_preprint = Paper(title="T", abstract="a", doi="10.21203/rs.3.rs-1/v1")
    check("Paper dataclass identifies preprint on construction", p_preprint.is_preprint is True)
    p_journal = Paper(title="T", abstract="a", doi="10.1001/jamapsychiatry.2025.1317")
    check("Paper dataclass leaves journal paper unflagged", p_journal.is_preprint is False)

    # 3. Parsers
    # OpenAlex type=preprint
    oa_payload = {"results": [{
        "id": "https://openalex.org/W1", "title": "OA Preprint",
        "publication_year": 2024, "cited_by_count": 0,
        "abstract_inverted_index": {"An": [0], "abstract.": [1]},
        "authorships": [{"author": {"display_name": "Author A"}}],
        "primary_location": {"source": {"display_name": ""},
                             "landing_page_url": "https://example.org/1"},
        "doi": "10.1000/ordinary-doi",
        "type": "preprint",
    }]}

    class FakeOAResp:
        def json(self):
            return oa_payload

    # Europe PMC source=PPR vs source=MED
    epmc_payload = {"resultList": {"result": [
        {
            "id": "PPR123", "source": "PPR", "title": "EPMC Preprint",
            "pubYear": "2024", "authorList": {"author": [{"fullName": "Author B"}]},
            "abstractText": "EPMC preprint abstract.", "doi": "10.1000/ordinary-epmc-doi",
            "pubType": "preprint",
        },
        {
            "id": "MED123", "source": "MED", "title": "EPMC Journal Paper",
            "pubYear": "2024", "authorList": {"author": [{"fullName": "Author C"}]},
            "abstractText": "EPMC journal abstract.", "doi": "10.1000/ordinary-med-doi",
            "journalInfo": {"journal": {"title": "Journal of Medicine"}},
        },
    ]}}

    class FakeEPMCResp:
        def json(self):
            return epmc_payload

    # arXiv Atom XML sample
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <published>2024-01-01T00:00:00Z</published>
        <title>arXiv Paper</title>
        <summary>arXiv abstract</summary>
        <author><name>Author D</name></author>
        <arxiv:doi>10.1000/arxiv-doi</arxiv:doi>
        <arxiv:journal_ref>Nature Energy 9, 1 (2024)</arxiv:journal_ref>
      </entry>
    </feed>"""

    class FakeArxivResp:
        text = atom

    real_get = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeOAResp()
        oa_papers = sources._openalex_page("test query", limit=1)
        check("OpenAlex type=preprint sets is_preprint", oa_papers[0].is_preprint is True)

        sources._get_with_retry = lambda url, params, headers: FakeEPMCResp()
        epmc_papers = sources.search_europe_pmc("test query", limit=2)
        check("Europe PMC PPR source sets is_preprint", epmc_papers[0].is_preprint is True)
        check("Europe PMC MED source leaves is_preprint False", epmc_papers[1].is_preprint is False)

        sources._get_with_retry = lambda url, params, headers: FakeArxivResp()
        arxiv_papers = sources.search_arxiv("test query", limit=1)
        check("arXiv search unconditionally sets is_preprint", arxiv_papers[0].is_preprint is True)
    finally:
        sources._get_with_retry = real_get

    # 4. Rendering
    paper_preprint = Paper(
        title="Preprint Study",
        abstract="Abstract of preprint",
        authors=["Alice Smith"],
        doi="10.21203/rs.3.rs-9924877/v1",
        year=2024,
        venue="",
    )
    paper_journal = Paper(
        title="Cochrane Review",
        abstract="Abstract of Cochrane review",
        authors=["Bob Jones"],
        doi="10.1002/14651858.CD001",
        year=2023,
        venue="Cochrane Database Syst Rev",
    )
    cited_pair = [paper_preprint, paper_journal]

    t_html = render._table_html(cited_pair, {})
    check("Table 1 HTML marks preprint in Source column", "Preprint" in t_html)
    check("Table 1 HTML keeps journal venue", "Cochrane Database Syst Rev" in t_html)

    t_md = render._table_markdown(cited_pair, {})
    check("Table 1 Markdown marks preprint in Source column", "Preprint" in t_md)
    check("Table 1 Markdown keeps journal venue", "Cochrane Database Syst Rev" in t_md)

    article_payload = {
        "title": "A Test Review Article",
        "abstract": "Summary with references [1] and [2].",
        "keywords": ["testing"],
        "evidence_note": "Evidence note [1].",
        "featured_study": {"source_index": 1, "method": "Trial", "results": "Results"},
        "sections": [
            {"heading": "Introduction", "paragraphs": ["Intro referencing [1] and [2]."]},
            {"heading": "Conclusions", "paragraphs": ["Conclusion [1]."]},
        ],
        "key_points": ["Key point referencing [2]."],
        "glossary": [],
        "references": [1, 2],
    }

    h_out = render.render_article(article_payload, cited_pair, "test topic", None, None, None)
    md_out = render.render_markdown(article_payload, cited_pair, "test topic", None, None, None)

    marker = "(preprint, not peer reviewed)"
    check("HTML render has preprint marker exactly once", h_out.count(marker) == 1)
    check("Markdown render has preprint marker exactly once", md_out.count(marker) == 1)

    # 5. _merge_duplicate adoption
    kept_no_doi = Paper(title="Duplicate Study", abstract="Abs", doi="", url="")
    dup_preprint_doi = Paper(title="Duplicate Study", abstract="Abs", doi="10.21203/rs.3.rs-9924877/v1")
    sources._merge_duplicate(kept_no_doi, dup_preprint_doi)
    check("adopting preprint DOI in merge sets is_preprint", kept_no_doi.is_preprint is True)

    kept_published = Paper(title="Published Paper", abstract="Abs", doi="10.1001/jamapsychiatry.2025.1317", is_preprint=False)
    dup_arxiv = Paper(title="Published Paper", abstract="Abs", doi="10.48550/arXiv.1", is_preprint=True)
    sources._merge_duplicate(kept_published, dup_arxiv)
    check("published paper does not inherit preprint flag across merge when kept has DOI", kept_published.is_preprint is False)


def test_retracted_records_never_reach_the_writer() -> None:
    """A clinical evidence tool that cites a retracted paper without saying so
    has a gap that matters (issue #242). OpenAlex's `is_retracted` and
    Crossref's `update-to`/`is-corrected-by` are one field away.
    """
    import inspect

    from articlegen import render, sources, writer
    from articlegen.sources import Paper

    # 1. The field is requested.
    check("OpenAlex asks for is_retracted", "is_retracted" in sources._OA_FIELDS)

    # 2. _openalex_page reads it.
    oa_payload = {"results": [
        {
            "id": "https://openalex.org/W1", "title": "Retracted work",
            "publication_year": 2024, "cited_by_count": 0,
            "abstract_inverted_index": {"An": [0], "abstract.": [1]},
            "authorships": [{"author": {"display_name": "Author A"}}],
            "primary_location": {"source": {"display_name": ""},
                                 "landing_page_url": "https://example.org/1"},
            "doi": "10.1000/retracted-doi",
            "type": "article",
            "is_retracted": True,
        },
        {
            "id": "https://openalex.org/W2", "title": "Sound work",
            "publication_year": 2024, "cited_by_count": 0,
            "abstract_inverted_index": {"An": [0], "abstract.": [1]},
            "authorships": [{"author": {"display_name": "Author B"}}],
            "primary_location": {"source": {"display_name": ""},
                                 "landing_page_url": "https://example.org/2"},
            "doi": "10.1000/sound-doi",
            "type": "article",
        },
    ]}

    class FakeOAResp:
        def json(self):
            return oa_payload

    real_get = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeOAResp()
        oa_papers = sources._openalex_page("test query", limit=2)
        check("OpenAlex is_retracted=True is read", oa_papers[0].is_retracted is True)
        check("absent is_retracted reads False, not None", oa_papers[1].is_retracted is False)
    finally:
        sources._get_with_retry = real_get

    # 3. check_record_status
    real_get = sources._get_with_retry
    real_ttl = sources._CACHE_TTL
    sources._CACHE_TTL = 0
    urls_called: list[str] = []

    def fake_status_get(url, params, headers):
        urls_called.append(url)
        if url == sources.OPENALEX_URL:
            class R:
                def json(self):
                    return {"results": [{"id": "W1", "is_retracted": True}]}
            return R()
        if url.startswith("https://api.crossref.org"):
            class R:
                def json(self):
                    return {"message": {"update-to": [{"type": "correction"}]}}
            return R()
        raise AssertionError(f"unexpected url {url}")

    try:
        sources._get_with_retry = fake_status_get
        p = Paper(title="T", abstract="a", doi="10.1000/ss-doi", source="Semantic Scholar")
        sources.check_record_status(p)
        check("Semantic Scholar record flagged retracted", p.is_retracted is True)
        check("Semantic Scholar record gets a correction note", p.correction_note == "Corrected")

        urls_called.clear()
        p_oa = Paper(title="T", abstract="a", doi="10.1000/oa-doi", source="OpenAlex")
        sources.check_record_status(p_oa)
        check("OpenAlex-sourced record makes no OpenAlex retraction call",
              sources.OPENALEX_URL not in urls_called)
        check("OpenAlex-sourced record still makes the Crossref call",
              any(u.startswith("https://api.crossref.org") for u in urls_called))

        def fake_is_corrected_by(url, params, headers):
            if url == sources.OPENALEX_URL:
                class R:
                    def json(self):
                        return {"results": []}
                return R()
            class R:
                def json(self):
                    return {"message": {"relation": {"is-corrected-by": [{"id": "10.1/x"}]}}}
            return R()

        sources._get_with_retry = fake_is_corrected_by
        p2 = Paper(title="T", abstract="a", doi="10.1000/corrected-doi")
        sources.check_record_status(p2)
        check("is-corrected-by relation alone yields Corrected", p2.correction_note == "Corrected")

        def fake_eoc(url, params, headers):
            if url == sources.OPENALEX_URL:
                class R:
                    def json(self):
                        return {"results": []}
                return R()
            class R:
                def json(self):
                    return {"message": {"update-to": [{"type": "expression_of_concern"}]}}
            return R()

        sources._get_with_retry = fake_eoc
        p3 = Paper(title="T", abstract="a", doi="10.1000/eoc-doi")
        sources.check_record_status(p3)
        check("expression_of_concern update type yields the matching note",
              p3.correction_note == "Expression of concern")

        def fake_empty(url, params, headers):
            if url == sources.OPENALEX_URL:
                class R:
                    def json(self):
                        return {"results": []}
                return R()
            class R:
                def json(self):
                    return {"message": {}}
            return R()

        sources._get_with_retry = fake_empty
        p4 = Paper(title="T", abstract="a", doi="10.1000/empty-doi")
        sources.check_record_status(p4)
        check("no update relation leaves correction_note empty", p4.correction_note == "")

        calls = []
        sources._get_with_retry = lambda url, params, headers: calls.append(url)
        p5 = Paper(title="T", abstract="a", doi="")
        sources.check_record_status(p5)
        check("a paper with no DOI makes no calls at all", calls == [])

        def fake_fails(url, params, headers):
            raise sources.SearchFailure("boom")

        sources._get_with_retry = fake_fails
        p6 = Paper(title="T", abstract="a", doi="10.1000/fails-doi")
        sources.check_record_status(p6)  # must not raise
        check("a soft failure leaves the paper untouched",
              p6.is_retracted is False and p6.correction_note == "")
    finally:
        sources._get_with_retry = real_get
        sources._CACHE_TTL = real_ttl

    # 4. resolve_pmcid keeps its pinned error-handling shape.
    src = inspect.getsource(sources.resolve_pmcid)
    check("resolve_pmcid keeps its two named failures", src.count("except SearchFailure") == 2)
    check("resolve_pmcid keeps its two named failure log lines",
          src.count("failed for {doi}") == 2)
    check("resolve_pmcid calls the status check", "check_record_status(" in src)

    # 5. _merge_duplicate: retraction IS OR'd, unlike is_preprint.
    kept = Paper(title="Dup", abstract="a", doi="10.1000/kept-doi", is_retracted=False)
    dup = Paper(title="Dup", abstract="a", doi="10.1000/dup-doi", is_retracted=True)
    sources._merge_duplicate(kept, dup)
    check("is_retracted IS OR'd across a merge", kept.is_retracted is True)

    kept_published = Paper(title="Published Paper", abstract="Abs",
                           doi="10.1001/jamapsychiatry.2025.1317", is_preprint=False)
    dup_arxiv = Paper(title="Published Paper", abstract="Abs",
                      doi="10.48550/arXiv.1", is_preprint=True)
    sources._merge_duplicate(kept_published, dup_arxiv)
    check("is_preprint is still not OR'd across a merge when kept already has a DOI",
          kept_published.is_preprint is False)

    # 6. The writer never sees a retracted record — the "done means".
    papers = [
        Paper(title="First paper", abstract="Abstract one", doi="10.1000/p1", year=2020),
        Paper(title="Retracted paper", abstract="Abstract two", doi="10.1000/p2",
              year=2021, is_retracted=True),
        Paper(title="Third paper", abstract="Abstract three", doi="10.1000/p3", year=2022),
    ]
    curation = {
        "relevance": {1: "direct", 2: "direct", 3: "direct"},
        "most_relevant_index": 1,
        "counts": {"direct": 3, "related": 0, "tangential": 0},
    }
    context, _ = writer._writer_context("topic", papers, "", curation, kind="briefing")
    check("retracted paper's title is not in the writer context",
          "Retracted paper" not in context)
    check("SOURCE 3 numbering is not re-packed", "SOURCE 3" in context)
    check("SOURCE 2 block is gone", "SOURCE 2\n" not in context)
    check("the context names a retracted record as withheld",
          "retracted" in context.lower())

    format_only = writer._format_sources(papers, omit={2})
    check("_format_sources omits by index the same way", "Retracted paper" not in format_only)
    check("_format_sources keeps the untouched numbering", "SOURCE 3" in format_only)

    # 7. Table 1 and Methods.
    retracted_paper = Paper(title="Retracted Study", abstract="Abs", doi="10.1000/r1",
                            year=2021, venue="Some Journal", is_retracted=True)
    journal_paper = Paper(title="Journal Study", abstract="Abs", doi="10.1000/j1",
                          year=2021, venue="Another Journal")
    cited_pair = [retracted_paper, journal_paper]
    check("Table 1 HTML prints Retracted", "Retracted" in render._table_html(cited_pair, {}))
    check("Table 1 Markdown prints Retracted", "Retracted" in render._table_markdown(cited_pair, {}))

    check("_retracted_count counts screened retracted records",
          render._retracted_count(papers) == 1)

    with_retractions = render._methods_paragraphs(None, 40, 8, "topic", n_retracted=2)["search"]
    check("Methods names the retraction count", "2" in with_retractions)
    check("Methods says 'retracted'", "retracted" in with_retractions.lower())
    without_retractions = render._methods_paragraphs(None, 40, 8, "topic", n_retracted=0)["search"]
    check("Methods says nothing about retraction when there is none",
          "retracted" not in without_retractions.lower())
    for word in ("GRADE", "risk of bias", "quality", "appraisal"):
        check(f"Methods retraction sentence carries no {word!r}",
              word.lower() not in with_retractions.lower())

    # 8. The correction note reaches both renderers, and a corrected paper is
    # never omitted.
    corrected_paper = Paper(title="Corrected Study", abstract="Abs", doi="10.1000/c1",
                            year=2022, venue="Journal X", correction_note="Corrected")
    plain_paper = Paper(title="Plain Study", abstract="Abs", doi="10.1000/pl1",
                        year=2022, venue="Journal Y")
    correction_pair = [corrected_paper, plain_paper]
    article_payload = {
        "title": "A Test Review Article",
        "abstract": "Summary with references [1] and [2].",
        "keywords": ["testing"],
        "evidence_note": "Evidence note [1].",
        "featured_study": {"source_index": 1, "method": "Trial", "results": "Results"},
        "sections": [
            {"heading": "Introduction", "paragraphs": ["Intro referencing [1] and [2]."]},
            {"heading": "Conclusions", "paragraphs": ["Conclusion [1]."]},
        ],
        "key_points": ["Key point referencing [2]."],
        "glossary": [],
        "references": [1, 2],
    }
    h_out = render.render_article(article_payload, correction_pair, "test topic", None, None, None)
    md_out = render.render_markdown(article_payload, correction_pair, "test topic", None, None, None)
    check("HTML render has the correction note exactly once", h_out.count("(Corrected)") == 1)
    check("Markdown render has the correction note exactly once", md_out.count("(Corrected)") == 1)

    corrected_curation = {
        "relevance": {1: "direct", 2: "direct"},
        "most_relevant_index": 1,
        "counts": {"direct": 2, "related": 0, "tangential": 0},
    }
    corrected_context, _ = writer._writer_context(
        "topic", correction_pair, "", corrected_curation, kind="briefing")
    check("a corrected paper is not omitted from the writer context",
          "Corrected Study" in corrected_context)


def test_arxiv_rate_limit_is_honoured() -> None:
    """arXiv asks for three seconds between requests; there is no key to throttle.

    Ignoring it gets the egress IP blocked, and on the hosted backend that IP is
    shared with every other user of the deployment.
    """
    from articlegen import sources

    class FakeResp:
        text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    slept = []
    real_get, real_sleep = sources._get_with_retry, sources.time.sleep
    real_last = sources._arxiv_last_call
    try:
        sources._get_with_retry = lambda url, params, headers: FakeResp()
        sources.time.sleep = lambda s: slept.append(s)
        sources._arxiv_last_call = 0.0
        sources.search_arxiv("first")
        check("the first call in a cold process does not wait", not slept)
        sources.search_arxiv("second")
        check("a back-to-back call waits out the interval",
              len(slept) == 1 and 0 < slept[0] <= sources._ARXIV_MIN_INTERVAL)
    finally:
        sources._get_with_retry, sources.time.sleep = real_get, real_sleep
        sources._arxiv_last_call = real_last


def test_methods_names_only_sources_that_answered() -> None:
    """The Methods section must not claim a database that returned nothing.

    `_DATABASES` was a hardcoded constant, so every article stated that both
    Semantic Scholar and OpenAlex had been searched — including while one of
    them was 429ing every call. That made the claim false in every article the
    pipeline produced, in the one section that exists to state what was actually
    done. It survived the first fix as a fallback for drafts whose provenance
    carried no `databases`, which reintroduced exactly the same false claim on
    that path, so there is now no fallback at all: an unrecorded search says so.
    """
    from articlegen import sources
    from articlegen.render import render_article, render_markdown
    from articlegen.sources import DATABASE_NAMES, Paper

    papers = [Paper(title="P", abstract="a", year=2024, source="OpenAlex")]
    article = {
        "title": "T", "abstract": "A" * 200, "keywords": [], "evidence_note": "",
        "featured_study": {"source_index": 1, "why": "w", "method": "m", "results": "r"},
        "sections": [{"heading": "Introduction", "paragraphs": ["Prose [1]."]},
                     {"heading": "Conclusions", "paragraphs": ["More [1]."]}],
        "key_points": [], "glossary": [], "references": [1],
    }

    only_openalex = render_article(article, papers, "t", None, {},
                                   {"databases": ["OpenAlex"], "queries": ["q"]})
    check("names the source that answered", "OpenAlex" in only_openalex)
    check("does not name the silent source", "Semantic Scholar" not in only_openalex)

    # Three databases must read "A, B and C", not "A and B and C" — the join
    # was only ever exercised with two until Europe PMC was added.
    from articlegen.render import _join_list
    check("one name joins to itself", _join_list(["A"]) == "A")
    check("two names join with 'and'", _join_list(["A", "B"]) == "A and B")
    check("three names join as a list", _join_list(["A", "B", "C"]) == "A, B and C")

    three = render_article(article, papers, "t", None, {},
                           {"databases": ["A", "B", "C"], "queries": ["q"]})
    check("Methods lists three databases grammatically",
          "from A, B and C" in three and "A and B and C" not in three)

    both = render_article(article, papers, "t", None, {},
                          {"databases": list(DATABASE_NAMES.values()), "queries": ["q"]})
    check("names both when both answered",
          "OpenAlex" in both and "Semantic Scholar" in both)

    # Drafts written before provenance carried `databases` must still render —
    # and must name no database at all rather than guessing at a plausible pair.
    legacy = render_article(article, papers, "t", None, {}, {"queries": ["q"]})
    check("legacy drafts without `databases` still render", "Candidate records" in legacy)
    check("an unrecorded search names no database",
          "OpenAlex" not in legacy and "Semantic Scholar" not in legacy
          and "Europe PMC" not in legacy)
    check("and says so instead of guessing",
          "databases searched were not recorded" in legacy)

    legacy_md = render_markdown(article, papers, "t", None, {}, {"queries": ["q"]})
    check("markdown makes the same admission",
          "databases searched were not recorded" in legacy_md
          and "OpenAlex" not in legacy_md)

    # A source that refuses once is not retried for the remaining queries:
    # three tries with backoff is ~10s, and the limits are per-minute.
    real = (sources.search_semantic_scholar, sources.search_openalex,
            sources.search_europe_pmc, sources.search_arxiv)
    calls = {"ss": 0, "oa": 0, "ep": 0, "ax": 0}
    try:
        sources.clear_search_cache()

        def failing_ss(q, limit=15):
            calls["ss"] += 1
            raise sources.SearchFailure("HTTP 429 after 3 attempts")

        def working_oa(q, limit=15):
            calls["oa"] += 1
            return [Paper(title=f"{q}-{calls['oa']}", abstract="a", year=2024)]

        def failing_ep(q, limit=15):
            calls["ep"] += 1
            raise sources.SearchFailure("HTTP 503 after 3 attempts")

        def failing_ax(q, limit=15):
            calls["ax"] += 1
            raise sources.SearchFailure("HTTP 503 after 3 attempts")

        sources.search_semantic_scholar = failing_ss
        sources.search_openalex = working_oa
        sources.search_europe_pmc = failing_ep
        sources.search_arxiv = failing_ax
        outcomes: list[dict] = []
        got = sources.gather_evidence(["q1", "q2", "q3"], outcomes=outcomes)

        check("each failing source is tried once, not per query",
              calls["ss"] == 1 and calls["ep"] == 1 and calls["ax"] == 1)
        check("the working source is still tried every query", calls["oa"] == 3)
        check("the run still succeeds on one source", len(got) == 3)
        # Three sources fail on q1 and are skipped for q2 and q3: 3 x 2.
        check("skips are recorded, not silent",
              sum(1 for o in outcomes if "skipped" in o["error"]) == 6)
        answered = {o["source"] for o in outcomes if o["count"]}
        check("only the answering source counts as answered", answered == {"openalex"})
    finally:
        sources.clear_search_cache()
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = real


def test_citation_renumbering() -> None:
    from articlegen.render import _citation_map, _remap_citations
    from articlegen.sources import Paper

    papers = [Paper(title=f"P{i}", abstract="a", year=2000 + i) for i in range(1, 21)]
    article = {"references": [6, 7, 18, 8, 15]}
    cited, m = _citation_map(article, papers)
    check("cited papers in references order", [p.title for p in cited] == ["P6", "P7", "P18", "P8", "P15"])
    check("map renumbers source indices", m == {6: 1, 7: 2, 18: 3, 8: 4, 15: 5})
    check("remap single marker", _remap_citations("x [18] y", m) == "x [3] y")
    check("remap combined marker", _remap_citations("a [7, 8] b", m) == "a [2, 4] b")
    # The dropped marker takes its leading space with it — leaving it produced
    # "z  w" here and a stranded full stop ("…the market .") in a real article.
    check("remap drops unknown marker without leaving a gap",
          _remap_citations("z [99] w", m) == "z w")


def test_journal_citation_style() -> None:
    """Superscript markers after the punctuation, runs of 3+ collapsed to a range."""
    from articlegen.render import (
        _link_citations, _plain_citations, _shift_markers_after_punctuation,
    )

    check("marker moves after the full stop",
          _shift_markers_after_punctuation("a claim [1].") == "a claim.[1]")
    check("marker moves after a semicolon",
          _shift_markers_after_punctuation("first [2]; second") == "first;[2] second")
    check("marker hugs the preceding word",
          _shift_markers_after_punctuation("evidence [3] is thin") == "evidence[3] is thin")

    valid = set(range(1, 8))
    single = _link_citations("x.[3]", valid)
    check("single marker becomes a superscript link",
          '<sup class="cite">' in single and 'href="#ref-3"' in single)
    pair = _link_citations("x.[1, 3]", valid)
    check("pair is comma-joined, not ranged", ">1</a>,<a" in pair)
    run = _link_citations("x.[2, 3, 4]", valid)
    check("run of three collapses to a range", ">2</a>–<a" in run and ">4</a>" in run)
    check("unsorted markers are ordered", _link_citations("x.[5, 2]", valid).index(">2<")
          < _link_citations("x.[5, 2]", valid).index(">5<"))
    check("marker with no matching source is left alone",
          _link_citations("x.[99]", valid) == "x.[99]")
    check("markdown collapses runs too", _plain_citations("x.[2, 3, 4]") == "x.[2–4]")


def test_reference_formatting() -> None:
    from articlegen.render import _format_author, _reference_authors, _short_author
    from articlegen.sources import Paper

    check("author -> surname, initial", _format_author("Susanne Diekelmann") == "Diekelmann, S.")
    check("particle stays with the surname", _format_author("Hans Van Dongen") == "Van Dongen, H.")
    check("middle name becomes a second initial",
          _format_author("Hans P Van Dongen") == "Van Dongen, H. P.")
    check("mononym survives", _format_author("Aristotle") == "Aristotle")

    two = Paper(title="T", abstract="a", authors=["Susanne Diekelmann", "Jan Born"])
    many = Paper(title="T", abstract="a", authors=["Lulu Xie", "Hongyi Kang", "Qiwu Xu", "et al."])
    four = Paper(title="T", abstract="a", authors=["A One", "B Two", "C Three", "D Four"])
    check("two authors joined with an ampersand",
          _reference_authors(two) == "Diekelmann, S. & Born, J.")
    check("trailing 'et al.' collapses the list", _reference_authors(many) == "Xie, L. et al.")
    check("author line always closes with a stop",
          _reference_authors(Paper(title="T", abstract="a", authors=["Borelli"])) == "Borelli."
          and _reference_authors(Paper(title="T", abstract="a", authors=[])) == "Unknown authors.")
    check("more than three authors collapses too", _reference_authors(four) == "One, A. et al.")
    check("short form for tables", _short_author(two) == "Diekelmann & Born")
    check("short form collapses many", _short_author(many) == "Xie et al.")

    # Corporate/consortium authors pass through whole (issue #62): a digit or an
    # organisational word marks a name no initials should be invented from.
    check("consortium with a digit is not split",
          _format_author("GBD 2019 Collaborators") == "GBD 2019 Collaborators")
    check("organisational word is not split",
          _format_author("WHO Expert Committee") == "WHO Expert Committee")
    check("a person named after none of the org words still splits",
          _format_author("Sofia Network") == "Sofia Network"
          and _format_author("Sofia Networks") == "Networks, S.")
    corporate = Paper(title="T", abstract="a", authors=["GBD 2019 Collaborators"])
    check("corporate reference line survives intact",
          _reference_authors(corporate) == "GBD 2019 Collaborators.")
    check("corporate short form survives intact",
          _short_author(corporate) == "GBD 2019 Collaborators")


def test_prose_style_check() -> None:
    """The style gate must catch magazine register and pass real journal prose."""
    from articlegen.style import check_style, errors, revision_brief

    magazine = {
        "abstract": "Ever wondered what your brain does at night? It's incredible.",
        "sections": [{"heading": "The night shift", "paragraphs": [
            "The last two decades have dramatically inverted that picture.",
            "Scientists have definitively proven that sleep matters, and in order to "
            "understand why, we ran through the literature ourselves.",
        ]}],
        "key_points": ["Sleep clearly matters!"],
    }
    found = {i["rule"] for i in check_style(magazine)["issues"]}
    for rule in ("rhetorical-question", "second-person", "contraction", "booster",
                 "overclaim", "first-person", "exclamation", "wordiness"):
        check(f"style catches {rule}", rule in found)

    brief = revision_brief(check_style(magazine))
    check("revision brief locates and quotes each offence",
          "[abstract]" in brief and "Offending text:" in brief
          and "Addresses the reader directly" in brief)

    from articlegen import demo
    demo_report = check_style(demo.SAMPLE_ARTICLE)
    check("demo prose passes the style gate", not errors(demo_report))
    check("style stats are reported",
          demo_report["stats"]["sentences"] > 20
          and demo_report["stats"]["hedges_per_sentence"] > 0)

    # The reviewing frame journals themselves use is allowed; other first person is not.
    allowed = {"sections": [{"heading": "H", "paragraphs": [
        "Here we review the evidence for clearance during sleep."]}]}
    denied = {"sections": [{"heading": "H", "paragraphs": [
        "Our results show that clearance increases during sleep."]}]}
    check("'here we review' is permitted",
          not any(i["rule"] == "first-person" for i in check_style(allowed)["issues"]))
    check("'our results' is not",
          any(i["rule"] == "first-person" for i in check_style(denied)["issues"]))

    long_sentence = {"sections": [{"heading": "H", "paragraphs": [
        "This sentence " + "goes on and on " * 15 + "without stopping."]}]}
    check("long sentences are flagged",
          any(i["rule"] == "long-sentence" for i in check_style(long_sentence)["issues"]))

    # Density rules need enough prose to mean anything.
    thin = {"sections": [{"heading": "H", "paragraphs": ["A flat claim about a thing."]}]}
    check("density rules stay quiet on short drafts",
          not any(i["rule"] == "under-hedged" for i in check_style(thin)["issues"]))
    # Flat, unhedged assertion at the length where density figures start to mean
    # something (the gate needs both a sentence count and a word count).
    flat = (
        "The compound binds the receptor and increases the rate of clearance in "
        "every model tested. The effect is dose dependent across the full range "
        "studied. Uptake rises with temperature in each preparation examined. "
        "The pathway drives waste removal from the interstitial space. Levels of "
        "the metabolite fall after a single treatment. The response is linear "
        "throughout the measured interval. Binding saturates at the highest dose "
        "administered. Efflux doubles overnight in every animal studied. The "
        "volume of the interstitial space expands during rest. Transport slows "
        "with age in all cohorts examined. Flow returns to baseline by morning "
        "in each experiment. The mechanism accounts for the whole of the observed "
        "difference between the groups."
    )
    unhedged = {"sections": [{"heading": "H", "paragraphs": [flat] * 3}]}
    report = check_style(unhedged)
    check("under-hedged prose is flagged",
          any(i["rule"] == "under-hedged" for i in report["issues"]))


def test_rules_do_not_reject_real_journal_prose() -> None:
    """The register rules must not fire on genuinely published writing.

    Every rule here is a guess about what journal prose looks like, and the guesses
    have been wrong in ways no synthetic fixture could reveal. Checked against real
    abstracts, an earlier version of the first-person rule flagged five of eight,
    because its allowlist required "we" immediately followed by an approved verb —
    so "we also review", "we searched", "we aimed to" all read as a human author
    intruding. Three more false positives were pure clinical notation: "Axis I",
    "I2 = 70.6%" (the meta-analysis heterogeneity statistic) and "US $16.3 million"
    all matched a first-person pronoun.

    `tests/real_abstracts.json` is a frozen corpus of real abstracts, stored rather
    than fetched so the suite stays offline and deterministic. Two entries carry a
    documented `expect_register_errors` — one rhetorical "we", and one genuine
    author-voice "we hope" that the house style bans on purpose, which doubles as a
    positive control that the rule still bites.
    """
    import json

    from articlegen.style import check_style, errors

    register = {"first-person", "second-person", "contraction", "booster",
                "overclaim", "rhetorical-question", "exclamation"}
    path = os.path.join(os.path.dirname(__file__), "real_abstracts.json")
    corpus = json.load(open(path, encoding="utf-8"))
    check("corpus is a usable size", len(corpus) >= 12)

    unexpected = []
    for entry in corpus:
        article = {"sections": [{"heading": "Introduction",
                                 "paragraphs": [entry["abstract"]]}]}
        fired = sorted({i["rule"] for i in errors(check_style(article))} & register)
        if fired != entry["expect_register_errors"]:
            unexpected.append((entry["title"][:50], fired, entry["expect_register_errors"]))
    if unexpected:
        for title, fired, expected in unexpected:
            print(f"      {title}: fired {fired}, expected {expected}")
    check(f"register rules match expectations on {len(corpus)} real abstracts",
          not unexpected)

    check("every documented exception carries a reason",
          all(e.get("note") for e in corpus if e["expect_register_errors"]))

    # The corpus must not be so permissive that the rules stop working. These are
    # the things the house style genuinely bans.
    for text, expected in [
        ("We believe this is the most important question.", "first-person"),
        ("Our findings show a clear benefit.", "first-person"),
        ("You should note the risk before prescribing.", "second-person"),
        ("This clearly proves the point.", "booster"),
        ("It doesn't replicate.", "contraction"),
    ]:
        fired = {i["rule"] for i in errors(check_style(
            {"sections": [{"heading": "Introduction", "paragraphs": [text]}]}))}
        check(f"still catches {expected}: {text[:34]!r}", expected in fired)

    # Clinical notation that must never read as a first-person pronoun.
    for text in ("Axis I comorbidity was recorded in most participants.",
                 "Heterogeneity was substantial (I2 = 70.6%).",
                 "Costs reached US $16.3 million by 2030.",
                 "A Phase I trial enrolled 40 participants."):
        fired = {i["rule"] for i in errors(check_style(
            {"sections": [{"heading": "Introduction", "paragraphs": [text]}]}))}
        check(f"not first person: {text[:38]!r}", "first-person" not in fired)


def test_unverified_figures_are_marked_inline() -> None:
    """The flag has to travel with the number, not sit 90 lines below it.

    A draft's first Key point stated an odds ratio of 0.66 (95% CI 0.43-0.99);
    the fact that those figures could not be located appeared about ninety
    lines later, with no way to tell which figure it meant — while the number
    carried a superscript citation, the strongest "this came from that paper"
    signal on the page. Someone copies the Key points into a team email, the
    numbers travel and every qualification stays behind (issue #92).
    """
    from articlegen import render
    from articlegen.render import render_article, render_markdown
    from articlegen.sources import Paper

    papers = [Paper(title="S1", abstract="a", year=2020, authors=["Ann Ab"]),
              Paper(title="S2", abstract="b", year=2021, authors=["Bo Cd"])]
    article = {
        "title": "T", "abstract": "The odds ratio was 0.66 [1].",
        "sections": [
            {"heading": "Introduction",
             "paragraphs": ["An effect of 0.66 was seen [1]. Also 112% of baseline [1]."]},
            {"heading": "Conclusions", "paragraphs": ["Unresolved [1]."]}],
        "key_points": ["Odds ratio 0.66 favours the intervention [1].",
                       "A separate trial found 1.24 [2]."],
        "featured_study": {"source_index": 1, "method": "m", "results": "RR 1.24 overall"},
        "references": [1, 2], "keywords": [],
    }
    verification = {"unverified": ["0.66"], "misattributed": ["1.24"], "total": 6}
    provenance = {"queries": ["q"], "databases": ["Europe PMC"], "model": "m"}

    h = render_article(article, papers, "topic", None, verification, provenance)
    md = render_markdown(article, papers, "topic", None, verification, provenance)

    # Every quotable unit: abstract, body prose, key points, Box 1.
    check("html flags the figure inside the key points",
          '0.66<sup class="flag"' in h[h.index('<aside class="key-points"'):])
    check("html flags the figure inside Box 1",
          '1.24<sup class="flag"' in h[h.index('<aside class="display box"'):])
    check("html flags the abstract's figure",
          '0.66<sup class="flag"' in h[:h.index("<h2>Introduction")])
    check("markdown flags key points and Box 1",
          "0.66†" in md and "1.24‡" in md and "RR 1.24‡" in md)

    # Two categories, two marks, and the mark links to the paragraph that
    # explains it.
    check("unverified and misattributed are distinguishable",
          '0.66<sup class="flag" title="This figure could not be located' in h
          and '1.24<sup class="flag" title="This figure was found only' in h)
    check("the mark links to the Limitations paragraph",
          'href="#limitations"' in h and 'id="limitations"' in h)
    check("and Limitations names the mark",
          "marked †" in h and "marked ‡" in h)

    # A number that was never flagged must not pick up a mark, and a flagged
    # figure must not be found inside a longer number or inside a citation
    # marker — "12" would otherwise match inside "[12]".
    check("unflagged numbers are untouched",
          "112%" in h and '112%<sup class="flag"' not in h)
    cited_only = dict(article, key_points=["Twelve sources [12]."])
    h12 = render_article(cited_only, papers, "topic", None,
                         {"unverified": ["12"], "total": 1}, provenance)
    check("a flagged figure is not found inside a citation marker",
          '12<sup class="flag"' not in h12)

    # The grounding line under Key points, so the copied block carries it.
    note_at = h.index('class="key-points-note"')
    check("the grounding note sits above the points", note_at < h.index("<ul>", note_at))
    check("and names both marks", "† marks" in h and "‡ marks" in h)
    check("markdown carries the note too",
          md.index("Every figure in this review was checked") < md.index("- Odds ratio"))

    # Derived, never assumed — the same rule as every other provenance
    # statement. No verification means no claim that anything was checked.
    check("a clean check says so",
          "Every figure in this review was located in"
          in render._grounding_note({"unverified": [], "misattributed": [], "total": 4}, False))
    check("no verification means no grounding claim",
          render._grounding_note(None, False) == ""
          and render._grounding_note({"unverified": [], "total": 0}, False) == "")
    check("and the wording follows what was read",
          "open-access full text" in render._grounding_note({"unverified": [], "total": 2}, True))


def test_clinical_directives_are_an_error() -> None:
    """Reporting what studies found is the job; instructing a clinician is not.

    A shipped draft carried a titration protocol — "starting with a low-dose
    exposure of 15 minutes per day... titrated upward by 15 minutes each week" —
    for a population the same article said had zero studies (issue #102). The
    footer disclaimer does no work against that.

    The pairs below are the rule's specification. Each "reports" line is real
    review prose that must stay legal; each "instructs" line is the same
    clinical content turned into advice. A rule that cannot tell them apart is
    the wrong rule, not a strict one.
    """
    import json

    from articlegen import render
    from articlegen.style import check_style, errors

    def fired(text: str) -> set:
        return {i["rule"] for i in errors(check_style(
            {"sections": [{"heading": "Introduction", "paragraphs": [text]}]}))}

    instructs = [
        "Treatment should be initiated at a low-dose exposure of 15 minutes per "
        "day, titrated upward by 15 minutes each week.",
        "Clinicians should prescribe the lowest effective dose.",
        "Monitor serum levels every four weeks.",
        "The starting dose is 25 mg at night.",
        "Ensuring therapeutic antipsychotic levels is an absolute prerequisite.",
        "Exposure must be titrated according to response.",
        "Patients should be referred for specialist review.",
    ]
    for text in instructs:
        check(f"instruction is an error: {text[:44]!r}",
              "clinical-directive" in fired(text))

    # The negative controls matter more than the positives here. Every one of
    # these is ordinary, correct review prose about the same clinical subject
    # matter, and an over-eager rule kills the article's ability to report.
    reports = [
        "Participants received 10,000 lux for 30 minutes each morning.",
        "The trial titrated the dose to response over eight weeks.",
        "Dosing varied across trials, from 25 mg to 100 mg daily.",
        "Screening was performed at baseline and at 12 weeks.",
        "Treatment of the underlying condition was reported in three studies.",
        "Monitoring of adherence was inconsistent across the cohort studies.",
        "Screen time was associated with poorer sleep in two cohorts.",
        # Idioms that borrow a clinical verb for something else.
        "These findings should be treated as provisional.",
        "The syndrome should be referred to as post-exertional malaise.",
        # Trial-arm names are reports. A Luna briefing of RADAR was branded a
        # working draft because `_PRESCRIPTION_RE` treated "dose reduction" as
        # advice. Instructional dose changes still fail via starting/target dose
        # or a modal + clinical act, tested above.
        "A separate recurrent-psychosis trial found no social-functioning "
        "advantage from dose reduction over maintenance at two years.",
        "The RADAR trial assigned adults to gradual antipsychotic dose "
        "reduction or maintenance.",
        # A recommendation for research is not a recommendation for care, and
        # it is standard in a Conclusions section.
        "Future trials should monitor adherence more closely.",
        "Further research should assess whether the effect persists.",
    ]
    for text in reports:
        check(f"report is not an error: {text[:44]!r}",
              "clinical-directive" not in fired(text))

    # The rule against the corpus. One abstract fires: a Lancet trial report
    # concluding "WBRT and stereotactic radiosurgery should... be standard
    # treatment". Those authors ran the trial and may say that; articlegen is a
    # synthesis and may only report that they said it — the same argument as
    # the investigator-voice finding in docs/journal-style.md §12.
    corpus = json.load(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_corpus.json"),
        encoding="utf-8"))
    hits = [e for e in corpus if "clinical-directive" in fired(e["abstract"])]
    check("exactly one corpus abstract is a clinical directive", len(hits) == 1)
    check("and it is the one recommending a standard treatment",
          hits and "standard treatment" in hits[0]["abstract"])

    # demo.SAMPLE_ARTICLE must always pass every rule (CLAUDE.md invariant).
    from articlegen import demo
    check("the demo article carries no directive",
          "clinical-directive" not in {i["rule"] for i in errors(check_style(demo.SAMPLE_ARTICLE))})

    # The writer is told, not only checked. A rule the prompt never mentions is
    # a revision round trip on every draft.
    from articlegen import writer
    check("the writer prompt carries the prohibition",
          "NEVER GIVE CLINICAL ADVICE" in writer._WRITER_SYSTEM
          and "not even a hedged one" in writer._WRITER_SYSTEM)
    check("and the briefing prompt carries it too",
          "NEVER GIVE CLINICAL ADVICE" in writer._BRIEFING_SYSTEM
          and "not even a hedged one" in writer._BRIEFING_SYSTEM)
    check("and the reader is told when it survived",
          "clinical-directive" in render._STYLE_FAILURE_WORDING)


def test_full_text_dependencies_fail_loudly_enough_to_diagnose() -> None:
    """Soft failure is right for the reader and useless for the operator.

    The full-text path depends on four keyless services — Europe PMC search,
    Europe PMC fetch, DOI resolution and Unpaywall — each rate-limited against
    Render's shared egress IP, each failing soft. When Unpaywall blocks the
    contact address, every article silently drops to abstracts-only, Methods
    correctly reports fewer full texts, and nothing anywhere points at the
    cause (issue #104).
    """
    import inspect

    from articlegen import pipeline as articlegen_pipeline
    from articlegen import sources, web

    # 1. The swallowed failures now say what happened, with the DOI.
    src = inspect.getsource(sources.resolve_pmcid)
    check("resolve_pmcid takes a logger", "log" in inspect.signature(sources.resolve_pmcid).parameters)
    check("no failure is swallowed in silence",
          src.count("except SearchFailure") == 2 and "except SearchFailure:\n            pass" not in src)
    for service in ("europe_pmc DOI lookup failed", "unpaywall lookup failed"):
        check(f"the log names the service: {service!r}", service in src)
    check("and names the DOI", src.count("failed for {doi}") == 2)

    # The pipeline has to pass the logger, or none of the above is reachable.
    check("the pipeline passes its logger through",
          "resolve_pmcid(paper, log=log)" in inspect.getsource(
              articlegen_pipeline._retrieve_full_texts)
          and "_retrieve_full_texts(" in inspect.getsource(
              articlegen_pipeline.generate_draft))

    # 2. /api/diag probes Unpaywall, which the three search probes never touch.
    diag = inspect.getsource(web.ArticleGenHandler._handle_diag)
    check("/api/diag probes the full-text dependency too", "probe_unpaywall()" in diag)
    check("and reports it under its own key", '"full_text": unpaywall' in diag)

    # The probe reports rather than raises: a diagnostic endpoint that 500s
    # when the thing it diagnoses is down answers nothing.
    real = sources._get_with_retry
    try:
        def refuse(*a, **kw):
            raise sources.SearchFailure("HTTP 403 (blocked contact address)")
        sources._get_with_retry = refuse
        out = sources.probe_unpaywall()
        check("a blocked probe reports instead of raising",
              out["source"] == "unpaywall" and "403" in out["error"])

        class FakeResp:
            @staticmethod
            def json():
                return {"doi": "x"}          # no is_oa field

        sources._get_with_retry = lambda *a, **kw: FakeResp()
        out = sources.probe_unpaywall()
        check("a changed response shape is caught, not read as a miss",
              "unexpected response shape" in out["error"])

        class ClosedResp:
            @staticmethod
            def json():
                return {"is_oa": False}

        sources._get_with_retry = lambda *a, **kw: ClosedResp()
        out = sources.probe_unpaywall()
        check("a known open-access DOI reported closed is an error",
              "reports this known open-access DOI as closed" in out["error"])

        class OkResp:
            @staticmethod
            def json():
                return {"is_oa": True, "best_oa_location": {"url": "http://x"}}

        sources._get_with_retry = lambda *a, **kw: OkResp()
        out = sources.probe_unpaywall()
        check("a healthy Unpaywall reports no error",
              out["error"] == "" and out["is_oa"] is True)
    finally:
        sources._get_with_retry = real


def test_first_visit_does_not_dead_end() -> None:
    """The longest possible wait must not end in "you needed a key".

    A stranger opened the site, typed a theme, tapped, sat through the ~50s
    Render cold start, and used to get the server's 400 as a raw browser
    alert() telling them to open a gear icon (issue #95). Generating is host
    paid now, so the page must not ask for a key and must not fail as an
    alert.
    """
    import inspect
    import re as _re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "index.html"), encoding="utf-8").read()

    # 1. Neither entry point waits on a visitor key.
    check("the page does not gate on a visitor key", "requireKey()" not in page)
    check("ideas still fetches", "apiUrl('/api/ideas')" in page)
    check("draft still fetches", "apiUrl('/api/draft')" in page)

    # 2. The backend is woken on load, and a slow request says why it is slow.
    check("the backend is warmed on page load",
          "warmBackend()" in page and "/api/health" in page)
    check("a slow request explains itself while it is still running",
          "whileWaking(" in page and "Waking the free server" in page)

    # 3. No alert() anywhere; failures land in the page with something to press.
    code = "\n".join(_re.findall(r"<script>(.*?)</script>", page, _re.S))
    code = _re.sub(r"//[^\n]*", "", code)
    check("no browser alert() survives", "alert(" not in code)
    check("failures land in the progress card",
          'id="progressError"' in page and "showProgressError(" in page)
    check("and offer a retry that repeats the same action",
          "retryLastAction()" in page and "lastAction = function ()" in page)
    check("a long dropped request does not claim the backend is down",
          "Search was still running" in page)

    # 4. No setup card, no settings, no version badge.
    check("there is no first-run key card", 'id="setupCard"' not in page)
    check("there is no settings panel", 'id="settingsModal"' not in page)
    check("the version badge is gone", "SyncFix" not in page and "v2.3" not in page)

    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    check("the landing view does not link the drafts index",
          'href="drafts/"' not in page)
    check("and the topic input is above the visitor gallery",
          page.index('id="themeInput"') < page.index('id="publicBand"'))
    check("README says no key is needed",
          "No key and no account needed" in readme)
    check("README says the public site generates for free",
          "free on the public site" in readme)

    # 5. The timeline covers the whole plausible run, not just the first 30s.
    marks = sorted(int(m) for m in _re.findall(r"\}, (\d{4,6})\)", page))
    check("progress messages continue past 30 seconds",
          any(m >= 55000 for m in marks) and any(m >= 80000 for m in marks))

    # Server side: an unexpected fault gets a sentence, not a stack trace. A
    # visitor was once shown 500 characters of raw JSON on their phone.
    from articlegen.web import ArticleGenHandler

    for fn in (ArticleGenHandler._handle_ideas, ArticleGenHandler._handle_draft):
        # Only the catch-all matters. `NoPapersFound` carries a message written
        # for the visitor and is deliberately passed through as it is.
        generic = inspect.getsource(fn)
        generic = generic[generic.index("except Exception"):]
        check(f"{fn.__name__} does not return the raw exception",
              "_unexpected(" in generic and "str(exc)" not in generic)

    unexpected = inspect.getsource(ArticleGenHandler._unexpected)
    check("the raw detail still reaches the log", "_log_stage(" in unexpected)
    check("but an actionable message is passed through", "_ACTIONABLE" in unexpected)


def test_the_landing_page_leads_with_the_input() -> None:
    """The topic input is the first thing on the page, above the browse links.

    #152 and #111 put finished reviews above the key prompt, on the theory that
    a stranger should see output before being asked for anything. Generating is
    free on the hosted backend now, so the ask is the input, not a key — the
    input leads and the shared-visitor list follows it. The drafts/ index is
    the local CLI review surface, not a public browse link.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "index.html"), encoding="utf-8").read()

    check("the topic input comes before the shared-visitor list",
          page.index('id="themeInput"') < page.index('id="publicBand"'))
    check("the published-briefings band is gone",
          'href="drafts/"' not in page and "All published briefings" not in page)
    check("the drafts index still exists on disk",
          os.path.exists(os.path.join(root, "drafts", "index.html")))
    wf = open(os.path.join(root, ".github", "workflows", "deploy-pages.yml"), encoding="utf-8").read()
    check("Pages deploy excludes drafts/",
          "exclude 'drafts'" in wf and "path: '_site'" in wf)
    check("the removed per-draft cards are not left behind",
          "Recent evidence briefings" not in page)
    check("the head frames the output as a briefing, not an article generator",
          "Research-Grounded Articles on Mobile" not in page)
    check("the product name is Evidence brief",
          "<title>Evidence brief" in page
          and ">Evidence brief</a>" in page)
    check("ArticleGen is gone from the page", "ArticleGen" not in page)
    check("the library is briefings, not articles",
          "Your briefings" in page and "Your articles" not in page)
    check("the description names the statistic check",
          "Every statistic is checked against the cited papers" in page)
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    check("README is titled Evidence brief", readme.startswith("# Evidence brief\n"))
    check("README lead is the product description",
          "Sourced evidence briefings from a question. Every statistic is checked against"
          in readme)
    check("README live site is the evidence-brief Pages URL",
          "https://bartholomewtj.github.io/evidence-brief/" in readme
          and "github.io/article-generator" not in readme)
    health = open(os.path.join(root, ".github", "workflows", "health.yml"),
                  encoding="utf-8").read()
    check("health check curls the evidence-brief Pages URL",
          "https://bartholomewtj.github.io/evidence-brief/" in health
          and "github.io/article-generator" not in health)
    check("landing page uses the general recipe",
          "recipe: general" in page and "#6ec4bb" in page)
    check("invented sky accent is gone",
          "#38bdf8" not in page)


def test_the_web_app_does_not_take_a_visitor_key() -> None:
    """The page has no Settings, no key box, and does not send a key.

    A remembered OpenRouter key on bartholomewtj.github.io was readable by
    every other Pages site on that origin (issue #113). Removing the
    component is the fix. The backend must also ignore a crafted `key` in
    the payload, or the UI removal is only cosmetic.
    """
    import inspect

    from articlegen import web

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()

    check("no settings modal", 'id="settingsModal"' not in page)
    check("no API key input", 'id="apiKeyInput"' not in page)
    check("no remember-key checkbox", 'id="rememberKey"' not in page)
    check("no Settings nav item", ">Settings</button>" not in page)
    check("ideas request does not send a key or model",
          "JSON.stringify({ theme, guidance: styleGuidance(guidance), n: 6 })" in page)
    check("draft request does not send a key or model",
          "style: styleGuidance(style),\n            search_terms: terms" in page)
    check("the page does not store a visitor key",
          "function saveApiKey" not in page and "function activeKey" not in page)
    check("the page clears leftover keys",
          "articlegen_key_openrouter" in page
          and "removeItem" in page[page.index("purgeLegacyGistCredentials"):
                                   page.index("purgeLegacyGistCredentials") + 500])
    check("README does not tell visitors to paste a key",
          "Where your key is kept" not in readme
          and "Settings still accepts" not in readme)

    cred = inspect.getsource(web._credentials)
    check("the server ignores a payload key",
          'payload.get("key")' not in cred)


def test_the_front_end_has_one_article_list() -> None:
    """One library, one storage key, one renderer.

    There were two of each: a "Draft Review Queue" over
    articlegen_local_drafts and a "Published Library" over
    articlegen_published_library. Same articles, duplicated search / delete /
    clear-all / open, and saving to the second stored a second full copy of the
    article HTML — ~65KB against a ~5MB quota, written with a bare setItem that
    silently lost the article when it threw.
    """
    page = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"),
        encoding="utf-8",
    ).read()

    check("only one personal library view", page.count('id="libraryView"') == 1
          and 'id="publishedView"' not in page)
    check("only one personal search box",
          page.count('id="librarySearchInput"') == 1
          and 'id="publishedSearchInput"' not in page)
    check("the visitor gallery is a separate view, not a second library",
          'id="publicListView"' in page
          and 'id="publicPreview"' in page)

    # The legacy keys may still be *touched* only inside the one-time migration,
    # which folds them in and deletes them. A localStorage call on one of them
    # anywhere else means a second store came back. (Prose mentions are fine —
    # match on the call, not the name.)
    import re as _re

    migration = page[page.index("function migrateLegacyLibraries"):]
    migration = migration[:migration.index("\n  }\n")]
    for legacy in ("articlegen_local_drafts", "articlegen_published_library"):
        calls = _re.findall(rf"localStorage\.\w+\(\s*'{legacy}'", page)
        in_migration = _re.findall(rf"localStorage\.\w+\(\s*'{legacy}'", migration)
        check(f"{legacy} is only touched by the migration",
              len(calls) == len(in_migration) and in_migration)
        check(f"and the migration removes {legacy}",
              f"localStorage.removeItem('{legacy}')" in migration)

    # A bare setItem for the library is the bug this replaced: on
    # QuotaExceededError it threw away the article the user just waited for.
    body = page[page.index("const LIBRARY_KEY"):]
    check("library writes go through writeLibrary, which handles a full quota",
          "function writeLibrary" in body and "catch (e)" in body
          and f"localStorage.setItem(LIBRARY_KEY" in body)
    check("nothing else writes the library key directly",
          body.count("localStorage.setItem(LIBRARY_KEY") == 1)


def test_article_in_the_web_app_cannot_run_scripts() -> None:
    """The reader iframe is sandboxed, and what goes into it carries no script.

    The iframe used to run same-origin with no sandbox, and `#read=` / `#p=`
    links mean the HTML in it is not merely model-written but attacker-choosable:
    one localStorage.getItem reaches the visitor's saved articles. Two halves fix
    it and both have to hold — the sandbox attribute, and an article rendered
    without the scripted toolbar so the sandbox costs nothing (issue #100).
    """
    import inspect
    from articlegen import demo, render, web

    standalone = render.render_article(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, "t")
    embedded = render.render_article(
        demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, "t", standalone=False)

    check("the embedded article has no script tag", "<script" not in embedded)
    check("no inline event handlers either", "onclick" not in embedded)
    check("and never touches localStorage", "localStorage" not in embedded)
    check("the toolbar goes with them", 'class="toolbar"' not in embedded)
    check("but the article itself is all still there",
          "<h1>" in embedded and "References" in embedded
          and "Methods" in embedded and "Table 1" in embedded)

    # A file written to drafts/ is opened on its own, so it keeps all three.
    check("a standalone draft still carries its toolbar",
          'class="toolbar"' in standalone and "<script" in standalone)

    page = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"),
        encoding="utf-8",
    ).read()
    frame = page[page.index('<iframe id="articleIframe"'):]
    frame = frame[:frame.index(">") + 1]
    check("the reader iframe is sandboxed", "sandbox=" in frame)
    # allow-scripts would hand the whole thing back. allow-same-origin on its own
    # does not grant script execution; it is what keeps contentDocument reachable
    # for the theme sync, edit mode and save-to-library.
    check("and never with allow-scripts", "allow-scripts" not in frame)
    check("but keeps allow-same-origin", "allow-same-origin" in frame)

    src = inspect.getsource(web.ArticleGenHandler._handle_draft)
    check("the API returns the script-free copy", "standalone=False" in src)


def test_desktop_article_uses_the_full_window() -> None:
    """On a desktop the article is the remaining window, not a boxed column.

    A clamp to 52rem plus a 1120px iframe card left a phone-width page sitting
    in the middle of a wide screen. Newly rendered articles drop the column
    cap; the web app's reader, above 1100px, drops the card and fills the
    space beside the sidebar.
    """
    from articlegen import demo, render

    html = render.render_article(demo.SAMPLE_BRIEFING, demo.SAMPLE_PAPERS, "t")
    check("article CSS does not cap the column at 52rem",
          "clamp(40rem, 72vw, 52rem)" not in html)
    check("article main has no boxed max-width",
          "max-width: 40rem" not in html and "max-width: 52rem" not in html)
    check("article main uses the window", "max-width: none" in html)

    page = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"),
        encoding="utf-8",
    ).read()
    check("the reader is not a 1120px card",
          "min(92vw, 1120px)" not in page and "min(88vw, 1040px)" not in page)
    check("desktop reader takes the remaining window",
          "main.container:has(#readerView.active)" in page
          and "max-width: none" in page)
    check("desktop reader drops the iframe card chrome",
          "#readerView.active .reader-frame-wrapper" in page
          and "border: none" in page)
    check("the build stamp is hidden while reading",
          "body:has(#readerView.active) .build-stamp" in page)


def test_browser_back_from_an_article_returns_home() -> None:
    """Opening an article writes a hash; Back used to leave the reader on screen.

    The page is one HTML file with views toggled in JS. Opening an article sets
    #read= / #g= / #p=, which is a history entry, but nothing listened for
    hashchange — so the browser Back button cleared the hash and the article
    stayed on screen. Home next to the wordmark is the in-page equivalent.
    """
    page = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"),
        encoding="utf-8",
    ).read()

    home = page[page.index("function goHome("):]
    home = home[:home.index("\n  }\n")]
    check("goHome shows the landing view", "showView('themeView')" in home)
    check("and strips the article hash without adding history",
          "history.replaceState" in home)

    check("hashchange sends an empty hash back to the landing view",
          "addEventListener('hashchange'" in page
          and "showView('themeView')" in page[page.index("addEventListener('hashchange'"):])
    check("opening an article goes through setArticleHash",
          "function setArticleHash(" in page
          and "window.location.hash = hash" not in page)

    check("a home control sits next to the wordmark",
          'class="brand"' in page and 'class="home-btn"' in page
          and 'onclick="goHome()"' in page)
    check("both the phone header and the desktop sidebar have it",
          page.count('class="brand"') == 2
          and page.count('class="home-btn"') == 2)


def test_health_reports_which_build_is_running() -> None:
    """The deployment must be able to say what it is running, and from where.

    `/api/diag` exists because a deployment's behaviour is undiagnosable from
    outside; the same was true of its *identity*. "Is the running backend the
    code I merged?" and "which branch does this host deploy from?" both needed a
    dashboard login, which is what made renaming the default branch risky — the
    deploy could stop silently and nothing observable would say so (#47).
    """
    from articlegen import web

    for var in ("RENDER_GIT_COMMIT", "RENDER_GIT_BRANCH"):
        os.environ.pop(var, None)
    check("no build keys off Render", web._build_info() == {})

    os.environ["RENDER_GIT_COMMIT"] = "0123456789abcdef0123456789abcdef01234567"
    os.environ["RENDER_GIT_BRANCH"] = "main"
    info = web._build_info()
    check("the commit is reported short", info.get("commit") == "0123456")
    check("the deploying branch is reported", info.get("branch") == "main")

    os.environ["RENDER_GIT_BRANCH"] = ""
    check("an empty value is omitted, not reported blank",
          "branch" not in web._build_info())
    for var in ("RENDER_GIT_COMMIT", "RENDER_GIT_BRANCH"):
        os.environ.pop(var, None)

    import inspect
    src = inspect.getsource(web.ArticleGenHandler.do_GET)
    check("health carries it", "_build_info()" in src)


def test_openalex_reaches_for_recent_work_as_well() -> None:
    """The evidence pool skewed old, and ranking alone could not fix it.

    OpenAlex's relevance ordering returns old, heavily-cited work: measured over
    three topics the median year of a 20-paper page was 2006-2016, with 13-18 of
    20 published before 2015 (issue #38). Ranking only reorders what it is given,
    and a bigger page makes it worse — one page of 50 returned *more* pre-2015
    work, because it goes deeper into the same ordering.

    So the source now issues a second, date-filtered query over the same terms
    and merges. Live, that moved the median to 2018-2020 and tripled the count of
    2020-or-later papers while leaving the pre-2015 count untouched — recent work
    reaches the pool without a cutoff excluding foundational work.
    """
    from articlegen import sources
    from articlegen.sources import Paper

    calls = []

    def fake_page(query, limit, from_year=None):
        calls.append({"query": query, "limit": limit, "from_year": from_year})
        if from_year:
            return [Paper(title="Recent study", abstract="a", year=from_year + 3,
                          doi="10.1/recent"),
                    Paper(title="Shared study", abstract="a", year=from_year + 1,
                          doi="10.1/shared")]
        return [Paper(title="Old classic", abstract="a", year=1998, doi="10.1/old"),
                Paper(title="Shared study", abstract="a", year=from_year or 2020,
                      doi="10.1/shared")]

    real = sources._openalex_page
    try:
        sources._openalex_page = fake_page
        papers = sources.search_openalex("sleep", limit=10)
    finally:
        sources._openalex_page = real

    check("both a plain and a dated query are issued", len(calls) == 2)
    check("the plain query carries no date filter", calls[0]["from_year"] is None)
    check("the companion query does", isinstance(calls[1]["from_year"], int))
    import datetime as _dt
    check("and reaches back a sensible window",
          0 < _dt.date.today().year - calls[1]["from_year"] <= 15)
    check("both use the same search terms", calls[0]["query"] == calls[1]["query"])

    titles = [p.title for p in papers]
    check("the old classic is kept", "Old classic" in titles)
    check("recent work is added", "Recent study" in titles)
    check("the overlap is deduplicated", titles.count("Shared study") == 1)

    # If the companion query refuses, the relevance results must still stand.
    def half_failing(query, limit, from_year=None):
        if from_year:
            raise sources.SearchFailure("HTTP 429")
        return [Paper(title="Only result", abstract="a", year=2001)]

    attempts = []

    def half_failing_counted(query, limit, from_year=None):
        attempts.append(from_year)
        if from_year:
            raise sources.SearchFailure("HTTP 429")
        return [Paper(title="Only result", abstract="a", year=2001)]

    try:
        sources._recency_query_refused = False
        sources._openalex_page = half_failing_counted
        survived = sources.search_openalex("sleep", limit=10)
        # A second query in the same run must not re-attempt the dead one: the
        # limits are per-minute, so retrying costs three tries and ~10s each.
        sources.search_openalex("sleep again", limit=10)
    finally:
        sources._openalex_page = real
        sources._recency_query_refused = False
    check("a refused recency query does not lose the plain results",
          [p.title for p in survived] == ["Only result"])
    check("and is not retried for the rest of the run",
          sum(1 for a in attempts if a) == 1)


def test_claude_cli_provider() -> None:
    """Drafting on a Claude subscription, which issues no API key.

    The invariant worth pinning is that this provider stays opt-in. It answers
    as whoever is signed into the CLI on the machine, so auto-selecting it on a
    threaded server would answer every visitor's request from the host's own
    seat — and no deployment host has a `claude` binary anyway.
    """
    import json as _json

    from articlegen import llm, web

    saved = {v: os.environ.get(v) for v in
             ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER")}
    try:
        for v in saved:
            os.environ.pop(v, None)

        check("cli: prefix routes to the CLI, stripped to the alias",
              llm.resolve_provider("cli:sonnet") == ("claude-cli", "sonnet"))
        check("a bare cli: takes the default model",
              llm.resolve_provider("cli:") == ("claude-cli", llm.CLAUDE_CLI_DEFAULT_MODEL))
        check("ARTICLEGEN_PROVIDER can select it",
              (os.environ.update({"ARTICLEGEN_PROVIDER": "claude-cli"}) or
               llm.resolve_provider()) == ("claude-cli", llm.CLAUDE_CLI_DEFAULT_MODEL))
        os.environ.pop("ARTICLEGEN_PROVIDER")

        # The point of the whole guard: nothing about the environment should
        # ever land a caller here by accident.
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-x"
        check("a present Anthropic key does not select the CLI",
              llm.resolve_provider()[0] == "anthropic")
        os.environ.pop("ANTHROPIC_API_KEY")
        check("no keys at all still falls back to OpenRouter, not the CLI",
              llm.resolve_provider()[0] == "openrouter")
        check("an sk-ant- key routes to the API, never the subscription",
              llm.resolve_provider(api_key="sk-ant-x")[0] == "anthropic")

        check("the CLI model is not offered to the web app",
              llm.CLAUDE_CLI_DEFAULT_MODEL not in web.ALLOWED_MODELS
              and not any(m.startswith(llm.CLAUDE_CLI_PREFIX) for m in web.ALLOWED_MODELS))
        check("the web app drops a cli: model rather than honouring it",
              web._requested_model({"model": "cli:opus"}) is None)

        # -- the subprocess contract -------------------------------------
        captured = {}

        class FakeProc:
            returncode = 0
            stderr = ""
            stdout = _json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "duration_ms": 1200, "usage": {"input_tokens": 3, "output_tokens": 9},
                "result": '```json\n{"ok": true}\n```',
            })

        def fake_run(args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            return FakeProc()

        import subprocess
        real_run, real_which = subprocess.run, __import__("shutil").which
        try:
            subprocess.run = fake_run
            __import__("shutil").which = lambda name: "/fake/claude"
            out = llm.generate_json("PROMPT", {"type": "object"},
                                    system="SYS", model="cli:sonnet", deep=True)
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        args = captured["args"]
        check("fenced JSON from the CLI is still parsed", out == {"ok": True})
        check("the alias is passed, not the cli: name",
              "sonnet" in args and "cli:sonnet" not in args)
        check("effort is set high, since subscription time is not per-token",
              args[args.index("--effort") + 1] == llm.CLAUDE_CLI_EFFORT)
        check("the schema and the caller's system text ride on stdin, not in argv",
              "JSON Schema:" in captured["kwargs"]["input"]
              and "SYS" in captured["kwargs"]["input"]
              and "JSON Schema:" not in " ".join(args)
              and "SYS" not in " ".join(args))
        # `claude` is a .cmd shim on Windows, so cmd.exe builds the command
        # line and its ceiling is 8,191 characters, not the 32,767 of a native
        # CreateProcess. _WRITER_SYSTEM (8,168) plus the article schema (3,102)
        # busted it at the article stage, four calls into a run. Nothing that
        # scales with the article, the schema or the sources may go in argv.
        check("argv stays small enough for the cmd.exe 8,191-char ceiling",
              len(" ".join(args)) < 2000)
        check("MCP servers are suppressed, a measured 10x prompt-token tax",
              "--strict-mcp-config" in args
              and args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}')
        check("no tools, so the model answers instead of working",
              args[args.index("--tools") + 1] == "")
        check("the prompt goes on stdin, never in argv",
              "PROMPT" in captured["kwargs"]["input"] and "PROMPT" not in " ".join(args))
        check("the format demand is last, after the sources, not only up front",
              captured["kwargs"]["input"].rstrip().endswith("no YAML."))
        check("argv still carries the JSON-only contract",
              "one JSON object" in args[args.index("--system-prompt") + 1])
        check("it runs outside the repo, so no CLAUDE.md is auto-discovered",
              captured["kwargs"]["cwd"] and "articlegen" not in captured["kwargs"]["cwd"])

        # -- prose replies ------------------------------------------------
        # The failure this provider actually has. The API paths are given a
        # response_format and cannot return prose; the first real call here
        # answered a JSON-schema prompt in YAML and cost the whole run at the
        # first of eight stages.
        check("a JSON object is recovered from a reply wrapped in prose",
              llm._extract_json_object('Sure!\n```json\n{"a": {"b": 1}}\n```\nHope that helps')
              == '{"a": {"b": 1}}')
        check("braces and escaped quotes inside strings do not end the object",
              llm._extract_json_object('{"a": "has } and \\" quote"} trailing')
              == '{"a": "has } and \\" quote"}')
        check("a reply with no object at all is passed through untouched",
              llm._extract_json_object("core_entity: safety planning") ==
              "core_entity: safety planning")

        # -- deterministic repair (#147) ---------------------------------------
        # 3 of 5 write_article calls on cli:opus returned non-JSON at least once,
        # and one run died after its retry on "Expecting ',' delimiter: line 1
        # column 4942" — a complete article object one comma short.
        check("a missing comma between two values is repaired",
              llm._repair_json('{"a": "one" "b": "two"}') == {"a": "one", "b": "two"})
        check("and the same slip between two objects in an array",
              llm._repair_json(
                  '{"sections": [{"heading": "H1"} {"heading": "H2"}]}')
              == {"sections": [{"heading": "H1"}, {"heading": "H2"}]})
        check("a trailing comma before } or ] is dropped",
              llm._repair_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3})
        check("a bare newline inside a string is escaped",
              llm._repair_json('{"a": "line one\nline two"}')
              == {"a": "line one\nline two"})
        # The acceptance rule is parses AND dict. This repairs into valid JSON
        # that no caller of generate_json can use.
        check("a repair that yields a list is refused",
              llm._repair_json('[1, 2,]') is None)
        # A refusal is prose. There is nothing in it to salvage, and inventing a
        # dict from an apology would hand the pipeline a fake article.
        check("a refusal does not repair into anything",
              llm._repair_json("I'm sorry, I can't help with that.") is None)
        check("neither does a YAML reply",
              llm._repair_json("core_entity: safety planning\nqueries:\n  - a") is None)
        # The pass is a no-op on anything that already parses: in valid JSON a
        # value is only ever followed by , : } ] or the end.
        _valid = '{"a": "x, y", "b": [1, 2], "c": {"d": "has } and \\" quote"}}'
        check("valid JSON passes through the repair untouched",
              llm._repair_json_text(_valid) == _valid)

        near_miss_calls = []

        def near_miss_proc(args_, **kwargs):
            near_miss_calls.append(kwargs["input"])

            class P:
                returncode, stderr = 0, ""
                stdout = _json.dumps({
                    "type": "result", "subtype": "success", "is_error": False,
                    "usage": {}, "duration_ms": 1,
                    "result": '{"article": "body" "title": "T"}',
                })
            return P()

        try:
            subprocess.run = near_miss_proc
            __import__("shutil").which = lambda name: "/fake/claude"
            out = llm.generate_json("P", {"type": "object"}, model="cli:sonnet")
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        check("a near-miss reply is repaired on the first call, no retry spent",
              out == {"article": "body", "title": "T"} and len(near_miss_calls) == 1)

        unfixable_calls = []

        def unfixable_proc(args_, **kwargs):
            unfixable_calls.append(kwargs["input"])

            class P:
                returncode, stderr = 0, ""
                stdout = _json.dumps({
                    "type": "result", "subtype": "success", "is_error": False,
                    "usage": {}, "duration_ms": 1,
                    "result": "Just plain conversational prose with no JSON at all.",
                })
            return P()

        try:
            subprocess.run = unfixable_proc
            __import__("shutil").which = lambda name: "/fake/claude"
            llm.generate_json("P", {"type": "object"}, model="cli:sonnet")
            check("an unfixable reply raises after two calls", False)
        except RuntimeError as exc:
            check("an unfixable reply raises after two calls",
                  len(unfixable_calls) == 2 and "twice" in str(exc))
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        calls = []

        def yaml_then_json(args_, **kwargs):
            calls.append(kwargs["input"])

            class P:
                returncode, stderr = 0, ""
                stdout = _json.dumps({
                    "type": "result", "subtype": "success", "is_error": False,
                    "usage": {}, "duration_ms": 1,
                    "result": ('core_entity: safety planning\nqueries:\n  - a'
                               if len(calls) == 1 else '{"recovered": true}'),
                })
            return P()

        try:
            subprocess.run = yaml_then_json
            __import__("shutil").which = lambda name: "/fake/claude"
            out = llm.generate_json("P", {"type": "object"}, model="cli:sonnet")
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        check("a YAML reply is retried rather than losing the run",
              out == {"recovered": True} and len(calls) == 2)
        check("the retry tells the model its last reply was unusable",
              "was not valid JSON" in calls[1] and "P" in calls[1])
        check("the retry is not infinite", len(calls) == 2)

        # A refusal is a successful invocation carrying an apology. It has to
        # fail here, not later as "invalid JSON" pointing at that apology.
        class RefusedProc(FakeProc):
            stdout = _json.dumps({"type": "result", "subtype": "error_during_execution",
                                  "is_error": True, "result": "I can't help with that."})

        try:
            subprocess.run = lambda args, **kw: RefusedProc()
            __import__("shutil").which = lambda name: "/fake/claude"
            llm.generate_json("P", {"type": "object"}, model="cli:opus")
            check("a refusal raises rather than parsing as JSON", False)
        except RuntimeError as exc:
            check("a refusal raises rather than parsing as JSON",
                  "error_during_execution" in str(exc))
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        try:
            __import__("shutil").which = lambda name: None
            llm.generate_json("P", {"type": "object"}, model="cli:opus")
            check("a missing binary names the fix", False)
        except RuntimeError as exc:
            check("a missing binary names the fix", "not on PATH" in str(exc))
        finally:
            __import__("shutil").which = real_which
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def test_revision_replaces_blocks_rather_than_the_article() -> None:
    """The style revision returns the blocks it changed, not a new article.

    Rewriting all 3,586 words to fix three sentences was the most expensive call
    in a measured run — 26,632 output tokens against the 24,191 that wrote the
    article, at 5x the price of input. It is also the riskier shape: every
    untouched paragraph that comes back regenerated is another chance to drop a
    citation marker.

    What has to hold is that a partial reply merges into a complete article, and
    that a reply naming blocks the draft does not have changes nothing.
    """
    from articlegen import writer

    article = {
        "title": "Old title",
        "abstract": "Old abstract.",
        "key_points": ["old point"],
        "featured_study": {"source_index": 1, "method": "old method",
                           "results": "old results", "limitations": "old limits"},
        "sections": [
            {"heading": "Introduction", "paragraphs": ["intro one", "intro two"]},
            {"heading": "Boarding times", "paragraphs": ["boarding one"]},
            {"heading": "Conclusions", "paragraphs": ["conclusion one"]},
        ],
        "references": [1, 2, 3],
    }

    revised, applied = writer.apply_revisions(article, [
        {"where": "Boarding times", "replacement": ["fixed one", "fixed two"]},
        {"where": "key points", "replacement": ["fixed point"]},
        {"where": "featured study/method", "replacement": ["fixed method"]},
    ])

    check("a named section is replaced by its new paragraphs",
          revised["sections"][1]["paragraphs"] == ["fixed one", "fixed two"])
    check("the checker's spelling of a block is understood",
          revised["key_points"] == ["fixed point"])
    check("a featured-study field is reachable",
          revised["featured_study"]["method"] == "fixed method")
    check("untouched blocks survive verbatim",
          revised["sections"][0]["paragraphs"] == ["intro one", "intro two"]
          and revised["abstract"] == "Old abstract."
          and revised["featured_study"]["results"] == "old results")
    check("the structure the style gate counts is unchanged",
          len(revised["sections"]) == 3 and revised["references"] == [1, 2, 3])
    check("the original is not mutated",
          article["sections"][1]["paragraphs"] == ["boarding one"]
          and article["key_points"] == ["old point"])
    check("every applied block is reported", len(applied) == 3)

    # A heading the model invented means it restructured the article, which is
    # not what a style revision may do. Appending it would let the revision add
    # sections the gate never asked for.
    _, applied = writer.apply_revisions(article, [
        {"where": "A section that does not exist", "replacement": ["text"]},
        {"where": "Boarding times", "replacement": []},
    ])
    check("an unknown target is skipped, not appended", applied == [])

    # -- the call itself ------------------------------------------------
    captured = {}

    def fake_generate(prompt, schema, **kwargs):
        captured["schema"], captured["system"] = schema, kwargs.get("system")
        return {"edits": [{"where": "Introduction", "replacement": ["revised intro"]}]}

    real = writer.generate_json
    try:
        writer.generate_json = fake_generate
        out = writer.revise_prose(article, "BRIEF")
        check("revise_prose still returns a whole article",
              out["title"] == "Old title" and len(out["sections"]) == 3)
        check("with the edit merged in",
              out["sections"][0]["paragraphs"] == ["revised intro"])
        check("the patch schema is the one sent", "edits" in captured["schema"]["properties"])
        check("and the patch system prompt with it",
              "ONLY the blocks you changed" in captured["system"])

        # too-few-sections cannot be patched — the section is not there to
        # replace — so that one path still asks for the whole article.
        writer.generate_json = lambda p, s, **kw: (
            captured.update(schema=s, system=kw.get("system")) or dict(article))
        writer.revise_prose(article, "BRIEF", rewrite_whole=True)
        check("a whole rewrite still asks for the full article schema",
              "sections" in captured["schema"]["properties"]
              and "edits" not in captured["schema"]["properties"])

        # A reply that changes nothing must not be logged as a revision.
        writer.generate_json = lambda p, s, **kw: {"edits": []}
        try:
            writer.revise_prose(article, "BRIEF")
            check("an empty patch raises rather than faking a revision", False)
        except RuntimeError as exc:
            check("an empty patch raises rather than faking a revision",
                  "no edit that matched" in str(exc))
    finally:
        writer.generate_json = real

    # The one rule whose fix is a section that does not exist yet.
    import inspect

    from articlegen import pipeline
    src = inspect.getsource(pipeline.enforce_style)
    check("enforce_style asks for a whole rewrite only on too-few-sections",
          'i["rule"] == "too-few-sections"' in src and "rewrite_whole=rewrite_whole" in src)


def test_revision_carries_sources_only_when_they_can_be_used() -> None:
    """The sources ride along only for a failure that rewording cannot fix.

    `revision_brief` already splits the two cases: a substance failure is told to
    pull specific findings out of the sources, a register failure is told to
    reword and add nothing. Sending 20 abstracts and 60,000 characters of full
    text to fix a contraction is ~30,000 input tokens the model is forbidden to
    use.

    The pairing is the invariant: whenever the brief says "go back to the
    SOURCES", the sources must actually be there.
    """
    from articlegen import pipeline
    from articlegen.style import SUBSTANCE_RULES, revision_brief

    stats = {"sentences": 20, "mean_sentence_words": 22.0, "hedges_per_sentence": 0.3,
             "passive_ratio": 0.2}

    def report_with(rule, severity="error"):
        return {"issues": [{"rule": rule, "severity": severity, "where": "whole article",
                            "detail": "d", "excerpt": ""}], "stats": stats}

    seen = {}

    def fake_revise(article, brief, **kwargs):
        seen["papers"], seen["brief"] = kwargs.get("papers"), brief
        raise RuntimeError("stop here — only the call matters")

    papers = ["paper-stand-in"]
    saved = pipeline.revise_prose, pipeline.check_style
    try:
        pipeline.revise_prose = fake_revise

        for rule in ("contraction", "second-person", "booster"):
            pipeline.check_style = lambda a, **kw: report_with(rule)
            pipeline.enforce_style({"sections": []}, papers=papers)
            check(f"{rule}: a rewording fix travels without the sources",
                  seen["papers"] is None)
            check(f"{rule}: and the brief does not ask for them",
                  "go back to the SOURCES" not in seen["brief"].lower()
                  and "SOURCES below" not in seen["brief"])

        for rule in sorted(SUBSTANCE_RULES - {"under-length"}):
            pipeline.check_style = lambda a, **kw: report_with(rule)
            pipeline.enforce_style({"sections": []}, papers=papers)
            check(f"{rule}: a thinness fix still gets the sources",
                  seen["papers"] == papers)
            check(f"{rule}: and the brief tells the model to use them",
                  "SOURCES below" in seen["brief"])

        # under-length is a warning, never an error, so it cannot reach here on
        # its own — but paired with one that can, the sources must still travel.
        pipeline.check_style = lambda a, **kw: {
            "issues": [{"rule": "under-length", "severity": "warning",
                        "where": "whole article", "detail": "d", "excerpt": ""},
                       {"rule": "recycled-phrasing", "severity": "error",
                        "where": "whole article", "detail": "d", "excerpt": ""}],
            "stats": stats}
        pipeline.enforce_style({"sections": []}, papers=papers)
        check("a mixed report with a substance error still gets the sources",
              seen["papers"] == papers)
    finally:
        pipeline.revise_prose, pipeline.check_style = saved

    # The brief's own wording is what the split is keyed on, so a change to one
    # without the other is the failure this catches.
    substance = revision_brief(report_with("recycled-phrasing"))
    register = revision_brief(report_with("contraction"))
    check("the substance brief still names the sources", "SOURCES below" in substance)
    check("the register brief still forbids new material",
          "do not introduce new claims" in register)


def test_warnings_ride_along_on_a_revision() -> None:
    """Warning-level style findings ride along only when errors triggered a revision.

    `check_style` flags long sentences, wordiness and passive voice as warnings.
    They never trigger a revision on their own — `enforce_style` returns early
    when there are no errors — but once the model is revising anyway, warnings
    in the same blocks ride along in `revision_brief()` as secondary items to fix
    (#145).

    The acceptance rule and source-passing split stay keyed strictly on errors.
    """
    from articlegen import demo, pipeline, style
    from articlegen.style import (
        LONG_SENTENCE_WORDS, RIDE_ALONG_WARNINGS, SUBSTANCE_RULES, check_style, errors, revision_brief,
    )

    # 1. Real prose, end to end: fixture trips a contraction error and a long-sentence warning.
    long_sentence = "This analysis indicates that " + " ".join(["evidence"] * (LONG_SENTENCE_WORDS + 5)) + "."
    article_1 = {
        "sections": [{
            "heading": "Introduction",
            "paragraphs": [f"It's clear that results matter. {long_sentence}"]
        }]
    }
    rep_1 = check_style(article_1)
    rules_fired_1 = {i["rule"] for i in rep_1["issues"]}
    check("fixture produces a contraction error", "contraction" in rules_fired_1)
    check("fixture produces a long-sentence warning", "long-sentence" in rules_fired_1)
    brief_1 = revision_brief(rep_1)
    check("brief contains the contraction error first", "Contraction" in brief_1)
    check("brief contains the secondary warning section",
          "Also fix these while you are in the same blocks" in brief_1)
    check("long-sentence warning rides along in the brief", "word sentence; split it" in brief_1)

    # 2. Every long sentence rides along, not just the first.
    long_sentence_2 = "A secondary review shows that " + " ".join(["finding"] * (LONG_SENTENCE_WORDS + 5)) + "."
    article_2 = {
        "sections": [{
            "heading": "Introduction",
            "paragraphs": [
                f"It's clear that results matter. {long_sentence}",
                f"They're observing that {long_sentence_2}",
            ]
        }]
    }
    rep_2 = check_style(article_2)
    brief_2 = revision_brief(rep_2)
    check("every long sentence rides along in the brief",
          brief_2.count("word sentence; split it") == 2)

    # 3. Warnings alone buy no revision.
    called = []

    def fake_revise(article, brief, **kwargs):
        called.append(True)
        raise RuntimeError("revise_prose should not be called for warnings alone")

    stats = {"sentences": 20, "mean_sentence_words": 22.0, "hedges_per_sentence": 0.3,
             "passive_ratio": 0.2}

    def report_with(rule, severity="error"):
        return {"issues": [{"rule": rule, "severity": severity, "where": "whole article",
                            "detail": "d", "excerpt": ""}], "stats": stats}

    saved = pipeline.revise_prose, pipeline.check_style
    try:
        pipeline.revise_prose = fake_revise
        pipeline.check_style = lambda a, **kw: report_with("long-sentence", severity="warning")
        dummy_art = {"sections": [{"heading": "Introduction", "paragraphs": ["Prose."]}]}
        res_art, res_rep = pipeline.enforce_style(dummy_art)
        check("warnings alone do not trigger revise_prose", len(called) == 0)
        check("enforce_style returns original article unchanged when warnings-only", res_art == dummy_art)
    finally:
        pipeline.revise_prose, pipeline.check_style = saved

    # 4. under-length does not ride along.
    mixed_report = {
        "issues": [
            {"rule": "contraction", "severity": "error", "where": "Introduction",
             "detail": "Contraction 'it\\'s'", "excerpt": "it's"},
            {"rule": "under-length", "severity": "warning", "where": "whole article",
             "detail": "500 words of body prose", "excerpt": ""},
        ],
        "stats": stats,
    }
    mixed_brief = revision_brief(mixed_report)
    check("under-length warning does not ride along in the brief",
          "under-length" not in mixed_brief and "words of body prose" not in mixed_brief)
    check("register brief with under-length warning still forbids new material",
          "do not introduce new claims or numbers" in mixed_brief)
    check("and does not ask for sources", "SOURCES below" not in mixed_brief)

    # 5. The curated sample is untouched.
    sample_rep = check_style(demo.SAMPLE_ARTICLE)
    check("curated demo.SAMPLE_ARTICLE has zero style errors", errors(sample_rep) == [])
    sample_brief = revision_brief(sample_rep)
    check("warning-only sample brief has no 'Also fix these' section",
          "Also fix these" not in sample_brief)


def test_a_second_style_pass_runs_only_after_progress() -> None:
    """A second revision pass is allowed, and only after the first one worked.

    Two of three measured runs ended with exactly one residual style error after
    a productive revision (3 -> 1 and 2 -> 1), which is what a second pass is
    sized to clear. Whether an unfixed error reaches the page is
    `SENDABLE_BLOCKING_RULES`' business (#169). So `enforce_style` may go round
    twice — but the second pass is gated on the first having been accepted, and
    acceptance means strictly fewer errors. An error the model cannot fix costs
    one call, never a loop (#146).
    """
    from articlegen import pipeline

    stats = {"sentences": 20, "mean_sentence_words": 22.0, "hedges_per_sentence": 0.3,
             "passive_ratio": 0.2}

    def report_with(n_errors, rule="contraction"):
        return {"issues": [{"rule": rule, "severity": "error", "where": "whole article",
                            "detail": "d", "excerpt": ""} for _ in range(n_errors)],
                "stats": stats}

    article = {"sections": [{"heading": "Introduction", "paragraphs": ["Prose."]}],
               "references": []}

    def run(error_counts):
        """Drive enforce_style with a scripted sequence of error counts.

        `error_counts[0]` is the initial check; each later entry is the check on
        the revision produced by that pass. Returns (passes_run, final_report).
        """
        seq = list(error_counts)
        calls = {"check": 0, "revise": 0, "sources": []}

        def fake_check(a, **kw):
            i = min(calls["check"], len(seq) - 1)
            calls["check"] += 1
            return report_with(seq[i])

        def fake_revise(a, brief, **kwargs):
            calls["revise"] += 1
            calls["sources"].append(kwargs.get("papers"))
            return dict(a)

        saved = pipeline.revise_prose, pipeline.check_style
        try:
            pipeline.revise_prose = fake_revise
            pipeline.check_style = fake_check
            out_article, out_report = pipeline.enforce_style(article)
        finally:
            pipeline.revise_prose, pipeline.check_style = saved
        return calls, out_report

    # 1. Improves, then clears: two passes run and the draft comes back clean.
    calls, report = run([3, 1, 0])
    check("a productive first pass buys a second", calls["revise"] == 2)
    check("and a cleared draft is what comes back",
          not [i for i in report["issues"] if i["severity"] == "error"])

    # 2. Improves, then stalls: the cap stops it at two.
    calls, report = run([3, 1, 1])
    check("progress then a stall stops at the two-pass cap", calls["revise"] == 2)
    check("and the better of the two drafts is kept",
          len([i for i in report["issues"] if i["severity"] == "error"]) == 1)

    # 3. No improvement on the first pass: no second pass at all.
    calls, _ = run([2, 2])
    check("a pass that did not improve buys no second pass", calls["revise"] == 1)

    # 4. A revision that gets worse is discarded and stops the loop.
    calls, report = run([2, 3])
    check("a worse revision buys no second pass", calls["revise"] == 1)
    check("and the original draft is kept",
          len([i for i in report["issues"] if i["severity"] == "error"]) == 2)

    # 5. The cap is two, stated once.
    check("the pass cap is a named constant set to two",
          pipeline.MAX_STYLE_PASSES == 2)

    # 6. The substance split still applies on the second pass: if what is left
    #    is a thinness failure, the sources travel with that call too.
    seq = [("recycled-phrasing", 3), ("recycled-phrasing", 1), ("recycled-phrasing", 0)]
    calls = {"check": 0, "revise": 0, "sources": []}

    def fake_check(a, **kw):
        rule, n = seq[min(calls["check"], len(seq) - 1)]
        calls["check"] += 1
        return report_with(n, rule=rule)

    def fake_revise(a, brief, **kwargs):
        calls["revise"] += 1
        calls["sources"].append(kwargs.get("papers"))
        return dict(a)

    papers = ["paper-stand-in"]
    saved = pipeline.revise_prose, pipeline.check_style
    try:
        pipeline.revise_prose = fake_revise
        pipeline.check_style = fake_check
        pipeline.enforce_style(article, papers=papers)
    finally:
        pipeline.revise_prose, pipeline.check_style = saved
    check("the sources travel on both passes of a substance failure",
          calls["sources"] == [papers, papers])


def test_tangential_sources_stay_out_of_the_writer_prompt() -> None:
    """Background-only sources cost tokens the relevance gate exists to refuse.

    `curate_sources` defines "tangential" as good for background or framing
    only, and the pipeline already refuses to spend a full-text fetch on one.
    They were still arriving at `write_article` with a full abstract each.

    The trap is the numbering. The SOURCE index *is* the citation scheme —
    `render` maps the writer's markers back through it — so dropping a source
    must leave a gap, never re-pack the list.
    """
    from articlegen import writer
    from articlegen.sources import Paper

    papers = [Paper(title=f"P{i}", abstract=f"abstract of paper {i}", year=2024)
              for i in range(1, 6)]
    relevance = {1: "direct", 2: "tangential", 3: "related",
                 4: "tangential", 5: "direct"}

    rendered = writer._format_sources(papers, relevance,
                                      omit={i for i, r in relevance.items()
                                            if r == "tangential"})
    check("tangential abstracts are gone",
          "abstract of paper 2" not in rendered and "abstract of paper 4" not in rendered)
    check("the ones that earned their place stay",
          all(f"abstract of paper {i}" in rendered for i in (1, 3, 5)))
    check("the surviving sources keep their original numbers",
          "SOURCE 3 [related to topic]" in rendered
          and "SOURCE 5 [direct to topic]" in rendered)
    check("and the numbering is not re-packed to close the gaps",
          "SOURCE 2" not in rendered and "SOURCE 4" not in rendered)

    # -- what write_article actually sends -------------------------------
    captured = {}

    def fake_generate(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return {"title": "t", "sections": []}

    real = writer.generate_json
    try:
        writer.generate_json = fake_generate
        writer.write_article("topic", papers, curation={
            "relevance": relevance,
            "counts": {"direct": 2, "related": 1, "tangential": 2},
            "most_relevant_index": 1,
        })
    finally:
        writer.generate_json = real

    prompt = captured["prompt"]
    check("write_article drops them too", "abstract of paper 2" not in prompt)
    check("the tally still counts them, so the prompt must say they are missing",
          "2 tangential" in prompt and "NOT reproduced below" in prompt)
    check("and warn about the gaps it just created", "gaps" in prompt)

    # The degraded case: curation failed and labelled nothing. Dropping every
    # source because none is labelled "direct" would leave the writer with an
    # empty payload, which is worse than a bloated one.
    try:
        writer.generate_json = fake_generate
        writer.write_article("topic", papers, curation={})
    finally:
        writer.generate_json = real
    check("an unlabelled run keeps every source",
          all(f"abstract of paper {i}" in captured["prompt"] for i in range(1, 6)))


def test_the_writer_cites_a_working_set() -> None:
    """The candidate pool is 40 so the relevance gate has something to throw away
    (#141). The shipped drafts still cited almost everything: safety-planning 20/20,
    seclusion 17/20 (#167). The writer must cite a working set of about 12, while
    the candidate pool stays 40.
    """
    from articlegen import render, sources, writer
    from articlegen.sources import Paper

    # 1. The pool did not shrink.
    check("candidate pool default stays 40", sources.DEFAULT_MAX_PAPERS == 40)
    check("curation abstracts are not truncated", writer.CURATION_ABSTRACT_CHARS is None)

    # 2. The constant exists and is 12.
    check("TARGET_CITED_SOURCES is 12", writer.TARGET_CITED_SOURCES == 12)

    # 3. One rule string, in both prompts.
    check("working-set rule in briefing system prompt",
          writer._WORKING_SET_RULE in writer._BRIEFING_SYSTEM)
    check("working-set rule in writer system prompt",
          writer._WORKING_SET_RULE in writer._WRITER_SYSTEM)
    check("working-set rule text contains target count",
          str(writer.TARGET_CITED_SOURCES) in writer._WORKING_SET_RULE)

    # 4. The inclusion instruction is gone.
    check("inclusion instruction removed from writer prompt",
          "cite the related and tangential sources too" not in writer._WRITER_SYSTEM)
    check("writer prompt notes tangential sources were withheld",
          "sources labelled tangential have already been withheld" in writer._WRITER_SYSTEM)

    # 5. Full-text variants still carry the rule.
    check("working-set rule survives in full-text writer prompt",
          writer._WORKING_SET_RULE in writer._WRITER_SYSTEM_FULLTEXT)
    check("working-set rule survives in full-text briefing prompt",
          writer._WORKING_SET_RULE in writer._BRIEFING_SYSTEM_FULLTEXT)

    # 6. The run's own numbers reach the model.
    papers_40 = [Paper(title=f"P{i}", abstract=f"Abstract {i}", year=2024) for i in range(1, 41)]
    relevance_40 = {i: ("tangential" if i <= 4 else ("direct" if i <= 20 else "related")) for i in range(1, 41)}
    curation_40 = {
        "relevance": relevance_40,
        "counts": {"direct": 16, "related": 20, "tangential": 4},
        "most_relevant_index": 5,
    }
    captured = {}

    def fake_generate(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return {"title": "t", "sections": [], "question": "q", "answer": "a", "findings": [], "unknowns": [], "open_these": []}

    real = writer.generate_json
    try:
        writer.generate_json = fake_generate
        writer.write_briefing("topic", papers_40, curation=curation_40)
        briefing_prompt = captured["prompt"]
        writer.write_article("topic", papers_40, curation=curation_40)
        article_prompt = captured["prompt"]
    finally:
        writer.generate_json = real

    for name, prompt in (("briefing", briefing_prompt), ("article", article_prompt)):
        check(f"{name} prompt includes WORKING SET", "WORKING SET" in prompt)
        check(f"{name} prompt reports screened count", "40 records were screened" in prompt)
        check(f"{name} prompt reports shown count", "36 are reproduced below" in prompt)
        check(f"{name} prompt specifies target cited count",
              f"AT MOST {writer.TARGET_CITED_SOURCES} of them" in prompt)

    # 7. A thin pool is not asked for twelve.
    papers_6 = [Paper(title=f"P{i}", abstract=f"Abstract {i}", year=2024) for i in range(1, 7)]
    relevance_6 = {1: "tangential", 2: "direct", 3: "direct", 4: "related", 5: "related", 6: "related"}
    curation_6 = {
        "relevance": relevance_6,
        "counts": {"direct": 2, "related": 3, "tangential": 1},
        "most_relevant_index": 2,
    }
    try:
        writer.generate_json = fake_generate
        writer.write_briefing("topic", papers_6, curation=curation_6)
    finally:
        writer.generate_json = real
    thin_prompt = captured["prompt"]
    check("thin pool reports 5 reproduced below", "5 are reproduced below" in thin_prompt)
    check("thin pool does not ask for about 12", "about 12" not in thin_prompt)
    check("thin pool does not ask for at most 12", "at most 12" not in thin_prompt)

    # 8. Methods prints screened and cited as two different numbers.
    prov = {"queries": ["q"], "databases": ["Europe PMC"], "date": "21 August 2026"}
    m_html = render._methods_html(prov, screened=40, n_cited=12, topic="x")
    m_md = "\n".join(render._methods_markdown(prov, screened=40, n_cited=12, topic="x"))
    check("methods HTML shows screened and cited as two different numbers",
          "leaving 40" in m_html and "12 were cited here" in m_html)
    check("methods markdown shows screened and cited as two different numbers",
          "leaving 40" in m_md and "12 were cited here" in m_md)

    # 9. Methods does not over-claim the Read column.
    prov_ft = {
        "queries": ["q"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
        "full_text_sources": [1, 2, 3, 4, 5],
    }
    m_ft_partial = render._methods_html(prov_ft, screened=40, n_cited=12, topic="x", n_full_cited=3)
    check("methods HTML specifies cited count when full-text cited differs from fetched",
          "3 of which are cited here and marked in Table 1" in m_ft_partial)
    check("methods HTML omits bare marker when full-text cited differs",
          "(marked in Table 1)" not in m_ft_partial)

    m_ft_all = render._methods_html(prov_ft, screened=40, n_cited=12, topic="x", n_full_cited=5)
    check("methods HTML uses bare marker when all fetched full texts are cited",
          "(marked in Table 1)" in m_ft_all)


def test_the_cite_ceiling_scales_with_the_direct_count() -> None:
    """The cite ceiling scales with the evidence that is actually on-topic (#188).

    Four Grok 4.6 briefings (21 Aug 2026) showed that a flat cite target of 12
    padded thin-direct pools with related sources and weak primary studies.
    The ask now scales to min(12, n_direct + 2) floored at MIN_CITED_SOURCES (5),
    pins question and title to the user's topic, and prefers read full texts over
    abstract-only case reports.
    """
    from articlegen import writer
    from articlegen.sources import Paper

    # 1. cite_target calculations
    check("n_direct=7, shown=30 scales to 9", writer.cite_target(7, 30) == 9)
    check("n_direct=21, shown=30 caps at 12", writer.cite_target(21, 30) == 12)
    check("n_direct=0, shown=30 floors at 5", writer.cite_target(0, 30) == 5)
    check("n_direct=None (unlabelled), shown=30 returns 12", writer.cite_target(None, 30) == 12)
    check("n_direct=3, shown=4 caps at shown 4", writer.cite_target(3, 4) == 4)

    # 2. Constants
    check("MIN_CITED_SOURCES is 5", writer.MIN_CITED_SOURCES == 5)
    check("TARGET_CITED_SOURCES is 12", writer.TARGET_CITED_SOURCES == 12)

    # 3. Prompt level cite targets
    papers_30 = [Paper(title=f"P{i}", abstract=f"Abstract {i}", year=2024) for i in range(1, 31)]
    relevance_thin = {i: ("direct" if i <= 7 else ("related" if i <= 27 else "tangential")) for i in range(1, 31)}
    curation_thin = {
        "relevance": relevance_thin,
        "counts": {"direct": 7, "related": 20, "tangential": 3},
        "most_relevant_index": 1,
    }
    captured = {}

    def fake_generate(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return {"title": "t", "sections": [], "question": "q", "answer": "a", "findings": [], "unknowns": [], "open_these": []}

    real = writer.generate_json
    try:
        writer.generate_json = fake_generate
        writer.write_briefing("user topic", papers_30, curation=curation_thin)
        thin_prompt = captured["prompt"]
    finally:
        writer.generate_json = real

    check("thin-direct briefing asks for at most 9", "at most 9 of them" in thin_prompt.lower())
    check("thin-direct briefing does not ask for at most 12", "at most 12" not in thin_prompt.lower())
    check("thin-direct briefing does not ask for about 12", "about 12" not in thin_prompt.lower())

    # 4. Rich-direct pool asks for at most 12
    relevance_rich = {i: ("direct" if i <= 21 else ("related" if i <= 27 else "tangential")) for i in range(1, 31)}
    curation_rich = {
        "relevance": relevance_rich,
        "counts": {"direct": 21, "related": 6, "tangential": 3},
        "most_relevant_index": 1,
    }
    try:
        writer.generate_json = fake_generate
        writer.write_briefing("user topic", papers_30, curation=curation_rich)
        rich_prompt = captured["prompt"]
    finally:
        writer.generate_json = real

    check("rich-direct briefing asks for at most 12", "at most 12 of them" in rich_prompt.lower())

    # 5. Rule text in _WORKING_SET_RULE
    check("_WORKING_SET_RULE contains AT MOST", "AT MOST" in writer._WORKING_SET_RULE)
    check("_WORKING_SET_RULE contains two-related cap", "at most two" in writer._WORKING_SET_RULE.lower())
    check("_WORKING_SET_RULE no longer contains Cite about", "Cite about" not in writer._WORKING_SET_RULE)

    # 6. Full-text preference in prompt
    papers_with_ft = [
        Paper(title=f"P{i}", abstract=f"Abstract {i}", year=2024,
              full_text=f"Full text content for paper {i}" if i in (2, 5) else "")
        for i in range(1, 7)
    ]
    curation_ft = {
        "relevance": {i: "direct" for i in range(1, 7)},
        "counts": {"direct": 6, "related": 0, "tangential": 0},
    }
    try:
        writer.generate_json = fake_generate
        writer.write_briefing("user topic", papers_with_ft, curation=curation_ft)
        ft_prompt = captured["prompt"]
        writer.write_briefing("user topic", papers_30, curation=curation_thin)
        no_ft_prompt = captured["prompt"]
    finally:
        writer.generate_json = real

    check("full text present: DEEP READS in prompt", "DEEP READS" in ft_prompt)
    check("full text present: names SOURCE 2, 5", "SOURCE 2, 5" in ft_prompt)
    check("full text present: contains preference sentence", "Prefer them:" in ft_prompt)
    check("no full text: DEEP READS not in prompt", "DEEP READS" not in no_ft_prompt)
    check("no full text: preference sentence not in prompt", "Prefer them:" not in no_ft_prompt)

    # 7. Topic fidelity
    check("topic fidelity rule in _BRIEFING_SYSTEM",
          writer._TOPIC_FIDELITY_RULE in writer._BRIEFING_SYSTEM)
    check("topic fidelity rule in _WRITER_SYSTEM",
          writer._TOPIC_FIDELITY_RULE in writer._WRITER_SYSTEM)
    check("topic fidelity rule in _BRIEFING_SYSTEM_FULLTEXT",
          writer._TOPIC_FIDELITY_RULE in writer._BRIEFING_SYSTEM_FULLTEXT)
    check("topic fidelity rule in _WRITER_SYSTEM_FULLTEXT",
          writer._TOPIC_FIDELITY_RULE in writer._WRITER_SYSTEM_FULLTEXT)
    check("narrow the population in _TITLE_RULE",
          "narrow the population" in writer._TITLE_RULE)
    check("narrow the population in briefing schema question description",
          "narrowed population" in writer._BRIEFING_SCHEMA["properties"]["question"]["description"])
    check("briefing prompt contains topic line guidance",
          "reader's question, as typed" in thin_prompt)

    # 8. Schema title rule sharing
    check("both schemas share _TITLE_RULE",
          writer._ARTICLE_SCHEMA["properties"]["title"]["description"] ==
          writer._BRIEFING_SCHEMA["properties"]["title"]["description"] ==
          writer._TITLE_RULE)


def test_gemini_cli_provider() -> None:
    """Drafting on a Gemini subscription, through the Antigravity CLI.

    Same opt-in invariant as `claude-cli`: it answers as whoever is signed in
    on this machine, so nothing about the environment may select it and the web
    app must never offer it.

    The transport is what needs pinning. `agy` ignores stdin, and the article
    prompt runs to ~95,000 characters against a 32,767-character Windows
    command line, so the prompt is handed over as an inlined `@file` reference
    and only small fixed flags go in argv. It does enforce the schema, which
    the Claude CLI cannot, so the parsed object comes straight out of the
    envelope.
    """
    import json as _json

    from articlegen import llm, web

    saved = {v: os.environ.get(v) for v in
             ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARTICLEGEN_PROVIDER")}
    try:
        for v in saved:
            os.environ.pop(v, None)

        check("agy: prefix routes to the Antigravity CLI, stripped to the model",
              llm.resolve_provider("agy:gemini-3.1-pro-high")
              == ("gemini-cli", "gemini-3.1-pro-high"))
        check("a bare agy: takes the default model",
              llm.resolve_provider("agy:") == ("gemini-cli", llm.GEMINI_CLI_DEFAULT_MODEL))
        check("no keys at all still falls back to OpenRouter, not the CLI",
              llm.resolve_provider()[0] == "openrouter")
        check("the CLI model is not offered to the web app",
              llm.GEMINI_CLI_DEFAULT_MODEL not in web.ALLOWED_MODELS
              and not any(m.startswith(llm.GEMINI_CLI_PREFIX) for m in web.ALLOWED_MODELS))
        check("the web app drops an agy: model rather than honouring it",
              web._requested_model({"model": "agy:gemini-3.6-flash-high"}) is None)

        # -- the subprocess contract -------------------------------------
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            # Read back what was written into the scratch directory before the
            # TemporaryDirectory that holds it is torn down.
            prompt_path = args[args.index("-p") + 1].lstrip("@")
            captured["prompt"] = open(prompt_path, encoding="utf-8").read()
            captured["schema"] = _json.load(
                open(args[args.index("--json-schema") + 1], encoding="utf-8"))

            class P:
                returncode, stderr = 0, ""
                stdout = _json.dumps({
                    "status": "SUCCESS", "duration_seconds": 1.5,
                    "usage": {"input_tokens": 3, "output_tokens": 9},
                    "response": '{"ok": true}',
                    "structured_output": {"ok": True},
                })
            return P()

        import subprocess
        real_run, real_which = subprocess.run, __import__("shutil").which
        try:
            subprocess.run = fake_run
            __import__("shutil").which = lambda name: "/fake/agy"
            out = llm.generate_json("PROMPT", {"type": "object"}, system="SYS",
                                    model="agy:gemini-3.6-flash-high", deep=True)
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        args = captured["args"]
        check("the enforced structured output is used directly", out == {"ok": True})
        check("the model name is passed without the agy: prefix",
              args[args.index("--model") + 1] == "gemini-3.6-flash-high")

        check("the schema is enforced, not merely requested",
              captured["schema"] == {"type": "object"})
        # The whole reason for the file: argv cannot carry the article prompt.
        check("the prompt is an inlined @file, never an argument",
              args[args.index("-p") + 1].startswith("@")
              and "PROMPT" not in " ".join(args))
        check("the caller's system text rides in that file",
              "SYS" in captured["prompt"] and "PROMPT" in captured["prompt"])
        check("argv stays small regardless of the sources",
              len(" ".join(args)) < 2000)
        check("slash-command expansion is off, since a source may start with /",
              "--disable-slash-commands" in args)
        check("it runs outside the repo",
              captured["kwargs"]["cwd"] and "articlegen" not in captured["kwargs"]["cwd"])

        # -- reasoning tier --------------------------------------------------
        # Running the shallow stages a tier down was tried and reverted. It is a
        # one-line change that looks free, so this pins the outcome rather than
        # leaving the next person to rediscover it: at -low, curation agreed with
        # -high on only 14 of 20 relevance labels and collapsed everything toward
        # "related", which is the gate that stops topic drift. See the note above
        # GEMINI_CLI_DEFAULT_MODEL in llm.py.
        shallow = {}

        def capture_shallow(args_, **kwargs):
            shallow["args"] = args_

            class P:
                returncode, stderr = 0, ""
                stdout = _json.dumps({"status": "SUCCESS", "response": "{}",
                                      "structured_output": {"ok": True}, "usage": {}})
            return P()

        try:
            subprocess.run = capture_shallow
            __import__("shutil").which = lambda name: "/fake/agy"
            llm.generate_json("P", {"type": "object"}, model="agy:gemini-3.6-flash-high")
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which
        check("a shallow call still runs the model the operator named",
              shallow["args"][shallow["args"].index("--model") + 1]
              == "gemini-3.6-flash-high")
        check("no tier-rewriting helper survives the revert",
              not hasattr(llm, "_gemini_cli_model"))

        # -- failures ------------------------------------------------------
        class Failed:
            returncode, stderr = 0, ""
            stdout = _json.dumps({"status": "ERROR", "response": "quota exhausted"})

        try:
            subprocess.run = lambda args_, **kw: Failed()
            __import__("shutil").which = lambda name: "/fake/agy"
            llm.generate_json("P", {"type": "object"}, model="agy:")
            check("a non-SUCCESS status raises rather than parsing as JSON", False)
        except RuntimeError as exc:
            check("a non-SUCCESS status raises rather than parsing as JSON",
                  "status=ERROR" in str(exc))
        finally:
            subprocess.run = real_run
            __import__("shutil").which = real_which

        try:
            __import__("shutil").which = lambda name: None
            llm.generate_json("P", {"type": "object"}, model="agy:")
            check("a missing binary names the fix", False)
        except RuntimeError as exc:
            check("a missing binary names the fix", "not on PATH" in str(exc))
        finally:
            __import__("shutil").which = real_which
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def test_disclosure_is_above_the_fold_and_derived() -> None:
    """Issue #91: six published pages carried no AI disclosure at all, and the
    masthead's only hint was "Not peer reviewed" in the smallest grey type on
    the page — which to a non-academic reads as *preprint*, a far higher trust
    category than "a language model wrote this". The banner has to sit under
    the <h1>, at full contrast, in both output formats and on the index.

    Its grounding half is *derived* from the cited papers, never hardcoded —
    the same no-fallback rule that issue #75 was filed over.
    """
    from articlegen import render
    from articlegen.sources import Paper

    art = {"title": "T", "abstract": "A.", "sections": [], "key_points": [],
           "references": [1, 2], "keywords": []}
    abstracts = [Paper(title="a", abstract="x", year=2020),
                 Paper(title="b", abstract="x", year=2021)]
    mixed = [Paper(title="a", abstract="x", full_text="body", year=2020),
             Paper(title="b", abstract="x", year=2021)]
    full = [Paper(title="a", abstract="x", full_text="body", year=2020),
            Paper(title="b", abstract="x", full_text="body", year=2021)]

    check("banner names the author when nothing is read in full",
          render._disclosure_banner(abstracts)
          == "Written by a language model from the cited research, from their "
             "abstracts alone. No human author wrote or checked this text. "
             "Not peer reviewed.")
    check("banner counts full texts, and counts them like Table 1",
          "from 1 full text and the abstracts of the other 1"
          in render._disclosure_banner(mixed))
    check("banner drops the abstract clause when every source was read in full",
          "from their full texts" in render._disclosure_banner(full)
          and "abstracts" not in render._disclosure_banner(full))

    html_out = render.render_article(art, mixed, "night shift work",
                                     curation={1: "direct", 2: "related"})
    md_out = render.render_markdown(art, mixed, "night shift work",
                                    curation={1: "direct", 2: "related"})
    check("html banner sits directly under the h1",
          html_out.index("</h1>") < html_out.index('class="ai-disclosure"')
          < html_out.index('class="meta-line"'))
    check("html banner is full contrast, not muted",
          ".ai-disclosure {\n    font-family" in html_out
          and "color: var(--ink); \n" not in html_out
          and "font-size: 0.92rem; font-weight: 600" in html_out)
    check("md banner is the second line of the header block",
          md_out.index("# T") < md_out.index("No human author wrote or checked")
          < md_out.index("Generated "))
    for name, out in (("HTML", html_out), ("Markdown", md_out)):
        check(f"{name} disclosure says a model wrote it, not that it was assisted",
              "Written by a language model" in out and "AI assistance" not in out)

    index_html = render._INDEX_TEMPLATE.format(count=0, items="")
    check("index warns before the list, not after",
          index_html.index("idx-disclosure") < index_html.index("<ul>"))
    check("index says no human checked any of them",
          "No human author wrote or checked any of them" in index_html)


def test_display_items_are_selected_once_for_both_formats() -> None:
    """HTML and Markdown must not each decide what a display item contains.

    Box 1, Fig. 1 and Table 1 are the parts a model cannot fabricate, so both
    renderers have to show the same facts. They used to compute those facts
    independently — the same index validation, field picking, relevance
    labelling and year bucketing written twice — which is a correctness risk,
    not just duplication: the two could disagree and nothing would notice
    (issue #46). Selection now lives in one place; the renderers only format.
    """
    import inspect

    from articlegen import demo, render

    cited, cite_map = render._citation_map(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS)
    labels = render._display_relevance(cite_map, demo.SAMPLE_CURATION)

    # Both renderers must go through the shared selectors.
    for fn, selector in (
        (render._box_html, "_box_parts"), (render._box_markdown, "_box_parts"),
        (render._table_html, "_table_rows"), (render._table_markdown, "_table_rows"),
        (render._figure_html, "_figure_series"), (render._figure_markdown, "_figure_series"),
    ):
        check(f"{fn.__name__} uses {selector}", selector in inspect.getsource(fn))

    # And the facts they carry must actually agree.
    rows = render._table_rows(cited, labels)
    check("a row per cited source", len(rows) == len(cited))
    table_html = render._table_html(cited, labels)
    table_md = render._table_markdown(cited, labels)
    for row in rows:
        for value in (str(row["year"]), row["study"], row["design"]):
            if value == "—":
                continue
            check(f"both tables carry {value[:26]!r}",
                  value in table_md and (value in table_html or html_escaped(value, table_html)))

    series = render._figure_series(cited, labels)
    check("the figure has data to plot", series is not None)
    fig_md = render._figure_markdown(cited, labels)
    for (label, _), tally in zip(series["buckets"], series["counts"]):
        check(f"markdown figure reports bucket {label}",
              f"- {label}: {sum(tally.values())}" in fig_md)

    box = render._box_parts(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, cite_map)
    check("the featured study resolves", box is not None)
    box_html, box_md = (
        render._box_html(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, cite_map),
        render._box_markdown(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, cite_map),
    )
    check("both boxes name the same study",
        box["paper"].title in box_md and html_escaped(box["paper"].title, box_html))
    check("both boxes carry the method", box["method"][:40] in box_md)

    # A bad index must fail the same way in both, not render half a box.
    broken = dict(demo.SAMPLE_ARTICLE, featured_study={"source_index": 999})
    check("an out-of-range featured study yields nothing",
          render._box_parts(broken, demo.SAMPLE_PAPERS, cite_map) is None
          and render._box_html(broken, demo.SAMPLE_PAPERS, cite_map) == ""
          and render._box_markdown(broken, demo.SAMPLE_PAPERS, cite_map) == "")


def test_figure_one_counts_study_designs() -> None:
    """Fig. 1 counts cited sources by study design with fallback to year histogram.

    Table 1 drops 'Cited by' and adds 'Design'. Citation counts stay on the reference list.
    """
    from articlegen import demo, render, sources
    from articlegen.sources import Paper, classify_design, paper_design

    # 1. Design mode fires with distinct recognisable designs
    papers = [
        Paper(title="Efficacy of light therapy: a systematic review and meta-analysis", abstract="a", year=2020, citation_count=50),
        Paper(title="Safety planning for self-harm: a randomized controlled trial", abstract="a", year=2021, citation_count=30),
        Paper(title="Containment in psychiatric wards: a cluster-randomised trial", abstract="a", year=2022, citation_count=20),
        Paper(title="Incidence of depression: a prospective cohort study", abstract="a", year=2023, citation_count=15),
        Paper(title="Staff experiences of seclusion: a qualitative interview study", abstract="a", year=2024, citation_count=10),
        Paper(title="General overview of clinical services", abstract="a", year=2025, citation_count=5),
    ]
    labels = {1: "direct", 2: "direct", 3: "related", 4: "related", 5: "related", 6: "tangential"}
    series = render._figure_series(papers, labels)
    check("design mode fires when metadata supports it", series is not None and series["mode"] == "design")
    valid_labels = set(sources.DESIGN_LABELS.values())
    bucket_labels = [label for label, _ in series["buckets"]]
    check("bucket labels drawn from DESIGN_LABELS", all(b in valid_labels for b in bucket_labels))
    total_in_buckets = sum(sum(t.values()) for t in series["counts"])
    check("per-bucket totals sum to number of cited sources", total_in_buckets == len(papers))

    # 2. HTML says which axis it is & caption disclaims quality appraisal
    html_fig = render._figure_html(papers, labels)
    check("HTML figure contains Study design", "Study design" in html_fig)
    check("HTML figure does not contain Year of publication", "Year of publication" not in html_fig)
    check("HTML figure caption disclaims quality appraisal",
          "it is not a quality appraisal" in html_fig or "not a quality appraisal" in html_fig)

    # 3. Markdown agrees with HTML
    md_fig = render._figure_markdown(papers, labels)
    check("Markdown figure contains study design", "study design" in md_fig)
    for (label, _), tally in zip(series["buckets"], series["counts"]):
        check(f"Markdown figure reports bucket {label}", f"- {label}: {sum(tally.values())}" in md_fig)

    # 4. Fallback: mostly unlabelled (demo.SAMPLE_PAPERS shape)
    sample_cited = [Paper(title=f"Study {i}", abstract="a", year=2020 + i) for i in range(1, 7)]
    sample_labels = {i: "direct" for i in range(1, 7)}
    fb_series = render._figure_series(sample_cited, sample_labels)
    check("mostly unlabelled falls back to year mode", fb_series is not None and fb_series["mode"] == "year")
    fb_html = render._figure_html(sample_cited, sample_labels)
    check("fallback HTML contains Year of publication", "Year of publication" in fb_html)

    # 5. Fallback: single category
    all_trials = [
        Paper(title=f"Treatment {i}: a randomised controlled trial", abstract="a", year=2020 + i)
        for i in range(1, 7)
    ]
    single_series = render._figure_series(all_trials, sample_labels)
    check("single category falls back to year mode", single_series is not None and single_series["mode"] == "year")

    # 6. Table 1 demotes citation counts, adds Design, references keep Cited by
    t_html = render._table_html(papers, labels)
    t_md = render._table_markdown(papers, labels)
    check("Table 1 HTML drops Cited by header", "<th>Cited by</th>" not in t_html)
    check("Table 1 HTML has Design header", "<th>Design</th>" in t_html)
    check("Table 1 Markdown drops Cited by header", "Cited by |" not in t_md)
    check("Table 1 Markdown has Design header", "| Design |" in t_md)

    article = {
        "title": "A Review",
        "abstract": "Abstract [1].",
        "sections": [{"heading": "Introduction", "paragraphs": ["Text [1]."]}],
        "key_points": ["Point [1]."],
        "references": [1, 2, 3, 4, 5, 6],
    }
    rendered = render.render_article(article, papers, "A topic", curation={"relevance": labels})
    rendered_md = render.render_markdown(article, papers, "A topic", curation={"relevance": labels})
    check("render_article HTML reference list keeps Cited by", "ref-cites" in rendered and "Cited by" in rendered)
    check("render_markdown reference list keeps Cited by", "Cited by" in rendered_md)

    # 7. classify_design and paper_design negative controls
    p_proto = Paper(title="Protocol for a prospective cohort study of outcomes", abstract="a")
    check("study protocol classifies as other", classify_design(p_proto) == "other")
    p_survey = Paper(title="Survey of national policy and practice", abstract="a")
    check("plain survey without design keywords classifies as other", classify_design(p_survey) == "other")
    p_rct = Paper(title="A randomised controlled trial of intervention", abstract="a")
    check("randomised controlled trial classifies as trial", classify_design(p_rct) == "trial")
    p_qual = Paper(title="A qualitative interview study", abstract="a")
    check("qualitative study classifies as qualitative", classify_design(p_qual) == "qualitative")
    p_obs = Paper(title="A prospective cohort study", abstract="a")
    check("cohort study classifies as observational", classify_design(p_obs) == "observational")

    # Invariants on label mappings
    check("DESIGN_LABELS and DESIGN_DISPLAY_ORDER keys match",
          set(sources.DESIGN_LABELS) == set(sources.DESIGN_DISPLAY_ORDER))
    for p in (p_proto, p_survey, p_rct, p_qual, p_obs, papers[0]):
        check(f"classify_design returns valid label for {p.title[:20]}",
              classify_design(p) in sources.DESIGN_LABELS)

    # Positives from title
    p_scoping = Paper(title="Coercive practice in acute care: a scoping review", abstract="a")
    check("scoping review classifies as scoping", classify_design(p_scoping) == "scoping")
    p_narrative = Paper(title="A narrative review of interventions", abstract="a")
    check("narrative review classifies as narrative", classify_design(p_narrative) == "narrative")
    p_consensus = Paper(title="Consensus statement on safety planning", abstract="a")
    check("consensus statement classifies as consensus", classify_design(p_consensus) == "consensus")
    p_case_rep = Paper(title="Clozapine toxicity: a case report", abstract="a")
    check("case report classifies as case", classify_design(p_case_rep) == "case")
    p_case_ser = Paper(title="Adverse events in inpatient wards: a case series", abstract="a")
    check("case series classifies as case", classify_design(p_case_ser) == "case")

    # Positives from publication_types alone (with plain titles)
    check("meta-analysis pubType classifies as synthesis",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("meta-analysis",))) == "synthesis")
    check("systematic review pubType classifies as synthesis",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("systematic review",))) == "synthesis")
    check("scoping review pubType classifies as scoping",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("scoping review",))) == "scoping")
    check("case reports pubType classifies as case",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("case reports",))) == "case")
    check("randomized controlled trial pubType classifies as trial",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("randomized controlled trial",))) == "trial")
    # Hyphenated spellings
    check("hyphenated case-report pubType classifies as case",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("case-report",))) == "case")
    check("hyphenated systematic-review pubType classifies as synthesis",
          classify_design(Paper(title="Study on outcomes", abstract="a", publication_types=("systematic-review",))) == "synthesis")

    # Negative controls
    p_case_ctrl = Paper(title="A case-control study of risk factors", abstract="a")
    check("case-control study classifies as observational", classify_design(p_case_ctrl) == "observational")
    p_sys_scoping = Paper(title="A systematic scoping review of care", abstract="a")
    check("systematic scoping review classifies as synthesis", classify_design(p_sys_scoping) == "synthesis")
    p_sys_lit = Paper(title="A systematic literature review of care", abstract="a")
    check("systematic literature review classifies as synthesis", classify_design(p_sys_lit) == "synthesis")
    p_proto_narr = Paper(title="Protocol for a narrative review of care", abstract="a")
    check("protocol for narrative review classifies as other", classify_design(p_proto_narr) == "other")
    p_bare_rev = Paper(title="Care models in psychiatry", abstract="a", publication_types=("review",))
    check("bare review pubType classifies as other", classify_design(p_bare_rev) == "other")

    # Fetch order (paper_design returns 'other' for scoping, narrative, consensus, case)
    for p in (p_scoping, p_narrative, p_consensus, p_case_rep, p_case_ser):
        check(f"paper_design demotes {classify_design(p)} to other", paper_design(p) == "other")
        check(f"paper_design maps {classify_design(p)} to DESIGN_ORDER", paper_design(p) in sources.DESIGN_ORDER)

    for p in (p_proto, p_survey, p_rct, p_qual, p_obs, papers[0]):
        check(f"paper_design maps {classify_design(p)} to DESIGN_ORDER",
              paper_design(p) in ("synthesis", "trial", "other"))

    # Table 1 prints 'Other' instead of '—'
    survey_rows = render._table_rows([p_survey], {1: "direct"})
    check("Table 1 prints Other instead of dash for unclassifiable paper", survey_rows[0]["design"] == "Other")
    all_rows = render._table_rows(papers, labels)
    check("Table 1 dash not in design column for mixed set", all(r["design"] != "—" for r in all_rows))
    check("Table 1 HTML contains Other cell", "<td>Other</td>" in render._table_html([p_survey], {1: "direct"}))

    # Fig. 1 width scaling
    nine_papers = [
        Paper(title="Systematic review of care", abstract="a", year=2020),
        Paper(title="Randomised controlled trial of care", abstract="a", year=2021),
        Paper(title="Prospective cohort study of care", abstract="a", year=2022),
        Paper(title="Qualitative interview study of care", abstract="a", year=2023),
        Paper(title="Adverse events: a case report", abstract="a", year=2024),
        Paper(title="Scoping review of care", abstract="a", year=2025),
        Paper(title="Narrative review of care", abstract="a", year=2026),
        Paper(title="Consensus statement on care", abstract="a", year=2026),
        Paper(title="General overview of services", abstract="a", year=2026),
    ]
    nine_labels = {i: "direct" for i in range(1, 10)}
    nine_fig = render._figure_html(nine_papers, nine_labels)
    check("nine-bucket figure widens viewBox to 860", 'viewBox="0 0 860 250"' in nine_fig)
    check("five-bucket figure keeps viewBox at 660", 'viewBox="0 0 660 250"' in html_fig)


def html_escaped(value: str, haystack: str) -> bool:
    import html as _h
    return value in haystack or _h.escape(value) in haystack


def test_failed_style_gate_is_visible_in_the_article() -> None:
    """A draft that failed the prose check must not look like one that passed.

    The gate ran, the revision was attempted, and if it did not clear the errors
    the article shipped with no sign of it — a reader had no way to tell a clean
    draft from one carrying five unfixed errors (issue #53). The house style puts
    warnings in the Limitations paragraph rather than in callout boxes, so that
    is where this goes, next to the unverified-figure warning.
    """
    from articlegen import render
    from articlegen.sources import Paper

    papers = [Paper(title="P1", abstract="a", year=2024, doi="10.1/a")]
    counts = {"direct": 1, "related": 0, "tangential": 0}
    clean = {"issues": [], "stats": {}}
    failed = {"issues": [
        {"rule": "echoed-abstract", "severity": "error", "where": "Introduction",
         "detail": "d", "excerpt": ""},
        {"rule": "echoed-abstract", "severity": "error", "where": "key points",
         "detail": "d", "excerpt": ""},
        {"rule": "bundled-citations", "severity": "error", "where": "whole article",
         "detail": "d", "excerpt": ""},
        {"rule": "under-length", "severity": "warning", "where": "whole article",
         "detail": "d", "excerpt": ""},
    ], "stats": {}}

    ok = render._assessment_html(papers, counts, {"unverified": []}, clean)
    check("a clean draft says nothing about the check",
          "automated check of the writing" not in ok)

    bad = render._assessment_html(papers, counts, {"unverified": []}, failed)
    check("a failed draft says so", "automated check of the writing" in bad)
    check("in the reader's terms, not the rule's",
          "repeats the abstract rather than adding to it" in bad
          and "echoed-abstract" not in bad)
    check("the same rule twice is one fault to a reader",
          bad.count("repeats the abstract") == 1)
    check("faults are joined readably", " and " in bad)
    check("warnings do not trigger it",
          "under-length" not in bad and "under length" not in bad)
    check("it says the revision was tried", "revision was attempted" in bad)
    check("and lands in Limitations, not a callout box",
          "Limitations." in bad and "⚠" not in bad)

    md = "\n".join(render._assessment_markdown(papers, counts, {"unverified": []}, failed))
    check("markdown carries the same warning", "automated check of the writing" in md)

    # An absent report is not the same as a clean one, but it must not crash or
    # invent a warning — legacy drafts have no style report at all.
    legacy = render._assessment_html(papers, counts, {"unverified": []}, None)
    check("a missing report warns about nothing",
          "automated check of the writing" not in legacy)

    # The pipeline must actually hand it over, or none of this is reachable.
    import inspect
    from articlegen import cli, web
    for name, src in (("cli", inspect.getsource(cli.cmd_draft) + inspect.getsource(cli._write_draft_files)),
                      ("web", inspect.getsource(web.ArticleGenHandler._handle_draft))):
        check(f"{name} passes the style report to the renderer",
              "draft.style_report" in src)


def test_only_sendable_defects_brand_the_page() -> None:
    """The working-draft Limitations line prints only for sendable-blocking defects.

    Prose nits (recycled-phrasing, repeated-opener) and warnings (under-length)
    stay in the CLI log and in style_report; they do not brand the page (#169).
    Clinical directives, surviving substance rules, and residual unverified/misattributed
    figures do brand the page.
    """
    from articlegen import render, style
    from articlegen.sources import Paper

    papers = [Paper(title="P1", abstract="a", year=2024, doi="10.1/a")]
    counts = {"direct": 1, "related": 0, "tangential": 0}
    clean_report = {"issues": [], "stats": {}}
    clean_verification = {"unverified": [], "misattributed": []}

    def limitations_for(style_report=None, verification=None):
        return " ".join(
            render._assessment_paragraphs(
                papers, counts, verification or clean_verification, style_report or clean_report
            )["limitations"]
        )

    # 1. Nits do not brand. A style_report whose only errors are recycled-phrasing
    # and repeated-opener produces limitations containing neither "working draft"
    # nor "journal prose conventions".
    nits_report = {
        "issues": [
            {"rule": "recycled-phrasing", "severity": "error", "where": "whole article",
             "detail": "recycled text", "excerpt": "sample text"},
            {"rule": "repeated-opener", "severity": "error", "where": "whole article",
             "detail": "repeated opener", "excerpt": "The study found"},
        ],
        "stats": {},
    }
    nits_limitations = limitations_for(style_report=nits_report)
    check("nits do not brand the page with working draft",
          "working draft" not in nits_limitations)
    check("nits do not print the journal prose conventions sentence",
          "journal prose conventions" not in nits_limitations)

    # 2. A clinical directive brands. A report with a clinical-directive error produces
    # both the prose-check sentence ("instructs the reader on treatment") and
    # "working draft rather than a finished review".
    directive_report = {
        "issues": [
            {"rule": "clinical-directive", "severity": "error", "where": "Introduction",
             "detail": "clinical directive detail", "excerpt": "titrate upward"},
        ],
        "stats": {},
    }
    directive_limitations = limitations_for(style_report=directive_report)
    check("clinical directive prints prose-check sentence",
          "instructs the reader on treatment" in directive_limitations)
    check("clinical directive brands as working draft",
          "working draft rather than a finished review" in directive_limitations)

    # 3. A surviving substance failure brands. Same with too-few-sections.
    substance_report = {
        "issues": [
            {"rule": "too-few-sections", "severity": "error", "where": "whole article",
             "detail": "too few sections", "excerpt": ""},
        ],
        "stats": {},
    }
    substance_limitations = limitations_for(style_report=substance_report)
    check("surviving substance failure prints prose-check sentence",
          "covers the topic in fewer sections" in substance_limitations)
    check("surviving substance failure brands as working draft",
          "working draft rather than a finished review" in substance_limitations)

    # 4. A mixed report names only the blocking fault. clinical-directive +
    # recycled-phrasing together -> the sentence names the directive and does not
    # contain "reuses phrasing between sections".
    mixed_report = {
        "issues": [
            {"rule": "clinical-directive", "severity": "error", "where": "Introduction",
             "detail": "d", "excerpt": ""},
            {"rule": "recycled-phrasing", "severity": "error", "where": "whole article",
             "detail": "d", "excerpt": ""},
        ],
        "stats": {},
    }
    mixed_limitations = limitations_for(style_report=mixed_report)
    check("mixed report names the blocking fault",
          "instructs the reader on treatment" in mixed_limitations)
    check("mixed report does not name the nit",
          "reuses phrasing between sections" not in mixed_limitations)
    check("mixed report brands as working draft",
          "working draft rather than a finished review" in mixed_limitations)

    # 5. Residual figures brand, with no style errors at all. style_report with no
    # errors, verification={"unverified": ["42%"]} -> the unverified sentence is
    # still there and the working-draft sentence prints. Same for {"misattributed": ["18%"]}.
    unverified_limitations = limitations_for(
        style_report=clean_report, verification={"unverified": ["42%"], "misattributed": []}
    )
    check("unverified figures produce the unverified sentence",
          "could not be located" in unverified_limitations and "42%" in unverified_limitations)
    check("unverified figures brand as working draft",
          "working draft rather than a finished review" in unverified_limitations)

    misattributed_limitations = limitations_for(
        style_report=clean_report, verification={"unverified": [], "misattributed": ["18%"]}
    )
    check("misattributed figures produce the misattributed sentence",
          "other than the one its sentence credits" in misattributed_limitations and "18%" in misattributed_limitations)
    check("misattributed figures brand as working draft",
          "working draft rather than a finished review" in misattributed_limitations)

    # 6. A clean draft says nothing. No style errors, verification={"unverified": [], "misattributed": []}
    # -> neither string appears anywhere in the limitations.
    clean_limitations = limitations_for(style_report=clean_report, verification=clean_verification)
    check("clean draft produces no working draft branding",
          "working draft" not in clean_limitations)
    check("clean draft produces no prose check sentence",
          "journal prose conventions" not in clean_limitations)

    # 7. under-length stays out. It is severity "warning", so a report carrying it
    # as a warning brands nothing — assert that, and assert "under-length" in style.SUBSTANCE_RULES
    # so the exemption in SENDABLE_BLOCKING_RULES is still subtracting a name that exists.
    under_length_report = {
        "issues": [
            {"rule": "under-length", "severity": "warning", "where": "whole article",
             "detail": "short", "excerpt": ""},
        ],
        "stats": {},
    }
    under_length_limitations = limitations_for(style_report=under_length_report)
    check("under-length warning does not brand as working draft",
          "working draft" not in under_length_limitations)
    check("under-length warning does not print prose check sentence",
          "journal prose conventions" not in under_length_limitations)
    check("under-length is in style.SUBSTANCE_RULES",
          "under-length" in style.SUBSTANCE_RULES)

    # 8. The exemptions are still real names. {"recycled-phrasing", "repeated-opener",
    # "under-length"} <= style.SUBSTANCE_RULES, and neither of the first two is in
    # render.SENDABLE_BLOCKING_RULES.
    exemptions = {"recycled-phrasing", "repeated-opener", "under-length"}
    check("exemptions are all in style.SUBSTANCE_RULES",
          exemptions <= style.SUBSTANCE_RULES)
    check("recycled-phrasing is not in render.SENDABLE_BLOCKING_RULES",
          "recycled-phrasing" not in render.SENDABLE_BLOCKING_RULES)
    check("repeated-opener is not in render.SENDABLE_BLOCKING_RULES",
          "repeated-opener" not in render.SENDABLE_BLOCKING_RULES)

    # 9. The rules still fire and still buy a revision. Nothing in style.py changed:
    # assert "recycled-phrasing" in style.SUBSTANCE_RULES and that style.revision_brief
    # on a recycled-phrasing report still asks for the fix (the brief text contains the rule's detail).
    check("recycled-phrasing remains in style.SUBSTANCE_RULES",
          "recycled-phrasing" in style.SUBSTANCE_RULES)
    brief = style.revision_brief({
        "issues": [
            {"rule": "recycled-phrasing", "severity": "error", "where": "whole article",
             "detail": "reused sentence across sections", "excerpt": "verbatim text repeated here"},
        ],
        "stats": {},
    })
    check("style.revision_brief asks for recycled-phrasing fix",
          "reused sentence across sections" in brief)
    check("style.revision_brief inverts to request source material for substance rules",
          "SOURCES below" in brief)


def test_evidence_assessment_is_wholly_deterministic() -> None:
    """Nothing countable in the Evidence assessment may be model-written.

    The section exists to state honestly how good the evidence is, and the
    model's `evidence_note` repeatedly contradicted the counts printed two lines
    above it: one article claimed "13 out of 20 sources" beside a computed 9;
    another claimed "5 sources", "the majority is related" and "some are
    tangential" against a computed 3 cited, all direct, none related, none
    background. The field is gone from the schema (issue #54).

    Legacy drafts that still carry one must render without it rather than
    reprinting a tally that may be wrong.
    """
    from articlegen import render, writer
    from articlegen.sources import Paper

    check("evidence_note is out of the schema",
          "evidence_note" not in writer._ARTICLE_SCHEMA["properties"])
    check("and is not required", "evidence_note" not in writer._ARTICLE_SCHEMA["required"])
    check("the writer is told not to state counts",
          "Do NOT state counts or tallies" in writer._WRITER_SYSTEM)

    papers = [Paper(title="P1", abstract="a", year=2024, doi="10.1/a")]
    counts = {"direct": 1, "related": 0, "tangential": 0}
    legacy = render._assessment_html(papers, counts, {"unverified": []})
    check("a legacy note is not rendered", "13 out of 20" not in legacy)
    check("the deterministic opening survives", "Of the 1 source cited" in legacy)

    # Number agreement: the section that exists to be precise printed
    # "1 address the review question directly".
    check("singular subject takes a singular verb", "1 addresses the review question" in legacy)
    check("singular 'source', not 'sources'", "1 sources cited" not in legacy)
    check("singular related reads 'is related'",
          "0 are related" in legacy)

    one_each = render._assessment_html(
        papers, {"direct": 0, "related": 1, "tangential": 1}, {"unverified": []})
    check("one related reads 'is related'", "1 is related" in one_each)
    check("one background reads 'provides'", "1 provides background only" in one_each)

    plural = render._assessment_html(
        papers * 3, {"direct": 2, "related": 3, "tangential": 0}, {"unverified": []})
    check("plurals still read correctly",
          "Of the 3 sources cited, 2 address" in plural and "3 are related" in plural)

    md = "\n".join(render._assessment_markdown(papers, counts, {"unverified": []}))
    check("markdown agrees with the html", "Of the 1 source cited, 1 addresses" in md)


def test_house_style_is_fixed_not_a_preference() -> None:
    """There is one register, and the front end must not offer alternatives to it.

    The Tone selector used to fold a choice into the `style_note` the writer is
    given, defaulting to "Engaging Science Journalism (Wired/Quanta style)" —
    while `style.py` deterministically bans second person, contractions,
    boosters and rhetorical questions, which is precisely that register. Three
    of its four options asked for prose the checker then rejected, so the
    setting could only make the output worse.

    `docs/journal-style.md` defines the register and `style.py` enforces it, so
    the tone is part of the house style rather than something a reader picks.
    The selector is gone and the label is a constant.

    Article length and evidence depth were removed for the same reason: every
    option except the house artefact + strict empirical asked for prose the
    substance rules then failed. All three are constants now; only the output
    language is still selectable. The length constant is the briefing, not a
    journal Review — `--long` is CLI-only and is not a front-end preference.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "index.html")
    html = open(path, encoding="utf-8").read()

    check("the tone selector is gone", 'id="prefTone"' not in html)
    check("and no tone map survives it", "toneMap" not in html)
    for banned in ("Wired", "Quanta", "ELI5", "Executive Briefing"):
        check(f"no alternative register is offered: {banned}", banned not in html)

    # The writer is still told the register explicitly; it just isn't selectable.
    check("the tone label is a single constant", html.count("const TONE_LABEL") == 1)
    check("and it names the academic register",
          "Formal Academic & Technical" in html)
    check("styleGuidance still sends a tone", "'Tone: ' + p.toneLabel" in html)
    check("which now resolves to the constant", "toneLabel: TONE_LABEL" in html)

    # Length and evidence depth went the same way and for the same reason: the
    # short lengths and the narrative/balanced depths asked for prose the
    # substance rules in style.py then failed. One combination survives, so it
    # is a constant rather than a selector.
    for gone in ("prefLength", "prefDepth"):
        check(f"the {gone} selector is gone", f'id="{gone}"' not in html)
    for label, value in (("LENGTH_LABEL", "Evidence briefing"),
                         ("DEPTH_LABEL", "Strict Empirical Focus")):
        check(f"{label} is a single constant", html.count(f"const {label}") == 1)
        check(f"and it names {value}", value in html)
    check("styleGuidance still sends a length", "'Length: ' + p.lengthLabel" in html)
    check("and an evidence focus", "'Evidence focus: ' + p.depthLabel" in html)

    # Language is still the reader's choice.
    check("prefLang still offered", 'id="prefLang"' in html)


def test_register_rules_are_scoped_to_the_synthesis_voice() -> None:
    """The register rules model one voice, and the corpus proves which.

    `tests/style_corpus.json` is 20 high-cited abstracts stratified across
    article type (primary / systematic review / narrative review) and domain
    (clinical psychiatry / neuroscience / health services), 20 distinct journals.

    Each is labelled by the voice it is written in. A primary trial report says
    "we randomly assigned patients" because those authors ran the trial; a
    synthesis speaks about other people's work. `articlegen` only ever writes the
    second, so banning the first is correct — but it means primary-research
    abstracts are *not* a negative control for the first-person rule, which is
    the trap the older corpus had to paper over with per-entry exceptions.

    The separation is clean, and that is the assertion: every investigator-voice
    entry fires a register rule, no synthesis-voice entry does.
    """
    import json

    from articlegen.style import check_style, errors

    register = {"first-person", "second-person", "contraction", "booster",
                "overclaim", "rhetorical-question", "exclamation"}
    path = os.path.join(os.path.dirname(__file__), "style_corpus.json")
    corpus = json.load(open(path, encoding="utf-8"))

    check("corpus is 20 articles", len(corpus) == 20)
    check("every type/domain cell is represented",
          len({(e["type"], e["domain"]) for e in corpus}) == 9)
    check("drawn from many journals, not one house style",
          len({e["journal"] for e in corpus}) >= 15)

    mismatched = []
    for entry in corpus:
        article = {"sections": [{"heading": "Introduction",
                                 "paragraphs": [entry["abstract"]]}]}
        fired = sorted({i["rule"] for i in errors(check_style(article))} & register)
        if fired != entry["expect_register_errors"]:
            mismatched.append((entry["title"][:48], fired, entry["expect_register_errors"]))
    for title, fired, expected in mismatched:
        print(f"      {title}: fired {fired}, expected {expected}")
    check("register rules match expectations on all 20", not mismatched)

    synthesis = [e for e in corpus if e["register"] == "synthesis"]
    investigator = [e for e in corpus if e["register"] == "investigator"]
    check("the corpus carries both voices", synthesis and investigator)
    check("no synthesis-voice abstract trips a register rule",
          all(not e["expect_register_errors"] for e in synthesis))
    check("every investigator-voice abstract does",
          all(e["expect_register_errors"] for e in investigator))


def test_hedging_floor_is_calibrated_against_body_prose() -> None:
    """The hedging floor is validated — against body prose, not abstracts.

    #56 recorded that `MIN_HEDGES_PER_SENTENCE = 0.20` had never been checked
    against the register it polices: abstracts run at a median of 0.031, but
    they compress and assert, and 18 of the 20 in `style_corpus.json` are too
    short for the density gate to apply at all.

    `body_prose_measurements.json` closes that gap: the body paragraphs of 18
    open-access reviews (Lancet, Lancet Psychiatry, BMJ, PLoS, Frontiers…),
    abstract and references excluded. They hedge at a **median of 0.22** — the
    0.20 floor is very nearly the median of published review prose. The guess
    was right; it had simply been compared with the wrong text type.

    Only the measurements are stored, not the articles: the statistics are the
    evidence, and the repo has no business carrying other people's full texts.
    """
    import json
    import statistics

    from articlegen import style

    path = os.path.join(os.path.dirname(__file__), "body_prose_measurements.json")
    corpus = json.load(open(path, encoding="utf-8"))
    check("body-prose corpus is a usable size", len(corpus) >= 15)
    check("every entry is identifiable", all(e.get("pmcid") for e in corpus))
    check("drawn from several journals", len({e["journal"] for e in corpus}) >= 10)
    check("covers all three domains", len({e["domain"] for e in corpus}) == 3)

    hedges = [e["hps"] for e in corpus]
    median = statistics.median(hedges)
    check(f"body prose hedges near the floor (median {median:.3f})",
          abs(median - style.MIN_HEDGES_PER_SENTENCE) < 0.05)
    check("which is why the floor stands", style.MIN_HEDGES_PER_SENTENCE == 0.20)

    # Unlike abstracts, real body prose is long enough to be judged at all.
    check("all of it clears the density gate",
          all(e["sentences"] > style.MIN_SENTENCES_FOR_DENSITY
              and e["body_words"] > style.MIN_WORDS_FOR_DENSITY for e in corpus))

    # And it uses many distinct hedges, which is what hedge-monotony assumes.
    distinct = [e["distinct"] for e in corpus]
    check(f"body prose varies its hedges (median {statistics.median(distinct):.0f} distinct)",
          statistics.median(distinct) >= 8)
    check("so demanding variety is fair once there are enough hedges",
          style.MIN_HEDGES_FOR_MONOTONY >= 8)

    passive = [e["passive"] for e in corpus]
    check("the passive threshold clears review prose too",
          sum(1 for p in passive if p > style.MAX_PASSIVE_RATIO) == 0)


def test_density_thresholds_are_documented_against_the_corpus() -> None:
    """The density thresholds are house preferences, not measurements — say so.

    Measured over the 20-article corpus: hedging runs at a median of 0.03 markers
    per sentence and 17 of 20 sit below the 0.20 floor, half using no hedge at
    all. `docs/journal-style.md` §15 cites corpus work reporting one marker every
    two to three sentences, but that figure is for *whole research articles*,
    where the Discussion carries most of the hedging — not for abstracts, which
    compress and assert.

    So this test does not assert that the floor is right. It pins the measured
    distribution, so that changing a threshold without re-measuring fails here,
    and records that abstracts cannot calibrate a rule aimed at body prose (18 of
    20 are too short for the density gate to even apply).
    """
    import json
    import statistics

    from articlegen import style

    path = os.path.join(os.path.dirname(__file__), "style_corpus.json")
    corpus = json.load(open(path, encoding="utf-8"))
    hedges = [e["measured"]["hedges_per_sentence"] for e in corpus]
    passive = [e["measured"]["passive_ratio"] for e in corpus]
    sentence_words = [e["measured"]["mean_sentence_words"] for e in corpus]

    check("published abstracts hedge far below the floor",
          statistics.median(hedges) < style.MIN_HEDGES_PER_SENTENCE / 2)
    check("most of the corpus would fail the hedging floor",
          sum(1 for h in hedges if h < style.MIN_HEDGES_PER_SENTENCE) >= 15)
    check("and most are too short for the density gate to apply",
          sum(1 for e in corpus
              if e["measured"]["sentences"] > style.MIN_SENTENCES_FOR_DENSITY
              and e["measured"]["words"] > style.MIN_WORDS_FOR_DENSITY) <= 4)

    # These two are calibrated: real prose sits comfortably inside them, so they
    # flag genuine outliers rather than the norm.
    check("the passive ratio threshold clears real prose",
          sum(1 for p in passive if p > style.MAX_PASSIVE_RATIO) <= 3)
    check("mean sentence length matches the 15-30 guidance",
          sum(1 for w in sentence_words if 15 <= w <= 30) >= 17)

    # A measured corpus is the guard the docs promise; keep the recorded stats
    # honest by recomputing one of them.
    entry = corpus[0]
    report = style.check_style(
        {"sections": [{"heading": "Introduction", "paragraphs": [entry["abstract"]]}]})
    check("recorded stats still match a fresh run",
          round(report["stats"]["passive_ratio"], 3) == entry["measured"]["passive_ratio"])


def test_statistic_verification() -> None:
    from articlegen.verify import check_statistics
    from articlegen.sources import Paper

    papers = [Paper(title="P1", abstract="the effect was 0.53 overall and 12% responded", year=2010)]
    article = {
        "abstract": "s", "evidence_note": "",
        "featured_study": {"source_index": 1, "why": "", "method": "", "results": "RR 4.91"},
        "sections": [{"heading": "H", "paragraphs": ["fell 0.53 [1] but SMD -0.90 [1]"]}],
        "key_points": ["12% responded [1]"], "references": [1],
    }
    v = check_statistics(article, papers)
    check("flags absent figure 4.91", "4.91" in v["unverified"])
    check("flags absent figure -0.90", "-0.90" in v["unverified"])
    check("passes present figure 0.53", "0.53" not in v["unverified"])
    check("passes present figure 12%", "12%" not in v["unverified"])

    legacy = {
        "standfirst": "SMD -0.77 headline", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": ["x"], "pull_quote": "and 8.88 more"}],
        "key_takeaways": ["12% responded [1]"], "references": [1],
    }
    v_legacy = check_statistics(legacy, papers)
    check("pre-journal-format drafts are still checked",
          "-0.77" in v_legacy["unverified"] and "8.88" in v_legacy["unverified"])

    # -- attribution: a cited sentence is checked against its own sources, and
    #    a hit in some other source is a distinct failure, not a pass (#101) --
    two = [
        Paper(title="P1", abstract="light therapy was tested; the effect was 0.53", year=2010),
        Paper(title="P2", abstract="a separate trial reported SMD 2.71", year=2012),
    ]
    swapped = {
        "abstract": "", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": [
            "The trial found SMD 2.71 [1].",
            "A related trial found 0.53 [2].",
            "Neither reported 6.66 [1].",
        ]}],
        "key_points": [], "references": [1, 2],
    }
    v_swap = check_statistics(swapped, two)
    check("a figure real in another source is not a pass",
          "2.71" not in v_swap["unverified"] and "0.53" not in v_swap["unverified"])
    check("it is reported as misattributed instead",
          "2.71" in v_swap["misattributed"] and "0.53" in v_swap["misattributed"])
    check("a figure in no source at all stays unverified",
          "6.66" in v_swap["unverified"] and "6.66" not in v_swap["misattributed"])

    uncited = {
        "abstract": "An overall effect of 2.71 was reported.", "evidence_note": "",
        "sections": [], "key_points": [], "references": [],
    }
    v_uncited = check_statistics(uncited, two)
    check("a sentence citing nothing has no attribution to break",
          not v_uncited["misattributed"] and not v_uncited["unverified"])

    # -- integers carrying a clinical unit are checked; bare ones are not ----
    clinical = [Paper(title="P", abstract="Participants (n = 441) received 7,000 lux for "
                                          "30 minutes; serum was 50 ng/mL in 109 cases.")]
    quantities = {
        "abstract": "", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": [
            "Light at 7,000 lux for 30 minutes reached 50 ng/mL in 109 cases among "
            "441 participants [1].",
            "A second protocol used 9,500 lux for 90 minutes [1].",
        ]}],
        "key_points": [], "references": [1],
    }
    v_q = check_statistics(quantities, clinical)
    for grounded in ("7,000 lux", "30 minutes", "50 ng/mL", "109 cases", "441 participants"):
        check(f"clinical quantity is checked and passes: {grounded}",
              grounded not in v_q["unverified"] and v_q["total"] >= 5)
    check("absent clinical quantities are flagged",
          "9,500 lux" in v_q["unverified"] and "90 minutes" in v_q["unverified"])

    bare = {"abstract": "The protocol had 3 stages and ran in 2019.", "evidence_note": "",
            "sections": [], "key_points": [], "references": []}
    check("an integer with no clinical unit is not a figure",
          check_statistics(bare, clinical)["total"] == 0)

    # -- title-only figure is grounded (#189) -------------------------------
    title_only_papers = [
        Paper(title="Seclusion fell 37% after the intervention", abstract="No numbers here.", year=2021)
    ]
    title_art = {
        "abstract": "", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": ["Seclusion fell 37% [1]."]}],
        "key_points": [], "references": [1],
    }
    v_title = check_statistics(title_art, title_only_papers)
    check("a figure stated in a paper's title is grounded",
          "37%" not in v_title["unverified"] and "37%" not in v_title["misattributed"])

    # -- hyphenated range is ONE figure, not two (#189) ---------------------
    range_papers = [
        Paper(title="P1", abstract="the interval was 4.4-5.2 and another was from 4.4 to 5.2", year=2020),
        Paper(title="P2", abstract="the reduction was from -0.38 to -0.12", year=2022),
    ]
    grounded_range_art = {
        "abstract": "", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": [
            "The interval was 4.4-5.2 [1].",
            "A second interval was 4.4–5.2 [1].",
            "The negative reduction was -0.38--0.12 [2].",
        ]}],
        "key_points": [], "references": [1, 2],
    }
    v_gr = check_statistics(grounded_range_art, range_papers)
    check("grounded hyphenated range passes", "4.4-5.2" not in v_gr["unverified"])
    check("en-dash range against hyphenated source passes", "4.4–5.2" not in v_gr["unverified"])
    check("negative range against 'from to' source passes", "-0.38--0.12" not in v_gr["unverified"])
    check("single sentence with range has total == 1",
          check_statistics({"abstract": "", "evidence_note": "",
                            "sections": [{"heading": "H", "paragraphs": ["the interval was 4.4-5.2 [1]"]}],
                            "key_points": [], "references": [1]}, range_papers)["total"] == 1)

    absent_range_art = {
        "abstract": "", "evidence_note": "",
        "sections": [{"heading": "H", "paragraphs": ["The interval was 9.1-9.8 [1]."]}],
        "key_points": [], "references": [1],
    }
    v_ar = check_statistics(absent_range_art, range_papers)
    check("absent hyphenated range is flagged as one figure", "9.1-9.8" in v_ar["unverified"])
    check("hyphenated range is not split into a negative number",
          "-9.8" not in v_ar["unverified"] and "9.8" not in v_ar["unverified"])


def test_flagged_figures_buy_one_revision() -> None:
    """A draft with unverified figures buys one revision pass, a clean draft buys none.

    `enforce_statistics` runs `check_statistics` and, if any figures are unverified
    or misattributed, asks the model once (`MAX_STATISTIC_PASSES = 1`) to drop the
    figure, state it in words, or move the citation. The model is forbidden from
    inventing numbers or sources, enforced deterministically: a revision that
    increases `total` is refused (#189).
    """
    from articlegen import pipeline, verify
    from articlegen.sources import Paper

    papers = [Paper(title="P1", abstract="the effect was 0.53", year=2020)]
    article = {
        "form": "briefing",
        "question": "Q?",
        "answer": "A.",
        "findings": ["Finding one with 4.91 [1]."],
        "unknowns": ["Unknown."],
        "references": [1],
    }

    # 1. Clean first write buys nothing (no LLM call, original article object returned).
    calls = {"check": 0, "revise": 0}

    def fake_check_clean(a, p):
        calls["check"] += 1
        return {"unverified": [], "misattributed": [], "total": 4, "details": []}

    def fake_revise_never(a, brief, **kwargs):
        calls["revise"] += 1
        return dict(a)

    saved_check = pipeline.check_statistics
    saved_revise = pipeline.revise_statistics
    saved_style = pipeline.check_style
    try:
        pipeline.check_statistics = fake_check_clean
        pipeline.revise_statistics = fake_revise_never
        out_art, out_v, out_st = pipeline.enforce_statistics(article, papers)
        check("clean first write costs zero revision calls", calls["revise"] == 0)
        check("clean first write returns original article", out_art is article)
        check("clean first write returns clean verification", not out_v["unverified"])
    finally:
        pipeline.check_statistics = saved_check
        pipeline.revise_statistics = saved_revise
        pipeline.check_style = saved_style

    # 2. Flags buy exactly one call (MAX_STATISTIC_PASSES == 1).
    check("MAX_STATISTIC_PASSES is 1", pipeline.MAX_STATISTIC_PASSES == 1)

    calls = {"check": 0, "revise": 0}

    def fake_check_flags(a, p):
        calls["check"] += 1
        return {"unverified": ["4.91"], "misattributed": ["2.71"], "total": 2,
                "details": [{"figure": "4.91", "kind": "unverified", "sentence": "Finding one with 4.91 [1].", "cited": [1]}]}

    def fake_revise_once(a, brief, **kwargs):
        calls["revise"] += 1
        return dict(a)

    try:
        pipeline.check_statistics = fake_check_flags
        pipeline.revise_statistics = fake_revise_once
        pipeline.enforce_statistics(article, papers)
        check("flags buy exactly one revision call", calls["revise"] == 1)
    finally:
        pipeline.check_statistics = saved_check
        pipeline.revise_statistics = saved_revise
        pipeline.check_style = saved_style

    # 3. A revision that adds a number is refused (total increases).
    calls = {"check": 0, "revise": 0}
    logs = []

    def fake_check_seq_increase(a, p):
        calls["check"] += 1
        if calls["check"] == 1:
            return {"unverified": ["4.91"], "misattributed": [], "total": 1,
                    "details": [{"figure": "4.91", "kind": "unverified", "sentence": "Finding one with 4.91 [1].", "cited": [1]}]}
        # Revision fixed 4.91, but added 2 new numbers: total is now 2 > 1
        return {"unverified": [], "misattributed": [], "total": 2, "details": []}

    try:
        pipeline.check_statistics = fake_check_seq_increase
        pipeline.revise_statistics = fake_revise_once
        out_art, out_v, _ = pipeline.enforce_statistics(article, papers, log=logs.append)
        check("a revision that introduces new numbers is rejected", out_art is article)
        check("rejection log mentions new numbers", any("introduced new numbers" in m for m in logs))
    finally:
        pipeline.check_statistics = saved_check
        pipeline.revise_statistics = saved_revise
        pipeline.check_style = saved_style

    # 4. A revision that fixes flags is accepted, and style report is recomputed.
    calls = {"check": 0, "revise": 0, "style": 0}

    revised_art = dict(article, findings=["Finding one [1]."])

    def fake_check_seq_success(a, p):
        calls["check"] += 1
        if calls["check"] == 1:
            return {"unverified": ["4.91"], "misattributed": [], "total": 1,
                    "details": [{"figure": "4.91", "kind": "unverified", "sentence": "Finding one with 4.91 [1].", "cited": [1]}]}
        return {"unverified": [], "misattributed": [], "total": 0, "details": []}

    def fake_revise_success(a, brief, **kwargs):
        calls["revise"] += 1
        return revised_art

    def fake_check_style(a, **kw):
        calls["style"] += 1
        return {"issues": [], "stats": {"recomputed": True}}

    try:
        pipeline.check_statistics = fake_check_seq_success
        pipeline.revise_statistics = fake_revise_success
        pipeline.check_style = fake_check_style
        out_art, out_v, out_st = pipeline.enforce_statistics(article, papers)
        check("revision is accepted", out_art == revised_art)
        check("verification is updated to clean", len(out_v["unverified"]) == 0)
        check("check_style was called to recompute style report", calls["style"] == 1)
        check("style report is recomputed one", out_st.get("stats", {}).get("recomputed") is True)
    finally:
        pipeline.check_statistics = saved_check
        pipeline.revise_statistics = saved_revise
        pipeline.check_style = saved_style

    # 5. A raising revise_statistics keeps the draft and returns original verification.
    def fake_revise_raise(a, brief, **kwargs):
        raise RuntimeError("LLM failure")

    try:
        pipeline.check_statistics = fake_check_flags
        pipeline.revise_statistics = fake_revise_raise
        out_art, out_v, _ = pipeline.enforce_statistics(article, papers)
        check("raising revision keeps the original draft", out_art is article)
        check("raising revision keeps original verification", "4.91" in out_v["unverified"])
    finally:
        pipeline.check_statistics = saved_check
        pipeline.revise_statistics = saved_revise
        pipeline.check_style = saved_style

    # 6. The brief names the three fixes and forbids new numbers.
    sample_v = {
        "unverified": ["4.91"],
        "misattributed": ["2.71"],
        "total": 2,
        "details": [
            {"figure": "4.91", "kind": "unverified", "sentence": "Trial found RR 4.91 [1].", "cited": [1]},
            {"figure": "2.71", "kind": "misattributed", "sentence": "Trial found SMD 2.71 [2].", "cited": [2]},
        ],
    }
    brief = verify.revision_brief(sample_v)
    check("brief contains unverified figure", "4.91" in brief)
    check("brief contains misattributed figure", "2.71" in brief)
    check("brief contains unverified sentence", "Trial found RR 4.91 [1]." in brief)
    check("brief contains misattributed sentence", "Trial found SMD 2.71 [2]." in brief)
    check("brief mentions deleting figure", "Delete the figure" in brief or "delete" in brief.lower())
    check("brief explicitly forbids new numbers", "MUST NOT introduce any" in brief or "no new number" in brief.lower())


def test_ranking() -> None:
    from articlegen.sources import Paper, _rank_score

    terms = {"schizophrenia", "light"}
    on_topic = Paper(title="Light therapy in schizophrenia", abstract="light schizophrenia trial", year=2020, citation_count=50)
    famous = Paper(title="A review of depression", abstract="depression mood", year=2011, citation_count=5000)
    check("on-topic outranks famous off-topic", _rank_score(on_topic, terms) > _rank_score(famous, terms))


def _sample_draft():
    from articlegen.sources import Paper

    papers = [
        Paper(title=f"Study {i}", abstract="a", year=2000 + i, authors=[f"Ann A{'bcdef'[i - 1]}"],
              venue=f"Journal {i}", citation_count=100 * i, doi=f"10/{i}")
        for i in range(1, 6)
    ]
    article = {
        "title": "Bright light therapy in schizophrenia",
        "abstract": "A summary paragraph of the evidence.",
        "keywords": ["schizophrenia", "light therapy"],
        "evidence_note": "Only one source is directly on schizophrenia [1].",
        "featured_study": {"source_index": 2, "why": "Best trial.", "method": "RCT, n=40.",
                           "results": "Improved."},
        "sections": [
            {"heading": "Introduction", "paragraphs": ["Claim [1] and [2]."]},
            {"heading": "Trial evidence", "paragraphs": ["More [2]."]},
            {"heading": "Conclusions", "paragraphs": ["Unresolved [1]."]},
        ],
        "key_points": ["Point [1]."],
        "glossary": [{"term": "Lux", "definition": "A unit of illuminance."}],
        "references": [1, 2, 3],
    }
    curation = {
        "relevance": {1: "direct", 2: "related", 3: "tangential"},
        "most_relevant_index": 2,
        "counts": {"direct": 1, "related": 1, "tangential": 1},
    }
    verification = {"unverified": ["-0.90", "4.91"], "misattributed": ["1.24"], "total": 6}
    # `databases` names the sources that actually answered. A real run always
    # records it; nothing is inferred when it is absent (see
    # test_methods_names_only_sources_that_answered).
    provenance = {
        "queries": ["light therapy schizophrenia"],
        "databases": ["Semantic Scholar Graph API", "OpenAlex"],
        "model": "test-model",
    }
    return article, papers, curation, verification, provenance


def test_recency_actually_counts() -> None:
    """Recency must be able to outweigh citation count, within reason.

    It used to be `year / 1000`, spanning 0.02 across two decades while the
    citation term spans 0-4 — arithmetically a rounding error. A real search
    returned a median paper year of 2013 with only 2 of 20 papers from 2020 or
    later, which pushes the writer towards broad old reviews that carry fewer
    specific findings than recent trials.
    """
    from articlegen.sources import Paper, _rank_score, RECENCY_HALF_LIFE

    now = 2026
    terms: set[str] = set()
    old_famous = Paper(title="A", abstract="a", year=2003, citation_count=2750)
    recent_solid = Paper(title="B", abstract="b", year=2025, citation_count=100)
    check("a recent solid paper beats an old famous one",
          _rank_score(recent_solid, terms, now) > _rank_score(old_famous, terms, now))

    # But citations must still count — not merely "newest wins".
    recent_ignored = Paper(title="C", abstract="c", year=2026, citation_count=0)
    recent_cited = Paper(title="D", abstract="d", year=2026, citation_count=500)
    check("among equally recent papers, citations still decide",
          _rank_score(recent_cited, terms, now) > _rank_score(recent_ignored, terms, now))

    # And topic relevance still dominates both — it's the primary sort key.
    off_topic_new = Paper(title="unrelated", abstract="unrelated", year=2026, citation_count=9999)
    on_topic_old = Paper(title="shift work sleep", abstract="shift work sleep",
                         year=2005, citation_count=1)
    topic_terms = {"shift", "work", "sleep"}
    check("topic relevance still outranks recency and fame",
          _rank_score(on_topic_old, topic_terms, now) > _rank_score(off_topic_new, topic_terms, now))

    # Beyond the half-life the recency bonus is spent, not negative.
    ancient = Paper(title="E", abstract="e", year=now - RECENCY_HALF_LIFE - 30, citation_count=10)
    older_still = Paper(title="F", abstract="f", year=now - RECENCY_HALF_LIFE - 60, citation_count=10)
    check("recency decays to zero rather than going negative",
          _rank_score(ancient, terms, now) == _rank_score(older_still, terms, now))

    check("a missing year does not crash or win",
          _rank_score(Paper(title="G", abstract="g", year=None), terms, now)[1] >= 0)


def test_render_blocks() -> None:
    from articlegen.render import render_article, render_markdown, _is_clinical

    article, papers, curation, verification, provenance = _sample_draft()
    topic = "sunlight for schizophrenia"
    h = render_article(article, papers, topic, curation, verification, provenance)
    md = render_markdown(article, papers, topic, curation, verification, provenance)

    check("html article-type label", "Evidence Review" in h)
    check("html abstract run-in head", 'class="run-in-head">Abstract' in h)
    check("html keywords printed", 'class="keywords"' in h and "schizophrenia" in h)
    check("html key points box", 'class="key-points"' in h and "Key points" in h)
    check("html Box 1 for the featured study", "Box 1 |" in h and "Study 2" in h)
    check("html Table 1 of cited evidence", "Table 1 |" in h and "<table>" in h)
    check("html Fig. 1 of the evidence base", "Fig. 1 |" in h and "<svg" in h)
    check("html methods states the search", "Methods" in h and "light therapy schizophrenia" in h)
    check("html names the databases", "OpenAlex" in h and "Semantic Scholar" in h)
    check("html limitations replace warning boxes",
          "Limitations." in h and "could not be located" in h and "-0.90" in h)
    check("html limitations name misattributed figures separately",
          "1.24" in h and "other than the one its sentence credits" in h)
    check("html no emoji warnings", "⚠" not in h)
    check("html glossary", "Glossary" in h and "Lux" in h)
    check("html back matter", "Competing interests" in h and "Data availability" in h)
    check("html references are Vancouver-style",
          '<span class="ref-authors">Ab, A.</span>' in h
          and ">Study 1.</a> <em>Journal 1</em> (2001)." in h)
    check("html clinical disclaimer", "Not medical or clinical advice" in h)
    check("html has no magazine furniture",
          "pull" not in h and "kicker" not in h and "standfirst" not in h)

    check("md abstract", "**Abstract.**" in md)
    check("md key points", "## Key points" in md)
    check("md Box 1", "**Box 1 | Most relevant source:" in md)
    check("md Box 1 disclaims the appraisal it is not making",
          "no quality appraisal was performed" in md)
    check("md Table 1", "**Table 1 |" in md and "| Ref. | Study |" in md)
    check("md Fig. 1", "**Fig. 1 |" in md)
    check("md methods", "## Methods" in md and "**Search strategy.**" in md)
    check("md evidence assessment", "## Evidence assessment" in md and "**Limitations.**" in md)
    check("md additional information", "## Additional information" in md)
    check("md clinical disclaimer", "Not medical or clinical advice" in md)
    check("clinical detection on", _is_clinical(topic, article))
    check("clinical detection off",
          not _is_clinical("gravity batteries", {"title": "Storage", "abstract": "x"}))


def test_display_item_placement() -> None:
    """Display items interleave with the body and every one is emitted exactly once."""
    from articlegen.render import render_article

    article, papers, curation, verification, provenance = _sample_draft()
    h = render_article(article, papers, "sunlight for schizophrenia", curation, verification, provenance)
    check("each display item appears once",
          h.count("Box 1 |") == 1 and h.count("Fig. 1 |") == 1 and h.count("Table 1 |") == 1)
    check("figure precedes box in the body", h.index("Fig. 1 |") < h.index("Box 1 |"))
    check("table sits in the back matter, after Methods",
          h.index("Table 1 |") > h.index("<h2>Methods"))
    check("table precedes the reference list", h.index("Table 1 |") < h.index("References"))
    check("key points sit before the conclusions",
          h.index("<h2>Introduction") < h.index("Key points") < h.index("<h2>Conclusions"))

    # A one-section article still gets all three, appended rather than interleaved.
    short = dict(article, sections=[{"heading": "Introduction", "paragraphs": ["Only [1]."]}])
    h_short = render_article(short, papers, "topic", curation, None, provenance)
    check("short articles keep every display item",
          all(k in h_short for k in ("Box 1 |", "Fig. 1 |", "Table 1 |")))


def test_legacy_draft_fields() -> None:
    """Drafts written against the pre-journal schema still render."""
    from articlegen.render import render_article
    from articlegen.sources import Paper

    papers = [Paper(title="Old study", abstract="a", year=2011, authors=["Ann Old"])]
    legacy = {
        "title": "An older draft", "standfirst": "The old deck line.",
        "evidence_note": "", "featured_study": {},
        "sections": [{"heading": "H", "paragraphs": ["Text [1]."], "pull_quote": "quote"}],
        "key_takeaways": ["Old point [1]."], "references": [1],
    }
    h = render_article(legacy, papers, "legacy topic")
    check("standfirst is used as the abstract", "The old deck line." in h)
    check("key_takeaways render as key points", "Old point" in h and "Key points" in h)
    check("pull quote is dropped", "quote" not in h)


def test_demo_and_index() -> None:
    import tempfile
    from articlegen import demo
    from articlegen.render import render_article, render_markdown, build_index

    h = render_article(demo.SAMPLE_BRIEFING, demo.SAMPLE_PAPERS, "Sample topic",
                       demo.SAMPLE_CURATION, None, demo.SAMPLE_PROVENANCE)
    md = render_markdown(demo.SAMPLE_BRIEFING, demo.SAMPLE_PAPERS, "Sample topic",
                         demo.SAMPLE_CURATION, None, demo.SAMPLE_PROVENANCE)
    check("demo briefing html names the artefact", "Evidence briefing" in h)
    check("demo briefing has findings and unknowns",
          "What the evidence shows" in h and "What remains open" in h)
    check("demo briefing has three papers to open", "Three papers to open" in h)
    check("demo briefing has no journal Box 1", "Box 1 |" not in h)
    check("demo markdown renders as a briefing", md.startswith("**EVIDENCE BRIEFING**"))
    review = render_article(demo.SAMPLE_ARTICLE, demo.SAMPLE_PAPERS, "Sample topic",
                            demo.SAMPLE_CURATION, None, demo.SAMPLE_PROVENANCE)
    check("the parked Review sample still renders display items",
          all(k in review for k in ("Box 1 |", "Fig. 1 |", "Table 1 |", "Evidence Review")))
    check("demo Review sections run Introduction -> Conclusions",
          demo.SAMPLE_ARTICLE["sections"][0]["heading"] == "Introduction"
          and demo.SAMPLE_ARTICLE["sections"][-1]["heading"].startswith("Conclusions"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "2026-01-01-x.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(h)
        idx = build_index(d)
        check("index builds", os.path.exists(idx))
        idx_html = open(idx, encoding="utf-8").read()
        import datetime as _dt
        mtime = _dt.datetime.fromtimestamp(os.path.getmtime(path))
        check("drafts index uses day-month-year",
              mtime.strftime("%d %b %Y").lstrip("0") in idx_html)
        check("drafts index is not month-first American",
              mtime.strftime("%b %d, %Y") not in idx_html)


def test_dates_are_australian() -> None:
    """Every printed date is day-month-year. American month-first was the
    drafts-index format (`Aug 24, 2026`) and the browser default on this
    machine (`toLocaleDateString()` with no locale).
    """
    import datetime as _dt
    import inspect
    from articlegen import pipeline, render, writer

    check("article dates are day month year",
          render._au_date(_dt.date(2026, 8, 4)) == "4 August 2026")
    check("datetime dates are day month year",
          render._au_datetime(_dt.datetime(2026, 8, 24, 18, 2))
          == "24 Aug 2026 18:02")
    check("pipeline Methods date uses the same helper",
          "_au_date()" in inspect.getsource(pipeline))

    check("date rule in briefing system prompt",
          writer._DATE_RULE in writer._BRIEFING_SYSTEM)
    check("date rule in writer system prompt",
          writer._DATE_RULE in writer._WRITER_SYSTEM)
    check("date rule survives in full-text briefing prompt",
          writer._DATE_RULE in writer._BRIEFING_SYSTEM_FULLTEXT)
    check("date rule survives in full-text writer prompt",
          writer._DATE_RULE in writer._WRITER_SYSTEM_FULLTEXT)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    check("gallery dates pin Australian locale",
          "toLocaleDateString('en-AU'" in page)
    check("gallery dates go through formatAuDate",
          "formatAuDate(" in page)
    check("bare toLocaleDateString() is gone",
          "toLocaleDateString()" not in page)


def test_visitor_gallery() -> None:
    """Shared briefings are auto-published, capped, and not the personal library.

    Generating publishes. The hosted backend has no disk, so the durable
    copy is a public gist — listed from the front end without waiting on
    Render. A gist-scoped token cannot rewrite the generator.
    """
    import inspect
    import json as json_mod
    import tempfile
    from unittest.mock import patch

    import requests

    from articlegen import gallery, web

    check("empty html is refused", gallery.validate_html("") is not None)
    check("a script tag is refused",
          gallery.validate_html("<h1>Hi</h1><script>x</script>") is not None)
    check("html without a heading is refused",
          gallery.validate_html("<p>no heading</p>") is not None)
    check("a normal briefing html is accepted",
          gallery.validate_html("<h1>A question</h1><p>findings</p>") is None)
    too_big = "<h1>x</h1>" + ("a" * (gallery.GALLERY_MAX_HTML + 10))
    check("oversize html is refused", gallery.validate_html(too_big) is not None)

    saved = {k: os.environ.get(k) for k in (
        "ARTICLEGEN_GALLERY_TOKEN", "ARTICLEGEN_STATELESS")}
    original_dir = gallery.GALLERY_DIR
    original_max = gallery.GALLERY_MAX
    try:
        os.environ.pop("ARTICLEGEN_GALLERY_TOKEN", None)
        os.environ.pop("ARTICLEGEN_STATELESS", None)
        check("a local server can accept shares", gallery.gallery_enabled())
        os.environ["ARTICLEGEN_STATELESS"] = "1"
        check("stateless without a token cannot", not gallery.gallery_enabled())
        os.environ["ARTICLEGEN_GALLERY_TOKEN"] = "ghp_test"
        check("a gist token enables the hosted gallery", gallery.gallery_enabled())

        os.environ.pop("ARTICLEGEN_GALLERY_TOKEN", None)
        os.environ.pop("ARTICLEGEN_STATELESS", None)
        tmp = tempfile.mkdtemp()
        gallery.GALLERY_DIR = tmp
        gallery.GALLERY_MAX = 2
        first = gallery.publish("First question", "<h1>First</h1><p>a</p>")
        second = gallery.publish("Second question", "<h1>Second</h1><p>b</p>")
        third = gallery.publish("Third question", "<h1>Third</h1><p>c</p>")
        items = gallery.list_items()
        check("newest shared briefing is first", items[0]["id"] == third["id"])
        check("the gallery is capped", len(items) == 2)
        check("the oldest dropped off the index",
              first["id"] not in {i["id"] for i in items})
        check("the dropped file is gone",
              not os.path.exists(os.path.join(tmp, os.path.basename(first["html_url"]))))
        check("local html_url is a gallery path",
              second["html_url"].startswith("/gallery/"))

        os.environ["ARTICLEGEN_STATELESS"] = "1"
        os.environ.pop("ARTICLEGEN_GALLERY_TOKEN", None)
        check("try_publish is a no-op when the gallery is off",
              gallery.try_publish("T", "<h1>T</h1><p>a</p>") is None)
        os.environ["ARTICLEGEN_GALLERY_TOKEN"] = "ghp_test"
        with patch.object(gallery, "publish", return_value={"id": "x"}):
            check("try_publish returns the item when the gallery is on",
                  gallery.try_publish("T", "<h1>T</h1><p>a</p>")["id"] == "x")
        with patch.object(gallery, "publish", side_effect=gallery.GalleryError("no")):
            check("try_publish swallows GalleryError",
                  gallery.try_publish("T", "<h1>T</h1><p>a</p>") is None)
        with patch.object(gallery, "publish", side_effect=RuntimeError("boom")):
            check("try_publish swallows unexpected errors",
                  gallery.try_publish("T", "<h1>T</h1><p>a</p>") is None)
    finally:
        gallery.GALLERY_DIR = original_dir
        gallery.GALLERY_MAX = original_max
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Gist publish: create article gist, rewrite index, drop overflow.
    class _Resp:
        def __init__(self, payload, status=200):
            self.status_code = status
            self._payload = payload
            self.content = b"{}" if payload is not None else b""

        def json(self):
            return self._payload

    calls = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "json": json,
                      "auth": bool((headers or {}).get("Authorization"))})
        if method == "POST" and url.endswith("/gists"):
            return _Resp({"id": "newgist1"})
        if method == "GET" and gallery.GALLERY_INDEX_GIST in url:
            return _Resp({
                "files": {
                    gallery.GALLERY_INDEX_FILE: {
                        "content": json_mod.dumps({"items": [
                            {"id": "old1", "title": "Old", "html_url": "u1"},
                            {"id": "old2", "title": "Older", "html_url": "u2"},
                        ]})
                    }
                }
            })
        if method == "PATCH":
            return _Resp({})
        if method == "DELETE":
            return _Resp({}, status=204)
        return _Resp({}, status=500)

    saved_token = os.environ.get("ARTICLEGEN_GALLERY_TOKEN")
    try:
        os.environ["ARTICLEGEN_GALLERY_TOKEN"] = "ghp_test"
        gallery.GALLERY_MAX = 2
        with patch.object(requests, "request", fake_request):
            item = gallery.publish("Gist title", "<h1>Gist</h1><p>ok</p>")
        check("gist publish returns the new id", item["id"] == "newgist1")
        check("gist html_url is a raw gist url",
              item["html_url"].startswith(
                  "https://gist.githubusercontent.com/bartholomewtj/newgist1/"))
        methods = [c["method"] for c in calls]
        check("gist publish creates then updates the index",
              methods[0] == "POST" and "GET" in methods and "PATCH" in methods)
        check("overflow gists are deleted", "DELETE" in methods)
        check("the gist token is sent as a header, not a query",
              all(c["auth"] for c in calls) and all(
                  "ghp_test" not in (c["url"] or "") for c in calls))
        patch_body = next(c["json"] for c in calls if c["method"] == "PATCH")
        index = json_mod.loads(
            patch_body["files"][gallery.GALLERY_INDEX_FILE]["content"])
        check("the new briefing is first in the index",
              index["items"][0]["id"] == "newgist1")
        check("the gist index is capped too", len(index["items"]) <= 2)
    finally:
        gallery.GALLERY_MAX = original_max
        if saved_token is None:
            os.environ.pop("ARTICLEGEN_GALLERY_TOKEN", None)
        else:
            os.environ["ARTICLEGEN_GALLERY_TOKEN"] = saved_token

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    check("front end gist id matches the server",
          f"GALLERY_INDEX_GIST = '{gallery.GALLERY_INDEX_GIST}'" in page)
    check("front end gist file matches the server",
          f"GALLERY_INDEX_FILE = '{gallery.GALLERY_INDEX_FILE}'" in page)

    draft_src = inspect.getsource(web.ArticleGenHandler._handle_draft)
    check("generating publishes to the gallery",
          "gallery.try_publish" in draft_src)
    check("generate does not go through the manual gallery POST",
          "_handle_publish_gallery" not in draft_src)
    check("the share-to-gallery button is gone",
          "shareGalleryBtn" not in page and "shareToGallery" not in page)
    select = page[page.index("async function selectDraft"):]
    select = select[:select.index("\n  }\n")]
    check("a finished draft refreshes the public list",
          "loadPublicPreview" in select)
    check("the page does not POST /api/gallery from generate",
          "/api/gallery" not in select)

    health = inspect.getsource(web.ArticleGenHandler.do_GET)
    check("health reports whether the gallery is on",
          '"gallery"' in health or "gallery.gallery_enabled" in health)
    check("GET /api/gallery is served", "_handle_get_gallery" in health)
    post = inspect.getsource(web.ArticleGenHandler.do_POST)
    check("POST /api/gallery is the publish path",
          "_handle_publish_gallery" in post)
    check("publish is rate-limited",
          "_over_rate_limit" in inspect.getsource(
              web.ArticleGenHandler._handle_publish_gallery))


def test_web_server() -> None:
    import json
    from io import BytesIO
    from articlegen.web import ArticleGenHandler

    class DummyRequest:
        def makefile(self, *args, **kwargs):
            return BytesIO(b"GET /api/drafts HTTP/1.1\r\nHost: localhost\r\n\r\n")

    class FakeSocket:
        def __init__(self):
            self.rfile = BytesIO(b"GET /api/drafts HTTP/1.1\r\nHost: localhost\r\n\r\n")
            self.wfile = BytesIO()

        def sendall(self, data):
            self.wfile.write(data)

        def makefile(self, mode, *args, **kwargs):
            if "r" in mode:
                return self.rfile
            return self.wfile

    sock = FakeSocket()
    handler = ArticleGenHandler(sock, ("127.0.0.1", 8000), None)
    output = sock.wfile.getvalue().decode("utf-8")
    check("web handler GET /api/drafts returns 200 OK", "200 OK" in output and "application/json" in output)


def test_run_log_never_carries_the_topic() -> None:
    """A run log line carries counts, durations and fixed vocabulary — never content.

    The shared host is stateless because visitor research topics are sensitive.
    An allowlist (RUN_FIELDS) ensures no topic, prompt, search term, query string,
    article text, API key, or IP address reaches the log.
    """
    import json as json_mod
    from types import SimpleNamespace
    from articlegen import web

    # 1. Check forbidden field names are absent from web.RUN_FIELDS
    forbidden = (
        "topic", "theme", "question", "title", "search_terms", "queries",
        "html", "markdown", "api_key", "ip", "client_ip", "error_message",
    )
    for name in forbidden:
        check(f"forbidden field {name!r} not in RUN_FIELDS", name not in web.RUN_FIELDS)

    sentinels = [
        "SENTINEL-TOPIC",
        "SENTINEL-QUERY-1",
        "SENTINEL-QUERY-2",
        "SENTINEL-NAMED-QUERY",
        "sk-or-v1-SENTINEL-KEY",
        "203.0.113.9",
        "SENTINEL-UNVERIFIED-42",
        "SENTINEL-MISATTRIBUTED-12",
        "SENTINEL-CLINICAL",
        "SENTINEL-EXCERPT",
        "SENTINEL-TITLE",
    ]

    fake_draft = SimpleNamespace(
        papers=[{"title": "Paper 1"}],
        cited_refs=[1],
        curation={
            "counts": {"direct": 1, "related": 0, "tangential": 0},
            "relevance": {1: "direct"},
        },
        provenance={
            "queries": ["SENTINEL-QUERY-1", "SENTINEL-QUERY-2"],
            "named_sources": {
                "added": 1,
                "queries": ["SENTINEL-NAMED-QUERY"],
            },
            "full_text_sources": [1],
            "full_text_via": {"papers": 1, "europe_pmc": 0},
            "model": "openai/gpt-5.6-luna",
        },
        verification={
            "total": 2,
            "unverified": ["SENTINEL-UNVERIFIED-42%"],
            "misattributed": ["SENTINEL-MISATTRIBUTED-12%"],
        },
        style_report={
            "stats": {"sentences": 20, "mean_sentence_words": 18, "hedges_per_sentence": 0.3, "passive_ratio": 0.1},
            "issues": [
                {"severity": "error", "rule": "clinical-directive", "where": "Conclusions", "detail": "SENTINEL-CLINICAL", "excerpt": "SENTINEL-EXCERPT"}
            ],
        },
        article={"title": "SENTINEL-TITLE"},
        topic="SENTINEL-TOPIC",
    )

    fields = web.draft_run_fields(fake_draft)
    check("screened count is 1", fields["screened"] == 1)
    check("cited count is 1", fields["cited"] == 1)
    check("cited_direct count is 1", fields["cited_direct"] == 1)
    check("direct count is 1", fields["direct"] == 1)
    check("full_text count is 1", fields["full_text"] == 1)
    check("named_added count is 1", fields["named_added"] == 1)
    check("named_queries is int count 1", fields["named_queries"] == 1 and isinstance(fields["named_queries"], int))
    check("figures count is 2", fields["figures"] == 2)
    check("unverified is int count 1", fields["unverified"] == 1 and isinstance(fields["unverified"], int))
    check("misattributed is int count 1", fields["misattributed"] == 1 and isinstance(fields["misattributed"], int))
    check("style_errors count is 1", fields["style_errors"] == 1)
    check("style_rules lists rule name", fields["style_rules"] == ["clinical-directive"])
    check("working_draft is boolean True", fields["working_draft"] is True)

    raw_payload = {
        "topic": "SENTINEL-TOPIC",
        "api_key": "sk-or-v1-SENTINEL-KEY",
        "html": "<h1>SENTINEL-TITLE</h1>",
        "ip": "203.0.113.9",
        "queries": ["SENTINEL-QUERY-1"],
    }
    extra = {**fields, **raw_payload}
    record = web.build_run_record("/api/draft", "POST", 200, 1234, extra)
    record_json = json_mod.dumps(record)

    for sentinel in sentinels:
        check(f"sentinel absent from run record: {sentinel}", sentinel not in record_json)

    handler = web.ArticleGenHandler.__new__(web.ArticleGenHandler)
    handler._run_extra = {}
    handler._note_run(topic="SENTINEL-TOPIC", api_key="sk-or-v1-SENTINEL-KEY", screened=5)
    check("topic dropped by _note_run", "topic" not in handler._run_extra)
    check("api_key dropped by _note_run", "api_key" not in handler._run_extra)
    check("screened kept by _note_run", handler._run_extra.get("screened") == 5)

    check("build_run_record ok is True for 200", web.build_run_record("/api/draft", "POST", 200, 100)["ok"] is True)
    check("build_run_record ok is False for 500", web.build_run_record("/api/draft", "POST", 500, 100)["ok"] is False)


def test_every_endpoint_logs_one_run_line() -> None:
    """Every request to /api/ideas, /api/draft and /api/gallery emits one JSON line to stderr."""
    import io
    import json as json_mod
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    from articlegen import gallery, web
    from articlegen.pipeline import NoPapersFound

    saved_env = {k: os.environ.get(k) for k in ("ARTICLEGEN_ANALYTICS_GIST", "ARTICLEGEN_PUBLIC_OPENROUTER_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        os.environ.pop("ARTICLEGEN_ANALYTICS_GIST", None)
        os.environ.pop("ARTICLEGEN_PUBLIC_OPENROUTER_KEY", None)
        os.environ["OPENROUTER_API_KEY"] = "sk-test-fake"

        def make_handler(method: str, path: str, body: dict | None = None, raw_body: bytes | None = None) -> tuple[web.ArticleGenHandler, list[dict], io.StringIO]:
            handler = web.ArticleGenHandler.__new__(web.ArticleGenHandler)
            handler.client_address = ("127.0.0.1", 12345)
            handler.path = path

            if raw_body is not None:
                encoded = raw_body
            elif body is not None:
                encoded = json_mod.dumps(body).encode("utf-8")
            else:
                encoded = b""

            handler.rfile = io.BytesIO(encoded)
            handler.headers = {"Content-Length": str(len(encoded))}

            sent = []
            def fake_send_json(data: dict, status: int = 200) -> None:
                handler._run_status = status
                sent.append({"data": data, "status": status})
            handler._send_json = fake_send_json

            stderr_buf = io.StringIO()
            return handler, sent, stderr_buf

        # 1. ideas success
        handler, sent, buf = make_handler("POST", "/api/ideas", {"theme": "delirium"})
        with patch.object(sys, "stderr", buf), patch.object(web, "generate_ideas", return_value=["Idea 1", "Idea 2"]):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("ideas success emits exactly 1 run line", len(lines) == 1)
        check("ideas kind == 'run'", lines[0]["kind"] == "run")
        check("ideas endpoint == '/api/ideas'", lines[0]["endpoint"] == "/api/ideas")
        check("ideas status == 200", lines[0]["status"] == 200)
        check("ideas n_ideas == 2", lines[0]["n_ideas"] == 2)
        check("theme text not in record", "delirium" not in json_mod.dumps(lines[0]))

        # 2. ideas missing theme (400)
        handler, sent, buf = make_handler("POST", "/api/ideas", {"theme": ""})
        with patch.object(sys, "stderr", buf):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("ideas missing theme emits 1 run line", len(lines) == 1)
        check("ideas missing theme status == 400", lines[0]["status"] == 400)
        check("ideas missing theme error == 'missing_theme'", lines[0].get("error") == "missing_theme")

        # 3. ideas raising exception (500)
        handler, sent, buf = make_handler("POST", "/api/ideas", {"theme": "delirium"})
        with patch.object(sys, "stderr", buf), patch.object(web, "generate_ideas", side_effect=RuntimeError("Provider exploded")):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("ideas error emits 1 run line", len(lines) == 1)
        check("ideas error status == 500", lines[0]["status"] == 500)
        check("ideas error == 'RuntimeError'", lines[0].get("error") == "RuntimeError")

        # 4. draft success
        fake_draft = SimpleNamespace(
            papers=[{"title": "Paper 1"}],
            cited_refs=[1],
            curation={"counts": {"direct": 1, "related": 0, "tangential": 0}, "relevance": {1: "direct"}},
            provenance={"queries": ["query 1"], "named_sources": {"added": 0, "queries": []}, "full_text_sources": [], "model": "openai/gpt-5.6-luna"},
            verification={"total": 0, "unverified": [], "misattributed": []},
            style_report={"stats": {"sentences": 10, "mean_sentence_words": 15, "hedges_per_sentence": 0.3, "passive_ratio": 0.1}, "issues": []},
            article={
                "form": "briefing",
                "title": "Test Title",
                "question": "What is the evidence on seclusion?",
                "answer": "Answer here.",
                "keywords": ["seclusion"],
                "findings": ["Finding one [1]."],
                "unknowns": ["Unknown one."],
                "papers_to_open": [{"source": 1, "why": "Landmark trial"}],
            },
            topic="test topic",
            summary=lambda: "1 of 1 screened sources cited",
        )
        handler, sent, buf = make_handler("POST", "/api/draft", {"topic": "seclusion"})
        with patch.object(sys, "stderr", buf), \
             patch.object(web, "generate_draft", return_value=fake_draft), \
             patch.object(gallery, "try_publish", return_value={"id": "g1", "title": "Test Title", "html_url": "u"}):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("draft success emits 1 run line", len(lines) == 1)
        check("draft status == 200", lines[0]["status"] == 200)
        check("draft endpoint == '/api/draft'", lines[0]["endpoint"] == "/api/draft")
        check("draft screened == 1", lines[0]["screened"] == 1)
        check("draft topic not in json", "seclusion" not in json_mod.dumps(lines[0]))
        check("draft response includes the gallery item",
              sent[0]["data"].get("gallery", {}).get("id") == "g1")

        handler, sent, buf = make_handler("POST", "/api/draft", {"topic": "seclusion"})
        with patch.object(sys, "stderr", buf), \
             patch.object(web, "generate_draft", return_value=fake_draft), \
             patch.object(gallery, "try_publish", return_value=None):
            handler.do_POST()
        check("gallery miss does not fail the draft", sent[0]["status"] == 200)
        check("gallery miss omits the gallery key", "gallery" not in sent[0]["data"])

        # 5. draft raising NoPapersFound (422)
        handler, sent, buf = make_handler("POST", "/api/draft", {"topic": "seclusion"})
        with patch.object(sys, "stderr", buf), patch.object(web, "generate_draft", side_effect=NoPapersFound("No literature found")):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("draft NoPapersFound emits 1 run line", len(lines) == 1)
        check("draft NoPapersFound status == 422", lines[0]["status"] == 422)
        check("draft error == 'NoPapersFound'", lines[0].get("error") == "NoPapersFound")

        # 6. gallery publish success
        handler, sent, buf = make_handler("POST", "/api/gallery", {"html": "<h1>Test</h1><p>content</p>", "title": "Title"})
        with patch.object(sys, "stderr", buf), patch.object(gallery, "publish", return_value={"id": "g1", "title": "Title", "html_url": "url"}):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("gallery publish emits 1 run line", len(lines) == 1)
        check("gallery publish status == 200", lines[0]["status"] == 200)
        check("gallery publish endpoint == '/api/gallery'", lines[0]["endpoint"] == "/api/gallery")

        # 7. gallery publish GalleryError (503)
        handler, sent, buf = make_handler("POST", "/api/gallery", {"html": "<h1>Test</h1><p>content</p>"})
        with patch.object(sys, "stderr", buf), patch.object(gallery, "publish", side_effect=gallery.GalleryError("Gist store unavailable")):
            handler.do_POST()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("gallery error emits 1 run line", len(lines) == 1)
        check("gallery error status == 503", lines[0]["status"] == 503)
        check("gallery error == 'GalleryError'", lines[0].get("error") == "GalleryError")

        # 8. gallery GET list
        handler, sent, buf = make_handler("GET", "/api/gallery")
        with patch.object(sys, "stderr", buf), patch.object(gallery, "list_items", return_value=[{"id": "1"}, {"id": "2"}]):
            handler.do_GET()
        lines = [json_mod.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        check("gallery GET emits 1 run line", len(lines) == 1)
        check("gallery GET status == 200", lines[0]["status"] == 200)
        check("gallery GET method == 'GET'", lines[0]["method"] == "GET")
        check("gallery GET items == 2", lines[0]["items"] == 2)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_run_log_gist_is_optional_and_gist_scoped() -> None:
    """Run log private gist sink is opt-in, capped, and never raises."""
    import json as json_mod
    from unittest.mock import patch
    from articlegen import gallery

    saved_env = {k: os.environ.get(k) for k in ("ARTICLEGEN_ANALYTICS_GIST", "ARTICLEGEN_GALLERY_TOKEN")}
    try:
        # 1. Unset: analytics_enabled is False, append_run makes 0 calls
        os.environ.pop("ARTICLEGEN_ANALYTICS_GIST", None)
        os.environ["ARTICLEGEN_GALLERY_TOKEN"] = "ghp_fake"
        check("analytics_enabled is False when gist id is unset", not gallery.analytics_enabled())
        github_calls = []
        with patch.object(gallery, "_github", side_effect=lambda *args, **kwargs: github_calls.append(args)):
            gallery.append_run({"kind": "run", "status": 200})
        check("append_run makes zero calls when gist is unset", len(github_calls) == 0)

        # 2. Set: calls GET then PATCH on /gists/<id>
        os.environ["ARTICLEGEN_ANALYTICS_GIST"] = "secret_gist_123"
        os.environ["ARTICLEGEN_GALLERY_TOKEN"] = "ghp_fake"
        check("analytics_enabled is True when gist and token are set", gallery.analytics_enabled())

        calls = []
        def fake_github(method: str, path: str, body: dict | None = None) -> dict:
            calls.append({"method": method, "path": path, "body": body})
            if method == "GET":
                return {
                    "files": {
                        gallery.ANALYTICS_FILE: {
                            "content": json_mod.dumps({"kind": "run", "t": "2026-08-23T01:00:00Z", "id": 1}) + "\n"
                        }
                    }
                }
            if method == "PATCH":
                return {}
            return {}

        with patch.object(gallery, "_github", fake_github):
            gallery.append_run({"kind": "run", "t": "2026-08-23T02:00:00Z", "id": 2})

        check("append_run called GET then PATCH", [c["method"] for c in calls] == ["GET", "PATCH"])
        check("append_run called the right gist id", all(c["path"] == "/gists/secret_gist_123" for c in calls))
        patch_body = calls[1]["body"]
        content = patch_body["files"][gallery.ANALYTICS_FILE]["content"]
        lines = [json_mod.loads(l) for l in content.splitlines() if l.strip()]
        check("written file has 2 lines", len(lines) == 2)
        check("newest record is last", lines[-1]["id"] == 2)

        # 3. Max lines capping
        calls.clear()
        seed_lines = [json_mod.dumps({"kind": "run", "id": i}) for i in range(gallery.ANALYTICS_MAX_LINES + 5)]
        def fake_github_overflow(method: str, path: str, body: dict | None = None) -> dict:
            calls.append({"method": method, "path": path, "body": body})
            if method == "GET":
                return {
                    "files": {
                        gallery.ANALYTICS_FILE: {
                            "content": "\n".join(seed_lines) + "\n"
                        }
                    }
                }
            return {}

        with patch.object(gallery, "_github", fake_github_overflow):
            gallery.append_run({"kind": "run", "id": "latest"})

        content = calls[1]["body"]["files"][gallery.ANALYTICS_FILE]["content"]
        capped_lines = [json_mod.loads(l) for l in content.splitlines() if l.strip()]
        check("capped to ANALYTICS_MAX_LINES", len(capped_lines) == gallery.ANALYTICS_MAX_LINES)
        check("latest line is at the end", capped_lines[-1]["id"] == "latest")

        # 4. Errors in _github are swallowed (GalleryError and RuntimeError)
        with patch.object(gallery, "_github", side_effect=gallery.GalleryError("API 500")):
            gallery.append_run({"kind": "run"})  # must not raise
        check("append_run swallows GalleryError", True)

        with patch.object(gallery, "_github", side_effect=RuntimeError("unexpected network error")):
            gallery.append_run({"kind": "run"})  # must not raise
        check("append_run swallows RuntimeError", True)

        # 5. File name and isolation
        check("append_run targets ANALYTICS_FILE", gallery.ANALYTICS_FILE == "articlegen-runs.jsonl")
        check("append_run does not target GALLERY_INDEX_FILE", gallery.ANALYTICS_FILE != gallery.GALLERY_INDEX_FILE)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_ungrounded_citations_leave_no_trace() -> None:
    """A model does cite SOURCE numbers it was never given. The marker is
    dropped — but it used to leave its leading space behind, stranding the
    sentence's full stop as "…the market ." in a shipped article."""
    from articlegen.render import _remap_citations, _shift_markers_after_punctuation

    def rendered(text, mapping):
        return _shift_markers_after_punctuation(_remap_citations(text, mapping))

    check("a dropped marker takes its space with it",
          rendered("ahead of the market [15].", {1: 1}) == "ahead of the market.")
    check("kept markers still hug their word",
          rendered("a claim [1].", {1: 1}) == "a claim.[1]")
    check("dropping one of a bundle keeps the rest",
          rendered("both agree [1, 9].", {1: 1}) == "both agree.[1]")
    check("a dropped marker mid-sentence closes up",
          rendered("dropped [9] here [2].", {2: 2}) == "dropped here.[2]")


def test_second_hand_figures_are_a_last_resort() -> None:
    """A draft opened its Introduction on three figures the writer never saw at
    first hand (e.g. "A meta-analysis cited within a Canadian pilot study estimated
    ... 14.4%"). The number is real in the quoting paper, so verify.check_statistics
    confirms the quotation, but nobody checked whether the original paper reported it.

    The writer is instructed to avoid second-hand figures in load-bearing slots
    (title, abstract, key_points, opening claim of Introduction) when first-hand
    alternatives exist, and to keep "cited within"-style attribution when they are
    unavoidable. This test pins that the rule survives every system-prompt derivation
    and does not duplicate substitution targets.
    """
    from articlegen import writer

    # 1. Rule is in _WRITER_SYSTEM
    check("second-hand figure rule is in writer system prompt",
          "SECOND-HAND" in writer._WRITER_SYSTEM)

    # 2. Names the slots it protects
    for slot in ("`title`", "`abstract`", "`key_points`", "Introduction"):
        check(f"second-hand figure rule protects slot: {slot}",
              slot in writer._WRITER_SYSTEM)

    # 3. Survives every derivation
    derivations = (
        ("_WRITER_SYSTEM_FULLTEXT", writer._WRITER_SYSTEM_FULLTEXT),
        ("_REVISE_SYSTEM", writer._REVISE_SYSTEM),
        ("_REVISE_SYSTEM_FULLTEXT", writer._REVISE_SYSTEM_FULLTEXT),
        ("_REVISE_PATCH_SYSTEM", writer._REVISE_PATCH_SYSTEM),
        ("_REVISE_PATCH_SYSTEM_FULLTEXT", writer._REVISE_PATCH_SYSTEM_FULLTEXT),
        ("_BRIEFING_SYSTEM", writer._BRIEFING_SYSTEM),
        ("_BRIEFING_SYSTEM_FULLTEXT", writer._BRIEFING_SYSTEM_FULLTEXT),
        ("_REVISE_BRIEFING_SYSTEM", writer._REVISE_BRIEFING_SYSTEM),
        ("_REVISE_BRIEFING_SYSTEM_FULLTEXT", writer._REVISE_BRIEFING_SYSTEM_FULLTEXT),
        ("_REVISE_BRIEFING_PATCH_SYSTEM", writer._REVISE_BRIEFING_PATCH_SYSTEM),
        ("_REVISE_BRIEFING_PATCH_SYSTEM_FULLTEXT", writer._REVISE_BRIEFING_PATCH_SYSTEM_FULLTEXT),
    )
    for name, prompt in derivations:
        check(f"second-hand figure rule survives into {name}",
              "SECOND-HAND" in prompt)

    # 4. No substitution target appears twice in _WRITER_SYSTEM
    for old, _ in writer._FULLTEXT_SUBSTITUTIONS:
        check(f"substitution target appears exactly once: {old[:40]}…",
              writer._WRITER_SYSTEM.count(old) == 1)

    # 5. Preserves honest attribution phrasing
    for name, prompt in (("_WRITER_SYSTEM", writer._WRITER_SYSTEM),) + derivations:
        check(f"'cited within' attribution phrasing preserved in {name}",
              "cited within" in prompt)


def test_full_text_grounding() -> None:
    """Full-text mode: fetch only what is truly open access, show a bounded
    excerpt, tell the model the truth about its inputs, and verify statistics
    against exactly what was shown — never against text the writer did not see.
    """
    from articlegen import render, sources, verify, writer
    from articlegen.sources import Paper, full_text_excerpts

    # -- search carries the fetchability facts ------------------------------
    payload = {"resultList": {"result": [
        {"id": "1", "source": "MED", "title": "OA paper", "abstractText": "An abstract.",
         "pubYear": "2026", "pmcid": "PMC123", "isOpenAccess": "Y", "inEPMC": "Y"},
        {"id": "2", "source": "MED", "title": "OA elsewhere only", "abstractText": "A.",
         "pubYear": "2026", "pmcid": "PMC124", "isOpenAccess": "Y", "inEPMC": "N"},
    ]}}

    class FakeResp:
        def __init__(self, data=None, text=""):
            self._data, self.text = data, text
        def json(self):
            return self._data

    real = sources._get_with_retry
    try:
        sources._get_with_retry = lambda url, params, headers: FakeResp(payload)
        papers = sources.search_europe_pmc("q", limit=2)
    finally:
        sources._get_with_retry = real
    check("pmcid captured from the search", papers[0].pmcid == "PMC123")
    check("in-EPMC open access marks fetchable", papers[0].is_open_access)
    check("OA hosted elsewhere is not fetchable", not papers[1].is_open_access)

    # -- JATS parsing -------------------------------------------------------
    jats = """<article><body>
      <sec><title>Methods</title><p>We enrolled 441 participants [ 3 ] over two years.</p></sec>
      <sec><title>Results</title><p>Readmission fell (OR 0.66) [2, 5].</p></sec>
      <sec><title>Acknowledgements</title><p>We thank everyone.</p></sec>
    </body></article>"""
    text = sources._parse_fulltext_xml(jats)
    check("sections flattened in order", text.index("enrolled") < text.index("Readmission"))
    check("back matter dropped", "thank" not in text)
    check("the paper's own citation brackets are stripped",
          "[ 3 ]" not in text and "[2, 5]" not in text)
    check("statistics survive the stripping", "OR 0.66" in text)
    check("recognised JATS sections get paperfetch marker lines",
          "## Methods" in text and "## Results" in text)
    check("the marker precedes the section body",
          text.index("## Methods") < text.index("enrolled"))

    # -- fetch gating and cache --------------------------------------------
    calls = []
    try:
        sources._get_with_retry = (
            lambda url, params, headers: (calls.append(url), FakeResp(text=jats))[1])
        sources.clear_search_cache()
        no_pmcid = Paper(title="n", abstract="a")
        check("no PMCID -> no fetch", sources.fetch_full_text(no_pmcid) == "")
        oa = Paper(title="o", abstract="a", pmcid="PMC9", is_open_access=True)
        first = sources.fetch_full_text(oa)
        again = sources.fetch_full_text(oa)
        check("OA paper fetches and parses", "enrolled 441" in first)
        check("second fetch is served from cache", again == first and len(calls) == 1)
    finally:
        sources._get_with_retry = real
        sources.clear_search_cache()

    # -- dedupe enrichment: the kept copy learns the duplicate's PMCID ------
    dup_title = "The same study twice"
    oa_copy = Paper(title=dup_title, abstract="a", pmcid="PMC77", is_open_access=True)
    plain_copy = Paper(title=dup_title, abstract="a")
    reals = (sources.search_semantic_scholar, sources.search_openalex,
             sources.search_europe_pmc, sources.search_arxiv)
    try:
        sources.search_semantic_scholar = lambda q, limit=15: []
        sources.search_openalex = lambda q, limit=15: [plain_copy]
        sources.search_europe_pmc = lambda q, limit=15: [oa_copy]
        sources.search_arxiv = lambda q, limit=15: []
        merged = sources.gather_evidence(["q"], use_cache=False)
    finally:
        (sources.search_semantic_scholar, sources.search_openalex,
         sources.search_europe_pmc, sources.search_arxiv) = reals
    check("dedupe keeps one copy", len(merged) == 1)
    check("which inherits the duplicate's PMCID",
          merged[0].pmcid == "PMC77" and merged[0].is_open_access)

    # -- excerpt budgeting: the shared writer/verifier contract -------------
    big = "x" * (sources.FULLTEXT_PER_PAPER_CHARS + 5000)
    ranked = [Paper(title=f"t{i}", abstract="a", full_text=big) for i in range(7)]
    ranked.insert(1, Paper(title="no ft", abstract="a"))
    ex = full_text_excerpts(ranked)
    check("papers without full text are skipped", 2 not in ex)
    check("per-paper cap applies", len(ex[1]) == sources.FULLTEXT_PER_PAPER_CHARS)
    check("total budget bounds the set",
          sum(len(v) for v in ex.values()) <= sources.FULLTEXT_TOTAL_CHARS)
    check("rank order decides who is left abstract-only",
          1 in ex and max(ex) < len(ranked))

    # -- payload formatting -------------------------------------------------
    two = [Paper(title="ft", abstract="A1.", full_text="Full body text 12.5% here."),
           Paper(title="ab", abstract="A2.")]
    shown = full_text_excerpts(two)
    block = writer._format_sources(two, None, shown)
    check("full text rides along under its source", "Full text (open access" in block
          and "Full body text 12.5% here." in block)
    check("abstract-only sources say so", "no open-access full text" in block)
    check("no excerpts means no full-text notes at all",
          "Full text (open access" not in writer._format_sources(two))

    # -- the system prompt tells the truth about its inputs -----------------
    for old, new in writer._FULLTEXT_SUBSTITUTIONS:
        check(f"substitution target still present: {old[:40]}…", old in writer._WRITER_SYSTEM)
        check(f"and replaced in the full-text variant: {new[:40]}…",
              new in writer._WRITER_SYSTEM_FULLTEXT
              and old not in writer._WRITER_SYSTEM_FULLTEXT)
        check(f"briefing prompt still has the target: {old[:40]}…", old in writer._BRIEFING_SYSTEM)
        check(f"and the briefing full-text variant is substituted: {new[:40]}…",
              new in writer._BRIEFING_SYSTEM_FULLTEXT
              and old not in writer._BRIEFING_SYSTEM_FULLTEXT)

    # -- verification checks the shown excerpt, not the unseen tail ---------
    tail_figure = "the unseen tail says 77.7%"
    paper = Paper(title="t", abstract="Abstract only.",
                  full_text="Shown text reports 12.5% improvement. "
                            + "y" * sources.FULLTEXT_PER_PAPER_CHARS + tail_figure)
    art = {"abstract": "It improved by 12.5% but also 77.7%.", "sections": [],
           "key_points": [], "references": []}
    result = verify.check_statistics(art, [paper])
    check("figure in the shown excerpt verifies", "12.5%" not in result["unverified"])
    check("figure only in the unseen tail stays unverified",
          "77.7%" in result["unverified"])

    # -- Methods and Table 1 report what happened ---------------------------
    prov_ft = {"queries": ["q"], "databases": ["Europe PMC"], "model": "m",
               "full_text_sources": [1, 3]}
    parts = render._methods_paragraphs(prov_ft, 10, 2, "topic")
    check("Methods reports full texts read", "full texts of 2 sources were retrieved"
          in parts["handling"] and "read alongside them" in parts["handling"])
    parts_abs = render._methods_paragraphs({"queries": ["q"], "databases": ["Europe PMC"]},
                                           10, 2, "topic")
    check("abstracts-only Methods wording unchanged",
          "full texts were not retrieved" in parts_abs["handling"])

    # -- Methods describes the check that actually runs (issue #101) ---------
    for name, handling in (("full-text", parts["handling"]), ("abstracts-only",
                                                              parts_abs["handling"])):
        check(f"{name} Methods does not overclaim the statistical check",
              "Every numerical value" not in handling
              and "quantities carrying a clinical unit" in handling
              and "its own sentence cites" in handling
              and "other than the one cited" in handling)

    # The retraction must name what was searched: on a full-text draft the
    # check read more than the abstracts, and saying otherwise understates it.
    check("unverified wording follows what was read",
          "abstracts of the cited sources" in render._unverified_sentence(["9.9"], False)
          and "open-access full text where one was retrieved"
          in render._unverified_sentence(["9.9"], True))
    ft_cited = [Paper(title="a", abstract="x", full_text="body", year=2020),
                Paper(title="b", abstract="x", year=2021)]
    table = render._table_html(ft_cited, {1: "direct", 2: "related"})
    check("Table 1 grows a Read column in full-text mode",
          "<th>Read</th>" in table and "Full text" in table and "Abstract" in table)
    plain_table = render._table_html(
        [Paper(title="a", abstract="x", year=2020)], {1: "direct"})
    check("Read column is always present in Table 1",
          "<th>Read</th>" in plain_table and "Abstract" in plain_table)
    md = render._table_markdown(ft_cited, {1: "direct", 2: "related"})
    check("markdown table matches", "Read |" in md and "Full text |" in md)

    # -- every provenance statement agrees with Table 1 (issue #75) ----------
    # A shipped article said "full texts of 7 sources were retrieved" in Methods
    # and "prepared from abstracts alone" in Limitations, plus an
    # "Abstract-derived synthesis" masthead and a "(not full texts)" colophon:
    # Methods read provenance, the other four were hardcoded. The contradiction
    # is what these assert against, so keep them keyed to the *wording* a reader
    # sees rather than to the helpers that produce it.
    # `references` is load-bearing: `cited` is resolved from it, and an empty
    # list means nothing is cited, so every count below would be 0.
    art_ft = {"title": "T", "abstract": "A.", "sections": [], "key_points": [],
              "references": [1, 2], "keywords": []}
    html_ft = render.render_article(art_ft, ft_cited, "night shift work",
                                    curation={1: "direct", 2: "related"},
                                    provenance=prov_ft)
    md_ft = render.render_markdown(art_ft, ft_cited, "night shift work",
                                   curation={1: "direct", 2: "related"},
                                   provenance=prov_ft)
    for name, out in (("HTML", html_ft), ("Markdown", md_ft)):
        check(f"{name} never claims abstracts-only when a full text was read",
              "abstracts alone" not in out
              and "not full texts" not in out
              and "not the full texts" not in out
              and "Abstract-derived synthesis" not in out)
        check(f"{name} states the mixed grounding instead",
              "Full text read for 1 of 2 sources" in out)

    # The abstracts-only wording must survive untouched — it is the truthful one
    # whenever nothing was read in full, which is any topic with no open-access
    # literature.
    abs_cited = [Paper(title="a", abstract="x", year=2020)]
    art_abs = dict(art_ft, references=[1])
    html_abs = render.render_article(art_abs, abs_cited, "night shift work",
                                     curation={1: "direct"}, provenance={})
    check("abstracts-only article still says so",
          "abstracts alone" in html_abs and "Abstract-derived synthesis" in html_abs)


def test_section_aware_excerpts() -> None:
    """Results and discussion are shown even when they sit past a head-slice.

    The first 12,000 characters of a paper are the title, intro and methods.
    A figure that only appears in Results used to be unseen by the writer and
    unverifiable. With paperfetch-style ``## Results`` markers, the excerpt
    spends its budget on abstract, results, discussion/conclusions, methods,
    introduction — so that figure verifies. References never enter, so a
    volume number in the list cannot falsely verify a figure.
    """
    from articlegen import sources, verify
    from articlegen.sources import Paper, full_text_excerpts

    figure = "41.7%"
    intro = "Background on seclusion. " * 500
    check("the intro padding does not itself contain the figure",
          figure not in intro)

    marked = (
        "## Abstract\n\n"
        "A cluster trial of seclusion reduction.\n\n"
        "## Introduction\n\n"
        f"{intro}\n\n"
        "## Methods\n\n"
        "We randomised 12 wards.\n\n"
        "## Results\n\n"
        f"Seclusion episodes fell {figure} in the intervention wards.\n\n"
        "## Discussion\n\n"
        "The reduction held at six months.\n\n"
        "References\n\n"
        "Smith J. J Hosp 2020;12:99.9.\n"
    )
    check("the results figure sits past a head-slice",
          figure not in marked[:sources.FULLTEXT_PER_PAPER_CHARS])

    paper = Paper(title="Seclusion reduction trial",
                  abstract="A cluster trial of seclusion reduction.",
                  full_text=marked)
    shown = full_text_excerpts([paper])[1]
    check("the excerpt contains the results figure", figure in shown)
    check("sections appear in excerpt-priority order, not document order",
          shown.index("## Abstract") < shown.index("## Results")
          < shown.index("## Discussion") < shown.index("## Methods"))
    check("the long introduction is not taken first",
          shown.index("## Results") < (shown.find("## Introduction")
                                       if "## Introduction" in shown else len(shown)))
    check("references stay out of the excerpt",
          "99.9" not in shown and "Smith J" not in shown)
    check("marked text still fills the per-paper cap",
          len(shown) == sources.FULLTEXT_PER_PAPER_CHARS)

    art = {"abstract": f"Seclusion fell {figure} [1].", "sections": [],
           "key_points": [], "references": []}
    result = verify.check_statistics(art, [paper])
    check("a results-section figure verifies where a head-slice would miss it",
          figure not in result["unverified"])

    head = Paper(title=paper.title, abstract=paper.abstract,
                 full_text=marked[:sources.FULLTEXT_PER_PAPER_CHARS])
    head_result = verify.check_statistics(art, [head])
    check("the same figure is unverified against a head-slice of the same text",
          figure in head_result["unverified"])

    # Europe PMC JATS is mapped the same way as paperfetch: markers, no
    # reference list, heading not repeated in the body.
    jats = (
        "<article>"
        "<front><article-meta><abstract><p>A cluster trial of seclusion "
        "reduction.</p></abstract></article-meta></front>"
        "<body>"
        f"<sec><title>Introduction</title><p>{intro}</p></sec>"
        "<sec><title>Materials and Methods</title><p>We randomised 12 wards.</p></sec>"
        f"<sec><title>Results</title><p>Seclusion episodes fell {figure} "
        "in the intervention wards.</p></sec>"
        "<sec><title>Discussion</title><p>The reduction held at six months.</p></sec>"
        "<sec><title>References</title><p>Smith J. J Hosp 2020;12:99.9.</p></sec>"
        "</body></article>"
    )
    parsed = sources._parse_fulltext_xml(jats)
    check("JATS parse emits Results and Methods markers",
          "## Results" in parsed and "## Methods" in parsed)
    check("JATS Materials and Methods maps to ## Methods",
          "## Methods" in parsed and "Materials and Methods" not in parsed)
    check("JATS parse drops the reference list", "99.9" not in parsed)
    check("JATS abstract is marked", "## Abstract" in parsed)
    jats_shown = full_text_excerpts([
        Paper(title="t", abstract="a", full_text=parsed)])[1]
    check("JATS results figure reaches the excerpt", figure in jats_shown)

    both = (
        "## Discussion\n\nThe finding held.\n\n"
        "## Conclusions\n\nWe conclude nothing new.\n"
    )
    either = full_text_excerpts([
        Paper(title="t", abstract="a", full_text=both)])[1]
    check("discussion wins the discussion-or-conclusions slot",
          "## Discussion" in either and "## Conclusions" not in either)

    conclusions_only = (
        "## Methods\n\nWe did a thing.\n\n"
        "## Conclusions\n\nThe rate was 8.8%.\n"
    )
    conc = full_text_excerpts([
        Paper(title="t", abstract="a", full_text=conclusions_only)])[1]
    check("conclusions fill the slot when there is no discussion",
          "## Conclusions" in conc and "8.8%" in conc)

    unmarked = (
        "Intro text. Results: seclusion fell 41.7%.\n"
        "References\n"
        "Smith J. J Hosp 2020;12:99.9.\n"
    )
    plain = full_text_excerpts([
        Paper(title="t", abstract="a", full_text=unmarked)])[1]
    check("unmarked text still drops the reference list", "99.9" not in plain)
    check("unmarked text keeps the results figure", "41.7%" in plain)


def test_pipeline_fetches_full_text() -> None:
    """The pipeline stage itself: fetch only cited direct/related sources,
    then rewrite with those texts.

    Fetch used to run *before* the writer chose citations, so FULLTEXT_TARGET
    filled with uncited OA while cited Cochrane reviews stayed unread. Groq
    used to skip the step entirely; that path is gone. Tangential sources are
    still never fetched, even if cited.
    """
    from articlegen import pipeline
    from articlegen.sources import Paper

    papers = [Paper(title=f"p{i}", abstract="a", pmcid=f"PMC{i}", is_open_access=True)
              for i in range(1, 5)]
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [], "references": [1]}
    curation = {"relevance": {1: "direct", 2: "tangential", 3: "related", 4: "direct"},
                "most_relevant_index": 1,
                "counts": {"direct": 2, "related": 1, "tangential": 1}}
    fetched_pmcids: list[str] = []
    writes: list[list[int]] = []

    saved = (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
             pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
             pipeline.enforce_style)
    try:
        pipeline.plan_queries = lambda topic, **kw: (["q"], "core")
        def fake_gather(queries, **kw):
            kw.get("outcomes", []).append(
                {"source": "europe_pmc", "query": "q", "count": 4, "error": "", "cached": False})
            return papers
        pipeline.gather_evidence = fake_gather
        pipeline.curate_sources = lambda topic, p, **kw: curation

        def fake_write(topic, p, **kw):
            writes.append([i for i, x in enumerate(p, 1) if x.full_text])
            return dict(article)
        pipeline.write_article = fake_write
        pipeline.write_briefing = fake_write
        pipeline.fetch_full_text = (
            lambda p, use_cache=True: (fetched_pmcids.append(p.pmcid), "body text")[1])
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})

        draft = pipeline.generate_draft("topic")
        check("only cited sources are fetched (uncited direct OA is skipped)",
              fetched_pmcids == ["PMC1"])
        check("first draft is abstracts-only, rewrite sees the cited full text",
              writes == [[], [1]])
        check("papers carry their full text", draft.papers[0].full_text == "body text")
        check("provenance records which cited sources were read in full",
              draft.provenance["full_text_sources"] == [1])

        fetched_pmcids.clear()
        writes.clear()
        for p in papers:
            p.full_text = ""
            p.full_text_via = ""
        article["references"] = [1, 3, 4]
        draft = pipeline.generate_draft("topic")
        check("cited direct before related, tangential never",
              fetched_pmcids == ["PMC1", "PMC4", "PMC3"])
        check("rewrite sees all three cited full texts",
              writes[0] == [] and writes[1] == [1, 3, 4])
        check("provenance lists the cited full texts in index order",
              draft.provenance["full_text_sources"] == [1, 3, 4])

        # Unlabelled sources are never fetched. Reaching this through
        # generate_draft is no longer possible — an empty curation is a hard
        # stop (#168) — but the ordering function is where the rule lives, so
        # it is asserted here directly.
        from articlegen.sources import full_text_order
        check("no relevance labels means nothing is eligible for full text",
              full_text_order(papers, {}) == [])
    finally:
        (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
         pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
         pipeline.enforce_style) = saved


def test_unlabelled_sources_stop_the_run() -> None:
    """Curation swallows every exception and returns empty labels; the pipeline
    used to log a warning and write anyway, which turned a failed relevance
    gate into a normal-looking briefing with no topic-drift protection and no
    full text (#168). An unlabelled pool now raises CurationFailed before the
    writer or the named-source pass run.
    """
    import inspect
    from articlegen import pipeline, writer
    from articlegen.sources import Paper

    papers = [Paper(title=f"p{i}", abstract="abstract text", pmcid=f"PMC{i}", is_open_access=True)
              for i in range(1, 4)]
    logged: list[str] = []

    def _boom(*a, **kw):
        raise AssertionError("the writer must not run")

    saved_pipeline = (
        pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
        pipeline.write_article, pipeline.write_briefing, pipeline.enforce_style,
        pipeline._named_source_pass,
    )
    saved_writer = (writer.generate_json,)
    try:
        pipeline.plan_queries = lambda topic, **kw: (["q"], "core")
        def fake_gather(queries, **kw):
            kw.get("outcomes", []).append(
                {"source": "europe_pmc", "query": "q", "count": 3, "error": "", "cached": False})
            return papers
        pipeline.gather_evidence = fake_gather
        pipeline.curate_sources = lambda topic, p, **kw: {
            "relevance": {}, "most_relevant_index": None, "counts": {},
            "error": "RuntimeError: provider exploded",
        }
        pipeline.write_briefing = pipeline.write_article = _boom
        pipeline.enforce_style = _boom
        named_called = [0]
        def fake_named(*a, **kw):
            named_called[0] += 1
            return {"queries": [], "added": 0}
        pipeline._named_source_pass = fake_named

        raised = None
        try:
            pipeline.generate_draft("topic", log=logged.append)
        except pipeline.CurationFailed as exc:
            raised = exc

        check("empty labelling stops the run", raised is not None)
        check("and is caught by every existing NoPapersFound handler",
              isinstance(raised, pipeline.NoPapersFound))
        check("and is not blamed on the scholarly APIs",
              raised is not None and raised.sources_failed is False)
        check("the message names the reason",
              raised is not None and "provider exploded" in str(raised))
        check("the message says the write was not charged",
              raised is not None and "charged" in str(raised).lower())
        check("the writer is never called", True)
        check("the named-source pass is never reached", named_called[0] == 0)
        check("the WARNING log line survives",
              any("no usable labels" in line for line in logged))

        # curate_sources still degrades soft and now says why
        def _raise_nope(*a, **kw):
            raise RuntimeError("nope")
        writer.generate_json = _raise_nope
        res_exc = writer.curate_sources("topic", papers)
        check("curate_sources returns a dict on provider error", isinstance(res_exc, dict))
        check("curate_sources relevance is empty on provider error", res_exc.get("relevance") == {})
        check("curate_sources error names the exception", "nope" in res_exc.get("error", ""))

        writer.generate_json = lambda *a, **kw: {"assessments": []}
        res_empty = writer.curate_sources("topic", papers)
        check("curate_sources returns a dict on empty assessments", isinstance(res_empty, dict))
        check("curate_sources relevance is empty on empty assessments", res_empty.get("relevance") == {})
        check("curate_sources error describes empty assessments",
              bool(res_empty.get("error")) and isinstance(res_empty.get("error"), str))

        check("the named-source pass still degrades soft",
              "CurationFailed" not in inspect.getsource(pipeline._named_source_pass))
    finally:
        (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_article, pipeline.write_briefing, pipeline.enforce_style,
            pipeline._named_source_pass,
        ) = saved_pipeline
        (writer.generate_json,) = saved_writer


def test_full_text_comes_from_the_papers_cli_when_it_is_there() -> None:
    """The papers CLI (paperfetch) is tried first for open-access full text,
    falling back to Europe PMC when absent or failing soft.
    """
    import json
    import tempfile
    from dataclasses import dataclass
    from articlegen import paperfetch, pipeline, sources
    from articlegen.sources import Paper

    @dataclass
    class FakeProc:
        stdout: str = ""
        stderr: str = ""
        returncode: int = 0

    class FakeResp:
        def __init__(self, data=None, text=""):
            self._data, self.text = data, text

        def json(self):
            return self._data if self._data is not None else {}

    real_run = paperfetch.subprocess.run
    real_which = paperfetch.shutil.which
    real_get_with_retry = sources._get_with_retry

    def reset_paperfetch():
        paperfetch._AVAILABLE = None
        paperfetch._ARGV = []
        paperfetch._WARNED = False
        paperfetch._BATCH_OK = None
        paperfetch._BATCH_CACHE = {}
        sources.clear_search_cache()

    try:
        # 1. `ok` -> text returned, [3] stripped, full_text_via set to "papers"
        reset_paperfetch()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write("Body text with 441 participants [3].")
            tf_path = tf.name

        try:
            ok_record = json.dumps({
                "status": "ok", "doi": "10.1234/sample", "read": tf_path,
            })
            run_calls = []
            paperfetch.subprocess.run = lambda argv, **kw: (
                run_calls.append((argv, kw)), FakeProc(stdout=ok_record, returncode=0)
            )[1]
            paperfetch.shutil.which = lambda cmd: "papers" if cmd == "papers" else None

            p1 = Paper(title="p1", abstract="abs", doi="10.1234/sample")
            text1 = sources.fetch_full_text(p1)
            check("ok record returns text", "Body text with 441 participants" in text1)
            check("brackets stripped from papers text", "[3]" not in text1)
            check("paper.full_text_via is papers", p1.full_text_via == "papers")
            check("subprocess.run was called once", len(run_calls) == 1)

            # 2. Cached by DOI (different spelling: https://doi.org/10.1234/sample)
            p2 = Paper(title="p2", abstract="abs", doi="https://doi.org/10.1234/sample")
            text2 = sources.fetch_full_text(p2)
            check("second fetch with equivalent DOI served from cache", text2 == text1)
            check("cached hit sets full_text_via", p2.full_text_via == "papers")
            check("subprocess was not called a second time", len(run_calls) == 1)
        finally:
            if os.path.exists(tf_path):
                try:
                    os.remove(tf_path)
                except OSError:
                    pass

        # 3. Every non-ok status falls through to Europe PMC
        fake_xml = "<article><body><sec><title>Body</title><p>Europe PMC body text</p></sec></body></article>"
        epmc_calls = []
        sources._get_with_retry = lambda url, params, headers: (
            epmc_calls.append(url), FakeResp(text=fake_xml)
        )[1]

        def raise_timeout(argv):
            raise paperfetch.subprocess.TimeoutExpired(argv, 120)

        non_ok_cases = [
            ("queued_ckn", lambda argv, **kw: FakeProc(stdout=json.dumps({"status": "queued_ckn"}), returncode=2)),
            ("no_doi", lambda argv, **kw: FakeProc(stdout=json.dumps({"status": "no_doi"}), returncode=1)),
            ("exit 1 empty stdout", lambda argv, **kw: FakeProc(stdout="", returncode=1)),
            ("TimeoutExpired", lambda argv, **kw: raise_timeout(argv)),
            ("bad JSON", lambda argv, **kw: FakeProc(stdout="not json", returncode=0)),
            ("missing file", lambda argv, **kw: FakeProc(stdout=json.dumps({"status": "ok", "read": "C:\\nonexistent\\missing_12345.txt"}), returncode=0)),
        ]

        for case_name, runner in non_ok_cases:
            reset_paperfetch()
            epmc_calls.clear()
            paperfetch.shutil.which = lambda cmd: "papers"
            paperfetch.subprocess.run = runner

            # Test paper with both DOI and PMCID/is_open_access
            p = Paper(title="p", abstract="a", doi="10.9999/x", pmcid="PMC123", is_open_access=True)
            res = sources.fetch_full_text(p)
            check(f"non-ok case '{case_name}' falls through to Europe PMC", "Europe PMC body text" in res)
            check(f"non-ok case '{case_name}' sets full_text_via to europe_pmc", p.full_text_via == "europe_pmc")
            check(f"non-ok case '{case_name}' made Europe PMC request", len(epmc_calls) == 1)

        # 4. Not on PATH -> today's behaviour
        reset_paperfetch()
        epmc_calls.clear()
        run_called = []
        paperfetch.shutil.which = lambda cmd: None
        os.environ.pop("ARTICLEGEN_PAPERS_CMD", None)
        paperfetch.subprocess.run = lambda argv, **kw: (run_called.append(argv), FakeProc())[1]

        p_no_papers = Paper(title="p", abstract="a", doi="10.8888/y", pmcid="PMC456", is_open_access=True)
        res_no_papers = sources.fetch_full_text(p_no_papers)
        check("papers not on PATH -> available is False", not paperfetch.available())
        check("papers not on PATH -> subprocess.run never called", len(run_called) == 0)
        check("papers not on PATH -> Europe PMC produces text", "Europe PMC body text" in res_no_papers)
        check("papers not on PATH -> full_text_via is europe_pmc", p_no_papers.full_text_via == "europe_pmc")

        # 5. ARTICLEGEN_PAPERS_CMD="python -m papers"
        reset_paperfetch()
        cmd_calls = []
        os.environ["ARTICLEGEN_PAPERS_CMD"] = "python -m papers"
        try:
            paperfetch.subprocess.run = lambda argv, **kw: (
                cmd_calls.append((argv, kw)),
                FakeProc(stdout=json.dumps({"status": "no_doi"}), returncode=1),
            )[1]
            paperfetch.fetch_via_papers("10.5555/cmd")
            check("ARTICLEGEN_PAPERS_CMD was split into argv list", len(cmd_calls) == 1)
            argv_used, kw_used = cmd_calls[0]
            check("argv is list", isinstance(argv_used, list))
            check("argv matches expected split command + get + doi",
                  argv_used == ["python", "-m", "papers", "get", "10.5555/cmd"])
            check("shell=True is absent from kwargs", "shell" not in kw_used or not kw_used["shell"])
        finally:
            os.environ.pop("ARTICLEGEN_PAPERS_CMD", None)

        # 6. Pipeline loop: DOI-only paper fetched when papers available, skipped when not
        reset_paperfetch()
        doi_paper = Paper(title="doi only", abstract="a", doi="10.7777/doionly")
        article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
                   "key_points": [], "glossary": [], "references": [1]}
        curation = {"relevance": {1: "direct"}, "most_relevant_index": 1, "counts": {"direct": 1}}

        saved_pipe = (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
                      pipeline.write_article, pipeline.fetch_full_text, pipeline.enforce_style)
        try:
            pipeline.plan_queries = lambda topic, **kw: (["q"], "core")
            pipeline.gather_evidence = lambda queries, **kw: (
                kw.get("outcomes", []).append(
                    {"source": "openalex", "query": "q", "count": 1, "error": "", "cached": False}
                ),
                [doi_paper],
            )[1]
            pipeline.curate_sources = lambda topic, p, **kw: curation
            pipeline.write_article = lambda topic, p, **kw: dict(article)
            pipeline.write_briefing = pipeline.write_article
            pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})

            # With papers available:
            logs_with = []
            paperfetch.shutil.which = lambda cmd: "papers"
            paperfetch.subprocess.run = lambda argv, **kw: FakeProc(
                stdout=json.dumps({"mailto_set": True, "s2_key_set": True}),
                returncode=0)
            def fake_fetch_ft_papers(p, **kw):
                p.full_text_via = "papers"
                return "text from papers"
            pipeline.fetch_full_text = fake_fetch_ft_papers

            reset_paperfetch()
            draft_with = pipeline.generate_draft("topic", log=logs_with.append)
            check("pipeline fetches DOI-only paper when papers available",
                  draft_with.provenance["full_text_sources"] == [1])
            check("pipeline log contains 'via papers'", any("via papers" in ln for ln in logs_with))
            check("provenance records full_text_via breakdown",
                  draft_with.provenance.get("full_text_via") == {"papers": 1, "europe_pmc": 0})

            # With papers unavailable:
            logs_without = []
            paperfetch.shutil.which = lambda cmd: None
            doi_paper.full_text = ""
            doi_paper.full_text_via = ""
            reset_paperfetch()
            draft_without = pipeline.generate_draft("topic", log=logs_without.append)
            check("pipeline skips DOI-only paper when papers unavailable",
                  draft_without.provenance["full_text_sources"] == [])
            check("pipeline stopped because names no open-access copy",
                  any("1 had no open-access copy" in ln for ln in logs_without))
        finally:
            (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
             pipeline.write_article, pipeline.fetch_full_text, pipeline.enforce_style) = saved_pipe

        # 7. No env leak & PAPERS_MAILTO derivation
        reset_paperfetch()
        env_calls = []
        paperfetch.shutil.which = lambda cmd: "papers"
        paperfetch.subprocess.run = lambda argv, **kw: (
            env_calls.append((argv, kw)),
            FakeProc(stdout=json.dumps({"status": "no_doi"}), returncode=1),
        )[1]
        os.environ["OPENALEX_MAILTO"] = "test@example.com"
        os.environ.pop("PAPERS_MAILTO", None)
        before_env = dict(os.environ)
        try:
            paperfetch.fetch_via_papers("10.1111/envtest")
            after_env = dict(os.environ)
            check("fetch_via_papers does not leak into os.environ", before_env == after_env)
            check("subprocess received PAPERS_MAILTO from OPENALEX_MAILTO",
                  env_calls[0][1].get("env", {}).get("PAPERS_MAILTO") == "test@example.com")
            check("subprocess sets PAPERS_NO_CKN_QUEUE so drafts do not fill the miss list",
                  env_calls[0][1].get("env", {}).get("PAPERS_NO_CKN_QUEUE") == "1")
        finally:
            os.environ.pop("OPENALEX_MAILTO", None)
    finally:
        paperfetch.subprocess.run = real_run
        paperfetch.shutil.which = real_which
        sources._get_with_retry = real_get_with_retry
        reset_paperfetch()


def test_queued_ckn_counts_as_no_open_access() -> None:
    """`queued_ckn` and `no_oa` are both paywalled, not "OA but returned no text".

    Four Grok 4.6 runs (21 Aug 2026) stopped on target of 5, and paywalled
    landmarks (Tesnières 2026) were logged as OA failures because `papers`
    returning `queued_ckn` produced empty text, which the loop counted as
    `fetch_failed` (#191). `no_oa` is the public `paperfetch-oa` package's own
    name for the same outcome (#173) and is covered by the same checks. A
    timeout or unreadable PDF still is a failed OA fetch. The stop-reason and
    read-subset-skew lines still print.
    """
    import json
    from dataclasses import dataclass
    from articlegen import paperfetch, pipeline, sources
    from articlegen.sources import Paper

    @dataclass
    class FakeProc:
        stdout: str = ""
        stderr: str = ""
        returncode: int = 0

    real_run = paperfetch.subprocess.run
    real_which = paperfetch.shutil.which

    def reset_paperfetch():
        paperfetch._AVAILABLE = None
        paperfetch._ARGV = []
        paperfetch._WARNED = False
        paperfetch._BATCH_OK = None
        paperfetch._BATCH_CACHE = {}
        sources.clear_search_cache()

    def draft_logs(papers, runner):
        article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
                   "key_points": [], "glossary": [], "references": [1]}
        curation = {
            "relevance": {i: "direct" for i in range(1, len(papers) + 1)},
            "most_relevant_index": 1,
            "counts": {"direct": len(papers)},
        }
        lines = []
        saved = (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_article, pipeline.write_briefing, pipeline.enforce_style,
            pipeline._named_source_pass,
        )
        try:
            pipeline.plan_queries = lambda topic, **kw: (["q"], "core")

            def gather(queries, **kw):
                kw.get("outcomes", []).append(
                    {"source": "openalex", "query": "q", "count": len(papers),
                     "error": "", "cached": False})
                return papers

            pipeline.gather_evidence = gather
            pipeline.curate_sources = lambda t, p, **kw: curation
            pipeline.write_article = lambda t, p, **kw: dict(article)
            pipeline.write_briefing = pipeline.write_article
            pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
            pipeline._named_source_pass = lambda *a, **kw: {"queries": [], "added": 0}
            paperfetch.shutil.which = lambda cmd: "papers" if cmd == "papers" else None
            paperfetch.subprocess.run = runner
            reset_paperfetch()
            pipeline.generate_draft("topic", log=lines.append)
        finally:
            (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
             pipeline.write_article, pipeline.write_briefing, pipeline.enforce_style,
             pipeline._named_source_pass) = saved
            reset_paperfetch()
        return "\n".join(lines)

    try:
        # Status travels back; the string helper stays a string.
        reset_paperfetch()
        paperfetch.shutil.which = lambda cmd: "papers"
        paperfetch.subprocess.run = lambda argv, **kw: FakeProc(
            stdout=json.dumps({"status": "queued_ckn"}), returncode=2)
        text, status = paperfetch.fetch_via_papers_with_status("10.9999/paywalled")
        check("queued_ckn returns empty text", text == "")
        check("queued_ckn status is queued_ckn", status == "queued_ckn")
        check("fetch_via_papers still returns a string",
              isinstance(paperfetch.fetch_via_papers("10.9999/paywalled"), str))
        check("queued_ckn is in NOT_OA_STATUSES",
              "queued_ckn" in paperfetch.NOT_OA_STATUSES)

        # DOI-only paper: no Europe PMC fallthrough, flag set, empty text.
        reset_paperfetch()
        paperfetch.shutil.which = lambda cmd: "papers"
        paperfetch.subprocess.run = lambda argv, **kw: FakeProc(
            stdout=json.dumps({"status": "queued_ckn"}), returncode=2)
        paywalled = Paper(title="paywalled landmark", abstract="a",
                          doi="10.9999/paywalled")
        body = sources.fetch_full_text(paywalled)
        check("queued_ckn DOI-only fetch is empty", body == "")
        check("queued_ckn sets full_text_not_oa", paywalled.full_text_not_oa is True)
        check("queued_ckn does not set full_text_via", paywalled.full_text_via == "")

        # Pipeline log: paywalled landmark is no-open-access, not an OA miss.
        paywalled_for_draft = Paper(title="paywalled landmark", abstract="a",
                                    doi="10.9999/paywalled")
        queued = draft_logs(
            [paywalled_for_draft],
            lambda argv, **kw: FakeProc(
                stdout=json.dumps({"status": "queued_ckn"}), returncode=2),
        )
        check("queued_ckn run still prints the stop-reason line",
              "stopped because:" in queued)
        check("queued_ckn run still prints the read-subset-skew line",
              "read-subset skew:" in queued)
        check("queued_ckn increments no-open-access",
              "1 had no open-access copy" in queued)
        check("queued_ckn does not count as OA-but-empty",
              "0 were open access but returned no text" in queued)

        # `no_oa` is the public paperfetch-oa package's equivalent status (#173) —
        # same meaning as queued_ckn, no CKN ladder behind it.
        reset_paperfetch()
        paperfetch.shutil.which = lambda cmd: "papers"
        paperfetch.subprocess.run = lambda argv, **kw: FakeProc(
            stdout=json.dumps({"status": "no_oa"}), returncode=2)
        text, status = paperfetch.fetch_via_papers_with_status("10.9999/paywalled")
        check("no_oa returns empty text", text == "")
        check("no_oa status is no_oa", status == "no_oa")
        check("no_oa is in NOT_OA_STATUSES",
              "no_oa" in paperfetch.NOT_OA_STATUSES)

        # Negative controls: a fetch failure is not a confirmed absence of OA.
        check("unreadable_pdf is not in NOT_OA_STATUSES",
              "unreadable_pdf" not in paperfetch.NOT_OA_STATUSES)
        check("retry is not in NOT_OA_STATUSES",
              "retry" not in paperfetch.NOT_OA_STATUSES)
        check("no_doi is not in NOT_OA_STATUSES",
              "no_doi" not in paperfetch.NOT_OA_STATUSES)

        reset_paperfetch()
        paperfetch.shutil.which = lambda cmd: "papers"
        paperfetch.subprocess.run = lambda argv, **kw: FakeProc(
            stdout=json.dumps({"status": "no_oa"}), returncode=2)
        no_oa_paper = Paper(title="paywalled landmark", abstract="a",
                            doi="10.9999/paywalled")
        body = sources.fetch_full_text(no_oa_paper)
        check("no_oa DOI-only fetch is empty", body == "")
        check("no_oa sets full_text_not_oa", no_oa_paper.full_text_not_oa is True)
        check("no_oa does not set full_text_via", no_oa_paper.full_text_via == "")

        no_oa_for_draft = Paper(title="paywalled landmark", abstract="a",
                                doi="10.9999/paywalled")
        no_oa_run = draft_logs(
            [no_oa_for_draft],
            lambda argv, **kw: FakeProc(
                stdout=json.dumps({"status": "no_oa"}), returncode=2),
        )
        check("no_oa increments no-open-access",
              "1 had no open-access copy" in no_oa_run)
        check("no_oa does not count as OA-but-empty",
              "0 were open access but returned no text" in no_oa_run)

        # Contrast: a timeout on a DOI-only paper is still a failed OA fetch.
        timed_out = Paper(title="timeout paper", abstract="a",
                          doi="10.9999/timeout")

        def raise_timeout(argv, **kw):
            raise paperfetch.subprocess.TimeoutExpired(argv, 120)

        timed = draft_logs([timed_out], raise_timeout)
        check("timeout increments fetch_failed",
              "1 were open access but returned no text" in timed)
        check("timeout does not increment no-open-access",
              "0 had no open-access copy" in timed)

        check("paywalled cited sources are listed under one heading",
              "Paywalled cited sources (one CKN download away):" in queued)
        check("the paywalled DOI is on the list",
              "10.9999/paywalled" in queued)
    finally:
        paperfetch.subprocess.run = real_run
        paperfetch.shutil.which = real_which
        reset_paperfetch()


def test_ckn_loop() -> None:
    """draft --queue-ckn uses `papers queue`; rerun skips search and sees ingest."""
    import inspect
    import tempfile
    from articlegen import cli, paperfetch, pipeline, sources, web
    from articlegen.pipeline import Draft
    from articlegen.sources import Paper

    class FakeProc:
        def __init__(self, stdout="", returncode=0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    real_run = paperfetch.subprocess.run
    real_which = paperfetch.shutil.which

    def reset():
        paperfetch._AVAILABLE = None
        paperfetch._ARGV = []
        paperfetch._WARNED = False
        paperfetch._BATCH_OK = None
        paperfetch._BATCH_CACHE = {}
        paperfetch._TRIED = {}
        sources.clear_search_cache()

    check("cmd_draft does not call papers queue itself",
          "queue_paywalled" not in inspect.getsource(cli.cmd_draft))
    check("cmd_draft still goes through generate_draft",
          "generate_draft(" in inspect.getsource(cli.cmd_draft))
    check("the hosted app never queues for CKN",
          "queue-ckn" not in inspect.getsource(web)
          and "queue_paywalled" not in inspect.getsource(web))
    check("rerun skips search",
          "gather_evidence(" not in inspect.getsource(pipeline.rerun_draft))
    check("rerun skips curation",
          "curate_sources(" not in inspect.getsource(pipeline.rerun_draft))
    check("rerun skips plan_queries",
          "plan_queries(" not in inspect.getsource(pipeline.rerun_draft))
    check("get still sets PAPERS_NO_CKN_QUEUE",
          'env["PAPERS_NO_CKN_QUEUE"] = "1"' in inspect.getsource(paperfetch._child_env))

    paywalled = Paper(
        title="Closed landmark", abstract="a", doi="10.9999/paywalled",
        venue="JAMA", year=2018, full_text_not_oa=True, oa_tried="europepmc,unpaywall",
    )
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [], "references": [1],
               "findings": ["a finding [1]"]}
    original = Draft(
        topic="topic",
        article=article,
        papers=[paywalled],
        curation={"relevance": {1: "direct"}, "counts": {"direct": 1}},
        provenance={"queries": ["q"], "databases": ["OpenAlex"],
                    "full_text_sources": [], "date": "1 September 2026"},
    )

    try:
        reset()
        paperfetch.shutil.which = lambda cmd: "papers" if cmd == "papers" else None
        queue_calls = []

        def queue_runner(argv, **kw):
            queue_calls.append(list(argv))
            return FakeProc(stdout=json.dumps({"status": "queued", "doi": "10.9999/paywalled"}))

        paperfetch.subprocess.run = queue_runner
        n = paperfetch.queue_paywalled(original.paywalled_cited())
        check("queue_paywalled queues one DOI", n == 1)
        check("queue_paywalled uses papers queue, not get",
              "queue" in queue_calls[0] and "get" not in queue_calls[0])
        check("queue_paywalled passes title", "--title" in queue_calls[0])
        check("queue_paywalled passes tried from the first run",
              "europepmc,unpaywall" in queue_calls[0])

        saved_generate = cli.generate_draft
        cli.generate_draft = lambda *a, **kw: original
        with tempfile.TemporaryDirectory() as d:
            saved_cwd = os.getcwd()
            os.chdir(d)
            try:
                queue_calls.clear()
                check("draft without the flag does not queue",
                      cli.main(["draft", "topic", "--name", "x"]) == 0)
                check("no papers queue process on a default draft", queue_calls == [])
                queue_calls.clear()
                check("draft --queue-ckn exits 0",
                      cli.main(["draft", "topic", "--name", "y", "--queue-ckn"]) == 0)
                check("draft --queue-ckn called papers queue",
                      any("queue" in c for c in queue_calls))
            finally:
                os.chdir(saved_cwd)
        cli.generate_draft = saved_generate

        # rerun: first-run paper was paywalled; ingest now returns full text.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write("Ingested full text: 41.7% responded.")
            tf_path = tf.name
        reset()
        paperfetch.shutil.which = lambda cmd: "papers" if cmd == "papers" else None

        def ingest_runner(argv, **kw):
            argv = list(argv)
            if "status" in argv:
                return FakeProc(stdout=json.dumps({"mailto_set": True, "s2_key_set": True}))
            return FakeProc(stdout=json.dumps({
                "status": "ok", "doi": "10.9999/paywalled", "read": tf_path,
            }))

        paperfetch.subprocess.run = ingest_runner
        saved = (
            pipeline.write_briefing, pipeline.write_article,
            pipeline.enforce_style, pipeline.enforce_statistics,
        )
        rewritten = []

        def fake_compose(topic, papers, **kw):
            rewritten.append([p.full_text for p in papers])
            out = dict(article)
            out["findings"] = ["41.7% responded [1]"]
            return out

        pipeline.write_briefing = fake_compose
        pipeline.write_article = fake_compose
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
        pipeline.enforce_statistics = lambda a, papers, **kw: (a, {"unverified": []}, kw.get("style_report") or {})
        try:
            with tempfile.TemporaryDirectory() as d:
                saved_cwd = os.getcwd()
                os.chdir(d)
                try:
                    os.makedirs("drafts")
                    man = os.path.join("drafts", "first.json")
                    with open(man, "w", encoding="utf-8") as f:
                        json.dump(original.to_dict(), f)
                    check("rerun exits 0", cli.main(["rerun", man]) == 0)
                    rerun_path = os.path.join("drafts", "first-rerun.json")
                    check("rerun writes a new manifest", os.path.isfile(rerun_path))
                    with open(rerun_path, encoding="utf-8") as f:
                        second = Draft.from_dict(json.load(f))
                    check("rerun attached ingested full text",
                          "41.7%" in (second.papers[0].full_text or ""))
                    check("Methods records the second run's full-text sources",
                          second.provenance.get("full_text_sources") == [1])
                    check("the writer saw the ingested text",
                          rewritten and "41.7%" in (rewritten[0][0] or ""))
                    check("rerun kept the first run's queries",
                          second.provenance.get("queries") == ["q"])
                finally:
                    os.chdir(saved_cwd)
        finally:
            (pipeline.write_briefing, pipeline.write_article,
             pipeline.enforce_style, pipeline.enforce_statistics) = saved
            os.unlink(tf_path)
    finally:
        paperfetch.subprocess.run = real_run
        paperfetch.shutil.which = real_which
        reset()


def test_refresh_reports_new_direct() -> None:
    """`refresh` re-runs a manifest's queries, labels only what's new, and
    prints the new direct records without writing anything unless --rewrite."""
    import inspect
    import io
    import contextlib
    import tempfile
    from articlegen import cli, pipeline, sources
    from articlegen.pipeline import Draft
    from articlegen.sources import Paper

    check("refresh_draft never plans queries",
          "plan_queries(" not in inspect.getsource(pipeline.refresh_draft))
    check("refresh_draft never writes a file",
          "open(" not in inspect.getsource(pipeline.refresh_draft))

    known = Paper(title="Known study", abstract="a", doi="10.1111/known",
                  venue="JAMA", year=2020)
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [], "references": [1],
               "findings": ["a finding [1]"]}
    original = Draft(
        topic="topic",
        article=article,
        papers=[known],
        curation={"relevance": {1: "direct"}, "counts": {"direct": 1}},
        provenance={"queries": ["seclusion reduction"], "databases": ["OpenAlex"],
                    "full_text_sources": [], "date": "1 September 2026"},
    )

    dup = Paper(title="Known study", abstract="a", doi="10.1111/known",
                venue="JAMA", year=2020)
    new = Paper(title="Brand new trial", abstract="b", doi="10.2222/new",
                venue="BMJ", year=2026)

    saved_gather = pipeline.gather_evidence
    saved_curate = pipeline.curate_sources
    saved_compose = (pipeline.write_briefing, pipeline.write_article)
    saved_enforce = (pipeline.enforce_style, pipeline.enforce_statistics)

    gather_calls = []
    curate_calls = []
    compose_calls = []

    def fake_gather(queries, **kw):
        gather_calls.append(list(queries))
        return [dup, new]

    def fake_curate(topic, papers, **kw):
        curate_calls.append(list(papers))
        return {"relevance": {1: "direct"}, "counts": {"direct": 1}}

    def fake_compose(topic, papers, **kw):
        compose_calls.append(topic)
        out = dict(article)
        out["references"] = [1, 2]
        out["findings"] = ["a finding [1]", "a new finding [2]"]
        return out

    pipeline.gather_evidence = fake_gather
    pipeline.curate_sources = fake_curate
    pipeline.write_briefing = fake_compose
    pipeline.write_article = fake_compose
    pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
    pipeline.enforce_statistics = lambda a, papers, **kw: (a, {"unverified": []}, kw.get("style_report") or {})

    try:
        sources.clear_search_cache()
        with tempfile.TemporaryDirectory() as d:
            saved_cwd = os.getcwd()
            os.chdir(d)
            try:
                os.makedirs("drafts")
                man = os.path.join("drafts", "first.json")
                with open(man, "w", encoding="utf-8") as f:
                    json.dump(original.to_dict(), f)
                with open(man, "rb") as f:
                    before = f.read()
                before_listing = sorted(os.listdir("drafts"))

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    check("refresh exits 0", cli.main(["refresh", man]) == 0)
                out = buf.getvalue()

                check("refresh re-ran the manifest's queries",
                      gather_calls and gather_calls[0] == ["seclusion reduction"])
                check("only the new record was labelled",
                      len(curate_calls) == 1 and len(curate_calls[0]) == 1
                      and curate_calls[0][0].doi == "10.2222/new")
                check("stdout names the new record", "Brand new trial" in out)
                check("stdout shows the manifest's date", "1 September 2026" in out)
                check("stdout carries the refresh summary",
                      "REFRESH_SUMMARY: 1 new direct record" in out)
                check("stdout does not report the already-screened record as new",
                      "Known study" not in out)

                with open(man, "rb") as f:
                    after = f.read()
                check("the manifest is untouched", before == after)
                check("no refresh files were written",
                      not os.path.isfile(os.path.join("drafts", "first-refresh.json"))
                      and not os.path.isfile(os.path.join("drafts", "first-refresh.html"))
                      and not os.path.isfile(os.path.join("drafts", "first-refresh.md"))
                      and not os.path.isfile(os.path.join("drafts", "index.html")))
                check("drafts/ directory listing is unchanged",
                      sorted(os.listdir("drafts")) == before_listing)
                check("no compose call happened without --rewrite", compose_calls == [])

                compose_calls.clear()
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    check("refresh --rewrite exits 0",
                          cli.main(["refresh", man, "--rewrite"]) == 0)
                refresh_path = os.path.join("drafts", "first-refresh.json")
                check("--rewrite wrote a new manifest", os.path.isfile(refresh_path))
                with open(refresh_path, encoding="utf-8") as f:
                    rewritten = Draft.from_dict(json.load(f))
                check("the rewritten draft has both papers", len(rewritten.papers) == 2)
                check("the second paper is the new record",
                      rewritten.papers[1].doi == "10.2222/new")
                check("the new record is labelled direct",
                      rewritten.curation.get("relevance", {}).get(2) == "direct")
                check("provenance records what it was refreshed from",
                      rewritten.provenance.get("refreshed_from", {}).get("date")
                      == "1 September 2026")
            finally:
                os.chdir(saved_cwd)
    finally:
        pipeline.gather_evidence = saved_gather
        pipeline.curate_sources = saved_curate
        pipeline.write_briefing, pipeline.write_article = saved_compose
        pipeline.enforce_style, pipeline.enforce_statistics = saved_enforce
        sources.clear_search_cache()


def test_one_papers_process_per_run() -> None:
    """18 eligible DOIs make one `papers get` process, and a missing mailto
    is named once before that get.

    Per-DOI `papers get` was up to 18 processes, each with a 120s timeout,
    so paperfetch's Semantic Scholar 429 memo died with every process. Batch
    mode (paperfetch PR #36) takes the ordered list on stdin (`papers get -`)
    and prints one JSON line per DOI. An older CLI that treats `-` as a query
    still falls back to one process per DOI.
    """
    import json
    import tempfile
    from dataclasses import dataclass
    from articlegen import paperfetch, pipeline, sources
    from articlegen.sources import Paper

    @dataclass
    class FakeProc:
        stdout: str = ""
        stderr: str = ""
        returncode: int = 0

    real_run = paperfetch.subprocess.run
    real_which = paperfetch.shutil.which
    saved_env = {k: os.environ.get(k) for k in
                 ("PAPERS_MAILTO", "OPENALEX_MAILTO", "UNPAYWALL_EMAIL",
                  "SEMANTIC_SCHOLAR_API_KEY", "ARTICLEGEN_PAPERS_CMD",
                  # The assertions below assume the local prefetch limit
                  # (`MAX_FULLTEXT_REQUESTS`), not the hosted one
                  # (`FULLTEXT_TARGET`) — this test must not depend on
                  # whatever the ambient shell happens to have set.
                  "ARTICLEGEN_STATELESS")}

    def reset_paperfetch():
        paperfetch._AVAILABLE = None
        paperfetch._ARGV = []
        paperfetch._WARNED = False
        paperfetch._BATCH_OK = None
        paperfetch._BATCH_CACHE = {}
        sources.clear_search_cache()

    def argv_cmd(argv):
        argv = list(argv)
        if "status" in argv:
            return "status"
        if "get" in argv and argv[-1] == "-":
            return "get-batch"
        if "get" in argv:
            return "get-one"
        return "other"

    n = pipeline.MAX_FULLTEXT_REQUESTS
    dois = [f"10.9999/batch-{i}" for i in range(1, n + 1)]
    papers = [Paper(title=f"p{i}", abstract="a", doi=doi)
              for i, doi in enumerate(dois, start=1)]
    order = list(range(1, n + 1))
    text_path = ""

    try:
        for key in saved_env:
            os.environ.pop(key, None)
        paperfetch.shutil.which = lambda cmd: "papers" if cmd == "papers" else None

        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write("Batch full text body.")
            text_path = tf.name

        def run_retrieve(runner):
            reset_paperfetch()
            paperfetch.subprocess.run = runner
            lines = []
            fetched, eligible, no_oa, failed, spent, stopped = (
                pipeline._retrieve_full_texts(papers, order, lines.append))
            return fetched, eligible, no_oa, failed, spent, stopped, lines

        # 1. Batch CLI: one status, one get -, 18 stdin lines, target of 5 kept.
        calls = []

        def batch_runner(argv, **kw):
            calls.append((list(argv), kw.get("input") or "", argv_cmd(argv)))
            kind = argv_cmd(argv)
            if kind == "status":
                return FakeProc(stdout=json.dumps({
                    "mailto_set": False, "s2_key_set": False,
                }))
            if kind == "get-batch":
                stdin_dois = [ln.strip() for ln in (kw.get("input") or "").splitlines()
                              if ln.strip()]
                lines = [json.dumps({
                    "status": "ok", "doi": d, "read": text_path,
                }) for d in stdin_dois]
                return FakeProc(stdout="\n".join(lines) + "\n", returncode=0)
            doi = argv[-1]
            return FakeProc(stdout=json.dumps({
                "status": "ok", "doi": doi, "read": text_path,
            }))

        fetched, eligible, no_oa, failed, spent, stopped, logs = run_retrieve(batch_runner)
        kinds = [c[2] for c in calls]
        batch_calls = [c for c in calls if c[2] == "get-batch"]
        one_calls = [c for c in calls if c[2] == "get-one"]
        status_calls = [c for c in calls if c[2] == "status"]
        stdin_dois = [ln.strip() for ln in batch_calls[0][1].splitlines() if ln.strip()] if batch_calls else []
        log_text = "\n".join(logs)

        check("18 eligible DOIs make one papers get process",
              kinds.count("get-batch") == 1 and kinds.count("get-one") == 0)
        check("the batch process is papers get -",
              batch_calls[0][0][-2:] == ["get", "-"])
        check("stdin carries every eligible DOI, in order", stdin_dois == dois)
        check("papers status ran once", len(status_calls) == 1)
        check("status ran before the get",
              kinds.index("status") < kinds.index("get-batch"))
        check("a run with no mailto says so in one line",
              sum(1 for ln in logs if "mailto_set" in ln) == 1)
        check("the missing line names mailto_set and s2_key_set",
              "papers: missing mailto_set, s2_key_set" in log_text)
        check("preflight log is the first papers line",
              next(ln for ln in logs if "papers:" in ln).startswith("  papers: missing"))
        check("the loop still stops at FULLTEXT_TARGET",
              len(fetched) == pipeline.FULLTEXT_TARGET
              and stopped == f"target of {pipeline.FULLTEXT_TARGET} reached")
        check("the first FULLTEXT_TARGET papers were attached",
              fetched == list(range(1, pipeline.FULLTEXT_TARGET + 1)))
        check("attached text came from the batch read path",
              all(papers[i - 1].full_text == "Batch full text body." for i in fetched))

        # 2. Older CLI: one JSON object for `get -` (treats `-` as the query).
        for paper in papers:
            paper.full_text = ""
            paper.full_text_via = ""
            paper.full_text_not_oa = False
        calls.clear()

        def old_cli_runner(argv, **kw):
            calls.append((list(argv), kw.get("input") or "", argv_cmd(argv)))
            kind = argv_cmd(argv)
            if kind == "status":
                return FakeProc(stdout=json.dumps({
                    "mailto_set": True, "s2_key_set": True,
                }))
            if kind == "get-batch":
                return FakeProc(stdout=json.dumps({
                    "status": "no_doi", "agent_next": "notify_human",
                }), returncode=1)
            doi = argv[-1]
            return FakeProc(stdout=json.dumps({
                "status": "ok", "doi": doi, "read": text_path,
            }))

        fetched_old, *_rest, logs_old = run_retrieve(old_cli_runner)
        kinds_old = [c[2] for c in calls]
        check("older papers falls back to one process per DOI",
              kinds_old.count("get-batch") == 1
              and kinds_old.count("get-one") == n)
        check("the fallback log names batch/stdin",
              any("no batch/stdin" in ln for ln in logs_old))
        check("fallback still fills FULLTEXT_TARGET",
              len(fetched_old) == pipeline.FULLTEXT_TARGET)
        check("a configured mailto is not reported as missing",
              not any("mailto_set" in ln for ln in logs_old))
    finally:
        paperfetch.subprocess.run = real_run
        paperfetch.shutil.which = real_which
        reset_paperfetch()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if os.path.exists(text_path):
            try:
                os.remove(text_path)
            except OSError:
                pass


def test_hosted_fulltext_prefetch_is_the_target() -> None:
    """Hosted drafts send FULLTEXT_TARGET DOIs to papers, not the request cap."""
    from articlegen import pipeline

    saved = os.environ.get("ARTICLEGEN_STATELESS")
    try:
        os.environ.pop("ARTICLEGEN_STATELESS", None)
        check("local prefetch is the request cap",
              pipeline._fulltext_prefetch_limit() == pipeline.MAX_FULLTEXT_REQUESTS)
        os.environ["ARTICLEGEN_STATELESS"] = "1"
        check("hosted prefetch is the target",
              pipeline._fulltext_prefetch_limit() == pipeline.FULLTEXT_TARGET)
    finally:
        if saved is None:
            os.environ.pop("ARTICLEGEN_STATELESS", None)
        else:
            os.environ["ARTICLEGEN_STATELESS"] = saved


def test_papers_batch_keeps_partial_results_on_timeout() -> None:
    """A killed `papers get -` still yields the JSON lines it already printed."""
    import subprocess
    import tempfile

    from articlegen import paperfetch

    real_run = paperfetch.subprocess.run
    real_which = paperfetch.shutil.which
    paperfetch._AVAILABLE = True
    paperfetch._ARGV = ["papers"]
    paperfetch._BATCH_OK = None
    paperfetch._BATCH_CACHE = {}
    text_path = ""
    try:
        paperfetch.shutil.which = (
            lambda cmd: "papers" if cmd == "papers" else None)
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write("Kept from the timed-out batch.")
            text_path = tf.name

        def runner(argv, **kw):
            raise subprocess.TimeoutExpired(
                cmd=argv, timeout=1,
                output=(json.dumps({
                    "status": "ok", "doi": "10.1/a", "read": text_path,
                }) + "\n{\"status\": \"ok\", \"doi\": \"10.1/b\", \"read\": \""),
            )

        paperfetch.subprocess.run = runner
        results = paperfetch.fetch_many_via_papers(
            ["10.1/a", "10.1/b", "10.1/c"], timeout=1)
        check("completed JSON line is kept",
              results[0][0] == "Kept from the timed-out batch.")
        check("truncated and unreached DOIs are empty, not a fallback to per-DOI",
              results[1] == ("", "") and results[2] == ("", "")
              and len(results) == 3)
    finally:
        paperfetch.subprocess.run = real_run
        paperfetch.shutil.which = real_which
        paperfetch._AVAILABLE = None
        paperfetch._ARGV = []
        paperfetch._BATCH_OK = None
        paperfetch._BATCH_CACHE = {}
        if text_path and os.path.exists(text_path):
            try:
                os.remove(text_path)
            except OSError:
                pass


def test_full_text_order_favours_reviews_and_trials() -> None:
    """Deep reads go to direct systematic reviews and trials first, not the newest papers.

    #143 fixed 'deep reads went to old, heavily-cited work' by sorting on recency
    within relevance. That created the opposite skew: in the seclusion draft,
    Gaynes 2017 (the only systematic appraisal of adult acute settings) was
    abstract-only while newer primary pilots and child reviews were read in full
    (#166). Relevance tier first, then study design (reviews -> trials -> other),
    then year, then search rank.

    The negative controls in paper_design are the specification.
    """
    from articlegen.sources import Paper, full_text_order, paper_design

    # 1. Full ordering: direct review (2016) beats newer direct trial (2019)
    # and newer direct primary (2024); related review follows all directs.
    papers = [
        Paper(title="Study 1: a systematic review and meta-analysis", abstract="a", year=2016),  # 1: direct synthesis (2016)
        Paper(title="Study 2: observational cohort study", abstract="a", year=2024),            # 2: direct other (2024)
        Paper(title="Study 3: a randomised controlled trial", abstract="a", year=2019),         # 3: direct trial (2019)
        Paper(title="Study 4: an umbrella review", abstract="a", year=2025),                     # 4: related synthesis (2025)
        Paper(title="Study 5: systematic review of tangential topic", abstract="a", year=2025), # 5: tangential synthesis (2025)
        Paper(title="Study 6: unlabelled paper", abstract="a", year=2026),                      # 6: unlabelled
        Paper(title="Study 7: direct undated primary study", abstract="a"),                     # 7: direct undated other
    ]
    relevance = {1: "direct", 2: "direct", 3: "direct", 4: "related", 5: "tangential", 7: "direct"}

    order = full_text_order(papers, relevance)
    check("direct review, direct trial, direct primary, direct undated, then related review",
          order == [1, 3, 2, 7, 4])
    check("relevance outranks design: direct other precedes related review",
          order.index(2) < order.index(4))
    check("tangential sources are never offered for fetch", 5 not in order)
    check("an unlabelled source is never offered either", 6 not in order)

    # 2. Recency inside a design tier: newer trial beats older trial (#143 behaviour survives)
    trials = [
        Paper(title="Trial A: a randomised controlled trial", abstract="a", year=2019),
        Paper(title="Trial B: a randomised controlled trial", abstract="a", year=2024),
    ]
    check("recency decides within a design tier (newer first)",
          full_text_order(trials, {1: "direct", 2: "direct"}) == [2, 1])

    # 3. Ties fall back to incoming search rank
    same = [Paper(title=f"p{i}", abstract="a", year=2020) for i in range(1, 5)]
    check("equal tier, design, and year keeps the incoming rank order",
          full_text_order(same, {i: "direct" for i in range(1, 5)}) == [1, 2, 3, 4])
    check("no labels means nothing is fetched",
          full_text_order(same, {}) == [])

    # 4. paper_design directly: positives and negative controls
    # synthesis positives
    check("paper_design identifies systematic review and meta-analysis in title",
          paper_design(Paper(title="Efficacy of light therapy: a systematic review and meta-analysis", abstract="a")) == "synthesis")
    check("paper_design identifies Cochrane Database of Systematic Reviews venue",
          paper_design(Paper(title="Interventions for seclusion", venue="Cochrane Database of Systematic Reviews", abstract="a")) == "synthesis")
    check("paper_design identifies systematic review in publication_types",
          paper_design(Paper(title="Seclusion reduction", abstract="a", publication_types=("systematic review",))) == "synthesis")

    # trial positives
    check("paper_design identifies randomised controlled trial in title",
          paper_design(Paper(title="Containment in acute wards: a randomised controlled trial", abstract="a")) == "trial")
    check("paper_design identifies cluster-randomized trial in title",
          paper_design(Paper(title="Safety planning: a cluster-randomized trial", abstract="a")) == "trial")
    check("paper_design identifies Randomized Controlled Trial in publication_types",
          paper_design(Paper(title="Crisis planning", abstract="a", publication_types=("randomized controlled trial",))) == "trial")
    check("paper_design identifies RCT acronym in title",
          paper_design(Paper(title="Safewards: an RCT in acute psychiatric wards", abstract="a")) == "trial")

    # other (the negative controls that are the specification)
    check("paper_design demotes study protocol of an RCT to other",
          paper_design(Paper(title="Protocol for a randomised controlled trial of seclusion reduction", abstract="a")) == "other")
    check("paper_design demotes narrative review to other",
          paper_design(Paper(title="Seclusion in acute wards: a narrative review", abstract="a")) == "other")
    check("paper_design ignores bare 'review' in publication_types",
          paper_design(Paper(title="Care models in psychiatry", abstract="a", publication_types=("review",))) == "other")
    check("paper_design ignores bare 'trials' in title",
          paper_design(Paper(title="The trials of implementing a new model of care", abstract="a")) == "other")
    check("paper_design treats plain cohort study as other",
          paper_design(Paper(title="A cohort study of seclusion in mental health wards", abstract="a")) == "other")
    check("paper_design treats scoping review without systematic as other",
          paper_design(Paper(title="A scoping review of restraint reduction", abstract="a")) == "other")

    # 5. Preprints are never down-ranked
    check("a preprint of a trial still ranks as a trial",
          paper_design(Paper(title="Intervention: a randomised trial", abstract="a", is_preprint=True)) == "trial")


def test_pmcid_is_resolved_by_doi() -> None:
    """A paper found via OpenAlex can still reach its open-access full text.

    Only the Europe PMC *search* returns pmcid/inEPMC, so before `resolve_pmcid`
    a paper was fetchable only if that search happened to be the one that found
    it. Everything from OpenAlex was abstract-only regardless of licence — on a
    real run, 9 of 11 available full texts were lost that way.
    """
    from articlegen import pipeline, sources
    from articlegen.sources import Paper

    class FakeResp:
        def __init__(self, data=None, text=""):
            self._data, self.text = data, text
        def json(self):
            return self._data

    def record(item):
        return {"resultList": {"result": ([item] if item else [])}}

    known = {
        "10.3389/fpsyt.2017.00156": {"pmcid": "PMC5572353", "isOpenAccess": "Y", "inEPMC": "Y"},
        "10.1002/ams2.182": {"pmcid": "PMC5667234", "isOpenAccess": "N", "inEPMC": "Y"},
    }
    calls: list[str] = []

    def fake_get(url, params, headers):
        # `resolve_pmcid` now also runs `check_record_status` (#242), which
        # calls OpenAlex and Crossref by DOI. This test is about the Europe
        # PMC lookup specifically, so only that call is tracked in `calls`;
        # the retraction/correction checks get an empty, harmless answer.
        if url == sources.OPENALEX_URL:
            return FakeResp({"results": []})
        if url.startswith("https://api.crossref.org"):
            return FakeResp({"message": {}})
        query = (params or {}).get("query", "")
        calls.append(query)
        doi = query.removeprefix('DOI:"').removesuffix('"')
        return FakeResp(record(known.get(doi)))

    real = sources._get_with_retry
    try:
        sources._get_with_retry = fake_get
        sources.clear_search_cache()

        oa = Paper(title="t", abstract="a", doi="https://doi.org/10.3389/fpsyt.2017.00156")
        check("a DOI lookup finds the open-access record", sources.resolve_pmcid(oa))
        check("the pmcid is written back onto the paper", oa.pmcid == "PMC5572353")

        sources.resolve_pmcid(oa)
        check("an already-resolved paper costs no second request", len(calls) == 1)

        cached = Paper(title="t2", abstract="a", doi="10.3389/fpsyt.2017.00156")
        check("a second paper with the same DOI is served from cache",
              sources.resolve_pmcid(cached) and len(calls) == 1)

        not_oa = Paper(title="t3", abstract="a", doi="10.1002/ams2.182")
        check("a PMCID without open access is not fetchable", not sources.resolve_pmcid(not_oa))

        unknown = Paper(title="t4", abstract="a", doi="10.9999/nope")
        check("a DOI Europe PMC does not hold stays abstract-only",
              not sources.resolve_pmcid(unknown) and unknown.pmcid == "")

        no_doi = Paper(title="t5", abstract="a")
        before = len(calls)
        check("no DOI means no lookup",
              not sources.resolve_pmcid(no_doi) and len(calls) == before)
    finally:
        sources._get_with_retry = real
        sources.clear_search_cache()

    # -- the pipeline stops at the target, not at the end of the list --------
    papers = [Paper(title=f"p{i}", abstract="a", doi=f"10.1/{i}") for i in range(1, 11)]
    curation = {"relevance": {i: "direct" for i in range(1, 11)},
                "most_relevant_index": 1, "counts": {"direct": 10}}
    article = {"title": "t", "abstract": "x", "keywords": [], "sections": [],
               "key_points": [], "glossary": [],
               "references": list(range(1, 11))}
    resolved: list[str] = []

    saved = (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
             pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
             pipeline.resolve_pmcid, pipeline.enforce_style)
    try:
        pipeline.plan_queries = lambda topic, **kw: (["q"], "core")
        def fake_gather(queries, **kw):
            kw.get("outcomes", []).append(
                {"source": "europe_pmc", "query": "q", "count": 10, "error": "", "cached": False})
            return papers
        pipeline.gather_evidence = fake_gather
        pipeline.curate_sources = lambda topic, p, **kw: curation
        pipeline.write_article = lambda topic, p, **kw: dict(article)
        pipeline.write_briefing = pipeline.write_article
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})

        def fake_resolve(paper, use_cache=True, log=None):
            resolved.append(paper.doi)
            paper.pmcid, paper.is_open_access = "PMC" + paper.doi[-1], True
            return True
        pipeline.resolve_pmcid = fake_resolve
        pipeline.fetch_full_text = lambda p, use_cache=True: "body text"

        draft = pipeline.generate_draft("topic")
        check("the pipeline stops at the full-text target",
              draft.provenance["full_text_sources"] == [1, 2, 3, 4, 5])
        check("it does not resolve DOIs it will never fetch", len(resolved) == 5)
        check("the target matches what the excerpt budget can show",
              pipeline.FULLTEXT_TARGET * sources.FULLTEXT_PER_PAPER_CHARS
              <= sources.FULLTEXT_TOTAL_CHARS)

        # -- a topic with no open-access literature must not spend a request
        #    per paper against the one API that reliably answers -------------
        for p in papers:
            p.pmcid, p.is_open_access, p.full_text = "", False, ""
        resolved.clear()
        pipeline.resolve_pmcid = (
            lambda paper, use_cache=True, log=None: (resolved.append(paper.doi), False)[1])
        draft = pipeline.generate_draft("topic")
        check("no open access means no full text",
              draft.provenance["full_text_sources"] == [])
        check("lookups are bounded when nothing is fetchable",
              len(resolved) <= pipeline.MAX_FULLTEXT_REQUESTS)
    finally:
        (pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
         pipeline.write_article, pipeline.write_briefing, pipeline.fetch_full_text,
         pipeline.resolve_pmcid, pipeline.enforce_style) = saved


def test_unpaywall_fallback_in_resolve_pmcid() -> None:
    """When Europe PMC does not yield an open-access PMCID copy, Unpaywall is queried."""
    from articlegen import sources
    from articlegen.sources import Paper

    class FakeResp:
        def __init__(self, data=None):
            self._data = data
        def json(self):
            return self._data

    calls: list[str] = []

    def fake_get(url, params, headers):
        calls.append(url)
        if "api.unpaywall.org" in url:
            if "10.1000/unpaywall-pmcid" in url:
                return FakeResp({
                    "is_oa": True,
                    "best_oa_location": {
                        "pmcid": "PMC9876543",
                        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9876543",
                    },
                })
            elif "10.1000/unpaywall-oa-no-pmcid" in url:
                return FakeResp({
                    "is_oa": True,
                    "best_oa_location": {
                        "url": "https://journal.org/paper.pdf",
                    },
                })
            elif "10.1000/unpaywall-closed" in url:
                return FakeResp({"is_oa": False, "best_oa_location": None})
        return FakeResp({"resultList": {"result": []}})

    real = sources._get_with_retry
    try:
        sources._get_with_retry = fake_get
        sources.clear_search_cache()

        p1 = Paper(title="t1", abstract="a", doi="10.1000/unpaywall-pmcid")
        check("Unpaywall fallback resolves open-access paper with PMCID", sources.resolve_pmcid(p1))
        check("PMCID extracted from Unpaywall", p1.pmcid == "PMC9876543")
        check("is_open_access set to True", p1.is_open_access is True)

        p2 = Paper(title="t2", abstract="a", doi="10.1000/unpaywall-oa-no-pmcid")
        check("Unpaywall without PMCID returns False for fetchable fulltext", not sources.resolve_pmcid(p2))
        check("is_open_access set to True for OA paper without PMCID", p2.is_open_access is True)
        check("PMCID remains empty", p2.pmcid == "")

        p3 = Paper(title="t3", abstract="a", doi="10.1000/unpaywall-closed")
        check("Closed access paper returns False", not sources.resolve_pmcid(p3))
        check("is_open_access remains False", p3.is_open_access is False)
    finally:
        sources._get_with_retry = real
        sources.clear_search_cache()


def test_named_papers_in_abstracts_are_looked_up() -> None:
    """Papers named in the top abstracts are looked up in a second gather pass.

    Pinned by issue #165: after curation, abstracts are scanned for DOIs and study
    names, an extra gather runs (capped), new records are appended and re-curated,
    and provenance records the pass.
    """
    from articlegen import pipeline, sources
    from articlegen.sources import Paper

    # Setup papers: top abstracts contain names and DOIs
    p1 = Paper(
        title="Review of containment methods",
        abstract="The Safewards trial (doi: 10.1001/jama.2015.1234) showed substantial reduction.",
        year=2024, doi="10.1000/review1",
    )
    p2 = Paper(
        title="Early psychosis interventions",
        abstract="In the RAISE-ETP study, comprehensive care improved outcomes.",
        year=2023, doi="10.1000/review2",
    )
    p3 = Paper(
        title="Third relevant paper",
        abstract="A standard abstract with no named trials or DOIs.",
        year=2022, doi="10.1000/review3",
    )
    p4 = Paper(
        title="Fourth paper",
        abstract="Mentions 10.9999/fourth.abstract in a 4th-ranked abstract.",
        year=2021, doi="10.1000/review4",
    )
    p5 = Paper(
        title="Fifth paper",
        abstract="Fifth abstract background.",
        year=2020, doi="10.1000/review5",
    )
    initial_papers = [p1, p2, p3, p4, p5]

    initial_curation = {
        "relevance": {1: "direct", 2: "direct", 3: "related", 4: "related", 5: "tangential"},
        "most_relevant_index": 1,
        "counts": {"direct": 2, "related": 2, "tangential": 1},
    }
    article = {
        "title": "Topic", "abstract": "Summary", "keywords": [], "sections": [],
        "key_points": [], "glossary": [], "references": [1],
    }

    safewards_matched = Paper(
        title="The Safewards cluster randomised controlled trial in acute psychiatric wards",
        abstract="Original Safewards trial abstract text.",
        year=2015, doi="10.1001/jama.2015.1234",
    )

    gather_calls: list[dict] = []
    curate_calls: list[list[Paper]] = []

    saved = (
        pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
        pipeline.write_briefing, pipeline.write_article, pipeline.fetch_full_text,
        pipeline.enforce_style, pipeline.resolve_pmcid,
    )
    try:
        pipeline.plan_queries = lambda topic, **kw: (["query 1"], "core")

        def fake_gather(queries, **kw):
            gather_calls.append({
                "queries": list(queries),
                "patient": kw.get("patient", True),
                "exhausted": kw.get("exhausted"),
                "max_papers": kw.get("max_papers"),
            })
            if len(gather_calls) == 1:
                # Main search returns initial papers
                outcomes = kw.get("outcomes")
                if outcomes is not None:
                    outcomes.append({"source": "europe_pmc", "query": queries[0], "count": 5, "error": "", "cached": False})
                return list(initial_papers)
            else:
                # Named pass returns matched paper
                return [safewards_matched]

        pipeline.gather_evidence = fake_gather

        def fake_curate(topic, papers, **kw):
            curate_calls.append(list(papers))
            if len(curate_calls) == 1:
                return dict(initial_curation)
            # Second call for new papers: 1-based index
            return {"relevance": {1: "direct"}, "most_relevant_index": 1, "counts": {"direct": 1}}

        pipeline.curate_sources = fake_curate
        pipeline.write_briefing = lambda topic, p, **kw: dict(article)
        pipeline.write_article = pipeline.write_briefing
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
        pipeline.resolve_pmcid = lambda p, **kw: False
        pipeline.fetch_full_text = lambda p, **kw: ""

        draft = pipeline.generate_draft("topic")

        # 1. Second gather_evidence call happened
        check("second gather_evidence call happened", len(gather_calls) == 2)
        second_queries = gather_calls[1]["queries"]
        check("second gather includes the DOI printed in top abstract",
              "10.1001/jama.2015.1234" in second_queries)
        check("second gather includes the trial name",
              "Safewards trial" in second_queries)
        check("second gather includes study name from second top abstract",
              "RAISE-ETP study" in second_queries)
        check("only top NAMED_SOURCE_SCAN abstracts were scanned (4th abstract not scanned)",
              "10.9999/fourth.abstract" not in second_queries)

        # 2. Second call shares exhausted set and has patient=False
        check("second call receives patient=False", gather_calls[1]["patient"] is False)
        check("second call receives the same exhausted set object as the first",
              gather_calls[0]["exhausted"] is gather_calls[1]["exhausted"]
              and isinstance(gather_calls[1]["exhausted"], set))

        # 3. New paper is appended at N+1 without moving existing papers
        check("first paper identity is preserved at index 0", draft.papers[0] is p1)
        check("second paper identity is preserved at index 1", draft.papers[1] is p2)
        check("new paper is appended at index 5 (1-based index 6)",
              len(draft.papers) == 6 and draft.papers[5] is safewards_matched)

        # 4. Curate_sources was called a second time with only new papers
        check("second curate_sources received exactly the new papers",
              len(curate_calls) == 2 and curate_calls[1] == [safewards_matched])
        check("merged relevance holds label for new paper at index 6",
              draft.curation["relevance"].get(6) == "direct")
        check("merged relevance retains existing labels 1..5",
              all(draft.curation["relevance"].get(i) == initial_curation["relevance"][i] for i in range(1, 6)))
        check("curation counts are recomputed",
              draft.curation["counts"] == {"direct": 3, "related": 2, "tangential": 1})

        # 5. Provenance records named_sources
        check("provenance contains named_sources", "named_sources" in draft.provenance)
        check("provenance records queries and added count",
              draft.provenance["named_sources"]["added"] == 1
              and "10.1001/jama.2015.1234" in draft.provenance["named_sources"]["queries"])

        # 6. Caps check: abstract with 20 DOIs requests at most NAMED_SOURCE_LIMIT
        gather_calls.clear()
        curate_calls.clear()
        many_dois_abstract = " ".join(f"10.1000/doi{i}" for i in range(20))
        p_many = Paper(title="Many DOIs", abstract=many_dois_abstract, year=2024, doi="10.1000/many")

        def fake_gather_caps(queries, **kw):
            gather_calls.append({"queries": list(queries)})
            if len(gather_calls) == 1:
                return [p_many]
            # Return 20 matching papers
            return [Paper(title=f"Paper {q}", abstract="a", doi=q) for q in queries] * 4

        pipeline.gather_evidence = fake_gather_caps
        pipeline.curate_sources = lambda topic, p, **kw: {"relevance": {i: "direct" for i in range(1, len(p) + 1)}, "most_relevant_index": 1, "counts": {"direct": len(p)}}

        draft_caps = pipeline.generate_draft("topic")
        check("queries requested is capped at NAMED_SOURCE_LIMIT",
              len(gather_calls[1]["queries"]) <= sources.NAMED_SOURCE_LIMIT)
        check("new records added is capped at NAMED_SOURCE_LIMIT",
              len(draft_caps.papers) <= 1 + sources.NAMED_SOURCE_LIMIT)

        # 7. Nothing named -> no second gather and no named_sources key
        gather_calls.clear()
        plain_paper = Paper(title="Plain", abstract="No names or DOIs here.", year=2024, doi="10.1000/plain")
        pipeline.gather_evidence = lambda queries, **kw: (gather_calls.append(queries), [plain_paper])[1]
        pipeline.curate_sources = lambda topic, p, **kw: {"relevance": {1: "direct"}, "most_relevant_index": 1, "counts": {"direct": 1}}

        draft_none = pipeline.generate_draft("topic")
        check("nothing named -> gather_evidence called only once", len(gather_calls) == 1)
        check("nothing named -> no named_sources in provenance", "named_sources" not in draft_none.provenance)
    finally:
        (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_briefing, pipeline.write_article, pipeline.fetch_full_text,
            pipeline.enforce_style, pipeline.resolve_pmcid,
        ) = saved


def test_named_references_reads_names_not_noise() -> None:
    """The extractor pulls DOIs and named studies, rejecting noise and apparatus words."""
    from articlegen.sources import NAMED_SOURCE_LIMIT, _NAMED_STOPLIST, named_references

    # Positive controls
    pos1 = named_references("the Safewards cluster randomised controlled trial")
    check("extracts Safewards trial", pos1 == ["Safewards trial"])

    pos2 = named_references("(doi: 10.1001/jama.2015.1234)")
    check("extracts DOI from abstract", pos2 == ["10.1001/jama.2015.1234"])

    pos3 = named_references("the RAISE-ETP study")
    check("extracts RAISE-ETP study", pos3 == ["RAISE-ETP study"])

    pos4 = named_references("a stepped-wedge trial (STAR)")
    check("extracts parenthesised acronym as STAR trial", pos4 == ["STAR trial"])

    # Negative controls
    check("rejects sentence-initial 'This study'", named_references("This study found…") == [])
    check("rejects sentence-initial 'Recent trials'", named_references("Recent trials suggest…") == [])
    check("rejects apparatus acronym (RCT)", named_references("a randomised controlled trial (RCT)") == [])
    check("rejects PRISMA guideline mention", named_references("reported using PRISMA") == [])
    check("rejects sentence-initial 'The trial'", named_references("The trial was registered") == [])

    # Sweep all 16 abstracts in tests/real_abstracts.json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_abstracts_path = os.path.join(root, "tests", "real_abstracts.json")
    with open(real_abstracts_path, encoding="utf-8") as f:
        real_abstracts = json.load(f)

    check("real_abstracts has 16 entries", len(real_abstracts) == 16)
    for i, entry in enumerate(real_abstracts, start=1):
        refs = named_references(entry.get("abstract", ""))
        check(f"real abstract {i} extraction stays within NAMED_SOURCE_LIMIT",
              len(refs) <= NAMED_SOURCE_LIMIT)
        for ref in refs:
            tokens = ref.lower().split()
            for tok in tokens:
                clean_tok = tok.strip(".,;:\"'()")
                # The trailing study noun is permitted (e.g. trial, study); verify name tokens aren't stoplisted apparatus words
                if clean_tok in ("trial", "study", "programme", "program", "cohort", "rct", "intervention"):
                    continue
                check(f"real abstract {i} extracted query token {tok!r} is not in stoplist",
                      clean_tok not in _NAMED_STOPLIST)


def test_generic_named_lookups_are_skipped() -> None:
    """A generic 'name' is never looked up and names that match most records are dropped (#190)."""
    from articlegen import pipeline
    from articlegen.sources import (
        NAMED_MATCH_MIN_MATCHES, NAMED_MATCH_RATE_MAX, Paper,
        filter_named_matches, named_references,
    )

    # Negative controls (each must return [])
    negatives = [
        "RESULTS Twelve studies were included in the review.",
        "We included Twelve studies of seclusion.",
        "We searched for English-language trials of seclusion reduction.",
        "The ED intervention reduced restraint use.",
        "Three ED interventions were compared.",
        "Data from the US trial and UK trials were combined.",
        "A population-based cohort study followed adults.",
        "BACKGROUND Several trials have examined restraint.",
        # Live Luna runs looked these up from Cochrane abstracts (#190 missed them).
        "The Cochrane Schizophrenia Group's trial register was searched.",
        "We identified RCTs trial reports of pimozide versus placebo.",
        "The International Clinical trial registry was searched for unpublished studies.",
    ]
    for text in negatives:
        check(f"named_references rejects: {text[:35]!r}", named_references(text) == [])

    # Positive controls (must still be extracted)
    check("named_references extracts Safewards trial",
          named_references("the Safewards cluster randomised controlled trial") == ["Safewards trial"])
    check("named_references extracts RAISE-ETP study",
          named_references("In the RAISE-ETP study, comprehensive care improved outcomes.") == ["RAISE-ETP study"])
    check("named_references extracts DOI",
          named_references("(doi: 10.1001/jama.2015.1234)") == ["10.1001/jama.2015.1234"])
    check("named_references extracts parenthesised acronym",
          named_references("a stepped-wedge trial (STAR)") == ["STAR trial"])
    check("named_references orders DOI first then real name",
          named_references("In the Safewards trial (doi: 10.1001/jama.2015.1234), restraint dropped.") == [
              "10.1001/jama.2015.1234", "Safewards trial"
          ])

    # filter_named_matches tests
    # 1. 16 records, a name request matching 15 of them -> request dropped, none of its records kept
    records_16 = [
        Paper(title=f"Twelve study comparison in adult unit #{i}", abstract="text")
        for i in range(16)
    ]
    records_16[15] = Paper(title="Completely unrelated study of restraint", abstract="text")
    kept, dropped = filter_named_matches(records_16, ["Twelve study"])
    check("filter_named_matches drops request matching 15 of 16 records", dropped == ["Twelve study"])
    check("filter_named_matches keeps 0 records when only request is dropped", kept == [])

    # 2. Same 16 records, a name request matching 2 -> kept
    records_16_safewards = [
        Paper(title=f"General psychiatric unit report #{i}", abstract="text")
        for i in range(16)
    ]
    records_16_safewards[0] = Paper(title="Safewards model implementation in acute wards", abstract="text")
    records_16_safewards[1] = Paper(title="Safewards outcomes in inpatient psychiatry", abstract="text")
    kept, dropped = filter_named_matches(records_16_safewards, ["Safewards trial"])
    check("filter_named_matches keeps request matching 2 of 16 records", dropped == [])
    check("filter_named_matches keeps matching records for surviving request", len(kept) == 2)

    # 3. A DOI request matching every record -> never dropped (a DOI identifies one work)
    doi_records = [
        Paper(title=f"Paper #{i}", abstract="text", doi="10.1001/jama.2015.1234")
        for i in range(10)
    ]
    kept, dropped = filter_named_matches(doi_records, ["10.1001/jama.2015.1234"])
    check("filter_named_matches never drops a DOI request (identifies one work)", dropped == [] and len(kept) == 10)

    # 4. 3 records, a name matching 2 (0.67 rate, below NAMED_MATCH_MIN_MATCHES) -> kept
    small_records = [
        Paper(title="Safewards trial part 1", abstract="text"),
        Paper(title="Safewards trial part 2", abstract="text"),
        Paper(title="Unrelated study", abstract="text"),
    ]
    kept, dropped = filter_named_matches(small_records, ["Safewards trial"])
    check("filter_named_matches keeps high-rate match below MIN_MATCHES floor", dropped == [] and len(kept) == 2)

    # 5. A record matching one dropped and one surviving request -> kept
    records_mixed = [
        Paper(title="Twelve study of Safewards in psychiatric wards", abstract="text")
    ]
    records_mixed.extend([
        Paper(title=f"Twelve study comparison #{i}", abstract="text") for i in range(15)
    ])
    kept, dropped = filter_named_matches(records_mixed, ["Twelve study", "Safewards trial"])
    check("filter_named_matches drops generic name and keeps specific one",
          dropped == ["Twelve study"])
    check("filter_named_matches keeps record that matches surviving request",
          len(kept) == 1 and kept[0].title == "Twelve study of Safewards in psychiatric wards")

    # 6. Empty records and empty requests -> ([], [])
    check("filter_named_matches returns ([], []) for empty inputs", filter_named_matches([], []) == ([], []))
    check("filter_named_matches returns ([], []) for empty records", filter_named_matches([], ["Safewards trial"]) == ([], []))
    check("filter_named_matches returns ([], []) for empty requests", filter_named_matches(small_records, []) == ([], []))

    # Pipeline end-to-end: a second gather that returns records all matching one generic-behaving name adds no new papers
    p_init = Paper(title="Initial paper", abstract="The Safewards trial showed reduction.", year=2024, doi="10.1000/p1")
    initial_curation = {
        "relevance": {1: "direct"},
        "most_relevant_index": 1,
        "counts": {"direct": 1, "related": 0, "tangential": 0},
    }
    article = {
        "title": "Topic", "abstract": "Summary", "keywords": [], "sections": [],
        "key_points": [], "glossary": [], "references": [1],
    }
    # Second gather returns 16 records that all match a generic term, but do not match Safewards
    generic_results = [
        Paper(title=f"Twelve study comparison in psychiatric wards #{i}", abstract="text", year=2020 + i, doi=f"10.2000/g{i}")
        for i in range(16)
    ]

    saved = (
        pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
        pipeline.write_briefing, pipeline.write_article, pipeline.fetch_full_text,
        pipeline.enforce_style, pipeline.resolve_pmcid,
    )
    gather_calls: list[dict] = []
    try:
        pipeline.plan_queries = lambda topic, **kw: (["query 1"], "core")

        def fake_gather(queries, **kw):
            gather_calls.append(dict(kw))
            if len(gather_calls) == 1:
                return [p_init]
            return generic_results

        pipeline.gather_evidence = fake_gather
        pipeline.curate_sources = lambda topic, papers, **kw: dict(initial_curation)
        pipeline.write_briefing = lambda topic, p, **kw: dict(article)
        pipeline.write_article = pipeline.write_briefing
        pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
        pipeline.resolve_pmcid = lambda p, **kw: False
        pipeline.fetch_full_text = lambda p, **kw: ""

        draft = pipeline.generate_draft("topic")
        check("pipeline adds no generic-behaving records to draft.papers",
              len(draft.papers) == 1 and draft.papers[0].title == "Initial paper")
    finally:
        (
            pipeline.plan_queries, pipeline.gather_evidence, pipeline.curate_sources,
            pipeline.write_briefing, pipeline.write_article, pipeline.fetch_full_text,
            pipeline.enforce_style, pipeline.resolve_pmcid,
        ) = saved


def test_named_sources_merge_without_renumbering() -> None:
    """merge_candidates enriches duplicates and appends new records; named_matches filters properly."""
    from articlegen.sources import Paper, merge_candidates, named_matches

    # 1. named_matches
    p_doi = Paper(title="Any Title", abstract="a", doi="10.1001/jama.2015.1234")
    check("named_matches accepts exact DOI",
          named_matches(p_doi, "10.1001/jama.2015.1234"))
    check("named_matches accepts formatted DOI URL",
          named_matches(p_doi, "https://doi.org/10.1001/jama.2015.1234"))
    check("named_matches rejects different DOI",
          not named_matches(p_doi, "10.1001/other.5678"))

    p_title = Paper(
        title="The Safewards cluster randomised controlled trial in acute psychiatric wards",
        abstract="a",
        doi="10.1016/j.ijnurstu.2015.07.001",
    )
    check("named_matches accepts matching study name",
          named_matches(p_title, "Safewards trial"))
    check("named_matches rejects non-matching study name",
          not named_matches(p_title, "RAISE-ETP study"))

    # 2. merge_candidates dedupe and append
    original_p1 = Paper(title="Original Paper Title", abstract="orig abstract", doi="10.1234/test.doi")
    pool = [original_p1]

    # Duplicates by DOI (different formats) and title
    dup1 = Paper(title="Original Paper Title Subtitle", abstract="dup abstract", doi="https://doi.org/10.1234/TEST.DOI", year=2021)
    dup2 = Paper(title="original paper title", abstract="dup2 abstract", doi="doi: 10.1234/test.doi", citation_count=42)

    appended = merge_candidates(pool, [dup1, dup2])
    check("duplicates are not appended", appended == [])
    check("pool length unchanged after duplicates", len(pool) == 1)
    check("original paper identity preserved", pool[0] is original_p1)
    check("metadata enriched onto original paper (year)", pool[0].year == 2021)
    check("metadata enriched onto original paper (citation_count)", pool[0].citation_count == 42)

    # Genuine new papers with limit
    new1 = Paper(title="New Paper One", abstract="a", doi="10.9999/new1")
    new2 = Paper(title="New Paper Two", abstract="a", doi="10.9999/new2")
    new3 = Paper(title="New Paper Three", abstract="a", doi="10.9999/new3")

    appended_new = merge_candidates(pool, [new1, new2, new3], limit=2)
    check("merge_candidates honours limit", len(appended_new) == 2 and appended_new == [new1, new2])
    check("pool has 3 papers", len(pool) == 3)
    check("existing indices never move", pool[0] is original_p1 and pool[1] is new1 and pool[2] is new2)


def test_referenced_works_fetch_asks_openalex_correctly() -> None:
    """The one-hop citation fetches (issue #243) ask OpenAlex exactly what they claim to.

    Pins the request shape for openalex_work, openalex_records and
    openalex_citing_works: filters, select fields, and the no-request early exits.
    """
    from articlegen import sources
    from articlegen.sources import Paper

    check("referenced_works is selected on every OpenAlex record",
          "referenced_works" in sources._OA_FIELDS)

    calls: list[tuple[str, dict]] = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    real_get = sources._get_with_retry
    try:
        # openalex_work: a paper with a DOI sends filter=doi:<doi> and returns
        # the work id plus reference-list ids.
        def fake_get_work(url, params, headers):
            calls.append((url, dict(params)))
            return FakeResp({"results": [{"id": "https://openalex.org/W100",
                                           "referenced_works": ["https://openalex.org/W1",
                                                                 "https://openalex.org/W2"]}]})

        sources._get_with_retry = fake_get_work
        work_id, ref_ids = sources.openalex_work(Paper(title="A review", abstract="a", doi="10.1/review"))
        check("openalex_work sends filter=doi:<doi>",
              "filter=doi:10.1/review" in calls[-1][1].get("filter", "") or
              calls[-1][1].get("filter") == "filter=doi:10.1/review" or
              calls[-1][1].get("filter") == "doi:10.1/review")
        check("openalex_work returns the work id", work_id == "https://openalex.org/W100")
        check("openalex_work returns the reference-list ids",
              ref_ids == ["https://openalex.org/W1", "https://openalex.org/W2"])

        calls.clear()
        no_doi_id, no_doi_refs = sources.openalex_work(Paper(title="No DOI", abstract="a", doi=""))
        check("a paper with no DOI makes no request", calls == [])
        check("no DOI returns empty id and empty list", no_doi_id == "" and no_doi_refs == [])

        # openalex_records: batched request, filter with OR-joined ids and has_abstract.
        calls.clear()

        def fake_get_records(url, params, headers):
            calls.append((url, dict(params)))
            return FakeResp({"results": [
                {"id": "https://openalex.org/W1", "title": "Referenced paper",
                 "abstract_inverted_index": {"Some": [0], "abstract": [1]}},
            ]})

        sources._get_with_retry = fake_get_records
        recs = sources.openalex_records(["W1", "W2"])
        check("openalex_records makes exactly one request", len(calls) == 1)
        check("openalex_records filter contains openalex_id:W1|W2",
              "openalex_id:W1|W2" in calls[-1][1]["filter"])
        check("openalex_records filter contains has_abstract:true",
              "has_abstract:true" in calls[-1][1]["filter"])
        check("openalex_records returns one paper", len(recs) == 1)

        calls.clear()
        empty_recs = sources.openalex_records([])
        check("an empty id list makes no request", calls == [])
        check("an empty id list returns no papers", empty_recs == [])

        calls.clear()
        many_ids = [f"W{i}" for i in range(sources.REFERENCED_ID_LIMIT + 20)]
        sources.openalex_records(many_ids)
        sent_filter = calls[-1][1]["filter"]
        sent_ids = sent_filter.split(",")[0].split(":", 1)[1].split("|")
        check("more than REFERENCED_ID_LIMIT ids are trimmed to that many",
              len(sent_ids) == sources.REFERENCED_ID_LIMIT)

        # openalex_citing_works: cites: filter, sorted by citation count.
        calls.clear()

        def fake_get_citing(url, params, headers):
            calls.append((url, dict(params)))
            return FakeResp({"results": []})

        sources._get_with_retry = fake_get_citing
        sources.openalex_citing_works("W9")
        check("openalex_citing_works sends filter=cites:W9",
              "cites:W9" in calls[-1][1]["filter"])
        check("openalex_citing_works sorts by cited_by_count:desc",
              calls[-1][1].get("sort") == "cited_by_count:desc")

        calls.clear()
        empty_citing = sources.openalex_citing_works("")
        check("an empty work id makes no request", calls == [])
        check("an empty work id returns no papers", empty_citing == [])
    finally:
        sources._get_with_retry = real_get


def test_referenced_works_merge_without_renumbering() -> None:
    """One-hop citation follow-up (issue #243) merges without renumbering, capped, shared exhausted.

    Pinned by the "Done means" of issue #243: a referenced work merges into the
    pool without moving any existing record, only the new records are
    re-curated, and the pass shares the run's exhausted set.
    """
    from articlegen import pipeline, sources
    from articlegen.sources import Paper

    p1 = Paper(title="A systematic review and meta-analysis of seclusion reduction",
               abstract="review abstract", year=2023, doi="10.1000/synthesis1",
               publication_types=["systematic review"])
    p2 = Paper(title="A randomised controlled trial of seclusion reduction",
               abstract="trial abstract", year=2022, doi="10.1000/trial1",
               publication_types=["randomized controlled trial"])
    p3 = Paper(title="Related background paper", abstract="related", year=2021, doi="10.1000/related1")
    p4 = Paper(title="Direct but neither design", abstract="direct other", year=2020, doi="10.1000/other1")
    p5 = Paper(title="Tangential paper", abstract="tangential", year=2019, doi="10.1000/tangential1")
    papers = [p1, p2, p3, p4, p5]

    curation = {
        "relevance": {1: "direct", 2: "direct", 3: "related", 4: "direct", 5: "tangential"},
        "most_relevant_index": 1,
        "counts": {"direct": 3, "related": 1, "tangential": 1},
    }

    work_calls: list[Paper] = []
    citing_calls: list[str] = []
    curate_calls: list[list[Paper]] = []

    known_referenced = Paper(title="A known referenced record", abstract="ref abstract",
                              year=2010, doi="10.1000/known-referenced", citation_count=500)
    new_papers_offered = [known_referenced] + [
        Paper(title=f"New referenced record {i}", abstract="a", doi=f"10.1000/new{i}", citation_count=i)
        for i in range(11)
    ]
    check("twelve distinct new papers are offered", len(new_papers_offered) == 12)

    saved = (pipeline.openalex_work, pipeline.openalex_records,
             pipeline.openalex_citing_works, pipeline.curate_sources)
    try:
        def fake_openalex_work(paper):
            work_calls.append(paper)
            return (f"W-{paper.doi}", [f"ref-{paper.doi}-{i}" for i in range(3)])

        def fake_openalex_records(ids):
            return list(new_papers_offered)

        def fake_openalex_citing_works(work_id, limit=sources.REFERENCED_CITED_LIMIT):
            citing_calls.append(work_id)
            return []

        def fake_curate_sources(topic, new_papers, **kw):
            curate_calls.append(list(new_papers))
            return {"relevance": {i + 1: "direct" for i in range(len(new_papers))},
                    "counts": {"direct": len(new_papers)}}

        pipeline.openalex_work = fake_openalex_work
        pipeline.openalex_records = fake_openalex_records
        pipeline.openalex_citing_works = fake_openalex_citing_works
        pipeline.curate_sources = fake_curate_sources

        exhausted: set[str] = set()
        result = pipeline._referenced_source_pass(
            "seclusion", papers, curation, exhausted, log=lambda m: None,
        )

        check("existing pool entries keep their positions and titles",
              papers[0] is p1 and papers[1] is p2 and papers[2] is p3
              and papers[3] is p4 and papers[4] is p5)
        check("existing relevance labels unchanged",
              all(curation["relevance"][i] == {1: "direct", 2: "direct", 3: "related",
                                                4: "direct", 5: "tangential"}[i]
                  for i in range(1, 6)))
        check("a known referenced record is present in the pool after the call",
              any(p.doi == "10.1000/known-referenced" for p in papers))
        check("twelve offered, eight kept",
              len(papers) == 5 + sources.NAMED_SOURCE_LIMIT)
        check("curate_sources called exactly once with only the new records",
              len(curate_calls) == 1 and len(curate_calls[0]) == sources.NAMED_SOURCE_LIMIT
              and all(p not in (p1, p2, p3, p4, p5) for p in curate_calls[0]))
        check("new relevance keys start at 6",
              min(k for k in curation["relevance"] if k > 5) == 6)
        merged_rel = curation["relevance"]
        expected_counts = {
            level: sum(1 for v in merged_rel.values() if v == level)
            for level in ("direct", "related", "tangential")
        }
        check("curation counts match the merged relevance map",
              curation["counts"] == expected_counts)
        check("only direct syntheses become seeds (tangential paper never a seed)",
              p5 not in work_calls)
        check("only direct syntheses become seeds (related paper never a seed)",
              p3 not in work_calls)
        check("at most REFERENCED_SEEDS syntheses are seeds",
              sum(1 for p in work_calls if p is p1) <= sources.REFERENCED_SEEDS)
        check("the direct trial is followed forward via citing works", citing_calls == [f"W-{p2.doi}"])
        check("result reports added count", result["added"] == sources.NAMED_SOURCE_LIMIT)

        # Shared exhausted set: already-exhausted skips every call.
        work_calls.clear()
        citing_calls.clear()
        curate_calls.clear()
        papers2 = [Paper(title=p.title, abstract=p.abstract, year=p.year, doi=p.doi,
                          publication_types=list(p.publication_types)) for p in papers[:5]]
        curation2 = {
            "relevance": {1: "direct", 2: "direct", 3: "related", 4: "direct", 5: "tangential"},
            "most_relevant_index": 1,
            "counts": {"direct": 3, "related": 1, "tangential": 1},
        }
        pre_exhausted = {"openalex"}
        result2 = pipeline._referenced_source_pass(
            "seclusion", papers2, curation2, pre_exhausted, log=lambda m: None,
        )
        check("already-exhausted openalex makes no calls",
              work_calls == [] and citing_calls == [] and curate_calls == [])
        check("already-exhausted openalex returns added == 0", result2["added"] == 0)

        # A SearchFailure from openalex_work is swallowed and marks exhausted.
        papers3 = [Paper(title=p.title, abstract=p.abstract, year=p.year, doi=p.doi,
                          publication_types=list(p.publication_types)) for p in papers[:5]]
        curation3 = {
            "relevance": {1: "direct", 2: "direct", 3: "related", 4: "direct", 5: "tangential"},
            "most_relevant_index": 1,
            "counts": {"direct": 3, "related": 1, "tangential": 1},
        }
        exhausted3: set[str] = set()

        def raising_openalex_work(paper):
            raise sources.SearchFailure("OpenAlex refused")

        pipeline.openalex_work = raising_openalex_work
        result3 = pipeline._referenced_source_pass(
            "seclusion", papers3, curation3, exhausted3, log=lambda m: None,
        )
        check("a SearchFailure does not raise (call returned)", result3 is not None)
        check("a SearchFailure marks openalex exhausted", "openalex" in exhausted3)
    finally:
        (pipeline.openalex_work, pipeline.openalex_records,
         pipeline.openalex_citing_works, pipeline.curate_sources) = saved


def test_methods_names_the_named_source_pass() -> None:
    """Methods describes the targeted named-source pass when present, and omits it otherwise."""
    from articlegen import render

    prov_with_named = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
        "model": "claude-opus-5",
        "named_sources": {
            "queries": ["10.1001/jama.2015.1234", "Safewards <trial>"],
            "added": 1,
        },
    }

    html_out = render._methods_html(prov_with_named, screened=10, n_cited=3, topic="seclusion")
    md_out = "\n".join(render._methods_markdown(prov_with_named, screened=10, n_cited=3, topic="seclusion"))

    check("html contains named sources sentence",
          "A second, targeted search then looked up 2 works named in the most relevant abstracts" in html_out)
    check("html contains added count sentence",
          "which added 1 further record to the pool." in html_out)
    check("html escapes special characters in queries",
          "Safewards &lt;trial&gt;" in html_out)

    check("markdown contains named sources sentence",
          "A second, targeted search then looked up 2 works named in the most relevant abstracts" in md_out)
    check("markdown contains added count sentence",
          "which added 1 further record to the pool." in md_out)

    prov_zero_added = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
        "named_sources": {
            "queries": ["Safewards trial"],
            "added": 0,
        },
    }
    html_zero = render._methods_html(prov_zero_added, screened=10, n_cited=3, topic="seclusion")
    check("zero added reports 'no further records to the pool.'",
          "which added no further records to the pool." in html_zero)

    prov_none = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
    }
    html_none = render._methods_html(prov_none, screened=10, n_cited=3, topic="seclusion")
    md_none = "\n".join(render._methods_markdown(prov_none, screened=10, n_cited=3, topic="seclusion"))

    check("absent named_sources leaves HTML methods unchanged",
          "second, targeted search" not in html_none)
    check("absent named_sources leaves Markdown methods unchanged",
          "second, targeted search" not in md_none)


def test_methods_names_the_referenced_works_pass() -> None:
    """Methods describes the one-hop citation follow-up pass (issue #243) when present."""
    from articlegen import render

    prov_with_referenced = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
        "model": "claude-opus-5",
        "referenced_sources": {
            "seeds": ["A systematic review of seclusion <trial>"],
            "cited_seed": "The landmark trial",
            "added": 2,
        },
    }

    html_out = render._methods_html(prov_with_referenced, screened=10, n_cited=3, topic="seclusion")
    md_out = "\n".join(render._methods_markdown(prov_with_referenced, screened=10, n_cited=3, topic="seclusion"))

    check("html contains the citation follow-up sentence",
          "Citations were then followed one step through OpenAlex" in html_out)
    check("html contains the added count",
          "which added 2 further records to the pool." in html_out)
    check("html escapes special characters in seed titles",
          "seclusion &lt;trial&gt;" in html_out)

    check("markdown contains the citation follow-up sentence",
          "Citations were then followed one step through OpenAlex" in md_out)
    check("markdown contains the added count",
          "which added 2 further records to the pool." in md_out)

    prov_zero_added = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
        "referenced_sources": {
            "seeds": [],
            "cited_seed": "The landmark trial",
            "added": 0,
        },
    }
    html_zero = render._methods_html(prov_zero_added, screened=10, n_cited=3, topic="seclusion")
    check("zero added reports 'no further records to the pool.'",
          "which added no further records to the pool." in html_zero)

    prov_none = {
        "queries": ["main query"],
        "databases": ["Europe PMC"],
        "date": "21 August 2026",
    }
    html_none = render._methods_html(prov_none, screened=10, n_cited=3, topic="seclusion")
    md_none = "\n".join(render._methods_markdown(prov_none, screened=10, n_cited=3, topic="seclusion"))

    check("absent referenced_sources leaves HTML methods unchanged",
          "Citations were then followed" not in html_none)
    check("absent referenced_sources leaves Markdown methods unchanged",
          "Citations were then followed" not in md_none)


def _validate(instance, schema: dict, path: str = "") -> list[str]:
    """The slice of JSON Schema `_ARTICLE_SCHEMA` actually uses.

    A dependency for this would be a third one in a project with two, and the
    schema is deliberately plain: object, array, string, integer, required.
    """
    errors: list[str] = []
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str,
             "integer": int, "number": (int, float), "boolean": bool}
    if kind in types and not isinstance(instance, types[kind]):
        return [f"{path or '<root>'}: expected {kind}, got {type(instance).__name__}"]
    if kind == "object":
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required, missing")
        for key, sub in (schema.get("properties") or {}).items():
            if key in instance:
                errors += _validate(instance[key], sub, f"{path}.{key}")
    elif kind == "array" and schema.get("items"):
        for i, item in enumerate(instance):
            errors += _validate(item, schema["items"], f"{path}[{i}]")
    return errors


def test_model_comparison_harness() -> None:
    """The default costs 2.5x the alternative, and that trade was taken on purpose.

    `anthropic/claude-opus-5` is $5/$25 per Mtok; `anthropic/claude-sonnet-5` is
    $2/$10. #85 asked whether the cheaper model was good enough and was settled
    by decision, not by measurement: Opus stays, because prose quality is the
    product. The harness survives as a general cost/quality probe for a future
    pair, so the arithmetic it reports still has to be right.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import compare_models

    # The prices are the reason the issue exists, so the arithmetic is worth
    # pinning: same tokens, 2.5x the cost.
    opus = compare_models.cost("anthropic/claude-opus-5", 100_000, 20_000)
    sonnet = compare_models.cost("anthropic/claude-sonnet-5", 100_000, 20_000)
    check("the estimate matches the published prices",
          opus == "$1.000" and sonnet == "$0.400")
    check("and reproduces the 2.5x gap the issue is about",
          float(opus[1:]) / float(sonnet[1:]) == 2.5)
    # A wrong dollar figure is worse than none: list prices go stale, and an
    # unknown model must not be silently priced as a known one.
    check("an unknown model is not priced", compare_models.cost("x/y", 1, 1) == "n/a")
    check("and neither is a run whose tokens were not captured",
          compare_models.cost("anthropic/claude-opus-5", 0, 0) == "n/a")

    # It must read the provider's own accounting, not guess.
    line = "[articlegen] openrouter anthropic/claude-opus-5 in=1234 cached=99 out=567"
    m = compare_models._USAGE.search(line)
    check("token counts are read from the provider log",
          m is not None and m.group(1) == "1234" and m.group(3) == "567")

    # The report has to say what it cannot measure. The whole value of the
    # pipeline is adjudication, and a contrast-marker count is not that.
    rendered = []
    fake = [{"model": f"m{i}", "cited": 10, "papers": 20, "counts": {},
             "sections": 6, "words": 1400, "figures": 12, "unverified": 1,
             "misattributed": 0, "style_errors": [], "contrast": 9,
             "contrast_per_sentence": 0.1, "tokens_in": 0, "tokens_out": 0}
            for i in (1, 2)]
    compare_models.print = lambda *a, **k: rendered.append(" ".join(map(str, a)))
    try:
        compare_models.report(fake)
    finally:
        del compare_models.print
    out = "\n".join(rendered)
    check("the report names what it cannot measure", "ADJUDICATE" in out)
    check("and labels the contrast count as a hint, not a score",
          "not a score" in out and "hint only" in out)
    check("and says the activity log is the authority on cost",
          "activity log is the authority" in out)
    check("it reports the checks the pipeline already runs",
          "style errors" in out and "unverified figures" in out
          and "misattributed figures" in out)


def test_full_text_run_says_why_it_stopped() -> None:
    """"4 of 19" is not an answer until you know which limit bound.

    On the best article so far, 4 of 19 cited sources had retrievable
    open-access full text. Whether the pipeline stopped early on its own
    request cap or whether only 4 sources were actually available needs
    completely different fixes — and the old log reported the count and then
    *asserted* availability ("the remaining cited sources have no open-access
    copy"), so the two were indistinguishable (issue #84).
    """
    from articlegen import pipeline
    from articlegen.sources import Paper

    def run(n_papers, open_access_upto, target=None, cap=None):
        papers = [Paper(title=f"p{i}", abstract="a", year=2000 + i,
                        citation_count=i * 10,
                        pmcid=f"PMC{i}" if i <= open_access_upto else "",
                        is_open_access=i <= open_access_upto)
                  for i in range(1, n_papers + 1)]
        curation = {"relevance": {i: "direct" for i in range(1, n_papers + 1)},
                    "most_relevant_index": 1, "counts": {"direct": n_papers}}
        lines = []
        saved = (pipeline.plan_queries, pipeline.gather_evidence,
                 pipeline.curate_sources, pipeline.write_article,
                 pipeline.write_briefing, pipeline.fetch_full_text,
                 pipeline.enforce_style,
                 pipeline.FULLTEXT_TARGET, pipeline.MAX_FULLTEXT_REQUESTS)
        try:
            if target is not None:
                pipeline.FULLTEXT_TARGET = target
            if cap is not None:
                pipeline.MAX_FULLTEXT_REQUESTS = cap
            pipeline.plan_queries = lambda topic, **kw: (["q"], "c")

            def gather(queries, **kw):
                kw.get("outcomes", []).append({"source": "europe_pmc", "query": "q",
                                               "count": n_papers, "error": "",
                                               "cached": False})
                return papers

            pipeline.gather_evidence = gather
            pipeline.curate_sources = lambda t, p, **kw: curation
            pipeline.write_article = lambda t, p, **kw: {
                "title": "t", "abstract": "x", "keywords": [], "sections": [],
                "key_points": [], "glossary": [],
                "references": list(range(1, n_papers + 1))}
            pipeline.write_briefing = pipeline.write_article
            pipeline.fetch_full_text = lambda p, use_cache=True: "body"
            pipeline.enforce_style = lambda a, **kw: (a, {"issues": [], "stats": {}})
            pipeline.generate_draft("topic", log=lines.append)
        finally:
            (pipeline.plan_queries, pipeline.gather_evidence,
             pipeline.curate_sources, pipeline.write_article,
             pipeline.write_briefing, pipeline.fetch_full_text,
             pipeline.enforce_style,
             pipeline.FULLTEXT_TARGET, pipeline.MAX_FULLTEXT_REQUESTS) = saved
        return "\n".join(lines)

    # Availability bound it: plenty of budget, only two open-access sources.
    scarce = run(8, open_access_upto=2)
    check("a run short of the target says why", "stopped because:" in scarce)
    check("and names availability when that is the reason",
          "ran out of eligible sources" in scarce
          and "had no open-access copy" in scarce)
    check("it no longer asserts availability it did not check",
          "the remaining cited sources have no open-access copy" not in scarce)

    # The request cap bound it: everything is open access, cap set below need.
    capped = run(8, open_access_upto=8, target=8, cap=3)
    check("a capped run says the cap bound", "request cap of 3 reached" in capped)
    check("and flags that the code, not the literature, is the constraint",
          "MAX_FULLTEXT_REQUESTS" in capped and "NOTE:" in capped)

    # The target bound it: the ordinary healthy case.
    satisfied = run(8, open_access_upto=8, target=2)
    check("a satisfied run says the target was reached",
          "target of 2 reached" in satisfied)
    check("and does not warn about a cap that never bit", "NOTE:" not in satisfied)

    # The skew, measured rather than asserted. Limitations already tells the
    # reader the deeply-read subset skews open access; nobody had checked how.
    check("every run reports the read-subset skew", "read-subset skew:" in scarce)
    check("with both axes the concern is about",
          "median year" in scarce and "median citations" in scarce)
    check("and says so plainly when there is nothing to compare",
          "not comparable" in pipeline._read_subset_skew(
              [Paper(title="p", abstract="a", year=2020)], []))


def test_curation_truncation_is_off_until_measured() -> None:
    """The token saving is real; the risk to the relevance gate is untested.

    Truncating the abstracts sent to curation would cut ~39,000 input tokens to
    roughly 15,000 for 20 papers (the default pool then; it is now
    `DEFAULT_MAX_PAPERS`). But curation *is* the relevance gate, and two
    things downstream read its labels: `style._required_sections` scales the
    section floor with the `direct` count, and `write_article` omits
    `tangential` sources from the prompt entirely (issue #117).

    The failure mode is not theoretical — curation at a cheaper reasoning tier
    agreed with the full tier on only 14 of 20 labels, and every disagreement
    collapsed toward "related".
    """
    import inspect

    from articlegen import pipeline, writer
    from articlegen.sources import Paper

    check("the pipeline still sends full abstracts to curation",
          writer.CURATION_ABSTRACT_CHARS is None)

    long_abstract = ("Bright light therapy was tested in schizophrenia. " * 40).strip()
    papers = [Paper(title="A study", abstract=long_abstract, year=2020)]

    full = writer._format_sources(papers)
    check("no limit means the abstract is untouched", long_abstract in full)

    short = writer._format_sources(papers, abstract_chars=400)
    check("a limit shortens the prompt", len(short) < len(full))
    check("the title survives truncation", "Title: A study" in short)
    check("and the cut is marked, so the model is not misled about completeness",
          "[…]" in short)
    kept = short.split("Abstract: ")[1].replace(" […]", "").strip()
    check("the kept text is a prefix of the original", long_abstract.startswith(kept))

    # An abstract already under the limit must come through whole — marking it
    # would say text was cut when none was.
    brief = [Paper(title="B", abstract="Short abstract.", year=2021)]
    check("an abstract under the limit is left alone",
          "[…]" not in writer._format_sources(brief, abstract_chars=400))

    # `curate_sources` must actually honour the parameter, or the harness
    # compares two identical runs and reports a confident false pass.
    sent = []
    real = writer.generate_json
    try:
        writer.generate_json = lambda prompt, schema, **kw: (
            sent.append(prompt), {"assessments": []})[1]
        writer.curate_sources("topic", papers)
        writer.curate_sources("topic", papers, abstract_chars=400)
    finally:
        writer.generate_json = real
    check("curate_sources honours abstract_chars", len(sent[1]) < len(sent[0]))
    check("and defaults to the module constant, which is off",
          "[…]" not in sent[0] and "[…]" in sent[1])
    check("the pipeline passes no limit of its own",
          "abstract_chars" not in inspect.getsource(pipeline.generate_draft))

    # The harness that would settle it, and its acceptance rule. The rule is
    # the part worth pinning: overall agreement is the wrong measure, because a
    # run that collapsed every label to "related" would score well on it.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harness = open(os.path.join(root, "tools", "compare_curation.py"),
                   encoding="utf-8").read()
    check("the comparison harness exists", "def compare(" in harness)
    check("it runs both curations over the same fetched papers",
          harness.count("curate_sources(topic, papers") == 2)
    check("and only direct and tangential are treated as gating",
          'CRITICAL = ("direct", "tangential")' in harness)
    check("it says outright that agreement on 'related' is not enough",
          "'related' alone is not enough" in harness)


def test_the_candidate_pool_is_big_enough_to_curate() -> None:
    """The pool default is one constant, and 20 was too small to be curated.

    Three runs on 2026-08-15 collected exactly 20 candidates and cited 16-19 of
    them — the relevance gate discarding almost nothing — and a landmark cluster
    RCT named by a run's own planned query never made the pool (issue #141).

    The number lived in four places, and one of them was a hardcoded argument in
    the web handler rather than a default. Raising the pipeline default alone
    would have left the deployed web app on 20 with nothing to show for it.
    """
    import inspect

    from articlegen import cli, pipeline, sources, web, writer

    check("the pool default is 40", sources.DEFAULT_MAX_PAPERS == 40)

    for mod, fn in ((sources, sources.gather_evidence),
                    (pipeline, pipeline.generate_draft)):
        default = inspect.signature(fn).parameters["max_papers"].default
        check(f"{fn.__name__} defaults to the constant",
              default == sources.DEFAULT_MAX_PAPERS)

    parser = cli.build_parser()
    args = parser.parse_args(["draft", "a topic"])
    check("the CLI flag defaults to the constant",
          args.max_papers == sources.DEFAULT_MAX_PAPERS)

    handler = inspect.getsource(web.ArticleGenHandler._handle_draft)
    check("the web handler reads the constant rather than its own number",
          "max_papers=DEFAULT_MAX_PAPERS" in handler)

    # The whole point of one constant: no caller may carry a second copy of the
    # number. A stale literal here is how the web path would sit on the old pool
    # while everything else moved.
    for mod in (cli, pipeline, sources, web):
        check(f"{mod.__name__} hardcodes no candidate cap of its own",
              "max_papers=20" not in inspect.getsource(mod)
              and "max_papers: int = 20" not in inspect.getsource(mod))

    # The bigger pool is paid for in curation tokens, never in truncation.
    # Truncating at 400 chars was measured in #117 and destabilised the gate.
    check("a bigger pool did not buy itself truncated abstracts",
          writer.CURATION_ABSTRACT_CHARS is None)


def test_claude_md_still_describes_this_code() -> None:
    """Every file, test and constant CLAUDE.md names must still exist.

    The doc says "Read this first" and is loaded into every session, so it is
    trusted on sight — which makes a wrong line worse than a missing one. By
    the audit on 12 August it named `.github/workflows/pages.yml` (the file is
    `deploy-pages.yml`), documented three FULLTEXT constants that no longer
    matched the code, listed `SUBSTANCE_RULES` without `under-length`, and
    described Groq as the default provider after Groq had been removed in the
    same session (issue #114).

    The CI gate makes someone *think* about the doc on every code PR. This
    catches the drift that survives thinking about it: a rename lands, the doc
    still names the old thing, and nobody re-reads the paragraph around it.
    """
    import re as _re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Both halves of the split (#99). CLAUDE.md holds the invariants,
    # docs/decisions.md the settled history — and drift is as bad in the
    # second: a post-mortem naming a function that no longer exists sends the
    # next reader looking for it.
    # Those files left the public tree; skip when they are not at repo root
    # (they live in archive/ on this machine, gitignored).
    names = ("CLAUDE.md", "docs/decisions.md")
    if not all(os.path.isfile(os.path.join(root, name)) for name in names):
        print("SKIP test_claude_md_still_describes_this_code (docs not in this checkout)")
        return
    doc = "\n".join(
        open(os.path.join(root, name), encoding="utf-8").read()
        for name in names
    )

    # 1. Paths. Anything backticked that looks like a repo file must be one.
    #    This is the check that would have caught `pages.yml`.
    #
    #    ...except where the docs are *quoting* the wrong name, which is what
    #    the #114 post-mortem does: "it named `pages.yml` (the file is
    #    `deploy-pages.yml`)". Same shape as the Groq rule below — a
    #    retrospective marker on the line is what separates a record of the
    #    error from a claim about the code.
    searched = ("", "articlegen", "tests", "docs", ".github/workflows", "deploy")
    retrospective = ("the file is", "no longer", "used to", "was renamed",
                     "renamed", "it named", "removed")
    lines_for = {}
    for line in doc.splitlines():
        for path in _re.findall(r"`([\w./-]+\.(?:py|md|yml|yaml|html|json|txt|sh))`", line):
            lines_for.setdefault(path, []).append(line.lower())

    for path, lines in sorted(lines_for.items()):
        if path.startswith(("http", "~")):
            continue
        if all(any(m in ln for m in retrospective) for ln in lines):
            continue     # only ever mentioned as a thing that changed
        found = any(os.path.exists(os.path.join(root, d, path)) for d in searched)
        check(f"the docs name a file that exists: {path}", found)

    # 2. Test names. The doc cites guard tests by name; a renamed test leaves
    #    the sentence pointing at nothing, and prose that points at nothing
    #    stops being a pointer and becomes folklore.
    suites = ""
    for name in ("test_offline.py", "test_journal_conformance.py"):
        suites += open(os.path.join(root, "tests", name), encoding="utf-8").read()
    for test_name in sorted(set(_re.findall(r"`(test_\w+)`", doc))):
        check(f"CLAUDE.md cites a test that exists: {test_name}",
              f"def {test_name}(" in suites)

    # 3. Named constants. Same argument, and the FULLTEXT trio is the case that
    #    actually drifted while the paragraph around it read as authoritative.
    import inspect

    # Every module, not a subset — a constant named in the docs is stale
    # wherever it lives, and a partial sweep fails on correct text.
    from articlegen import (gallery, llm, paperfetch, pipeline, render, sources, style,
                            verify, web, writer)

    modules = (gallery, llm, paperfetch, pipeline, render, sources, style, verify, web, writer)
    # Env vars and front-end constants are not module attributes, so a plain
    # `hasattr` sweep would fail on a dozen names that are perfectly current.
    # Falling back to "appears anywhere in the code" still catches the case
    # that matters: a constant renamed or deleted appears nowhere at all.
    haystack = "".join(inspect.getsource(m) for m in modules)
    haystack += open(os.path.join(root, "index.html"), encoding="utf-8").read()
    for const in sorted(set(_re.findall(r"`([A-Z][A-Z0-9_]{4,})`", doc))):
        if const in ("CLAUDE", "README", "SOURCE", "HTTP", "TODO", "JATS"):
            continue
        check(f"CLAUDE.md names a constant that exists: {const}",
              any(hasattr(m, const) for m in modules) or const in haystack)

    # 4. A removed provider must not read as *current*. Groq was described as
    #    the default provider throughout the doc for a whole session after it
    #    was deleted from the code. Explaining why it went is exactly what the
    #    doc is for, so only present-tense currency is banned.
    from articlegen.llm import _PROVIDER_DEFAULT_MODELS

    check("groq really is gone from the code", "groq" not in _PROVIDER_DEFAULT_MODELS)
    currency = ("is the default", "default provider", "currently the default",
                "we use groq", "runs on groq")
    # Recounting the error is legitimate — "Groq described as the default after
    # it was deleted" is a sentence this file should contain. A retrospective
    # marker anywhere on the line is what separates the history from a claim.
    retrospective = ("was", "were", "had", "after", "until", "used to", "removed",
                     "gone", "no longer", "era", "reinstate", "old")
    for line in doc.splitlines():
        lowered = line.lower()
        if "groq" in lowered and any(m in lowered for m in currency):
            check(f"CLAUDE.md does not call groq current: {line.strip()[:56]!r}",
                  any(m in lowered for m in retrospective))
    check("and the doc says outright that it was removed",
          "groq was removed" in doc.lower())


def test_briefing_is_the_default_artefact() -> None:
    """The sendable page is a briefing. The Review is `--long`, not the default.

    The site already framed the job as a sourced evidence briefing. The artefact
    was still a ~3,000-word journal Review. This pins the cut: default render
    is the briefing, ideas are questions not magazine pitches, and the parked
    Review writer still exists.
    """
    from articlegen import demo, ideas
    from articlegen.cli import build_parser
    from articlegen.style import check_style, errors as style_errors
    from articlegen.writer import _BRIEFING_SYSTEM, write_article, write_briefing

    check("ideas are not an editor for a popular-science publication",
          "ideas editor for a popular-science publication" not in ideas._IDEAS_SYSTEM)
    check("ideas prompt asks for briefing questions",
          "evidence-briefing" in ideas._IDEAS_SYSTEM)
    check("demo briefing passes the style check",
          not style_errors(check_style(demo.SAMPLE_BRIEFING)))
    check("demo Review still passes the style check",
          not style_errors(check_style(demo.SAMPLE_ARTICLE)))
    check("the briefing prompt is not a magazine feature",
          "popular-science" in _BRIEFING_SYSTEM
          and "NOT writing a magazine feature" in _BRIEFING_SYSTEM)

    parser = build_parser()
    check("draft defaults to a briefing",
          not parser.parse_args(["draft", "a topic"]).long)
    check("draft --long is the Review path",
          parser.parse_args(["draft", "a topic", "--long"]).long)
    check("write_briefing and write_article are both still callable",
          callable(write_briefing) and callable(write_article))


def test_titles_describe_the_question() -> None:
    """A `--long` title names the question. It does not answer it.

    The Review path asked for "the subject and the finding" and got
    "Brief hospital admission by self-referral reduces involuntary care and
    self-harm ... in borderline personality disorder" — a causal claim in the
    one field nothing downstream checks. `verify.check_statistics` never reads
    titles and `style.py` has no title rule, so the prompt is the only control
    there is (issue #170).

    The briefing schema already had the rule. This pins that both schemas now
    read it from one string, that the old wording is gone, and that both system
    prompts carry the prohibition. Deliberately no regex title-ban in style.py:
    "reduces" is a legitimate word in a descriptive title and a crude ban would
    fail good titles. Measure first, per the issue.
    """
    from articlegen.writer import (_ARTICLE_SCHEMA, _BRIEFING_SCHEMA, _TITLE_RULE,
                                   _BRIEFING_SYSTEM, _REVISE_PATCH_SYSTEM,
                                   _REVISE_SYSTEM, _WRITER_SYSTEM,
                                   _WRITER_SYSTEM_FULLTEXT)

    article_title = _ARTICLE_SCHEMA["properties"]["title"]["description"]
    briefing_title = _BRIEFING_SCHEMA["properties"]["title"]["description"]

    check("the Review and the briefing share one title rule",
          article_title == briefing_title == _TITLE_RULE)
    check("the rule forbids claiming the result", "no result claimed" in _TITLE_RULE)
    check("and shows what that means",
          "reduces" in _TITLE_RULE and "Right:" in _TITLE_RULE)
    check("the rule asks for population, intervention/exposure and outcome",
          all(word in _TITLE_RULE
              for word in ("population", "intervention or exposure", "outcome")))
    check("the old 'subject and the finding' wording is gone",
          "the subject and the finding" not in article_title)

    line = "TITLE: descriptive. Names the question. Does not claim the result."
    check("the Review prompt carries the title rule", line in _WRITER_SYSTEM)
    check("the briefing prompt still does", line in _BRIEFING_SYSTEM)
    for name, prompt in (("revise", _REVISE_SYSTEM),
                         ("revise-patch", _REVISE_PATCH_SYSTEM),
                         ("full-text", _WRITER_SYSTEM_FULLTEXT)):
        check(f"the {name} prompt inherits it", line in prompt)


def test_real_articles_still_match_the_schema() -> None:
    """Every article the pipeline has to render must satisfy the writer's schema.

    Fixtures encode yesterday's payloads: each replays one observed shape, so a
    model that stops honouring `_ARTICLE_SCHEMA` is structurally undetectable
    (issue #98). Nothing validated any real article output against the schema
    it was generated from — so schema and renderer could drift apart silently,
    which is how a legacy field ends up load-bearing without anyone noticing.
    """
    from articlegen import demo
    from articlegen.writer import _ARTICLE_SCHEMA, _BRIEFING_SCHEMA

    articles = [("demo.SAMPLE_ARTICLE", demo.SAMPLE_ARTICLE)]
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import test_journal_conformance as conformance

        articles += [(name, article) for name, article, *_ in conformance.fixtures()]
    except Exception as exc:      # the conformance suite is a sibling, not a dep
        check(f"conformance fixtures are readable ({exc})", False)

    for name, article in articles:
        errors = _validate(article, _ARTICLE_SCHEMA)
        check(f"{name} matches the article schema", not errors)
        if errors:
            for e in errors[:5]:
                print("     " + e)

    # The legacy fields render but are gone from the schema. That is deliberate
    # and documented; this pins the direction so a future schema change does
    # not quietly resurrect one as required.
    required = set(_ARTICLE_SCHEMA.get("required", []))
    for legacy in ("standfirst", "key_takeaways", "pull_quote"):
        check(f"{legacy} is not required by the schema", legacy not in required)

    briefing = {k: v for k, v in demo.SAMPLE_BRIEFING.items() if k != "form"}
    briefing_errors = _validate(briefing, _BRIEFING_SCHEMA)
    check("demo.SAMPLE_BRIEFING matches the briefing schema", not briefing_errors)
    if briefing_errors:
        for e in briefing_errors[:5]:
            print("     " + e)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for fn in (
        test_provider_resolution, test_per_request_api_key,
        test_anthropic_generate_behaviour,
        test_openrouter_request_is_asserted_on_the_payload,
        test_output_ceilings_follow_the_default_model,
        test_every_provider_reports_what_it_sent,
        test_model_comparison_harness,
        test_full_text_run_says_why_it_stopped,
        test_curation_truncation_is_off_until_measured,
        test_the_candidate_pool_is_big_enough_to_curate,
        test_claude_md_still_describes_this_code,
        test_real_articles_still_match_the_schema,
        test_openrouter_routing,
        test_openrouter_refusal_falls_back,
        test_refusal_fallbacks,
        test_pipeline_is_shared,
        test_idea_search_terms_reach_the_draft,
        test_idea_pico_reaches_the_curator,
        test_paraphrase_terms_buy_a_distinct_query,
        test_dead_sources_fail_before_the_caller_is_billed,
        test_draft_summary, test_run_manifest_round_trips, test_rate_limit,
        test_keepalive_connection_reuse, test_substance_checks,
        test_source_failures_are_distinguishable,
        test_first_semantic_scholar_refusal_buys_one_patient_round,
        test_read_timeout_does_not_retry,
        test_preflight_stops_when_a_source_answers,
        test_search_cache, test_search_cache_survives_a_new_process,
        test_front_end_models_match_the_allowlist,
        test_public_generation_forces_luna_on_the_hosted_key,
        test_polite_pool_identification, test_europe_pmc_parsing,
        test_table_one_reports_sample_size_registration_and_funding,
        test_sample_size_never_reads_a_year_or_a_page_number,
        test_arxiv_parsing, test_titles_arrive_without_markup,
        test_candidate_papers_dedupe_by_doi,
        test_preprints_are_marked_as_preprints,
        test_retracted_records_never_reach_the_writer,
        test_arxiv_rate_limit_is_honoured,
        test_ungrounded_citations_leave_no_trace,
        test_second_hand_figures_are_a_last_resort,
        test_full_text_grounding, test_section_aware_excerpts,
        test_pipeline_fetches_full_text,
        test_unlabelled_sources_stop_the_run,
        test_full_text_comes_from_the_papers_cli_when_it_is_there,
        test_queued_ckn_counts_as_no_open_access,
        test_ckn_loop,
        test_refresh_reports_new_direct,
        test_one_papers_process_per_run,
        test_hosted_fulltext_prefetch_is_the_target,
        test_papers_batch_keeps_partial_results_on_timeout,
        test_full_text_order_favours_reviews_and_trials,
        test_pmcid_is_resolved_by_doi,
        test_unpaywall_fallback_in_resolve_pmcid,
        test_named_papers_in_abstracts_are_looked_up,
        test_named_references_reads_names_not_noise,
        test_generic_named_lookups_are_skipped,
        test_named_sources_merge_without_renumbering,
        test_referenced_works_fetch_asks_openalex_correctly,
        test_referenced_works_merge_without_renumbering,
        test_methods_names_the_named_source_pass,
        test_methods_names_the_referenced_works_pass,
        test_methods_names_only_sources_that_answered,
        test_citation_renumbering, test_journal_citation_style, test_reference_formatting,
        test_prose_style_check, test_rules_do_not_reject_real_journal_prose,
        test_the_front_end_has_one_article_list,
        test_visitor_gallery,
        test_article_in_the_web_app_cannot_run_scripts,
        test_desktop_article_uses_the_full_window,
        test_browser_back_from_an_article_returns_home,
        test_health_reports_which_build_is_running,
        test_openalex_reaches_for_recent_work_as_well,
        test_claude_cli_provider,
        test_gemini_cli_provider,
        test_tangential_sources_stay_out_of_the_writer_prompt,
        test_the_writer_cites_a_working_set,
        test_the_cite_ceiling_scales_with_the_direct_count,
        test_revision_replaces_blocks_rather_than_the_article,
        test_revision_carries_sources_only_when_they_can_be_used,
        test_warnings_ride_along_on_a_revision,
        test_a_second_style_pass_runs_only_after_progress,
        test_disclosure_is_above_the_fold_and_derived,
        test_display_items_are_selected_once_for_both_formats,
        test_figure_one_counts_study_designs,
        test_failed_style_gate_is_visible_in_the_article,
        test_only_sendable_defects_brand_the_page,
        test_evidence_assessment_is_wholly_deterministic,
        test_unverified_figures_are_marked_inline,
        test_clinical_directives_are_an_error,
        test_full_text_dependencies_fail_loudly_enough_to_diagnose,
        test_first_visit_does_not_dead_end,
        test_the_landing_page_leads_with_the_input,
        test_the_web_app_does_not_take_a_visitor_key,
        test_house_style_is_fixed_not_a_preference,
        test_briefing_is_the_default_artefact,
        test_titles_describe_the_question,
        test_register_rules_are_scoped_to_the_synthesis_voice,
        test_hedging_floor_is_calibrated_against_body_prose,
        test_density_thresholds_are_documented_against_the_corpus,
        test_statistic_verification, test_flagged_figures_buy_one_revision,
        test_ranking, test_recency_actually_counts, test_render_blocks,
        test_display_item_placement, test_legacy_draft_fields,
        test_demo_and_index, test_web_server,
        test_run_log_never_carries_the_topic,
        test_every_endpoint_logs_one_run_line,
        test_run_log_gist_is_optional_and_gist_scoped,
    ):
        print(f"\n# {fn.__name__}")
        # One crash used to abort the whole run. Several tests mutate shared
        # state — env vars, `sources.search_*`, `requests.post` — and not all
        # restore it, so an early crash left the rest running against a dirty
        # environment or not running at all (issue #98). The conformance suite
        # already wraps each predicate; this matches it.
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {fn.__name__} raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}")

    if "--live" in argv:
        print("\n# live_smoke")
        live_smoke()

    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL PASS'}")
    return 1 if FAILURES else 0


def live_smoke() -> None:
    """One real call to each seam. Opt-in: `python tests/test_offline.py --live`.

    Everything above fakes the transport, which is the right default — but the
    faking is exactly what makes provider drift undetectable, and every
    production failure on record happened on this seam (issue #98). This is the
    smallest thing that would have caught them: one cheap `generate_json` and
    one `gather_evidence` query.

    Costs a few cents of real credit and one request against the shared
    scholarly quota, so it never runs in the default suite.
    """
    from articlegen import llm
    from articlegen.sources import gather_evidence

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        check("live: a key is available", False)
        return
    schema = {"type": "object",
              "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    try:
        out = llm.generate_json(
            "Reply with {\"answer\": \"ok\"} and nothing else.", schema,
            system="You return JSON matching the schema.", deep=False)
        check("live: the provider honours the schema",
              isinstance(out, dict) and isinstance(out.get("answer"), str))
    except Exception as exc:
        check(f"live: generate_json succeeded ({type(exc).__name__}: {exc})", False)

    outcomes: list[dict] = []
    try:
        papers = gather_evidence(["shift work sleep"], max_papers=3, per_query=3,
                                 topic="shift work sleep", outcomes=outcomes,
                                 use_cache=False)
        check(f"live: at least one scholarly API answered ({len(papers)} papers)",
              any(not o["error"] for o in outcomes))
        for outcome in outcomes:
            if outcome["error"]:
                print(f"     {outcome['source']}: {outcome['error']}")
    except Exception as exc:
        check(f"live: gather_evidence succeeded ({type(exc).__name__}: {exc})", False)


if __name__ == "__main__":
    raise SystemExit(main())

"""HTTP server and JSON API for the web app.

Endpoints:
- GET  /api/drafts: list drafts on disk (local mode only; empty when shared)
- GET  /api/gallery: list briefings on the public gallery
- POST /api/gallery: publish one briefing (generate already does this)
- POST /api/ideas:  generate briefing questions from a theme
- POST /api/draft:  run the full evidence-grounded research & draft pipeline

Runs in two modes, because a laptop and a shared host want opposite things:

**Local** (`articlegen web`) — writes each draft into `drafts/` and rebuilds the
review queue, matching what the CLI does. This is the default.

**Shared** (`ARTICLEGEN_STATELESS=1`, what the public deployment sets) — renders
the article, lists it on the public gallery, and returns it in the response.
Nothing is written to `drafts/` on the host. A common `drafts/` directory would
make every visitor's article readable at a guessable URL. The gallery copy is a
public GitHub gist; generating puts it there. The browser also keeps its own
copies in localStorage.

The web app does not take a visitor key. A shared host holds
`ARTICLEGEN_PUBLIC_OPENROUTER_KEY` (or `OPENROUTER_API_KEY`), plans and screens
with `OPENROUTER_PUBLIC_MODEL` and writes with `OPENROUTER_PUBLIC_WRITER_MODEL`.
A `key` field in the request body is ignored. Drafts run as polled jobs
(`async: true`), because a hosted run outlives the proxy's ~100s window.
The rate limits are a spend cap, not just a scholarly-API cap. Local
`articlegen web` uses the same env vars; without one, generate returns 400.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import gallery, llm, paperfetch
from .ideas import generate_ideas
from .pipeline import NoPapersFound, generate_draft
from .render import (
    _draft_title,
    _working_draft_sentence,
    build_index,
    render_article,
    render_markdown,
)
from .sources import DEFAULT_MAX_PAPERS, gather_evidence, probe_unpaywall
from .style import errors as style_errors
from .writer import clean_pico, clean_search_terms

DRAFTS_DIR = "drafts"

# Models a caller may ask for by name. Anything else is ignored and the provider
# layer picks from the key it was given. The public hosted path always writes
# with OPENROUTER_PUBLIC_MODEL even if the payload names something else —
# otherwise a crafted POST on the hosted key would bill Opus.
ALLOWED_MODELS = frozenset({
    llm.ANTHROPIC_DEFAULT_MODEL,
    llm.OPENROUTER_DEFAULT_MODEL,
    llm.OPENROUTER_PUBLIC_MODEL,
})

# Every key a run line may carry. The record is built by filtering against this
# set, so a field nobody thought about cannot reach the log. Nothing here is
# content: no topic, theme, search term, query, article text, key or address.
RUN_FIELDS = frozenset({
    # every endpoint
    "kind", "t", "endpoint", "method", "status", "ok", "ms",
    "model", "key", "commit", "stateless", "error",
    # /api/ideas
    "n_asked", "n_ideas", "guidance",
    # /api/draft
    "screened", "cited", "cited_direct", "direct", "related", "tangential",
    "full_text", "full_text_via", "named_added", "named_queries",
    "referenced_added",
    "style_errors", "style_rules", "figures", "unverified", "misattributed",
    "working_draft",
    # /api/gallery
    "backend", "items", "html_kb",
})

RUN_ENDPOINTS = ("/api/ideas", "/api/draft", "/api/gallery")

# Shared hosts set this. Local runs leave it unset and keep writing to drafts/.
STATELESS = os.environ.get("ARTICLEGEN_STATELESS", "").strip().lower() in ("1", "true", "yes")

# Comma-separated origins allowed to call the API from a browser. Default "*"
# suits localhost; the deployment pins it to the GitHub Pages origin.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ARTICLEGEN_ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# Per-IP throttle. Every draft costs the caller LLM tokens but costs *us* calls
# against the shared OpenAlex / Semantic Scholar quotas, which are attached to
# this server's IP — one abusive client can get the whole deployment throttled.
RATE_LIMIT_WINDOW = 3600
RATE_LIMIT_MAX = int(os.environ.get("ARTICLEGEN_RATE_LIMIT", "20"))

# ...and an aggregate ceiling, because the per-IP limit protects nothing the
# scholarly APIs care about. They meter against this server's single egress IP,
# so upstream load scales with visitor count: one popular share of the link
# exhausts the shared quota for everybody while every individual stays politely
# under 20 (#96). Six busy visitors' worth by default.
RATE_LIMIT_TOTAL = int(os.environ.get("ARTICLEGEN_RATE_LIMIT_TOTAL", "120"))

_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}
_rate_hits_all: list[float] = []

# Whether X-Forwarded-For can be believed. Render sets it; a direct-to-internet
# server must not trust it, because then any caller can pick their own bucket
# by sending the header themselves. Render also injects RENDER_GIT_COMMIT, so
# the deployment auto-detects and a self-hoster behind their own proxy sets
# ARTICLEGEN_TRUST_PROXY=1.
TRUST_PROXY = (
    os.environ.get("ARTICLEGEN_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")
    or bool(os.environ.get("RENDER_GIT_COMMIT", "").strip())
)


def _client_ip(peer: str, forwarded_for: str | None) -> str:
    """The address to charge this request to.

    Behind Render's load balancer `client_address[0]` is the *proxy*, so every
    visitor shared one bucket: one abuser locked out everybody, which is the
    exact failure the throttle exists to prevent, and a direct attacker could
    not be attributed at all (#96).

    The **rightmost** X-Forwarded-For entry is the one to use, not the leftmost.
    A caller can send their own X-Forwarded-For header; the proxy appends the
    real peer address to whatever arrived, so the leftmost entry is attacker-
    controlled and the rightmost is the one hop we actually trust.
    """
    if not TRUST_PROXY or not forwarded_for:
        return peer
    parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
    return parts[-1] if parts else peer


def _rate_limited(client_ip: str) -> str | None:
    """Charge this request; return the reason it was refused, or None to allow.

    Two sliding windows: one per address, one across every address. The second
    is what the upstream quota actually experiences.
    """
    now = time.time()
    with _rate_lock:
        _rate_hits_all[:] = [t for t in _rate_hits_all if now - t < RATE_LIMIT_WINDOW]
        hits = [t for t in _rate_hits.get(client_ip, []) if now - t < RATE_LIMIT_WINDOW]
        if len(hits) >= RATE_LIMIT_MAX:
            _rate_hits[client_ip] = hits
            return (f"Rate limit reached ({RATE_LIMIT_MAX} requests/hour). "
                    "Try again later.")
        if len(_rate_hits_all) >= RATE_LIMIT_TOTAL:
            _rate_hits[client_ip] = hits
            return ("This shared server has reached its hourly total across all "
                    f"visitors ({RATE_LIMIT_TOTAL}/hour) and cannot search the "
                    "scholarly databases again yet. This is not your limit — try "
                    "again later, or run the generator locally.")
        hits.append(now)
        _rate_hits[client_ip] = hits
        _rate_hits_all.append(now)
        # Opportunistic sweep so the dict doesn't grow without bound.
        if len(_rate_hits) > 2048:
            for ip in [k for k, v in _rate_hits.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW]:
                _rate_hits.pop(ip, None)
    return None


def _build_info() -> dict:
    """Which commit, from which branch, this process is actually running.

    Same reasoning as `/api/diag`: everything works locally, so questions about
    the *deployment* are otherwise unanswerable from outside. "Is the running
    backend the code I just merged?" and "which branch does the host deploy
    from?" both needed a dashboard login to answer, which meant a rename of the
    default branch could stop deploys without anything visible from here (#47).

    Render injects these; anywhere else they are simply absent, and the keys are
    omitted rather than reported as empty. Both values are public facts about a
    public repository.
    """
    info = {}
    commit = os.environ.get("RENDER_GIT_COMMIT", "").strip()
    branch = os.environ.get("RENDER_GIT_BRANCH", "").strip()
    if commit:
        info["commit"] = commit[:7]
    if branch:
        info["branch"] = branch
    return info


def _requested_model(payload: dict) -> str | None:
    """The caller's model, accepted only from a known list.

    The value reaches an LLM API, so it is matched against the models this
    project actually supports rather than forwarded as given. `None` means the
    provider layer decides from whichever key was supplied.
    """
    requested = (payload.get("model") or "").strip()
    return requested if requested in ALLOWED_MODELS else None


def _hosted_openrouter_key() -> str | None:
    """The key the shared host pays with, if it has one.

    `ARTICLEGEN_PUBLIC_OPENROUTER_KEY` is the dedicated public-billing secret.
    `OPENROUTER_API_KEY` is the fallback so a local `articlegen web` with the
    usual env var already set can generate. Neither is ever returned from an
    API or written to a log line.
    """
    for var in ("ARTICLEGEN_PUBLIC_OPENROUTER_KEY", "OPENROUTER_API_KEY"):
        raw = os.environ.get(var, "").strip()
        if raw:
            return raw
    return None


def public_generation_enabled() -> bool:
    """True when the host has a key and visitors can generate."""
    return _hosted_openrouter_key() is not None


def _credentials(payload: dict) -> tuple[str | None, str | None]:
    """(api_key, model) for this request.

    The web app does not take a visitor key. A hosted key pays, and the model
    is forced to Luna so a crafted POST cannot select Opus on the public bill.
    A `key` in the payload is ignored. No key anywhere: both are None and
    `_missing_key` refuses.
    """
    hosted = _hosted_openrouter_key()
    if hosted:
        return hosted, llm.OPENROUTER_PUBLIC_MODEL
    return None, _requested_model(payload)


def _key_mode() -> str:
    """Who is paying: 'public' (the host's key) or 'none'."""
    if _hosted_openrouter_key():
        return "public"
    return "none"


def _resolve_model_name(model: str | None, api_key: str | None) -> str | None:
    try:
        return llm.resolve_provider(model, api_key)[1]
    except Exception:
        return model


def build_run_record(
    endpoint: str, method: str, status: int, ms: int | float, extra: dict | None = None
) -> dict:
    """One run's log line. Only RUN_FIELDS survive."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {
        "kind": "run",
        "t": now_utc,
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "ok": status < 400,
        "ms": int(ms),
        "stateless": STATELESS,
    }
    commit = _build_info().get("commit")
    if commit:
        rec["commit"] = commit
    if extra:
        for k, v in extra.items():
            if k in RUN_FIELDS:
                rec[k] = v
    return rec


def draft_run_fields(draft) -> dict:
    """Counts describing one finished draft. Never any of its text."""
    try:
        papers = getattr(draft, "papers", None) or []
        cited_refs = getattr(draft, "cited_refs", None) or []
        curation = getattr(draft, "curation", None) or {}
        counts = curation.get("counts") or {}
        relevance = curation.get("relevance") or {}
        provenance = getattr(draft, "provenance", None) or {}
        named_sources = provenance.get("named_sources") or {}
        referenced_sources = provenance.get("referenced_sources") or {}
        verification = getattr(draft, "verification", None) or {}
        style_report = getattr(draft, "style_report", None)

        st_errors = style_errors(style_report) if style_report else []
        style_rules = sorted({err["rule"] for err in st_errors if "rule" in err})

        fields = {
            "screened": len(papers),
            "cited": len(cited_refs),
            "cited_direct": sum(1 for r in cited_refs if relevance.get(r) == "direct"),
            "direct": counts.get("direct", 0),
            "related": counts.get("related", 0),
            "tangential": counts.get("tangential", 0),
            "full_text": len(provenance.get("full_text_sources") or []),
            "named_added": named_sources.get("added", 0),
            "named_queries": len(named_sources.get("queries") or []),
            "referenced_added": referenced_sources.get("added", 0),
            "style_errors": len(st_errors),
            "style_rules": style_rules,
            "figures": verification.get("total", 0),
            "unverified": len(verification.get("unverified") or []),
            "misattributed": len(verification.get("misattributed") or []),
            "working_draft": bool(_working_draft_sentence(style_report, verification)),
        }
        if "full_text_via" in provenance:
            fields["full_text_via"] = provenance["full_text_via"]
        return fields
    except Exception:
        return {}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "article"


class ArticleGenHandler(SimpleHTTPRequestHandler):
    # SimpleHTTPRequestHandler defaults to HTTP/1.0, which has no keep-alive:
    # the server closes the connection after every response. Browsers and
    # reverse proxies pool connections, so they reuse one the server has
    # already hung up on and the request fails instantly — intermittently,
    # depending on whether that particular request drew a pooled or a fresh
    # connection. It looks like flaky networking and isn't: roughly every
    # other fetch from the deployed front end failed in ~140ms.
    #
    # curl hides the bug completely, because each invocation opens its own
    # connection. Only a connection-pooling client sees it.
    #
    # HTTP/1.1 requires an accurate Content-Length (or chunked) on every
    # response or the client hangs waiting for a body. _send_json sets it,
    # SimpleHTTPRequestHandler sets it for files and send_error, and
    # do_OPTIONS sends an explicit zero below.
    protocol_version = "HTTP/1.1"

    _run_status: int = 0
    _run_extra: dict | None = None

    def _note_run(self, **fields) -> None:
        """Record fields for this request's run line. Unknown keys are dropped."""
        if self._run_extra is None:
            self._run_extra = {}
        for k, v in fields.items():
            if k in RUN_FIELDS:
                self._run_extra[k] = v

    def _emit_run(self, endpoint: str, method: str, started: float) -> None:
        try:
            ms = int((time.time() - started) * 1000)
            status = self._run_status or 200
            record = build_run_record(endpoint, method, status, ms, self._run_extra)
            sys.stderr.write(json.dumps(record) + "\n")
            try:
                gallery.append_run(record)
            except Exception as exc:
                sys.stderr.write(f"[web] run-log gist write failed: {type(exc).__name__}\n")
        except Exception:
            pass

    def log_message(self, format_str: str, *args) -> None:
        sys.stderr.write(f"[web] {format_str % args}\n")

    def _missing_key(self, api_key: str | None) -> bool:
        """Reject a keyless request in the caller's language, not the server's.

        Without this the provider layer raises "OPENROUTER_API_KEY environment
        variable is not set", which is true and useless to someone using the web
        app — they have no environment to set. Fail only when the host has no
        key either.
        """
        if api_key or public_generation_enabled() or os.environ.get("ANTHROPIC_API_KEY"):
            return False
        self._note_run(error="missing_key")
        self._send_json(
            {"error": "This server has no OpenRouter key, so it cannot generate. "
                      "Set OPENROUTER_API_KEY (or ARTICLEGEN_PUBLIC_OPENROUTER_KEY) "
                      "on the host."},
            status=400,
        )
        return True

    # Messages that name a cause the caller can act on. Everything else is an
    # internal fault and gets the generic sentence — a visitor was once shown
    # `RuntimeError("... returned invalid JSON: ..." + 500 characters of raw
    # JSON)` on their phone (#95). The full text still goes to the server log,
    # which is where it is useful.
    _ACTIONABLE = ("api key", "credit", "rate limit", "quota", "spending limit",
                   "refused", "content_filter", "too long", "unsupported model")

    def _unexpected(self, doing: str, exc: Exception) -> str:
        """A sentence for the visitor; the detail goes to the log."""
        self._note_run(error=type(exc).__name__)
        detail = f"{type(exc).__name__}: {exc}"
        self._log_stage(f"ERROR while {doing}: {detail}")
        lowered = str(exc).lower()
        if any(hint in lowered for hint in self._ACTIONABLE):
            return str(exc)
        return (f"Something went wrong {doing}, and it was not your request's fault. "
                "Try again — if it keeps happening, the backend log has the detail.")

    def _over_rate_limit(self) -> bool:
        """Charge the throttle for a request that is about to do real work.

        Checked after validation, not before: a malformed request costs nothing
        upstream, and locking someone out for a typo in the form is a worse
        failure than the flood it would prevent.
        """
        refusal = _rate_limited(
            _client_ip(self.client_address[0], self.headers.get("X-Forwarded-For"))
        )
        if refusal is None:
            return False
        self._note_run(error="rate_limited")
        self._send_json({"error": refusal}, status=429)
        return True

    def _cors_origin(self) -> str | None:
        """Echo the caller's origin when it's allowed. None means: send no header."""
        if ALLOWED_ORIGINS == ["*"]:
            return "*"
        origin = self.headers.get("Origin")
        return origin if origin in ALLOWED_ORIGINS else None

    def _send_cors_headers(self) -> None:
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            if origin != "*":
                self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, data: dict, status: int = 200) -> None:
        self._run_status = status
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        # Explicit zero: under HTTP/1.1 a response with neither Content-Length
        # nor chunked encoding leaves the client waiting for a body.
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        started = time.time()
        self._run_status, self._run_extra = 0, {}
        path_without_slash = self.path.rstrip("/")
        is_run_endpoint = path_without_slash in RUN_ENDPOINTS or self.path in RUN_ENDPOINTS
        try:
            if self.path in ("/api/drafts", "/api/drafts/"):
                self._handle_get_drafts()
                return
            if self.path.startswith("/api/draft/"):
                self._handle_job(self.path[len("/api/draft/"):].strip("/"))
                return
            if self.path in ("/api/health", "/api/health/"):
                public = public_generation_enabled()
                body = {"ok": True, "stateless": STATELESS, "public": public,
                        "gallery": gallery.gallery_enabled(),
                        **_build_info()}
                if public:
                    body["public_model"] = llm.OPENROUTER_PUBLIC_MODEL
                    body["public_writer_model"] = llm.OPENROUTER_PUBLIC_WRITER_MODEL
                self._send_json(body)
                return
            if self.path in ("/api/gallery", "/api/gallery/"):
                self._handle_get_gallery()
                return
            if self.path in ("/api/diag", "/api/diag/"):
                self._handle_diag()
                return
            super().do_GET()
        finally:
            if is_run_endpoint:
                endpoint = "/api/gallery" if path_without_slash == "/api/gallery" else self.path
                self._emit_run(endpoint, "GET", started)

    def do_POST(self) -> None:
        started = time.time()
        self._run_status, self._run_extra = 0, {}
        try:
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                self._note_run(error="invalid_content_length")
                self._send_json({"error": "Invalid Content-Length"}, status=400)
                return
            # Ideas and draft payloads are small. Share to gallery sends the
            # article HTML; GALLERY_MAX_HTML is 400 KB, and the JSON envelope
            # needs a little more. Anything past that is not a real request.
            if content_length > 512 * 1024:
                self._note_run(error="body_too_large")
                self._send_json({"error": "Request body too large."}, status=413)
                return

            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(post_data.decode("utf-8")) if post_data else {}
            except Exception:
                self._note_run(error="invalid_json")
                self._send_json({"error": "Invalid JSON payload"}, status=400)
                return
            if not isinstance(payload, dict):
                self._note_run(error="invalid_json")
                self._send_json({"error": "Invalid JSON payload"}, status=400)
                return

            if self.path == "/api/ideas":
                self._handle_ideas(payload)
            elif self.path == "/api/draft":
                self._handle_draft(payload)
            elif self.path == "/api/gallery":
                self._handle_publish_gallery(payload)
            else:
                self._send_json({"error": "Endpoint not found"}, status=404)
        finally:
            if self.path in RUN_ENDPOINTS:
                self._emit_run(self.path, "POST", started)

    def _handle_get_gallery(self) -> None:
        """List shared briefings. Empty when the store is missing, never 500."""
        items = gallery.list_items()
        self._note_run(backend=gallery._backend(), items=len(items))
        self._send_json({
            "enabled": gallery.gallery_enabled(),
            "items": items,
        })

    def _handle_publish_gallery(self, payload: dict) -> None:
        """Publish one briefing. Generate already lists it; this is the retry path."""
        self._note_run(backend=gallery._backend())
        html = payload.get("html")
        if not isinstance(html, str):
            self._note_run(error="invalid_html")
            self._send_json({"error": "Generate a briefing first."}, status=400)
            return
        self._note_run(html_kb=len(html.encode("utf-8")) // 1024)
        title = payload.get("title") if isinstance(payload.get("title"), str) else ""
        reason = gallery.validate_html(html)
        if reason:
            self._note_run(error="invalid_html")
            self._send_json({"error": reason}, status=400)
            return
        if self._over_rate_limit():
            return
        try:
            item = gallery.publish(title, html)
        except gallery.GalleryError as exc:
            self._note_run(error=type(exc).__name__)
            self._send_json({"error": str(exc)}, status=503)
            return
        except Exception as exc:
            self._send_json(
                {"error": self._unexpected("sharing to the gallery", exc)},
                status=500,
            )
            return
        self._send_json({"ok": True, "item": item})

    def _handle_get_drafts(self) -> None:
        if not os.path.exists(DRAFTS_DIR):
            self._send_json({"drafts": []})
            return

        html_files = [
            p for p in glob.glob(os.path.join(DRAFTS_DIR, "*.html"))
            if os.path.basename(p) != "index.html"
        ]
        html_files.sort(key=os.path.getmtime, reverse=True)

        results = []
        for path in html_files:
            name = os.path.basename(path)
            title = _draft_title(path)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            md_name = name[:-5] + ".md"
            has_md = os.path.exists(os.path.join(DRAFTS_DIR, md_name))
            results.append({
                "filename": name,
                "title": title,
                "date": mtime,
                "html_url": f"/drafts/{name}",
                "md_url": f"/drafts/{md_name}" if has_md else None,
            })

        self._send_json({"drafts": results})

    def _handle_ideas(self, payload: dict) -> None:
        theme = (payload.get("theme") or "").strip()
        guidance = (payload.get("guidance") or "").strip()
        api_key, model = _credentials(payload)
        resolved = _resolve_model_name(model, api_key)
        try:
            n = max(1, min(int(payload.get("n") or 6), 12))
        except (TypeError, ValueError):
            n = 6

        self._note_run(
            key=_key_mode(),
            model=resolved,
            n_asked=n,
            guidance=bool(guidance),
        )

        if not theme:
            self._note_run(error="missing_theme")
            self._send_json({"error": "Please provide a theme."}, status=400)
            return
        if self._missing_key(api_key) or self._over_rate_limit():
            return

        prompt_theme = theme
        if guidance:
            prompt_theme = f"{theme} — guidance: {guidance[:300]}"

        try:
            ideas = generate_ideas(prompt_theme, n=n, api_key=api_key,
                                   model=model)
            self._note_run(n_ideas=len(ideas))
            self._send_json({"theme": theme, "ideas": ideas})
        except Exception as exc:
            self._send_json({"error": self._unexpected("generating ideas", exc)}, status=500)

    def _draft_params(self, payload: dict) -> dict | None:
        """Validate a draft request. Sends the 400 and returns None when it fails."""
        topic = (payload.get("topic") or "").strip()
        style = (payload.get("style") or "").strip()
        api_key, model = _credentials(payload)
        resolved = _resolve_model_name(model, api_key)

        self._note_run(
            key=_key_mode(),
            model=resolved,
        )

        if not topic:
            self._note_run(error="missing_topic")
            self._send_json({"error": "Please provide a briefing question."}, status=400)
            return None

        if len(topic) > 300:
            self._note_run(error="topic_too_long")
            self._send_json({"error": "Topic is too long (300 characters max)."}, status=400)
            return None

        raw_terms = payload.get("search_terms")
        if raw_terms is not None and not isinstance(raw_terms, (list, tuple)):
            self._note_run(error="bad_search_terms")
            self._send_json({"error": "search_terms must be a list of strings."}, status=400)
            return None
        search_terms = clean_search_terms(raw_terms)

        raw_pico = payload.get("pico")
        if raw_pico is not None and not isinstance(raw_pico, dict):
            self._note_run(error="bad_pico")
            self._send_json({"error": "pico must be an object of strings."}, status=400)
            return None
        pico = clean_pico(raw_pico)

        return {
            "topic": topic, "style": style, "api_key": api_key, "model": model,
            "resolved": resolved, "search_terms": search_terms, "pico": pico,
        }

    def _handle_draft(self, payload: dict) -> None:
        params = self._draft_params(payload)
        if params is None:
            return
        if self._missing_key(params["api_key"]) or self._over_rate_limit():
            return

        # `async: true` is what the current front end sends: the run continues
        # in a thread and the page polls GET /api/draft/<job>. Without it the
        # request blocks until the briefing is written, as it always did — kept
        # for a cached old page and for scripts, but a run that outlives
        # Cloudflare's ~100s window in front of Render is lost on that path.
        if payload.get("async"):
            job_id = start_draft_job(params)
            if job_id is None:
                self._note_run(error="busy")
                self._send_json(
                    {"error": "The server is writing as many briefings as it can at "
                              "once. Try again in a minute."}, status=429)
                return
            self._send_json({"ok": True, "job": job_id, "status": "running"}, status=202)
            return

        status, body, fields = produce_draft(params, log=self._log_stage)
        self._note_run(**fields)
        self._send_json(body, status=status)

    def _handle_job(self, job_id: str) -> None:
        """Where a polling page reads a running draft's stage, then its result."""
        job = job_snapshot(job_id)
        if job is None:
            self._send_json(
                {"status": "missing",
                 "error": "This briefing run is not on the server any more. The host "
                          "may have restarted; start it again."}, status=404)
            return
        self._send_json(job)

    def _handle_diag(self) -> None:
        """Can this host reach the scholarly APIs at all?

        Needs no API key, because it exercises only the keyless half of the
        pipeline. Locally everything works, so when a deployment reports "no
        papers found" the question is whether *that host* is being refused —
        and there is no way to answer it from outside without asking the host.

        Two consequences of that follow. It bypasses the search cache, because a
        cached answer cannot tell you what the sources are doing right now. And
        it is therefore throttled like any other expensive endpoint: each call
        spends real requests against the same shared quota drafting depends on,
        and an unmetered probe is a way for anyone who finds the URL to exhaust
        the very quota the throttle exists to protect.
        """
        if self._over_rate_limit():
            return
        outcomes: list[dict] = []
        try:
            papers = gather_evidence(
                ["shift work sleep"], max_papers=3, per_query=3,
                topic="shift work sleep", outcomes=outcomes, use_cache=False,
            )
            count = len(papers)
        except Exception as exc:
            count, outcomes = 0, [{"source": "gather_evidence", "query": "",
                                   "count": 0, "error": f"{type(exc).__name__}: {exc}"}]
        # The full-text path depends on a fourth keyless service that the three
        # search probes above never touch. It fails soft, so a block shows up
        # only as quietly halved full-text coverage — right for the reader,
        # invisible to the operator, and undiagnosable from outside (#104).
        try:
            unpaywall = probe_unpaywall()
        except Exception as exc:
            unpaywall = {"source": "unpaywall", "error": f"{type(exc).__name__}: {exc}"}

        self._send_json({
            "papers_found": count,
            "sources": outcomes,
            "full_text": unpaywall,
            "openalex_mailto_set": bool(paperfetch.env_get("OPENALEX_MAILTO")),
            "semantic_scholar_key_set": bool(paperfetch.env_get("SEMANTIC_SCHOLAR_API_KEY")),
        })

    def _log_stage(self, message: str) -> None:
        """Pipeline progress to the server log. Never carries the caller's key."""
        sys.stderr.write(f"[draft] {message}\n")


# --------------------------------------------------------------- draft jobs
#
# A hosted briefing takes longer than Cloudflare will hold a request open in
# front of Render (~100s): the stronger writer alone runs 40-80s, after 30-60s
# of search and screening. So the page starts a job and polls it. The job lives
# in this process only — a restart loses it, and the poll says so — and is
# dropped JOB_TTL seconds after it finishes. Nothing is written to disk.

JOB_TTL = 30 * 60
MAX_RUNNING_JOBS = int(os.environ.get("ARTICLEGEN_MAX_JOBS", "4"))

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# The pipeline's own log lines, mapped to what a visitor can follow. Only the
# stage name and a count ever leave the server: never a query, a title or an
# error's detail.
_STAGE_PATTERNS = (
    (re.compile(r"^Planning search queries"), "planning", lambda m: "Planning the searches"),
    (re.compile(r"^Fetching journal articles"), "searching",
     lambda m: "Searching Semantic Scholar, OpenAlex, Europe PMC and arXiv"),
    (re.compile(r"^Collected (\d+) candidate papers"), "searching",
     lambda m: f"Found {m.group(1)} candidate papers"),
    (re.compile(r"^Assessing source relevance"), "screening",
     lambda m: "Rating how directly each paper answers the question"),
    (re.compile(r"relevance: (\d+) direct / (\d+) related"), "screening",
     lambda m: f"{m.group(1)} papers address the question directly, {m.group(2)} are related"),
    (re.compile(r"^Writing the (?:briefing|review)"), "writing",
     lambda m: "Writing a first draft from the abstracts"),
    (re.compile(r"^Fetching open-access full texts"), "reading",
     lambda m: "Fetching open-access full texts of the cited papers"),
    (re.compile(r"full text retrieved for (\d+) cited"), "reading",
     lambda m: (f"Read {m.group(1)} full text{'s' if m.group(1) != '1' else ''}"
                if m.group(1) != "0" else "No open-access full text found; using abstracts")),
    (re.compile(r"^Rewriting with cited full text"), "writing",
     lambda m: "Rewriting the briefing with the full texts"),
    (re.compile(r"^Prose style:"), "checking", lambda m: "Checking the prose"),
    (re.compile(r"^Statistics:"), "checking",
     lambda m: "Checking every figure against the paper it cites"),
)


def public_stage(message: str) -> tuple[str, str] | None:
    """(stage, message) a visitor may see for one pipeline log line, or None."""
    text = (message or "").strip()
    for pattern, stage, render in _STAGE_PATTERNS:
        m = pattern.search(text)
        if m:
            return stage, render(m)
    return None


def _unexpected_message(doing: str, exc: Exception, log) -> str:
    """The sentence a visitor sees for an internal failure; detail goes to the log."""
    log(f"ERROR while {doing}: {type(exc).__name__}: {exc}")
    lowered = str(exc).lower()
    if any(hint in lowered for hint in ArticleGenHandler._ACTIONABLE):
        return str(exc)
    return (f"Something went wrong {doing}, and it was not your request's fault. "
            "Try again — if it keeps happening, the backend log has the detail.")


def produce_draft(params: dict, log) -> tuple[int, dict, dict]:
    """Run the pipeline for validated `params`. Returns (status, body, run fields).

    The one copy used by the blocking request and by a job thread alike.
    """
    topic = params["topic"]
    search_terms, pico = params["search_terms"], params["pico"]
    try:
        draft = generate_draft(
            topic, style_note=params["style"][:500], max_papers=DEFAULT_MAX_PAPERS,
            api_key=params["api_key"], model=params["model"], log=log,
            search_terms=search_terms,
            pico=pico,
            # Hosted only: the host's key writes with the stronger model.
            writer_model=(llm.OPENROUTER_PUBLIC_WRITER_MODEL
                          if _hosted_openrouter_key() else None),
        )
    except NoPapersFound as exc:
        # 503 when the upstream APIs are the problem: it's not the caller's
        # request that was unprocessable, and the distinction tells them
        # whether to reword the topic or simply try again.
        return (503 if exc.sources_failed else 422, {"error": str(exc)},
                {"error": type(exc).__name__})
    except Exception as exc:
        return (500, {"error": _unexpected_message("writing the briefing", exc, log)},
                {"error": type(exc).__name__})

    fields = dict(draft_run_fields(draft),
                  model=draft.provenance.get("model") or params["resolved"])

    render_args = (
        draft.article, draft.papers, draft.topic,
        draft.curation, draft.verification, draft.provenance,
        draft.style_report,
    )
    # The copy sent back goes into a sandboxed iframe in the front end, so
    # it carries no scripts and no toolbar (see render_article's docstring).
    # The copy written to drafts/ below is opened directly in a browser with
    # no app around it, so it keeps both — same file the CLI produces.
    article_html = render_article(*render_args, standalone=False)
    article_md = render_markdown(*render_args)

    response = {
        "ok": True,
        "title": draft.article.get("title", topic),
        "stem": f"{datetime.date.today().isoformat()}-{_slugify(topic)}",
        "sources_count": len(draft.cited_refs),
        "summary": draft.summary(),
        "html": article_html,
        "markdown": article_md,
    }

    if not STATELESS:
        # Local mode: mirror the CLI so `articlegen queue` sees the same drafts.
        os.makedirs(DRAFTS_DIR, exist_ok=True)
        html_name = f"{response['stem']}.html"
        md_name = f"{response['stem']}.md"
        with open(os.path.join(DRAFTS_DIR, html_name), "w", encoding="utf-8") as f:
            f.write(render_article(*render_args))
        with open(os.path.join(DRAFTS_DIR, md_name), "w", encoding="utf-8") as f:
            f.write(article_md)
        build_index(DRAFTS_DIR)
        response["html_url"] = f"/drafts/{html_name}"
        response["md_url"] = f"/drafts/{md_name}"

    item = gallery.try_publish(response["title"], article_html)
    if item:
        response["gallery"] = item
    return 200, response, fields


def _prune_jobs(now: float) -> None:
    for job_id in [k for k, j in _jobs.items()
                   if j.get("finished") and now - j["finished"] > JOB_TTL]:
        del _jobs[job_id]


def start_draft_job(params: dict) -> str | None:
    """Start a draft in a thread. None when MAX_RUNNING_JOBS are already running."""
    import secrets

    now = time.time()
    with _jobs_lock:
        _prune_jobs(now)
        if sum(1 for j in _jobs.values() if j["status"] == "running") >= MAX_RUNNING_JOBS:
            return None
        job_id = secrets.token_hex(16)
        _jobs[job_id] = {"status": "running", "stage": "planning",
                         "message": "Planning the searches", "started": now,
                         "finished": None, "result": None, "error": None,
                         "http_status": None}

    def log(message: str) -> None:
        sys.stderr.write(f"[draft {job_id[:8]}] {message}\n")
        stage = public_stage(message)
        if stage:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job["stage"], job["message"] = stage

    def work() -> None:
        try:
            status, body, fields = produce_draft(params, log)
        except Exception as exc:  # produce_draft catches; this is the backstop
            status, body, fields = 500, {"error": _unexpected_message(
                "writing the briefing", exc, log)}, {"error": type(exc).__name__}
        finished = time.time()
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="done" if status < 400 else "error",
                           finished=finished, http_status=status,
                           result=body if status < 400 else None,
                           error=body.get("error") if status >= 400 else None)
        # The run line a blocking request would have written when it returned.
        try:
            record = build_run_record(
                "/api/draft", "POST", status, (finished - now) * 1000,
                dict({"key": _key_mode(), "model": params["resolved"]}, **fields))
            sys.stderr.write(json.dumps(record) + "\n")
            gallery.append_run(record)
        except Exception as exc:
            sys.stderr.write(f"[web] run-log write failed: {type(exc).__name__}\n")

    threading.Thread(target=work, name=f"draft-{job_id[:8]}", daemon=True).start()
    return job_id


def job_snapshot(job_id: str) -> dict | None:
    """What a poll returns: stage while running, the briefing once done."""
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        return None
    with _jobs_lock:
        _prune_jobs(time.time())
        job = _jobs.get(job_id)
        if job is None:
            return None
        end = job["finished"] or time.time()
        out = {"status": job["status"], "stage": job["stage"], "message": job["message"],
               "elapsed": int(end - job["started"])}
        if job["status"] == "done":
            out["result"] = job["result"]
        elif job["status"] == "error":
            out["error"] = job["error"]
            out["http_status"] = job["http_status"]
        return out


def run_server(port: int = 8000, directory: str = ".") -> None:
    os.chdir(directory)
    handler = ArticleGenHandler
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    mode = "stateless (nothing written to disk)" if STATELESS else f"local (writing to {DRAFTS_DIR}/)"
    print(f"🚀 Article Generator running at http://localhost:{port}/  — {mode}")
    if ALLOWED_ORIGINS != ["*"]:
        print(f"   CORS restricted to: {', '.join(ALLOWED_ORIGINS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

"""Command-line interface — a three-stage workflow:

    articlegen ideas  "<theme>"        # 1. briefing questions to pick from
    articlegen draft  "<title>"        # 2. research + briefing, into drafts/
    articlegen queue                   # (re)build / open the drafts index
    articlegen render drafts/x.json    # rebuild HTML + Markdown from a run manifest
    articlegen refresh drafts/x.json   # re-run the queries; list what is new

Stages 1 and 2 are the two human gates: you choose a question, then you review the briefing.
`draft --long` writes the parked journal-style Review instead.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import webbrowser

from . import demo
from .ideas import format_ideas_console, generate_ideas, ideas_to_markdown
from .llm import resolve_provider
from . import paperfetch
from .pipeline import (Draft, NoPapersFound, RefreshReport, generate_draft,
                       refresh_draft, rerun_draft)
from .render import build_index, render_article, render_markdown
from .sources import DEFAULT_MAX_PAPERS

IDEAS_DIR = "ideas"
DRAFTS_DIR = "drafts"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "article"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _open_in_browser(path: str) -> None:
    try:
        webbrowser.open(f"file://{os.path.abspath(path)}")
    except Exception:
        pass  # headless / no browser — the path is already logged


def _api_error(exc: Exception, model: str | None = None) -> int:
    # Resolve with the model the caller actually asked for. Without it this
    # reported whatever the *default* provider would have been — so a failed
    # `--model cli:sonnet` run blamed the default provider, and advised setting
    # a key that would not have been used.
    provider, resolved = resolve_provider(model)
    _log(f"The {provider} call failed (model {resolved}): {exc}")
    if provider == "claude-cli":
        _log(
            "This provider runs on your Claude subscription through the `claude` CLI. "
            "Check `claude --version` works and that you are signed in, or pick a "
            "provider that takes an API key."
        )
    else:
        _log(
            "Set OPENROUTER_API_KEY (https://openrouter.ai/keys) or "
            "ANTHROPIC_API_KEY, and try again. To draft on a Claude "
            "subscription with no key at all, use --model cli:opus."
        )
    return 1


def cmd_ideas(args) -> int:
    _log(f"Generating ideas for: {args.theme}")
    try:
        ideas = generate_ideas(args.theme, n=args.n, model=args.model)
    except Exception as exc:
        return _api_error(exc, args.model)

    os.makedirs(IDEAS_DIR, exist_ok=True)
    out_path = args.output or os.path.join(IDEAS_DIR, f"{_slugify(args.theme)}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ideas_to_markdown(args.theme, ideas))

    print(format_ideas_console(ideas))
    _log("")
    _log(f"Saved {len(ideas)} ideas to {out_path}")
    _log('Pick one, then run:  articlegen draft "<title>" --open')
    return 0


def cmd_draft(args) -> int:
    try:
        draft = generate_draft(
            args.topic,
            style_note=args.style,
            max_papers=args.max_papers,
            model=args.model,
            log=_log,
            long=getattr(args, "long", False),
            search_terms=[t.strip() for t in (getattr(args, "queries", "") or "").split(",") if t.strip()],
        )
    except NoPapersFound as exc:
        _log(str(exc))
        return 1
    except Exception as exc:
        return _api_error(exc, args.model)

    os.makedirs(DRAFTS_DIR, exist_ok=True)
    date = datetime.date.today().isoformat()
    stem = args.name or f"{date}-{_slugify(args.topic)}"
    return _write_draft_files(
        draft, DRAFTS_DIR, stem,
        open_html=args.open,
        queue_ckn=getattr(args, "queue_ckn", False),
    )


def _maybe_queue_ckn(draft: Draft, queue_ckn: bool) -> None:
    if not queue_ckn:
        return
    paywalled = draft.paywalled_cited()
    if not paywalled:
        _log("  --queue-ckn: no paywalled cited sources")
        return
    n = paperfetch.queue_paywalled(paywalled, log=_log)
    _log(f"  queued {n} DOI(s) for CKN")


def _write_draft_files(
    draft: Draft,
    out_dir: str,
    stem: str,
    *,
    open_html: bool,
    queue_ckn: bool,
    refresh_index: bool = True,
) -> int:
    html_path = os.path.join(out_dir, f"{stem}.html")
    md_path = os.path.join(out_dir, f"{stem}.md")
    manifest_path = os.path.join(out_dir, f"{stem}.json")
    render_args = (
        draft.article, draft.papers, draft.topic,
        draft.curation, draft.verification, draft.provenance,
        draft.style_report,
    )
    _write_rendered(render_args, html_path, md_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(draft.to_dict(), f, ensure_ascii=False, indent=2)
    _log(f"Draft ready ({len(draft.article.get('references') or [])} sources cited):")
    _log(f"  HTML:     {html_path}")
    _log(f"  Markdown: {md_path}")
    _log(f"  Manifest: {manifest_path}")
    if refresh_index and os.path.isdir(DRAFTS_DIR):
        index_path = build_index(DRAFTS_DIR)
        _log(f"  Queue:    {index_path}")
    print(f"EVIDENCE_SUMMARY: {draft.summary()}")
    _maybe_queue_ckn(draft, queue_ckn)
    if open_html:
        _open_in_browser(html_path)
    return 0


def _write_rendered(render_args: tuple, html_path: str, md_path: str) -> None:
    """Write the HTML and Markdown for one draft, through the shared renderers."""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_article(*render_args))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(*render_args))


def cmd_render(args) -> int:
    """Rebuild the HTML and Markdown from a run manifest, next to it."""
    manifest_path = args.manifest
    try:
        with open(manifest_path, encoding="utf-8") as f:
            draft = Draft.from_dict(json.load(f))
    except (OSError, ValueError, KeyError) as exc:
        _log(f"Could not read the manifest {manifest_path}: {exc}")
        return 1

    stem, _ = os.path.splitext(manifest_path)
    html_path, md_path = f"{stem}.html", f"{stem}.md"
    render_args = (
        draft.article, draft.papers, draft.topic,
        draft.curation, draft.verification, draft.provenance,
        draft.style_report,
    )
    _write_rendered(render_args, html_path, md_path)
    _log(f"Rendered from {manifest_path}:")
    _log(f"  HTML:     {html_path}")
    _log(f"  Markdown: {md_path}")
    if args.open:
        _open_in_browser(html_path)
    return 0


def cmd_rerun(args) -> int:
    """Reuse a run's pool and labels; fetch full text again; write a new draft."""
    manifest_path = args.manifest
    try:
        with open(manifest_path, encoding="utf-8") as f:
            draft = Draft.from_dict(json.load(f))
    except (OSError, ValueError, KeyError) as exc:
        _log(f"Could not read the manifest {manifest_path}: {exc}")
        return 1
    try:
        draft = rerun_draft(
            draft,
            model=args.model,
            log=_log,
            long=getattr(args, "long", False),
        )
    except Exception as exc:
        return _api_error(exc, args.model)

    out_dir = os.path.dirname(os.path.abspath(manifest_path)) or DRAFTS_DIR
    stem, _ = os.path.splitext(os.path.basename(manifest_path))
    if not stem.endswith("-rerun"):
        stem = f"{stem}-rerun"
    return _write_draft_files(
        draft, out_dir, stem,
        open_html=args.open,
        queue_ckn=getattr(args, "queue_ckn", False),
    )


def cmd_refresh(args) -> int:
    """Re-run a manifest's queries and list new direct records; rewrite only with --rewrite."""
    manifest_path = args.manifest
    try:
        with open(manifest_path, encoding="utf-8") as f:
            draft = Draft.from_dict(json.load(f))
    except (OSError, ValueError, KeyError) as exc:
        _log(f"Could not read the manifest {manifest_path}: {exc}")
        return 1

    try:
        report = refresh_draft(
            draft, model=args.model, log=_log,
            rewrite=args.rewrite, long=getattr(args, "long", False),
        )
    except NoPapersFound as exc:
        _log(str(exc))
        return 1
    except Exception as exc:
        return _api_error(exc, args.model)

    if report.curation_error:
        _log(f"Could not label the new records ({report.curation_error}); "
             "nothing is reported as new.")
        return 1

    if report.new_direct:
        print(f"New direct records since {report.date} "
              f"({len(report.queries)} queries re-run, {report.screened} new record(s) screened):")
        for i, paper in enumerate(report.new_direct, start=1):
            print(f"  {i}. {paper.title} ({paper.year}, {paper.venue})")
            if paper.link:
                print(f"     {paper.link}")
        if report.new_related:
            print(f"({len(report.new_related)} new related record(s) also found)")
    else:
        print(f"No new direct records since {report.date}.")

    print(f"REFRESH_SUMMARY: {report.summary()}")

    if report.draft is None:
        if report.new_direct:
            _log("Nothing was written. Add --rewrite to rewrite the briefing with these records.")
        return 0

    out_dir = os.path.dirname(os.path.abspath(manifest_path)) or DRAFTS_DIR
    stem, _ = os.path.splitext(os.path.basename(manifest_path))
    if not stem.endswith("-refresh"):
        stem = f"{stem}-refresh"
    return _write_draft_files(
        report.draft, out_dir, stem,
        open_html=args.open,
        queue_ckn=getattr(args, "queue_ckn", False),
    )


def cmd_queue(args) -> int:
    if not os.path.isdir(DRAFTS_DIR):
        _log(f"No {DRAFTS_DIR}/ folder yet — run `articlegen draft \"<title>\"` first.")
        return 1
    index_path = build_index(DRAFTS_DIR)
    _log(f"Review queue: {index_path}")
    if args.open:
        _open_in_browser(index_path)
    return 0


def cmd_demo(args) -> int:
    output_path = args.output or "demo.html"
    sample = demo.SAMPLE_ARTICLE if getattr(args, "long", False) else demo.SAMPLE_BRIEFING
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_article(
            sample, demo.SAMPLE_PAPERS, "the functions of sleep in the brain",
            demo.SAMPLE_CURATION, None, demo.SAMPLE_PROVENANCE,
        ))
    _log(f"Demo written to {output_path}")
    if args.open:
        _open_in_browser(output_path)
    return 0


def cmd_web(args) -> int:
    from .web import run_server
    port = args.port
    _log(f"Starting mobile web app server at http://localhost:{port}/")
    if args.open:
        _open_in_browser(f"http://localhost:{port}/")
    run_server(port=port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="articlegen",
        description=(
            "Evidence brief: sourced evidence briefings from a question. "
            "Pick a briefing question, auto-collate research, and prepare a "
            "sourced one-page briefing for your review — as a self-contained "
            "HTML file (plus Markdown), grounded in journal-article abstracts "
            "and open-access full texts. Every statistic is checked against "
            "the cited papers."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model to use; the name picks the provider. cli:opus / cli:sonnet run on "
            "your Claude subscription through the Claude Code CLI, and "
            "agy:gemini-3.6-flash-high on your Gemini subscription through the "
            "Antigravity CLI (both need no API key and are local only); "
            "vendor/model -> OpenRouter; claude-* -> Anthropic. Default: auto "
            "— anthropic/claude-opus-5 (OpenRouter is the default provider), or "
            "claude-fable-5 when only an Anthropic key is set."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ideas = sub.add_parser("ideas", help="Stage 1: turn a theme into briefing questions to pick from")
    p_ideas.add_argument("theme", help="Broad theme or interest area")
    p_ideas.add_argument("-n", type=int, default=6, help="How many ideas to generate (default: 6)")
    p_ideas.add_argument("-o", "--output", help="Output .md path (default: ideas/<theme>.md)")
    p_ideas.set_defaults(func=cmd_ideas)

    p_draft = sub.add_parser("draft", help="Stage 2: research + write a sourced briefing")
    p_draft.add_argument("topic", help="The briefing question to write")
    p_draft.add_argument("--name", help="Draft filename stem (default: <date>-<slug>)")
    p_draft.add_argument("--style", default="", help='Optional extra constraint, e.g. "Australian spelling"')
    p_draft.add_argument(
        "--queries", default="",
        help="Comma-separated search terms from the idea card. They are searched as "
             "given; the planner may add one more specific query.",
    )
    p_draft.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS,
                         help=f"Max candidate papers (default: {DEFAULT_MAX_PAPERS})")
    p_draft.add_argument(
        "--long", action="store_true",
        help="Write the journal-style Review instead of the default briefing",
    )
    p_draft.add_argument("--open", action="store_true", help="Open the draft in your browser when done")
    p_draft.add_argument(
        "--queue-ckn", action="store_true",
        help="Add paywalled cited DOIs to the CKN pickup list (papers queue). Off by default.",
    )
    p_draft.set_defaults(func=cmd_draft)

    p_render = sub.add_parser(
        "render",
        help="Rebuild the HTML and Markdown from a run manifest (drafts/<stem>.json), no API calls",
    )
    p_render.add_argument("manifest", help="Path to the .json manifest a draft run wrote")
    p_render.add_argument("--open", action="store_true", help="Open the rendered HTML in your browser")
    p_render.set_defaults(func=cmd_render)

    p_rerun = sub.add_parser(
        "rerun",
        help="Reuse a run's pool and labels, fetch full text again, write a new briefing",
    )
    p_rerun.add_argument("manifest", help="Path to the .json manifest a draft run wrote")
    p_rerun.add_argument(
        "--long", action="store_true",
        help="Write the journal-style Review instead of the default briefing",
    )
    p_rerun.add_argument("--open", action="store_true", help="Open the new draft in your browser")
    p_rerun.add_argument(
        "--queue-ckn", action="store_true",
        help="Add remaining paywalled cited DOIs to the CKN pickup list",
    )
    p_rerun.set_defaults(func=cmd_rerun)

    p_refresh = sub.add_parser(
        "refresh",
        help="Re-run a manifest's queries and list new direct records; rewrites only with --rewrite",
    )
    p_refresh.add_argument("manifest", help="Path to the .json manifest a draft run wrote")
    p_refresh.add_argument("--rewrite", action="store_true",
                           help="Rewrite the briefing with the new records, into <stem>-refresh.json")
    p_refresh.add_argument("--long", action="store_true",
                           help="With --rewrite, write the journal-style Review instead of the briefing")
    p_refresh.add_argument("--open", action="store_true",
                           help="With --rewrite, open the new draft in your browser")
    p_refresh.add_argument("--queue-ckn", action="store_true",
                           help="With --rewrite, add paywalled cited DOIs to the CKN pickup list")
    p_refresh.set_defaults(func=cmd_refresh)

    p_queue = sub.add_parser("queue", help="(Re)build and optionally open the drafts review index")
    p_queue.add_argument("--open", action="store_true", help="Open the queue in your browser")
    p_queue.set_defaults(func=cmd_queue)

    p_demo = sub.add_parser("demo", help="Render a built-in sample (no API calls) to preview the design")
    p_demo.add_argument("-o", "--output", help="Output HTML path (default: demo.html)")
    p_demo.add_argument(
        "--long", action="store_true",
        help="Render the parked Review sample instead of the briefing",
    )
    p_demo.add_argument("--open", action="store_true", help="Open the sample in your browser")
    p_demo.set_defaults(func=cmd_demo)

    p_web = sub.add_parser("web", help="Launch local mobile web app server")
    p_web.add_argument("-p", "--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    p_web.add_argument("--open", action="store_true", help="Open web app in your browser")
    p_web.set_defaults(func=cmd_web)

    return parser


def _make_output_unicode_safe() -> None:
    """Stop non-ASCII output from crashing the run on a legacy Windows console.

    The Windows default console encoding is cp1252, which can't encode the ⚠ in
    the evidence summary or the emoji in the server banner — so `draft` and `web`
    both died with UnicodeEncodeError partway through, after doing all the work.
    Replacing unencodable characters loses a glyph; raising loses the article.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # already fine, or not a real stream (captured in tests)


def main(argv: list[str] | None = None) -> int:
    _make_output_unicode_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


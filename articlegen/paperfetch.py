"""Full text via the `papers` CLI (the separate paperfetch project).

Optional. When `papers` is not installed, `available()` is False and the
pipeline behaves exactly as it did before this module existed.

A draft sends the ordered DOI list to `papers get -` in one process (one JSON
line back per DOI, in input order) so per-process memos such as Semantic
Scholar's 429 skip survive the whole list. An older `papers` that rejects
batch/stdin falls back to one process per DOI. `papers status` runs once per
run, before the first get, and the log names any missing `mailto_set` /
`s2_key_set`.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from typing import Callable

# Resolved once per process. shutil.which is a PATH walk and the pipeline asks
# per paper. None means "not looked yet".
_AVAILABLE: bool | None = None
_ARGV: list[str] = []
# The "not installed" line is worth saying once per process, not per paper.
_WARNED = False
# None = not tried this process. True/False after the first N>1 batch attempt.
_BATCH_OK: bool | None = None
# Filled by fetch_many_via_papers; fetch_via_papers_with_status reads it so
# the pipeline loop does not start a second process per DOI.
_BATCH_CACHE: dict[str, tuple[str, str]] = {}

DEFAULT_TIMEOUT = 120.0
STATUS_TIMEOUT = 15.0
STATUS_FLAGS = ("mailto_set", "s2_key_set")

# papers statuses that mean "no open-access copy", not "OA but empty".
# `queued_ckn` is Unpaywall (and the rest of the ladder) saying the paper is
# paywalled; logging that as a failed OA fetch called landmarks fetch failures
# (#191). `no_oa` is the public `paperfetch-oa` package's equivalent status —
# same meaning, different name because it has no CKN ladder to queue into
# (#173). Other non-ok statuses (`unreadable_pdf`, `retry`, `no_doi`) stay
# fetch failures: those are not a confirmed absence of OA.
NOT_OA_STATUSES = frozenset({"queued_ckn", "no_oa"})

# Filled from the last not-OA `papers` record per DOI (`tried` field).
_TRIED: dict[str, str] = {}

# Tests reset process state by setting paperfetch._AVAILABLE = None,
# paperfetch._ARGV = [], paperfetch._WARNED = False, paperfetch._BATCH_OK = None,
# paperfetch._BATCH_CACHE = {}, paperfetch._TRIED = {}.


def _command() -> list[str]:
    global _ARGV
    if _ARGV:
        return list(_ARGV)
    raw = os.environ.get("ARTICLEGEN_PAPERS_CMD", "").strip()
    if raw:
        _ARGV = shlex.split(raw, posix=False)
    else:
        which = shutil.which("papers")
        _ARGV = [which] if which else []
    return list(_ARGV)


def available(log: Callable[[str], None] | None = None) -> bool:
    """True when the papers CLI executable or command is configured and available."""
    global _AVAILABLE, _WARNED
    if _AVAILABLE is None:
        cmd = _command()
        _AVAILABLE = bool(cmd)
    if not _AVAILABLE and not _WARNED:
        if callable(log):
            log("  papers CLI not installed; full text limited to Europe PMC")
        _WARNED = True
    return _AVAILABLE


# Scholarly keys paperfetch (and this search) share. Process env first;
# on Windows, the User environment if this process started before setx.
_PAPERS_KEYS = (
    "SEMANTIC_SCHOLAR_API_KEY",
    "NCBI_API_KEY",
    "CORE_API_KEY",
    "LIBKEY_API_KEY",
    "PAPERS_MAILTO",
    "OPENALEX_MAILTO",
    "UNPAYWALL_EMAIL",
)


def _user_env(name: str) -> str:
    """Windows User-environment value, or ""."""
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            raw, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(raw or "").strip()


def env_get(name: str) -> str:
    """Process env, then Windows User env. Never writes os.environ."""
    val = (os.environ.get(name) or "").strip()
    return val or _user_env(name)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in _PAPERS_KEYS:
        val = env_get(name)
        if val:
            env[name] = val
        else:
            env.pop(name, None)
    mailto = (
        (os.environ.get("PAPERS_MAILTO") or "").strip()
        or (os.environ.get("OPENALEX_MAILTO") or "").strip()
        or (os.environ.get("UNPAYWALL_EMAIL") or "").strip()
        or env_get("PAPERS_MAILTO")
        or env_get("OPENALEX_MAILTO")
        or env_get("UNPAYWALL_EMAIL")
        or ""
    )
    if mailto:
        env["PAPERS_MAILTO"] = mailto
    else:
        env.pop("PAPERS_MAILTO", None)
    # Private paperfetch writes a CKN miss-list line on `queued_ckn`.
    # A draft must not fill that pickup list. Public paperfetch-oa
    # ignores the variable.
    env["PAPERS_NO_CKN_QUEUE"] = "1"
    return env


def _run(
    argv: list[str],
    *,
    timeout: float,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=_child_env(),
    )


def preflight(log: Callable[[str], None] | None = None) -> dict:
    """Run `papers status` once and log any missing mailto_set / s2_key_set.

    Local only; does not fetch. Returns the status object, or {} if status
    could not be read. Missing keys that are not booleans on the object are
    ignored, so a mocked `get` response is not reported as a config gap.
    """
    if not available(log):
        return {}
    cmd = _command()
    if not cmd:
        return {}
    try:
        proc = _run(cmd + ["status"], timeout=STATUS_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return {}
    try:
        record = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(record, dict):
        return {}
    missing = [flag for flag in STATUS_FLAGS if record.get(flag) is False]
    if missing and callable(log):
        log("  papers: missing " + ", ".join(missing))
    return record


def fetch_via_papers(
    doi: str,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> str:
    """Fetch open-access full text via the `papers` CLI.

    Runs `papers get <doi>` as a subprocess without a shell.
    Never raises into the pipeline; on any error, timeout, non-ok status or missing
    file, returns "" so callers can fall back gracefully.
    """
    return fetch_via_papers_with_status(doi, timeout=timeout, log=log)[0]


def tried_for(doi: str) -> str:
    """Resolvers listed on the last not-OA record for this DOI, or ""."""
    return _TRIED.get(doi, "")


def _text_from_record(
    record: dict,
    doi: str,
    log: Callable[[str], None] | None,
) -> tuple[str, str]:
    status = record.get("status") or ""
    if status in NOT_OA_STATUSES:
        _TRIED[doi] = str(record.get("tried") or "")
    if status != "ok":
        if callable(log):
            log(f"  papers: {status} for {doi}")
        return "", str(status)

    path = record.get("read")
    if not path:
        return "", status

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), status
    except OSError as exc:
        if callable(log):
            log(f"  papers could not read file {path}: {exc}")
        return "", status


def _parse_ndjson(stdout: str | None, *, skip_bad_tail: bool = False) -> list[dict] | None:
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    out: list[dict] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return out if skip_bad_tail and out else None
        if not isinstance(record, dict):
            return out if skip_bad_tail and out else None
        out.append(record)
    return out


def _try_batch(
    dois: list[str],
    timeout: float,
    log: Callable[[str], None] | None,
) -> list[tuple[str, str]] | None:
    """One `papers get -` with the DOI list on stdin. None means try per-DOI."""
    cmd = _command()
    if not cmd:
        return None
    argv = cmd + ["get", "-"]
    stdin = "".join(doi + "\n" for doi in dois)
    try:
        proc = _run(argv, timeout=timeout, stdin=stdin)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "ignore")
        records = _parse_ndjson(stdout, skip_bad_tail=True)
        if records:
            if callable(log):
                log(f"  papers batch timed out after {len(records)}/{len(dois)}")
            pairs = [_text_from_record(record, doi, log)
                     for doi, record in zip(dois, records)]
            pairs.extend([("", "")] * (len(dois) - len(pairs)))
            return pairs
        if callable(log):
            log(f"  papers batch timed out: {exc}")
        # No JSON at all: do not cache five empty misses. Caller falls back
        # to one DOI at a time.
        return None
    except (OSError, ValueError) as exc:
        if callable(log):
            log(f"  papers batch failed: {exc}")
        return None

    records = _parse_ndjson(proc.stdout)
    if records is None:
        return None
    if len(records) == len(dois):
        return [_text_from_record(record, doi, log)
                for doi, record in zip(dois, records)]
    # New papers prints one config_error JSON line for the whole batch
    # (missing mailto, empty stdin) and exits 1. That is not an old CLI.
    if (len(records) == 1
            and records[0].get("status") == "config_error"):
        status = "config_error"
        if callable(log):
            reason = records[0].get("reason") or status
            log(f"  papers: {reason}")
        return [("", status)] * len(dois)
    return None


def fetch_many_via_papers(
    dois: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Fetch many DOIs, one `papers get -` process when the CLI supports it.

    Returns one `(text, status)` pair per input DOI, in input order. A single
    DOI uses the per-DOI path (old `papers get -` treats `-` as the query).
    An older `papers` that returns one JSON object for many stdin lines, or
    a usage error, falls back to one process per DOI for the rest of the
    process.
    """
    global _BATCH_OK, _BATCH_CACHE
    _BATCH_CACHE = {}
    if not dois or not available(log):
        return [("", "")] * len(dois)

    results: list[tuple[str, str]] | None = None
    if _BATCH_OK is not False and len(dois) > 1:
        results = _try_batch(dois, timeout * len(dois), log)
        if results is not None:
            _BATCH_OK = True
        else:
            _BATCH_OK = False
            if callable(log):
                log("  papers: older CLI (no batch/stdin); fetching one DOI at a time")

    if results is None:
        results = [
            fetch_via_papers_with_status(doi, timeout=timeout, log=log)
            for doi in dois
        ]

    _BATCH_CACHE = {doi: pair for doi, pair in zip(dois, results)}
    return results


def fetch_via_papers_with_status(
    doi: str,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Like fetch_via_papers, plus the papers JSON status.

    Returns `(text, status)`. `status` is the CLI's `status` field when JSON
    parsed (`ok`, `queued_ckn`, `no_oa`, `no_doi`, …), or `""` on transport/parse
    failure. `queued_ckn` and `no_oa` both mean no open-access copy, not a
    failed OA fetch.
    """
    if not doi or not available(log):
        return "", ""
    cached = _BATCH_CACHE.get(doi)
    if cached is not None:
        return cached

    try:
        cmd = _command()
        if not cmd:
            return "", ""
        argv = cmd + ["get", doi]

        try:
            proc = _run(argv, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            if callable(log):
                log(f"  papers fetch failed for {doi}: {exc}")
            return "", ""

        try:
            record = json.loads(proc.stdout)
        except (json.JSONDecodeError, Exception) as exc:
            if callable(log):
                sample = (proc.stdout or proc.stderr or "")[:200]
                log(f"  papers returned invalid JSON for {doi} ({exc}): {sample!r}")
            return "", ""

        if not isinstance(record, dict):
            if callable(log):
                log(f"  papers returned non-dict JSON for {doi}")
            return "", ""

        return _text_from_record(record, doi, log)
    except Exception as exc:
        if callable(log):
            log(f"  papers fetch unexpected error for {doi}: {exc}")
        return "", ""


def queue_paywalled(
    papers: list,
    log: Callable[[str], None] | None = None,
) -> int:
    """Add paywalled papers to the CKN list via `papers queue`. Returns how many.

    Uses `papers queue`, not `papers get`. `PAPERS_NO_CKN_QUEUE` on get is
    unchanged: a draft still cannot fill the list unless the caller asked.
    """
    if not papers:
        return 0
    if not available(log):
        if callable(log):
            log("  papers CLI not installed; nothing queued")
        return 0
    cmd = _command()
    queued = 0
    for paper in papers:
        doi = (getattr(paper, "doi", "") or "").removeprefix("https://doi.org/").strip()
        if not doi:
            continue
        argv = cmd + ["queue", doi]
        title = getattr(paper, "title", "") or ""
        if title:
            argv += ["--title", title]
        venue = getattr(paper, "venue", "") or ""
        if venue:
            argv += ["--journal", venue]
        year = getattr(paper, "year", None)
        if year:
            argv += ["--year", str(year)]
        tried = getattr(paper, "oa_tried", "") or "unpaywall"
        argv += ["--tried", tried]
        try:
            proc = _run(argv, timeout=STATUS_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            if callable(log):
                log(f"  papers queue failed for {doi}: {exc}")
            continue
        try:
            row = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError):
            if callable(log):
                log(f"  papers queue returned invalid JSON for {doi}")
            continue
        status = (row or {}).get("status") or ""
        if callable(log):
            log(f"  queued {doi} ({status})")
        queued += 1
    return queued

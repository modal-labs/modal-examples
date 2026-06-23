from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import re
import ssl
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import modal
except ImportError:  # pragma: no cover - exercised when Modal is not installed.
    modal = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://modal.com"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_RETRIES = 1
DEFAULT_WORKERS = 32
DEFAULT_MAX_FILE_BYTES = 2_000_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)
SEC_USER_AGENT = "examples@modal.com"

_BARE_URL_RE = re.compile(r"https?://[^\s<>'\"`]+")
_MARKDOWN_LINK_START_RE = re.compile(r"!?\[[^\]]*]\(")
_GITHUB_REF_RE = re.compile(r"^https://github\.com/([^/]+)/([^/@]+?)(?:\.git)?@(.+)$")
_GITHUB_RELEASE_DOWNLOAD_RE = re.compile(
    r"^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/?$"
)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\""
_SKIP_SCHEMES = {"mailto", "tel", "javascript", "data"}
_PLACEHOLDER_TOKENS = (
    "app_id",
    "my-tailscale-machine",
    "your-",
    "your_",
)
_SITE_ABSOLUTE_PREFIXES = (
    "/apps",
    "/docs",
    "/pricing",
    "/secrets",
    "/settings",
)
_DEFAULT_EXCLUDES = (
    ".git/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "__pycache__/**",
    "venv/**",
)


@dataclass(frozen=True)
class LinkRecord:
    url: str
    target: str
    kind: str
    file: str
    line: int
    source: str
    reason: str = ""


@dataclass(frozen=True)
class CheckResult:
    target: str
    status: str
    reason: str
    http_status: int | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class BaselineAnalysis:
    known_targets: set[str]
    new_targets: dict[str, list[LinkRecord]]
    new_locations: dict[str, list[LinkRecord]]
    fixed_targets: set[str]
    unchecked_targets: set[str]

    @property
    def has_failures(self) -> bool:
        return bool(self.new_targets or self.new_locations)


@dataclass(frozen=True)
class HttpCheckResponse:
    status: int
    reason: str
    headers: Mapping[str, str]


def _clean_link_target(url: str) -> str:
    url = url.strip().strip("<>")
    while url and url[-1] in _TRAILING_PUNCTUATION:
        if url[-1] == ")" and url.count("(") >= url.count(")"):
            break
        url = url[:-1]
    return url


def _target_without_fragment(url: str) -> str:
    url, _fragment = urldefrag(url)
    return url


def _safe_urlsplit(url: str):
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _external_target(url: str) -> str:
    if match := _GITHUB_REF_RE.match(url):
        owner, repo, ref = match.groups()
        return f"https://github.com/{owner}/{repo}/tree/{ref}"
    if match := _GITHUB_RELEASE_DOWNLOAD_RE.match(url):
        owner, repo, tag = match.groups()
        return f"https://github.com/{owner}/{repo}/releases/tag/{tag}"

    parts = urlsplit(_target_without_fragment(url))
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path or "/",
            parts.query,
            "",
        )
    )


def _is_private_hostname(hostname: str) -> bool:
    hostname = hostname.lower()
    if hostname in {"localhost", "0.0.0.0"}:
        return True
    if hostname.endswith(".local"):
        return True
    return hostname.startswith("127.")


def _skip_external_reason(url: str) -> str:
    lower_url = url.lower()
    if "..." in url or "…" in url:
        return "placeholder URL"
    if any(token in lower_url for token in _PLACEHOLDER_TOKENS):
        return "placeholder URL"
    if "{" in url or "}" in url:
        return "templated URL"
    parts = _safe_urlsplit(url)
    if parts is None:
        return "invalid URL"
    if not parts.netloc:
        return "missing hostname"
    hostname = parts.hostname or ""
    if _is_private_hostname(hostname):
        return "local-only URL"
    if hostname == "example.com" or hostname.endswith(".example.com"):
        return "example placeholder URL"
    if hostname.endswith((".modal.host", ".modal.run")):
        return "Modal app URL"
    return ""


def _classify_link(
    raw_url: str,
    file_path: Path,
    line_number: int,
    source: str,
    *,
    root: Path,
    base_url: str,
) -> LinkRecord:
    url = _clean_link_target(raw_url)
    file = file_path.relative_to(root).as_posix()

    if url.startswith("//"):
        url = f"https:{url}"

    def record(target: str, kind: str, reason: str = "") -> LinkRecord:
        return LinkRecord(url, target, kind, file, line_number, source, reason)

    if not url:
        return record("", "skip", "empty link")

    parts = _safe_urlsplit(url)
    if parts is None:
        return record(url, "invalid", "invalid URL")

    if parts.scheme in {"http", "https"}:
        if not parts.netloc:
            return record(url, "invalid", "missing hostname")
        target = _external_target(url)
        if reason := _skip_external_reason(target):
            return record(target, "skip", reason)
        return record(target, "external")

    if parts.scheme:
        if parts.scheme in _SKIP_SCHEMES:
            return record(url, "skip", f"{parts.scheme}: link")
        return record(url, "invalid", f"unsupported URL scheme: {parts.scheme}")

    if url.startswith("#"):
        return record(url, "skip", "same-page anchor")

    if "{" in url or "}" in url:
        return record(url, "skip", "templated URL")

    if url.startswith(_SITE_ABSOLUTE_PREFIXES):
        target = _external_target(urljoin(base_url, url))
        if reason := _skip_external_reason(target):
            return record(target, "skip", reason)
        return record(target, "external")

    target_path = Path(unquote(parts.path))
    if target_path.is_absolute():
        resolved = (root / target_path.relative_to("/")).resolve()
    else:
        resolved = (file_path.parent / target_path).resolve()
    return record(str(resolved), "local")


def extract_links_from_file(
    file_path: Path, *, root: Path, base_url: str
) -> Iterator[LinkRecord]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not _line_allows_link_collection(line):
            continue

        markdown_spans: list[tuple[int, int]] = []

        for target, span in _iter_markdown_targets(line):
            markdown_spans.append(span)
            yield _classify_link(
                target,
                file_path,
                line_number,
                "markdown",
                root=root,
                base_url=base_url,
            )

        for match in _BARE_URL_RE.finditer(line):
            if any(
                match.start() >= start and match.start() < end
                for start, end in markdown_spans
            ):
                continue
            yield _classify_link(
                match.group(0),
                file_path,
                line_number,
                "bare",
                root=root,
                base_url=base_url,
            )


def _line_allows_link_collection(line: str) -> bool:
    return line.lstrip().startswith("#")


def _iter_markdown_targets(line: str) -> Iterator[tuple[str, tuple[int, int]]]:
    for match in _MARKDOWN_LINK_START_RE.finditer(line):
        target_start = match.end()
        index = target_start
        depth = 0

        while index < len(line):
            char = line[index]
            if char == "\\":
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    target = line[target_start:index].split(None, 1)[0]
                    yield target, (match.start(), index + 1)
                    break
                depth -= 1
            index += 1


def _tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if p.is_file()]

    return [root / name.decode() for name in result.stdout.split(b"\0") if name]


def _is_excluded(path: Path, root: Path, patterns: Iterable[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def collect_links(
    *,
    root: Path = REPO_ROOT,
    base_url: str = DEFAULT_BASE_URL,
    excludes: Iterable[str] = (),
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[LinkRecord]:
    root = root.resolve()
    patterns = tuple(_DEFAULT_EXCLUDES) + tuple(excludes)
    records: list[LinkRecord] = []
    seen: set[tuple[str, str, int, str]] = set()

    for path in _tracked_files(root):
        if _is_excluded(path, root, patterns):
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue

        for record in extract_links_from_file(path, root=root, base_url=base_url):
            key = (record.url, record.file, record.line, record.source)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)

    return records


def _load_records(path: Path) -> list[LinkRecord]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("links", [])
    return [LinkRecord(**item) for item in data]


def _write_records(path: Path, records: list[LinkRecord]) -> None:
    payload = {
        "links": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _http_request(url: str, *, method: str, timeout: int) -> HttpCheckResponse:
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "User-Agent": _user_agent_for_url(url),
    }
    request = Request(url, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        if method == "GET":
            response.read(1024)
        return HttpCheckResponse(
            response.status,
            response.reason,
            dict(response.headers.items()),
        )


def _user_agent_for_url(url: str) -> str:
    parts = _safe_urlsplit(url)
    hostname = parts.hostname if parts else ""
    if hostname == "sec.gov" or hostname.endswith(".sec.gov"):
        return SEC_USER_AGENT
    return USER_AGENT


def _http_error_response(exc: HTTPError) -> HttpCheckResponse:
    return HttpCheckResponse(exc.code, exc.reason, dict(exc.headers.items()))


def _attempt_http_check(url: str, *, timeout: int) -> HttpCheckResponse:
    try:
        response = _http_request(url, method="HEAD", timeout=timeout)
        if response.status < 400:
            return response
    except HTTPError:
        # Some hosts reject HEAD while accepting GET, so we fall through to GET
        # before deciding the link is broken. Network/TLS errors propagate.
        pass

    try:
        return _http_request(url, method="GET", timeout=timeout)
    except HTTPError as exc:
        return _http_error_response(exc)


def _is_anti_bot_challenge(headers: Mapping[str, str]) -> bool:
    normalized = {key.lower(): value.lower() for key, value in headers.items()}
    return normalized.get("cf-mitigated") == "challenge"


def check_external_target(
    target: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> CheckResult:
    started = time.monotonic()
    last_error = ""

    for attempt in range(retries + 1):
        try:
            response = _attempt_http_check(target, timeout=timeout)
            status = response.status
            reason = response.reason
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if 200 <= status < 400:
                return CheckResult(target, "ok", reason, status, elapsed_ms)
            if _is_anti_bot_challenge(response.headers):
                return CheckResult(
                    target,
                    "skipped",
                    "blocked by anti-bot challenge",
                    status,
                    elapsed_ms,
                )
            if status in {405, 429}:
                skip_reason = (
                    "method not allowed by target"
                    if status == 405
                    else "rate-limited by target"
                )
                return CheckResult(
                    target,
                    "skipped",
                    skip_reason,
                    status,
                    elapsed_ms,
                )
            if status >= 500 and attempt < retries:
                last_error = f"HTTP {status} {reason}"
                time.sleep(1 + attempt)
                continue
            return CheckResult(
                target, "broken", f"HTTP {status} {reason}", status, elapsed_ms
            )
        except TimeoutError as exc:
            last_error = f"timeout: {exc}"
        except URLError as exc:
            last_error = f"network error: {exc.reason}"
        except ssl.SSLError as exc:
            last_error = f"TLS error: {exc}"
        except OSError as exc:
            last_error = f"network error: {exc}"

        if attempt < retries:
            time.sleep(1 + attempt)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return CheckResult(
        target, "broken", last_error or "request failed", None, elapsed_ms
    )


def check_local_record(record: LinkRecord) -> CheckResult:
    if record.kind == "skip":
        return CheckResult(record.target or record.url, "skipped", record.reason)
    if record.kind == "invalid":
        return CheckResult(record.target or record.url, "broken", record.reason)
    if record.kind == "local":
        if Path(record.target).exists():
            return CheckResult(record.target, "ok", "local file exists")
        return CheckResult(record.target, "broken", "local file does not exist")
    raise ValueError(f"Cannot check {record.kind} link locally without routing")


def _unique_external_targets(records: Iterable[LinkRecord]) -> list[str]:
    seen: set[str] = set()
    targets: list[str] = []
    for record in records:
        if record.kind != "external":
            continue
        if record.target in seen:
            continue
        seen.add(record.target)
        targets.append(record.target)
    return targets


def _check_records(
    records: list[LinkRecord],
    check_targets: Callable[[list[str]], list[CheckResult]],
) -> list[CheckResult]:
    results: list[CheckResult] = [
        check_local_record(record) for record in records if record.kind != "external"
    ]
    results.extend(check_targets(_unique_external_targets(records)))
    return results


def check_records_locally(
    records: list[LinkRecord],
    *,
    timeout: int,
    retries: int,
    workers: int,
) -> list[CheckResult]:
    def check_targets(targets: list[str]) -> list[CheckResult]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    check_external_target, target, timeout=timeout, retries=retries
                )
                for target in targets
            ]
            return [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

    return _check_records(records, check_targets)


def check_records_with_modal(
    records: list[LinkRecord],
    *,
    timeout: int,
    retries: int,
) -> list[CheckResult]:
    if check_url_on_modal is None:
        raise RuntimeError("Modal is not installed; run with --mode local instead")

    def check_targets(targets: list[str]) -> list[CheckResult]:
        remote_results = check_url_on_modal.map(
            targets,
            [timeout] * len(targets),
            [retries] * len(targets),
            return_exceptions=True,
        )
        return [
            CheckResult(targets[index], "broken", repr(remote_result))
            if isinstance(remote_result, Exception)
            else CheckResult(**remote_result)
            for index, remote_result in enumerate(remote_results)
        ]

    return _check_records(records, check_targets)


def _group_occurrences(records: Iterable[LinkRecord]) -> dict[str, list[LinkRecord]]:
    grouped: dict[str, list[LinkRecord]] = {}
    for record in records:
        grouped.setdefault(record.target or record.url, []).append(record)
    return grouped


def _print_occurrences(records: list[LinkRecord], *, max_occurrences: int) -> None:
    for record in records[:max_occurrences]:
        print(f"  - {record.file}:{record.line} [{record.source}] {record.url}")
    if len(records) > max_occurrences:
        print(f"  - ... {len(records) - max_occurrences} more occurrence(s)")


def print_collection_summary(records: list[LinkRecord]) -> None:
    counts = Counter(record.kind for record in records)
    unique_external = len(_unique_external_targets(records))
    print(
        "Collected "
        f"{len(records)} links "
        f"({unique_external} unique external; "
        f"{counts.get('local', 0)} local; "
        f"{counts.get('skip', 0)} skipped; "
        f"{counts.get('invalid', 0)} invalid)."
    )


def print_check_summary(
    records: list[LinkRecord],
    results: list[CheckResult],
    *,
    max_occurrences: int,
) -> None:
    grouped_records = _group_occurrences(records)
    broken = [result for result in results if result.status == "broken"]
    skipped = [result for result in results if result.status == "skipped"]
    ok = [result for result in results if result.status == "ok"]

    print(
        f"Checked {len(results)} targets: {len(ok)} ok, {len(skipped)} skipped, {len(broken)} broken."
    )
    if not broken:
        return

    print("\nBroken links:")
    for result in sorted(broken, key=lambda item: item.target):
        status = (
            f"HTTP {result.http_status}"
            if result.http_status is not None
            else result.reason
        )
        print(f"- {result.target} ({status})")
        occurrences = grouped_records.get(result.target, [])
        _print_occurrences(occurrences, max_occurrences=max_occurrences)


def _write_results(path: Path, results: list[CheckResult]) -> None:
    payload = {
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_baseline(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text())
    baseline: dict[str, set[str]] = {}
    for item in data.get("links", []):
        baseline[item["target"]] = set(item.get("files", []))
    return baseline


def _broken_results(results: Iterable[CheckResult]) -> list[CheckResult]:
    return [result for result in results if result.status == "broken"]


def _broken_baseline(
    records: list[LinkRecord], results: list[CheckResult]
) -> dict[str, set[str]]:
    grouped_records = _group_occurrences(records)
    baseline: dict[str, set[str]] = {}
    for result in _broken_results(results):
        files = {record.file for record in grouped_records.get(result.target, [])}
        baseline[result.target] = files
    return baseline


def _write_baseline(
    path: Path, records: list[LinkRecord], results: list[CheckResult]
) -> None:
    baseline = _broken_baseline(records, results)
    payload = {
        "description": (
            "Known broken prose links. CI fails only for new broken targets "
            "or known broken targets appearing in new files."
        ),
        "links": [
            {
                "target": target,
                "files": sorted(files),
            }
            for target, files in sorted(baseline.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def analyze_baseline(
    records: list[LinkRecord],
    results: list[CheckResult],
    baseline: dict[str, set[str]],
) -> BaselineAnalysis:
    grouped_records = _group_occurrences(records)
    broken_targets = {result.target for result in _broken_results(results)}
    skipped_targets = {
        result.target for result in results if result.status == "skipped"
    }
    known_targets: set[str] = set()
    new_targets: dict[str, list[LinkRecord]] = {}
    new_locations: dict[str, list[LinkRecord]] = {}

    for target in sorted(broken_targets):
        records_for_target = grouped_records.get(target, [])
        current_files = {record.file for record in records_for_target}
        baseline_files = baseline.get(target)
        if baseline_files is None:
            new_targets[target] = records_for_target
            continue

        known_targets.add(target)
        new_files = current_files - baseline_files
        if new_files:
            new_locations[target] = [
                record for record in records_for_target if record.file in new_files
            ]

    unchecked_targets = set(baseline) & skipped_targets
    fixed_targets = set(baseline) - broken_targets - unchecked_targets
    return BaselineAnalysis(
        known_targets=known_targets,
        new_targets=new_targets,
        new_locations=new_locations,
        fixed_targets=fixed_targets,
        unchecked_targets=unchecked_targets,
    )


def print_baseline_summary(analysis: BaselineAnalysis, *, max_occurrences: int) -> None:
    print(
        "\nBaseline comparison: "
        f"{len(analysis.known_targets)} known broken, "
        f"{len(analysis.new_targets)} new broken target(s), "
        f"{len(analysis.new_locations)} known target(s) in new file(s), "
        f"{len(analysis.fixed_targets)} baseline target(s) no longer broken, "
        f"{len(analysis.unchecked_targets)} baseline target(s) skipped."
    )

    def print_target_group(header: str, mapping: dict[str, list[LinkRecord]]) -> None:
        if not mapping:
            return
        print(f"\n{header}")
        for target, records in sorted(mapping.items()):
            print(f"- {target}")
            _print_occurrences(records, max_occurrences=max_occurrences)

    print_target_group("New broken targets:", analysis.new_targets)
    print_target_group(
        "Known broken targets added to new files:", analysis.new_locations
    )

    if analysis.fixed_targets:
        print("\nBaseline targets no longer broken:")
        for target in sorted(analysis.fixed_targets):
            print(f"- {target}")

    if analysis.unchecked_targets:
        print("\nBaseline targets skipped this run:")
        for target in sorted(analysis.unchecked_targets):
            print(f"- {target}")


def _parse_args(argv: list[str] | None, *, default_mode: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and check links in modal-examples."
    )
    parser.add_argument(
        "command",
        choices=["collect", "check"],
        default="check",
        nargs="?",
    )
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for site-absolute links like /docs/guide/images.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional git-style path pattern to exclude. May be repeated.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Read collected links from this JSON file instead of collecting again.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write collected links or check results to this JSON file.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON file of known broken links. Only new broken links fail.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite --baseline with the current broken link set.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "modal"],
        default=default_mode,
        help="Where to run external HTTP checks.",
    )
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-occurrences", type=int, default=5)
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None, *, default_mode: str = "local") -> int:
    args = _parse_args(argv, default_mode=default_mode)
    root = Path(args.root).resolve()

    if args.input:
        records = _load_records(args.input)
    else:
        records = collect_links(
            root=root,
            base_url=args.base_url,
            excludes=args.exclude,
            max_file_bytes=args.max_file_bytes,
        )

    print_collection_summary(records)

    if args.command == "collect":
        if args.output:
            _write_records(args.output, records)
        return 0

    if args.mode == "modal":
        results = check_records_with_modal(
            records, timeout=args.timeout, retries=args.retries
        )
    else:
        results = check_records_locally(
            records,
            timeout=args.timeout,
            retries=args.retries,
            workers=args.workers,
        )

    if args.output:
        _write_results(args.output, results)
    print_check_summary(records, results, max_occurrences=args.max_occurrences)

    if args.update_baseline:
        if not args.baseline:
            raise SystemExit("--update-baseline requires --baseline")
        _write_baseline(args.baseline, records, results)
        print(f"\nUpdated baseline: {args.baseline}")
        return 0

    if args.baseline:
        analysis = analyze_baseline(records, results, _load_baseline(args.baseline))
        print_baseline_summary(analysis, max_occurrences=args.max_occurrences)
        return 1 if analysis.has_failures else 0

    return 1 if _broken_results(results) else 0


if modal is not None:
    app = modal.App("example-link-checker")
    image = modal.Image.debian_slim(python_version="3.11")

    @app.function(image=image, max_containers=100, retries=1, timeout=60)
    def check_url_on_modal(target: str, timeout: int, retries: int) -> dict[str, Any]:
        return asdict(check_external_target(target, timeout=timeout, retries=retries))

    @app.local_entrypoint()
    def main(*arglist: str) -> None:
        exit_code = run_cli(list(arglist), default_mode="modal")
        if exit_code:
            raise SystemExit(exit_code)

else:
    check_url_on_modal = None


if __name__ == "__main__":
    raise SystemExit(run_cli(default_mode="local"))

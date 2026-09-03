"""Resumable, checksum-verified downloads.

Two behaviours here are deliberate and worth reading before changing.

**Mirrors are never used silently.** A failed primary URL raises with manual
download instructions and the expected destination path. Falling through to a
mirror of unknown provenance would mean a result whose data source is not
recorded anywhere (ADR-0011). ``allow_mirror=True`` opts in explicitly, and the
URL actually used is recorded.

**A server that advertises ``Accept-Ranges`` may still ignore ``Range``.**
GitHub Pages does exactly this: it returns ``200 OK`` with the full body rather
than ``206 Partial Content``. Appending that body to an existing partial file
produces a corrupt, double-prefixed file that then fails its checksum with a
thoroughly confusing message. So a resume is accepted only when the response is
``206`` *and* its ``Content-Range`` start matches what we already have.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

__all__ = ["DownloadError", "DownloadResult", "fetch", "sha256_file"]

_CHUNK = 1 << 20  # 1 MiB
_TIMEOUT = (15, 120)  # (connect, read) seconds


def _progress_disabled(quiet: bool) -> bool:
    """Disable tqdm when quiet, or when stderr is not a terminal.

    A redirected run otherwise writes one progress line per update; a 500 MB
    download produced a 39 KB log of nothing but bar redraws.
    """
    return quiet or not sys.stderr.isatty()


class DownloadError(RuntimeError):
    """A dataset artifact could not be obtained. Always actionable."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    url: str
    sha256: str
    n_bytes: int
    resumed: bool
    cached: bool  # the file was already present and verified; nothing transferred


def sha256_file(path: Path, chunk: int = _CHUNK) -> str:
    """Streaming SHA-256 of a file (these are up to 500 MB; do not slurp)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _manual_instructions(urls: tuple[str, ...], dest: Path, reason: str) -> str:
    lines = [
        f"Could not download {dest.name}: {reason}",
        "",
        "Download it manually and place it at exactly:",
        f"  {dest}",
        "",
        "Known sources (primary first):",
    ]
    lines += [f"  {i + 1}. {u}" for i, u in enumerate(urls)]
    lines += [
        "",
        "Then re-run the same command; the file will be checksummed and recorded.",
    ]
    return "\n".join(lines)


def _attempt(url: str, dest: Path, quiet: bool) -> tuple[int, bool]:
    """Download *url* into *dest*, resuming from ``<dest>.part`` when possible.

    Returns ``(n_bytes, resumed)``.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=_TIMEOUT, headers=headers) as r:
        r.raise_for_status()

        resumed = False
        if have:
            content_range = r.headers.get("Content-Range", "")
            starts_where_we_stopped = content_range.replace("bytes ", "").startswith(f"{have}-")
            if r.status_code == 206 and starts_where_we_stopped:
                resumed = True
            else:
                # The server ignored Range (GitHub Pages does this) or answered
                # from a different offset. Appending would corrupt the file.
                part.unlink()
                have = 0

        declared = r.headers.get("Content-Length")
        total = (int(declared) + have) if declared is not None else None

        mode = "ab" if resumed else "wb"
        with (
            part.open(mode) as fh,
            tqdm(
                total=total,
                initial=have,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                disable=_progress_disabled(quiet),
                leave=False,
            ) as bar,
        ):
            for block in r.iter_content(chunk_size=_CHUNK):
                if block:
                    fh.write(block)
                    bar.update(len(block))

    n_bytes = part.stat().st_size
    shutil.move(str(part), str(dest))
    return n_bytes, resumed


def fetch(
    urls: tuple[str, ...] | list[str],
    dest: Path,
    expected_sha256: str | None = None,
    allow_mirror: bool = False,
    quiet: bool = False,
) -> DownloadResult:
    """Obtain *dest*, verifying against *expected_sha256* when one is recorded.

    A file that is already present is checksummed rather than re-downloaded. If
    it fails verification it is **not** silently replaced: that would erase the
    evidence that something changed.
    """
    urls = tuple(urls)
    if not urls:
        raise DownloadError(f"no source URLs registered for {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        digest = sha256_file(dest)
        if expected_sha256 and digest != expected_sha256:
            raise DownloadError(
                f"{dest} exists but its checksum does not match the recorded value.\n"
                f"  recorded: {expected_sha256}\n"
                f"  on disk:  {digest}\n"
                "Delete the file deliberately if you intend to re-download it. It is "
                "not being overwritten automatically, because a mismatch is evidence."
            )
        return DownloadResult(
            path=dest,
            url=(dest.as_uri()),
            sha256=digest,
            n_bytes=dest.stat().st_size,
            resumed=False,
            cached=True,
        )

    candidates = urls if allow_mirror else urls[:1]
    failures: list[str] = []
    for url in candidates:
        if url.startswith("ftp://"):
            failures.append(f"{url}: ftp:// is not fetched automatically")
            continue
        try:
            n_bytes, resumed = _attempt(url, dest, quiet)
        except (requests.RequestException, OSError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue

        digest = sha256_file(dest)
        if expected_sha256 and digest != expected_sha256:
            raise DownloadError(
                f"{dest.name} downloaded from {url} but failed checksum verification.\n"
                f"  recorded: {expected_sha256}\n"
                f"  received: {digest}\n"
                "The upstream file changed or the transfer was corrupted."
            )
        return DownloadResult(
            path=dest, url=url, sha256=digest, n_bytes=n_bytes, resumed=resumed, cached=False
        )

    reason = "; ".join(failures) if failures else "no usable source"
    if not allow_mirror and len(urls) > 1:
        reason += (
            f" (only the primary URL was tried; {len(urls) - 1} mirror(s) are registered "
            "but are not used silently -- pass --allow-mirror to opt in)"
        )
    raise DownloadError(_manual_instructions(urls, dest, reason))

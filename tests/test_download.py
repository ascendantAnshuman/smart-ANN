"""Phase 1: download resume, checksum enforcement, and the no-silent-mirror rule.

The resume tests matter because GitHub Pages advertises ``Accept-Ranges`` and
then ignores ``Range``, returning 200 with the whole body. Appending that to a
partial file yields a corrupt, double-prefixed artifact whose checksum failure
points nowhere useful.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import requests

from smartann.data.download import DownloadError, fetch, sha256_file

PAYLOAD = b"0123456789" * 500  # 5000 bytes
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    """Minimal stand-in for ``requests.Response`` in streaming mode."""

    def __init__(self, body: bytes, status_code: int = 200, headers: dict[str, str] | None = None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(len(body))}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size: int = 1) -> Any:
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _install(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def fake_get(url: str, **kw: Any) -> FakeResponse:
        seen.append(dict(kw.get("headers") or {}))
        return handler(url, kw)

    monkeypatch.setattr(requests, "get", fake_get)
    return seen


def test_plain_download_and_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, lambda u, kw: FakeResponse(PAYLOAD))
    dest = tmp_path / "f.bin"
    res = fetch(("https://example/f.bin",), dest, quiet=True)

    assert dest.read_bytes() == PAYLOAD
    assert res.sha256 == DIGEST
    assert res.n_bytes == len(PAYLOAD)
    assert not res.cached


def test_resume_appends_only_on_a_valid_206(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(PAYLOAD[:2000])

    def handler(url: str, kw: dict[str, Any]) -> FakeResponse:
        return FakeResponse(
            PAYLOAD[2000:],
            status_code=206,
            headers={
                "Content-Length": str(len(PAYLOAD) - 2000),
                "Content-Range": f"bytes 2000-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
            },
        )

    headers = _install(monkeypatch, handler)
    res = fetch(("https://example/f.bin",), dest, quiet=True)

    assert headers[0]["Range"] == "bytes=2000-"
    assert res.resumed is True
    assert dest.read_bytes() == PAYLOAD
    assert res.sha256 == DIGEST


def test_server_ignoring_range_restarts_instead_of_corrupting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub Pages behaviour: Range requested, 200 with the full body returned."""
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(PAYLOAD[:2000])

    _install(monkeypatch, lambda u, kw: FakeResponse(PAYLOAD, status_code=200))
    res = fetch(("https://example/f.bin",), dest, quiet=True)

    assert res.resumed is False
    assert dest.read_bytes() == PAYLOAD, "the partial prefix must not have been kept"
    assert res.sha256 == DIGEST


def test_206_from_the_wrong_offset_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "f.bin"
    dest.with_suffix(".bin.part").write_bytes(PAYLOAD[:2000])

    def handler(url: str, kw: dict[str, Any]) -> FakeResponse:
        # Claims to resume from 1000 although we asked for 2000.
        return FakeResponse(
            PAYLOAD,
            status_code=206,
            headers={"Content-Range": f"bytes 1000-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"},
        )

    _install(monkeypatch, handler)
    res = fetch(("https://example/f.bin",), dest, quiet=True)
    assert res.resumed is False
    assert dest.read_bytes() == PAYLOAD


def test_checksum_mismatch_after_download_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, lambda u, kw: FakeResponse(b"different content"))
    with pytest.raises(DownloadError, match="failed checksum"):
        fetch(("https://example/f.bin",), tmp_path / "f.bin", expected_sha256=DIGEST, quiet=True)


def test_existing_file_is_verified_not_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD)
    seen = _install(monkeypatch, lambda u, kw: pytest.fail("should not have downloaded"))

    res = fetch(("https://example/f.bin",), dest, expected_sha256=DIGEST, quiet=True)
    assert res.cached is True
    assert seen == []


def test_existing_file_with_bad_checksum_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatch is evidence that something changed; keep it."""
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"stale")
    _install(monkeypatch, lambda u, kw: FakeResponse(PAYLOAD))

    with pytest.raises(DownloadError, match="does not match the recorded value"):
        fetch(("https://example/f.bin",), dest, expected_sha256=DIGEST, quiet=True)
    assert dest.read_bytes() == b"stale"


def test_mirrors_are_not_used_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tried: list[str] = []

    def handler(url: str, kw: dict[str, Any]) -> FakeResponse:
        tried.append(url)
        return FakeResponse(b"", status_code=404)

    _install(monkeypatch, handler)
    urls = ("https://primary/f.bin", "https://mirror/f.bin")

    with pytest.raises(DownloadError) as exc:
        fetch(urls, tmp_path / "f.bin", quiet=True)

    assert tried == ["https://primary/f.bin"], "the mirror must not be tried by default"
    assert "--allow-mirror" in str(exc.value)
    assert "Download it manually" in str(exc.value)
    assert str(tmp_path / "f.bin") in str(exc.value)


def test_allow_mirror_opts_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, kw: dict[str, Any]) -> FakeResponse:
        if "primary" in url:
            return FakeResponse(b"", status_code=500)
        return FakeResponse(PAYLOAD)

    _install(monkeypatch, handler)
    res = fetch(
        ("https://primary/f.bin", "https://mirror/f.bin"),
        tmp_path / "f.bin",
        allow_mirror=True,
        quiet=True,
    )
    assert res.url == "https://mirror/f.bin"


def test_sha256_file_streams_correctly(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(PAYLOAD)
    assert sha256_file(p) == DIGEST

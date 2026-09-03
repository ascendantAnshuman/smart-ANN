"""Shared test fixtures.

The autouse fixture below is a safety rail, added after a Phase 0 CLI test --
which asserted that ``smartann data prepare squad`` raised
``NotImplementedError`` -- silently turned into a live-fire command once Phase 1
implemented that subcommand. It downloaded 35 MB and began writing a chunk
database into the developer's real data root.

No test may touch the real data or results roots. Anything that genuinely needs
network or bulk data is marked ``slow`` and opts in explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_data_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SMARTANN_DATA_ROOT / SMARTANN_RESULTS_ROOT at a throwaway directory."""
    data_root = tmp_path / "data-root"
    results_root = tmp_path / "results-root"
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SMARTANN_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SMARTANN_RESULTS_ROOT", str(results_root))
    return data_root


@pytest.fixture(autouse=True)
def no_accidental_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Fail loudly if an unmarked test reaches the network.

    Tests that mock ``requests.get`` themselves are unaffected, because their
    monkeypatch is applied after this one.
    """
    if request.node.get_closest_marker("slow"):
        return

    import requests

    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "network access from an unmarked test. Mock requests.get, or mark the "
            "test with @pytest.mark.slow if it genuinely needs the network."
        )

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "head", blocked)

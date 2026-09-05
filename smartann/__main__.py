"""Entry point for ``python -m smartann``.

The package and the module path both work, deliberately: ``python -m
smartann.cli`` is what the runbook documented for thirty-nine commits, and
``python -m smartann`` is what a reader would try first. Neither may be a
silent no-op again (ADR-0042).
"""

from __future__ import annotations

from smartann.cli import app

if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    app()

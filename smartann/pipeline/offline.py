"""12.5 -- prove the query path works with the network taken away.

`NetworkWitness` in `querytrace.py` *counts* what a query sends. This *blocks*
it. The two are deliberately separate instruments: a witness that refused
connections would change the behaviour it was built to observe, and a blocker
that merely counted would prove nothing.

**Why blocking is a stronger claim than counting.** A query that sends zero
bytes today might be one dependency upgrade away from checking for a model
update on first use. Counting notices that after the fact, on whichever run
someone happens to inspect. Blocking turns it into a failure, immediately, in
the test suite. The claim "this works offline" is a claim about what the code
*cannot* do, and only a blocker can test that.

**What is blocked.** Outbound connections, datagram sends, and name resolution.
Name resolution is included because a DNS lookup is itself traffic leaving the
device: a pipeline that resolves a hostname and then fails has still told a
resolver what it was looking for.

**Loopback is blocked too.** It would be convenient to allow 127.0.0.1, and
wrong: the point is that nothing contacts anything, and a local server is still
a service the pipeline depends on being up.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NetworkBlockedError", "OfflineReport", "block_network"]


class NetworkBlockedError(RuntimeError):
    """Raised when code under `block_network` tries to reach the network."""


@dataclass
class OfflineReport:
    """What was attempted while the network was unavailable."""

    attempts: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.attempts

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": list(self.attempts),
            "clean": self.clean,
            "how_measured": (
                "outbound connects, datagram sends and name resolution were patched to "
                "raise. An empty list means nothing was attempted, not that nothing "
                "succeeded."
            ),
        }


@contextmanager
def block_network() -> Iterator[OfflineReport]:
    """Make every outbound network operation raise, for the duration of the block.

    The report is yielded immediately and filled in as attempts occur, so a
    caller that catches `NetworkBlockedError` can still see what was tried.
    """
    report = OfflineReport()
    saved: dict[str, Any] = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "sendto": socket.socket.sendto,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
    }

    def refuse(what: str) -> Any:
        def blocked(*args: Any, **_kwargs: Any) -> Any:
            target = args[1] if len(args) > 1 else (args[0] if args else "?")
            report.attempts.append(f"{what} -> {target!r}")
            raise NetworkBlockedError(
                f"{what} to {target!r} was blocked. This code path is required to work "
                f"with no network at all, and it just tried to use one."
            )

        return blocked

    def refuse_module(what: str) -> Any:
        def blocked(*args: Any, **_kwargs: Any) -> Any:
            target = args[0] if args else "?"
            report.attempts.append(f"{what} -> {target!r}")
            raise NetworkBlockedError(
                f"{what} for {target!r} was blocked. This code path is required to work "
                f"with no network at all, and it just tried to use one."
            )

        return blocked

    socket.socket.connect = refuse("socket.connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse("socket.connect_ex")  # type: ignore[method-assign]
    socket.socket.sendto = refuse("socket.sendto")  # type: ignore[method-assign]
    socket.create_connection = refuse_module("socket.create_connection")
    socket.getaddrinfo = refuse_module("socket.getaddrinfo")
    try:
        yield report
    finally:
        socket.socket.connect = saved["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = saved["connect_ex"]  # type: ignore[method-assign]
        socket.socket.sendto = saved["sendto"]  # type: ignore[method-assign]
        socket.create_connection = saved["create_connection"]
        socket.getaddrinfo = saved["getaddrinfo"]

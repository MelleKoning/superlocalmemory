# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fake every urllib exit a test's code under test can take, with one mock.

Code that talks to the daemon reaches the network two ways: plain
``urllib.request.urlopen`` (health probes) and the egress gate
``core.outbound_http.urlopen``, which opens through its own no-redirect
opener (``outbound_http._send``). A test that fakes only the first lets the
second reach a real socket. ``patch_urlopen`` fakes both with the same mock,
so call order, ``side_effect`` sequences and ``call_count`` read exactly as
they did when there was a single exit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

_GATE_SEND = "superlocalmemory.core.outbound_http._send"


@contextlib.contextmanager
def patch_urlopen(*args: Any, **kwargs: Any) -> Iterator[MagicMock]:
    """``patch("urllib.request.urlopen", ...)`` that also covers the gate."""
    with patch("urllib.request.urlopen", *args, **kwargs) as fake, \
            patch(_GATE_SEND, new=fake):
        yield fake


def setattr_urlopen(monkeypatch: Any, fake: Any) -> None:
    """``monkeypatch`` both exits with ``fake``."""
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr(_GATE_SEND, fake)

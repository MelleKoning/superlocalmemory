"""LanceDB's process-wide loop cannot be stopped by close() (upstream
lancedb/lancedb#2133, #4031). What SLM relies on instead: it is a daemon
thread, so it can never keep SLM from exiting. If an upgrade changes that,
this fails and the close() note in vector/lancedb_backend.py must be revisited.
"""

import pytest

background_loop = pytest.importorskip("lancedb.background_loop")


def test_the_lancedb_background_loop_is_a_daemon_thread() -> None:
    loop = background_loop.BackgroundEventLoop()
    try:
        assert loop.thread.daemon is True
        assert loop.thread.name == "LanceDBBackgroundEventLoop"
    finally:
        loop.loop.call_soon_threadsafe(loop.loop.stop)
        loop.thread.join(5)

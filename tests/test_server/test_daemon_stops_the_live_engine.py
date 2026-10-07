# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A daemon stop closes the engine in use, not the one it started with."""

from types import SimpleNamespace

from superlocalmemory.server.live_engine import engine_to_close


def test_after_a_settings_change_the_published_engine_is_the_one_closed():
    started, live = object(), object()
    assert engine_to_close(SimpleNamespace(engine=live), started) is live


def test_without_a_published_engine_the_start_up_one_is_closed():
    started = object()
    assert engine_to_close(SimpleNamespace(), started) is started
    assert engine_to_close(SimpleNamespace(engine=None), started) is started

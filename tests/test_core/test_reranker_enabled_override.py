# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""SLM_RERANKER_ENABLED is a process-only override for the lite
bot-host profile. It must win over the persisted `use_cross_encoder` choice
when set, leave it alone when unset, and never leak back into config.json
through a later save() — an env override that silently became a permanent
on-disk setting would be its own, worse bug.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.config import RetrievalConfig, SLMConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SLM_RERANKER_ENABLED", raising=False)
    yield


def test_unset_env_falls_through_to_the_persisted_value_true():
    rc = RetrievalConfig(use_cross_encoder=True)
    assert rc.reranker_enabled() is True


def test_unset_env_falls_through_to_the_persisted_value_false():
    rc = RetrievalConfig(use_cross_encoder=False)
    assert rc.reranker_enabled() is False


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "NO", "off", "Off"])
def test_env_false_spellings_override_a_persisted_true(monkeypatch, value):
    monkeypatch.setenv("SLM_RERANKER_ENABLED", value)
    rc = RetrievalConfig(use_cross_encoder=True)
    assert rc.reranker_enabled() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", "anything-truthy"])
def test_env_true_spellings_override_a_persisted_false(monkeypatch, value):
    monkeypatch.setenv("SLM_RERANKER_ENABLED", value)
    rc = RetrievalConfig(use_cross_encoder=False)
    assert rc.reranker_enabled() is True


def test_override_never_mutates_the_persisted_field(monkeypatch):
    """The override must be read-only: it decides behaviour, it does not
    change what asdict()/save() would write out."""
    monkeypatch.setenv("SLM_RERANKER_ENABLED", "false")
    rc = RetrievalConfig(use_cross_encoder=True)
    assert rc.reranker_enabled() is False
    assert rc.use_cross_encoder is True, "the env override must not mutate the field"


def test_save_then_reload_never_persists_the_env_override(tmp_path, monkeypatch):
    """Round-trip through disk with the override active, then read it back
    with the override gone: the file must still say what the user chose,
    not what the override made the process do."""
    config_path = tmp_path / "config.json"
    cfg = SLMConfig(base_dir=tmp_path)
    cfg.retrieval.use_cross_encoder = True
    cfg.save(config_path)

    monkeypatch.setenv("SLM_RERANKER_ENABLED", "false")
    assert cfg.retrieval.reranker_enabled() is False  # override is in effect
    cfg.save(config_path)  # saving while overridden must not flip the file
    monkeypatch.delenv("SLM_RERANKER_ENABLED", raising=False)

    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk["retrieval"]["use_cross_encoder"] is True, (
        "SLM_RERANKER_ENABLED leaked into config.json — "
        "a process-only override must never become a permanent setting"
    )

    reloaded = SLMConfig.load(config_path)
    assert reloaded.retrieval.reranker_enabled() is True

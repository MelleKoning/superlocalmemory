# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The check has to be where the write is, not beside it.

A validation endpoint the caller may or may not have called is not a guard. The
route that persists the embedding model is the only place a mismatched vector
width can actually be stopped, so the refusal lives there and returns 409.

It fails open on anything it cannot determine — a store with no vectors, a model
server that is not running — because refusing on "I could not tell" would block
a legitimate first-time setup, which is the common case.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.server.routes.v3_api import _refuse_incompatible_embedding


class _Emb:
    def __init__(self, provider="", ollama_model="", ollama_base_url=""):
        self.provider = provider
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url


class _Config:
    def __init__(self, base_dir: Path, provider: str = ""):
        self.base_dir = base_dir
        self.embedding = _Emb(provider=provider)


def _store_holding(tmp_path: Path, width: int | None) -> Path:
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute(
        "CREATE TABLE embedding_metadata (vec_rowid INTEGER PRIMARY KEY,"
        " fact_id TEXT, model_name TEXT, dimension INTEGER)"
    )
    if width is not None:
        conn.execute(
            "INSERT INTO embedding_metadata VALUES (1,'f','m',?)", (width,)
        )
    conn.commit()
    conn.close()
    return tmp_path


def test_a_different_width_is_refused(tmp_path) -> None:
    _store_holding(tmp_path, 768)
    result = _refuse_incompatible_embedding(
        _Config(tmp_path), _Emb(), "mxbai-embed-large", 1024,
    )
    assert result is not None, "a 1024-wide model was accepted into a 768 store"
    assert result.status_code == 409
    body = result.body.decode()
    assert "768" in body and "1024" in body
    # 4.1.21 named `slm db migrate`, which only applies schema migrations.
    assert "slm embedder switch mxbai-embed-large --dimension 1024" in body, (
        "refusing without naming the command that actually re-embeds")
    assert "slm db migrate" not in body


def test_the_same_width_is_allowed(tmp_path) -> None:
    _store_holding(tmp_path, 768)
    assert _refuse_incompatible_embedding(
        _Config(tmp_path), _Emb(), "some-other-768d-model", 768,
    ) is None


def test_a_store_with_no_vectors_can_take_anything(tmp_path) -> None:
    _store_holding(tmp_path, None)
    assert _refuse_incompatible_embedding(
        _Config(tmp_path), _Emb(), "mxbai-embed-large", 1024,
    ) is None


def test_no_store_at_all_is_not_a_refusal(tmp_path) -> None:
    assert _refuse_incompatible_embedding(
        _Config(tmp_path / "nothing-here"), _Emb(), "mxbai-embed-large", 1024,
    ) is None


def test_a_broken_check_does_not_block_a_save(tmp_path, monkeypatch) -> None:
    """Failing open is deliberate; failing closed would lock users out."""
    class Exploding:
        @property
        def base_dir(self):
            raise RuntimeError("cannot resolve the data root")
        embedding = _Emb()

    assert _refuse_incompatible_embedding(
        Exploding(), _Emb(), "anything", 1024,
    ) is None


def test_the_ollama_probe_measures_the_model_being_saved(tmp_path, monkeypatch) -> None:
    """It measured the model already configured, so it always matched.

    Probing the current model returns the width the store already holds, which
    equals the stored width by definition — so the guard passed every time and
    protected nothing. These tests never set the Ollama provider, so the probe
    path was never exercised at all.
    """
    _store_holding(tmp_path, 768)

    asked: list[str] = []

    class _Probe:
        def __init__(self, dimension):
            self.ok = True
            self.dimension = dimension
            self.message = ""

    import superlocalmemory.core.ollama_validator as validator

    def fake(model_name, role, *, base_url="", timeout=60.0):
        asked.append(model_name)
        return _Probe(4096 if model_name == "qwen3-embedding:8b" else 768)

    monkeypatch.setattr(validator, "validate_ollama_model", fake)

    old = _Emb(provider="ollama", ollama_model="nomic-embed-text")
    result = _refuse_incompatible_embedding(
        _Config(tmp_path, provider="ollama"), old, "qwen3-embedding:8b", 768,
    )

    assert asked == ["qwen3-embedding:8b"], (
        f"the guard probed {asked} — it must ask about the model being saved"
    )
    assert result is not None, "a 4096-wide model was accepted into a 768 store"
    assert result.status_code == 409


def test_a_declared_width_cannot_talk_past_a_measurement(tmp_path, monkeypatch) -> None:
    """A caller declaring 768 does not make a 4096-wide model safe."""
    _store_holding(tmp_path, 768)

    class _Probe:
        ok = True
        dimension = 4096
        message = ""

    import superlocalmemory.core.ollama_validator as validator
    monkeypatch.setattr(
        validator, "validate_ollama_model",
        lambda *a, **k: _Probe(),
    )

    result = _refuse_incompatible_embedding(
        _Config(tmp_path, provider="ollama"),
        _Emb(provider="ollama", ollama_model="nomic-embed-text"),
        "mxbai-embed-large", 768,
    )
    assert result is not None, "the declared width overrode the measured one"


def test_switching_to_a_local_model_is_probed(tmp_path, monkeypatch) -> None:
    """The guard read the CURRENT provider, so switching to one never probed.

    Switching is exactly when the width changes, so reading only the provider
    already in place meant the check never ran at the moment it was needed.
    """
    _store_holding(tmp_path, 768)
    asked: list[str] = []

    class _Probe:
        ok = True
        message = ""
        def __init__(self, dimension):
            self.dimension = dimension

    import superlocalmemory.core.ollama_validator as validator

    def fake(model_name, role, *, base_url="", timeout=60.0):
        asked.append(model_name)
        return _Probe(4096)

    monkeypatch.setattr(validator, "validate_ollama_model", fake)

    # Currently no provider at all; the caller is switching TO the local one and
    # declaring a width that happens to match the store.
    result = _refuse_incompatible_embedding(
        _Config(tmp_path), _Emb(provider=""), "qwen3-embedding:8b", 768,
        new_provider="ollama",
    )

    assert asked == ["qwen3-embedding:8b"], (
        "switching to a local model did not probe it"
    )
    assert result is not None, (
        "a 4096-wide model entered a 768 store because the declared width lied"
    )

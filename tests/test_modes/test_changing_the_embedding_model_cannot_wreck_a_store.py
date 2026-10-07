# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Two vector widths in one store is the one change that fails silently.

Every other misconfiguration announces itself. This one does not: a similarity
search over vectors of mismatched widths still returns ten results, ranked, with
scores. They are noise, and nothing in the system can tell.

So the switch is refused, not warned about, and these tests hold that line —
including the case that must still be allowed, because a false alarm about the
one genuinely dangerous change is how a real alarm gets ignored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest

from superlocalmemory.core.ollama_validator import (
    EMBEDDING,
    GENERATION,
    check_embedding_model_change,
    same_embedding_model,
    stored_embedding_dimension,
    validate_ollama_model,
)

OLLAMA = "http://localhost:11434"


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{OLLAMA}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama is not running; start it with `ollama serve` to run these",
)


@pytest.fixture()
def store_with_768d_vectors(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE embedding_metadata ("
        " vec_rowid INTEGER PRIMARY KEY, fact_id TEXT, model_name TEXT,"
        " dimension INTEGER)"
    )
    conn.execute(
        "INSERT INTO embedding_metadata VALUES (1, 'f1', 'nomic-embed-text', 768)"
    )
    conn.commit()
    conn.close()
    return db


class _Response:
    """Just enough of an httpx response for the validator to read."""

    def __init__(self, status_code: int, payload: object = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ""

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _answers(monkeypatch, response) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: response)


def _raises(monkeypatch, exc) -> None:
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(httpx, "post", boom)


# --- the refusal that matters ----------------------------------------------

def test_a_wider_model_is_refused_not_warned_about(
    monkeypatch, store_with_768d_vectors
) -> None:
    _answers(monkeypatch, _Response(200, {"embeddings": [[0.0] * 1024]}))
    result = check_embedding_model_change(
        "mxbai-embed-large", db_path=store_with_768d_vectors
    )
    assert result.allowed is False
    assert result.stored_dimension == 768
    assert result.new_dimension == 1024
    assert "768" in result.message and "1024" in result.message
    # 4.1.21 named `slm db migrate`, which never re-embeds anything.
    assert "slm embedder switch mxbai-embed-large --dimension 1024" in result.message, (
        "refusing without saying how to proceed leaves the user stuck"
    )


def test_the_same_width_is_allowed(monkeypatch, store_with_768d_vectors) -> None:
    _answers(monkeypatch, _Response(200, {"embeddings": [[0.0] * 768]}))
    result = check_embedding_model_change(
        "some-other-768d-model", db_path=store_with_768d_vectors
    )
    assert result.allowed is True


def test_an_empty_store_can_take_any_model(monkeypatch, tmp_path) -> None:
    _answers(monkeypatch, _Response(200, {"embeddings": [[0.0] * 1024]}))
    result = check_embedding_model_change(
        "mxbai-embed-large", db_path=tmp_path / "nothing-here.db"
    )
    assert result.allowed is True
    assert "nothing has to be rebuilt" in result.message


def test_two_names_for_one_model_are_not_a_change(
    monkeypatch, store_with_768d_vectors
) -> None:
    """A familiar name softens the message. It never replaces the measurement.

    This test used to assert the opposite — that a recognised name skips the
    probe entirely — and that shortcut was the hole. Names are not widths:
    ``qwen3-embedding:0.6b`` and ``qwen3-embedding:8b`` looked like one model to
    the name check and emit 1024 and 4096 numbers respectively.
    """
    _answers(monkeypatch, _Response(200, {"embeddings": [[0.0] * 768]}))
    result = check_embedding_model_change(
        "nomic-embed-text",
        db_path=store_with_768d_vectors,
        current_model="nomic-ai/nomic-embed-text-v1.5",
    )
    assert result.allowed is True
    assert result.new_dimension == 768, "the width was not measured"


def test_a_familiar_name_does_not_excuse_a_different_width(
    monkeypatch, store_with_768d_vectors
) -> None:
    """The case that made the shortcut dangerous."""
    _answers(monkeypatch, _Response(200, {"embeddings": [[0.0] * 4096]}))
    result = check_embedding_model_change(
        "qwen3-embedding:8b",
        db_path=store_with_768d_vectors,
        current_model="qwen3-embedding:0.6b",
    )
    assert result.allowed is False, (
        "a 4096-wide model was allowed into a 768 store because its name looked "
        "like the one already in use"
    )
    assert result.new_dimension == 4096


def test_a_stopped_server_can_no_longer_be_talked_past(
    monkeypatch, store_with_768d_vectors
) -> None:
    """If the width cannot be measured, the switch does not happen."""
    _raises(monkeypatch, httpx.ConnectError("refused"))
    result = check_embedding_model_change(
        "nomic-embed-text",
        db_path=store_with_768d_vectors,
        current_model="nomic-ai/nomic-embed-text-v1.5",
    )
    assert result.allowed is False
    assert "ollama serve" in result.message


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text", True),
        ("nomic-embed-text:latest", "nomic-embed-text", True),
        ("nomic-embed-text", "mxbai-embed-large", False),
        ("bge-m3", "bge-large", False),
        # An Ollama tag is part of the identity, not packaging. These two emit
        # 1024 and 4096 numbers.
        ("qwen3-embedding:0.6b", "qwen3-embedding:8b", False),
        ("mxbai-embed-large:335m", "mxbai-embed-large:1b", False),
    ],
)
def test_which_names_mean_the_same_weights(left, right, same) -> None:
    assert same_embedding_model(left, right) is same


# --- the failures a user can actually cause --------------------------------

def test_a_missing_model_says_how_to_get_it(monkeypatch) -> None:
    _answers(monkeypatch, _Response(404, None, 'model "typo" not found'))
    probe = validate_ollama_model("typo", EMBEDDING)
    assert probe.ok is False
    assert "ollama pull typo" in probe.message


def test_a_stopped_server_says_how_to_start_it(monkeypatch) -> None:
    _raises(monkeypatch, httpx.ConnectError("refused"))
    probe = validate_ollama_model("nomic-embed-text", EMBEDDING)
    assert probe.ok is False
    assert "ollama serve" in probe.message


def test_a_slow_server_is_not_reported_as_a_missing_model(monkeypatch) -> None:
    _raises(monkeypatch, httpx.ReadTimeout("slow"))
    probe = validate_ollama_model("nomic-embed-text", EMBEDDING, timeout=5)
    assert probe.ok is False
    assert "ollama list" in probe.message
    assert "pull" not in probe.message


def test_a_chat_model_in_the_embedding_field_is_named_as_such(monkeypatch) -> None:
    _answers(monkeypatch, _Response(501, None, "This server does not support embeddings"))
    probe = validate_ollama_model("llama3.2", EMBEDDING)
    assert probe.ok is False
    assert "not an embedding model" in probe.message


def test_an_unknown_role_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        validate_ollama_model("nomic-embed-text", "summarising")


def test_a_store_that_holds_nothing_reports_no_width(tmp_path) -> None:
    assert stored_embedding_dimension(tmp_path / "absent.db") is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    assert stored_embedding_dimension(empty) is None


# --- against a real server, when there is one -------------------------------

@needs_ollama
def test_the_real_embedding_model_reports_its_real_width() -> None:
    probe = validate_ollama_model("nomic-embed-text", EMBEDDING)
    if not probe.ok:
        pytest.skip(f"nomic-embed-text is not pulled here: {probe.message}")
    assert probe.dimension == 768


@needs_ollama
def test_a_real_generation_model_answers() -> None:
    probe = validate_ollama_model("llama3.2", GENERATION)
    if not probe.ok:
        pytest.skip(f"llama3.2 is not pulled here: {probe.message}")
    assert probe.dimension is None

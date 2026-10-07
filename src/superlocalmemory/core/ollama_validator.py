# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ask the local model server whether a model exists, before trusting it.

Mode B runs on Ollama, and a user picking a model there picks it by typing a
name. Two things then go wrong silently.

**The name is wrong.** Nothing happens at the moment of the mistake; the config
saves, the daemon starts, and every write quietly falls back to a worse path.
The error a user eventually sees is unrelated to what they did.

**The name is right and the shape is different.** Every embedding model emits a
vector of a fixed width. `nomic-embed-text` emits 768 numbers,
`mxbai-embed-large` emits 1024. Vectors of different widths cannot be compared,
so a store holding both answers similarity questions with noise — and it does so
without failing, because nothing in a similarity search knows the difference
between a bad answer and a good one. **This is the single silent-and-catastrophic
change a user can make**, so it is refused rather than warned about, and the
refusal says the one command that would make it safe.

There are two Ollama roles and they are separate models with separate names:
the embedding model (`embedding.ollama_model`) turns text into vectors, and the
generation model (`llm.model` when the provider is Ollama) writes summaries.
They are validated independently because a machine can perfectly well have one
and not the other.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "EMBEDDING",
    "GENERATION",
    "OllamaProbe",
    "DimensionChange",
    "validate_ollama_model",
    "stored_embedding_dimension",
    "check_embedding_model_change",
    "same_embedding_model",
]

logger = logging.getLogger(__name__)

EMBEDDING = "embedding"
GENERATION = "generation"

DEFAULT_BASE_URL = "http://localhost:11434"

#: Long enough for a cold model load, short enough that a wedged server is not
#: mistaken for a slow one.
_CONNECT_TIMEOUT = 3.0
_RESPONSE_TIMEOUT = 60.0

_PROBE_TEXT = "superlocalmemory model probe"
_PROBE_PROMPT = "Reply with the single word: ok"


@dataclass(frozen=True)
class OllamaProbe:
    """What the server said when asked about one model."""

    ok: bool
    message: str
    dimension: int | None = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.ok


@dataclass(frozen=True)
class DimensionChange:
    """A model change that would put two vector widths in one store."""

    allowed: bool
    message: str
    stored_dimension: int | None = None
    new_dimension: int | None = None


def same_embedding_model(left: str, right: str) -> bool:
    """True when two names are two spellings of one model.

    ``nomic-ai/nomic-embed-text-v1.5`` is the HuggingFace name and
    ``nomic-embed-text`` is the Ollama pull name for the same weights. Warning
    about a dimension change between them would be a false alarm, and a false
    alarm about the one thing that is genuinely dangerous is how a real warning
    gets ignored.
    """
    def base(name: str) -> str:
        stem = name.strip().lower().rsplit("/", 1)[-1]
        # An Ollama tag is PART OF THE IDENTITY, not packaging.
        # ``qwen3-embedding:0.6b`` emits 1024 numbers and ``qwen3-embedding:8b``
        # emits 4096, and stripping the tag made this function call them the
        # same model — which then skipped the width check that exists to stop
        # exactly that swap. Only ``:latest`` means "whatever the bare name
        # means", so only that one is dropped.
        if stem.endswith(":latest"):
            stem = stem[: -len(":latest")]
        # A HuggingFace revision suffix on an otherwise identical name is
        # packaging; ``nomic-embed-text-v1.5`` and ``nomic-embed-text`` are the
        # same weights under two registries' conventions.
        if ":" not in stem:
            for suffix in ("-v1.5", "-v1", "-v2"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
        return stem

    return base(left) == base(right)


def validate_ollama_model(
    model_name: str,
    role: str = EMBEDDING,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = _RESPONSE_TIMEOUT,
) -> OllamaProbe:
    """Ask the server to actually use the model, and report what happened.

    Listing the installed models is not enough: a name can be present and the
    model still fail to load. So this runs the smallest real request for the
    role and reads the answer, which is also the only way to learn an embedding
    model's width.
    """
    if role not in (EMBEDDING, GENERATION):
        raise ValueError(f"role must be {EMBEDDING!r} or {GENERATION!r}, got {role!r}")
    name = (model_name or "").strip()
    if not name:
        return OllamaProbe(False, "No model name given.")

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a hard dependency
        return OllamaProbe(False, "httpx is not installed, so Ollama cannot be reached.")

    url = base_url.rstrip("/")
    request_timeout = httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT)

    try:
        if role == EMBEDDING:
            response = httpx.post(
                f"{url}/api/embed",
                json={"model": name, "input": [_PROBE_TEXT]},
                timeout=request_timeout,
            )
        else:
            response = httpx.post(
                f"{url}/api/generate",
                json={"model": name, "prompt": _PROBE_PROMPT, "stream": False},
                timeout=request_timeout,
            )
    except httpx.ConnectError:
        return OllamaProbe(
            False,
            f"Ollama is not running at {url}. Start it with: ollama serve",
        )
    except httpx.TimeoutException:
        return OllamaProbe(
            False,
            f"Ollama did not answer within {timeout:.0f}s. The model may still be "
            f"downloading — check with: ollama list",
        )
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        return OllamaProbe(False, f"Could not reach Ollama at {url}: {exc}")

    if response.status_code == 404 or _looks_missing(response):
        return OllamaProbe(
            False,
            f"Ollama has no model called {name!r}. Run: ollama pull {name}",
        )
    if role == EMBEDDING and _refuses_embeddings(response):
        return OllamaProbe(
            False,
            f"{name!r} is not an embedding model — the server refused the "
            f"request. Pick a model built for embeddings, such as "
            f"nomic-embed-text.",
        )
    if response.status_code != 200:
        return OllamaProbe(
            False,
            f"Ollama answered {response.status_code} for {name!r}: "
            f"{response.text[:200]}",
        )

    try:
        payload = response.json()
    except ValueError:
        return OllamaProbe(False, f"Ollama returned something that is not JSON for {name!r}.")

    if role == EMBEDDING:
        vectors = payload.get("embeddings") or []
        if not vectors or not isinstance(vectors[0], list) or not vectors[0]:
            return OllamaProbe(
                False,
                f"{name!r} answered but returned no vector, so it is not an "
                f"embedding model. Pick one built for embeddings, such as "
                f"nomic-embed-text.",
            )
        width = len(vectors[0])
        return OllamaProbe(True, f"{name} emits {width}-dimensional vectors.", width)

    text = str(payload.get("response") or "").strip()
    if not text:
        return OllamaProbe(
            False,
            f"{name!r} answered but wrote nothing, so it cannot be used to "
            f"generate summaries.",
        )
    return OllamaProbe(True, f"{name} answered a probe prompt.")


def _refuses_embeddings(response: object) -> bool:
    """A generation-only model answers an embedding request with a refusal.

    The server's own wording is about its build flags and means nothing to
    somebody who typed a chat model's name into an embedding field.
    """
    status = getattr(response, "status_code", 0)
    try:
        lowered = response.text.lower()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        lowered = ""
    if status == 501:
        return True
    return "does not support embed" in lowered or "not support embeddings" in lowered


def _looks_missing(response: object) -> bool:
    """Ollama reports an unknown model as a 400 with a body that says so."""
    try:
        body = response.text  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        return False
    lowered = body.lower()
    return "not found" in lowered and "model" in lowered


def stored_embedding_dimension(db_path: str | Path) -> int | None:
    """The vector width this store already holds, or None if it holds none."""
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT dimension FROM embedding_metadata "
            "WHERE dimension IS NOT NULL AND dimension > 0 LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return int(row[0]) if row else None


def check_embedding_model_change(
    new_model: str,
    *,
    db_path: str | Path,
    current_model: str = "",
    base_url: str = DEFAULT_BASE_URL,
) -> DimensionChange:
    """Decide whether switching the embedding model is safe for this store.

    Refuses rather than warns. A warning printed into a log is not a decision,
    and the outcome of getting this wrong is a store whose similarity search is
    quietly meaningless.
    """
    # The width is asked for FIRST, always. Recognising two names as the same
    # model may soften the message, but it must never stand in for measuring —
    # a name that merely looks familiar is exactly how a different-width model
    # would get through.
    probe = validate_ollama_model(new_model, EMBEDDING, base_url=base_url)
    if not probe.ok:
        return DimensionChange(False, probe.message)

    familiar = bool(current_model) and same_embedding_model(current_model, new_model)
    stored = stored_embedding_dimension(db_path)
    if familiar and (stored is None or stored == probe.dimension):
        return DimensionChange(
            True,
            f"{new_model} is another name for the model already in use, and it "
            f"emits the same {probe.dimension}-dimensional vectors.",
            stored,
            probe.dimension,
        )
    if stored is None:
        return DimensionChange(
            True,
            f"{probe.message} This store holds no vectors yet, so nothing has to "
            f"be rebuilt.",
            None,
            probe.dimension,
        )
    if stored == probe.dimension:
        return DimensionChange(True, probe.message, stored, probe.dimension)

    return DimensionChange(
        False,
        f"{new_model} emits {probe.dimension}-dimensional vectors and this store "
        f"holds {stored}-dimensional ones. Vectors of different widths cannot be "
        f"compared, so every memory already stored would become unfindable by "
        f"meaning. Switch with: slm embedder switch {new_model} --dimension "
        f"{probe.dimension} (re-indexes in the background).",
        stored,
        probe.dimension,
    )

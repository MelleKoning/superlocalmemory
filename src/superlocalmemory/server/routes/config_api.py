# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Dashboard config endpoints — storage, daemon, mesh, trust, forgetting.

Each section provides GET (read current) and PUT (validate + persist) routes.
Writes use the same direct-JSON approach as the evolution config endpoint:
  1. Read config.json as a raw dict (fail-open → defaults when absent/corrupt).
  2. Update only the targeted keys; all other keys — including 'mode' — survive.
  3. Atomic write via a temp file + os.replace (no torn writes).

Auth: follows the same pattern as the auto-capture/auto-recall/auto-invoke
config endpoints — no route-level auth guard.  Write-identity is enforced by
the middleware layer wired in unified_daemon.py / ui.py for non-localhost
callers.  Tests use a bare FastAPI app (no middleware) and monkeypatch
MEMORY_DIR to a tmp_path.

Restart-required semantics:
  - graph_backend / vector_backend: changes take effect only on daemon restart.
  - daemon_port / daemon_legacy_port: changes take effect only on daemon restart.
  - mesh, trust, and forgetting are persisted safely but require restart
    because no complete supported worker-rebind transaction exists for them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.server.config_file import read_config, update_config
from superlocalmemory.server.routes.helpers import MEMORY_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["config"])


def _require_admin(request: Request) -> None:
    """SEC-H-01: system-configuration mutations are admin-only (MANAGE).

    Machine auth (loopback / credential) is enforced by the daemon middleware;
    this adds the RBAC layer so a logged-in non-admin (viewer/member) in company
    mode cannot change the daemon port, swap the LLM key, or alter the forgetting
    curve. The machine owner keeps MANAGE (personal mode is unaffected). Call
    this BEFORE the handler's try/except so the 401/403 is not swallowed into a
    500.
    """
    from superlocalmemory.server.rbac_enforce import require_manage
    require_manage(request)

# ---------------------------------------------------------------------------
# Allowed backend values
# ---------------------------------------------------------------------------

_GRAPH_BACKENDS = frozenset({"auto", "sqlite", "cozo"})
_VECTOR_BACKENDS = frozenset({"auto", "lancedb", "sqlite-vec"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Return the config.json path, resolved from the active MEMORY_DIR."""
    return MEMORY_DIR / "config.json"


def _read_config() -> dict:
    """Read one coherent config snapshot."""
    p = _config_path()
    try:
        return read_config(p)
    except (ValueError, OSError) as exc:
        logger.warning("config_api: could not read config.json: %s", exc)
        raise


def _update_config(mutator) -> dict:
    """Run one interprocess-locked read/modify/replace transaction."""
    return update_config(_config_path(), mutator)


# ---------------------------------------------------------------------------
# Pydantic request models  (extra="forbid" → 422 on unknown keys)
# ---------------------------------------------------------------------------


class StorageConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_backend: Optional[Annotated[str, Field(pattern=r"^(auto|sqlite|cozo)$")]] = None
    vector_backend: Optional[Annotated[str, Field(pattern=r"^(auto|lancedb|sqlite-vec)$")]] = None


class DaemonConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_timeout: Optional[int] = Field(None, ge=0)
    port: Optional[int] = Field(None, ge=1, le=65535)
    legacy_port: Optional[int] = Field(None, ge=1, le=65535)
    enable_legacy_port: Optional[StrictBool] = None


class MeshConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool


class TrustConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_trust_weighting: Optional[StrictBool] = None
    trust_first_party: Optional[StrictBool] = None
    promotion_min_trust: Optional[float] = Field(None, ge=0.0, le=1.0)


class ForgettingConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    alpha: Optional[float] = Field(None, gt=0.0)
    beta: Optional[float] = Field(None, gt=0.0)
    gamma: Optional[float] = Field(None, gt=0.0)
    delta: Optional[float] = Field(None, gt=0.0)
    min_strength: Optional[float] = Field(None, gt=0.0)
    max_strength: Optional[float] = Field(None, gt=0.0)
    archive_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    forget_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_rate: Optional[float] = Field(None, gt=0.0)
    forgetting_drift_scale: Optional[float] = Field(None, gt=0.0)
    # ge, not gt: zero is a meaningful setting — it turns trust-modulated decay
    # off, so every memory fades at the same rate regardless of where it came
    # from. A greater-than bound made the one value that disables the feature
    # the one value the API refused.
    trust_kappa: Optional[float] = Field(None, ge=0.0)
    scheduler_interval_minutes: Optional[int] = Field(None, ge=1)
    core_memory_immune: Optional[StrictBool] = None


# ---------------------------------------------------------------------------
# Default value constants (mirrors ForgettingConfig dataclass defaults)
# ---------------------------------------------------------------------------

_FORGETTING_DEFAULTS: dict = {
    "enabled": True,
    "alpha": 2.0,
    "beta": 1.5,
    "gamma": 1.0,
    "delta": 0.5,
    "min_strength": 0.1,
    "max_strength": 100.0,
    "archive_threshold": 0.2,
    "forget_threshold": 0.05,
    "learning_rate": 1.0,
    "forgetting_drift_scale": 0.5,
    "trust_kappa": 2.0,
    "scheduler_interval_minutes": 30,
    "core_memory_immune": True,
}


# ---------------------------------------------------------------------------
# GET /api/v3/storage/config
# ---------------------------------------------------------------------------


@router.get("/storage/config")
def get_storage_config(request: Request = None):  # noqa: B008 - FastAPI idiom
    """Return current storage backend configuration.

    base_dir is read-only — it is derived from the process namespace and
    cannot be changed via this endpoint.
    """
    try:
        data = _read_config()
        declared_graph = data.get("graph_backend", "auto")
        declared_vector = data.get("vector_backend", "auto")
        active_graph, active_vector = _active_backends(declared_graph, declared_vector)
        return {
            "graph_backend": declared_graph,
            "vector_backend": declared_vector,
            # What is actually answering queries. A store can be configured for
            # a backend it never successfully promoted to, and then the setting
            # describes an intention rather than the system.
            "graph_backend_active": active_graph,
            "vector_backend_active": active_vector,
            "backend_matches_configuration": (
                active_graph == declared_graph and active_vector == declared_vector
            ),
            "scale_engine_state": data.get("scale_engine_state", "local_core"),
            # What the last start did about moving this store onto the graph
            # and vector backends. A store that could not be moved says why
            # here rather than looking like one that was never asked.
            "automatic_promotion": getattr(
                request.app.state, "scale_autopromotion", None,
            ) if request is not None else None,
            "base_dir": data.get("base_dir", str(MEMORY_DIR)),
        }
    except Exception:
        logger.exception("get_storage_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def _active_backends(declared_graph: str, declared_vector: str) -> tuple[str, str]:
    """Which backends are really serving queries, not which were requested.

    A promotion writes the chosen backend into the configuration before the
    directory that holds it exists, and a promotion that never completed leaves
    the setting saying ``cozo`` while every query is answered by SQLite. The
    dashboard read the setting, so it agreed with the mistake.

    Resolved from the two things that must both be true for a backend to serve:
    its library imports, and its data directory is on disk.
    """
    def usable(module: str, directory: str) -> bool:
        # Importing it, not merely finding it. A package whose native extension
        # no longer matches the interpreter is present on disk and raises on
        # import, and "the file is there" would report it as serving queries it
        # cannot answer.
        import importlib

        if not (MEMORY_DIR / directory).is_dir():
            return False
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 - any import failure means unusable
            return False
        return True

    graph = "sqlite"
    if declared_graph in ("cozo", "auto") and usable("pycozo", "cozo"):
        graph = "cozo"

    vector = "sqlite-vec"
    if declared_vector in ("lancedb", "auto") and usable("lancedb", "lance"):
        vector = "lancedb"

    return graph, vector


# ---------------------------------------------------------------------------
# PUT /api/v3/storage/config
# ---------------------------------------------------------------------------


@router.put("/storage/config")
def put_storage_config(request: Request, body: StorageConfigUpdate):
    """Update graph_backend and/or vector_backend.

    Both fields require a daemon restart to take effect.
    Returns restart_required: true unconditionally.
    """
    _require_admin(request)
    try:
        def mutate(data: dict) -> None:
            if body.graph_backend is not None:
                data["graph_backend"] = body.graph_backend
            if body.vector_backend is not None:
                data["vector_backend"] = body.vector_backend

        data = _update_config(mutate)
        return {
            "graph_backend": data.get("graph_backend", "auto"),
            "vector_backend": data.get("vector_backend", "auto"),
            "base_dir": data.get("base_dir", str(MEMORY_DIR)),
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_storage_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/daemon/config
# ---------------------------------------------------------------------------


@router.get("/daemon/config")
def get_daemon_config():
    """Return current daemon configuration."""
    try:
        data = _read_config()
        return {
            "idle_timeout": data.get("daemon_idle_timeout", 0),
            "port": data.get("daemon_port", 8765),
            "legacy_port": data.get("daemon_legacy_port", 8767),
            "enable_legacy_port": data.get("daemon_enable_legacy_port", True),
        }
    except Exception:
        logger.exception("get_daemon_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/daemon/config
# ---------------------------------------------------------------------------


@router.put("/daemon/config")
def put_daemon_config(request: Request, body: DaemonConfigUpdate):
    """Update daemon configuration.

    Port / legacy_port changes require a daemon restart.
    restart_required is True if any port field is included in the request.
    """
    _require_admin(request)
    try:
        port_changed = False

        def mutate(data: dict) -> None:
            nonlocal port_changed
            if body.idle_timeout is not None:
                data["daemon_idle_timeout"] = body.idle_timeout
            if body.port is not None:
                data["daemon_port"] = body.port
                port_changed = True
            if body.legacy_port is not None:
                data["daemon_legacy_port"] = body.legacy_port
                port_changed = True
            if body.enable_legacy_port is not None:
                data["daemon_enable_legacy_port"] = body.enable_legacy_port

        data = _update_config(mutate)
        return {
            "idle_timeout": data.get("daemon_idle_timeout", 0),
            "port": data.get("daemon_port", 8765),
            "legacy_port": data.get("daemon_legacy_port", 8767),
            "enable_legacy_port": data.get("daemon_enable_legacy_port", True),
            "restart_required": port_changed,
        }
    except Exception:
        logger.exception("put_daemon_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/mesh/config
# ---------------------------------------------------------------------------


@router.get("/mesh/config")
def get_mesh_config():
    """Return current mesh configuration."""
    try:
        data = _read_config()
        return {"enabled": data.get("mesh_enabled", True)}
    except Exception:
        logger.exception("get_mesh_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/mesh/config
# ---------------------------------------------------------------------------


@router.put("/mesh/config")
def put_mesh_config(request: Request, body: MeshConfigUpdate):
    """Persist mesh state; restart is required to rebuild the mesh worker."""
    _require_admin(request)
    try:
        _update_config(
            lambda data: data.update({"mesh_enabled": body.enabled}),
        )
        return {"enabled": body.enabled, "restart_required": True}
    except Exception:
        logger.exception("put_mesh_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/trust/config
# ---------------------------------------------------------------------------


@router.get("/trust/config")
def get_trust_config():
    """Return current trust configuration.

    Fields are spread across three config sections:
      - retrieval.use_trust_weighting   (Bayesian trust in retrieval ranking)
      - injection.trust_first_party     (framing of injected context)
      - consolidation.promotion_min_trust (min trust required for promotion)
    """
    try:
        data = _read_config()
        retrieval = data.get("retrieval", {})
        injection = data.get("injection", {})
        consolidation = data.get("consolidation", {})
        return {
            "use_trust_weighting": retrieval.get("use_trust_weighting", True),
            "trust_first_party": injection.get("trust_first_party", False),
            "promotion_min_trust": consolidation.get("promotion_min_trust", 0.5),
        }
    except Exception:
        logger.exception("get_trust_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/trust/config
# ---------------------------------------------------------------------------


@router.put("/trust/config")
def put_trust_config(request: Request, body: TrustConfigUpdate):
    """Update trust configuration.

    Each field is stored in its canonical config.json sub-section.
    Unrelated keys in those sub-sections are preserved.
    """
    _require_admin(request)
    try:
        def mutate(data: dict) -> None:
            if body.use_trust_weighting is not None:
                retrieval = data.setdefault("retrieval", {})
                retrieval["use_trust_weighting"] = body.use_trust_weighting
            if body.trust_first_party is not None:
                injection = data.setdefault("injection", {})
                injection["trust_first_party"] = body.trust_first_party
            if body.promotion_min_trust is not None:
                consolidation = data.setdefault("consolidation", {})
                consolidation["promotion_min_trust"] = body.promotion_min_trust

        data = _update_config(mutate)
        retrieval = data.get("retrieval", {})
        injection = data.get("injection", {})
        consolidation = data.get("consolidation", {})
        return {
            "use_trust_weighting": retrieval.get("use_trust_weighting", True),
            "trust_first_party": injection.get("trust_first_party", False),
            "promotion_min_trust": consolidation.get("promotion_min_trust", 0.5),
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_trust_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/forgetting/config
# ---------------------------------------------------------------------------


@router.get("/forgetting/config")
def get_forgetting_config():
    """Return all Ebbinghaus forgetting configuration fields."""
    try:
        data = _read_config()
        stored = data.get("forgetting", {})
        # Return defaults for any field not yet in config.json
        result = {**_FORGETTING_DEFAULTS, **stored}
        # Keep only known fields
        return {k: result[k] for k in _FORGETTING_DEFAULTS}
    except Exception:
        logger.exception("get_forgetting_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/forgetting/config
# ---------------------------------------------------------------------------


@router.put("/forgetting/config")
def put_forgetting_config(request: Request, body: ForgettingConfigUpdate):
    """Update Ebbinghaus forgetting configuration.

    Only provided fields are changed; all other forgetting fields are
    preserved. Changes take effect after the daemon restarts.
    """
    _require_admin(request)
    try:
        updates = body.model_dump(exclude_none=True)

        def mutate(data: dict) -> None:
            stored = data.get("forgetting", {})
            merged = {**_FORGETTING_DEFAULTS, **stored, **updates}
            data["forgetting"] = merged

        data = _update_config(mutate)
        merged = data["forgetting"]
        return {
            **{k: merged[k] for k in _FORGETTING_DEFAULTS},
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_forgetting_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# Graph Pruning config (v3.8.4-G — GitHub #84)
# Dedicated endpoint so the forgetting config partial-update contract stays
# clean.  Callers that only want graph knobs do not need to know about the
# Ebbinghaus curve, and vice-versa.
# ---------------------------------------------------------------------------

_GRAPH_PRUNING_DEFAULTS: dict = {
    "max_degree_per_node": 100,
    "min_edge_weight": 0.0,
    "enabled": True,
}


class GraphPruningConfigUpdate(BaseModel):
    """Partial update model for graph pruning configuration.

    All fields are optional so a PUT with only ``max_degree_per_node`` does
    NOT reset ``min_edge_weight`` back to the default.  Existing values are
    preserved; only the supplied fields are overwritten.
    """

    model_config = ConfigDict(extra="forbid")

    max_degree_per_node: Optional[int] = Field(None, ge=1)
    min_edge_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    enabled: Optional[StrictBool] = None


# ---------------------------------------------------------------------------
# GET /api/v3/graph/config
# ---------------------------------------------------------------------------


@router.get("/graph/config")
def get_graph_config():
    """Return current graph thinning configuration.

    Returns all three fields with their defaults when config.json has no
    ``graph_pruning`` section (old installations).
    """
    try:
        data = _read_config()
        stored = data.get("graph_pruning", {})
        result = {**_GRAPH_PRUNING_DEFAULTS, **stored}
        # Return only known fields
        return {k: result[k] for k in _GRAPH_PRUNING_DEFAULTS}
    except Exception:
        logger.exception("get_graph_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/graph/config
# ---------------------------------------------------------------------------


@router.put("/graph/config")
def put_graph_config(request: Request, body: GraphPruningConfigUpdate):
    """Update graph thinning configuration.

    Only supplied fields are written; all other graph pruning fields are
    preserved.  Changes take effect at the next maintenance cycle without
    requiring a daemon restart (the scheduler reads graph_pruning live from
    the config object, which the daemon reloads from disk on each cycle).
    """
    _require_admin(request)
    try:
        updates = body.model_dump(exclude_none=True)

        def mutate(data: dict) -> None:
            stored = data.get("graph_pruning", {})
            merged = {**_GRAPH_PRUNING_DEFAULTS, **stored, **updates}
            data["graph_pruning"] = merged

        data = _update_config(mutate)
        merged = data["graph_pruning"]
        return {k: merged[k] for k in _GRAPH_PRUNING_DEFAULTS}
    except Exception:
        logger.exception("put_graph_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# Ollama model selection
# ---------------------------------------------------------------------------


class OllamaModelCheck(BaseModel):
    """A model a user is considering, and what they want to use it for."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., min_length=1, max_length=200)
    role: str = Field("embedding", pattern="^(embedding|generation)$")


@router.get("/ollama/models")
def get_ollama_models():
    """Which Ollama models are installed, and which two are in use.

    A user picking a model should be picking from a list, not typing a name and
    finding out later that they typed it wrong.
    """
    from superlocalmemory.core.ollama_validator import DEFAULT_BASE_URL

    try:
        data = _read_config()
        embedding = data.get("embedding") or {}
        llm = data.get("llm") or {}
        base_url = llm.get("base_url") or DEFAULT_BASE_URL

        installed: list[dict] = []
        reachable = True
        detail = ""
        try:
            import httpx

            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
            if response.status_code == 200:
                for entry in response.json().get("models", []):
                    installed.append({
                        "name": entry.get("name", ""),
                        "size": entry.get("size", 0),
                    })
            else:
                reachable = False
                detail = f"Ollama answered {response.status_code}."
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reachable = False
            detail = f"Ollama is not running at {base_url}. Start it with: ollama serve ({exc})"

        return {
            "reachable": reachable,
            "detail": detail,
            "base_url": base_url,
            "installed": sorted(installed, key=lambda m: m["name"]),
            "embedding_model": embedding.get("ollama_model", ""),
            "generation_model": llm.get("model", "") if llm.get("provider") == "ollama" else "",
            "stored_dimension": _stored_dimension(),
        }
    except Exception:
        logger.exception("get_ollama_models failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@router.get("/models/catalog")
def get_models_catalog():
    """The model catalogue plus what to recommend on THIS computer (4.1.22).

    Installed models come from ``get_ollama_models`` (the same one /api/tags
    read, so no second probe); ``local_recommendations`` ranks them by SLM's
    own extraction test and suggests pulls that fit this machine's memory.
    """
    from superlocalmemory.core import model_catalog
    from superlocalmemory.core.machine import total_ram_gb

    try:
        ollama = get_ollama_models()
        ollama = ollama if isinstance(ollama, dict) else {}
        names = [m.get("name", "") for m in ollama.get("installed", [])]
        ram = round(total_ram_gb(), 1) or None
        recs = model_catalog.recommend_local_llms(ram, names)
        return {
            **model_catalog.catalog(),
            "machine": {"ram_gb": ram},
            "ollama": {"reachable": bool(ollama.get("reachable")),
                       "detail": ollama.get("detail", ""), "installed": names},
            "local_recommendations": [
                {"model": r.model_id, "installed": r.installed, "fits": r.fits,
                 "reason": r.reason, "label": r.entry.label if r.entry else r.model_id}
                for r in recs],
            "best_local_llm": model_catalog.best_local_llm(ram, names),
        }
    except Exception:
        logger.exception("get_models_catalog failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@router.post("/ollama/validate")
def post_ollama_validate(request: Request, body: OllamaModelCheck):
    """Ask the server to actually use the model, before anything is saved.

    For the embedding role this also decides whether the switch is safe for this
    store: two vector widths cannot be compared, and a store holding both
    answers similarity questions with noise rather than failing.
    """
    _require_admin(request)
    from superlocalmemory.core.ollama_validator import (
        DEFAULT_BASE_URL,
        EMBEDDING,
        check_embedding_model_change,
        validate_ollama_model,
    )

    try:
        data = _read_config()
        llm = data.get("llm") or {}
        embedding = data.get("embedding") or {}
        base_url = llm.get("base_url") or DEFAULT_BASE_URL

        if body.role != EMBEDDING:
            probe = validate_ollama_model(body.model_name, body.role, base_url=base_url)
            return {
                "ok": probe.ok,
                "message": probe.message,
                "role": body.role,
                "model_name": body.model_name,
                "dimension": probe.dimension,
                "safe_to_apply": probe.ok,
            }

        decision = check_embedding_model_change(
            body.model_name,
            db_path=MEMORY_DIR / "memory.db",
            current_model=embedding.get("ollama_model", "")
                or embedding.get("model_name", ""),
            base_url=base_url,
        )
        return {
            "ok": decision.allowed,
            "message": decision.message,
            "role": body.role,
            "model_name": body.model_name,
            "dimension": decision.new_dimension,
            "stored_dimension": decision.stored_dimension,
            "safe_to_apply": decision.allowed,
        }
    except Exception:
        logger.exception("post_ollama_validate failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def _stored_dimension() -> int | None:
    from superlocalmemory.core.ollama_validator import stored_embedding_dimension

    return stored_embedding_dimension(MEMORY_DIR / "memory.db")

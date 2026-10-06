"""v3.5 release criteria.

These tests are the specification: product code changes until they pass. They
are not edited to accommodate the code — doing so defeats their only purpose.

Covered:
  P2  code-graph resolver wiring       — build_code_graph must succeed on real code
  P3  issue #119 dashboard config wipe — save must not destroy api_key or URL scheme
  P4  issue #112 private-LAN endpoints — trusted plain-HTTP LAN reranker must work

Every test asserts OBSERVABLE BEHAVIOUR, not implementation shape, so it
cannot be satisfied by special-casing the test.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# P2 — code graph must actually build (the FOREIGN KEY / dead-resolver defect)
# ─────────────────────────────────────────────────────────────────────────────
class TestP2CodeGraphBuilds:
    """`build_code_graph` currently fails on EVERY real repo.

    Root cause: code_graph/resolver.py is never imported by production code, so
    the parser's placeholder edges (`__unresolved__` -> `__call__<name>`) reach
    GraphStore with endpoints absent from graph_nodes, and the FK rejects them.
    """

    def _tiny_repo(self) -> pathlib.Path:
        """A repo whose only interesting property is an unresolved call.

        `bar()` is defined locally; `requests.get()` is external and can never
        resolve. Both shapes must be tolerated.
        """
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "a.py").write_text(
            "import requests\n\n\n"
            "def foo():\n"
            "    return bar()\n\n\n"
            "def bar():\n"
            "    return requests.get('https://example.invalid')\n",
            encoding="utf-8",
        )
        return d

    def test_parser_output_has_no_dangling_edges(self) -> None:
        """Every edge endpoint must exist in the node set BEFORE storage.

        This is the precise defect: a 2-function file yields 1 dangling edge.
        """
        from superlocalmemory.code_graph.config import CodeGraphConfig
        from superlocalmemory.code_graph.parser import CodeParser

        repo = self._tiny_repo()
        cfg = CodeGraphConfig(enabled=True, repo_root=repo)
        nodes, edges, _files = CodeParser(cfg).parse_all(repo)

        node_ids = {n.node_id for n in nodes}
        dangling = [
            (e.source_node_id, e.target_node_id)
            for e in edges
            if e.source_node_id not in node_ids or e.target_node_id not in node_ids
        ]
        assert not dangling, (
            f"{len(dangling)} edge(s) reference nodes that do not exist, e.g. "
            f"{dangling[:3]}. Either resolve placeholder targets or drop the "
            f"edge before it reaches GraphStore — these are what trigger "
            f"'FOREIGN KEY constraint failed'."
        )

    def test_build_code_graph_succeeds_and_is_non_empty(self, tmp_path) -> None:
        """The end-to-end contract: it must succeed AND actually store a graph.

        Guards both failure modes — the current hard error, and the equally
        useless 'success with an empty graph'.
        """
        import asyncio
        import os

        os.environ["SLM_DATA_DIR"] = str(tmp_path)
        from superlocalmemory.code_graph.config import CodeGraphConfig
        from superlocalmemory.code_graph.graph_store import GraphStore
        from superlocalmemory.code_graph.parser import CodeParser
        from superlocalmemory.code_graph.service import CodeGraphService

        repo = self._tiny_repo()
        cfg = CodeGraphConfig(enabled=True, repo_root=repo, db_path=tmp_path / "cg.db")
        svc = CodeGraphService(cfg)
        parser = CodeParser(cfg)
        nodes, edges, file_records = parser.parse_all(repo)
        store = GraphStore(svc.db)

        groups: dict[str, tuple[list, list, object]] = {
            fr.file_path: ([], [], fr) for fr in file_records
        }
        for n in nodes:
            if n.file_path in groups:
                groups[n.file_path][0].append(n)
        for e in edges:
            if e.file_path in groups:
                groups[e.file_path][1].append(e)

        # Must not raise FOREIGN KEY constraint failed.
        for fp, (ns, es, fr) in groups.items():
            store.store_file_nodes_edges(fp, ns, es, fr)

        assert svc.db.get_node_count() > 0, (
            "code graph stored zero nodes — build 'succeeded' but produced "
            "nothing, which is indistinguishable from broken to a user."
        )
        del asyncio  # imported for parity with the MCP path; unused here


# ─────────────────────────────────────────────────────────────────────────────
# P3 — issue #119: saving settings must not destroy credentials
# ─────────────────────────────────────────────────────────────────────────────
class TestP3ConfigSaveDoesNotWipeCredentials:
    """Reported by barrygfox. Mode C becomes unusable after any second save.

    The read endpoint deliberately redacts (SEC-L-01: returns netloc only, and
    never returns the key). The bug is that the WRITE path treats those redacted
    values as authoritative, so a no-op save persists a scheme-less host and a
    blank key.
    """

    def test_blank_api_key_on_save_preserves_stored_key(self) -> None:
        """A save that omits/blanks the key must mean 'unchanged', not 'clear'.

        Browsers never repopulate password inputs, so a blank key on submit is
        the NORMAL case, not an instruction to delete the credential.
        """
        import dataclasses

        from superlocalmemory.core.config import SLMConfig

        # LLMConfig is frozen (and therefore hashable) by design — build a new
        # instance rather than mutating, exactly as production code does.
        cfg = SLMConfig()
        cfg.llm = dataclasses.replace(
            cfg.llm,
            api_key="sk-test-EXISTING-do-not-wipe",
            api_base="https://provider.example.com/v1",
        )

        applied = _apply_settings_update(cfg, {"api_key": "", "provider": cfg.llm.provider})

        assert applied.llm.api_key == "sk-test-EXISTING-do-not-wipe", (
            "a blank api_key on save wiped the stored credential — this is the "
            "#119 data-loss defect"
        )

    def test_scheme_survives_read_modify_write_roundtrip(self) -> None:
        """Re-saving what the UI displayed must not strip https://.

        Step 2 of the reported repro: the pane pre-fills the host only, then a
        no-op save persists that scheme-less value.
        """
        import dataclasses
        from urllib.parse import urlparse

        from superlocalmemory.core.config import SLMConfig

        cfg = SLMConfig()
        cfg.llm = dataclasses.replace(
            cfg.llm, api_base="https://provider.example.com/v1"
        )

        displayed = urlparse(cfg.llm.api_base).netloc  # what GET /mode exposes
        applied = _apply_settings_update(cfg, {"endpoint": displayed, "api_key": ""})

        assert urlparse(applied.llm.api_base).scheme in ("http", "https"), (
            f"endpoint lost its scheme after a no-op save: "
            f"{applied.llm.api_base!r} — Mode C cannot connect without it"
        )
        assert applied.llm.api_base.rstrip("/").endswith("/v1"), (
            f"endpoint path was discarded: {applied.llm.api_base!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P4 — issue #112: trusted private-LAN plain-HTTP endpoints
# ─────────────────────────────────────────────────────────────────────────────
class TestP4TrustedPrivateLanEndpoints:
    """Reported by unfall103-debug (LXC/Docker on a private LAN, llama.cpp).

    Plain HTTP to a private-LAN reranker is currently refused outright, which
    disables reranking for every self-hosted local-AI deployment. The fix must
    make this possible WITHOUT silently allowing plain HTTP to the public
    internet.
    """

    def test_public_plain_http_is_still_refused_by_default(self) -> None:
        """Security floor — must not regress while fixing the LAN case."""
        from superlocalmemory.retrieval.remote_reranker import (
            validate_remote_reranker_config,
        )

        err = validate_remote_reranker_config("openai", "http://evil.example.com/v1/rerank")
        assert err, (
            "plain HTTP to a PUBLIC host must remain refused — relaxing this "
            "would send queries unencrypted across the internet"
        )

    def test_loopback_plain_http_allowed(self) -> None:
        """Existing behaviour that must keep working."""
        from superlocalmemory.retrieval.remote_reranker import (
            validate_remote_reranker_config,
        )

        assert not validate_remote_reranker_config(
            "openai", "http://127.0.0.1:8041/v1/rerank"
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://192.168.1.50:8041/v1/rerank",
            "http://10.0.0.7:8041/v1/rerank",
            "http://172.16.4.2:8041/v1/rerank",
        ],
    )
    def test_private_lan_plain_http_can_be_trusted(self, endpoint: str) -> None:
        """A private-LAN endpoint must be usable when the operator opts in.

        The mechanism is an implementation choice (config flag, allowlist, or
        RFC1918 detection) — this asserts only the outcome the reporter needs.
        Whatever the mechanism, it must be reachable from configuration.
        """
        from superlocalmemory.retrieval.remote_reranker import (
            validate_remote_reranker_config,
        )

        err = validate_remote_reranker_config("openai", endpoint)
        assert not err, (
            f"private-LAN endpoint {endpoint} rejected: {err!r}. Self-hosted "
            f"local-AI deployments (issue #112) cannot use reranking at all "
            f"until an operator-controlled trust path exists."
        )


def _apply_settings_update(cfg, payload: dict):
    """Adapter to the product's settings-save path.

    Implementer: point this at the real save routine used by the dashboard's
    POST handler. It exists so the assertions above test PRODUCT behaviour
    rather than a reimplementation. Do not weaken the assertions — wire this.
    """
    from superlocalmemory.server.routes import v3_api

    apply_fn = getattr(v3_api, "apply_settings_update", None)
    if apply_fn is None:
        pytest.fail(
            "No settings-update entry point found. Expose the dashboard's "
            "config-save logic as a testable function (e.g. "
            "v3_api.apply_settings_update(config, payload) -> config) so the "
            "#119 round-trip can be asserted against real product code."
        )
    return apply_fn(cfg, payload)

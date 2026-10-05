# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for ImportResolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.code_graph.config import CodeGraphConfig
from superlocalmemory.code_graph.models import EdgeKind, GraphEdge, GraphNode, NodeKind
from superlocalmemory.code_graph.resolver import ImportResolver, UnsupportedLanguageError


@pytest.fixture
def config() -> CodeGraphConfig:
    return CodeGraphConfig(enabled=True)


# ---------------------------------------------------------------------------
# Python import resolution
# ---------------------------------------------------------------------------

def test_resolve_python_relative_import(tmp_path: Path, config: CodeGraphConfig):
    """Resolve dotted path to .py file."""
    (tmp_path / "src" / "utils").mkdir(parents=True)
    (tmp_path / "src" / "utils" / "helpers.py").write_text("# helpers", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("src.utils.helpers", "main.py", "python")
    assert result is not None
    assert result == "src/utils/helpers.py"


def test_resolve_python_package_init(tmp_path: Path, config: CodeGraphConfig):
    """Resolve package import to __init__.py."""
    (tmp_path / "src" / "utils").mkdir(parents=True)
    (tmp_path / "src" / "utils" / "__init__.py").write_text("# pkg", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("src.utils", "main.py", "python")
    assert result is not None
    assert "__init__.py" in result


def test_resolve_python_external_package(tmp_path: Path, config: CodeGraphConfig):
    """External packages should return None."""
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("requests", "src/main.py", "python")
    assert result is None


# ---------------------------------------------------------------------------
# TypeScript import resolution
# ---------------------------------------------------------------------------

def test_resolve_ts_relative_import(tmp_path: Path, config: CodeGraphConfig):
    """Resolve relative TS import."""
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "validator.ts").write_text("// validator", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("./auth/validator", "src/main.ts", "typescript")
    assert result is not None
    assert "validator.ts" in result


def test_resolve_ts_index_file(tmp_path: Path, config: CodeGraphConfig):
    """Resolve directory import to index.ts."""
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "index.ts").write_text("// index", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("./auth", "src/main.ts", "typescript")
    assert result is not None
    assert "index.ts" in result


def test_resolve_ts_extension_priority(tmp_path: Path, config: CodeGraphConfig):
    """TS extension should be preferred over JS."""
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "utils.ts").write_text("// ts", encoding="utf-8")
    (tmp_path / "src" / "utils.js").write_text("// js", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("./utils", "src/main.ts", "typescript")
    assert result is not None
    assert result.endswith(".ts")


def test_resolve_ts_external_package(tmp_path: Path, config: CodeGraphConfig):
    """Bare package names should return None."""
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("express", "src/main.ts", "typescript")
    assert result is None


def test_resolve_ts_alias(tmp_path: Path, config: CodeGraphConfig):
    """Resolve @/ alias via tsconfig.json paths."""
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "validator.ts").write_text("// val", encoding="utf-8")
    tsconfig = {
        "compilerOptions": {
            "paths": {
                "@/*": ["src/*"]
            }
        }
    }
    (tmp_path / "tsconfig.json").write_text(json.dumps(tsconfig), encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)
    result = resolver.resolve("@/auth/validator", "src/main.ts", "typescript")
    assert result is not None
    assert "validator.ts" in result


# ---------------------------------------------------------------------------
# Symbol table
# ---------------------------------------------------------------------------

def test_build_symbol_table(config: CodeGraphConfig, tmp_path: Path):
    resolver = ImportResolver(tmp_path, config)
    nodes = [
        GraphNode(node_id="n1", name="authenticate", kind=NodeKind.METHOD,
                  qualified_name="a.py::Auth.authenticate", file_path="a.py"),
        GraphNode(node_id="n2", name="authenticate", kind=NodeKind.METHOD,
                  qualified_name="b.py::Other.authenticate", file_path="b.py"),
        GraphNode(node_id="n3", name="create_user", kind=NodeKind.FUNCTION,
                  qualified_name="c.py::create_user", file_path="c.py"),
    ]
    table = resolver.build_symbol_table(nodes)
    assert len(table["authenticate"]) == 2
    assert len(table["create_user"]) == 1


# ---------------------------------------------------------------------------
# Call target resolution
# ---------------------------------------------------------------------------

def test_resolve_call_targets_import_resolved(tmp_path: Path, config: CodeGraphConfig):
    """Import-resolved call should have confidence=1.0."""
    (tmp_path / "b.py").write_text("# b", encoding="utf-8")
    resolver = ImportResolver(tmp_path, config)

    nodes = [
        GraphNode(node_id="caller", name="foo", kind=NodeKind.FUNCTION,
                  qualified_name="a.py::foo", file_path="a.py"),
        GraphNode(node_id="target", name="bar", kind=NodeKind.FUNCTION,
                  qualified_name="b.py::bar", file_path="b.py"),
    ]
    edges = [
        GraphEdge(edge_id="e1", kind=EdgeKind.CALLS,
                  source_node_id="caller", target_node_id="__call__bar",
                  file_path="a.py", line=5),
    ]
    import_maps = {
        "a.py": {"bar": ("b", "bar")},
    }
    resolved = resolver.resolve_call_targets(nodes, edges, import_maps)
    assert len(resolved) == 1
    assert resolved[0].target_node_id == "target"
    assert resolved[0].confidence == 1.0


def test_resolve_call_targets_heuristic(tmp_path: Path, config: CodeGraphConfig):
    """Single global match should use heuristic confidence."""
    resolver = ImportResolver(tmp_path, config)

    nodes = [
        GraphNode(node_id="caller", name="foo", kind=NodeKind.FUNCTION,
                  qualified_name="a.py::foo", file_path="a.py"),
        GraphNode(node_id="target", name="bar", kind=NodeKind.FUNCTION,
                  qualified_name="c.py::bar", file_path="c.py"),
    ]
    edges = [
        GraphEdge(edge_id="e1", kind=EdgeKind.CALLS,
                  source_node_id="caller", target_node_id="__call__bar",
                  file_path="a.py", line=5),
    ]
    resolved = resolver.resolve_call_targets(nodes, edges, {})
    assert len(resolved) == 1
    assert resolved[0].confidence == config.heuristic_confidence


def test_resolve_call_targets_ambiguous(tmp_path: Path, config: CodeGraphConfig):
    """Multiple matches should pick closest with reduced confidence."""
    resolver = ImportResolver(tmp_path, config)

    nodes = [
        GraphNode(node_id="caller", name="foo", kind=NodeKind.FUNCTION,
                  qualified_name="src/a.py::foo", file_path="src/a.py"),
        GraphNode(node_id="t1", name="bar", kind=NodeKind.FUNCTION,
                  qualified_name="src/b.py::bar", file_path="src/b.py"),
        GraphNode(node_id="t2", name="bar", kind=NodeKind.FUNCTION,
                  qualified_name="lib/c.py::bar", file_path="lib/c.py"),
        GraphNode(node_id="t3", name="bar", kind=NodeKind.FUNCTION,
                  qualified_name="vendor/d.py::bar", file_path="vendor/d.py"),
    ]
    edges = [
        GraphEdge(edge_id="e1", kind=EdgeKind.CALLS,
                  source_node_id="caller", target_node_id="__call__bar",
                  file_path="src/a.py", line=5),
    ]
    resolved = resolver.resolve_call_targets(nodes, edges, {})
    assert len(resolved) == 1
    # Should pick src/b.py (closest to src/a.py)
    assert resolved[0].target_node_id == "t1"
    assert resolved[0].confidence == pytest.approx(config.heuristic_confidence * 0.8)


def test_resolve_call_targets_external_dropped(tmp_path: Path, config: CodeGraphConfig):
    """Calls with no matching symbol should be dropped."""
    resolver = ImportResolver(tmp_path, config)

    nodes = [
        GraphNode(node_id="caller", name="foo", kind=NodeKind.FUNCTION,
                  qualified_name="a.py::foo", file_path="a.py"),
    ]
    edges = [
        GraphEdge(edge_id="e1", kind=EdgeKind.CALLS,
                  source_node_id="caller", target_node_id="__call__external_func",
                  file_path="a.py", line=5),
    ]
    resolved = resolver.resolve_call_targets(nodes, edges, {})
    assert len(resolved) == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unsupported_language_raises(tmp_path: Path, config: CodeGraphConfig):
    resolver = ImportResolver(tmp_path, config)
    with pytest.raises(UnsupportedLanguageError):
        resolver.resolve("foo", "file.rs", "rust")


# ---------------------------------------------------------------------------
# Integration — resolver wired into parse_all (not isolation tests)
# ---------------------------------------------------------------------------

class TestResolverWiredToParseAll:
    """Verify that parse_all() runs the resolver pipeline and returns no
    dangling edges.  These tests exercise the production wire-up, not the
    resolver in isolation.
    """

    def _write_repo(self, d: Path, files: dict[str, str]) -> None:
        for name, content in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def test_single_file_no_dangling_after_parse_all(self, tmp_path: Path) -> None:
        """Single-file repo: every edge endpoint must be in the returned nodes."""
        from superlocalmemory.code_graph.parser import CodeParser

        self._write_repo(tmp_path, {
            "a.py": (
                "import os\n\n"
                "def helper():\n"
                "    return 42\n\n"
                "def caller():\n"
                "    return helper()\n"
            ),
        })
        cfg = CodeGraphConfig(enabled=True, repo_root=tmp_path)
        nodes, edges, _ = CodeParser(cfg).parse_all(tmp_path)
        node_ids = {n.node_id for n in nodes}
        dangling = [
            (e.source_node_id, e.target_node_id)
            for e in edges
            if e.source_node_id not in node_ids or e.target_node_id not in node_ids
        ]
        assert not dangling, f"dangling edges after parse_all: {dangling}"

    def test_cross_file_same_name_call_resolves(self, tmp_path: Path) -> None:
        """bar() defined in utils.py and called from main.py — Strategy 2 must
        resolve the call to the actual node (not drop it).
        """
        from superlocalmemory.code_graph.parser import CodeParser

        self._write_repo(tmp_path, {
            "utils.py": "def bar():\n    return 1\n",
            "main.py": (
                "from utils import bar\n\n"
                "def foo():\n"
                "    return bar()\n"
            ),
        })
        cfg = CodeGraphConfig(enabled=True, repo_root=tmp_path)
        nodes, edges, _ = CodeParser(cfg).parse_all(tmp_path)

        # All edges must have valid endpoints
        node_ids = {n.node_id for n in nodes}
        dangling = [e for e in edges
                    if e.source_node_id not in node_ids or e.target_node_id not in node_ids]
        assert not dangling, f"dangling edges: {[(e.source_node_id, e.target_node_id) for e in dangling]}"

        # The CALLS edge foo→bar must exist (not been silently dropped)
        calls_targets = {
            e.target_node_id for e in edges if e.kind == EdgeKind.CALLS
        }
        bar_nodes = {n.node_id for n in nodes if n.name == "bar"}
        assert calls_targets & bar_nodes, (
            "parse_all resolved 0 CALLS edges to the 'bar' function — "
            "resolver Strategy 2 (same-name, same-repo lookup) may be broken"
        )

    def test_external_library_calls_dropped(self, tmp_path: Path) -> None:
        """Calls to external libraries (requests.get) must be silently dropped,
        not propagated as dangling edges.
        """
        from superlocalmemory.code_graph.parser import CodeParser

        self._write_repo(tmp_path, {
            "a.py": (
                "import requests\n\n"
                "def fetch():\n"
                "    return requests.get('https://example.invalid')\n"
            ),
        })
        cfg = CodeGraphConfig(enabled=True, repo_root=tmp_path)
        nodes, edges, _ = CodeParser(cfg).parse_all(tmp_path)
        node_ids = {n.node_id for n in nodes}
        dangling = [e for e in edges
                    if e.source_node_id not in node_ids or e.target_node_id not in node_ids]
        assert not dangling, (
            f"external CALLS edges were not dropped: "
            f"{[(e.source_node_id, e.target_node_id) for e in dangling]}"
        )

    def test_imports_edges_with_unresolved_endpoints_dropped(
        self, tmp_path: Path
    ) -> None:
        """IMPORTS edges whose endpoints cannot be resolved must be dropped —
        they must NOT reach GraphStore as dangling endpoints.
        """
        from superlocalmemory.code_graph.parser import CodeParser

        self._write_repo(tmp_path, {
            "mod.py": (
                "import os\n"
                "import sys\n\n"
                "def run():\n"
                "    pass\n"
            ),
        })
        cfg = CodeGraphConfig(enabled=True, repo_root=tmp_path)
        nodes, edges, _ = CodeParser(cfg).parse_all(tmp_path)
        node_ids = {n.node_id for n in nodes}
        dangling = [e for e in edges
                    if e.source_node_id not in node_ids or e.target_node_id not in node_ids]
        assert not dangling, f"dangling IMPORTS edges survived: {dangling}"

    def test_parse_all_returns_at_least_one_node_per_file(
        self, tmp_path: Path
    ) -> None:
        """Smoke test: every .py file must produce at least the FILE node."""
        from superlocalmemory.code_graph.parser import CodeParser

        self._write_repo(tmp_path, {
            "a.py": "x = 1\n",
            "b.py": "y = 2\n",
        })
        cfg = CodeGraphConfig(enabled=True, repo_root=tmp_path)
        nodes, edges, file_records = CodeParser(cfg).parse_all(tmp_path)

        file_node_paths = {n.file_path for n in nodes if n.kind == NodeKind.FILE}
        assert "a.py" in file_node_paths
        assert "b.py" in file_node_paths
        assert len(file_records) == 2


# ---------------------------------------------------------------------------
# MUST-FIX 1: resolver wired into the incremental update path
# ---------------------------------------------------------------------------

try:
    import tree_sitter  # noqa: F401
    from tree_sitter_language_pack import get_parser as _ts_get_parser  # noqa: F401
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False


@pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter-language-pack not installed")
class TestIncrementalResolution:
    """Prove the resolver is wired into update_code_graph's parse path.

    Gate test: build a graph, edit a file, simulate incremental update,
    assert the CALLS edge for the edited file still exists and points at
    the correct target.  A test that only asserts "no exception" does NOT
    satisfy this gate.
    """

    def test_incremental_update_preserves_calls_edges(self, tmp_path: Path) -> None:
        """Build a 2-file graph.  Update one file.  CALLS edge to the other survives.

        Setup:
          a.py — defines foo() which calls bar()
          b.py — defines bar()

        After the full build, a CALLS edge foo→bar must exist.

        After incrementally updating a.py (adding a statement, keeping the call),
        the CALLS edge to bar() must still exist in the DB and target the SAME bar
        node (b.py was NOT re-parsed, so bar's node_id is unchanged).
        """
        import hashlib
        import time as _time

        from superlocalmemory.code_graph.config import CodeGraphConfig
        from superlocalmemory.code_graph.database import CodeGraphDatabase
        from superlocalmemory.code_graph.graph_store import GraphStore
        from superlocalmemory.code_graph.models import EdgeKind, FileRecord
        from superlocalmemory.code_graph.parser import CodeParser, _clean_and_resolve_edges

        repo = tmp_path / "repo"
        repo.mkdir()

        (repo / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        (repo / "a.py").write_text("def foo():\n    bar()\n", encoding="utf-8")

        cfg = CodeGraphConfig(enabled=True, repo_root=repo, db_path=tmp_path / "g.db")
        parser_instance = CodeParser(cfg)

        # ── Step 1: full build ─────────────────────────────────────────────
        nodes, edges, file_records = parser_instance.parse_all(repo)

        db = CodeGraphDatabase(tmp_path / "g.db")
        store = GraphStore(db)

        # Group by file and store
        file_groups: dict[str, tuple[list, list]] = {
            fr.file_path: ([], []) for fr in file_records
        }
        for n in nodes:
            if n.file_path in file_groups:
                file_groups[n.file_path][0].append(n)
        for e in edges:
            if e.file_path in file_groups:
                file_groups[e.file_path][1].append(e)
        # Adversarial order: a.py (caller) stored BEFORE b.py (callee).
        # This is the order that previously caused the CALLS edge to be
        # silently dropped.  commit_build_batch must produce correct results
        # in both directions.
        batch = [
            (fp, ns, es, next(r for r in file_records if r.file_path == fp))
            for fp, (ns, es) in file_groups.items()
        ]
        store.commit_build_batch(batch)

        # Confirm the full build produced a CALLS edge to bar()
        bar_after_build = next(
            (n for n in db.get_all_nodes() if n.name == "bar"), None
        )
        assert bar_after_build is not None, "full build must create a bar() node"

        calls_after_build = [
            e for e in db.get_all_edges()
            if e.kind == EdgeKind.CALLS and e.target_node_id == bar_after_build.node_id
        ]
        assert len(calls_after_build) >= 1, (
            "full build must produce a CALLS edge foo→bar via Strategy 3 "
            "(bar is globally unique, so the resolver picks it up)"
        )

        # ── Step 2: incremental update of a.py ────────────────────────────
        # Change the function body (adds a line) but KEEP the call to bar()
        new_src = b"def foo():\n    bar()\n    x = 1  # extra line\n"
        (repo / "a.py").write_bytes(new_src)

        # Simulate what update_code_graph does:
        #   parse_file → load DB nodes → _clean_and_resolve_edges → store
        file_nodes, file_edges, file_import_map = parser_instance.parse_file(
            Path("a.py"), new_src, "python"
        )
        db_nodes, _ = store.get_all_nodes_and_edges()
        resolution_universe = list(file_nodes) + [
            n for n in db_nodes if n.file_path != "a.py"
        ]
        resolved_edges = _clean_and_resolve_edges(
            resolution_universe,
            list(file_edges),
            {"a.py": file_import_map},
            repo,
            cfg,
        )
        fr_new = FileRecord(
            file_path="a.py",
            content_hash=hashlib.sha256(new_src).hexdigest(),
            mtime=(repo / "a.py").stat().st_mtime,
            language="python",
            node_count=len(file_nodes),
            edge_count=len(resolved_edges),
            last_indexed=_time.time(),
        )
        store.store_file_nodes_edges("a.py", file_nodes, resolved_edges, fr_new)

        # ── Step 3: verify CALLS edge survived ────────────────────────────
        # bar() in b.py was NOT re-parsed; its node_id is stable.
        bar_after_update = next(
            (n for n in db.get_all_nodes() if n.name == "bar"), None
        )
        assert bar_after_update is not None, "bar() node must still exist after update"
        assert bar_after_update.node_id == bar_after_build.node_id, (
            "b.py was not updated so bar's node_id must be unchanged"
        )

        calls_after_update = [
            e for e in db.get_all_edges()
            if e.kind == EdgeKind.CALLS and e.target_node_id == bar_after_update.node_id
        ]
        assert len(calls_after_update) >= 1, (
            f"CALLS edge foo→bar was lost after incremental update of a.py. "
            f"edges_before={len(calls_after_build)}, edges_after={len(calls_after_update)}. "
            f"The resolver is not wired correctly in the incremental path."
        )

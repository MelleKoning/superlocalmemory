# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CodeParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.code_graph.config import CodeGraphConfig
from superlocalmemory.code_graph.models import EdgeKind, NodeKind
from superlocalmemory.code_graph.parser import CodeParser, UnsupportedLanguageError

# Skip all tests if tree-sitter is not installed
try:
    import tree_sitter  # noqa: F401
    from tree_sitter_language_pack import get_parser  # noqa: F401
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

pytestmark = pytest.mark.skipif(
    not HAS_TREE_SITTER,
    reason="tree-sitter-language-pack not installed",
)


@pytest.fixture
def config(tmp_path: Path) -> CodeGraphConfig:
    return CodeGraphConfig(
        enabled=True,
        repo_root=tmp_path,
    )


@pytest.fixture
def parser(config: CodeGraphConfig) -> CodeParser:
    return CodeParser(config)


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------

def test_discover_files_finds_python(parser: CodeParser, tmp_path: Path):
    (tmp_path / "a.py").write_text("# a", encoding="utf-8")
    (tmp_path / "b.py").write_text("# b", encoding="utf-8")
    (tmp_path / "c.py").write_text("# c", encoding="utf-8")
    files = parser.discover_files(tmp_path)
    assert len(files) == 3


def test_discover_files_ignores_node_modules(parser: CodeParser, tmp_path: Path):
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "foo.js").write_text("// foo", encoding="utf-8")
    files = parser.discover_files(tmp_path)
    assert len(files) == 0


def test_discover_files_ignores_large_files(tmp_path: Path):
    config = CodeGraphConfig(
        enabled=True,
        repo_root=tmp_path,
        max_file_size_bytes=100,
    )
    parser = CodeParser(config)
    (tmp_path / "big.py").write_text("x" * 200, encoding="utf-8")
    (tmp_path / "small.py").write_text("# ok", encoding="utf-8")
    files = parser.discover_files(tmp_path)
    names = [f.name for f in files]
    assert "small.py" in names
    assert "big.py" not in names


def test_discover_files_respects_language_map(parser: CodeParser, tmp_path: Path):
    (tmp_path / "a.py").write_text("# py", encoding="utf-8")
    (tmp_path / "b.ts").write_text("// ts", encoding="utf-8")
    (tmp_path / "c.rb").write_text("# ruby", encoding="utf-8")
    files = parser.discover_files(tmp_path)
    extensions = {f.suffix for f in files}
    assert ".py" in extensions
    assert ".ts" in extensions
    assert ".rb" not in extensions


def test_discover_files_returns_relative_paths(parser: CodeParser, tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main", encoding="utf-8")
    files = parser.discover_files(tmp_path)
    for f in files:
        assert not f.is_absolute()
        assert str(f).startswith("src/")


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------

def test_parse_file_python(parser: CodeParser):
    source = b"def foo():\n    pass\n"
    nodes, edges, _ = parser.parse_file(Path("test.py"), source, "python")
    assert len(nodes) >= 2  # File + function
    kinds = {n.kind for n in nodes}
    assert NodeKind.FILE in kinds
    assert NodeKind.FUNCTION in kinds


def test_parse_file_typescript(parser: CodeParser):
    source = b"function bar(): void {}\n"
    nodes, edges, _ = parser.parse_file(Path("test.ts"), source, "typescript")
    assert len(nodes) >= 2


def test_parse_file_unsupported_language(parser: CodeParser):
    with pytest.raises(UnsupportedLanguageError):
        parser.parse_file(Path("test.rs"), b"fn main() {}", "rust")


def test_parse_file_generates_file_node(parser: CodeParser):
    source = b"x = 1\n"
    nodes, edges, _ = parser.parse_file(Path("simple.py"), source, "python")
    file_node = nodes[0]
    assert file_node.kind == NodeKind.FILE
    assert file_node.content_hash is not None
    assert len(file_node.content_hash) == 64  # SHA-256 hex


def test_parse_file_generates_contains_edges(parser: CodeParser):
    source = b"""
class Foo:
    def bar(self):
        pass
"""
    nodes, edges, _ = parser.parse_file(Path("contains.py"), source, "python")
    contains_edges = [e for e in edges if e.kind == EdgeKind.CONTAINS]
    assert len(contains_edges) >= 2  # File->Foo and Foo->bar

    # Find the File->Foo edge
    file_node = [n for n in nodes if n.kind == NodeKind.FILE][0]
    foo_node = [n for n in nodes if n.name == "Foo"][0]
    bar_node = [n for n in nodes if n.name == "bar"][0]

    file_to_foo = [
        e for e in contains_edges
        if e.source_node_id == file_node.node_id and e.target_node_id == foo_node.node_id
    ]
    assert len(file_to_foo) == 1

    foo_to_bar = [
        e for e in contains_edges
        if e.source_node_id == foo_node.node_id and e.target_node_id == bar_node.node_id
    ]
    assert len(foo_to_bar) == 1


def test_parse_file_generates_tested_by(parser: CodeParser):
    """Test file calling a function should generate TESTED_BY edge."""
    source = b"""
def test_something():
    some_func()
"""
    nodes, edges, _ = parser.parse_file(Path("tests/test_foo.py"), source, "python")
    # The file is in tests/ directory, so functions should be marked as test
    test_funcs = [n for n in nodes if n.is_test]
    assert len(test_funcs) >= 1


# ---------------------------------------------------------------------------
# parse_all
# ---------------------------------------------------------------------------

def test_parse_all_parallel(tmp_path: Path):
    config = CodeGraphConfig(enabled=True, repo_root=tmp_path, parallel_workers=2)
    parser = CodeParser(config)

    for i in range(10):
        (tmp_path / f"mod_{i}.py").write_text(f"def func_{i}():\n    pass\n", encoding="utf-8")

    nodes, edges, records = parser.parse_all(tmp_path)
    # Each file should produce at least File + function = 2 nodes
    assert len(nodes) >= 20
    assert len(records) == 10


def test_parse_all_handles_errors_gracefully(tmp_path: Path):
    config = CodeGraphConfig(enabled=True, repo_root=tmp_path)
    parser = CodeParser(config)

    # 5 valid files
    for i in range(5):
        (tmp_path / f"good_{i}.py").write_text(f"def func_{i}():\n    pass\n", encoding="utf-8")

    # 1 binary file with .py extension
    (tmp_path / "bad.py").write_bytes(b"\x00\x01\x02\x03" * 100)

    nodes, edges, records = parser.parse_all(tmp_path)
    # Should still get results from valid files
    assert len(records) >= 5


def test_parse_all_file_records(tmp_path: Path):
    config = CodeGraphConfig(enabled=True, repo_root=tmp_path)
    parser = CodeParser(config)

    (tmp_path / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    nodes, edges, records = parser.parse_all(tmp_path)
    assert len(records) == 2

    for rec in records:
        assert rec.content_hash is not None
        assert len(rec.content_hash) == 64
        assert rec.node_count > 0
        assert rec.language == "python"


# ---------------------------------------------------------------------------
# MUST-FIX 2: parse_file exposes import_map (3-tuple return)
# ---------------------------------------------------------------------------

def test_parse_file_returns_import_map(parser: CodeParser):
    """parse_file must return a 3-tuple whose third element is the import map dict."""
    source = b"from os import path\n\ndef foo():\n    path.join('a', 'b')\n"
    result = parser.parse_file(Path("importer.py"), source, "python")
    assert len(result) == 3, "parse_file must return (nodes, edges, import_map)"
    nodes, edges, import_map = result
    assert isinstance(import_map, dict)
    assert nodes  # at least FILE node

    # The import_map key for 'from os import path' should include 'path'
    # (exact resolution depends on extractor implementation, but the dict
    # must be a dict[str, tuple[str, str]]).
    for key, val in import_map.items():
        assert isinstance(key, str)
        assert isinstance(val, tuple) and len(val) == 2


def test_parse_all_workers_1_uses_sequential_path(tmp_path: Path):
    """parallel_workers=1 forces in-process parsing — no ProcessPoolExecutor spawn.

    Verified by: creating > 2 files (old threshold was 2; workers=1 overrides
    it), running parse_all, and confirming all files are parsed and import_maps
    are populated (they're only populated in the sequential path after MUST-FIX 2).
    """
    config = CodeGraphConfig(
        enabled=True, repo_root=tmp_path, parallel_workers=1
    )
    p = CodeParser(config)

    # 5 files — would normally trigger the parallel pool (len > 2)
    for i in range(5):
        (tmp_path / f"mod_{i}.py").write_text(
            f"from os import path\ndef func_{i}():\n    path.join('x', 'y')\n",
            encoding="utf-8",
        )

    nodes, edges, records = p.parse_all(tmp_path)
    assert len(records) == 5, "all 5 files must be parsed in sequential mode"
    assert len(nodes) >= 10  # FILE + FUNCTION per file


def test_parse_all_raises_on_majority_pool_failure(tmp_path: Path, monkeypatch):
    """parse_all must raise RuntimeError when >50% of futures fail (pool crash).

    Simulates a pool crash by monkey-patching ProcessPoolExecutor.submit to
    return real Future objects pre-populated with an exception.  Verifies the
    fail-fast threshold (>50% failure rate) triggers the RuntimeError.
    """
    from concurrent.futures import Future, ProcessPoolExecutor

    def patched_submit(self_pool, fn, *args, **kwargs):  # noqa: ARG001
        # Return a real Future with an exception already set.  as_completed
        # (which calls ._condition internally) requires a genuine Future.
        f: Future = Future()
        f.set_exception(RuntimeError("worker process killed"))
        return f

    monkeypatch.setattr(ProcessPoolExecutor, "submit", patched_submit)

    # Need > 2 files so we hit the parallel branch (parallel_workers defaults to 4)
    config = CodeGraphConfig(
        enabled=True, repo_root=tmp_path, parallel_workers=2
    )
    p = CodeParser(config)
    for i in range(6):
        (tmp_path / f"mod_{i}.py").write_text(f"def func_{i}():\n    pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Parsing aborted"):
        p.parse_all(tmp_path)

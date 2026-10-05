# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""SLM's own code calls no native LAPACK routine except the one it guards.

On macOS, NumPy, SciPy and PyTorch hand linear-algebra factorizations to
Apple's Accelerate LAPACK, which has reported out-of-bounds writes in
``dgeqrf`` (our 4.1.20 crash), ``dpotrf`` (scipy/scipy#26145) and a hang in
``dgesdd`` (numpy/numpy#32591). Vector norms and matrix products are BLAS,
not LAPACK, and stay allowed. Anything that factorizes a matrix must either
avoid LAPACK, as ``math/orthogonal.py`` does, or be listed here with a
Guard Malloc test of its own.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"

# numpy.linalg entries that do not call LAPACK.
_BLAS_ONLY = frozenset({"norm", "LinAlgError", "matmul", "vecdot", "matrix_transpose",
                        "outer", "tensordot", "trace", "diagonal", "vector_norm",
                        "matrix_norm", "multi_dot", "cross"})
_LAPACK_MODULES = ("numpy.linalg", "scipy.linalg", "scipy.sparse.linalg",
                   "torch.linalg", "sklearn.decomposition", "sklearn.linear_model")
_LAPACK_ATTRIBUTES = frozenset({"polyfit", "multivariate_normal", "lstsq", "pinv"})

# Reaches LAPACK through SciPy on purpose; guarded by
# tests/optimize/cache/test_boundary_fit_never_corrupts_memory.py.
_GUARDED = {("optimize/cache/boundary_store.py", "scipy.optimize")}


def _is_matrix_norm_call(node: ast.AST) -> bool:
    """``norm(a, ord)`` on a matrix runs an SVD for ord 2, -2 and 'nuc'."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    owner = node.func.value
    owner_name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
    if owner_name != "linalg" or node.func.attr != "norm":
        return False
    return len(node.args) > 1 or any(keyword.arg == "ord" for keyword in node.keywords)


def _findings(path: Path) -> list[str]:
    relative = path.relative_to(_PACKAGE).as_posix()
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if _is_matrix_norm_call(node):
            found.append(f"{relative}:{node.lineno} linalg.norm with ord (may run an SVD)")
        if isinstance(node, ast.Attribute):
            owner = node.value
            owner_name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
            if owner_name == "linalg" and node.attr not in _BLAS_ONLY:
                found.append(f"{relative}:{node.lineno} linalg.{node.attr}")
            elif node.attr in _LAPACK_ATTRIBUTES:
                found.append(f"{relative}:{node.lineno} .{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if (relative, module) in _GUARDED:
                continue
            names = {alias.name for alias in node.names}
            if module.startswith(_LAPACK_MODULES) or module.startswith("scipy.optimize") or (
                module in {"numpy", "scipy", "torch"} and "linalg" in names
            ):
                found.append(f"{relative}:{node.lineno} from {module} import {sorted(names)}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_LAPACK_MODULES):
                    found.append(f"{relative}:{node.lineno} import {alias.name}")
    return found


def test_no_source_file_calls_a_native_lapack_routine() -> None:
    findings = [hit for path in sorted(_PACKAGE.rglob("*.py")) for hit in _findings(path)]
    assert not findings, (
        "these calls reach native LAPACK (Accelerate on macOS); avoid LAPACK or add a "
        "Guard Malloc test and list the file in _GUARDED:\n" + "\n".join(findings)
    )


def test_the_guarded_exception_is_still_the_only_one() -> None:
    """If the guarded call goes away, the allow-list entry must go with it."""
    for relative, module in _GUARDED:
        source = (_PACKAGE / relative).read_text(encoding="utf-8")
        assert f"from {module} import" in source, f"{relative} no longer imports {module}"

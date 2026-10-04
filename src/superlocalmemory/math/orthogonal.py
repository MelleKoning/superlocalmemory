# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Seeded Haar-random orthogonal matrices without LAPACK.

The quantizers rotate every embedding by a fixed random orthogonal matrix.
It used to come from ``np.linalg.qr``. On macOS that call reaches
Accelerate's ``dgeqrf``, which writes outside its buffers for square inputs
of roughly 576..1000 rows — the 768-dimension default sits inside that range.
The damage is silent at the call and crashes the process later, anywhere.

This module computes the same matrix with Householder reflections in plain
numpy (BLAS level-2 products only): ``Q`` of ``H = QR`` with the Mezzadri
sign correction ``diag(R) > 0``, which makes ``Q`` unique and Haar-uniform
on O(d). It is built once per dimension and persisted by the callers, so the
extra cost over LAPACK (well under a second at d=768) is paid once.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mezzadri_q(matrix: NDArray) -> NDArray[np.float64]:
    """Orthogonal factor of a square ``matrix = QR`` with ``diag(R) >= 0``.

    Equal, to rounding, to ``Q @ diag(sign(diag(R)))`` from ``np.linalg.qr``.
    The input is not modified.
    """
    r = np.array(matrix, dtype=np.float64, copy=True)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {r.shape}")
    n = r.shape[0]

    # Forward pass: reduce r to upper-triangular R, keeping each reflector.
    reflectors: list[NDArray[np.float64] | None] = []
    for k in range(n):
        column = r[k:, k]
        norm = float(np.linalg.norm(column))
        if norm == 0.0:
            reflectors.append(None)
            continue
        v = column.copy()
        v[0] += norm if v[0] >= 0.0 else -norm
        v /= np.linalg.norm(v)
        r[k:, k:] -= 2.0 * np.outer(v, v @ r[k:, k:])
        reflectors.append(v)

    # Backward accumulation: Q = H_0 H_1 ... H_{n-1} applied to the identity.
    q = np.eye(n)
    for k in range(n - 1, -1, -1):
        v = reflectors[k]
        if v is None:
            continue
        q[k:, :] -= 2.0 * np.outer(v, v @ q[k:, :])

    # Mezzadri: flip each column whose R diagonal came out negative.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def haar_orthogonal(dimension: int, seed: int) -> NDArray[np.float64]:
    """The seeded rotation the quantizers use: Mezzadri QR of a Gaussian."""
    if dimension < 1:
        raise ValueError(f"dimension must be positive, got {dimension}")
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension))
    return mezzadri_q(gaussian)

# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Creating the quantization rotation matrix must not corrupt process memory.

On macOS, ``np.linalg.qr`` on a square matrix of roughly 576..1000 rows calls
Accelerate's ``dgeqrf``, which writes outside its buffers. A 768-dimension
encoder hit exactly that range on first use, and the damaged heap crashed the
process later at an unrelated point — often at interpreter shutdown. The
rotation is now built without LAPACK; these tests pin that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.core.config import PolarQuantConfig
from superlocalmemory.math.orthogonal import haar_orthogonal
from superlocalmemory.math.polar_quant import PolarQuantEncoder
from superlocalmemory.math.turbo_quant import TurboQuantEncoder
from tests.helpers.native_guard import (
    GUARD_MALLOC_AVAILABLE,
    SKIP_REASON,
    assert_no_stray_write,
    run_under_guard_malloc,
)


def _lapack_qr_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_args, **_kwargs):
        raise AssertionError("rotation generation must not call LAPACK QR")

    monkeypatch.setattr(np.linalg, "qr", _refuse)


@pytest.mark.parametrize("encoder_cls", [PolarQuantEncoder, TurboQuantEncoder])
def test_a_new_rotation_does_not_call_lapack_qr(
    encoder_cls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lapack_qr_is_refused(monkeypatch)
    config = PolarQuantConfig(
        dimension=64,
        rotation_matrix_path=str(tmp_path / "rotation.npy"),
        seed=42,
    )
    encoder = encoder_cls(config)
    rotation = encoder._S
    assert rotation.shape == (64, 64)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(64), atol=1e-12)


@pytest.mark.parametrize("dimension", [8, 32, 128])
def test_the_rotation_is_the_same_matrix_mezzadri_qr_gives(dimension: int) -> None:
    """Same seed, same matrix as before: Q of H = QR with diag(R) > 0."""
    rng = np.random.default_rng(7)
    h = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(h)  # sizes outside the faulting range
    expected = q @ np.diag(np.sign(np.diag(r)))

    np.testing.assert_allclose(haar_orthogonal(dimension, seed=7), expected, atol=1e-10)


def test_a_768_rotation_is_orthogonal_and_repeatable() -> None:
    first = haar_orthogonal(768, seed=42)
    second = haar_orthogonal(768, seed=42)
    assert np.array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(768), atol=1e-10)
    assert not np.array_equal(first, haar_orthogonal(768, seed=43))


@pytest.mark.skipif(not GUARD_MALLOC_AVAILABLE, reason=SKIP_REASON)
def test_creating_768_dimension_encoders_makes_no_stray_memory_write(tmp_path: Path) -> None:
    done = run_under_guard_malloc(
        f"""
        from superlocalmemory.core.config import PolarQuantConfig
        from superlocalmemory.math.polar_quant import PolarQuantEncoder
        from superlocalmemory.math.turbo_quant import TurboQuantEncoder
        for cls, name in ((PolarQuantEncoder, "polar"), (TurboQuantEncoder, "turbo")):
            cls(PolarQuantConfig(
                dimension=768,
                rotation_matrix_path={str(tmp_path)!r} + "/" + name + ".npy",
                seed=42,
            ))
        print("created")
        """,
        tmp_path,
    )
    assert_no_stray_write(done, "created")

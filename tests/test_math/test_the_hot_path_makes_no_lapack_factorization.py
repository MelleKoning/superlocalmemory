# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Saving, recalling and maintenance reach no Accelerate LAPACK factorization.

The source scan in ``test_no_native_lapack_in_source.py`` sees only SLM's own
calls. This test watches the native side as well: it builds a small library
that counts every call into Accelerate's matrix factorizations (QR, Cholesky,
LU, SVD, eigen and least-squares solvers, in all three of Accelerate's calling
conventions), loads it into a child interpreter, and runs the paths a running
SLM exercises. Anything NumPy, SciPy or NetworkX does on our behalf is counted
too. The count must stay at zero; matrix products and norms are BLAS and are
not counted.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.helpers.native_guard import FAULT_TO_EXIT_CODE

_FRAMEWORK = Path("/System/Library/Frameworks/Accelerate.framework")
_ACCELERATE = _FRAMEWORK / "Accelerate"  # served from the dyld shared cache, not disk
_SRC = Path(__file__).resolve().parents[2] / "src"
_ROUTINES = (
    "geqrf", "geqp3", "gelqf", "orgqr", "ormqr", "ungqr", "unmqr",
    "potrf", "potrs", "potri", "posv", "getrf", "getrs", "getri", "gesv",
    "gesdd", "gesvd", "gelsd", "gelss", "gelsy", "gels",
    "syevd", "syev", "syevr", "heevd", "heev", "geev", "gees",
    "sytrf", "sysv", "hetrf", "hesv", "trtrs", "trtri", "gehrd", "hseqr",
)
_FORMS = ("{}$NEWLAPACK$ILP64", "{}$NEWLAPACK", "{}_")

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("clang") is None or not _FRAMEWORK.is_dir(),
    reason="counts calls into Apple Accelerate; needs macOS and clang",
)


def _exported_symbols() -> list[str]:
    accelerate = ctypes.CDLL(str(_ACCELERATE))
    found = []
    for routine in _ROUTINES:
        for prefix in "sdcz":
            for form in _FORMS:
                name = form.format(prefix + routine)
                try:
                    accelerate[name]
                except AttributeError:
                    continue
                found.append(name)
    return found


def _build_counter(symbols: list[str], where: Path) -> Path:
    """Each counted symbol is interposed by a stub that bumps a counter and
    tail-branches to Accelerate, so arguments and results pass untouched."""
    assembly = [".section __TEXT,__text,regular,pure_instructions", ".p2align 2"]
    for index, name in enumerate(symbols):
        assembly += [
            f"_slm_count_{index}:",
            "    adrp x16, _slm_counts@PAGE",
            "    add x16, x16, _slm_counts@PAGEOFF",
            f"    mov x17, #{index * 8}",
            "    add x16, x16, x17",
            "    mov x17, #1",
            "    ldadd x17, x17, [x16]",
            f'    b "_{name}"',
        ]
    assembly += [".section __DATA,__interpose", ".p2align 3"]
    for index, name in enumerate(symbols):
        assembly += [f"    .quad _slm_count_{index}", f'    .quad "_{name}"']
    (where / "counter.s").write_text("\n".join(assembly) + "\n")
    (where / "counter.c").write_text(
        f"#include <stdint.h>\nuint64_t slm_counts[{len(symbols)}];\n"
    )
    library = where / "libslmcounter.dylib"
    subprocess.run(
        ["clang", "-mcpu=apple-m1", "-dynamiclib", "-O1", "-o", str(library),
         str(where / "counter.c"), str(where / "counter.s"), "-framework", "Accelerate"],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return library


_WORKLOAD = """
import ctypes, hashlib, json, sys
from pathlib import Path
import numpy as np
counts = (ctypes.c_uint64 * len(SYMBOLS)).in_dll(ctypes.CDLL(LIBRARY), "slm_counts")
from unittest.mock import patch
import networkx as nx
from networkx.algorithms.community import louvain_communities
from superlocalmemory.core.config import PolarQuantConfig, SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.graph_metrics import compute_graph_metrics
from superlocalmemory.core.maintenance import run_maintenance
from superlocalmemory.math.polar_quant import PolarQuantEncoder
from superlocalmemory.math.turbo_quant import TurboQuantEncoder
from superlocalmemory.optimize.cache.boundary_store import _fit_logistic_mle
from superlocalmemory.storage.models import Mode

class Embedder:
    is_available = True
    model_name = "deterministic-768"
    dimension = 768
    def embed(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        vector = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
        return (vector / np.linalg.norm(vector)).tolist()
    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]
    def compute_fisher_params(self, vector):
        return [0.0] * len(vector), [1.0] * len(vector)
    def close(self):
        return None

baseline = list(counts)  # NumPy's own import-time self-check is not ours
config = SLMConfig.for_mode(Mode.A, base_dir=Path(DATA))
config.retrieval.use_cross_encoder = False
engine = MemoryEngine(config)
with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=Embedder()):
    engine.initialize()
people = ["Alice", "Bob", "Project Atlas", "the billing service", "the London office"]
for i in range(24):
    engine.store(f"Note {i}: {people[i % 5]} approved control {i} with {people[(i + 2) % 5]}.")
for i in range(12):
    engine.recall(f"What did {people[i % 5]} approve about control {i}?", limit=5)
run_maintenance(engine.db, config, embedder=engine.embedder)
compute_graph_metrics(engine.db, engine.profile_id)
engine.close()
for cls, name in ((PolarQuantEncoder, "polar"), (TurboQuantEncoder, "turbo")):
    encoder = cls(PolarQuantConfig(dimension=768, rotation_matrix_path=DATA + "/" + name + ".npy", seed=7))
    encoder.encode(np.ones(768) / np.sqrt(768.0))
graph = nx.gnm_random_graph(300, 1500, seed=3, directed=True)
nx.pagerank(graph)
louvain_communities(graph.to_undirected(), seed=3)
lapack_before_fit = list(counts)
_fit_logistic_mle([(0.6, 0), (0.7, 0), (0.9, 1), (0.95, 1)], 0.95, 10.0)
after = list(counts)
print(json.dumps({
    "workload": {SYMBOLS[i]: after[i] - baseline[i] for i in range(len(SYMBOLS))
                 if lapack_before_fit[i] != baseline[i]},
    "boundary_fit": {SYMBOLS[i]: after[i] - lapack_before_fit[i] for i in range(len(SYMBOLS))
                     if after[i] != lapack_before_fit[i]},
}))
"""


def test_saving_recalling_and_maintenance_reach_no_lapack_factorization(tmp_path: Path) -> None:
    symbols = _exported_symbols()
    assert any(name.startswith("dgeqrf") for name in symbols)
    library = _build_counter(symbols, tmp_path)
    (tmp_path / "data").mkdir()
    script = FAULT_TO_EXIT_CODE + (
        f"SYMBOLS = {symbols!r}\nLIBRARY = {str(library)!r}\nDATA = {str(tmp_path / 'data')!r}\n"
        + textwrap.dedent(_WORKLOAD)
    )
    env = {
        **os.environ,
        "DYLD_INSERT_LIBRARIES": str(library),
        "PYTHONPATH": str(_SRC),
        "HOME": str(tmp_path / "home"),
        "SLM_DATA_DIR": str(tmp_path / "data"),
    }
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])

    assert report["workload"] == {}, f"LAPACK factorizations on the hot path: {report['workload']}"
    # The counter is live: the one guarded LAPACK user is seen, and only it.
    assert set(report["boundary_fit"]) <= {"dpotrf$NEWLAPACK", "dtrtrs$NEWLAPACK"}
    assert report["boundary_fit"].get("dpotrf$NEWLAPACK", 0) > 0

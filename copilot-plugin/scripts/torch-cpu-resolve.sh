#!/usr/bin/env bash
# torch-cpu-resolve.sh — should this install force a CPU-only torch wheel?
#
# PyPI ships CPU-only torch wheels for macOS and Windows, but on Linux the
# plain `torch==X.Y.Z` wheel pulls the CUDA runtime plus ~18 nvidia-* packages
# (triton included) — about 2.6 GiB on a host that may have no GPU at all
# (reference: https://docs.astral.sh/uv/guides/integration/pytorch/, which
# documents the same PyPI default for `uv pip`; plain pip resolves the same
# index). The official CPU-only index is documented at
# https://pytorch.org/get-started/locally/ as:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Sourced by ensure-venv.sh (never run directly). Pure decision logic — makes
# no network call and runs no pip — so it is unit-testable without the 2 GiB
# download it exists to avoid (tests/test_plugin_src/test_torch_cpu_resolve.sh).
#
# Environment (all optional):
#   SLM_TORCH_BACKEND   cpu  — force the CPU index regardless of GPU detection
#                        <anything else, e.g. cuda/cu124/auto> — leave pip's
#                        default resolution alone; the user has already decided
#   PIP_INDEX_URL / PIP_EXTRA_INDEX_URL — if the user already points pip at
#                        their own index, this never overrides it
#
# Safe under `set -euo pipefail`: every variable read has a default.

TORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"

# torch_cpu_should_force — exit 0 (true, in bash's backwards convention) when
# this host should get the CPU-only wheel instead of whatever pip would
# otherwise pick. False (exit 1) covers: not Linux, a GPU is visible, or the
# user already steered pip (their own index) or torch (SLM_TORCH_BACKEND set
# to anything but "cpu") themselves.
torch_cpu_should_force() {
    local platform
    platform="$(uname -s 2>/dev/null || echo unknown)"
    if [ "${platform}" != "Linux" ]; then
        # macOS (Darwin) already gets CPU-only wheels from plain PyPI; Windows
        # too. Nothing to override there.
        return 1
    fi

    case "${SLM_TORCH_BACKEND:-}" in
        cpu)
            return 0
            ;;
        "")
            : # unset — fall through to auto-detection below
            ;;
        *)
            # cuda / cuXXX / auto / anything else: the user already decided.
            return 1
            ;;
    esac

    if [ -n "${PIP_INDEX_URL:-}" ] || [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
        # The user already pointed pip somewhere specific. Do not fight it.
        return 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    if [ -e /proc/driver/nvidia/version ] || [ -e /dev/nvidia0 ]; then
        return 1
    fi

    return 0
}

# torch_cpu_pin <pin_file> — print the "torch==X.Y.Z" line from pin_file.
# Returns 1 (and prints nothing) when the file is missing or has no such
# line: fail-open, so a missing/stale pin file can never block an install
# that would otherwise succeed — it just means pip falls back to its
# ordinary resolution (the CUDA wheel on Linux) for this run.
torch_cpu_pin() {
    local pin_file="$1"
    if [ ! -f "${pin_file}" ]; then
        return 1
    fi
    local line
    line="$(grep -m1 '^torch==' "${pin_file}" 2>/dev/null || true)"
    if [ -z "${line}" ]; then
        return 1
    fi
    printf '%s\n' "${line}"
}

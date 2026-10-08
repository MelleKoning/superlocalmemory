#!/usr/bin/env bash
# ensure-venv.sh — SuperLocalMemory plugin venv bootstrap
#
# Called by Claude Code SessionStart hook. Idempotent: fast-path exits in <100ms
# on repeat invocations when requirements.txt is unchanged.
#
# Environment (set by Claude Code plugin runtime):
#   CLAUDE_PLUGIN_ROOT  — plugin installation dir (ephemeral, contains scripts/ + requirements.txt)
#   CLAUDE_PLUGIN_DATA  — persistent data dir (venv lives here, survives plugin updates)
#
# Exit codes: 0 = venv ready, or not needed   non-0 = failure (logged to stderr)
# All output goes to stderr only (stdout reserved for MCP stdio protocol).

set -euo pipefail

# ---------------------------------------------------------------------------
# :? guard — fail loudly if required env vars are unset or empty
# ---------------------------------------------------------------------------
: "${CLAUDE_PLUGIN_ROOT:?CLAUDE_PLUGIN_ROOT must be set (plugin installation directory)}"
: "${CLAUDE_PLUGIN_DATA:?CLAUDE_PLUGIN_DATA must be set (plugin persistent data directory)}"

# ---------------------------------------------------------------------------
# Redirect all output to stderr (MCP uses stdout for protocol messages)
# ---------------------------------------------------------------------------
exec 1>&2

# ---------------------------------------------------------------------------
# Only build a venv that will be used.
#
# slm-launch prefers an slm that is already installed, and SLM_LAUNCHER can name
# one outright. In those cases the plugin venv is never run, so neither its
# Python requirement nor its install cost should land on this session. The
# question is answered by slm-resolve.sh, the same code the launcher and the
# hooks use, so the three can never disagree about which slm is in play.
# ---------------------------------------------------------------------------
# shellcheck source=slm-resolve.sh
. "$(dirname "${BASH_SOURCE[0]}")/slm-resolve.sh"
if ! slm_choose; then
    # A venv cannot fix a launcher that is configured to use something else.
    # Report it, and leave the session alone: the MCP server reports it again.
    echo "SLM plugin: venv not needed — ${SLM_RESOLVE_ERROR}" >&2
    exit 0
fi
if [ "${SLM_CHOICE}" != "plugin" ]; then
    echo "SLM plugin: venv not needed — using ${SLM_CHOICE_BIN} (${SLM_CHOICE})." >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Python >= 3.12 guard
# ---------------------------------------------------------------------------
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    PY_VER=$(python3 --version 2>&1 || echo "unknown")
    echo "ERROR: SuperLocalMemory plugin requires Python >= 3.12, found: ${PY_VER}" >&2
    echo "Install Python 3.12+ and ensure it is first on PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REQ="${CLAUDE_PLUGIN_ROOT}/requirements.txt"
TORCH_CPU_PIN_FILE="${CLAUDE_PLUGIN_ROOT}/requirements-cpu-torch.txt"
VENV="${CLAUDE_PLUGIN_DATA}/venv"
SENTINEL="${CLAUDE_PLUGIN_DATA}/.venv-reqs.sha256"
VENV_TMP="${CLAUDE_PLUGIN_DATA}/venv.tmp"

# ---------------------------------------------------------------------------
# Compute sha256 of requirements.txt, and of the CPU-torch pin file when it
# exists (a pin bump alone must also trigger a rebuild).
# (cross-platform: prefer sha256sum, fall back to shasum -a 256)
# ---------------------------------------------------------------------------
_sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

hash_req() {
    local combined
    combined="$(_sha256_of "${REQ}")"
    if [ -f "${TORCH_CPU_PIN_FILE}" ]; then
        combined="${combined}:$(_sha256_of "${TORCH_CPU_PIN_FILE}")"
    fi
    # Fold the (possibly combined) digest down to one hash again so the
    # sentinel stays a single opaque token either way.
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "${combined}" | sha256sum | awk '{print $1}'
    else
        printf '%s' "${combined}" | shasum -a 256 | awk '{print $1}'
    fi
}

NEW_HASH=$(hash_req)

# ---------------------------------------------------------------------------
# Fast-path: venv python exists AND sentinel matches current requirements hash
# venv/bin/python3 (or python) is always present after `python3 -m venv`;
# the sentinel guards against stale requirements.
# In production, superlocalmemory also installs venv/bin/slm; both checks pass.
# ---------------------------------------------------------------------------
VENV_PYTHON="${VENV}/bin/python3"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="${VENV}/bin/python"
fi
if [ -x "${VENV_PYTHON}" ] && [ -f "${SENTINEL}" ] && [ "$(cat "${SENTINEL}")" = "${NEW_HASH}" ]; then
    echo "SLM plugin: venv up-to-date (sha256=${NEW_HASH:0:12}…), skipping install." >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Rebuild venv atomically: install to venv.tmp, rename to venv, write sentinel LAST
# ---------------------------------------------------------------------------
echo "SLM plugin: bootstrapping Python venv at ${VENV} …" >&2
echo "  requirements: ${REQ}" >&2
echo "  python3: $(python3 --version 2>&1)" >&2

# Clean up any partial previous attempt
rm -rf "${VENV_TMP}"

# Create fresh venv
python3 -m venv "${VENV_TMP}"

# Upgrade pip first (prefer binary to avoid source builds)
"${VENV_TMP}/bin/pip" install --upgrade pip --prefer-binary --quiet

# ---------------------------------------------------------------------------
# CPU-only torch on CPU-only Linux. Plain PyPI resolution of the
# `torch==X.Y.Z` requirement that requirements.txt pulls in transitively
# installs the CUDA build on Linux (torch + triton + ~18 nvidia-* packages,
# about 2.6 GiB) even when there is no GPU to use it. Pre-installing the
# pinned version from the official CPU index first means the later
# `pip install -r requirements.txt` below finds it already satisfied and
# never touches the CUDA wheels. macOS, Windows, a host with a GPU, and a
# user who already set SLM_TORCH_BACKEND or their own pip index are all left
# exactly as before (see torch-cpu-resolve.sh).
# shellcheck source=torch-cpu-resolve.sh
. "$(dirname "${BASH_SOURCE[0]}")/torch-cpu-resolve.sh"
if torch_cpu_should_force; then
    if TORCH_PIN="$(torch_cpu_pin "${TORCH_CPU_PIN_FILE}")"; then
        echo "SLM plugin: Linux, no GPU detected — installing ${TORCH_PIN} from ${TORCH_CPU_INDEX_URL} (set SLM_TORCH_BACKEND=cuda to opt out)." >&2
        "${VENV_TMP}/bin/pip" install \
            --require-virtualenv \
            --prefer-binary \
            --quiet \
            --index-url "${TORCH_CPU_INDEX_URL}" \
            "${TORCH_PIN}"
    fi
fi

# Install requirements
"${VENV_TMP}/bin/pip" install \
    --require-virtualenv \
    --prefer-binary \
    --quiet \
    -r "${REQ}"

echo "SLM plugin: install complete, activating venv." >&2

# Atomic swap: remove old venv (if any), rename tmp into place
rm -rf "${VENV}"
mv "${VENV_TMP}" "${VENV}"

# Write sentinel LAST — guarantees that a crash before this line triggers rebuild
echo "${NEW_HASH}" > "${SENTINEL}"

echo "SLM plugin: venv ready at ${VENV}/bin/slm" >&2

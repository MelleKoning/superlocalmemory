#!/usr/bin/env bash
# slm-resolve.sh — which SuperLocalMemory install does this plugin use?
#
# Sourced, never run. ONE answer to that question, shared by every script that
# needs it:
#   slm-launch      the MCP server
#   slm-run         every lifecycle hook
#   ensure-venv.sh  the plugin venv bootstrap (it only builds what will be used)
# Before this file each of them decided on its own, and they disagreed: the
# launcher fell back to the plugin venv, the hooks only ever tried a bare `slm`,
# and the bootstrap built a venv even when nothing would run it.
#
# Environment (all optional):
#   SLM_LAUNCHER       auto (default) — prefer an installed slm, else the plugin venv
#                      system         — require the slm on PATH
#                      plugin         — require the plugin's own venv
#                      <path>         — an explicit slm binary (leading ~ expanded)
#   CLAUDE_PLUGIN_DATA where a plugin-owned venv lives, when there is one
#
# After slm_choose returns 0:
#   SLM_CHOICE        system | plugin | explicit
#   SLM_CHOICE_BIN    the binary that choice names
#   SLM_CHOICE_READY  1 when SLM_CHOICE_BIN is executable now, else 0
#                     (a plugin venv that SessionStart has not built yet is 0)
# After slm_choose returns 1:
#   SLM_RESOLVE_ERROR what was asked for and why it cannot be honoured
#
# Safe under `set -euo pipefail`: every variable read has a default.

# Expand a leading ~ WITHOUT eval — eval on an environment value is a
# command-injection foot-gun. Only tilde expansion; pass expanded paths for vars.
_slm_expand_tilde() {
    case "$1" in
        "~"*) printf '%s' "${HOME}${1#\~}" ;;
        *)    printf '%s' "$1" ;;
    esac
}

# Where the plugin venv keeps slm. Empty when the host gives the plugin no data
# directory (only Claude Code sets CLAUDE_PLUGIN_DATA). A venv built on Windows
# keeps entry points in Scripts\ with an .exe suffix, and Git Bash is where these
# scripts run there.
#
# _slm_plugin_bin_var sets SLM_PLUGIN_BIN without a subshell (hooks run this on
# every tool call); slm_plugin_bin prints it, for callers that want a value.
_slm_plugin_bin_var() {
    SLM_PLUGIN_BIN=""
    [ -n "${CLAUDE_PLUGIN_DATA:-}" ] || return 0
    case "${OSTYPE:-}" in
        msys*|cygwin*|win32*) SLM_PLUGIN_BIN="${CLAUDE_PLUGIN_DATA}/venv/Scripts/slm.exe" ;;
        *)                    SLM_PLUGIN_BIN="${CLAUDE_PLUGIN_DATA}/venv/bin/slm" ;;
    esac
}

slm_plugin_bin() {
    _slm_plugin_bin_var
    printf '%s' "${SLM_PLUGIN_BIN}"
}

# The message for "nothing to run", naming both places that were looked in.
slm_not_found_message() {
    local _pbin
    _pbin="$(slm_plugin_bin)"
    printf '%s\n' "SLM plugin: no SuperLocalMemory found."
    printf '%s\n' "  Looked for: 'slm' on PATH, and ${_pbin:-a plugin venv (this host sets no plugin data dir)}."
    printf '%s'   "  Install it with:  pipx install superlocalmemory"
}

_slm_set_choice() {
    SLM_CHOICE="$1"
    SLM_CHOICE_BIN="$2"
    if [ -n "$2" ] && [ -x "$2" ]; then SLM_CHOICE_READY=1; else SLM_CHOICE_READY=0; fi
}

slm_choose() {
    SLM_CHOICE=""
    SLM_CHOICE_BIN=""
    SLM_CHOICE_READY=0
    SLM_RESOLVE_ERROR=""
    local _pbin _sys
    _slm_plugin_bin_var
    _pbin="${SLM_PLUGIN_BIN}"

    case "${SLM_LAUNCHER:-auto}" in
        auto)
            # 1. An slm already on PATH — the pip, pipx or npm install the user
            #    has. The common case, and the one that must not be forked.
            if _sys="$(command -v slm 2>/dev/null)" && [ -n "${_sys}" ]; then
                _slm_set_choice system "${_sys}"
                return 0
            fi
            # 2. No system install: the plugin's own venv, if this host gives the
            #    plugin somewhere to keep one. It may not be built yet.
            if [ -n "${_pbin}" ]; then
                _slm_set_choice plugin "${_pbin}"
                return 0
            fi
            # 3. Neither.
            SLM_RESOLVE_ERROR="$(slm_not_found_message)"
            return 1
            ;;
        system)
            if _sys="$(command -v slm 2>/dev/null)" && [ -n "${_sys}" ]; then
                _slm_set_choice system "${_sys}"
                return 0
            fi
            SLM_RESOLVE_ERROR="SLM_LAUNCHER=system but no 'slm' on PATH."
            return 1
            ;;
        plugin)
            if [ -z "${_pbin}" ]; then
                SLM_RESOLVE_ERROR="SLM_LAUNCHER=plugin but this host sets no CLAUDE_PLUGIN_DATA."
                return 1
            fi
            _slm_set_choice plugin "${_pbin}"
            return 0
            ;;
        *)
            _sys="$(_slm_expand_tilde "${SLM_LAUNCHER}")"
            if [ ! -x "${_sys}" ]; then
                SLM_RESOLVE_ERROR="SLM_LAUNCHER is not an executable slm binary: ${_sys}"
                return 1
            fi
            _slm_set_choice explicit "${_sys}"
            return 0
            ;;
    esac
}

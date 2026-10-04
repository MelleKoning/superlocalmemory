# B3 (3.7.9): the marketplace plugin must bundle the FULL auto-memory hook
# suite — a plugin-only install previously shipped only a minimal SessionStart,
# so external users got degraded memory (no checkpoint/topic-shift/pre-web/stop).
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

REQUIRED_EVENTS = {"SessionStart", "PostToolUse", "UserPromptSubmit", "Stop", "PreToolUse"}
# the value-add hooks a plugin-only install must have (were missing before)
REQUIRED_CMDS = {
    "slm hook mandate", "slm hook start", "slm hook checkpoint",
    "slm hook post_tool_outcome", "slm hook user_prompt_rehash",
    "slm hook topic_shift", "slm hook stop", "slm hook stop_outcome",
    "slm hook before_web",
}


# 4.1.20 (#141): plugin hooks run slm through scripts/slm-run, which finds the
# same slm the MCP launcher uses. Normalise that back to the slm command it runs.
_SLM_RUN = '"${CLAUDE_PLUGIN_ROOT}/scripts/slm-run" '


def _slm_command(command):
    command = command.split("2>")[0].split("||")[0].strip()
    if command.startswith(_SLM_RUN):
        command = "slm " + command[len(_SLM_RUN):]
    return command


def _events_and_cmds(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cmds = set()
    for arr in d["hooks"].values():
        for grp in arr:
            for hk in grp.get("hooks", []):
                cmds.add(_slm_command(hk["command"]))
    return set(d["hooks"].keys()), cmds


def test_plugin_src_hooks_cover_full_suite():
    events, cmds = _events_and_cmds(REPO / "plugin-src" / "hooks" / "hooks.json")
    assert REQUIRED_EVENTS <= events, f"missing events: {REQUIRED_EVENTS - events}"
    assert REQUIRED_CMDS <= cmds, f"missing hook commands: {REQUIRED_CMDS - cmds}"


def test_built_plugin_hooks_match_source():
    src = _events_and_cmds(REPO / "plugin-src" / "hooks" / "hooks.json")
    built = _events_and_cmds(REPO / "plugin" / "hooks" / "hooks.json")
    assert src == built, "plugin/hooks.json is stale — re-run scripts/build-plugin.mjs"
    assert REQUIRED_EVENTS <= built[0]

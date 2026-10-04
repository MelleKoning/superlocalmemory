# Why `.mcp.json` sets no profile and no data directory

It used to set both:

```json
"SLM_MCP_PROFILE": "code",
"SLM_DATA_DIR": "${CLAUDE_PLUGIN_DATA}"
```

Both are wrong for anyone who already uses SLM, and neither is needed by anyone
who does not.

**`SLM_DATA_DIR` pointed the plugin at its own private directory.** On a machine
with an existing store that means the editor talks to an empty one: measured on
the author's machine, `${CLAUDE_PLUGIN_DATA}` was 28 KB while the real store was
611 MB with 5,370 memories in it. Installing the plugin would have looked like
losing every memory. Omitted, SLM resolves its canonical data root, which is the
same store every other surface uses — and on a fresh machine that is a new store
anyway, so nothing is lost either way.

**`SLM_MCP_PROFILE: code` narrowed the tool set.** `code` (38 tools) drops
the 8 mesh tools among others. Forcing it overrode a wider profile the user had
deliberately configured. Omitted, the server falls back to the same no-profile
default as every other install — the 54-tool `full` surface — which is the
user's decision to narrow or not, not the plugin's.

`SLM_AGENT_ID` stays: it is attribution, not configuration, and it is what lets
memories written from Claude Code be told apart from every other agent.

A plugin should add capability. It should not quietly re-point the data it reads
or take tools away.

# Why the Windows launcher is not what Claude Code runs

`.mcp.json` names `${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch`, with no extension.
On macOS and Linux that is the bash launcher. On Windows it is not resolved to
`slm-launch.bat`: a host that starts an MCP server without a shell goes through
`CreateProcess`, which cannot start a batch file and appends only `.exe` to a
name that has no extension. Node's `child_process.spawn` without `shell` and
libuv behave the same way (libuv tries the literal name, then `.com`, then
`.exe`). The Windows CI runner checks this directly:
`tests/test_plugin/test_windows_mcp_spawn_premise.py`.

So editing `slm-launch.bat` changes nothing for a plugin install on Windows,
and `SLM_LAUNCHER` was deliberately not ported into it (#139). A single
`.mcp.json` has no per-platform command, so making the plugin's server start on
native Windows means changing what the command names — for example an `.exe`,
or `cmd` with `/c` — and that choice affects every platform. It is an open
decision, not something this file settles.

Hooks are different: Claude Code runs command hooks with bash (Git Bash on
Windows) or, without Git Bash, PowerShell. Under Git Bash the POSIX scripts —
`slm-run`, `ensure-venv.sh`, and the shared `slm-resolve.sh` — are what run.

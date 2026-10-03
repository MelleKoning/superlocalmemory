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

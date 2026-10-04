# SuperLocalMemory for Hermes

This is a native Hermes plugin. It adds namespaced SLM skills, slash commands,
advisor child agents, and lifecycle observation without editing a user's Hermes
configuration, memory-provider selection, MCP definitions, or existing hooks.

Install disabled, review it, then enable it through Hermes. Grant only the
`superlocalmemory` MCP server under the plugin's `mcp_allowlist`. Plugin state
is maintained by Hermes under `PLUGIN_DATA`; uninstall removes only plugin code
and optional plugin state, never an SLM database.

For updates, Hermes stages and validates the replacement before swapping it.
Disable removes this plugin's registrations but leaves all SLM data and Hermes
settings alone. Uninstall removes only this package and optional `PLUGIN_DATA`;
it never removes another plugin, an MCP server, an existing hook, or a profile.

`/slm <command> [args]` forwards the documented SLM CLI command surface. Each
top-level command also has a generated `/slm-<command>` alias. Destructive
commands require an explicit `CONFIRM` token in the slash invocation.

Use `/slm-agent <memory|governance|optimize|loop> <goal>` to launch one bounded
Hermes child agent. It is never auto-launched from a hook.

To use an SLM that runs on another computer, configure the `superlocalmemory`
MCP server in Hermes with its `https://` address, `Authorization: Bearer` key
and `ssl_verify` CA file, and set `settings.connection: remote` in this
plugin's entry. `/slm` then runs a fixed set of commands through that MCP
server, with no local SLM install; a save the server did not confirm is
reported as `NOT SAVED`. Only the exact value `remote` selects this mode. See
`docs/hermes.md#remote`.

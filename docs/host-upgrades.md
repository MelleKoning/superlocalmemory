# Host Integration Upgrades

Updating SuperLocalMemory updates the SLM executable. It does **not** silently
rewrite an AI application's configuration, hooks, plugin state, or workspace
instructions. Those files belong to the operator and may contain unrelated
tools, policies, or credentials.

Use the consented host-upgrade flow after upgrading through npm, pip, or a
repository checkout:

```bash
slm upgrade-hosts
```

This is a read-only preview. It detects existing SLM integrations and reports
what is safe to refresh. It does not create an SLM data directory, start a
daemon, change a configuration file, or enable the experimental PreToolUse
gate.

## Apply an upgrade

Review the preview, then target the hosts you approve:

```bash
slm upgrade-hosts --host codex --apply
slm upgrade-hosts --host cursor --apply
```

To refresh every *already integrated* host in one deliberate action:

```bash
slm upgrade-hosts --all-detected --apply
```

`--apply` without `--host` or `--all-detected` is rejected. This prevents a
package update from discovering and changing every supported application.

## What is changed

The command uses a merge-not-clobber rule. Existing portable MCP blocks are
verified and left intact so host-specific command paths, environment variables,
and unrelated MCP servers are not downgraded. For Codex, it may refresh only
SLM-owned skills, agents, and lifecycle hooks; the MCP block remains intact.

Claude Code's SLM integration is a marketplace plugin. Refresh it through
Claude Code's own plugin manager:

```bash
claude plugin update superlocalmemory@qualixar
```

Then restart the affected host application. Finally verify the local runtime:

```bash
slm doctor
```

## Your memory store after an upgrade

An upgrade that changes the memory store first takes a verified copy; see
[Restore points](restore-points.md). A few minutes after SLM first starts on the
new version it also checks the store once, read-only, and reports the result in
the dashboard under **Health → Memory store**, with **Repair now** if anything
needs fixing. See [Troubleshooting](troubleshooting.md#memory-store-check).

## First-time setup

For a new machine, use the interactive setup flow instead of the upgrade flow:

```bash
slm setup
```

The wizard detects integrations and asks before connecting them. For a single
new host, use `slm connect <host>`; see [IDE Setup](ide-setup.md) for supported
hosts and host-specific restart instructions.

## Package-manager behavior

The npm package creates a private Python runtime. The pip package installs into
the activated Python environment. Neither package manager is used to perform
host activation automatically: installation and host consent are separate
operations. This separation is intentional and keeps an upgrade reversible,
inspectable, and compatible with developer-managed configuration.

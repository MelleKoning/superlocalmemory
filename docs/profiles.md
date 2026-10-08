# Profiles

Profiles organize memory contexts inside one installation. Personal facts are
profile-scoped by default; shared and global recall are opt-in and subject to
the configured scope policy. Profiles are not an operating-system or tenant
security boundary: they all live in one database on one computer.

> **Two things share the word "profile".** A *memory profile* (this page) is a
> namespace of memories. An *MCP tool profile* (`SLM_MCP_PROFILE`: `core`,
> `code`, `full`, `power`, `mesh`) decides which tools a client sees. Switching a
> memory profile never changes the tool set, and the tool set is fixed when the
> MCP server starts. See [MCP Tools Reference](mcp-tools.md#which-tools-a-client-sees).

---

## What Profiles Are

A profile is a memory namespace. Each profile has its own memories and
knowledge graph, learned patterns and behavioural data, saved views, and audit
trail. Every row carries a profile id and queries are filtered by it, so recall
in one profile does not return another profile's personal memories. Shared and
global recall are opt-in; see [Shared memory](shared-memory.md).

## Default Profile

After installation there is one profile called `default`. Every memory goes
there unless you create and switch to another profile.

## Managing Profiles

### List profiles

```bash
slm profile list
```

```
Profiles:
  - default: default
  - work: work
  - client-acme: client-acme
```

`slm profile list --json` returns the same list. `slm status` shows which
profile is active.

### Create a profile

```bash
slm profile create work
slm profile create client-acme
```

### Switch profiles

```bash
slm profile switch work
```

Switching changes the active profile for the whole installation: the CLI, the
dashboard and every connected agent use it until you switch again. From an MCP
client the same switch is the `switch_profile` tool:

```json
{ "tool": "switch_profile", "arguments": { "profile_id": "work" } }
```

`switch_profile` is in the `core`, `code`, `full` and `power` tool sets and in
the default set; it is not in `mesh`, which is coordination only.

### Use another profile for one call

Most memory tools take a `profile_id`. A non-empty value routes that one call to
the named profile, which must already exist, and leaves the active profile
alone:

```json
{ "tool": "remember", "arguments": { "content": "Invoice due on the 5th", "profile_id": "client-acme" } }
```

An unknown profile id is refused, never created or treated as empty. This is
how an agent works in a second workspace without moving everyone else.

### Delete a profile

The dashboard's profile list can delete a profile. It cannot delete `default`
or the active profile. The profile's memories, and everything that makes them
findable and correctable, move into `default`; nothing is discarded. There is no
CLI command for it.

## Use Cases

### Work and personal

```bash
slm profile switch work        # work memories are captured and recalled
slm profile switch personal    # personal project memories are now active
```

### One profile per client

```bash
slm profile create client-alpha
slm profile create client-beta
slm profile switch client-alpha   # only Alpha's context is recalled
slm profile switch client-beta
```

### One profile for Web access

A web app connected through [Web access](remote-access/README.md) reaches only
the profile that was active when you turned Web access on.

## Retention and settings

Retention is not a per-profile setting. The lifecycle thresholds are changed
with the `set_retention_policy` MCP tool (`cold_after_days`, `archive_after_days`)
and apply to the whole installation. See [Compliance](compliance.md) for
deployment-level retention.

## How Profiles Work Internally

All profiles store memories in the same SQLite database with a profile
identifier on every row, and queries filter on it. The entity graph is built per
profile, so building it for one profile does not affect another.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

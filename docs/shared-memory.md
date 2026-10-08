# Multi-Scope (Shared) Memory

> **Status:** opt-in feature. **Off by default.** A fresh or unconfigured install
> keeps every memory private to its profile, and recall never surfaces another
> profile's data.

## Concepts

Every memory carries a **scope**:

| Scope | Who can recall it | How to set it |
|-------|-------------------|---------------|
| `personal` (default) | only the owning profile | nothing — this is the default |
| `shared` | the owner **and** the profiles listed in `shared_with` | `--scope shared --shared-with a,b` |
| `global` | every profile on the machine | `--scope global` |

Scope is a property of the **memory**: every atomic fact extracted from a memory
inherits the memory's scope.

Recall visibility is controlled separately by two flags, both **off by default**:

- `include_global` — also return `global` facts from other profiles
- `include_shared` — also return facts other profiles shared *with you*

Personal facts are *always* returned; the flags only add the shared/global rows.

## Enabling it

Shared memory is opt-in. You turn it on either **per call** or **persistently in config**.

### Per call (CLI)

```bash
# Write a global memory (visible to every profile)
slm remember "Deploy freeze starts Friday" --scope global

# Write a memory shared with specific profiles
slm remember "Q3 roadmap draft" --scope shared --shared-with alice,bob

# Recall including global + shared facts (opt-in for this query only)
slm recall "deploy freeze" --include-global --include-shared

# ...and explicitly exclude them again
slm recall "private note" --no-global --no-shared
```

### Per call (MCP)

```jsonc
// remember
{ "content": "Deploy freeze starts Friday", "scope": "global" }
{ "content": "Q3 roadmap draft", "scope": "shared", "shared_with": "alice,bob" }

// recall — omit the flags to use the configured default (off)
{ "query": "deploy freeze", "include_global": true, "include_shared": true }
```

### Persistently

```bash
slm config set scope.default_scope personal      # scope assigned to new writes
slm config set scope.recall_include_global true
slm config set scope.recall_include_shared true
```

These write a `scope` section in `config.json`. The dashboard's Settings page has the
same switches. You can also edit the section by hand:

```jsonc
{
  "mode": "a",
  "scope": {
    "default_scope": "personal",        // scope assigned to new writes
    "recall_include_global": true,      // surface global facts by default
    "recall_include_shared": true       // surface facts shared with you by default
  }
}
```

Omitting the `scope` section — or any field — keeps the safe defaults
(`personal` / `false` / `false`). An invalid value (a typo'd
`default_scope`, a negative weight) is ignored with a warning and falls back to the
safe default; it will not crash the CLI.

> **Wire semantics:** on the CLI/MCP boundary, leaving a flag unset sends `None`
> ("use the configured default"); passing an explicit `true`/`false` overrides config
> for that one call. The engine resolves `None` against your `ScopeConfig`, so there is
> exactly one place the default lives.

## Backward compatibility

- Memories saved before shared memory existed are `scope='personal'` and recall is unchanged.
- Per-profile isolation is preserved: another profile's `personal` facts are never
  visible, regardless of the flags.
- Turning the flags on is purely additive — it can only *add* global/shared rows, never
  remove or reorder your own results.
- Every direct read path (`recall`, `search`, `list_recent`, the `slm://recent`
  resource) is private by default; none surface another profile's data unless you opt in.

## Answer check and scope

[Answer check](answer-check.md) does not change what a recall is allowed to
return. It runs after scope filtering, on whichever results a given call's
`include_global`/`include_shared` flags already produced — it never performs
a separate, wider search of its own. A caller who cannot see a shared or
global memory will not see it via the answer check either.

## Known limits

- Cognitive-consolidation (CCQ) summary blocks are always `personal` for now; opting a
  CCQ block into shared/global is tracked for a later release.
- A consolidated summary inherits its cluster's scope only when the whole cluster agrees;
  a mixed-scope cluster yields a `personal` summary (most restrictive — never leaks).

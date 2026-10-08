# Deployment Tiers: Personal and Enterprise

SLM ships one binary. The deployment tier is a configuration preset: **Personal**
(the default) or **Enterprise**. The preset sets four switches. Each can be set on
its own in the `[deployment]` section of `config.toml`.

---

## Tier comparison

| Capability | Personal | Enterprise |
|-----------|:--------:|:----------:|
| Login gate (`require_login`) | off | on |
| PII redaction before memory storage (`pii_redaction`) | off | on |
| Retention scheduler (`retention_enabled`) | off | on |
| Hash-chained audit trail (`audit`) | on | on |
| RBAC (users, roles, workspaces) | on | on |
| GDPR export and erasure | on | on |
| Mesh coordination | on | on |
| Cloud backup | opt-in | opt-in |
| Scale Engine projections | opt-in | opt-in |

All capabilities are present in both tiers. The difference is default
configuration, not binary capability.

---

## Where the tier is set

The tier lives in `config.toml` in the data folder (`~/.superlocalmemory/`), in a
`[deployment]` table. It is separate from `config.json`, and `slm config set`
does not change it. If the file or the table is absent, the install is Personal.

```toml
[deployment]
mode = "enterprise"          # "personal" or "enterprise"
require_login = true         # each key below is optional
pii_redaction = true
retention_enabled = true
audit = true
```

A key you leave out takes the value of the chosen mode's preset. The values must
be real TOML booleans. A `[deployment]` table that cannot be read, has an unknown
`mode` or a non-boolean value is treated as Enterprise, so a typo can never
quietly weaken the controls. Restart SLM after editing it (`slm restart`).

The setup tools write the table for you. `slm reconfigure` re-runs the
interactive configurator, which asks for a deployment mode, and
`--deployment=enterprise` selects it directly. Personal installs get no
`[deployment]` table, because Personal is the default.

---

## Personal tier

The default for individual developers. No login is required and the loopback
owner is trusted. PII redaction and retention are off.

Use it when you are the only user of the machine or the SLM install, you do not
share memory with other users, and you want no authentication step.

---

## Enterprise tier

For teams, shared infrastructure, or any deployment where several people use the
same SLM instance. The preset turns on:

1. `require_login`: reading and saving memories needs a signed-in user. See
   [Teams, users and RBAC](rbac-teams.md).
2. `pii_redaction`: recognized PII patterns are redacted before memory content is
   stored.
3. `retention_enabled`: the retention scheduler runs hourly and applies your
   retention rules.
4. `audit`: the audit trail, which is already on in Personal.

### First run after enabling login

With login required there are no default credentials. Create the first user and
role in the dashboard under **Settings → Access**, as described in
[Teams, users and RBAC](rbac-teams.md) and [Company mode](company-mode.md).

---

## Retention rules

When `retention_enabled` is on, the retention scheduler checks every profile once
an hour and applies its named rules. Each rule has a name, a number of days, a
framework label (`custom` unless you give another) and an action:

| Action | What happens to an expired memory |
|--------|----------------------------------|
| `archive` | Moves to the archived lifecycle zone |
| `tombstone` | Moves to the archived zone and is marked purgeable |
| `notify` | Nothing changes; the count is reported so you can act |

Enforcement is soft: it changes a memory's state and never deletes the raw row.
A rule belongs to one profile, so different workspaces can have different rules.
A rule can be limited with `applies_to` to a `scope`, `fact_type`, `signal_type`
or `session_id`.

Rules are managed through the daemon's HTTP API, which needs SLM's own credential
and, where roles are in use, the manage permission:

```
POST   /api/compliance/retention-policy   {"name": ..., "retention_days": 30, "action": "archive", "category": "custom"}
DELETE /api/compliance/retention-policy?name=...
POST   /api/compliance/retention/enforce
```

`POST /api/compliance/retention/enforce` applies the active profile's rules
immediately. The MCP `set_retention_policy` tool is a different setting: it sets
the idle-day thresholds (`cold_after_days`, `archive_after_days`) of the normal
memory lifecycle.

---

## PII redaction

With `pii_redaction` on, SLM applies pattern-based redaction to memory content
before it is stored. Redaction is heuristic and does not catch every PII
pattern. Operators responsible for a compliance program should verify coverage
for their content.

---

## Related

- [Teams, users, and RBAC](rbac-teams.md)
- [GDPR and compliance controls](compliance.md)
- [Configuration reference](configuration.md)

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

# Migration from V2

Upgrade from SuperLocalMemory V2. Verify a backup before migrating; see the
rollback caveats below. There is no `slm migrate --dry-run`.

> **Schema migrations.** Newer releases add further schema changes on top of the
> V2 migration. They are additive, apply automatically at startup, and run in
> order. `slm db migrate --status` shows what has been applied and
> `slm db migrate --dry-run` previews pending ones without writing. They are
> **forward-only**: there is no `slm db migrate --rollback`. SLM refuses to run
> against a database written by a newer build, and holds back a migration whose
> dependency did not complete.
>
> Every update that changes the store first takes a verified copy, so you can go
> back to it with `slm db restore`, and `slm db prepare-downgrade` readies the
> store for the previous release when that is safe. See [Restore points and
> downgrades](restore-points.md). Otherwise, to revert an upgrade, restore a
> **verified pre-upgrade complete backup of the whole data root**: `slm serve stop`
> first so the WAL checkpoints, then copy all present `*.db` files with their
> `-wal` and `-shm` sidecars and `lance/` if present.

---

## What Changed Since V2

| Area | V2 | Now |
|------|----|----|
| **Retrieval** | Single-channel semantic search | Five candidate producers (Semantic + BM25 + Temporal + Spreading-Activation + Hopfield) -> RRF fusion + entity-graph post-fusion enhancement |
| **Modes** | One mode (cloud required for smart features) | Three modes: A (Local Guardian), B (Smart Local), C (Full Power) |
| **Math layer** | None | Fisher-Rao similarity, Sheaf consistency, Langevin lifecycle |
| **Ingestion** | Basic text storage | 11-step pipeline: entities, facts, emotions, beliefs, graph, and more |
| **Data directory** | `~/.claude-memory/` | `~/.superlocalmemory/` (the migrator attempts a legacy-path symlink; verify it) |
| **Consistency** | Manual | Automatic contradiction detection |
| **Recall quality** | Good | Significantly better on complex queries (multi-hop, temporal) |

**Compatibility boundary:** verify the commands, integrations, profiles, and
runtime artifacts your deployment relies on after migration. The migrator is a
data/schema migration, not a proof that every optional configuration, learned
state, or integration remains operational.

## Before You Migrate

1. **Update to the latest version:**

```bash
npm update -g superlocalmemory
```

2. **Check your current version:**

```bash
slm --version
# Prints the installed version
```

3. **Preserve both data roots before migration** (do not rely on a live
   `memory.db` copy):

```bash
slm serve stop
# Copy the complete legacy V2 source ~/.claude-memory/ to an encrypted/private
# destination. If ~/.superlocalmemory/ already exists, preserve it separately.
# Include every present .db plus -wal/-shm sidecar and lance/ directory.
# Verify owner-only modes (0600/0700); destination follows process umask.
ls -la ~/.claude-memory/
# Keep the daemon stopped through `slm migrate`; restart only after migration
# and verification have completed.
```

> No `slm migrate --dry-run` exists for the V2 migrator. For the additive
> database migrations that follow, the inspect commands are `slm db migrate
> --dry-run` and `slm db migrate --status`, forward-only.

## Run the Migration

```bash
slm migrate
# After migration completes and its checks pass:
slm serve start
```

The migrator performs these code-defined steps:

1. Creates a backup of your V2 database (verify it is complete and
   owner-only before proceeding)
2. Copies data from `~/.claude-memory/` to `~/.superlocalmemory/`
3. Attempts to create a legacy-path symlink (`~/.claude-memory/ -> ~/.superlocalmemory/`); verify it before relying on old IDE configs
4. Extends the database schema and inserts migrated facts

It does not configure an operating mode, run a SQLite integrity check, or
guarantee that every optional embedding/BM25 projection has been rebuilt.
Run the post-migration checks below and re-embed/rebuild any optional indexes
required by your deployment.

> The migration spans file copies, SQLite commits, and a rename/symlink — not a
> single global transaction. Do **not** treat it as globally
> transactional/zero-loss without a verified pre-upgrade backup.

The additive schema migrations that follow are applied automatically at startup.

## Migration boundaries

The migrator copies supported V2 memory data and attempts the legacy-path
symlink. Preserve and verify your pre-upgrade whole-root backup because it does
not prove complete continuity for optional indexes, configuration, audit/trust
history, or every runtime artifact in a customized installation.

## What Gets Added

The migration adds these capabilities to your existing data:

- BM25 token index for keyword search
- Entity graph nodes and edges
- Temporal event entries
- Fisher-Rao similarity metadata
- Sheaf consistency sections
- Langevin lifecycle state

These are available to configure after migration; do not assume every optional
projection has been materialized until its health/rebuild check succeeds.

## After Migration

### Verify

```bash
slm status --json
# or slm status for the text summary
slm db migrate --status   # shows which schema migrations are applied
```

Confirm:
- Configure and verify the intended operating mode; migration does not select one
- Memory count matches your V2 count (`slm status --json | jq '.data.fact_count'`)
- `slm db migrate --status` shows the expected migrations as applied

### Try a recall

```bash
slm recall "something you stored in V2"
```

Results should match or exceed V2 quality. Multi-producer retrieval can find memories that V2's single-channel search missed.

### Explore the new features

```bash
slm trace "your query"       # See channel-by-channel breakdown
slm health                   # Check math layer status
slm mode b                   # Try Smart Local mode (if Ollama is installed)
slm db integrity             # read-only store health report
slm ops status               # stuck or failed operations after an upgrade
```

## Rollback

Rollback of the V2 `slm migrate` (`slm migrate --rollback`) is **only** possible while a valid
pre-migration backup still exists and is **not** automatic or retained for
30 days. There is no automatic 30-day retention or timed deletion — verify the
backup file before migrating. Re-creating the backup during migration does not
guarantee a coherent cross-store set on the legacy per-file path
(`docs/cloud-backup.md`).

The additive schema migrations cannot be rolled back with a migrate flag: there is no
`slm db migrate --rollback`, and `slm migrate --rollback` is for the V2 migrator
only. To revert, use a restore point (`slm db restore`, see [Restore points and
downgrades](restore-points.md)) or restore a verified **pre-upgrade complete
backup of the whole data root** (stop the daemon first — `slm serve stop` — and include
WAL/SHM sidecars plus `lance/` if present). Copying a live `memory.db` alone
while the daemon runs is unsafe and does not guarantee a coherent restore set.

## IDE Configuration Updates

### Automatic (best effort)

The migrator attempts to create a legacy-path symlink for compatible IDE
configurations. Check that it exists and test each IDE integration after
migration; a symlink failure is reported as a warning rather than a global
migration failure.

### Manual (optional)

If you want to update your IDE configs to use the new path directly:

```bash
slm connect
```

This updates all detected IDE configs to point to `~/.superlocalmemory/` instead of relying on the symlink.

## FAQ

**Q: Will my IDE break during migration?**
It may require repair. Confirm the legacy-path symlink and run a real
connection/recall check in each IDE you use; use `slm connect` to update
detected configurations directly.

**Q: Do I need to reconfigure my API keys?**
Possibly. The V2 migrator copies the database; it does not prove migration of
separate configuration or credential files. Reconfigure or supply keys through
environment variables as needed, then test the provider path you use.

**Q: Can I run V2 and the current version side by side?**
No. The migration converts your database in place (with backup). No side-by-side.

**Q: What if migration fails halfway?**
The migration spans multiple file copies/SQLite commits and a symlink; it is **not**
a globally atomic/transactional switch. Keep the verified pre-upgrade complete
backup (offline whole-root copy with daemon stopped) and restore that if needed.
Do not rely on an unverified live `memory.db` copy.

**Q: I have multiple profiles. Are they all migrated?**
Verify them explicitly. Confirm each expected profile appears and that its
recall boundaries still behave correctly before retiring the recovery copy.

**Q: How big will my database get after migration?**
There is no supported fixed percentage. Size depends on the source data,
enabled indexes, and later model/vector artifacts. Measure the backup and the
completed target root before deleting any recovery copy.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

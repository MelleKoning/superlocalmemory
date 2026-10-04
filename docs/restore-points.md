# Restore Points and Downgrades

> SuperLocalMemory V4 documentation · 4.1.19+

Every update that changes the memory store first takes a verified copy. Those
copies are restore points. Copies from the last two updates are kept.

## List them

```bash
slm db restore-points
```

Each line shows the point id, when it was taken, why (for example
`migration`), the versions involved, its size and how many memories it holds.

## Go back to one

```bash
slm db restore <point_id>
slm restart
```

The restore is recorded and runs at the next start, before anything opens the
store. Restoring under a running engine is unsafe, so it always waits.

- Memories saved after the restore point are added back, unless you pass
  `--no-reimport`.
- It deletes again only what you, or a privacy erasure, deleted after the copy.
- Kinds you confirmed after the copy are kept, and erased profiles stay erased.
- It refuses a copy made by a newer version and works when the live store is
  damaged.
- `slm db restore --cancel` withdraws a restore that has not run yet. Once it
  starts, it cannot be cancelled half-way.

## Go back to the previous version

An older SLM refuses to open a store stamped by a newer one. To install the
previous release without losing anything:

```bash
slm db prepare-downgrade
# it names the version to go back to and prints the exact command, for example:
pipx install --force "superlocalmemory==4.1.19"   # or: npm install -g superlocalmemory@4.1.19
```

Preparation is allowed only when every change since that version is one the
older build safely ignores. It takes a verified copy first. Undo it with
`slm db prepare-downgrade --cancel`.

The same actions are available over the local HTTP API under `/api/upgrade/`
(`restore-points`, `restore/preview`, `restore`, `restore/cancel`,
`prepare-downgrade`, `prepare-downgrade/cancel`). For copies kept off this
machine, see
[cloud-backup.md](cloud-backup.md).

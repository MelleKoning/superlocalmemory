# SLM Error Reference

Errors reach you in four places: the CLI, MCP tool results, the daemon's HTTP
API, and the web apps connected through [Web access](remote-access/README.md).
Use `slm doctor` to diagnose most problems, and `slm doctor --fix` to repair the
ones it can.

## CLI

With `--json`, a failed command prints an envelope with `success: false` and an
`error` object holding a `code`, a `message`, and often a `hint` and
`retryable`. Without `--json` the message goes to the error stream.

| Code | When | What to do |
|------|------|-----------|
| `DAEMON_UNAVAILABLE` | A command that writes could not reach the SLM daemon. `reason` and `hint` say which of: process exited, recycled process id, unreachable port, or identity mismatch | Follow the hint: `slm serve start`, or `slm restart`, then `slm doctor`. Retryable |
| `INVALID_KIND` | `--kind` (or the MCP `kind`) is not one of the nine [memory kinds](memory-kinds.md). Nothing was retrieved or saved | Use one of the listed values |
| `INVALID_TIME_FILTER` | `--window`, `--as-of`, `--known-as-of` or `--valid-at` could not be read | Use a span such as `7d`, a range such as `2026-07-01..2026-07-31`, or an ISO 8601 time. Nothing was searched |
| `NO_STORE` | `slm db integrity` or `slm db repair` found no `memory.db` in the data folder | Check `SLM_DATA_DIR` and `slm status` |
| `WRONG_ROOT` | `slm db repair --apply` or `--undo` was given a `--root` that is not the data folder the command resolves to. Nothing was changed | Pass the `--root` the message names |
| `BUSY` | A repair is already running | Wait, then run again |
| `NOT_AUTHORIZED` | The caller is not allowed to make that change | Check your role on that profile |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | The command ran and failed (see the message) |
| 2 | The command refused to run: an unknown kind, a rejected `--replaces`, or a repair aimed at the wrong folder |

`slm gdpr verify` uses 0 for an intact receipt, 1 for a tampered one and 2 when
the receipt is not found. `slm remote check` and `slm warmup` exit non-zero when
a check fails.

## MCP tool results

Tools return a dictionary. Newer tools report `ok: false`; older ones report
`success: false` with an `error` string, so a caller should handle both. Codes
you may see:

| Code | Meaning |
|------|---------|
| `DAEMON_UNAVAILABLE` | The daemon did not answer; `retryable: true` |
| `INVALID_KIND` | The `kind` value does not parse. Raised before any retrieval |
| `INVALID_TIME_FILTER` | A time filter could not be read |
| `unknown_profile` | `profile_id` names a profile that does not exist. It is refused, never treated as empty |
| `NOT_AUTHORIZED` | Not allowed to write there |
| `remote_tool_not_allowed` | The caller came through remote access and the tool is host-only |

## Daemon HTTP API

Errors use ordinary HTTP status codes. A request body that fails validation gets
a `422` that says where and why, without repeating the value you sent. When a client sends too many requests the daemon
answers `429` with `{"error": "Too many requests."}` and a `Retry-After` header.
The limit is per client address and per window (60 seconds by default): 30
writes and 120 reads, set by `SLM_RATE_LIMIT_WRITE`, `SLM_RATE_LIMIT_READ` and
`SLM_RATE_LIMIT_WINDOW`. Local dashboard traffic on the same computer gets a much
higher limit.

## Web access

An app connected through Web access sees these codes in the tool result:

| Code | Meaning | What to do |
|------|---------|-----------|
| `connector_asleep` | Your computer has sent no heartbeat for over 45 seconds | Wake it and try again |
| `connector_offline`, `connector_unavailable`, `connector_closed`, `connector_error`, `connector_replaced`, `stale_connector` | The connection from your computer to the gateway is down or just replaced | Check the computer is online; `slm status`; the Connected apps page |
| `relay_timeout`, `origin_timeout` | The call passed its 25-second deadline | Try again; one call at a time |
| `relay_busy` | Too many calls in flight on this connection (limit 8) | Wait a few seconds and make one call at a time |
| `request_cancelled` | The call was cancelled before it finished | Try again |
| `DAILY_LIMIT_REACHED` | The free daily allowance is used up. It resets at midnight UTC | Continue without memory until then. Local use is unaffected |
| `TOOL_DENIED`, `INSUFFICIENT_SCOPE` | The app was not given that permission | Remove the app on the Connected apps page and add it again with the permission ticked |
| `REVOKED`, `connection_revoked`, `ENTITLEMENT_REQUIRED` | The app was removed, or Web access has ended | Turn Web access on again on the Connected apps page |

See [Troubleshooting](troubleshooting.md#web-access).

## Python exceptions

| Exception | Module | When raised |
|-----------|--------|-------------|
| `PoolError` | `mcp._pool_adapter` | A worker pool returned an error envelope (`{"ok": false}`): the worker crashed or timed out |
| `CapabilityError` | `core.engine_capabilities` | A light-mode MCP engine was asked for a full-mode operation (recall or store). Route it through the daemon |
| `SafeFsError` | `core.safe_fs` | A file-system safety check failed: a symlink, the wrong owner, or a cloud-synced directory |
| `QueueTimeoutError` | `core.recall_queue` | A queued recall did not finish before its deadline |
| `DeadLetterError` | `core.recall_queue` | A queued request used up its retries and was moved to the dead-letter queue |
| `QueueCancelledError` | `core.recall_queue` | Every subscriber withdrew before the worker finished |

`core.error_envelope.ErrorCode` also defines `RATE_LIMITED`, `QUEUE_FULL`,
`TIMEOUT`, `CANCELLED`, `DEAD_LETTER`, `DAEMON_DOWN` and `INTERNAL`, with an
envelope of `ok`, `error_code`, `error`, `request_id` and `retry_after_ms`.
Treat these as a library vocabulary for queue-backed callers, not a promise about
what the HTTP API returns.

Daemon logs are in the `logs` folder of the data folder (`daemon.log` and
`daemon-error.log`).

## Storage notes

### sqlite-vec (`fact_embeddings`) row deletes versus full-table deletes

Row-targeted deletes work: `DELETE FROM fact_embeddings WHERE rowid = ?` is the
supported primitive, used by the vector store's per-fact delete and by GDPR
erasure. A bare full-table `DELETE FROM fact_embeddings` with no `WHERE` clause
is rejected by sqlite-vec with `database disk image is malformed`. That error is
misleading: the database is not corrupt, and `PRAGMA integrity_check` and
`quick_check` correctly keep reporting `ok`. Do not replace the database over it.

Supported ways to clear or repair vector data:

- Per-fact removal through the product APIs (GDPR erase), which delete the vec0
  row, the metadata row and the row-map row together.
- `slm db repair`, which rewrites a vector that no longer matches its memory and
  removes LanceDB rows whose memory is gone. See
  [CLI Reference](cli-reference.md#embedding-models-and-store-health).
- A full rebuild through the embedding migration path (`slm embedder switch`),
  which drops and recreates the virtual table rather than deleting from it.

Never hand-edit the database with raw SQL against the `fact_embeddings*` virtual
tables; their shadow-table layout is a sqlite-vec internal and not a stable API.

# Recall: Filters, Time Travel and Trace

> SuperLocalMemory V4 documentation

`slm recall` (alias `slm search`) and the MCP `recall` tool search every
memory in the active profile across several channels — semantic, keyword
(BM25), temporal, associative (Hopfield) and spreading activation — fuse the
results, and rerank them.

## Narrowing a recall

| Flag | Keeps only |
|---|---|
| `--project NAME` | Memories saved under that project |
| `--saved-by AGENT` | Memories saved by that agent (for example `claude-desktop`) |
| `--about NAME` | Memories that mention that person, project or tool |
| `--kind KIND` | One of the nine [memory kinds](memory-kinds.md) |
| `--window SPAN` | An event-time range: `24h`, `7d`, `30d`, `1y`, or `2026-07-01..2026-07-31` |
| `--limit N` | At most N results (default 20) |

Filters apply on every recall path and before the [answer check](answer-check.md),
so the check judges only what you will see.

## Recall inside a project

| Flag (MCP / HTTP parameter) | Effect |
|---|---|
| `--prefer-project NAME` (`prefer_project`) | Memories saved under that project rank above others of similar relevance. Nothing is removed. |
| `--project NAME` (`project`) | Only that project's memories. If none of the memories found were saved under it, you get the unfiltered results and a line saying they are not narrowed. |

A project is a name or a path: `acme-billing`, `/Users/me/work/acme-billing`
and `ACME-Billing/` are the same project (the last part of the path, ignoring
case). `session_init` and `slm session open` prefer the project they are given.
The preference is bounded: a project memory passes another memory only when its
score is already at least 80% of that memory's, so a weak match never jumps a
strong one. It is applied after learned ranking (`SLM_RANKING`), on the final
order, so turning learning on does not undo it. The response's `project_scope` says what happened.

Memories saved before 4.1.21 carry a project only if one was passed to
`remember`. Session-end summaries (`[name] session ended ...`) are tagged with
their project by background maintenance after upgrading.

## Time travel

| Flag | Returns |
|---|---|
| `--as-of TIME` | Recall pinned to a point-in-time snapshot |
| `--known-as-of TIME` | Only facts SLM knew by that time |
| `--valid-at TIME` | Only facts that were true at that time |

`TIME` is ISO-8601, for example `2026-01-01T00:00:00+00:00`. Facts saved before
4.0.2 have no recorded time; add `--include-unknown` to include them.

```bash
slm recall "staging database" --known-as-of 2026-01-01T00:00:00+00:00
# No confident match.
```

## Scopes

Recall returns your own (`personal`) memories by default. Add
`--include-shared` or `--include-global` to read memories other profiles
shared. See [shared-memory.md](shared-memory.md).

## Seeing why a result ranked

```bash
slm trace "how do we roll back the API"
```

`trace` prints the per-channel score breakdown for each result.

## Reading the JSON

`slm recall --json` returns the results plus signals an agent can act on:

| Field | Meaning |
|---|---|
| `no_confident_match` | Nothing scored high enough to trust |
| `abstained`, `abstention_reason` | The answer check judged that the results do not answer the question. Results are still returned |
| `incomplete_channels` | Channels that could not run, for example while the embedding model is still loading. The human output says "Incomplete search" |
| `answer_check_status`, `answer_check_ran` | Whether the answer check ran, and why not when it did not |
| `channel_status` | Per-channel state for this recall |

Scores are uncalibrated ranking signals, not probabilities. See
[retrieval-score-contract.md](retrieval-score-contract.md).

## Summaries

```bash
slm summary session    # what one session covered
slm summary day        # a day's main topics
slm summary project    # what was worked on in a project
```

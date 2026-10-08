# Recall: Filters, Time Travel and Trace

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
| `--tag TAG` | Memories saved with that exact tag. Repeat the flag for several tags |
| `--tags-match all\|any` | With several tags: `all` (the default) needs every tag, `any` needs at least one |
| `--window SPAN` | An event-time range: `24h`, `7d`, `30d`, `1y`, or `2026-07-01..2026-07-31` |
| `--limit N` | At most N results (default 20) |

Filters apply on every recall path and before the [answer check](answer-check.md),
so the check judges only what you will see. They combine as AND.

A tag matches by its canonical identity: case, surrounding spaces and how it was
stored never matter. Unlike `--project`, a tag filter never falls back to
unfiltered results. An empty answer says why in `tag_scope`: nothing was ever
saved with that tag, or something has it but none of it matched the question.
Over MCP, `tags` takes a comma-separated string or a list (use the list when a
tag contains a comma).

## Recall inside a project

| Flag (MCP / HTTP parameter) | Effect |
|---|---|
| `--prefer-project NAME` (`prefer_project`) | Memories saved under that project rank above others of similar relevance. Nothing is removed. |
| `--project NAME` (`project`) | Only that project's memories. If none of the memories found were saved under it, you get the unfiltered results and a line saying they are not narrowed. |
| `--project-strict` (`project_strict`) | With `--project`: turn that fall-back off. Only that project's memories, even if there are none, for automations that must never act on another project's memories. |

A project is a name or a path: `acme-billing`, `/Users/me/work/acme-billing`
and `ACME-Billing/` are the same project (the last part of the path, ignoring
case). `session_init` and `slm session open` prefer the project they are given.
The preference is bounded: a project memory passes another memory only when its
score is already at least 80% of that memory's, so a weak match never jumps a
strong one. It is applied after learned ranking (`SLM_RANKING`), on the final
order, so turning learning on does not undo it. The response's `project_scope` says what happened.

A memory carries a project only if one was passed to `remember` when it was
saved. Session-end summaries (`[name] session ended ...`) are tagged with their
project by background maintenance.

## Time travel

| Flag | Returns |
|---|---|
| `--as-of TIME` | Recall pinned to a point-in-time snapshot |
| `--known-as-of TIME` | Only facts SLM knew by that time |
| `--valid-at TIME` | Only facts that were true at that time |

`TIME` is ISO-8601, for example `2026-01-01T00:00:00+00:00`. Facts saved before
4.0.2 have no recorded time; add `--include-unknown` to include them in the
strict `--known-as-of` and `--valid-at` filters. A time that cannot be read
stops the command before any search.

```bash
slm recall "staging database" --known-as-of 2026-01-01T00:00:00+00:00
# No confident match.
```

## Scopes

Recall returns your own (`personal`) memories by default. Add
`--include-shared` or `--include-global` to read memories other profiles
shared, and `--no-shared` or `--no-global` to leave them out when your
configuration includes them. See [Shared memory](shared-memory.md).

## Seeing why a result ranked

```bash
slm trace "how do we roll back the API"
```

`trace` prints the per-channel score breakdown for each result. Its default is
10 results.

## Reading the JSON

`slm recall --json` returns the results plus signals an agent can act on:

| Field | Meaning |
|---|---|
| `no_confident_match` | Nothing scored high enough to trust |
| `abstained`, `abstention_reason` | The answer check judged that the results do not answer the question. Results are still returned |
| `incomplete_channels` | Channels that could not run, for example while the embedding model is still loading. The human output says "Incomplete search" |
| `answer_check_status`, `answer_check_ran` | Whether the answer check ran, and why not when it did not |
| `channel_status` | Per-channel state for this recall: for example `warming` while the embedding model loads, or `no_embedding` |
| `answerability`, `answerability_reason` | `supported`, `unsupported` or `unjudged`. `abstained: false` alone never means the answer check ran; this does. `session_init` always reports `unjudged` |
| `query_id` | Identifies this recall; pass it to `report_outcome` as `recall_query_id` |
| `project_scope`, `tag_scope` | What the project and tag filters did |

Scores are uncalibrated ranking signals, not probabilities. See
[Retrieval score contract](retrieval-score-contract.md).

## Summaries

```bash
slm summary sessions        # sessions that have memories to summarise
slm summary session <id>    # what one session covered
slm summary day             # today's main topics, in this computer's time zone
slm summary project         # what was worked on in a project
```

Every summary says how much of the data it covered and lists the ids of the
memories it was built from. A summary of a session is usually partial, because
most memories carry no session, and it says so. The dashboard has the same
summaries under Memories → Summaries; the MCP tool is `get_memory_summary`.

## Saved views

A saved view is a recall query you run often, saved under a name. Running it is
running recall, so it ranks and checks the same way and gives the same answer on
an unchanged store. Every result shows its memory id. A view belongs to one
profile; no other profile can see or run it. No prompt is ever run over your
memories.

```bash
slm view create "Work log" "what did I ship" --window 7d
slm view list
slm view run "Work log"          # also: slm view show
slm view rename "Work log" "Shipped"
slm view delete "Shipped"        # removes the saved query only
```

Filters: `--kind`, `--window` (`24h`, `7d`, `30d`, `1y` or `2026-07-01..2026-07-31`)
and `--as-of`. Up to 50 results per view (`--limit`, default 10), 200 views per
profile. The dashboard has the same views under Memories → Saved views; the MCP
tools are `run_view` (read) and `manage_view` (write).

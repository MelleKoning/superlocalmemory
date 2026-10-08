# Architecture

A high-level overview of how SuperLocalMemory stores, organizes, and retrieves your memories.

## Local core and optional Web access

![Integrated SLM architecture: local memory capabilities, Laya/Jev answer checks and optional web connection](remote-access/assets/slm-integrated-architecture.svg)

The local core is the foundation and works fully without an account. **Web access** is an optional extra, managed from the dashboard's **Connected apps** page, that lets an AI app on the internet (ChatGPT, Claude on the web, Composio, Muse) use the same engine and database after you sign in with GitHub and approve what it may do. It leaves local tools and configuration untouched, and a gateway problem never stops local use. Web results do pass through the gateway and the AI service you connected, so keeping the database local does not mean nothing leaves the computer.

See [Web access](remote-access/README.md), [its architecture and trust boundaries](remote-access/architecture.md) and [how to connect an app](remote-access/onboarding.md). The ingestion and retrieval design below is the local engine reference.

Published LoCoMo evidence is maintained in [Benchmark Evidence](benchmarks.md),
including the original model, judge, and sample disclosures required to interpret
each result.

## What SLM is built from

Multi-channel retrieval over a local-first store, a governed write path (admission, per-store obligations, completion manifests), cross-store erasure, SLM-Mesh peer coordination, multi-scope memory with profiles, cache and compress context optimization, memory kinds, saved views and summaries, Entity Explorer and skill evolution surfaces, an optional Web access connection, and enterprise controls (roles, retention, hash-chained audit). Modes A/B/C describe locality and enrichment choices; they do **not** determine EU AI Act legal compliance — that depends on deployment context.

## Memory boundaries: profiles and scopes

Profiles partition the active workspace. Within a profile, a memory is
`personal` (default), `shared` with explicitly named profile readers, or
`global`. Read visibility across profiles is default-deny: recall includes
shared/global facts only when an operator enables the relevant policy or a
caller supplies the explicit scope option. This is local authorization and is
separate from trusted-peer SLM Mesh coordination.

---

## Design Principles

1. **Local-first.** Core memory state uses a local data root; optional integrations have separate network behavior.
2. **Explicit activation.** Installation does not implicitly add hooks, edit IDE configuration, start a daemon, or download a model.
3. **Multi-channel retrieval.** The engine can combine five candidate producers when their dependencies are healthy.
4. **Inspectable scoring.** Mathematical and heuristic signals remain distinct from calibrated answer confidence.

## System Overview

```
Your IDE (Claude, Cursor, VS Code, ...)        Web app (optional)
       |                                              |
       | MCP Protocol                                 | HTTPS + OAuth, via the gateway
       v                                              v
+------------------+                          +------------------+
| MCP Server       |  Tool-set-selected       | Companion        |  Outbound link,
|                  |  tools and resources     | (in the daemon)  |  same MCP server
+------------------+                          +------------------+
       |                                              |
       +----------------------+-----------------------+
                              v
                    +------------------+
                    | Memory Engine    |  Ingestion + Retrieval + Lifecycle
                    +------------------+
                              |
                              v
                    +------------------+
                    | SQLite Database  |  ~/.superlocalmemory/memory.db
                    +------------------+
```

## How Memories Are Stored (Ingestion)

An accepted write is first recorded in the M018 `ingestion_operations` table.
The source type and idempotency key own replay identity; reuse with different
immutable evidence is rejected.

| Durable phase | Contract |
|---|---|
| `raw` | Raw content, metadata, scope, session context, and trusted actor are durable. |
| `queryable` | A relational fact and its SQLite FTS projection are committed in the same transaction. |
| `enriching` | A lease-owning worker performs extraction, entity resolution, consolidation, graph, temporal, provenance, and embedding work. |
| checkpoint | Final fact IDs and completed derivation stages are committed before optional external projection. |
| `complete` | Every declared derivation and projector stage reports success. |
| `failed` | Evidence and checkpoint data remain inspectable and retryable with bounded backoff. |

Mode and configured dependencies determine how individual enrichment stages are
implemented. The state record reports what the released runtime actually
completed; documentation does not imply that an unavailable optional backend
participated.

## How Memories Are Retrieved (Recall)

Recall uses five candidate producers where their dependencies are available,
then fusion, optional reranking, and graph-based score enhancement.

### Candidate Producers

| Channel | How it works | Best for |
|---------|-------------|----------|
| **Semantic** | Dense vector similarity; Fisher-derived terms can affect later scoring where available | "Queries that mean the same thing but use different words" |
| **BM25** | Classic keyword matching with term frequency scoring | "Queries with specific names, codes, or exact terms" |
| **Temporal** | Matches based on time references and event ordering | "What did we decide last Friday?" or "Changes since the sprint started" |
| **Hopfield** | Associative retrieval over stored representations | Related decisions and linked technical context |
| **Spreading activation** | Traverses linked graph neighborhoods | Multi-hop related evidence |

### Fusion and Ranking

1. Available producers return candidates with scores and provenance.
2. Reciprocal Rank Fusion combines the ranked lists.
3. Optional rerankers refine the leading candidates when enabled.
4. Graph-derived evidence can modify later scoring without becoming a sixth producer.
5. The response returns the final ranked evidence and measured runtime fields.

No single producer is guaranteed to run in every mode. The runtime degrades to
the indexes that are healthy and reports its result provenance.

## Safe context injection

Stored text is data, even when it came from a local or first-party source. Every
runtime injection surface maps recalled results into one renderer that applies
configured budgets, redacts recognized secrets, neutralizes attempts to forge
the canonical boundary, and carries fact/source provenance. The rendered block
is explicitly reference-only evidence; instructions found inside do not gain
authority to call tools, change roles, or request secrets.

Cursor, Copilot, and Antigravity instruction files contain only
product-authored static protocol. Dynamic memories are fetched through MCP at
runtime and are not persisted into these high-trust files. See
[Auto-memory](auto-memory.md#memory-is-evidence-not-instruction) for how the
boundary works.

## Three Operating Modes

| Mode | Retrieval | LLM Usage | Data Location |
|------|-----------|-----------|---------------|
| **A: Local Guardian** | Candidate retrieval + math-informed scoring | None for core memory operations | Local data root; optional integrations may use the network |
| **B: Smart Local** | Candidate retrieval + local LLM enrichment | A model on this machine (Ollama by default, any local OpenAI-compatible server works) | Local data root; optional integrations may use the network |
| **C: Full Power** | Candidate retrieval plus configured provider-backed enrichment | Your own endpoint, or a cloud provider | Configured content is sent to the provider |

Mode A is the default. Core memory operations can run without a cloud model provider, but model and dependency downloads, connectors, cloud backup, and explicitly enabled integrations may use the network.

## Mathematical Foundations

SLM uses three mathematical layers. They solve specific practical problems.

### Fisher-Rao Similarity

**Problem:** Standard vector similarity treats all memories equally, regardless of how much evidence backs them.

**Solution:** Fisher-Rao geometry accounts for the statistical confidence of each memory's embedding. A memory accessed and confirmed many times gets a tighter confidence region. A memory stored once and never validated has wider uncertainty. Similarity scoring respects this difference.

**Effect:** Frequently validated memories rank higher. Uncertain memories are flagged for verification.

### Sheaf Consistency

**Problem:** Over time, you store contradictory memories. "We use PostgreSQL" and later "We migrated to MySQL." Simple retrieval returns both without flagging the conflict.

**Solution:** Sheaf cohomology can detect when memories attached to the same entity or topic contradict each other, and the system then records a *supersedes* link from the newer memory to the older one. That link is evidence, not a deletion: the older memory stays in the store and can still be recalled. The check that runs when a memory is stored is off by default (see [Configuration](configuration.md#consistency-checking-at-store-time)); you can still ask for contradictions on demand with the `consistency_check` MCP tool (in the `power` tool set). Nothing is hidden from you on the strength of an automatic contradiction check.

**Effect:** Contradictions can be surfaced for your review instead of being silently averaged.

### Langevin Lifecycle

**Problem:** Memory databases grow endlessly. Old memories dilute retrieval quality.

**Solution:** Langevin dynamics models each memory's lifecycle — from Active (frequently accessed) through Warm, Cold, and eventually Archived. The transition is not based on simple time rules but on a self-organizing dynamic that balances recency, access frequency, and information value.

**Effect:** Active memories stay prominent. Stale memories fade gracefully. Storage stays efficient.

## Bounded Loop Engine

A gate-verified iteration primitive. The engine runs laps until an independent gate passes or a hard bound trips. The agent's own claim that it is done is recorded per lap for audit, but does not terminate the run — only the independent gate can.

Three surfaces drive the same engine and the same durable ledger:

| Surface | Entry point |
|---------|-------------|
| CLI | `slm loop {demo\|history\|show}` |
| MCP | `slm_loop_run`, `slm_loop_history`, `slm_loop_show` |
| Slash command | `/slm-loop` |

Every lap is stored in the active SLM data root under tag `loop:<name>`, making the full run history queryable via `slm recall` and visible on the dashboard.

## Framework Adapters

Nine Python packages implement each framework's native memory interface (LangGraph, Semantic Kernel, Microsoft Agent Framework, LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK and OpenAI Agents), backed by the local SLM data root. Each adapter delegates persistence to the same SLM engine that powers the CLI and MCP surfaces. Optional SLM providers, connectors, and backup retain their documented network behavior. See [Framework Adapters](framework-adapters.md).

## Privacy and compliance controls

SuperLocalMemory provides controls that may support a deployment's privacy and compliance program. It is not a legal certification. Operators must assess the configured system, use case, data flows, and surrounding services.

- **Local core path.** Mode A can keep memory content in the configured local data root during core operations.
- **Erasure commands.** `slm forget` and `slm delete` remove selected local records, and `slm gdpr erase` removes a whole profile. Backups, exports, caches, derived indexes, and provider logs require separate verification.
- **Auditability.** Retrieval and lifecycle surfaces expose local records and diagnostics, subject to release-specific verification.
- **Policy controls.** Provenance, retention, and access-policy features are available for operator configuration.

Mode C sends queries to your configured endpoint or cloud provider. In that mode, that provider's compliance posture applies to those queries.

## Database

Core memory is SQLite-backed:

```
~/.superlocalmemory/memory.db
```

The data root also contains configuration, logs, models, queues, derived indexes, and optional backend state. Back up or migrate the documented data root rather than copying only `memory.db`. The core database uses WAL (Write-Ahead Logging) mode for concurrent access.

Key table groups:

- **Core:** memories, sessions, profiles
- **Knowledge:** atomic_facts, graph_edges, canonical_entities, temporal_events
- **Durable ingestion:** ingestion_operations (M018 operation state and raw evidence)
- **Retrieval indexes:** SQLite FTS plus configured derived indexes
- **Math layers:** per-fact similarity and lifecycle columns on `atomic_facts`
- **Compliance and provenance:** trust scores, provenance records, audit and retention controls

M017 additively adds scope to CCQ consolidation blocks. M018 is the canonical
expand-phase ingestion contract. The legacy `pending.db` spool remains only as
an offline compatibility input; replay submits its evidence into M018 before it
is marked done.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

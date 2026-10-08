# Dashboard Coverage

The dashboard is a local operational surface over the SLM daemon (`slm dashboard`).
It helps an operator inspect and control a deployment; it does not replace
CLI/MCP traces or make a health signal a guarantee of recall quality.

## Navigation structure

The sidebar groups the panes as follows.

### Overview group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Answer Check | which check runs, a Try it panel, how often recall said "I don't have that", timings, recent checks. See [Answer check](answer-check.md) | `slm recall` output and the response fields |
| Dashboard | runtime summary, daemon identity, storage and runtime health, diagnostics and recent activity | `slm status`, `slm doctor` |
| Brain | consolidation, behavioral patterns, feedback and outcomes, reward and soft-prompt state | mode and configuration, and the underlying data |

### Memory group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Knowledge Graph | graph neighborhoods, entities, scenes and relationships | `slm trace` for query-time graph participation |
| Memories | Recall Lab, **Summaries** and **Saved views** tabs, memory browsing, filtering, update and delete, provenance-oriented inspection | write identity and scope policy for mutations |
| Entity Explorer | compiled entity summaries and timelines | source facts and entity recompilation |
| Multi-Agent Memory | which agents save and read memory | the agent ids on the memories themselves |

### Intelligence group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Skill Evolution | opt-in lineage, budgets, trigger and verification outcomes | the operator's evolution policy and model/provider boundary |

### System group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Health | math-layer health, trust signals, connected agents, a **Memory store** card (the post-upgrade check and **Repair now**) and component health | `slm health`, `slm db integrity`, deployment resource limits |
| Governance | tabs for **Lifecycle** (memory ages, compaction preview, transitions), **Access & Users**, **Trust**, **Compliance** (retention policies, GDPR export and erasure, audit trail) and **Ingestion** | effective RBAC role on each API call; erasure completeness |
| Optimize | cache and compression controls, cache state and savings telemetry | actual proxy and MCP routing and invalidation behavior |

### Integrations group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| MCP & Tools | the active MCP tool profile and its tool count, the available profiles, per-IDE MCP configuration snippets, and a pointer to Connected apps. Changing the tool profile means editing the client's `.mcp.json` and restarting the client | `slm connect --list`, [MCP Tools Reference](mcp-tools.md) |
| Connected apps | optional [Web access](remote-access/README.md): link this computer with GitHub, choose what an app may do, copy the server URL and instructions, see each connected app and remove its access, turn Web access off | a real recall from the app |
| Mesh Peers | configured peers, inbox and outbox, pending messages and locks | authenticated transport on the real peer topology |
| Bounded Loops | the optional Bounded Loops evidence bridge and recorded loop runs | `slm loop history` |

### Config group

| Workspace | Product surface | What to verify separately |
|---|---|---|
| Settings | mode, provider, answer check, auto-capture and auto-recall, scope defaults and other product configuration | effective configuration and secret management |
| Cloud Backup | backup destinations, schedule status, last backup time, manual trigger, encryption | actual backup file presence and restore path |

## Data and trust boundary

Panels can be empty when a feature is disabled, no data has been produced, or
the selected mode lacks a configured dependency. Recalled or displayed memory
is evidence, not an instruction authority. SLM keeps dynamic memory out of
high-trust IDE instruction files and renders it through the bounded context
path at runtime.

The **Access & Users** tab needs the login gate switched on to be meaningful in
a multi-user deployment. In personal mode (single user, loopback owner), the
RBAC controls are visible but authentication is not enforced.

## Release verification

For a release or deployment witness, validate each enabled surface end to end:

```bash
slm doctor
slm health
slm remember "dashboard witness" --sync --json
slm trace "dashboard witness"
```

For optional systems, also execute a real cache invalidation, peer
authentication exchange, provider call, adapter import, or Scale Engine
prepare/verify/promote/rollback cycle as applicable. Do not infer that these
subsystems are active solely because their dashboard workspace is visible.

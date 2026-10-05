<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/slm-wordmark-dark.svg">
    <img src="assets/branding/slm-wordmark-light.svg" alt="SuperLocalMemory: local-first memory for AI agents" width="360">
  </picture>
</p>

# SuperLocalMemory: governed, local-first memory for AI agents

Claude Code, Codex, Cursor and other MCP clients forget what they learned when a session ends. SuperLocalMemory (SLM) gives them one long-term memory that lives on your machine: it learns from use, enforces who may read and erase what, coordinates many agents, and says "I don't have that" instead of guessing.

In Mode A, core remember and recall make no model-provider call. Anything that sends data out is a choice you make, and the docs say exactly what goes. Four arXiv papers describe the design ([research](#research)).

[![PyPI](https://img.shields.io/pypi/v/superlocalmemory)](https://pypi.org/project/superlocalmemory/)
[![npm](https://img.shields.io/npm/v/superlocalmemory)](https://www.npmjs.com/package/superlocalmemory)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![arXiv V4](https://img.shields.io/badge/arXiv-2608.08253-b31b1b)](https://arxiv.org/abs/2608.08253)
[![arXiv V3](https://img.shields.io/badge/arXiv-2603.14588-b31b1b)](https://arxiv.org/abs/2603.14588)

**[Install](https://www.superlocalmemory.com/install)** · **[Product walkthrough](https://www.superlocalmemory.com/demo)** · **[Demo video](https://www.youtube.com/watch?v=PMWW_ypsL60)** · **[CLI proof](docs/QUICK_PROOF.md)** · **[Release notes](CHANGELOG.md)**

```bash
npm install -g superlocalmemory   # primary route (Node 18+, Python 3.12+); or: pipx install superlocalmemory
slm setup                         # pick Mode A to keep everything on this machine
slm connect cursor                # or claude-code, codex, windsurf, zed ... 12 IDEs
```

npm installs SLM into a package-owned virtual environment. The other primary route is pip in a Python virtual environment you activate: `python3 -m venv .venv`, activate it, then `python -m pip install superlocalmemory`. Repository clone: `./scripts/install.sh install` (macOS, Linux) or `.\scripts\install.ps1 -Action Install` (Windows); see [CONTRIBUTING.md](CONTRIBUTING.md).

Runs on Apple Silicon macOS, 64-bit Windows and 64-bit Linux; Intel Mac and 32-bit Windows are not supported (the pinned `cryptography` has no build for them). No Docker, no required graph database, no API key.

## 30-second example

```console
$ slm remember "We deploy the API blue-green; rollback is a DNS flip." --kind decision
Queryable ✓ 1 facts (operation=ebca8e80...).
$ slm remember "Always run the migration dry-run before a release." --kind rule
$ slm remember "Staging DB is Postgres 16 on port 5433."

$ slm recall "how do we roll back the API"
  1. [0.67] We deploy the API blue-green; rollback is a DNS flip.
  2. [0.53] Always run the migration dry-run before a release.
  3. [0.52] Staging DB is Postgres 16 on port 5433.

$ slm recall "what did we decide about rollback" --kind decision
  1. [0.54] We deploy the API blue-green; rollback is a DNS flip.

$ slm remember "Staging DB moved to Postgres 17 on port 5434." --replaces 0f57a17af3dd46e5
Replaced ✓ 1 fact(s) of 0f57a17af3dd46e5.
To undo, run:
  slm review-correction ... rollback 1
$ slm recall "which port does staging postgres use"
  1. [0.68] Staging DB moved to Postgres 17 on port 5434.
```

Real output from a fresh Mode A install, trimmed. Scores rank; they are not probabilities. While the embedding model loads, recall says `Incomplete search`.

### Watch the product walkthrough

[![Watch the SuperLocalMemory demo](https://img.youtube.com/vi/PMWW_ypsL60/hqdefault.jpg)](https://www.youtube.com/watch?v=PMWW_ypsL60)

Five minutes: install, setup, recall, cache and compression.

## Why SuperLocalMemory: the moats

A vector store answers "what is similar". AI agent memory must also answer: is this still true, who may see it, can it be erased with proof, and does the agent actually have the answer?

**1. Governed memory, not a vector store.** Roles per workspace, personal / shared / global scopes with default-deny cross-profile recall, GDPR erasure with HMAC-verifiable receipts, retention rules and a hash-chained audit log. The [V4 paper](https://arxiv.org/abs/2608.08253) measures what the governed write path costs. Code: `src/superlocalmemory/access/`, `compliance/`.

**2. A zero-LLM core built on published math.** Five retrieval channels, fusion and the learned ranker run without a language model; V3 scored 60.4% on LoCoMo with no LLM anywhere ([benchmarks](#benchmarks-v3)). Fisher-information scoring, sheaf contradiction detection and Langevin lifecycle dynamics added 12.7 points ([V3 paper](https://arxiv.org/abs/2603.14588)).

**3. It says "I don't have that."** The [answer check](docs/answer-check.md) decides whether the top results answer the question, on your Mac (Laya) or online (Jev), and recall reports `abstained` instead of a confident wrong answer.

**4. Memory that learns, and cannot quietly get worse.** A Thompson-sampling bandit tunes channel weights and a LightGBM ranker learns from reported outcomes. A retrained ranker is promoted only after a shadow A/B test on live recalls, and rolled back automatically if NDCG@10 drops 2% or more. Code: `learning/shadow_test.py`, `learning/model_rollback.py`.

**5. Memory with a sense of time.** Every fact records when it happened and when SLM learned it. Ask what was true last month (`--valid-at`) or what SLM knew before a date (`--known-as-of`). Unused memories fade and lose vector precision ([V3.3 paper](https://arxiv.org/abs/2604.04514)).

**6. Many agents, one coordinated memory.** Every write records its agent, with Bayesian trust scores against poisoning ([V2 paper](https://arxiv.org/abs/2603.02240)). SLM-Mesh gives parallel sessions messages, locks and shared state.

**7. "Done" means a gate passed.** Bounded loops repeat a task until an independent check (tests, a schema, a linter) passes, never on the agent's word, and store every lap as auditable memory.

**8. Context you don't pay for twice.** Exact cache and reversible compression work through MCP tools or a skill, with no proxy, so your full context window stays intact.

SLM is part of Qualixar's AI Reliability Engineering work: agent memory that is observable, bounded and honest about what it doesn't know.

## Works with your agents

| Surface | What you get | Docs |
|---|---|---|
| Editor plugins | Claude Code, Codex, VS Code / Copilot, Antigravity, Hermes. Each ships 12 skills, 4 sub-agents and session hooks | [IDE setup](docs/ide-setup.md), [Hermes](docs/hermes.md) |
| `slm connect <ide>` | Writes the MCP config for 12 IDEs, including Cursor, Windsurf, Zed, JetBrains, Gemini CLI and Claude Desktop | [IDE setup](docs/ide-setup.md) |
| MCP | stdio (`slm mcp`) or HTTP at `http://127.0.0.1:8765/mcp/`; profiles from 8 to 103 tools | [MCP tools](docs/mcp-tools.md) |
| Framework adapters | LangGraph, LangChain, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, Microsoft Agent Framework, Google ADK, OpenAI Agents | [Framework adapters](docs/framework-adapters.md) |
| Python SDK and HTTP API | `MemoryEngine` in your code; the local REST API | [API reference](docs/api-reference.md) |
| Auto-capture hooks | `slm hooks install` for Claude Code, `--agent codex` for Codex | [Auto-memory](docs/auto-memory.md) |

Claude Code memory in two commands: `claude plugin marketplace add qualixar/superlocalmemory`, then `claude plugin install superlocalmemory@qualixar`.

## Architecture

![SuperLocalMemory capability architecture: modes, seven operating layers, Scale Engine, SLM-Mesh, delivery surfaces and opt-in adapters](docs/assets/slm-v37-capability-architecture.png)

*SQLite + sqlite-vec are canonical; CozoDB and LanceDB are parity-gated projections; SLM-Mesh coordinates trusted peers rather than replicating a distributed database; connectors are opt-in.* [Architecture docs](docs/ARCHITECTURE.md).

## Everything SLM does

### Memory and recall

| Capability | What you get | Docs |
|---|---|---|
| Hybrid recall | Semantic, keyword (BM25), temporal, associative (Hopfield) and spreading-activation channels, fused by reciprocal rank and reranked. `slm trace` shows each channel's score | [Recall](docs/recall.md) |
| Memory kinds | Nine kinds: fact, event, status, opinion, rule, decision, how-to, plan, correction. "What did we decide" favours decisions | [Memory kinds](docs/memory-kinds.md) |
| Standing rules | Confirmed rules (up to 10) and decisions (up to 5) load into every new agent session | [Memory kinds](docs/memory-kinds.md#standing-rules-at-session-start) |
| Replace and correct | `--replaces <id>` retires an old fact, kept and undoable. Edits go through reviewed corrections with rollback | [Corrections](docs/reviewed-corrections.md) |
| Time travel | `--as-of`, `--known-as-of`, `--valid-at`, and `--window 7d` or a date range | [Recall](docs/recall.md#time-travel) |
| Recall filters | `--project`, `--saved-by`, `--about`, `--kind`; applied before the answer check | [Recall](docs/recall.md#narrowing-a-recall) |
| Project-aware recall | New in 4.1.21: memories from your current project rank first, hiding nothing. `--project` narrows and says when nothing matched | [Recall](docs/recall.md) |
| Summaries and saved views | New in 4.1.21: `slm summary session`, `day` or `project`, each citing its memory ids; `slm view create "Work log" "what did I ship" --window 7d`, then `slm view run` | [Summaries](docs/recall.md#summaries), [Views](docs/recall.md#saved-views) |
| Knowledge graph | Entities, aliases, scenes and timelines; Entity Explorer in the dashboard | [Architecture](docs/ARCHITECTURE.md) |
| Code graph | Index a repo, then ask for blast radius, callers, review context and code search by meaning | [MCP tools](docs/mcp-tools.md) |
| Modes and providers | A: no model calls. B: a model on this machine (Ollama by default, or another local OpenAI-compatible server). C: your own endpoint or a cloud provider. Multilingual embedders work | [Configuration](docs/configuration.md) |

Recalled text is untrusted evidence: before it reaches a prompt, secrets are redacted, forged boundary markers neutralised and provenance attached.

### Learning in memory

- **Adaptive ranking.** A contextual Thompson-sampling bandit picks channel weights per query type; a LightGBM ranker trains on `report_outcome` and `report_feedback`. On an unchanged store, the same question gets the same ranking.
- **Guarded promotion.** A candidate ranker runs in shadow and is promoted only on a measured win; a later NDCG@10 drop of 2% or more restores the previous model.
- **Soft prompts.** Consolidation mines behavioural patterns and turns stable ones into soft prompts for new sessions.
- **Forgetting.** An Ebbinghaus retention cycle and a Langevin lifecycle move neglected memories toward archive and pull used ones back.
- **Skill evolution (opt-in).** Measures agent skills, proposes revisions and keeps lineage, under a budget and blind verification. [Skill evolution](docs/skill-evolution.md)

### Governance

| Control | What it does | Docs |
|---|---|---|
| Roles | Admin, member and viewer per workspace. A role in one workspace grants nothing in another | [Teams](docs/rbac-teams.md) |
| Company mode | Every read and write is attributed to a signed-in person; turn on with "Require login" | [Company mode](docs/company-mode.md) |
| Profiles and scopes | Isolated workspaces. Memories are `personal` by default; `shared` and `global` are opt-in | [Profiles](docs/profiles.md), [Scopes](docs/shared-memory.md) |
| GDPR | `slm gdpr export` (Art. 15/20), `slm gdpr erase` (Art. 17) across every store, `slm gdpr verify` checks the erasure receipt | [Compliance](docs/compliance.md) |
| Retention and audit | Per-profile retention rules; a hash-chained audit trail of stores, recalls, changes and erasures; ABAC policy checks | [Compliance](docs/compliance.md) |
| EU AI Act posture | A per-mode technical report (data locality, generative AI use). Risk class stays "undetermined"; no legal verdict | [Compliance](docs/compliance.md) |
| Write gate | Install token, API keys and remote keys decide who may write | [Auth write gate](docs/auth-write-gate.md) |
| Evidence export | Checksummed JSONL bundles: `slm evidence export`, `verify`, `import` | [CLI reference](docs/cli-reference.md#data-and-evidence) |
| Backup and restore | Cloud backup to GitHub or Google Drive, encrypted before upload. A restore point before each store update | [Cloud backup](docs/cloud-backup.md), [Restore points](docs/restore-points.md) |

Engineering controls that support a compliance program, not a certification.

### Multi-agent: SLM-Mesh and bounded loops

**Shared memory with attribution.** Claude Code, Codex, Cursor and Hermes share one store; each memory records its agent (`SLM_AGENT_ID`) and the dashboard shows per-agent activity.

**SLM-Mesh** coordinates sessions on one machine, or several machines with a shared secret: `mesh_peers`, `mesh_send`, `mesh_inbox`, `mesh_state`, `mesh_lock`, `mesh_events`, `mesh_status`, `mesh_summary`. Messages route across machines; locks and state are per machine. [Multi-machine](docs/multi-machine.md)

**Bounded loops** end only when an independent gate passes (tests, a linter, a schema, a recall condition), never because the agent says it is done. Runs end DONE, HALT, PAUSE, KILLED or ERROR, with each lap stored under `loop:<name>`. Run `slm loop demo`, the `slm_loop_*` MCP tools or `/slm-loop`; the separate Bounded Loops product can store its finished runs here as read-only evidence. [CLI reference](docs/cli-reference.md#bounded-loops-v380), [Bounded Loops bridge](docs/bounded-loops-bridge.md)

### Answer check: Laya and Jev

Answer check adds a second step after ranking; choose one in **Settings → Answer check**:

- **On this Mac (Laya):** a small model, about 1.1 GB, on Apple Silicon. Nothing leaves the machine.
- **Online with Jev:** TypeSafe or OpenRouter, with your own key and an explicit consent box. Sends the question and the top 3 memories.
- **Off.**

Only one runs at a time and the online option never turns itself on. Results are never hidden: recall marks them `abstained` and your agent decides. A repeat question gets the same verdict. The **Answer Check** tab shows verdicts, timing against the 3-second recall ceiling and the "I don't have that" rate; it never stores questions or memory text. [Answer check](docs/answer-check.md)

### Context optimisation: cache and compression

| Feature | What it does |
|---|---|
| Exact cache | `slm_cache_get` / `slm_cache_set` keep tool output and file reads with a TTL; a hit skips the repeat call |
| Compression | `slm_compress` shrinks large output; safe mode keeps JSON and code intact. A reversible result returns an id, and `slm_retrieve` restores the exact original |
| Three surfaces | Works through a proxy (`slm wrap claude`), MCP tools or a skill. Only the proxy caches the main conversation turn |
| Savings | `slm optimize savings --since 7` reports tokens and cost saved |
| Memory compression | Vector precision follows retention: fading memories are quantized to fewer bits, and consolidation folds clusters of faded memories into one gist |

All of it fails open. [Optimize](docs/optimize-overview.md), [Proxy setup](docs/proxy-setup.md)

### Remote access and teams

`slm remote` serves memory to other computers over TLS only. Each client gets one named key, bound to one profile, read-only or read-write; `slm remote keys revoke <name>` applies on the next request. Remote callers authenticate to read and never see this computer's paths or account. [Remote access](docs/distributed-deployment.md#remote-access-over-tls-4120), [Deployment tiers](docs/deployment-tiers.md)

### Scale and operations

- **[Scale Engine](docs/scale-engine.md):** SQLite stays canonical. Optional CozoDB graph and LanceDB vector copies go through prepare, verify, promote and rollback, and serve recall only once they match SQLite.
- **[Dashboard](docs/DASHBOARD-COVERAGE.md):** `slm dashboard`, 16 panes including Answer Check, Brain, Knowledge Graph, Governance, Optimize and Mesh Peers.
- **Durable writes:** each save moves raw → queryable → enriching → complete with a receipt; a failed step keeps the raw evidence and retries.
- **[Operations](docs/troubleshooting.md):** `slm doctor`, `status`, `health`, `restart`, `ops`; stuck operations are listed and resolved.

## MCP memory server: tool profiles

Pick how many tools your agent sees with `SLM_MCP_PROFILE`. Counts come from the server itself.

| Profile | Tools | For |
|---|---:|---|
| `core` | 18 | Remember, recall, sessions, optimize, correction review |
| `code` | 38 | Core plus code graph, memory kinds, Brain evidence, profile switching, bounded loops |
| `mesh` | 8 | SLM-Mesh coordination only |
| `full` (and unset) | 56 | Memory, kinds, Brain, optimize, skill evolution, mesh, loops, views and summaries |
| `power` | 68 | Full plus administration, lifecycle and diagnostics |
| `whole` | 103 | Every registered tool |

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

For stdio clients use `{"command": "slm", "args": ["mcp"]}`. An unknown profile name stops startup and lists the valid ones.

## Privacy and security

What leaves your machine, and when:

| Data | Leaves only when |
|---|---|
| Recall question and top 3 memories | You turn on the online answer check (Jev) and tick its consent |
| Question and top 20 memories | You also turn on Jev reordering, which has its own notice |
| Memory text | You choose Mode C, a cloud embedder or reranker, an Ollama on another computer, or Jev kind typing (its own consent) |
| Encrypted backup files | You connect GitHub or Google Drive backup. Files are encrypted before upload |
| Mesh messages | You configure SLM-Mesh peers |

Model downloads send no memory content. Credentials in memory text are redacted on every outbound path. Outbound requests never follow redirects, and forwarded-for headers count only from a proxy you name. See [Security policy](SECURITY.md) and [encryption at rest](docs/SECURITY-encryption-at-rest.md).

## Benchmarks (V3)

These numbers come from the published **V3** architecture paper, which V4 still runs. They are not a fresh V4 package run.

| V3 configuration | LoCoMo | Scope |
|---|---:|---|
| Mode A, retrieval + GPT-4.1-mini answers | 74.8% | 10 conversations, 1,276 questions |
| Mode A, raw (no LLM anywhere) | 60.4% | 10 conversations, 1,276 questions |
| Mode C, cloud embeddings + GPT-4.1-mini | 87.7% | 1 conversation (Conv-30), 81 questions |

Method, category breakdown and ablations: [docs/benchmarks.md](docs/benchmarks.md) and [arXiv:2603.14588](https://arxiv.org/abs/2603.14588). A LoCoMo score is comparable only when the subset, answer model and judge match.

## Research

Four arXiv preprints by Varun Pratap Bhardwaj describe SLM, newest first:

1. **V4 (2026):** [SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents](https://arxiv.org/abs/2608.08253), with Garima Singh and Arun Pratap Bhardwaj. arXiv:2608.08253, DOI [10.5281/zenodo.21853302](https://doi.org/10.5281/zenodo.21853302). Multi-scope isolation, role-based access, verified erasure, hash-chained audit and bi-temporal recall, with their measured cost.
2. **V3.3 (2026):** [SuperLocalMemory V3.3: The Living Brain](https://arxiv.org/abs/2604.04514), arXiv:2604.04514. Biologically inspired forgetting, cognitive quantization, multi-channel retrieval without an LLM.
3. **V3 (2026):** [SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory](https://arxiv.org/abs/2603.14588), arXiv:2603.14588. Fisher-information retrieval, Langevin lifecycle, sheaf contradiction detection; the LoCoMo results above.
4. **V2 (2026):** [SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning](https://arxiv.org/abs/2603.02240), arXiv:2603.02240.

Cite the V4 paper with [CITATION.cff](CITATION.cff) or GitHub's "Cite this repository" button:

```bibtex
@article{bhardwaj2026superlocalmemory,
  title   = {SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents},
  author  = {Bhardwaj, Varun Pratap and Singh, Garima and Bhardwaj, Arun Pratap},
  journal = {arXiv preprint arXiv:2608.08253},
  year    = {2026},
  doi     = {10.5281/zenodo.21853302}
}
```

## Documentation

**Start:** [Getting started](docs/getting-started.md) · [IDE setup](docs/ide-setup.md) · [Linux install](docs/install-linux.md) · [Quick proof](docs/QUICK_PROOF.md) · [Migrating from V2](docs/migration-from-v2.md)

**Use:** [Recall](docs/recall.md) · [Memory kinds](docs/memory-kinds.md) · [Answer check](docs/answer-check.md) · [Auto-memory](docs/auto-memory.md) · [Shared memory](docs/shared-memory.md) · [Optimize](docs/optimize-overview.md) · [Per-agent optimize](docs/optimize-per-agent.md) · [pi.dev](docs/pi-dev-integration.md)

**Reference:** [CLI](docs/cli-reference.md) · [MCP tools](docs/mcp-tools.md) · [Configuration](docs/configuration.md) · [Errors](docs/errors.md) · [Troubleshooting](docs/troubleshooting.md) · [Distributed deployment](docs/distributed-deployment.md) · [Privacy diagnostics](docs/privacy-diagnostics.md)

**Design:** [Architecture](docs/ARCHITECTURE.md) · [Score contract](docs/retrieval-score-contract.md) · [Compliance](docs/compliance.md) · [Benchmarks](docs/benchmarks.md)

## Upgrade

```bash
npm update -g superlocalmemory    # or, in your activated venv: python -m pip install --upgrade superlocalmemory
slm restart && slm doctor
slm upgrade-hosts                 # preview IDE and plugin updates; nothing changes until you --apply
```

Upgrades never move or delete memory; an update that changes the store takes a restore point first. Plugins update through your editor; see [host upgrades](docs/host-upgrades.md).

## Platform support

Python 3.12+, and Node 18+ for npm, on the systems listed under the install block. Laya needs Apple Silicon; elsewhere use Jev or leave the check off. The default embedding model (about 500 MB) downloads on first use or with `slm warmup`.

## Contributing and license

Issues and pull requests are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities through [SECURITY.md](SECURITY.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

SLM is licensed under [AGPL-3.0-or-later](LICENSE). For a commercial license, see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Copyright (c) 2026 Varun Pratap Bhardwaj / [Qualixar](https://qualixar.com). Website: [superlocalmemory.com](https://www.superlocalmemory.com).

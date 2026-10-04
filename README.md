<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/slm-wordmark-dark.svg">
    <img src="assets/branding/slm-wordmark-light.svg" alt="SuperLocalMemory: local-first memory for AI agents" width="360">
  </picture>
</p>

# SuperLocalMemory: local-first memory for AI agents

Claude Code, Codex, Cursor and other MCP clients lose what they learned when a session ends. SuperLocalMemory (SLM) gives them one long-term memory that lives on your machine, and it can tell you when it doesn't have the answer instead of guessing.

[![PyPI](https://img.shields.io/pypi/v/superlocalmemory)](https://pypi.org/project/superlocalmemory/)
[![npm](https://img.shields.io/npm/v/superlocalmemory)](https://www.npmjs.com/package/superlocalmemory)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.08253-b31b1b)](https://arxiv.org/abs/2608.08253)

```bash
pipx install superlocalmemory     # or: npm install -g superlocalmemory   (both need Python 3.12+)
slm setup                         # pick Mode A to keep everything on this machine
slm connect cursor                # or claude-code, codex, windsurf, zed ... 12 IDEs
```

Runs on Apple Silicon macOS, 64-bit Linux and 64-bit Windows. No Docker, no required graph database, no API key.

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

Real output from a fresh Mode A install, trimmed. The id comes from `slm recall --json`. Scores are ranking signals, not probabilities. Right after a start, the embedding model is still loading: recall still answers from keyword and time channels and says `Incomplete search` until the model is ready.

## Why SuperLocalMemory

- **Local-first by default.** Memory lives in SQLite on your disk. In Mode A, core remember and recall make no model-provider call. Anything that sends data out is a choice you make, and the docs say exactly what goes.
- **It says "I don't have that."** The optional [answer check](docs/answer-check.md) judges whether the top results actually answer the question, on your Mac or through Jev, and flags it when they don't.
- **Memory with a sense of time.** Every fact records when it happened and when SLM learned it. Ask what was true last month (`--valid-at`), or what SLM knew before a date (`--known-as-of`). Replaced facts stop being served as current.
- **One memory, every agent.** The same store serves Claude Code, Codex, Cursor, Windsurf, Hermes and any MCP client, so a decision saved in one tool is there in the next.

SLM is part of Qualixar's AI Reliability Engineering work: agent memory that is observable, bounded and honest about what it doesn't know.

## Works with your agents

| Surface | What you get | Docs |
|---|---|---|
| Editor plugins | Claude Code, Codex, VS Code / Copilot, Antigravity, Hermes. Each ships 12 skills, 4 sub-agents and session hooks | [IDE setup](docs/ide-setup.md), [Hermes](docs/hermes.md) |
| `slm connect <ide>` | Writes the MCP config for 12 IDEs: Antigravity, Claude Code, Claude Desktop, Codex, Continue, Cursor, Gemini CLI, JetBrains, OpenCode, VS Code / Copilot, Windsurf, Zed | [IDE setup](docs/ide-setup.md) |
| MCP | stdio (`slm mcp`) or HTTP at `http://127.0.0.1:8765/mcp/`, with tool profiles from 8 to 101 tools | [MCP tools](docs/mcp-tools.md) |
| Framework adapters | LangGraph, LangChain, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, Microsoft Agent Framework, Google ADK, OpenAI Agents | [Framework adapters](docs/framework-adapters.md) |
| Python SDK and HTTP API | `MemoryEngine` in your own code; a local REST API behind the dashboard | [API reference](docs/api-reference.md) |
| Auto-capture hooks | `slm hooks install` for Claude Code, `--agent codex` for Codex | [Auto-memory](docs/auto-memory.md) |
| Other computers | `slm remote`: HTTPS listener with named, revocable keys for agents on another machine | [Remote access](docs/distributed-deployment.md#remote-access-over-tls-4120) |

On Claude Code, `slm connect claude-code` points you to the plugin: `claude plugin marketplace add qualixar/superlocalmemory`, then `claude plugin install superlocalmemory@qualixar`.

## Everything SLM does

| Capability | What you get | Docs |
|---|---|---|
| Hybrid recall | Semantic, keyword (BM25), temporal, associative (Hopfield) and spreading-activation channels, fused and reranked. `slm trace` shows each channel's score | [Recall](docs/recall.md) |
| Answer check | A second step that says whether the results answer the question. Laya on your Mac or Jev online, one at a time; a dashboard tab with history and the "I don't have that" rate | [Answer check](docs/answer-check.md) |
| Memory kinds | Nine kinds: fact, event, status, opinion, rule, decision, how-to, plan, correction. "What did we decide" favours decisions | [Memory kinds](docs/memory-kinds.md) |
| Standing rules | Confirmed rules (up to 10) and decisions (up to 5) load into every new agent session | [Memory kinds](docs/memory-kinds.md#standing-rules-at-session-start) |
| Replace and correct | `--replaces <id>` retires an old fact, kept and undoable. Edits go through reviewed corrections with rollback | [Corrections](docs/reviewed-corrections.md) |
| Time travel | `--as-of`, `--known-as-of`, `--valid-at`, and `--window 7d` or a date range | [Recall](docs/recall.md#time-travel) |
| Recall filters | `--project`, `--saved-by`, `--about`, `--kind`; applied before the answer check | [Recall](docs/recall.md#narrowing-a-recall) |
| Summaries | `slm summary session`, `day` or `project` | [Recall](docs/recall.md#summaries) |
| Knowledge graph | Entities, aliases, scenes and timelines; Entity Explorer in the dashboard | [Architecture](docs/ARCHITECTURE.md) |
| Code graph | Build a graph of a repo, then ask for blast radius, review context and code search by meaning | [MCP tools](docs/mcp-tools.md) |
| Learning | Ranking adapts to what you use; learned patterns become soft prompts; opt-in skill evolution | [Skill evolution](docs/skill-evolution.md) |
| Cache and compress | Exact cache and safe compression through a proxy (`slm wrap claude`), MCP tools or a skill; `slm optimize savings` | [Optimize](docs/optimize-overview.md) |
| Multi-agent and mesh | Each memory records which agent saved it. SLM-Mesh gives sessions and machines messages, locks and shared state | [Multi-machine](docs/multi-machine.md) |
| Profiles and scopes | Isolated workspaces. Memories are personal by default; shared and global are opt-in | [Profiles](docs/profiles.md), [Scopes](docs/shared-memory.md) |
| Bounded loops | `slm loop`: repeat a task until an independent check passes, with every lap stored as memory | [CLI reference](docs/cli-reference.md#bounded-loops-v380) |
| Modes and providers | A: local. B: local Ollama. C: a cloud provider. Any OpenAI-compatible embedder, including multilingual models | [Configuration](docs/configuration.md) |
| Backup and restore | Cloud backup to GitHub or Google Drive, encrypted before upload. Restore points before each store update, and a safe path back to the previous version | [Cloud backup](docs/cloud-backup.md), [Restore points](docs/restore-points.md) |
| Evidence export | Checksummed JSONL bundles: `slm evidence export`, `verify`, `import` | [CLI reference](docs/cli-reference.md#data-and-evidence) |
| Teams and governance | Admin, member and viewer roles; sign-in; `slm gdpr export`, `erase` and `verify`; retention rules; hash-chained audit log | [Teams](docs/rbac-teams.md), [Compliance](docs/compliance.md) |
| Dashboard | `slm dashboard`: 16 panes, including Answer Check, Brain, Knowledge Graph, Memories, Health, Governance, Optimize, Mesh Peers and Cloud Backup | [Dashboard](docs/DASHBOARD-COVERAGE.md) |
| Operations | `slm doctor`, `status`, `health`, `restart`, `ops`; stuck operations listed and resolved | [Troubleshooting](docs/troubleshooting.md) |
| Scale Engine | Optional CozoDB and LanceDB copies, promoted only after they match SQLite | [Scale Engine](docs/scale-engine.md) |

## MCP memory server: tool profiles

Pick how many tools your agent sees with `SLM_MCP_PROFILE`. Counts come from the server itself.

| Profile | Tools | For |
|---|---:|---|
| `core` | 18 | Remember, recall, sessions, optimize, correction review |
| `code` | 38 | Core plus code graph, memory kinds, Brain evidence, profile switching, bounded loops |
| `mesh` | 8 | SLM-Mesh coordination only |
| `full` (and unset) | 54 | Memory, kinds, Brain, optimize, skill evolution, mesh, loops |
| `power` | 66 | Full plus administration, lifecycle and diagnostics |
| `whole` | 101 | Every registered tool |

```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```

For stdio clients use `{"command": "slm", "args": ["mcp"]}`. An unknown profile name stops startup and lists the valid ones.

## Answer Check: memory that says "I don't have that"

Relevance and "does this answer it" are different questions. The closest memory on file can still be the wrong answer, and an agent that treats it as the answer makes things up with confidence.

Answer check adds a second step after ranking. It reads the top results and decides whether they answer the question. Choose one, in **Settings → Answer check** in the dashboard:

- **On this Mac (Laya):** a small model, about 1.1 GB, on Apple Silicon. Nothing leaves the machine.
- **Online with Jev:** TypeSafe or OpenRouter, with your own key and an explicit consent box. Sends the question and the top 3 memories.
- **Off.**

Only one runs at a time, and nothing turns the online option on by itself. Each option shows whether it is ready, has a **Test** button, and is applied live with one **Save**. Results are never hidden: recall returns them with `abstained` and `abstention_reason` set, and your agent decides what to do. The same question gets the same verdict on a repeat. The **Answer Check** tab shows each verdict, its timing against the 3-second recall ceiling, and how often it said "I don't have that" over 24 hours, 7 days or 30 days. The history keeps outcomes and timings, never your questions or memory text.

## Privacy and security

What leaves your machine, and when:

| Data | Leaves only when |
|---|---|
| Recall question and top 3 memories | You turn on the online answer check (Jev) and tick its consent |
| Question and top 20 memories | You also turn on Jev reordering, which has its own notice |
| Memory text | You choose Mode C, a cloud embedder or reranker, an Ollama on another computer, or Jev kind typing (its own consent) |
| Encrypted backup files | You connect GitHub or Google Drive backup. Files are encrypted before upload |
| Mesh messages | You configure SLM-Mesh peers |

Model downloads (embedding model, Laya) fetch files but send no memory content. Credentials found in memory text are redacted on every outbound path, and a provider's key goes only to that provider.

Since 4.1.20, other computers must authenticate to read memory, not only to write it. SLM ignores forwarded-for headers unless you name a trusted proxy, so a proxy can't make a remote caller look local. SLM's own outbound requests never follow redirects, and the remote listener never answers with one, so a key is never sent through a redirect. Remote callers never see this computer's paths, account or environment. Remote agents connect through `slm remote`: TLS only, one named key per client, read-only keys available, and `slm remote keys revoke <name>` takes effect on the next request. See [Remote access](docs/distributed-deployment.md#remote-access-over-tls-4120), [Security policy](SECURITY.md) and [encryption at rest](docs/SECURITY-encryption-at-rest.md).

## Benchmarks (V3)

These numbers come from the published **V3** architecture paper, which V4 still runs. They are not a fresh V4 package run.

| V3 configuration | LoCoMo | Scope |
|---|---:|---|
| Mode A, retrieval + GPT-4.1-mini answers | 74.8% | 10 conversations, 1,276 questions |
| Mode A, raw (no LLM anywhere) | 60.4% | 10 conversations, 1,276 questions |
| Mode C, cloud embeddings + GPT-4.1-mini | 87.7% | 1 conversation (Conv-30), 81 questions |

Method, category breakdown and ablations: [docs/benchmarks.md](docs/benchmarks.md) and [arXiv:2603.14588](https://arxiv.org/abs/2603.14588). A LoCoMo score is comparable only when the subset, answer model and judge match.

## Research

SLM's design is described in the V4 paper, [SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents](https://arxiv.org/abs/2608.08253) (DOI [10.5281/zenodo.21853302](https://doi.org/10.5281/zenodo.21853302)). Earlier preprints: [information-geometric foundations, V3](https://arxiv.org/abs/2603.14588), [the Living Brain, V3.3](https://arxiv.org/abs/2604.04514), and [trust and behavioural foundations, V2](https://arxiv.org/abs/2603.02240). Cite it with [CITATION.cff](CITATION.cff) or GitHub's "Cite this repository" button:

```bibtex
@article{bhardwaj2026superlocalmemory,
  title   = {SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents},
  author  = {Bhardwaj, Varun Pratap and Singh, Garima and Bhardwaj, Arun Pratap},
  journal = {arXiv preprint arXiv:2608.08253},
  year    = {2026},
  doi     = {10.5281/zenodo.21853302}
}
```

## Upgrade

```bash
pipx upgrade superlocalmemory     # or: npm update -g superlocalmemory
slm restart && slm doctor
slm upgrade-hosts                 # preview IDE and plugin updates; nothing changes until you --apply
```

Upgrades never move or delete memory, and an update that changes the store takes a restore point first. Plugin updates follow your editor; see [host upgrades](docs/host-upgrades.md).

## Contributing and license

Issues and pull requests are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities through [SECURITY.md](SECURITY.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

SLM is licensed under [AGPL-3.0-or-later](LICENSE). For a commercial license, see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Copyright (c) 2026 Varun Pratap Bhardwaj / [Qualixar](https://qualixar.com). Website: [superlocalmemory.com](https://www.superlocalmemory.com).

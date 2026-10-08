# Auto-Memory

With hooks installed (`slm hooks install`, or the Claude Code and Codex plugins), SuperLocalMemory recalls relevant context at the start of a session and before your prompts, and your assistant can send decisions, bug fixes and preferences to it for capture. Hooks are explicit: nothing is installed until you consent.

---

## How Auto-Capture Works

When your assistant sends conversation text to SLM's `observe` step (through hooks or the MCP tool), SLM stores three kinds of information as memories:

| What gets captured | Example |
|-------------------|---------|
| **Decisions** | "Let's use WebSocket instead of SSE" |
| **Bug fixes** | "The crash was caused by a null pointer in the auth middleware" |
| **Preferences** | "I prefer functional components over class components" |

### What does NOT get captured

- Casual conversation and content that matches none of the decision, bug-fix or preference patterns
- Repeated content within the observation debounce window
- A category you turned off (see [Configuration](#configuration))
- Anything below the minimum confidence

### How the system decides what to capture

Auto-capture admission is a lightweight rules step. It matches configured
decision, bug-fix, and preference patterns and returns a category, reason, and
confidence. Rejected observations are not acknowledged as stored.

Accepted observations are submitted durably to M018 before the caller receives
`captured: true`. This is separate from materialization: the operation first
becomes `queryable`, then enrichment advances it to `complete` or records a
retryable `failed` state. Entropy, consolidation, graph, temporal, provenance,
and projector work belong to materialization, not admission.

## How Auto-Recall Works

When the recall hooks are installed, SLM searches for relevant memories when a session starts and before your prompts, and injects what it finds as context. Very short acknowledgements ("ok", "thanks") are skipped. MCP clients that do not run hooks get the session-start version through `session_init`.

### The flow

```
You ask a question
    |
    v
SLM runs a recall query using your question as the search input
    |
    v
Relevant memories are injected into the AI's context window
    |
    v
The AI responds with awareness of your past decisions, preferences, and project context
```

### What this looks like in practice

**Without SLM:**
> You: "What database should I use for the new service?"
> AI: Generic advice about PostgreSQL vs MySQL vs MongoDB...

**With SLM:**
> You: "What database should I use for the new service?"
> AI: "Based on your previous decision to standardize on PostgreSQL 16 (stored March 5), and your preference for managed services on AWS (stored February 20), I'd recommend Amazon RDS for PostgreSQL. Your auth service and payment service already use PostgreSQL, so this keeps the stack consistent."

The AI did not "remember" this on its own. SLM injected the relevant memories before the AI generated its response.

### Memory is evidence, not instruction

Recalled content can include old prompts, imported text, or hostile
instructions. SLM therefore renders it inside one reference-only untrusted
evidence boundary, with provenance, size budgets, recognized-secret redaction,
and forged-boundary neutralization. If the mandatory renderer fails, SLM omits
the memory context instead of falling back to raw text.

IDE instruction files contain only static product protocol. SLM retrieves
dynamic memory through MCP at runtime rather than copying recalled text into a
trusted Cursor, Copilot, or Antigravity rules file.

### Auto-recall and Answer check

When [Answer check](answer-check.md) is on (off until an on-device install
has passed its check), auto-recall puts the verdict above the memories as one
plain line. If the check decided none of the matching memories answer the
question, the line says so and tells the assistant to say it does not have the
answer or to ask, rather than presenting an unrelated memory as the answer.
The memories themselves are still listed, and the next explicit `recall`
returns the same full result set with the same verdict.

## Configuration

### Toggle auto-capture and auto-recall

The dashboard's **Settings** page has switches for auto-capture (enable, capture
decisions, capture bug fixes) and auto-recall (enable, recall on session start).
They are stored under `rules` in `~/.superlocalmemory/config.json`:

```json
{
  "rules": {
    "auto_recall": {
      "enabled": true,
      "on_session_start": true,
      "on_every_prompt": false,
      "max_memories_injected": 10,
      "relevance_threshold": 0.3
    },
    "auto_capture": {
      "enabled": true,
      "capture_decisions": true,
      "capture_bugs": true,
      "capture_preferences": true,
      "capture_session_summary": true,
      "min_confidence": 0.5
    }
  }
}
```

The values shown are the defaults. Set `enabled` to `false` to turn either off.
The MCP tools read these rules: `session_init` honors `enabled`,
`on_session_start` and `relevance_threshold`, and `observe` honors the capture
settings. The hook scripts for Claude Code and Codex are installed or removed with
`slm hooks install` and `slm hooks remove`. When auto-capture or auto-recall is
off, `slm remember` and `slm recall` still work.

| Setting | Default | Description |
|---------|---------|-------------|
| `auto_recall.relevance_threshold` | `0.3` | Minimum relevance (0.0 to 1.0) for `session_init` to include a memory. Lower gives more memories, possibly less relevant |
| `auto_capture.min_confidence` | `0.5` | Minimum confidence for `observe` to store something. Lower captures more |

> **Recall result count:** The default is 20 results per query. Override per call with the `--limit N` flag (CLI) or the `limit` parameter (MCP `recall` tool). There is no config file key for this default.

## Manual Override

You always have full control:

```bash
# Explicitly store something
slm remember "The API rate limit on production is 500 req/min, staging is 100 req/min"

# Explicitly recall
slm recall "rate limits"

# Delete by query (fuzzy) — preview then confirm
slm forget "old rate limit" --dry-run
slm forget "old rate limit" --yes

# Delete a specific fact by exact ID (precise)
slm delete QmFzZTY0RmFjdElk --yes
slm list --limit 20   # shows fact IDs for delete/update
```

Manual operations work regardless of auto-capture/auto-recall settings.
`slm forget` matches by query text (fuzzy); it does not take `--id`. Use `slm delete <fact_id>` for an exact ID.

## Learning Over Time

SLM's adaptive learning system observes which memories are recalled frequently, which are marked helpful or outdated, and adjusts its behavior:

With adaptive ranking enabled, memories that were reported helpful rank higher
in later recalls. Usage patterns are also kept for the dashboard's Living Brain
panel.

Learning is driven by **explicit feedback**. Recall itself is deliberately
read-only — it never opens a database writer, so that a busy recall path cannot
contend with the dashboard — which means reported feedback is what grows the
learning store. Feedback is recorded through the MCP `report_feedback` tool:

```
report_feedback(fact_id="<id>", feedback="relevant")     # helpful
report_feedback(fact_id="<id>", feedback="irrelevant")   # not useful
report_feedback(fact_id="<id>", feedback="partial")      # somewhat relevant
```

Each call returns `total_signals` and the current `phase`. The ranker's phases
are counted at 50 signals (phase 2) and 200 (phase 3). Adaptive ranking changes
result order only when you enable it with the `SLM_RANKING` environment variable
(`v1`, `v2` or `v2-ensemble`); otherwise feedback is recorded but does not
reorder recall.

You can see what the system has learned in the **Living Brain** panel of the
dashboard (`slm dashboard`), which reads the same store.

> **Note:** the `slm patterns`, `slm useful` and `slm learning` commands
> described in some older V2 documentation do not exist. Use `report_feedback`
> and the dashboard instead.

## Privacy

Core auto-capture and recall storage use the configured local data root. Mode A
does not require a cloud model provider for core memory operations, but optional
connectors, cloud backup, proxy providers, dependency/model downloads, and
other explicitly enabled integrations can use the network. Mode C sends the
constructed model request—including selected memory evidence—to your
configured endpoint or cloud provider. Review that endpoint or provider's
retention and privacy terms before use.

### Feedback query pseudonymization

Learning feedback stores a pseudonymized grouping key, not the raw query text.

- **Mechanism:** per-install keyed HMAC (`hmac.digest(key, query.encode("utf-8"), "sha256").hex()[:16]`) in `src/superlocalmemory/learning/feedback.py: _hash_query` — 16 hex characters (64-bit), **not encryption** (cannot be reversed to the query).
- **Correlation:** within a single install, the same query text produces the same `query_hash`, so repeat-query grouping is possible; across installs the keys differ.
- **Key storage:** 32-byte key at `.feedback-hash-key` beside the DB (`db_path.parent / ".feedback-hash-key"`), written atomically with mode `0600` (`src/superlocalmemory/learning/feedback.py: _load_or_create_hash_key`). Existing keys are re-chmodded to `0600` on load.
- **Read-only data root fallback:** if the key cannot be persisted (for example, a read-only data root), the collector falls back to a process-local key and logs a warning; this preserves privacy but **loses cross-restart grouping** because the next process generates a new key. See `SECURITY.md` for the operational caveat.

---

*SuperLocalMemory — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

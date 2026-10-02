# Answer Check

New in 4.1.18. Lets recall say "I don't have that" instead of guessing.
Until you choose, it uses the on-device check only once an on-device install
exists that passed its check; until then it behaves exactly as Off.

Recall already ranks memories by relevance. Answer check adds a separate
decision on top of the best few results: not "is this related" but "does
this actually answer the question." Results are never hidden or removed —
the response just also says, honestly, whether the top match is likely an
answer or just the nearest thing on file. That distinction is the point of
AI Reliability Engineering: an agent that can tell "related" from "answered"
fails safely instead of confidently making things up.

## The three choices (Settings → Answer check)

| Option | Where it runs | Needs |
|---|---|---|
| **On this Mac** (recommended) | Locally, about 1.1 GB, one-time download; about 1.1 GB of memory while on | A Mac with Apple Silicon |
| **Online with Jev** | A hosted provider you choose (TypeSafe or OpenRouter) | Your own key, plus explicit consent |
| **Off** | Nowhere | Nothing |

Only one runs at a time — turning one on turns the others off. Before anyone
chooses, the starting state (`auto`) runs the on-device check only once an
install exists that passed its check, and otherwise behaves exactly as Off.
It is never offered as a fourth option: the dashboard shows it as what it is
doing, and once you pick one of the three your choice sticks. It never
resolves to the online option by itself; that one always requires a
deliberate choice, a saved key, and a ticked consent box.

The settings live in one file for every mode (`answer_check.json` in the data
folder), so switching Mode A/B/C never brings back a withdrawn consent or
resets a choice. Withdrawing consent turns the online check off at once; SLM
confirms it stopped before saying so.

## Reordering results (optional, online only)

Inside **Online with Jev** there's one more switch: **Also use Jev to
reorder results** — off by default, never turned on by an upgrade. When it's
on, the same single request also asks Jev which of the top memories answers
most directly, and recall puts them in that order; the answer check then
looks at the three that now come first. It sends your question and the top
20 memories (instead of 3), so it has its own notice:

> When on, each recall sends your question and its top 20 memories to
> TypeSafe to choose the best order. Don't use this for client, confidential
> or personal material.

On a public benchmark of long conversations (LoCoMo, 1,531 questions), the
memory that answers the question came first 62% of the time with reordering
on, against 49% without it; it adds about 0.4 seconds to a recall.

Nothing is ever removed — only the order of the top results changes. A
memory that is nothing but a credential is left out of the reorder
entirely — never sent, never a reason to skip the rest. If the
provider is slow, unreachable, or answers something unexpected, results come
back in their usual order. It switches itself off (until you tick it again)
when you leave the online option, remove the key, untick the Jev notice, or
change provider. On this Mac never reorders. The System card shows
"· reordering on" while it runs.

## Setup

Everything is dashboard-driven; there's no config file to hand-edit.
**Settings → Answer check** has a **Set up** button for the on-device
option (with a progress bar while it downloads), a **Remove** button, and
an **Advanced — use an existing install** path for networks that block the
download (checked in the background; it must be the `python` inside a Python
environment, owned by you, not in a shared or temporary folder). The hosted
option has provider selection, a key field (only the last 4 characters are
ever shown back), **Test** (each test is one billed request), and a required
consent checkbox. The dashboard's System card shows a one-line summary of
which option is actually running. Changes need the dashboard's own
credential, which it sends automatically.

On Apple Silicon, the first-run setup wizard offers to install the
on-device option for you (defaults to yes; once it passes its check, the
answer check is on). Elsewhere, it points you to the same settings page for
the hosted option. An unattended setup (`slm setup --auto`, CI, no
terminal) skips that download unless `SLM_INSTALL_LAYA=1` is set.

The on-device model loads when the daemon starts, or the moment you switch
the check on — never inside a one-shot CLI command.

## Privacy

> **Shared Macs:** on a Mac used by several people with separate accounts, another account can change these settings through the local service. See [Shared Macs](https://github.com/qualixar/superlocalmemory/blob/main/docs/SECURITY-encryption-at-rest.md#shared-macs-and-other-multi-user-computers).

- **On this Mac**: nothing leaves the machine.
- **Online with Jev**: each checked recall sends your question and the top
  3 memories to the provider you chose — the top 20 if you also turned on
  reordering. Known vendor key formats, connection-URL passwords, and
  values after a credential label (`password=`, `token=`, `api_key: …`) are
  stripped first; a hex string or UUID only counts as a credential after a
  key-like label, so a commit id or record id is sent as it is. Personal
  details are not stripped — don't use this option on sensitive stores. If
  what the check would read includes a memory owned by another profile
  (shared or global), the check is skipped for that recall instead of
  sending it.
- **Off**: unchanged from before this feature existed.

The check only ever looks at results the recall was already about to
return to you; it never runs its own search and never sees more than scope
rules already allow.

## What you'll see

Response fields: `abstained`, `abstention_reason` (`judged_insufficient` —
found memories, none of them answer it; `evidence_floor` / `no_candidates` —
nothing confident was found at all, unrelated to this feature),
`answer_confidence`, `calibration_status`, `calibration_id`,
`answer_check_status`, `reranker_status`, `local_reranker_status`. See
[Retrieval Score Contract](Retrieval-Score-Contract).

Every recall carries `answer_check_status`: `judged` (a verdict applied),
`off` (no check runs), `skipped` (on but not asked this time — loading
context, the daemon's own start-up recalls, no results, too little of the
recall's time left, or the online check would have had to read another
profile's memory), `busy` (the on-device check was already answering
another recall), `warming` (still loading), or `unavailable` (timeout,
error, no key, or can't run here). Skipping costs the verdict only — recalls
that only load context (session start, auto-loaded context, the per-prompt
hook) and the daemon's own start-up recalls are never judged, sent, or
billed. Only the exact weights this release measured can abstain; other
weights still return a confidence, just never `abstained: true`.

`reranker_status` names which step produced the final order —
`jev_listwise` when the online check's optional reorder replaced it;
`local_reranker_status` keeps the local reranker's own status for that
recall, empty when nothing replaced it.

CLI (`slm recall`, `slm trace`) and MCP (`session_init`'s `context`) add one
plain-text line when the check has run:

```
Answer check: likely answered (confidence 0.91).
```

```
Answer check: none of these memories answers the question (confidence 0.18).
Say you don't have it, or ask — don't present these as the answer.
```

Agent instructions shipped for Claude Code, Codex, Cursor, Copilot,
Antigravity, and Hermes all say the same thing: don't present an abstained
set as the answer. A [bounded loop](Bounded-Loops)'s gate honors this too —
a recall judged insufficient can never pass it, and each lap asks the check
once, for the verdict only, never for the optional reorder.

## Third-party attribution

The on-device model, Laya, is downloaded separately at setup — it is not
bundled with SLM. It's built by Convai Innovations, under the Apache
License 2.0.

## Troubleshooting

- **No Apple Silicon** → use Online with Jev, or leave it off.
- **Download blocked by a company network** → Advanced — use an existing
  install, pointed at a copy from another machine.
- **Key test fails** → the message is plain: not accepted, no credit left,
  or rate-limited — act on what it says.
- **"Answer check runs in another SLM process"** → expected with more than
  one SLM process using the same data folder; only one runs the model.
- **"The saved key file was changed outside SLM"** → choose Remove key, then
  paste the key again.
- **Freeing the disk space** → the model and its environment (~1.1 GB) stay
  in `runtimes/laya` in the data folder until you choose **Remove**;
  switching to Online or Off, uninstalling SLM or downgrading leaves them.
  After uninstalling, delete `~/.superlocalmemory/runtimes/laya` (that folder
  only).

Full reference: [`docs/answer-check.md`](https://github.com/qualixar/superlocalmemory/blob/main/docs/answer-check.md).

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*

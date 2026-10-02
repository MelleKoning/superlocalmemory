# Answer Check
> SuperLocalMemory V4 Documentation
> https://superlocalmemory.com | Part of Qualixar

Lets your memory say "I don't have that" instead of guessing.

---

## What it is

Recall already ranks your stored memories by relevance and returns the best
matches. Answer check adds a second, separate step: it looks at the top few
results and decides whether they actually answer your question — not just
whether they're related to it.

Relevance and "does this answer it" are different questions. A memory can be
the closest match SLM has and still not answer what you asked — recall still
returns it (nothing is ever hidden), but the response now says, honestly,
whether that top result is likely to be an answer or just the nearest thing
on file. That distinction is the core of AI Reliability Engineering: an
agent that knows the difference between "found something related" and
"found the answer" fails safely instead of confidently making things up.

This is new in 4.1.18. Until you choose, it is in its starting state:
it uses the on-device check only once an on-device install exists that passed
its check (set up from the dashboard, or an existing install you pointed it
at); until then it behaves exactly as Off. It never turns the online option on
by itself.

## The three choices

Settings → Answer check offers three options:

- **On this Mac** (labeled "private (recommended)") — a small model runs
  locally. About 1.1 GB, downloaded once, and about 1.1 GB of memory while
  it is on. Nothing about your questions or memories leaves the machine.
  Needs a Mac with Apple Silicon.
- **Online with Jev** (labeled "needs a key") — your question and the top 3
  memories for it are sent to a hosted provider you choose and pay for
  directly (TypeSafe or OpenRouter), using your own key. Off by default, and
  it stays off until you add a key and tick a consent box.
- **Off** — no answer check runs.

**Before you choose** (every install, upgraded or new, starts here): the
on-device check runs only once an on-device install exists that passed its
check; until then nothing runs, exactly as with Off. This starting state is
never offered as a fourth option — the dashboard shows it as what it is doing
("On this Mac" once set up, otherwise "Off"). Once you pick one of the three,
your choice sticks: picking **Off** means a later **Set up** does not turn the
check on until you pick **On this Mac**.

Only one of these ever runs at a time. Turning one on turns the other off —
there is no mode where both run together, and no automatic mode that turns
the online option on by itself. Choosing "On this Mac" never requires a key
or an internet connection once the model is downloaded.

## Reordering results with Jev (optional)

Inside the **Online with Jev** option there is one more switch: **Also use
Jev to reorder results**. It is off by default, and upgrading never turns it
on.

When it's on, the request the answer check already makes also asks Jev which
of your top memories answers the question most directly, and recall puts
them in that order. It is still one request per recall, not two. The answer
check then looks at the three memories that now come first — the ones you
see first.

- **What it sends:** your question and its top 20 memories (instead of 3;
  fewer when a recall returns fewer) to the provider you chose. Each memory
  is shortened, and credential-shaped text is stripped out first, the same as
  for the answer check. A memory that is nothing but a credential is left
  out of the reorder entirely — never sent, and never a reason to skip the
  rest.
- **What it does for recall:** on a public benchmark of long conversations
  (LoCoMo, 1,531 questions), the memory that answers the question came first
  62% of the time with reordering on, against 49% without it. It adds about
  0.4 seconds to a recall.
- **What never changes:** nothing is removed or hidden. Only the order of the
  top results changes; everything below them stays where it was.
- **If anything goes wrong** — the provider is slow, unreachable, or answers
  something unexpected — recall returns your results in their usual order,
  exactly as if the switch were off.
- **On this Mac never reorders.** This switch only exists inside the online
  option.

It has its own notice, because it sends more than the answer check alone:

> When on, each recall sends your question and its top 20 memories to
> TypeSafe to choose the best order. Don't use this for client, confidential
> or personal material.

(The notice names the provider you actually chose.)

It switches itself off — and stays off until you tick it again — when you
switch the answer check to **On this Mac** or **Off**, remove the key, untick
the Jev notice, or change provider. The dashboard's System card line ends in
**· reordering on** while it is running.

## Setting it up

Everything here is configured from the dashboard. There is no config file
to hand-edit for this feature.

Open **Settings → Answer check**. You'll see:

1. **On this Mac — private (recommended)**, **Online with Jev — needs a
   key**, and **Off** as three radio options.
2. An **On this Mac** panel with a **Set up** button. Click it and SLM
   downloads and verifies the model in the background — you'll see a
   progress bar. When it finishes, the status reads "Ready." A **Remove**
   button stops the model and takes it back off your disk. An **Advanced**
   section lets you point at a Python install and model you already have,
   if you'd rather not download again (see *Company network blocks the
   download*, below). Checking that install loads the model once, so it
   runs in the background with a progress bar too; the dashboard tells you
   when it passes or why it didn't.
3. An **Online with Jev** panel: pick a provider (TypeSafe or OpenRouter),
   paste a key, and save it — only the last four characters are ever shown
   back to you. **Test** sends one check request so you know the key
   works before relying on it; each test is a real request to your provider
   and uses a little of your credit. A consent checkbox spells out exactly what
   gets sent before you can turn this option on. Below it, **Also use Jev
   to reorder results** is a separate, optional switch with its own notice
   (see *Reordering results with Jev*, above).

The dashboard's System card also shows an **Answer check** line summarizing
which option is active and whether it's ready.

If you're on a Mac with Apple Silicon and haven't set anything up yet, the
first-run setup wizard offers to install the on-device option for you
(Step 4d). It defaults to yes, and once the install passes its check the
answer check is on — say no and you can always do it later from the
dashboard. On any other machine, the wizard points you to the online option
instead.

An unattended setup (`slm setup --auto`, CI, or any run without a terminal)
never starts that download on its own; it's skipped and reported as such,
and you can set it up later from the dashboard. Set `SLM_INSTALL_LAYA=1`
before running setup to install it automatically in that case too.

## Privacy, by choice

> **Shared Macs:** on a Mac used by several people with separate accounts, another account can change these settings through the local service. See [Shared Macs](https://github.com/qualixar/superlocalmemory/blob/main/docs/SECURITY-encryption-at-rest.md#shared-macs-and-other-multi-user-computers).

- **On this Mac**: nothing about your questions or memories ever leaves
  the machine for this feature.
- **Online with Jev**: each recall that runs the check sends your question
  and the top 3 memories for it to the provider you chose. Known vendor key
  formats, a password inside a connection URL, and a value after a
  credential label (`password=`, `token=`, `api_key: …`) are stripped out
  first; a hex string or UUID only counts as a credential when it follows a
  key-like label, so a commit id, a checksum, or a record id is sent as it
  is. Ordinary personal details are not stripped, so don't turn this on for
  memory stores that hold client, confidential, or personal material. If
  what the check would read includes a memory owned by another profile — a
  shared or global result — the check is skipped for that recall instead of
  sending it: your consent to send covers your own memories, not a
  teammate's. You must tick the consent box before this option can run, and you can
  withdraw consent at any time, which turns it off immediately: SLM checks
  that the online check has actually stopped before it confirms, and if it
  ever cannot confirm that, the dashboard says so and asks you to restart
  SLM. Switching SLM's mode (A, B or C) never turns it back on. If you also
  turn on reordering, each recall sends the top 20 memories instead of 3 —
  it has its own notice, and it is off unless you tick it.
- **Off**: no change from how recall has always worked.
- Either way, the check only ever sees the handful of results recall was
  already about to return to you — it never does its own separate search,
  and it never sees more than you do.

## What your AI assistant sees

When the check is on, a recall response carries a few extra fields, and the
CLI and your assistant's session context get one extra line when there's
something to say:

- If none of the top results answer the question, you'll see:
  *"Answer check: none of these memories answers the question (confidence
  X.XX). Say you don't have it, or ask — don't present these as the
  answer."*
- If the top result looks like a real answer, you'll see:
  *"Answer check: likely answered (confidence X.XX)."*
- If the check hasn't run (it's off, or it's still starting up), there's no
  extra line at all — recall behaves exactly as it always has.

The memories themselves are never removed or hidden because of this check.
It adds a signal; it doesn't filter your results. The instructions shipped
for Claude Code, Codex, Cursor, Copilot, Antigravity, and Hermes all tell
the connected assistant the same thing: if the check says a set of memories
doesn't answer the question, say so, or ask — never present it as the
answer anyway. A [bounded loop](bounded-loops-bridge.md) waiting on a memory
honors this too: a recall the check calls insufficient can never pass that
loop's gate, even if it would otherwise look like a match. Each lap asks the
check once, for the verdict only — never for the optional reorder, even
when reordering is turned on.

## Technical reference

### Response fields

A canonical recall response adds:

| Field | Meaning |
|---|---|
| `abstained` | `true` when the top results are judged not to answer the question, or when nothing confident enough was found at all |
| `abstention_reason` | `judged_insufficient` — memories were found, but none of them answer it. `evidence_floor` or `no_candidates` — nothing confident was found in the first place (unrelated to this feature; see [Retrieval Score Contract](retrieval-score-contract.md)) |
| `answer_confidence` | How confident the check is that the top results answer the question, `0.0`–`1.0` |
| `calibration_status` | Whether that confidence number has been measured against known-answerable and known-unanswerable questions, or is unmeasured |
| `calibration_id` | Which backend and setup produced the verdict — useful for support, not meant to be parsed |
| `answer_check_status` | What happened to the check on this one recall — see below |
| `reranker_status` | Which step produced the final order. `jev_listwise` means the online check's optional reorder replaced the local reranker's order |
| `local_reranker_status` | What the local reranker reported before that reorder, if any. Empty when nothing replaced it |
| `score_contract_version` | Unchanged at `"2"` — see [Retrieval Score Contract](retrieval-score-contract.md) |

Results are **never removed** because of an abstention. `abstained` and
`abstention_reason` are a signal for the caller to act on, not a filter.

Every recall response carries `answer_check_status`, whether or not the
check is configured:

- `judged` — a verdict was produced and applied.
- `off` — no check runs for this engine; nothing is asked.
- `skipped` — the check is on, but wasn't asked for this one recall: it was
  loading context (session start, auto-loaded context, the per-prompt
  hook, or the daemon's own start-up recalls), there were no results to
  judge, too little of the recall's time budget was left, or the online
  option would have had to read a memory owned by another profile.
- `busy` — the on-device check was already answering another recall.
- `warming` — it's still loading its model.
- `unavailable` — it timed out, errored, has no key, or can't run here.

Skipping costs the verdict only; results are identical whether the check
ran or not. Recalls that only load context — session start, auto-loaded
context, the per-prompt hook — and the daemon's own start-up recalls are
never judged, never sent anywhere, and never billed.

Only the exact on-device weights this release measured can abstain: point
**Advanced** at a different snapshot or revision and the check still
returns an `answer_confidence`, but that recall can never report
`abstained: true` because of it.

### Where the settings live

The answer check's settings are kept in one file, `answer_check.json` in
your data folder (`~/.superlocalmemory/` unless you moved it), owner-only.
It is the same file whichever operating mode (A, B or C) you are in, so
switching modes can neither bring back a consent you withdrew nor reset a
choice you made. Installs that never touched the answer check have no such
file until the first change; earlier copies of these keys under `retrieval`
in `config.json` and `mode_a/b/c.json` are removed when it is created. A
damaged file is read as "no consent": it can only ever turn the online
option off.

The dashboard is the supported way to change them. Changes go through
`/api/v3/answer-check`, which accepts them only with one of SLM's own
credentials — the dashboard sends its install token automatically; a script
on the same machine must send `X-Install-Token` (or the daemon capability,
or an API key). Being on the same machine is not enough, because these
settings decide where memory text is sent and what is billed.

### Config keys

| Key | Values | Meaning |
|---|---|---|
| `sufficiency_judge` | `auto` (default) \| `laya` \| `jev` \| `off` | Which check runs. `auto` is the starting state before anyone chooses: the on-device check when an install exists that passed its check, otherwise exactly as `off`. It never resolves to the online option by itself, and the dashboard never offers it — choosing writes an explicit `laya`, `jev`, or `off` |
| `sufficiency_python`, `sufficiency_model`, `sufficiency_hf_home` | paths | Set automatically by the dashboard's "Set up" and "Advanced → Use an existing install" flows |
| `sufficiency_timeout_s` | seconds | How long the on-device check waits before giving up on a single recall |
| `sufficiency_jev_provider` | `typesafe` \| `openrouter` | Which hosted provider the online option uses |
| `sufficiency_jev_consent` | `true` \| `false` | Must be explicitly `true`, set only by ticking the consent box |
| `sufficiency_jev_timeout_s` | seconds | How long the online option waits for a provider reply |
| `sufficiency_jev_rerank` | `true` \| `false` (default) | Whether the online option also reorders the top results. Set only by the dashboard switch |
| `sufficiency_jev_rerank_consent` | `true` \| `false` (default) | Must be explicitly `true`, set only by ticking the switch next to its notice. Cleared whenever reordering is turned off |
| `sufficiency_jev_rerank_k` | `5`–`30`, default `20` | How many top results reordering sends and may reorder. A value that isn't a whole number of at least 1 means reordering is off. Not on the dashboard |

The provider key itself is never written to any of these files; it lives in
its own owner-only store, separate from the rest of your configuration.

The on-device model loads when the daemon starts, or the moment you switch
the check on from the dashboard — never inside a one-shot CLI command, so
running `slm recall` on its own never pays for loading it.

### CLI and MCP output

`slm recall` and `slm trace` print the same one-line summary shown above,
as plain text, whenever the check has run. The MCP `recall` tool instead
returns the fields themselves (`abstained`, `abstention_reason`,
`answer_confidence`, ...) for the caller to act on; MCP `session_init`
additionally prefixes its narrative `context` string with the same one-line
summary, since a host may consume `context` without reading the sibling
fields. When the check hasn't run (off, or still warming up), none of this
appears and output is byte-for-byte what it was before this feature
existed.

## Troubleshooting

### My Mac doesn't have Apple Silicon

"On this Mac" needs Apple Silicon. Use **Online with Jev** instead, or leave
the check off.

### My company network blocks the download

If "Set up" fails because the download couldn't reach the server, the
dashboard tells you: *"Couldn't reach the download server. If your network
blocks downloads, use 'Use an existing install'."* Open **Advanced — use an
existing install** under the On this Mac panel and point it at a Python
interpreter and model you already have (for example, one downloaded on
another machine on the same network and copied over). For the cache folder
you can give either your `HF_HOME` or the `hub` folder inside it.

SLM runs that interpreter, so it only accepts one it can trust: the `python`
inside a Python environment (`<environment>/bin/python`, next to a
`pyvenv.cfg` file) or the Python SLM itself runs on; owned by you (or the
system) and changeable by no other account; and not in a temporary folder or
under `/Users/Shared`. It is checked again every time SLM is about to run
it. If it is refused, the dashboard says why.

### "The saved key file was changed outside SLM"

The key file in SLM's private folder no longer looks like the one SLM wrote
(for example, it was replaced by a link), so SLM will not read it and the
online option stays off. Choose **Remove key**, then paste your key again.

### Removing the on-device model, uninstalling, or downgrading

The on-device model and its private Python environment (about 1.1 GB plus
the environment) live in `runtimes/laya` inside your data folder. **Remove**
in the dashboard stops the model and deletes that folder. Nothing else
deletes it: switching to **Online with Jev** or **Off** stops the model (it
no longer uses memory) but keeps the files so you can switch back without a
download; `pip uninstall`, `npm uninstall` and installing an older SLM leave
them too. To reclaim the space after uninstalling SLM, delete
`~/.superlocalmemory/runtimes/laya` (or `runtimes/laya` in the data folder
you chose) — that folder only. An install you pointed SLM at with **Use an
existing install** is never deleted by SLM.

### My key isn't working

Test results are plain-language:

- *"The key was not accepted."* — double-check the key, or generate a new one.
- *"The account has no credit left."* — top up with your provider.
- *"Too many requests — try again in a minute."* — you're being rate-limited; wait and retry.
- *"The service did not answer in time."* / *"The service could not be reached."* — a network or provider-side issue; try again shortly.

### "Answer check runs in another SLM process"

If more than one SLM process uses the same data folder (for example, a
terminal command running alongside the daemon), only one of them runs the
on-device model — the rest report their recalls as if the check were off,
rather than each loading a separate copy of the model. This is expected and
not an error. Recalls from the CLI, the MCP tools and the dashboard normally
go through the daemon, which is usually the one holding the model. A normal
install has one data folder; a second install with its own folder is a
separate SLM and runs its own copy.

---

### Third-party attribution

The on-device option downloads a small language model, separately from
SLM itself, the first time you set it up — it is not bundled with SLM. That
model, Laya, is built by Convai Innovations and distributed under the
Apache License 2.0.

---

*SuperLocalMemory V4 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*

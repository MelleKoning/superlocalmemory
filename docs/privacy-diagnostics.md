# Local privacy diagnostics

SuperLocalMemory keeps a bounded 31-day operational summary in the active SLM
data directory. It stores only daily counters, fixed client-family buckets,
latency buckets, coarse error classes, and aggregate cross-client transitions.
Memory content, queries, fact identifiers, filesystem locations, exception
messages, and identity values are not stored in this diagnostics database.

No network reporting endpoint exists. The aggregate file is never exported automatically.
To create a local support artifact, the operator must
explicitly run:

```bash
slm diagnostics export ./slm-diagnostics.json
```

The output is deterministic for unchanged counters and is written with owner
read/write permissions. Inspect it before choosing whether to share it. Running
the command does not enable recurring reporting or contact a remote service.

## Answer check (4.1.18)

[Answer check](answer-check.md) is a separate, opt-in feature with its own
network behavior, off by default:

- **On this Mac**: nothing about your questions or memories leaves the
  machine for this feature, at any time.
- **Online with Jev**: only while this option is turned on, each recall
  that runs the check sends your question and the top 3 memories for it to
  the provider you chose, using your own key. Text that looks like a
  credential is stripped out first; ordinary personal or confidential
  details are not, so treat this the same as any other data you choose to
  send to a third party. Turning the option off, or withdrawing consent,
  stops this immediately — nothing further is sent.
- **Off**: no change from how recall has always worked.

This is separate from the local diagnostics summary described above, which
does not record memory content, queries, or provider choices either way.

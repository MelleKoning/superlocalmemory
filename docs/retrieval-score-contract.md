# Retrieval Score Contract v2

This contract describes runtime result fields, not benchmark accuracy. For the
published LoCoMo figures and their protocol disclosures,
see [Benchmark Evidence](benchmarks.md).

SuperLocalMemory keeps retrieval ordering separate from confidence. A
retrieval score answers “how relevant is this stored fact to this query?” It
does not answer “how likely is a generated answer to be correct?”

## Result fields

| Field | Meaning | Public interpretation |
|---|---|---|
| `relevance_score` | Query-relative relevance after the configured retrieval and scoring pipeline | Bounded to `0.0..1.0`; compare only within a compatible SLM release and retrieval configuration |
| `ranking_score` | Internal ranking utility after optional adaptive or reranking adjustments | Diagnostic value; it is not a probability and is not guaranteed to be bounded or comparable across configurations |
| `memory_confidence` | Confidence stored with the underlying assertion | Memory metadata; it is not query relevance or answer correctness |
| `trust_score` | Trust signal attached to the stored evidence | An evidence-policy signal, not answer confidence |
| `rank_position` | One-based position in the returned result list | The observable order after ranking |

For one compatibility release, legacy `score` aliases `relevance_score` and
legacy `confidence` aliases `memory_confidence`. New integrations should use
the explicit field names.

## Response fields

Every canonical recall response declares:

```json
{
  "score_contract_version": "2",
  "calibration_status": "uncalibrated",
  "calibration_id": null,
  "answer_confidence": null,
  "abstained": false,
  "abstention_reason": null,
  "answer_check_status": "off",
  "reranker_status": "not_configured",
  "local_reranker_status": ""
}
```

By default, no backend judges whether the results answer the question, so
`calibration_status` is `uncalibrated`, `calibration_id` is `null`, and
`answer_confidence` is `null`. Consumers must not derive answer probability by
doubling, thresholding, averaging, or otherwise transforming retrieval scores.

`answer_check_status` reports what became of the answer check on this one
recall: `judged` (a verdict was applied), `off` (no check runs), `skipped`
(the check is on but wasn't asked — context loading, a system recall, no
results, too little of the recall's time budget left, or the online option
would have had to read another profile's memory), `busy` (the on-device
check was already answering another recall), `warming` (still loading its
model), or `unavailable` (timed out, errored, no key, or can't run here).
Skipping costs the verdict only — the memories, their order, and their
scores are unaffected either way.

`reranker_status` names whichever step produced the final order;
`jev_listwise` means the online answer check's optional reorder replaced
whatever the local reranker had produced. `local_reranker_status` keeps that
local status for the recall, empty when nothing replaced it.

When no result survives the evidence floor, `abstained` is `true` and
`abstention_reason` is `evidence_floor`. When candidate generation returns
nothing, the reason is `no_candidates`. Neither case involves a judgment about
the results — there simply were none to judge.

### Answer check

When [Answer check](answer-check.md) is turned on (Settings → Answer check —
off until an on-device install has passed its check), a separate decision
runs on the top results: not "is this
relevant" but "does this answer the question." That decision populates
`calibration_status`, `calibration_id`, and `answer_confidence` with real
values instead of the uncalibrated defaults above, and can set
`abstention_reason` to a third value, `judged_insufficient`: results were
found and are returned as before, but none of them were judged to answer the
question. `calibration_status` additionally distinguishes a measured
calibration from one that is not yet calibrated for its current backend and
settings — treat an unmeasured verdict as a confidence signal, not a
guarantee. Results are never removed because of this verdict; it is reported
alongside them, not used to filter them. See [Answer Check](answer-check.md)
for the full contract and setup.

### The `answer_check` request parameter

The daemon's `GET /recall` accepts an `answer_check` query parameter:
`full` (the default — the complete check, with the optional reorder when
that's on), `no_reorder` (the check's verdict only, never the reorder —
what the bounded-loop gate asks for), or `skip` (no check at all, for a
recall that isn't a question — context loading, health probes). Any other
value is refused with HTTP 400.

## Retrieval composition

The current engine can run five candidate producers when their dependencies
are healthy: dense semantic, BM25 lexical, temporal, Hopfield associative, and
spreading activation. Weighted reciprocal-rank fusion combines their output.
Entity-graph information can enhance a post-fusion score but does not create an
independent candidate in the current implementation. Optional reranking and
adaptive learning can alter `ranking_score`; they do not turn it into a
probability.

The exact channels that contributed to a result are available through trace
output. A missing optional dependency can change the active channel set, so
applications that need a locked retrieval topology should verify health and
trace metadata at startup.

## Calibration release gate

Calibrated confidence requires a frozen release candidate, held-out relevance
labels, declared corpus and query distribution, calibration identity, and
reported calibration and selective-risk metrics. Until that evidence exists,
the truthful contract remains uncalibrated with `answer_confidence: null`.

# SLM 4.1 — Reliability Evaluation Results

- Package version: **4.1.18** release candidate (commit 9f44b140, run from source; the installed package metadata still read 4.1.17)
- Python: 3.14.5
- Platform: macOS-26.4.1-arm64-arm-64bit-Mach-O
- Generated: 2026-10-02T20:19:39.923505+00:00

| Experiment | Guarantee | Metric | Trials | Held | Rate | Verdict |
|---|---|---|---:|---:|---:|:--:|
| exp1_erasure_completeness | Real Bm25Owner + TemporalOwner + VectorOwner (no sqlite-vec: embedding_metadata SQL table only, no ANN index) erase all wipe-tenant projection rows from bm25_tokens, fact_temporal_validity, and embedding_metadata; tombstones and a verifiable receipt are persisted; keep-tenant content-hash is identical before and after wipe (full-row tamper detection) | complete-erasure rate | 200 | 200 | 1.0000 | PASS |
| exp2_transaction_atomicity | MemoryTransactionService: committed op → manifest COMPLETE with both owners applied; faulted op → manifest DEGRADED, the successful owner's projection removed by real service.compensate() (tracking entry absent post-compensate, ledger=COMPENSATED), failed owner had no projection residue (ledger=FAILED) | manifest-correct rate | 200 | 200 | 1.0000 | PASS |
| exp2b_real_owner_manifest | Real Bm25Owner + TemporalOwner + VectorOwner (vector_store=None, ann_index=None) via MemoryTransactionService: (A) happy-path run() with all projections seeded → manifest COMPLETE, all three tables physically present; (B) Bm25 failure (no retrieval, no bm25_tokens) → manifest DEGRADED, Bm25 has zero partial residue, real svc.compensate('temporal') removes fact_temporal_validity rows to zero, obligation state COMPENSATED. | manifest-correct rate | 200 | 200 | 1.0000 | PASS |
| exp3_migration_downgrade | newer-stamped DB refused on deferred pass with zero mutation | refuse-and-preserve rate | 200 | 200 | 1.0000 | PASS |
| exp4_backup_restore_atomicity | partial-restore failure rolls live data back to pre-restore content | rollback+clean rate | 200 | 200 | 1.0000 | PASS |
| exp5_multitenant_isolation | personal rows of one tenant never leak to another on any read path | zero-leak rate | 200 | 200 | 1.0000 | PASS |
| exp6a_superseded_demotion | superseded facts kept but demoted 0.25x and re-ranked below valid | correct-demotion rate | 200 | 200 | 1.0000 | PASS |
| exp6b_decay_monotonic | Ebbinghaus retention non-increasing in age, bounded [0,1] | monotonic-decay rate | 200 | 200 | 1.0000 | PASS |
| exp6c_time_window | date-proximate events outrank distant ones; horizon excludes far past | correct-window rate | 200 | 200 | 1.0000 | PASS |
| exp7_generation_fence | Generation fence rejects a stale-epoch ADMISSION (WriteCoordinatorError with ValueError __cause__ 'epoch is stale', writer never called) and admits a fresh-epoch ADMISSION from runtime.remember() in the same runtime | fence-correct rate | 200 | 200 | 1.0000 | PASS |
| exp8_policy_registry | _DEFAULT_REGISTRY.evaluate(): REMEMBER allowed with reason=='allow' for OWNER/ADMIN/MEMBER; VIEWER→REMEMBER denied with insufficient_roles; unknown kind fail-open in local mode (unknown_kind_allow_local), fail-closed in company mode (unknown_kind_deny_company); unauthenticated actor denied with authentication_required (auth check before role check) | policy-correct rate | 200 | 200 | 1.0000 | PASS |
| exp12_learning_loop_ablation | only a real conversation plus observed engagement moves the posterior | max_abs_posterior_mean_shift | 600 | 600 | 0.3480 | PASS |
| exp13_governance_envelope_cost | every governed write records every declared envelope component | envelope_p50_ms | 200 | 200 | 7.1144 | PASS |

| exp_governed_latency | governed write envelope p50 | latency_ms | 200 | 200 | — | PASS |
**Aggregate: 3000/3000 trials upheld their guarantee (100.0000%).**

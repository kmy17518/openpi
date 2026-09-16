# Openpi post-baseline audit fixes — 2026-09-16

Scope: audit of `0cc8e355f7bac0976db1cc3139b1ff0379feea60..f588a73cd80a8be6b1d4edefc7b40652b095cbc3`, applied after history reorganization on `kmy17518/reorganized-openpi` (reorganized base `5851ae6`). The original checkout's uncommitted source/dependency work is not rewritten or staged here.

## Finding status

| Audit issue | Resolution | Regression coverage |
|---|---|---|
| 1. Repository-wide permanent LFS deletion | Removed permanent deletion and history rewriting. Only lower-numbered superseded resume paths are removed by normal commits; higher-step/other-ref data is retained. | `scripts/b1k/uploader_lifecycle_test.py` |
| 2. Stale-observation action chunk replanning | One prediction per request, execute only n of m; required requested size equals n. Andi-compatible `action` + `action_chunk` and no-reset-reply preserved. | Server tests and `examples/b1k/test_action_chunk_servers.py` using unmodified BEHAVIOR client |
| 3. Missing tasks silently omitted | Every requested task needs available episodes, including after explicit episode filtering, before data/stats construction. | `b1k_dataset_test.py` |
| 4. Dataset-specific prompts lost | Checkpoint metadata records exact resolved task prompts. Serving uses those by default, including custom tasks; explicit overrides remain available. | `b1k_artifacts_test.py`, including dataset-to-checkpoint assets and serving main |
| 5. Token budget lost at serving | Save/restore inference model settings, expose `--max-token-len` override, validate prompt against extracted state dimension before loading the policy. | `b1k_artifacts_test.py` |
| 6. Explicit delta mapping overlaps inferred mapping | Reserve explicit dimensions before inferring mappings; reject invalid indices and conflicting action assignments. | `b1k_delta_mapping_test.py` |
| 7. Metadata publication not retried | Reconcile LATEST/W&B/README even when checkpoint content exists; mark success and remove predecessors only afterward. | Failure-before/after-commit and restart injection tests |
| 8. Fresh run skips old step numbers | Generation UUID, generation-specific state/staging, Orbax checkpoint identities and remote provenance. Downloaded resume preserves generation ordering. | Stale state/staging, same-size remote replacement and download/resume tests |
| 9. Competing launchers overwrite same experiment | Nonblocking local run/GPU UUID leases span idle checks and training; publication locks exclude fresh-generation changes. | Concurrent launch/subprocess, overlapping GPU and parent-exit tests |
| 10. Filtered camera streams break image transforms | Reader-local camera metadata matches decoded streams and keeps supported image fields. | Full item-loading tests with image transform and RGB/depth selection |
| 11. Prefetch exhaustion and cleanup | Terminal state remembered; close/context cancellation, producer-owned upstream close, persistent-worker cleanup and trainer finally paths. | `data_loader_prefetch_test.py`, `train_b1k_lifecycle_test.py` |
| 12. Configured noise ignored in batched inference | Per-call noise overrides configured noise, otherwise generated defaults; shape checks and shared-sample broadcasting. | `policy_batch_test.py` |
| 13. Torso migration guard | Versioned action-representation metadata for stats/checkpoints; fail on known mismatch, explicit missing-metadata opt-in, legacy four-joint torso config. Resume also compares actual normalization arrays. | `b1k_artifacts_test.py`, including real Orbax save/restore |
| 14. Overstrict RNG test | Exact key/noise checks separated from output comparisons with floating-point tolerance. | CPU and GPU policy tests |

Independent review also identified and prompted fixes for delayed full-upload rollback, malformed completion timestamps, changed normalization values at resume, and lost generation timestamps when adopting downloaded checkpoints. They have dedicated regression coverage.

## Validation

- Broad available non-manual CPU suite: **297 passed**, with the two previously reproduced baseline failures explicitly deselected. Full-model CPU/training-integration tests were excluded as in the audit; GPU checks cover relevant numerical paths separately.
- Audit-specific regression panel: **222 passed**. The final generation-adoption change was then retested with all **77 uploader/lifecycle tests**.
- Four-GPU attention forward/gradient/cache, accumulation, optimizer/EMA, checkpoint restoration and prefetch checks: **7 passed**.
- Real-data long-video/low-dimensional action equivalence plus two-worker microbatch/prefetch checks: **3 passed**.
- Independent reviewer: focused CPU tests, small GPU tests, and offline race/failure reproductions; report stored with the workspace audit artifacts.
- CLI help construction for training, statistics and serving passed. Scoped lint, shell syntax and whitespace checks passed. Existing unrelated style debt was not rewritten.

The two baseline failures are FAST tokenizer initialization with the installed tokenizer stack and generic dataset access to missing `DataConfig.episodes_index`. They are outside the post-baseline introduced findings and remain unchanged.

## Operational changes and limits

- Recompute normalization statistics to obtain `b1k_metadata.json`. Missing metadata fails by default; after checking the convention, explicitly opt in via `--data.allow-legacy-assets` (training/resume) or `--allow-legacy-assets` (serving). Known mismatches are never bypassed. Use `--robot b1k/R1Pro-legacy-torso-delta` for matching old four-joint-delta checkpoints.
- New checkpoint prompts/token budgets restore automatically. Resuming with different saved/current normalization values fails before state restoration; use the checkpoint's assets.
- LFS storage is no longer reclaimed by the mirror. Remote history cleanup must be an explicit separate administrative action.
- Run/upload leases are local advisory locks. Cooperating processes must share the lock directory; arbitrary remote writers or other hosts are not coordinated.
- Older-generation higher-numbered folders may remain. Use `resume/LATEST.json` to choose the active generation/checkpoint.
- Prefetch `close(timeout)` returns false if an arbitrary upstream call remains blocked; cleanup finishes when that call returns. Python threads are not forcibly killed.
- Live Hub writes/deletion, full robot simulator rollouts, long-run convergence and the separate JAX/CUDA13 environment were not tested. Hub interactions in regressions are mocked; GPU/real-data checks use the original installed environment.

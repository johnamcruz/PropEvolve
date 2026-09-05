# V40 memory-pressure investigation

Date: 2026-09-04. Initial verdict: STOP pending resource correction, not a failed
economic recipe. Final gate: correction verified on the bounded three-worker
test and all 991 regression tests; R2 launch authorized after commit/push.
Production commit: `1a4728a4358a4d8f4edcd29e36e504a4d31a29e8`.
Read-only prior-learner comparison: `9f881af`.
Installed libraries: MLX 0.32.2, PyTorch 2.13.0, NumPy 2.4.6.

## Confirmed evidence

- Before reboot, OS memory-pressure reports listed V40 workers and the three
  suspended V39 workers among the largest Python consumers. Suspension did not
  release the old workers' memory. This is evidence of overlapping memory demand,
  not proof that one isolated component caused the entire freeze.
- After reboot, 46 obsolete campaign jobs restarted. They were unloaded; all 92
  obsolete PropEvolve campaign/sweep services were disabled. V40 was subsequently
  stopped and disabled at the user's request before correction and verification.
- V40 trial 1's episode-30 checkpoint references 30 replay shards totaling
  3,546,192,488 serialized bytes. A representative 13,261-transition episode
  contains 136,757,744 bytes of float32 observations: shape `(13262, 2578)`.
- The existing 500,000-transition limit allows about 5.16 GB of observations per
  worker, excluding metadata and episode-end rows. Three workers can therefore
  demand about 15.47 GB for observations alone. Other allocations require RAM too.
- The replay observation stacking and eager pickle restore predate V40.
- V40's mastery dictionary contains 16 reference tuples, about 2,045 JSON bytes.
  It does not duplicate whole episodes. However, actual episode-30 diagnostics
  report approximately 31.75-31.81 additional sequences per update. This expands
  the ordinary 16-sequence batch to approximately 48 sequences.

## Bounded real-data production learner comparison

One process at a time; identical saved V40 trial-1 checkpoint and first replay
episode; actual recurrent burn-in, production train_batch, labels, losses and
optimizer. The expanded arm adds 32 authenticated candidate sequences to the
same ordinary 16-sequence batch. This isolates batch-size cost; it does not
reproduce the entire evolving mastery bank or all campaign data.

| Arm | Sequences | Metal driver allocation after warmup | MLX free-cache bytes |
|---|---:|---:|---:|
| Prior learner loaded read-only from Git | 16 | 1,465,843,712 | 283,251,524 |
| Current learner | 16 | 1,471,250,432 | 288,628,548 |
| Current learner | 48 | 1,816,887,296 | 588,176,260 |

The 48-sequence arm was repeated for 12 updates. Driver allocation plateaued at
1,816,936,448 bytes from update 3 through update 12. MLX live allocation stayed
around 3.16 MB; free cache plateaued near 588.26 MB. Maximum process RSS increased
from about 916.55 MB after the first update to 935.43 MB after update 12.
Loss remained finite and declined; no boundary backtracking occurred in this
particular probe. It does not cover all later active-constraint patterns.

Separate raw recurrent forward/backward measurements also showed retained
allocator cache after returning from batch 48 to batch 16. Those measurements
used a full 104-step differentiable trace and are NOT production burn-in results.
Do not add Torch driver and MLX allocator numbers as independent physical RAM:
this is a shared Metal allocation path. RSS alone also understates compressed,
swapped and device memory.

Conclusion: recent fixes increase memory demand via effective batch expansion.
No unbounded learner leak was established by these bounded tests. Longer varying
batch/reset/constraint tests and phase-level process-footprint measurements remain
necessary. Three independent large replay stores plus leftover workers are a
proven resource-budget problem regardless of an additional leak.

## MLX options and boundaries

- Configure MLX's reusable buffer cache and benchmark the throughput tradeoff.
  A cache limit reclaims free cached buffers on subsequent allocation; zero
  disables caching. It does not delete live replay or model data.
  [MLX cache limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
- Do not treat the MLX allocation limit as a whole-process or machine-wide hard
  cap. It is a guideline for MLX allocation; it cannot budget every worker's
  NumPy/PyTorch/OS allocation.
  [MLX memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
- Unified CPU/GPU memory removes some transfers; it does not automatically share
  independent workers' replay stores. Converting the whole replay to MLX is not
  itself a remedy for excess retained data.
  [MLX unified memory](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst)

## Recommended next bounded implementation

1. Preserve V40's learning fixes, source examples, margins, losses and sampler.
2. Bound resident replay storage using read-only disk-backed observations and a
   bounded working set. Prove identical selected rows, resets, labels, new-pass
   promotion, mastery references and resumed Q-values before using it.
3. Add configuration-driven allocator/cache budgets and memory diagnostics;
   benchmark lower cache retention with actual loss/Q/gradient parity checks.
4. Measure one worker through replay growth and then aggregate concurrent demand.
   Increase concurrency only within measured machine headroom.
5. Add a separate disk-retention/free-space budget. Current compact screening
   cleanup removes rejected trials but retains feasible trials; it is not a
   global disk cap. Preserve summaries and designated checkpoint/replay dependencies.

The original study, queued recipes, saved passes, and checkpoints remain preserved.

## Implemented resource correction and matched evidence

- `training.replay_observation_storage` selects `memory` or `mmap`; the JSON
  default now uses `mmap`. Canonical pickle shards and descriptor schemas are
  unchanged. Read-only NPY observation sidecars are derived once per store
  lifetime, reused across checkpoints, rebuilt from authenticated canonical
  shards on restore, and pruned after checkpoint commit. Sampled minibatches,
  labels, resets, pairing, promotion and losses are unchanged.
- `runtime.mlx_cache_limit_bytes` is optional; null preserves the library choice.
  R2 explicitly uses 268,435,456 bytes (256 MiB) per worker. This bounds reusable
  cache, not live allocations or the entire process; small allocator overshoot
  is observable. Agent construction no longer clears this runtime setting.
- Read-only replay action selection copies just the requested observation to
  avoid exposing an unsafe writable Torch alias of the mapped archive.
- Optional `artifacts.maximum_retained_feasible_trials` keeps the best completed
  feasible family while compacting lower-ranked completed artifacts. R2 keeps
  three. Existing rejected-trial cleanup remains; JSON/JSONL diagnostics are
  preserved. Cleanup is serialized across callbacks, not across training jobs.
  Active trials are excluded. This bounds completed raw-trial count, not all
  unrelated files on the machine.

Ten identical real checkpoint episodes restored with the old heap storage had
a 1.4 GiB physical footprint; mapped restore was about 318 MiB. RSS is not the
right comparison because it includes clean, reclaimable mapped pages.

Three actual production updates on those same sampled batches produced exactly
the same reported losses under heap storage, mapped storage, and mapped storage
with the 256 MiB MLX cache budget: `7.803676605224609`, `7.303634166717529`,
`8.170903205871582`. The cache budget reduced the final measured footprint from
about 2.4 GiB to 2.0 GiB. End-to-end probe durations including restore/measurement
were approximately 61.5 versus 64.4 seconds; do not claim a speedup from this.

### Three-worker real-artifact test

Three isolated workers simultaneously restored 30 saved episodes each from
V40 trials 0, 1 and 2, then each performed four production MLX updates with
48 sequences per update (16 ordinary plus 32 authenticated paired sequences).
This uses the actual saved models, source replay, recurrent burn-in, losses and
optimizer; it does not launch or alter the Optuna study.

- Restored physical footprints: 375.2, 349.7, 353.6 MiB.
- Update footprints: approximately 2.0-2.2 GiB per worker.
- Metal driver allocations stayed around 1.51-1.52 billion bytes per worker.
- All workers completed with finite losses and no crash.
- System swap stayed at 1,037.38 MiB across the concurrent update check.
- Temporary sidecars were deleted when each diagnostic exited; original shards
  were linked read-only and were not modified.

This verifies the identified memory correction under a bounded concurrent
production-learner workload. It is not a full 100-trial soak or a guarantee against
all possible resource failures. R2 must retain normal runtime health checks.

### Regression seams

Tested read-only replay append/restore/eviction, tampered sidecar reconstruction,
exact sampling state, canonical shard authentication, new learned-pair promotion,
rehearsal after restore, exact greedy Q parity, both PyTorch and MLX E2E paths,
complete historical train-to-teacher-free-validation flow, optional JSON budgets,
MLX cache/output parity in a real subprocess, and one/three-job Optuna cleanup.
Final full regression: **991 passed**, 71 Optuna experimental-feature warnings,
218.35 seconds. The earlier legacy-default assertion was updated for the optional
null cache field; default legacy runtime behavior remains unchanged.

R2 keeps the same 100 trials, three jobs, grouped multivariate TPE, 50/50 screening,
search dimensions, objectives, labels, teachers, parent and learning mechanics.
Its study root is new so code/config identity is not silently changed underneath
the interrupted original journal. Re-enqueue the requested V39 Trial 1 and 26
recipes with native Optuna, alongside the unchanged baseline control.

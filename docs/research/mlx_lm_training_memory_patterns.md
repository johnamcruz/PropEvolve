# MLX-LM training memory patterns for the recurrent adapter

Date: 2026-09-04. Primary-source research and matched implementation evidence.
Sources below are live `main`/official documentation, not immutable release
pins. Verify against the installed MLX version before implementation.
The inspected local package metadata reports MLX 0.32.2; the compilation guide
also reports 0.32.2. Source links to `main` can still differ from that release.

## Applicable patterns

1. **Evaluate the complete step boundary.** MLX-LM compiles a training step
   with model, optimizer, and random state captured as inputs/outputs. After
   each iteration it evaluates state, loss/token accumulators, and pending
   gradients together. After an optimizer update the pending gradient becomes
   `None`; report accumulators are periodically reset. Its validation loop also
   evaluates its accumulated loss/tokens each batch. These are useful lifetime
   boundaries, not instructions to port the Torch-owned optimizer to MLX.
   [MLX-LM trainer, lines 166-204 and 224-342](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py#L166-L342)

2. **Clear free cache only after useful work completes.** The trainer calls
   `_clear_cache(threshold)` after evaluation; that helper clears only when
   cached bytes exceed the configured threshold (default zero). This offers a
   bounded cache-policy comparison, not a cure for live tensors. It also sets a
   wired limit from the device recommendation; do not copy that single-training
   process choice into three competing workers without a machine budget.
   [Trainer helper and configuration](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py#L19-L80),
   [training entry and iteration](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py#L216-L312)

3. **Separate live storage, reusable cache, and whole-process footprint.** MLX
   active bytes exclude free cached buffers. `set_cache_limit` reclaims cache
   on subsequent allocations; zero disables it. `clear_cache` empties that
   free cache, not referenced arrays. `set_memory_limit` is an evaluation
   guideline, not a whole-process hard cap; wired memory remains resident.
   Consequently, process footprint can grow while MLX active/cache metrics
   plateau, and MLX limits do not bound host replay, Torch, or every worker.
   [Official allocator API source](https://github.com/ml-explore/mlx/blob/main/python/src/memory.cpp)

4. **Evaluation is not Python ownership cleanup.** MLX normally detaches
   evaluated non-tracer nodes from their producer graphs; live arrays still
   own storage. The C++ array implementation uses shared ownership for data
   and graph inputs. An unused lazy output can retain a graph even if another
   output was evaluated. Most importantly, `mx.eval()` with no arrays returns
   immediately; it is not a universal flush of every outstanding graph.
   [Evaluation implementation](https://github.com/ml-explore/mlx/blob/main/mlx/transforms.cpp#L279-L346),
   [array ownership and detach](https://github.com/ml-explore/mlx/blob/main/mlx/array.cpp#L108-L190)

5. **Compile once, then audit actual signatures.** Shape, dtype, and argument
   count changes can recompile. Compile the outer transformed operation, not
   just its forward function; the existing compiled recurrent VJP already
   follows this pattern. MLX-LM sorts sequences by length and pads in buckets
   of 32. The transferable idea is to count and bound shape variants, not to
   change causal recurrent lengths or replay sampling. `shapeless=True` is
   unsafe when Python recurrence loops or reshapes depend on an input shape;
   it is not a drop-in switch. Allocator cache clearing does not establish that
   compilation variants were evicted.
   [Compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html),
   [trainer batching](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py#L97-L161)

6. **Keep the existing cross-framework synchronization.** MLX recommends an
   outer-iteration evaluation; adding one per recurrent timestep adds overhead.
   NumPy conversion and scalar reads can force partial evaluation. DLPack may
   share Metal storage but does not synchronize producer work. MLX functional
   autodiff does not remove the Torch bridge's saved-tensor lifetime needs;
   do not discard tensors required by later PCGrad backward calls.
   [Lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
   [framework conversion](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/numpy.rst#L69-L123),
   [functional autodiff](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)

## Concrete local hypothesis, not a leak diagnosis

In [`_MlxExecutionWorker._run`](../../src/propevolve/mlx_backend.py), successful
requests leave local references alive across loop branches and while waiting
for another request: request values, MLX inputs/outputs, cotangents, gradients,
and converted results. A later `memory` request does not overwrite all these.
Backward additionally builds forward outputs to size absent cotangents before
calling the separate compiled VJP; that forward graph can remain unevaluated.
This follows from the inspected control flow and function-local binding scope
([Python execution model](https://docs.python.org/3/reference/executionmodel.html)).
It establishes potentially avoidable last-request retention, not accumulating
per-episode leakage. Reference cleanup must preserve the returned DLPack tensor's
ownership and any live Torch backward context.

The supplied episode-8/9 slowdown, approximately 4.3 GB footprint per worker,
5.6 GB peaks, and 5.8 GB swap cannot be attributed to MLX cache from these
sources. An earlier fixed-workload plateau is compatible with later host/replay
growth, changing graph signatures, or overlapping worker peaks.

The smallest next diagnostic is a matched worker-lifetime comparison: identical
request/shape sequence and learning outputs; measure before/after request-local
release separately from free-cache clearing. Record actual shape signatures,
MLX active/cache/peak, Torch allocator/driver values, process footprint, swap,
and wall time at the same phase boundaries. Do not add overlapping allocator
counters as independent RAM. Only then consider cache-budget tuning or a
fixed-shape recurrence. Gradient checkpointing trades recomputation for saved
intermediates and should be deferred unless live activations are the measured
bottleneck ([checkpoint API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.checkpoint.html)).

## Correction and proof required before restarting

`RecurrentC51Agent.greedy_sequence_action_values` already has `@torch.no_grad()`.
Missing no-grad on that scoring function is therefore ruled out; do not add a
duplicate fix or attribute the slowdown to it.

The prior real-artifact storage benchmark called `train_batch` directly and
cleared the restored mastered-pair library. It did not exercise the complete
collection, violation scoring, mastery promotion/rehearsal, sparse replay, and
optimizer cycle. Its stable allocator measurements do not establish stability
of the sweep. Keep its results as a component baseline only.

Implementation order, conditional on matched measurements:

1. Test whether completed worker requests unnecessarily retain input/output
   storage. Compare request-local cleanup with unchanged numerical outputs,
   gradients, repeated backward, and returned tensor ownership.
2. Separately test unused-cache reclamation at completed work boundaries. Do
   not insert synchronization or cache clearing at every recurrent timestep.
3. Record batch/reset shape variants and whole-process footprint through the
   actual post-warmup production loop, not only fixed-shape optimizer calls.
4. Verify the complete loop with three concurrent workers beyond the previous
   episode-8/9 boundary, measuring swap growth and episode/update latency as
   well as active/cache bytes. A finite loss or surviving four updates alone
   is not the acceptance test.

No learning losses, replay membership/cadence, recurrent history, or search
parameters should change as a side effect of this memory correction. Do not
copy LLM-specific KV caches, LoRA, quantization, or a full optimizer rewrite;
none has been established as a remedy for this observed failure.

## Measured correction (three-worker acceptance still pending)

A real backward ownership test reproduced the worker retaining its completed
input batch. Clearing completed-request locals fixed that test while preserving
returned outputs. This was not sufficient: the full production prefix still
reached 3.6 GiB physical footprint at episode 8, whose 32 updates took part in
a 277.7-second episode. MLX live memory was only about 11 KB at that boundary;
the MLX cache alone was not the explanation.

The production reset helper split sequences into different reset-pattern groups
and temporal fragments before dispatching MLX. On a saved production replay,
two complete mastery-aware updates generated 5,073 worker calls and 214 distinct
operation signatures. Temporal fragment lengths varied from 1 to 65.

Moving the identical per-row reset mask into the batched recurrent primitive
reduced the same two updates to 76 calls and seven signatures. Total measured
time, including checkpoint/replay restoration, fell from 43.24 to 14.26 seconds.
The two loss values agreed within approximately 1e-6. This is a component
measurement, not a claimed threefold improvement in complete trial duration.
The Torch path, reset times, burn-in, losses, optimizer, replay membership, and
checkpoint parameters remain unchanged.

The optional JSON runtime setting `mps_cache_clear_threshold_bytes` defaults to
`null` (unchanged behavior). A nonnegative byte threshold permits unused Torch
and MLX cache reclamation after a complete mastery-aware update. It is not a
hard whole-process memory limit and does not free live tensors.

The reusable `scripts/audit_sweep_runtime.py` runs the real Optuna trial training
path with the original full curriculum budget, recording update work, operation
signatures, latency, physical footprint, and accelerator memory. Its bounded
stop occurs only after a completed episode diagnostic; it does not shorten
episodes or alter warmup, replay, losses, or schedules. Three simultaneous
12-episode prefixes are the required next acceptance test. Passing them will
not guarantee stability over every later trial or establish economic lift.

### Full-workload falsifier and consumed-gradient correction

The first three-worker check did **not** pass. All workers completed episode 8
with 32 updates and 1.9–2.1 GiB boundary footprints, but workers 2 and 1 exceeded
the 3 GiB diagnostic bound at episodes 9 and 10 (3.3 and 3.4 GiB). Worker 0
completed episode 12. These results are preserved as failure evidence rather
than weakening the bound.

The remaining reservation reproduced on saved production replay. An input-layer
gradient with shape `(128, 2578)` occupies 1,319,936 bytes. After a complete
update, retaining this already-consumed gradient could keep a 1 GiB Torch Metal
heap resident. Garbage collection and a second cache clear did not release it.
Explicitly discarding consumed gradients reduced driver allocation from
1,147,797,504 to 48,889,856 bytes and footprint from 1.4 GiB to 402.3 MiB.
The next update's loss agreed within approximately 5e-7 with the control.

This matches Torch's separate large-buffer pool and 1 GiB heap policy; a small
live allocation can keep a much larger heap reserved.
[Versioned allocator thresholds](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/mps/MPSAllocator.h#L19-L24),
[heap selection](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/mps/MPSAllocator.h#L125-L160).

The correction marks the end of the complete production replay update explicitly:
only then may configured cleanup zero consumed parameter gradients and reclaim
free cache. AdamW, PCGrad, post-step boundary repair, metric extraction, and
mastery promotion run first. Ordinary cache cleanup preserves live gradients;
the default `null` setting preserves prior behavior. Both mastery-enabled and
ordinary-replay E2E tests failed on retained gradients before the correction,
then passed while comparing four optimizer updates and teacher-free action/Q
outputs against the unchanged control. The complete regression suite passed:
1,007 tests, including both learning backends, replay, teacher-free inference,
campaign, and Optuna flow. The real V40 R3 sweep, with three workers beyond
episodes 8–10, remains the final acceptance test; this note does not claim it
has already passed. Screening stays 50 training / 50 validation, with the
same search and learning settings.

The full regression also exposed an independent concurrent artifact-cleanup
race: a winning trial could complete between the ranking snapshot and the
deletion snapshot. A deterministic two-thread Optuna E2E test reproduced the
winner's checkpoint being removed. Ranking and cleanup now use one immutable
trial snapshot. This changes no sampling, scoring, training, or promotion rule.

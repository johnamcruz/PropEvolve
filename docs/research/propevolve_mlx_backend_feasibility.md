# PropEvolve MLX backend feasibility

**Status:** bounded optional-backend implementation and evidence, 2026-09-02.

## Decision

**MLX is now available as an optional recurrent tensor backend inside the one
existing PropEvolve learner.** It does not implement a second learner. The
PyTorch `RecurrentC51Agent` still owns replay, C51, every auxiliary loss,
PCGrad, post-AdamW boundary repair, checkpoints, and teacher-free validation;
MLX executes only the shared LayerNorm/Linear/SiLU/GRU forward and reverse math
over shared Metal storage.

Therefore:

- V21/V36 configs and checkpoints remain unchanged; PyTorch remains the default.
- Selection is configuration-driven through `runtime.learner_backend` and fails
  closed unless MLX runs on MPS in eager FP32 mode.
- Existing PyTorch checkpoints warm-start directly because parameter ownership
  and the bundle schema remain in PyTorch.
- The stopped V36 sweep remains stopped until the separate replay-staging work
  is complete.

This conclusion is stronger than “MLX has a GRU.” The present learner is a [LayerNorm → Linear → SiLU → GRU → C51 head](../../src/propevolve/agent.py#L1044-L1085), but its training contract also includes exact reset segmentation, a detached 64-step burn-in, Double-Q categorical targets, action masks, teacher and economic cohorts, gradient clipping, target updates, teacher stripping, and checkpointed optimizer/RNG state. V36 additionally uses `pcgrad_preserve_economic_boundaries_v3`, which takes separate gradients and then projects the realized optimizer descent before a constrained repair loop ([agent.py](../../src/propevolve/agent.py#L4290-L4545)).

## What MLX can express, and where parity is risky

| Requirement | First-party evidence | Feasibility and PropEvolve risk |
|---|---|---|
| GRU and recurrent burn-in | MLX provides a batched [`mlx.nn.GRU`](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GRU.html) returning every time-step hidden state; the [official recurrent source](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/recurrent.py) exposes its gate layout. PyTorch documents its [`nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) equations and parameter layout. | Feasible. MLX has no `batch_first` option because its documented layout already is `NLD`. PropEvolve must explicitly reproduce grouping by reset pattern, segment boundaries, hidden resets, no-gradient burn-in, and hidden carry. GRU biases are not named/layout-identical, so a tested weight converter is required. MLX's implementation iterates through sequence positions in its source; performance for length 96 cannot be assumed. |
| C51 distributional Q | MLX provides [`take_along_axis`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html), softmax/log-softmax, and collision-safe indexed accumulation through [`array.at[idx].add`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html). | Feasible, but custom. PropEvolve's projection uses three `scatter_add_` calls, including the `lower == upper` case ([agent.py](../../src/propevolve/agent.py#L5610-L5630)). Plain indexed assignment is wrong when atoms collide; MLX explicitly says only `.at.add` applies all repeated-index updates. Indices must be clamped because [MLX indexing does not bounds-check](https://ml-explore.github.io/mlx/build/html/usage/indexing.html). Projection mass, row normalization, action selection, and gradients need golden tests. |
| Replay tensors and masks | MLX has NumPy-like arrays and [NumPy/DLPack interoperability](https://ml-explore.github.io/mlx/build/html/usage/numpy.html). | Retain the existing NumPy replay/sampler and freeze sampled transition IDs. Convert each complete batch at the learner boundary. Do not mix PyTorch and MLX inside the differentiated graph. MLX does not support boolean-mask reads in NumPy syntax; the current masked selections must become `where`-weighted reductions or explicit integer indices, with empty-cohort behavior tested. |
| Autograd and gradient conflict handling | MLX differentiation is function-based through [`grad`/`value_and_grad`](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html), supports parameter trees, and uses `stop_gradient`; it intentionally has no PyTorch `backward`, `zero_grad`, implicit `.grad`, or `requires_grad` workflow. | Base C51 is straightforward. V36 PCGrad is the highest correctness and implementation risk: the current code calls `torch.autograd.grad` separately for primary, safety, opportunity, and active economic boundaries, materializes unused leaves, projects their vectors, writes `.grad`, applies AdamW, projects the realized descent again, and may line-search to repair hard margins. This must be rewritten as pure parameter-tree math and compared component by component. |
| AdamW and clipping | MLX provides [`AdamW`](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html) and global [`clip_grad_norm`](https://ml-explore.github.io/mlx/build/html/python/optimizers.html). | Feasible with a critical default mismatch: MLX AdamW defaults `bias_correction=False`; PyTorch AdamW applies bias correction. MLX must set `bias_correction=True` and explicitly freeze the same betas, epsilon, learning rate, weight decay, clipping order, and parameter ordering. MLX also warns that not all optimizer constructor settings are stored in state, so the manifest must carry them. |
| Target network | MLX modules expose parameter trees and updates. | Feasible. Hard sync and the V36 `tau=0.005` soft update must operate over exactly the same online/target leaves and schedule (`target_sync_updates=1000`). |
| Save, load, exact resume | MLX supports [array formats including Safetensors](https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html), module `save_weights`/strict `load_weights`, and an [official optimizer-state recipe](https://ml-explore.github.io/mlx/build/html/python/optimizers.html#saving-and-loading). | Backend-local exact resume is feasible, but the current `propevolve_recurrent_c51_v1` `torch.save` bundle contains online/target/optimizer/scaler state, update count, NumPy RNG, support, config, and manifest ([agent.py](../../src/propevolve/agent.py#L5632-L5779)). MLX cannot transparently replace that physical format. Existing `.pt` files remain immutable and PyTorch-readable. A later MLX backend needs a backend-tagged bundle plus a deterministic Torch↔MLX weight converter and authenticated receipt; it must never masquerade as the v1 schema. |
| Teacher-free inference | Current `strip_teacher()` rebuilds both networks without the auxiliary output, resets optimizer/scaler, and `assert_teacher_free()` fails closed on teacher and training-only settings ([agent.py](../../src/propevolve/agent.py#L5540-L5590)). | Feasible, but it is a contract test rather than merely omitting a tensor. The MLX export must physically contain no teacher head or teacher inputs and must reproduce the same C51 probabilities, Q values, masks, and actions in the teacher-free evaluator. Teachers remain training-only. |
| Lazy evaluation | In MLX, operations build a graph until `mx.eval`; printing, scalar extraction, conversion, and saving also force evaluation ([lazy-evaluation guide](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)). | The training step must evaluate loss, updated parameters, and optimizer state together. PropEvolve's many tensor-to-Python diagnostics and control-flow checks can introduce synchronization and erase any speed advantage; performance must include them, not a stripped loss kernel. Timers must synchronize/evaluate before and after the measured region. |
| Compilation | [`mx.compile`](https://ml-explore.github.io/mlx/build/html/usage/compile.html) can compile forward/backward/update, expects pure functions or explicit captured state, and recompiles on shape/dtype changes unless shapeless compilation is used carefully. | Test eager first. Fixed `(16, 96, …)` batches make compilation plausible, but reset grouping, active boundary sets, and Python repair control flow can alter graph structure. Compiled MLX is a second performance arm only after eager numerical parity; it cannot rescue a failed eager semantic port. |
| Determinism | MLX exposes explicit [Threefry keys and seed/state APIs](https://ml-explore.github.io/mlx/build/html/python/random.html). | The same integer seed does not imply the same initialization or random stream across frameworks. V37 must load one frozen canonical parameter bundle and one immutable batch ledger; it may not compare independently initialized models. Record MLX, PyTorch, macOS, and hardware identities and test repeatability empirically. |

## Unified memory is useful, not a prior speed verdict

MLX's programming model is genuinely convenient: [all MLX arrays reside in Apple Silicon's unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), and CPU or GPU operations can consume them without an explicit array move. That does not imply that MLX will use less physical memory or run this recurrent learner faster.

The existing PyTorch baseline is not running on a discrete-memory GPU. [`torch.mps`](https://docs.pytorch.org/docs/stable/notes/mps.html) already executes through Metal/MPS on the same Apple Silicon unified-memory hardware. More specifically, MLX's official interoperability guide says PyTorch 2.12+ ordinary MPS tensors use shared Metal storage and can be imported by MLX without copying when compatible. PyTorch still exposes distinct CPU/MPS tensor ownership and PropEvolve currently stages NumPy replay into device tensors, whereas MLX exposes a device-agnostic array model. The fair conclusion is: **MLX may remove some framework-level staging and scheduling overhead, but unified memory alone is not evidence of a speed or capacity win over the installed PyTorch 2.13 MPS baseline.** Measure the complete update.

## Three-job Optuna behavior

V37 freezes `n_jobs=3` ([sweep config](../../config/sweeps/stage2_v37_mlx_trend_confluence_strength_tpe.json#L1-L15)), and the engine runs isolated trial subprocesses. MLX supports [multi-process distributed execution](https://ml-explore.github.io/mlx/build/html/usage/distributed.html), but that is evidence that processes can cooperate—not that three unrelated training graphs on one GPU are faster. MLX offers no documented GPU partition for three independent trials. Each process has separate model, optimizer, lazy graph, and caches while all contend for the same GPU and physical memory pool.

Consequently, `n_jobs=1` and `n_jobs=3` are separate benchmark conditions. A result that wins in isolation but swaps, thermally throttles, or loses aggregate throughput at three jobs is not compatible with the V37 sweep execution contract. V37 keeps the V36 economic search fixed and changes only the learner backend and replay prefetch setting.

## Port versus retain

Retain unchanged outside the backend:

- transition/replay objects, sampling rules, NumPy RNG ownership, sequence IDs, action enum, masks, causal reset declarations, configuration validation, teacher generation, temporal splits, Optuna study logic, economic evaluation, selection gates, and teacher-free deployment contract;
- the PyTorch implementation as semantic oracle and production fallback;
- every existing V21/V36 config, checkpoint, receipt, and result.

Port behind a V37-only experimental seam:

- the network math and exact PyTorch-GRU weight mapping;
- reset-aware burn-in/learning unroll, C51 gather/projection/loss, Double-Q target calculation, action masking, all V36 auxiliary losses, gradient component extraction, PCGrad/economic-boundary projection, AdamW/clipping, and target updates;
- backend-local serialization plus a manifest-preserving parameter converter;
- synchronized timing, MLX allocator metrics, and teacher-free inference adapter.

Do not port the replay system, teachers, Optuna sampler, evaluator, or economics. That would confound backend correctness with a second redesign.

## Smallest falsifiable V37 benchmark

### Frozen inputs

Create these artifacts once, hash them, and use them in every arm:

1. A Torch-independent `.npz` fixture drawn from authenticated development replay—not validation or sealed 2026 rows—with exact transition IDs, observations, next observations, actions, rewards, termination flags, valid-action masks, reset rows, training-valid rows, teacher targets, cohort/pair metadata, and population weights.
2. Two batches at the V21/V36 production shape: `batch_sequences=16`, `sequence_length=96`, `burn_in=64`, `hidden_dim=128`, `atoms=51`, support `[-3, 3]`, `gamma=.997`, `n_step=8`. Batch A must exercise resets and terminal/nonterminal C51 projection. Batch B must activate primary, safety, opportunity, and at least one economic-boundary constraint in V36.
3. One canonical FP32 parameter bundle exported from a fixed PyTorch initialization/checkpoint, including the online and target networks. Do not initialize MLX independently. Freeze the replay ledger and NumPy seed; no framework RNG may affect the tested updates.

The semantic oracle is PyTorch eager FP32 on CPU. Performance arms are PyTorch eager FP32 MPS versus MLX eager FP32 GPU. Only after MLX eager passes all correctness gates may the exact same MLX step be tested under `mx.compile`.

### Gates, in order

**Gate 1 — forward and recurrent semantics**

- Encoder, every GRU time-step state, burn-in final hidden, reset-segment output, C51 logits, projected distributions, and Q values: maximum absolute error `<= 2e-5` and maximum relative error `<= 2e-4` against the CPU oracle.
- Every projected row finite, nonnegative within `1e-7`, and sum within `1e-6` of one.
- Greedy actions identical on every row whose oracle top-two Q gap exceeds `1e-5`; explicitly report tied/near-tied rows.

**Gate 2 — gradients and update semantics**

- Each named V21/V36 loss component: absolute error `<= 1e-5` and relative error `<= 2e-4`.
- For every nonzero parameter-tree gradient component (primary, safety, opportunity, each active economic boundary, and final projected gradient): cosine similarity `>= 0.9999` and relative L2 error `<= 1e-3`; zero/unused leaves and the active constraint set must match exactly.
- Global pre/post-clip norms and PCGrad conflict/projection decisions must match; tolerance for scalar norms is relative `1e-4`.
- One AdamW step with MLX `bias_correction=True`: parameter-delta cosine `>= 0.9999`, relative L2 error `<= 1e-3`, and maximum absolute parameter difference `<= 2e-5`.
- After 100 updates over the frozen A/B ledger: no non-finite value, identical active-boundary decisions, at least `99.9%` action agreement excluding oracle Q gaps `<= 1e-5`, and mean categorical KL from oracle `<= 1e-4` with no row above `1e-3`.

**Gate 3 — save/reload and teacher-free boundary**

- At update 50, save and reload online weights, target weights, optimizer state/config, counter, support, and NumPy/MLX RNG metadata. Reloaded inference before the next update must be bitwise equal within MLX; update 51 must differ from the uninterrupted MLX trajectory by at most `1e-7` per parameter.
- Convert the frozen PyTorch bundle into MLX and back with a receipt containing source hash, destination hash, framework versions, parameter-name mapping, shapes, and dtypes. The round-trip PyTorch evaluator must pass Gate 1.
- Strip teachers physically. The exported policy must have no teacher parameters or teacher inputs, pass the same fail-closed assertions, and match the PyTorch teacher-free actions under Gate 1.

**Gate 4 — isolated runtime and memory**

- Reuse the existing fresh-subprocess benchmark pattern ([runtime_benchmark.py](../../src/propevolve/runtime_benchmark.py)), but use the frozen full-contract ledger rather than its synthetic easy batch.
- Per backend: 10 warm-up plus 100 measured updates, five fresh-process repetitions, alternating backend order. Synchronize before/after timing. Record median and p95 wall time/update, process RSS, swap delta, and thermal state. For MLX also record [`get_active_memory`, `get_cache_memory`, and `get_peak_memory`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html); for PyTorch record [`torch.mps.driver_allocated_memory`](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory). Treat allocator counters as framework metrics, not total process memory.
- MLX passes the value gate only if it is either **at least 15% faster** in median full-update wall time, or uses **at least 20% less peak process RSS** while being no more than **5% slower**. Correctness gates may not be relaxed to obtain this result.

**Gate 5 — the actual three-job condition**

- Launch three isolated processes simultaneously for each backend, each replaying its fixed 100-update ledger. Repeat three times with alternating backend order.
- Require zero crashes/OOMs, no warning/critical macOS memory pressure, aggregate swap growth `<= 512 MiB`, all correctness receipts passing, and no process p95 update latency above `2x` its isolated p95.
- Require either at least **15% higher aggregate updates/second** than PyTorch's three-job baseline, or at least **20% lower peak aggregate RSS** with makespan no more than **5% worse**. Otherwise MLX fails the V36-compatible sweep-use case even if its one-job result is attractive.

### Stop rule

Stop at the first failed gate; do not tune tolerances, objectives, batch composition, loss weights, or concurrency after looking at results. A pass authorizes only an optional backend POC on research rows. It does not authorize replacing PyTorch, changing V21/V36, or opening the sealed period.

## Implemented evidence

The implementation first exposed a performance failure: compiling only the
MLX forward function made the full multi-loss update slower because every
separate PCGrad backward rebuilt the MLX reverse graph. Compiling and caching
the recurrent VJP corrected that boundary.

On a preserved, authenticated V36 balance-pass replay artifact with the actual
production shape (`2578` observations, `16 x 96` sequences, `64` burn-in,
`hidden_dim=128`, `51` atoms), the bounded matched result was:

| Backend | Full update | Final loss | Relative drift |
|---|---:|---:|---:|
| PyTorch MPS | 776.11 ms | 4.10604572 | 0 |
| MLX recurrent adapter | 545.83 ms | 4.10604572 | 0 |

That is a `1.42x` full-update speedup after five warm-up and twenty measured
updates. The complete test
suite also passes, including identical recurrent reset/burn-in behavior,
PyTorch-to-MLX warm start, optimizer/checkpoint continuity, replay-driven
Expansion/Regime/Trend economic boundaries, PCGrad/boundary repair, physical
teacher stripping, and teacher-free evaluation.

The initial compiled bridge also exposed a process-lifecycle defect: PyTorch
executes a custom autograd backward on a worker thread, and MLX requires every
worker that uses `mx.compile` to clear its own streams before exit. Otherwise
the thread-local compile-cache destructor can run after Python finalization and
segfault in `PyGILState_Ensure`. The adapter now owns one long-lived MLX worker;
it creates and uses both compiled functions there, then clears that thread's
streams and joins it during `atexit`. The production E2E test, a dedicated
subprocess-exit regression, and the full suite complete without a new macOS
crash report. This follows the upstream MLX lifecycle requirement documented in
the [main-thread compile-cache cleanup fix](https://github.com/ml-explore/mlx/pull/4373).

The benchmark report is local run evidence and is intentionally not a packaged
training artifact. Multi-process sweep throughput remains a separate gate; no
sweep was restarted by this work.

## Replay-to-learner staging result

The durable replay remains packed NumPy data. Replacing that storage with MLX
arrays is not justified: MLX documents that NumPy-to-MLX conversion copies the
data, and a matched production-shape microbenchmark found the NumPy-to-MLX-to-
PyTorch path slower than direct NumPy-to-PyTorch MPS staging. Loading the full
replay into MLX would also duplicate a large persistent artifact in memory and
would change checkpoint and replay compatibility.

The accepted optimization is narrower. The learner now constructs one
contiguous causal observation batch with shape `(batch, sequence + 1,
observation_dim)`, transfers it once, and exposes current and next observations
as views. This replaces two NumPy stacks, two device transfers, and one device
concatenation. At the authenticated V36 production shape, the staging-only
benchmark improved from `11.80 ms` to `7.53 ms` (`1.57x`) and removed about
`32 MB` of duplicate live device observation buffers per update. The complete
MLX update improved from `545.83 ms` to `522.63 ms` with the same final loss
(`4.10604572`); the matched PyTorch path also benefits.

Replay sampling itself measured about `17.4 ms` per batch, roughly `3.3%` of
the complete MLX update. It is therefore a secondary optimization target. Any
future packed-batch sampler must preserve the exact sampled transition ledger,
sequence boundaries, reset rows, burn-in, pair metadata, and replay population
weights before it can replace the current sampler.

The next MLX performance candidates, in evidence order, are:

1. Profile the complete learner with Metal capture before changing graph
   boundaries.
2. Keep fixed batch and sequence shapes so compiled functions remain cached.
3. Consider compiling a larger pure-MLX update only if the remaining
   PyTorch/MLX synchronization is proven material; do not move C51, PCGrad, or
   AdamW merely for architectural symmetry.
4. Treat allocator cache limits as an operational control only if measured MLX
   cache growth or memory pressure becomes material. Frequent unconditional
   cache clearing is not a throughput optimization.

References: [MLX NumPy and DLPack interoperability](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/numpy.rst),
[MLX compilation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst),
and [MLX memory management](https://github.com/ml-explore/mlx/blob/main/docs/src/python/memory_management.rst).

## Overall verdict

MLX can accelerate the recurrent core without changing the V21/V36 learning
contracts because the existing PyTorch learner continues to own every semantic
boundary. The precise claim supported now is:

> The optional MLX recurrent adapter is numerically compatible and faster in a
> bounded one-process production-replay benchmark. It is not yet authorized for
> the three-worker sweep until replay staging and matched multi-process evidence
> pass.

PyTorch 2.13 MPS remains the authoritative default and fallback.

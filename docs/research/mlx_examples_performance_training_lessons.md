# MLX Examples lessons for PropEvolve performance and training

## Scope and verdict

This is a research-only source audit. It does not authorize changing or
interrupting the active V38 r2 sweep. Sources were frozen at
[`mlx-examples@796f5b5`](https://github.com/ml-explore/mlx-examples/tree/796f5b53cab69a3d48a44233ce21aae889e94a08)
and [`mlx@90846ad`](https://github.com/ml-explore/mlx/tree/90846adf0766785fb6560a6dedd32b1557b5794c),
plus [`mlx-data@2f431e9`](https://github.com/ml-explore/mlx-data/tree/2f431e90a06c33d2e3f78019b32c563e8fd8a71a),
and compared with PropEvolve `3296bc2`.

The official examples contain no C51, R2D2, experience replay, or recurrent-RL
trainer. They support **runtime engineering**, not a new learning algorithm.
PropEvolve already implements most of the high-value MLX practices: long-lived
compiled recurrent forward and VJP functions, FP32 parity, zero-copy-compatible
PyTorch-MPS/MLX DLPack views, one contiguous causal observation transfer,
explicit evaluation, and MLX allocator metrics. The strongest remaining
candidate is to make one reset-aware recurrent call fixed-shape and to follow
MLX's own GRU implementation by precomputing the whole sequence's input-gate
affine once. This must be an isolated post-sweep benchmark, not a live change.

## What the official sources show

- MLX compilation pays a first-call cost, caches a compiled function, and
  recompiles when input shape, dtype, or arity changes
  ([compile guide](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/compile.rst#L42-L77)).
  The Transformer LM compiles forward, backward, and optimizer update as one
  state-capturing step, then evaluates state once per iteration
  ([`transformer_lm/main.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/transformer_lm/main.py#L98-L114)).
  MLX recommends compiling the outermost transformed function
  ([compile guide](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/compile.rst#L421-L442)).
- `mx.eval`, scalar `.item()`, NumPy conversion, memory access, and saving force
  lazy work to execute. MLX recommends an evaluation at the outer optimization
  iteration rather than many partial evaluations, and warns that tensor-driven
  Python control flow evaluates arrays
  ([lazy evaluation](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/lazy_evaluation.rst#L64-L144)).
- MLX's GRU performs the input affine for every timestep in one `addmm` before
  its recurrent loop; only the hidden projection remains inside the loop
  ([`recurrent.py`](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/python/mlx/nn/layers/recurrent.py#L156-L198)).
  PropEvolve's current custom primitive instead performs the recurrent input
  projection inside its timestep loop
  ([`mlx_backend.py`](../../src/propevolve/mlx_backend.py#L50-L73)).
- MLX arrays use unified memory, but CPU/GPU overlap helps when work is
  independent; MLX inserts dependencies across streams when it is not
  ([unified memory](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/unified_memory.rst#L8-L48)).
  A GRU's timestep chain is dependent, so the guide is not evidence that moving
  individual gates to CPU will help.
- For PyTorch interoperability, `mx.asarray`/`mx.from_dlpack` can import ordinary
  PyTorch 2.12+ MPS tensors without a copy when their Metal buffer is shared;
  `mx.array` copies, and cross-framework DLPack does not synchronize pending
  Metal work
  ([NumPy/DLPack guide](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/numpy.rst#L75-L134)).
  PropEvolve already uses `mx.asarray`, `torch.as_tensor`, and an explicit MPS
  synchronize at the boundary
  ([`mlx_backend.py`](../../src/propevolve/mlx_backend.py#L76-L84),
  [`mlx_backend.py`](../../src/propevolve/mlx_backend.py#L243-L257)).
- The CIFAR and CVAE examples batch before bounded prefetch
  ([`cifar/dataset.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/cifar/dataset.py#L16-L29),
  [`cvae/dataset.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/cvae/dataset.py#L19-L35)).
  FLUX pre-encodes immutable inputs, concatenates them once, evaluates them, and
  then indexes batches
  ([`flux/trainer.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/flux/flux/trainer.py#L79-L98)).
  For seeded replay, order matters: the official data-layer docs state that
  ordinary stream prefetch is nondeterministic and provide `ordered_prefetch`
  when order must be retained
  ([`buffers_streams_samples.rst`](https://github.com/ml-explore/mlx-data/blob/2f431e90a06c33d2e3f78019b32c563e8fd8a71a/docs/src/buffers_streams_samples.rst#L105-L142)).
- FLUX shows separate compiled paths for ordinary steps and gradient
  accumulation, captures model/optimizer/random state, evaluates state and
  pending gradients together, and samples peak memory
  ([`dreambooth.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/flux/dreambooth.py#L183-L230),
  [`dreambooth.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/flux/dreambooth.py#L263-L285)).
- Current MLX exposes active, peak, and cache bytes; active bytes exclude cached
  buffers. Memory and cache limits are guidelines, while wired memory remains
  resident and must be below total memory
  ([memory API source](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/python/src/memory.cpp#L10-L124)).
  The WAN example clears cache only after deleting an entire model phase, not
  every iteration
  ([`video/wan2.1/txt2video.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/video/wan2.1/txt2video.py#L119-L142)).
- MLX Metal capture requires `MTL_CAPTURE_ENABLED=1`; an optional
  `MLX_METAL_DEBUG` build improves kernel labels and preserves source for Xcode
  inspection
  ([Metal debugger](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/dev/metal_debugger.rst#L6-L46)).
- MLX distributed training is replicated-model data parallelism with a shard per
  rank and averaged gradients
  ([official data-parallel example](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/examples/python/distributed_data_parallel.py#L1-L102)).
  This is not independent Optuna trial parallelism.

## Safe semantic-preserving candidates for after V38

These keep the frozen replay, C51/R2D2 losses, PCGrad, optimizer, batch and
sequence shapes, checkpoints, action contract, and teacher-free evaluator.
Each still needs the existing numerical-parity and full-update benchmark.

1. **Hoist the GRU input projection out of the timestep loop.** Compute
   `encoded @ W_ih.T + b_ih` once for the complete sequence, slice its three
   gates inside the recurrent loop, and leave the hidden projection recurrent.
   This matches MLX's own GRU organization and replaces up to 96 small
   per-timestep input matmuls with one batched matmul for a 96-step sequence.
2. **Add a final-hidden-only no-gradient path.** Burn-in and frozen target/
   retention passes often discard the full hidden sequence, but the bridge
   currently stacks, evaluates, and returns it. A compiled function that returns
   only the final hidden state preserves recurrence while avoiding an unused
   sequence output. MLX's lazy guide confirms unused outputs need not be
   evaluated, although their graph still has a construction cost.
3. **Batch host-control summaries.** The current Torch-owned learner has several
   accelerator scalar presence checks (`.any().item()`) inside one update. Form
   the same booleans together and cross the host boundary once where feasible;
   do not remove fail-closed validation or calculate inactive losses.
4. **Benchmark the already-existing one-batch prefetch seam.** V38 declares
   `training.prefetch_batches: 0`, while the trainer supports a bounded single
   producer. Test `1`, not an unbounded queue, and first prove an identical
   sampled-transition ledger. The upside is bounded because
   [prior local evidence](propevolve_mlx_backend_feasibility.md#replay-to-learner-staging-result)
   put replay sampling near 3.3% of update time. Retain PropEvolve's ordered
   future deque; do not replace it with nondeterministic stream prefetch.
5. **Keep the current zero-copy and checkpoint ownership.** Continue using
   `mx.asarray`/`torch.as_tensor`, explicit producer synchronization, contiguous
   batches, and the existing Torch checkpoint bundle. These are already the
   correct MLX lessons; replacing them would add copies or weaken resume.

## Benchmark-required ideas

1. **One fixed-shape reset-aware recurrent call.** PropEvolve currently groups
   reset patterns and invokes the MLX primitive on variable group sizes and
   segment lengths. That can create shape-specialized compilation variants and
   multiplies cross-framework evaluations. Pass the reset mask into one
   `(batch, sequence, feature)` compiled recurrence and zero hidden state before
   each marked timestep. This is mathematically compatible but structurally
   larger: compare every hidden state, logits, gradients, PCGrad decisions, and
   100-update trajectory before considering it safe.
2. **Batch the multiple recurrent cotangents used by PCGrad.** A `vmap` or one
   multi-cotangent reverse operation may amortize VJP and bridge overhead but
   can multiply activation/gradient memory. Test only after profiling proves
   repeated VJP calls dominate on the 16 GB machine.
3. **Enlarge the pure-MLX graph boundary only if the profile demands it.** The
   examples compile loss/backward/update together, but PropEvolve deliberately
   leaves C51, dynamic cohorts, PCGrad, AdamW repair, and checkpoint state in
   PyTorch. Porting those pieces is justified only if measured framework
   boundaries dominate; it requires a new full semantic oracle, not a refactor
   for symmetry. If attempted, capture online/optimizer state as inputs and
   outputs, target-network state as an input so target syncs cannot become stale
   compile-time constants, and `mx.random.state` if the graph samples.
4. **Treat batch size and gradient accumulation as learning experiments.** FLUX
   demonstrates how to compile accumulation paths, not that accumulation is
   equivalent for PropEvolve. Different recurrent batches change cohort balance,
   PCGrad conflicts, optimizer frequency, and economic-boundary repair. Freeze
   effective batch and update semantics or validate it as a new recipe.
5. **Tune cache limits only from three-process evidence.** Record process RSS,
   swap/pressure, plus MLX active/cache/peak bytes per trial. If cache, rather
   than active tensors, causes pressure, test a conservative per-process cache
   limit in an isolated three-worker benchmark. Do not infer a 16 GB-safe value
   from the per-process default.

## Inapplicable or risky ideas

- **Do not use `mlx.launch -n 3` for the existing sweep.** Data parallelism
  averages gradients for one replicated model; V38 needs three independent
  Optuna trials. On one M1 it also replicates model/optimizer/cache state without
  adding a GPU.
- **Do not enable `shapeless=True` on the current recurrent primitive.** The
  current Python `range(encoded.shape[-2])` makes the graph conditional on
  sequence length. MLX explicitly warns shapeless compilation is incorrect for
  shape-dependent graphs
  ([compile guide](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/usage/compile.rst#L446-L515)).
- **Do not copy Encodec's custom Metal LSTM into training.** That example is an
  inference implementation
  ([`encodec.py`](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/encodec/encodec.py#L14-L94)).
  A trainable custom kernel requires a correct custom VJP as well as forward
  parity; MLX documents that obligation explicitly
  ([custom-kernel guide](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/dev/custom_metal_kernels.rst#L321-L464)).
- **Do not use gradient checkpointing, quantization, FP16, smaller batches, or
  shorter sequences as transparent speedups.** Checkpointing recomputes
  intermediates; lower precision and altered batch/sequence geometry change the
  current numerical or recurrent-learning contract. They are fallback memory or
  new-recipe experiments only, not supported improvements for this compact FP32
  recurrent core.
- **Do not call `clear_cache()` every update or raise the wired limit.** The
  examples clear at phase boundaries. Frequent clearing defeats buffer reuse;
  wired memory on a 16 GB system can worsen pressure across three trials.
- **Do not move the durable replay wholesale into MLX.** NumPy-to-MLX normally
  copies, duplicates the resident replay, and changes artifact compatibility.
- **Do not adopt examples that save only weights/adapters.** Exact PropEvolve
  resume also needs online/target weights, optimizer and schedule configuration,
  counters, replay/sampler state, and RNG identities. MLX's optimizer docs warn
  that betas and epsilon are not necessarily stored in optimizer state
  ([optimizer docs](https://github.com/ml-explore/mlx/blob/90846adf0766785fb6560a6dedd32b1557b5794c/docs/src/python/optimizers.rst#L34-L67)).

## Smallest post-sweep benchmark

Use the existing authenticated replay benchmark and canonical FP32 weights.
First add observation only in the benchmark harness: distinct MLX input shapes,
recurrent-call/evaluation count, per-update median and p95, process RSS and swap,
and MLX active/cache/peak bytes. Capture one warmed full update in a fresh
subprocess; never attach Metal capture to the active sweep.

Run the candidates one at a time in this order: whole-sequence input projection,
final-hidden-only burn-in, fixed-shape reset mask, batched host summaries, then
prefetch `1`. For every arm freeze transition IDs/order, reset masks, weights,
and loss settings; compare all recurrent states, Q distributions, named loss and
gradient components, PCGrad/boundary decisions, parameter deltas, checkpoint
reload, and teacher-free actions before using runtime results. Retain only a
measured full-update or three-worker benefit with no correctness, memory, or
economic-contract regression.

No source reviewed here supports changing V38's objective, replay composition,
sequence horizon, optimizer, concurrency, or validation gate.

## Optimization 1 result

After the V38 r2 sweep was explicitly stopped, the complete-sequence recurrent
input projection was implemented with MLX `addmm`. On the identical
authenticated V38 replay batch, five warm-up plus twenty measured full learner
updates took `526.24 ms/update`, versus `575.53 ms/update` for the isolated
legacy primitive (`1.094x`, or about `8.6%` lower latency). Both produced the
same final loss, `4.109043121337891`. The complete regression suite and the
MLX reset/burn-in, learner-step, checkpoint, teacher-free, and shutdown parity
tests passed. Optimization 2 remains unimplemented.

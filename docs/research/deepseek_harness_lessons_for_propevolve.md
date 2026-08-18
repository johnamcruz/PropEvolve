# DeepSeek Harness lessons for PropEvolve's RL experimentation pipeline

## Scope and verdict

This is a source audit of DeepSeek AI's `deepseek-harness` at commit
[`99f6f02`](https://github.com/deepseek-ai/deepseek-harness/tree/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca).
It asks whether the repository contains ideas that can materially improve
PropEvolve's ability to discover a profitable recurrent RL policy. It does not
authorize changes to the active Stage 2A run.

The central finding is important: **DeepSeek Harness is an agent application
harness, not an RL-training harness.** DeepSeek describes it as an open-source
agent harness with an "everything is a plugin" architecture, and labels it a
developer preview with compatibility-breaking changes expected
([README](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/README.md#L5-L11)).
A complete source search found no policy optimizer, reward model, PPO/GRPO,
distributional RL, replay buffer, gradient algorithm, or model-training loop.
It therefore offers **no new loss, replay method, exploration method, or
recurrent-RL algorithm** for PropEvolve.

Its useful contribution is narrower: an event-sourced way to make long-running
work reconstructable, restartable, observable, bounded, and attributable. That
can improve RL *algorithm discovery* by making one-component ablations and
resume behavior trustworthy. It cannot make an ineffective A+ objective learn
by itself.

## What DeepSeek Harness actually does

### Architecture and execution algorithm

The runtime is a Cordis plugin tree. Model adapters, tools, persistence, and the
agent loop register services and typed events into a shared context, and their
registrations unwind when a plugin unloads
([architecture](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/architecture.md#L9-L27)).
Cordis expresses dependencies by named services, uses typed `emit`, `waterfall`,
`parallel`, and `serial` events, and treats registrations as reversible effects
([Cordis primer](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/cordis-primer.md#L7-L25)).

Its central algorithm is a conversation driver:

```text
claim input -> append turn/step events -> call an LLM -> execute tool calls
            -> append results -> repeat while more work is owed -> end turn
```

The exact step/turn flow is documented in the repository
([turn flow](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/architecture.md#L63-L90)).
This is agent orchestration, not policy learning.

### Data, replay, and resume

The authoritative data structure is an append-only `SessionEvent` log. Model
history, transcripts, telemetry, fork, and resume are projections of that log.
The key invariant is "model-visible means logged": anything sent to the model
must be reconstructable from durable events
([session log](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/architecture.md#L92-L96)).

This repository's use of the word `replay` means reconstructing an agent
session or replaying a recorded LLM response. It is **not experience replay**
and has no correspondence to PropEvolve's recurrent sequence sampler.

Persistence copies events into bounded write batches and exposes a flush
barrier. A cold load preserves a crash-interrupted turn and appends a synthetic
`interrupted` boundary instead of truncating already durable work
([persistence and recovery](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/subsystems/persistence.md#L9-L19)).
Its JSONL and SQLite backends implement the same append-only contract
([backends](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/subsystems/persistence.md#L231-L236)).

### Runtime and concurrency

DeepSeek Harness does not contain a distributed accelerator training runtime.
Its background-job registry is explicitly process-local, and its own docs say a
durable or cross-process backend would need new identity, restart, ownership,
and observation semantics
([jobs limitations](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/jobs/jobs/README.md#L36-L40)).

The workflow engine can fan out child agents under concurrency and total-work
caps, but it currently has no journal or restart support
([workflow contract and limits](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/workflow/workflow/README.md#L11-L19),
[workflow limitations](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/workflow/workflow/README.md#L53-L59)).
One worker thread per run prevents synchronous script work from blocking the
host and provides a force-termination boundary; it is explicitly not a
security sandbox
([worker runtime](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/workflow/workflow-worker-thread/README.md#L5-L22)).
It caps concurrency, total children, items per call, synchronous execution, and
dispose grace
([worker controls](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/workflow/workflow-worker-thread/README.md#L75-L86)).

For PropEvolve this supports bounded CPU-side analysis or evaluation fan-out,
not parallel MPS training. Competing MPS runs should remain serialized.

### Objectives, rollout controls, and evaluation

There is no trainable objective. The "goal" packages manage an agent task, and
the "trajectory" UI renders conversation activity; neither computes rewards or
updates model weights.

The repository's evaluation controls are software-quality controls:

- per-file coverage and contract-regression tests;
- real-provider end-to-end tests;
- deterministic transcript/snapshot replay;
- tests through the real assembled entry path;
- world-state verification rather than trusting an agent's self-report.

These policies are explicit in its testing guide
([test tiers](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/testing.md#L7-L19),
[world and entry-path verification](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/testing.md#L21-L35)).
They are useful analogies for testing PropEvolve's trainer and campaign runner,
but they provide no market evaluator, temporal split, reward, or promotion
criterion.

### Failure safeguards

The most reusable safeguards are:

- record independent terminal facts independently rather than hiding timeout,
  signal, and exit status behind one outcome;
- teardown must await quiescence instead of merely issuing cancellation;
- one bad observer must not break the core lifecycle;
- scrub secrets and use private unpredictable temporary paths
  ([defensive patterns](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/defensive-patterns.md#L7-L33));
- checkpoint intent before a model call or side-effecting tool, and fail closed
  if the checkpoint cannot be made durable
  ([checkpoint policy](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/session/session-checkpoint-policy/README.md#L5-L23));
- do not automatically retry a side effect whose outcome is unknown after a
  crash
  ([unknown outcomes](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/session/session-checkpoint-policy/README.md#L27-L44));
- apply explicit per-operation deadlines, while acknowledging that cooperative
  cancellation is not a hard kill
  ([timeout policy](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/packages/guard/timeout-policy/README.md#L18-L36)).

## What PropEvolve already has

PropEvolve already implements most of the high-value equivalents, so importing
Cordis or creating a second orchestration framework would be regression:

- [`orchestration.py`](../../src/propevolve/orchestration.py) uses the shared
  ML Training Loop, durable `JsonRunStore`, stage receipts, a run lock,
  authenticated candidate identities, and durable reasoning recovery.
- [`training.py`](../../src/propevolve/training.py) binds resume to the complete
  recipe, warm-start checkpoint, cache/teacher manifests, and runtime source
  hashes. Recovery restores model and optimizer state, environment RNG, replay
  content, replay sampler RNG/schedule, and progress; it fails closed when
  orphan diagnostics exist without a valid recovery checkpoint.
- Training and validation already append episode diagnostics to JSONL, preserve
  partial validation evidence, write atomic summaries and receipts, retain pass
  checkpoints immutably, short-circuit failed candidates, discard teachers
  before teacher-free validation, and bind final artifacts to hashes.
- [`replay.py`](../../src/propevolve/replay.py) already owns recurrent episode
  boundaries, burn-in-aware sequence construction, balanced cohorts, and a
  resumable sampler state. A Harness "session replay" must not replace it.

The active v20 contract also already freezes the recurrent architecture,
replay, optimizer, risk, evaluator, and budget while changing only the generic
canonical A+ comparison
([v20 contract](../experiments/stage2a_v20_generic_canonical_aplus.md)).

## Evidence-backed ideas worth reusing

### 1. One canonical training-influence ledger

Translate "model-visible means logged" into:

> **Gradient-affecting means logged or content-addressed.**

PropEvolve has the underlying data, but it is distributed among recovery
checkpoints, campaign state, candidate manifests, training diagnostics, and
validation diagnostics. The missing high-value projection is one append-only
index that links those existing artifacts at semantic boundaries.

The ledger should reference—not duplicate—the existing payloads:

- `run-started`: parent, effective recipe and single allowed delta, code/data/
  cache/teacher identities, seeds, and budgets;
- `checkpoint-committed`: episode/step boundary plus model, optimizer, replay,
  environment-RNG, sampler-RNG, and diagnostic-prefix hashes;
- `training-influence`: per checkpoint window, replay cohort mass, original
  Long/Short A+ pair mass, raw and scaled loss components, and the exact
  teacher/autonomy schedule values;
- `training-ended`: completion, short circuit, interruption, or failure as
  independent facts;
- `validation-started` and `validation-ended`: exact checkpoint and evaluator
  identity, teacher lookup count, pass/blow/near-blow/timeout and per-slice
  evidence;
- `candidate-handed-off`: authenticated artifacts and the frozen next stage.

The ledger is useful to algorithm discovery because it can prove whether an
auxiliary objective was active, received comparable replay mass, and changed
the intended Q separation before economics are compared. It would have made
the earlier 77% A+ pair-mass loss an immediate attributable training-influence
change rather than an after-the-fact inference.

It should be a projection over the existing trainer and ML Training Loop—not a
new event framework and not a second source of truth.

### 2. Semantic checkpoints before irreversible stage transitions

Harness makes durability a barrier before an external side effect. The direct
PropEvolve equivalent is to require an authenticated checkpoint and influence
prefix before:

- beginning teacher-free validation;
- selecting a retained policy;
- exporting a candidate;
- launching Stage 2B from Stage 2A.

PropEvolve already does much of this. The remaining value is to make the
boundary explicit in one ledger so an interrupted process cannot leave an
ambiguous "validation may have used this checkpoint" state. Never rerun an
unknown side effect automatically; verify its artifact/receipt first.

### 3. Capability seams only at experiment boundaries

Harness demonstrates that a stable capability interface can let providers vary
without changing consumers. PropEvolve can use that vocabulary narrowly for
the existing trainer, auxiliary objective, evaluator, and artifact store. This
would make an A+ loss ablation replace exactly one provider while the recurrent
C51/R2D2-like learner, replay population, evaluator, and campaign remain fixed.

This does **not** justify pluginizing individual tensors or hot-swapping losses
mid-run. A run's mounted components must be resolved once, dumped as an
effective composition, hashed, and frozen. Harness itself supports dumping the
actual composed plugin tree
([configuration layers](https://github.com/deepseek-ai/deepseek-harness/blob/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca/docs/architecture.md#L15-L37));
the PropEvolve analogue is an exact effective-recipe and component-identity
snapshot attached to every candidate.

## Smallest future experiments

These are future pipeline experiments only. They must not stop or alter the
active v20 run.

### Experiment 1: deterministic checkpoint/resume equivalence

On a small research-only training slice, run the same frozen recipe in two
ways:

1. uninterrupted through a declared checkpoint boundary;
2. stop at that boundary, reload the authenticated recovery checkpoint, and
   finish the same updates.

Compare the ledger projection, sampled-cohort hashes, component losses, final
weights/Q-values on a golden recurrent batch, and episode outputs within the
declared device tolerance. This is the cheapest test of whether resume preserves
the exact training influences needed for valid long Stage 2 runs.

Falsifier: any unexplained divergence in sample identities, schedule, recurrent
Q-values, or final evidence. Fix the first divergent boundary before more
expensive algorithm search.

### Experiment 2: one-component A+ objective replay ablation

If v20 fails its teacher-free transfer gate, freeze one authenticated training
checkpoint and a deterministic sequence-batch corpus from research data. From
identical weights and batches, run a short matched update audit with:

- the frozen baseline objective; and
- only the generic canonical A+ auxiliary objective changed.

Record original-side pair mass, winning-versus-failed candidate advantage,
WAIT/Long/Short gradient mass, recurrent golden-batch parity, and teacher-free
Q separation. Promote the objective to another full campaign only if it moves
the intended separation without side collapse or universal WAIT. This is not a
profitability test; it is a cheap mechanistic falsifier before 200–450 episode
compute. Economic acceptance still requires the unchanged temporal evaluator.

## What not to adopt

- Do not replace PropEvolve's recurrent C51/R2D2-like learner or episode-safe
  replay with Harness machinery; none exists there.
- Do not import the plugin framework wholesale or create parallel campaign,
  persistence, telemetry, or receipt systems.
- Do not hot-reload trainers, losses, teachers, or evaluators during a run.
  Harness supports dynamic plugins for an application; ML evidence requires a
  frozen run identity.
- Do not treat conversation-session replay as experience replay.
- Do not run concurrent MPS trainers because Harness can fan out worker tasks.
- Do not use an LLM agent, tool loop, or self-reported result as the economic
  verifier.
- Do not copy software snapshot tests as a substitute for chronological,
  teacher-free, prop-economic validation.
- Do not build a generic distributed runtime. Harness itself does not provide
  one, and the current single-MPS machine does not need one.

## Recommendation

DeepSeek Harness does not reveal a better RL algorithm for learning generic A+
Expansion patterns. PropEvolve should keep the active v20 algorithm and its
teacher-free falsifier unchanged.

The one Harness-inspired improvement likely to materially increase discovery
efficiency is a **thin append-only training-influence ledger derived from
existing artifacts**. It would make resumed runs, one-component ablations, and
Stage 2A-to-Stage 2B handoffs exactly attributable while avoiding another
orchestrator. Implement it only after the current run completes, and only if
the existing receipts cannot already answer the chosen ablation. The first use
should be the small deterministic resume test; the second, only if v20 fails,
should be the fixed-batch A+ objective ablation above.

This improves the probability of finding the right algorithm by reducing false
comparisons and wasted full campaigns. It does not itself improve pass rate,
blow rate, or near-blow behavior; those still require evidence that the learned
objective transfers to teacher-free economic outcomes.

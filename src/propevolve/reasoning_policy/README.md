# Reasoning-policy challenger

Optional, specialist-conditioned trading policy. The existing C51 training and
environment remain unchanged. This is an implementation checkpoint, not a
trained or economically validated replacement.

## Current path

1. `inputs.specialist_account_fields` reads aligned Expansion, Trend and Regime
   scores plus the existing normalized account observation. An optional audited
   Volume source joins this same interface without changing C51's teacher loader.
2. `context.RollingContext` supplies completed-bar history with an availability
   mask. `config/reasoning/context.json` starts with 20 bars; length is configurable.
3. `labels.label_entry_opportunity` reuses the existing net target-before-stop
   semantics, including next-open execution and adverse-first OHLC ambiguity.
4. `labels.label_actions` reconstructs the same simulator state for every legal
   action, then follows a declared continuation policy to pass/blow/timeout.
   These are realized conditional outcomes, not oracle pass probabilities.
5. `collector.collect_examples` produces separate market-understanding and action
   SFT records. Paths, policies, sources, economics and budgets are explicit inputs.
6. `dataset.write_supervised_dataset` writes disjoint chronological roles with
   full label reserves and a sealed boundary. A reviewed matching `audit.json`
   is required before the MLX preparation command can proceed.
7. `mlx_sft` creates encoded token/offset datasets and delegates adapter
   training to MLX-LM. SFT, RL and inference share one chat-tokenization boundary;
   the native dataset adapter does not wrap the prompt a second time.
   Inputs are loss-masked. Overlength samples fail rather than
   silently truncate. The first model candidate is quantized, so LoRA is QLoRA.
8. `policy.MLXActionPolicy` scores only legal action completions. These scores
   are log likelihoods, not C51 Q values. `evaluation.evaluate_policy` returns
   existing simulator outcomes and explicitly reports specialist dependence.

## Execution status

Do not launch yet. Work was paused at code-only implementation while another
training task was active. The first eight challenger tests passed before that
pause; subsequent additions and the full regression suite have NOT been run.
No MLX-LM model load, adapter update, checkpoint parity test or economic evaluation
has been executed. The selected model is provisional, not benchmark-selected.

The optional runtime dependency is the `reasoning` extra. It has not been
installed by this task. There is no model download at import time.

## Configurable backbone and causal trade context

`MLXActionPolicy.from_config(path)` reads `model`, `adapter_path`, and
`max_seq_length` from any JSON recipe, including the SFT recipe. The model can
be a compatible MLX-LM repository ID or local directory. Set `adapter_path` to
null for base-model inference; SFT requires an explicit new output directory.
Changing the base requires a matching adapter, not reuse of another model's LoRA
weights. Loading checks the native adapter metadata before loading large weights.
This is a compatibility guard, not a checkpoint content-authentication claim.

Prompt-template options inherit `config/reasoning/defaults.json` and can be
overridden with `chat_template_kwargs` in the recipe. Use the saved effective
training recipe for inference to preserve prompt parity. No model-name allowlist
or model-specific branch is needed. The decision interface stays
`decide(context, legal_actions)`; simulator actions and outcomes do not change.
Changing templates or models requires rebuilding tokenized views and revalidating
learning. Arbitrary models, tokenizer behavior, and economic equivalence are not
guaranteed by configuration alone.

The reasoning context also selects completed-trade-history fields by name:
current open-trade MFE/MAE in original-risk units, current R, giveback from MFE,
holding bars, and explicit open-position/risk-availability masks. They come from
one shared simulator snapshot used by collection and evaluation. They do not
extend or reorder the C51 observation. MFE/MAE are gross price excursions;
account equity and economic labels retain the simulator's fees. Final future
excursions are NOT inference inputs. These inputs can support learning exits,
but their presence is not evidence that the model has learned profitable exits.

The new config and excursion tests are written but unexecuted, including
Long/Short symmetry, future-price mutation, flat reset and model/adapter mismatch.

When execution is authorized and the source dataset audit passes, the preparation
entry point is:

```sh
python -m propevolve.reasoning_policy.mlx_sft --config config/reasoning/sft.json --view runs/reasoning-challenger/mlx-view
```

This loads the configured model to check quantization and exact token lengths,
but does not train unless `--train` is also supplied. Run from the repository
root or provide explicit resolved paths in the JSON. Existing output paths are
never overwritten by the wrapper.

## Explicit job stages

The module entry point consumes the job JSON, with no campaign-specific Python
edits. It never starts a stage automatically:

```sh
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json check
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json collect
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json prepare
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json train
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json rl
python -m propevolve.reasoning_policy.job --config config/reasoning/development.json evaluate
```

Run from the declared workspace root. Source recipes reuse existing market/cache
loaders and challenge configuration, not the C51 campaign runner. Collection uses
a hash-verified local C51 checkpoint as a declared continuation; its weights are
shared but each branch has independent recurrent state. This is a label generator,
not the reasoning learner. Dataset audit is a separate required stage of review:
collection never authors its own passing causality receipt. Both initial partial
and full rolling contexts can enter collection, matching inference.

Preparation is reusable only when its rendered data, audited source and effective
recipe still match. Model and adapter outputs are never silently overwritten.
`market_job.json` collects the market dataset; `development.json` collects the
separate action dataset. `market_sft.json` starts from the base, then
`policy_sft.json` warm-starts its matching adapter for account-aware action SFT.
The same model, LoRA shape and chat template must match at the handoff. All paths
and stages are JSON-controlled; none of these commands has been launched.

Market completions include both-side target-before-stop outcomes and full-horizon
gross MFE/MAE plus terminal net R. These future quantities appear only in targets,
never causal prompts. Full-horizon MFE is not an assertion that a stop-managed
trade survived to that extreme. Actual executable action labels still come from
the existing simulator.

### RL implementation status

`rl.py` implements a bounded actor-only clipped policy-gradient draft. It samples
legal actions from the reasoning policy in the **existing HistoricalChallengeEnv**
and uses existing simulator rewards, including full terminal outcomes. It does
not create another trading simulator or change any prop-firm rules.

Each group repeats one identical episode start under one frozen policy version.
The learner uses complete undiscounted return-to-go and a leave-one-out baseline
from other independently sampled trajectories in the group. PPO-style clipping,
old-policy categorical KL, entropy weight, minibatch/update budgets, gradient
clipping, learning rate and seed are JSON settings. Only the loaded LoRA leaves
are trainable. Inference and RL share the same differentiable completion scorer.

This is **not** full PPO with a critic/GAE, nor a claim of a faithful GRPO port.
Old-policy KL is not a fixed SFT-reference retention constraint. Long-horizon
credit assignment and economic lift remain to be tested. Outputs are immutable
adapter snapshots. Completed-group checkpoints include optimizer tensors,
NumPy/MLX RNG state, metrics and the next group index. Configure
`resume_checkpoint` to use one; incompatible source/policy/learning contracts
are rejected. Interrupted groups are replayed from the last completed group,
not resumed in the middle of a market episode. A real uninterrupted-versus-resumed
parity test is written but unexecuted. Normal evaluation is deterministic and
performs no gradient updates or promotion.

`checkpoint_keep` bounds intermediate snapshots. Cleanup only removes complete,
verified checkpoints of the same contract, protects an explicitly resumed parent,
and skips directories containing other files. Final adapters are not pruned.

## Volume interchange

Configure `volume_source` with `manifest` and `audit` paths and select the desired
`volume.<channel>` names in the context JSON. Channels and numerical bounds come
from that manifest, not assumptions about the unfinished Volume training job.
The interchange schema is `reasoning_specialist_cache_v1`, with:

- `kind`, `channels`, per-channel `bounds`, and completed-bar UTC-nanosecond semantics;
- per-ticker shards containing read-only `.npy` timestamps and value matrices;
- each shard's file digests, frozen `model_identity`, and `fit_end_ns`;
- a separate reviewed audit bound to the manifest digest.

Each shard must be fitted before its first scored row; multiple shards support
expanding-window OOF data. Coverage must match every requested market timestamp.
No missing value is fabricated. This loader does not export scores from the
unfinished sibling model. Its eventual export still needs verified channel and
artifact mapping; no sibling code is modified by this implementation.

## Explicit staged execution and screening

`python -m propevolve.reasoning_policy.workflow --config config/reasoning/workflow.json`
is an explicit future launch command, not an import-time or training-completion
listener. It has not been run. The initial plan begins with verification of an
already generated and reviewed market dataset, then market SFT, action-dataset
audit, action SFT, RL, and evaluation in separate
processes. Missing data/audits stop the flow; it never authors its own PASS audit.
Collection can be declared as a prior stage, but source and dataset reviews remain
required. Each stage declares its inputs, output receipts, log and timeout.
Completed stages are reused only when those recorded files match. A failed stage
remains BLOCKED; partial model outputs are never silently accepted or overwritten.
SFT native weight warm-start is not exact optimizer resume. The workflow does not
automatically recover a half-written SFT adapter; use a new declared output and
an explicitly reviewed warm-start if needed.

Evaluation records action counts, headroom, positive timeouts, pass/blow/timeout,
and decision logs. `evaluation_metrics.json` declares the near-blow definition and
screening limits. `REVIEW_CANDIDATE` is not promotion or sealed confirmation.
The initial 60%/zero-blow/10%-near-blow criteria are a research target, not evidence.
Teacher-dependent candidates fail a teacher-free requirement rather than being
misreported. Automatic live promotion is deliberately absent.

Evaluation inherits the SFT model/template JSON via `inherits`, overriding only
the adapter. Inheritance is filename-independent and cycles fail. Changing the
base model still requires compatible adapters and renewed parity evidence.

CPU reference/simulator tests and explicit-opt-in real MLX update/reload tests
are written. They have not been executed in this implementation session.

## Remaining before launch acceptance

- Resolve the source recipe and audited temporal roles from teacher manifests.
  Same source data does not mean in-sample teacher predictions are OOF inputs.
- Keep 2026 sealed; all development labels must resolve before that boundary.
- Fill reviewed source/continuation identities and explicit episode starts in
  the job JSON. Null placeholders are intentional blockers, not implicit defaults.
- Exercise all new tests and the existing environment/replay/campaign regression.
- Benchmark real tokenizer/model SFT and reload on the target 16 GB machine.
- Verify natural-frequency Long/Short/failure learning and chronological economics.
- Verify the RL draft with the real adapter, including update direction, frozen
  base, legal actions and save/reload parity, before any long RL campaign.
- Bind the completed Volume export to the interchange contract and test its
  alignment. The source model/export is not yet authenticated or available here.
- Implement and verify the selected teacher-free input path before claiming a
  replacement. The continuous-projector interface has been proposed for approval;
  the current specialist-conditioned path does not silently pretend to be it.

The first format uses named numeric specialist/account context, not thousands of
serialized FFM latent coordinates. Generated reasoning traces, continuous market
token projection, distillation to teacher-free inputs, and campaign promotion are
not claimed by this implementation checkpoint. The Stanford course note informs
the verified-feedback/offline-improvement workflow, not a guaranteed trading edge
or an automatically validated optimizer.

## Conditional retirement of R2D2

Do not delete the incumbent because this branch builds or its tests pass. Keep
the accepted checkpoint and a reproducible baseline until matched chronological
evaluation establishes challenger economic lift without unacceptable blow or
near-blow regression. Then archive the accepted baseline's recipe and evidence,
identify callers of C51-only learner/replay/teacher-loss code, and remove only
those obsolete paths in a separate reviewed change. Preserve the shared market
data, simulator, execution, prop-risk, label and evaluation modules. No such
retirement or old-run deletion is performed by the challenger implementation.
# Shared policy selection

`propevolve.policy.TradingPolicy` is the shared deterministic decision interface.
`R2D2Policy` and `ReasoningPolicy` inherit it and expose `reset()` and
`decide(PolicyInput)`. The R2D2 adapter owns recurrent state and its configured
reset horizon. The reasoning adapter consumes an episode-local rolling context.
Both return a legal action and scores tagged as Q-values or log likelihoods;
these scores are not equivalent or calibrated pass probabilities.

Set a job's `policy_config` to an arbitrary JSON path. For reasoning:

```json
{"kind": "reasoning", "model_config": "config/reasoning/evaluation.json"}
```

For R2D2, supply the actual existing checkpoint and its recurrent horizon:

```json
{"kind": "r2d2", "checkpoint": "runs/your-checkpoint.pt",
 "device": "cpu", "learner_backend": "pytorch", "recurrent_horizon": 150}
```

Paths resolve against the job's workspace root. The numbers/paths above are
examples, not Python defaults. `config/reasoning/shared_evaluation_job.json`
demonstrates selection while inheriting the existing job. Existing recipes
without `policy_config` retain their prior evaluation loading path.

Both adapters use the same challenge evaluator. R2D2 skips specialist context
construction; the current reasoning adapter still requires specialist inputs
and is explicitly reported as **not teacher-free**. This interface does not
implement the pending teacher-free embedding projector. C51 and LoRA/RL learning
algorithms remain separate, and the existing R2D2 campaign is unchanged.

Interface/E2E tests have been authored; execution is deferred while Volume
training is active. No runtime compatibility or learning result is claimed yet.

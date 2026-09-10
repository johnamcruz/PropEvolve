# Reasoning trading policy

Teacher-free reasoning-model policy built on the unchanged shared challenge
environment. It is not economically validated until temporal SFT and RL gates
pass.

## Current path

1. A market-distillation stage teaches Expansion, Trend and Regime semantics
   from aligned training-only targets. The deployed policy receives frozen FFM
   embeddings plus normalized account/trade state, never teacher outputs.
2. `context.RollingContext` supplies completed-bar history with an availability
   mask. `config/reasoning/context.json` starts with 20 bars; length is configurable.
3. Flat-state SFT labels rank WAIT/Long/Short using the configured minimum
   target-before-stop contract (2R before -1R by default), including next-open
   execution, costs and adverse-first OHLC ambiguity. Achieved 3R/4R levels add
   value but are not forced exits.
4. Positioned SFT labels rank HOLD/CLOSE from causal MFE, MAE, current R and
   giveback state. HOLD remains correct while meaningful additional favorable
   excursion is reachable; CLOSE becomes correct on deterioration. Pass/blow
   rewards do not define these trade-mastery labels.
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
   are log likelihoods, not action values. Evaluation reports five-action trade
   mastery separately from existing simulator challenge outcomes and explicitly
   reports specialist dependence.

## Responsibility boundary

### Development-data inference assessment

Before selecting a targeted trade-training subset, assess a frozen adapter on
an audited, prepared development pool:

```sh
python scripts/assess_reasoning_trade.py --config CONFIG.json --view PREPARED_VIEW --role train --output NEW_ASSESSMENT_DIRECTORY --root .
```

This uses the production batched evaluator and resolved embedding references.
It writes indexed `scores.jsonl` with action targets, prediction margins, ticker,
timestamps and available specialist metadata, plus `summary.json` with separate
entry/direction/management metrics for hierarchical policies and per-ticker
results. It does not update weights. Source targets and prepared targets must
match; sampled prepared views are rejected to prevent incorrect row joins.

Use training/development mistakes for future subset selection, alongside diverse
correct examples. Validation used to select training is development evidence;
never relabel it as untouched evaluation. A completed assessment does not prove
trade mastery or economic generalization.

Action SFT may consume a completed assessment through `targeted_sampling` in
JSON. The setting pins the assessment directory plus the SHA-256 identities of
`scores.jsonl` and `summary.json`, a per-ticker/action/year row budget, the
mistake fraction and a seed. Training fails closed unless the scores cover the
entire prepared training role and the summary names the same prepared-view
manifest. Each round rotates unresolved rows while retaining already-correct
rows, then balances all legal action classes. Validation remains the complete,
fixed chronological selection role; its errors never enter this sampler.

### Corrective trade-mastery campaign

The corrective reasoning workflow is a resumable sequence of frozen assessment
and short SFT rounds:

```sh
python -m propevolve.reasoning_policy.corrective_campaign \
  --config config/reasoning/corrective_trade_mastery_campaign.json
```

Each round assesses the accepted parent over the complete development role,
selects unresolved mistakes plus previously mastered anchors with balanced
action/ticker/year mass, fine-tunes a new immutable candidate, and reassesses
the parent and candidate over the same fixed chronological validation rows. A
candidate becomes the next parent only when mistake margins improve while every
WAIT/Long/Short/HOLD/CLOSE boundary retains its mastered rows and no task
regresses beyond the configured tolerance. A rejected candidate never becomes
the parent.

The state file receipts every assessment and adapter artifact by SHA-256. An
interrupted run resumes the incomplete round and skips only authenticated
completed phases. Training selection uses only the development assessment;
validation rows are acceptance evidence and never corrective examples. The
campaign does not use 2025 or sealed 2026 and does not contain challenge rewards.
After five-action trade mastery passes,
the accepted SFT adapter is the mandatory parent for the separate reasoning RL
stage that learns pass/blow/near-blow economics.

SFT owns trade mastery: WAIT/Long/Short setup selection, entry timing, HOLD
through valid continuation, and CLOSE on weakening, reversal or deteriorating
economics. RL starts only from an audited five-action SFT adapter and owns
challenge mastery: maximize pass rate under the unchanged profit target, MLL,
costs, fills and 30-day timeout. Evaluation reports action-boundary evidence
separately from pass/blow/near-blow economics.

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
alter the core market embedding. MFE/MAE are gross price excursions;
account equity and economic labels retain the simulator's fees. Final future
excursions are NOT inference inputs. These inputs can support learning exits,
but their presence is not evidence that the model has learned profitable exits.

The config and excursion regressions cover Long/Short symmetry, future-price
isolation, flat reset, and model/adapter mismatch. The explicit real-MLX smoke
also covers five-action updates, unseen-row learning, and save/reload parity.

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
loaders and challenge configuration. Scratch trade-mastery collection uses
economic barrier and position-path labels. Dataset audit is a separate required stage:
collection never authors its own passing causality receipt.

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

Evaluation records action counts, headroom, positive timeouts, pass/blow/timeout,
and decision logs. `trade_mastery_metrics.json` independently declares action,
win-rate, expectancy, 3R-average-winner and MFE-capture gates;
`evaluation_metrics.json` declares challenge pass/blow/near-blow limits.
`REVIEW_CANDIDATE` is not promotion or sealed confirmation.
The initial 60%/zero-blow/10%-near-blow criteria are a research target, not evidence.
Teacher-dependent candidates fail a teacher-free requirement rather than being
misreported. Automatic live promotion is deliberately absent.

Evaluation inherits the SFT model/template JSON via `inherits`, overriding only
the adapter. Inheritance is filename-independent and cycles fail. Changing the
base model still requires compatible adapters and renewed parity evidence.

CPU reference/simulator tests, MLX numeric tests, and the explicit real-MLX
five-action update/reload smoke have been executed in this implementation session.

## Remaining before launch acceptance

- Require positive WAIT/Long/Short/HOLD/CLOSE margins after save/reload on the
  inner temporal split, then repeat on unseen 2025.
- Confirm action SFT preserves the preceding Expansion/Trend/Regime distillation.
- Run full 2021-2024 SFT with best-checkpoint restoration and no 2025 tuning.
- Verify bounded RL updates preserve trade mastery while improving challenge
  economics in the unchanged simulator.
- Require the declared pass, blow and near-blow gates on unseen 2025. Keep 2026
  sealed for the final frozen confirmation.

## Policy selection

`propevolve.policy.TradingPolicy` is the deterministic decision interface.
`ReasoningPolicy` consumes an episode-local rolling context and exposes
`reset()` and `decide(PolicyInput)`. It returns a legal action and sequence
log-likelihood scores; these are not calibrated pass probabilities.

Set a job's `policy_config` to an arbitrary JSON path. For reasoning:

```json
{"kind": "reasoning", "model_config": "config/reasoning/evaluation.json"}
```

Paths resolve against the job's workspace root. The numbers/paths above are
examples, not Python defaults. `config/reasoning/shared_evaluation_job.json`
demonstrates selection while inheriting the existing job. The reasoning adapter
uses the frozen-embedding projector and is teacher-free at inference;
Expansion/Trend/Regime targets are training-only.

Passing mechanics tests are necessary but not an economic claim. The policy
remains unpromoted until the temporal SFT and challenge-economics gates above pass.
## Optional direct market distillation

`market_distillation` replaces verbose teacher-answer JSON with fixed query
positions and independent soft-probability losses. It keeps the same causal FFM
embedding projector, Qwen backbone, LoRA optimizer, and teacher-free policy.
All declared specialist channels must match the dataset exactly. No extra model
or prediction head is introduced. A null setting preserves token SFT.

Use `config/reasoning/qwen3_0_6b_market_probability_sft.json` for this objective.
The coverage variant adds deterministic rotating ticker/year/teacher-context
sampling. Its bins control training coverage, not legal actions or entry gates.
An epoch with coverage enabled is one sampled round, not a complete pool pass.
Validation remains fixed and is not selected by training errors.

Market distillation teaches Expansion/Trend/Regime. Existing action SFT retains
entry, direction, HOLD/CLOSE and economic supervision; RL retains challenge
objectives. Better teacher Brier scores do not establish better trading.

For the bounded direct-market-to-trade path:

```sh
python -m propevolve.reasoning_policy.workflow \
  --config config/diagnostics/market_distillation_candidate_workflow.json
```

This runs direct market distillation followed by trade SFT through the existing
workflow; it does not run old-model comparisons. Stage logs and immutable receipts
are written under the configured output directory. `prepared_sampling` explicitly
selects a fixed diagnostic slice; leave it null for full-pool training. Original
datasets are not modified. Software tests and teacher fit are not trade-mastery
or promotion evidence. 2026 remains sealed.

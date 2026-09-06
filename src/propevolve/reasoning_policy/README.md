# Reasoning-policy challenger

Optional, specialist-conditioned trading policy. The existing C51 training and
environment remain unchanged. This is an implementation checkpoint, not a
trained or economically validated replacement.

## Current path

1. `inputs.specialist_account_fields` reads aligned Expansion, Trend and Regime
   scores plus the existing normalized account observation.
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
7. `mlx_sft` creates native MLX-LM prompt/completion datasets and delegates adapter
   training to MLX-LM. Inputs are loss-masked. Overlength samples fail rather than
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

When execution is authorized and the source dataset audit passes, the preparation
entry point is:

```sh
python -m propevolve.reasoning_policy.mlx_sft --config config/reasoning/sft.json --view runs/reasoning-challenger/mlx-view
```

This loads the configured model to check quantization and exact token lengths,
but does not train unless `--train` is also supplied. Run from the repository
root or provide explicit resolved paths in the JSON. Existing output paths are
never overwritten by the wrapper.

## Remaining before a runnable experiment

- Resolve the source recipe and audited temporal roles from teacher manifests.
  Same source data does not mean in-sample teacher predictions are OOF inputs.
- Keep 2026 sealed; all development labels must resolve before that boundary.
- Wire approved real-data loading/collection into a job recipe; the collector
  currently exposes a Python API, not an automatic whole-history export job.
- Exercise all new tests and the existing environment/replay/campaign regression.
- Benchmark real tokenizer/model SFT and reload on the target 16 GB machine.
- Verify natural-frequency Long/Short/failure learning and chronological economics.
- Add the separate environment-RL learner only after the supervised challenger
  is stable; PPO/GRPO is NOT implemented by this SFT checkpoint.
- Volume inputs are not connected. No volume probabilities are fabricated.

The first format uses named numeric specialist/account context, not thousands of
serialized FFM latent coordinates. Generated reasoning traces, continuous market
token projection, distillation to teacher-free inputs, and campaign promotion are
not claimed by this implementation checkpoint.

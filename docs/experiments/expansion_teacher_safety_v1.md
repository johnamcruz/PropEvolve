# Expansion-teacher safety experiment v1

## Decision

Test whether temporary causal Expansion supervision improves challenge pass rate
and realized winning-R over the Mask-only safety recipe while retaining zero
selection blows. The deployed policy receives only frozen Chronos Mask
embeddings and account state.

## Frozen comparison

- Universe: nine training markets at 3 minutes; NQ deployment evaluation.
- Training: 2021-01-01 through 2024-12-31.
- Development selection: 2025. This period has already been inspected and is
  not sealed confirmation evidence.
- Sealed confirmation: 2026; inaccessible during training, reasoning, and
  recipe selection.
- Parent: `historical_mask_safety_replay_v1` economics, risk, agent, seed,
  environment-step budget, and evaluation code.
- Treatment: four soft Expansion probabilities supervise a temporary head on
  the shared recurrent representation during training only.
- Teacher: authenticated nine-market 3-minute Expansion checkpoint in
  `teachers/manifest.json`; targets end strictly before 2025.

## Gates

Primary feasibility is `selection.blow_rate == 0`. Among feasible candidates,
compare pass rate, timeout rate, mean terminal P&L, trade win rate, and average
winning-R against the matched Mask-only parent. Report Long/Short and temporal
slices and repeat across declared seeds before promotion. Recovery checkpoints
are never selected by their 2025 result; they only resume the fixed training
budget.

## Falsifier

REVISE or reject the teacher treatment if it introduces any selection blow,
fails to improve pass completion consistently across seeds, depends on teacher
inputs during greedy evaluation, or changes the frozen data/economic contract.
The 2026 period remains untouched until the complete recipe is frozen.

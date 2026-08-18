# Stage 2A v19 paired A+ contrastive learning

## Status

`READY FOR ONE MATCHED 100/200 RUN`

V18 remains the matched baseline. It completed 100 training episodes and
short-circuited before validation on its frozen teacher-free policy-health
gate, so v19 may now proceed without changing v18 evidence.

## Identity

- Experiment: `stage2a_v19_paired_aplus_contrastive`
- Process: `ML-RIGOR-v1`
- Repository owner: PropEvolve
- Opened: 2026-08-17
- Status: exploration/inner selection
- Parent lineage: the same immutable Stage 1 parent used by v18
- Baseline: completed v18 recipe and evidence
- V18 recipe SHA256: `b9066ab699740c71b96390eccc9e7b634b4fdb0f39e3f3fbd376b7c7e4693231`
- V18 terminal evaluation SHA256: `19b11c3ff37579fa27df74f359a2fc022b94cbb3c956445478d3e1db4d41be6a`
- Sealed period: 2026 remains untouched

## Decision and hypothesis

The decision remains teacher-free `WAIT`, `ENTER_LONG_1`, or
`ENTER_SHORT_1`. V19 changes only how the policy learns to rank an exact A+
entry against a similar failed entry.

The current v18 loss teaches exact actions and compares mean valid-versus-failed
cohorts. A mean comparison can hide hard examples. V19 adds direct pairwise
pressure so every present matched failed/valid cohort must rank the exact A+
entry above the failure.

For one side:

```text
delta_good = Q(correct side) - Q(WAIT)
delta_bad  = Q(failed side)  - Q(WAIT)

loss = softplus(pair_margin + delta_bad - delta_good)
```

Falsifiable claim: on the unchanged 2025 teacher-free selection, v19 should
improve exact Entry precision and pass conversion over v18, especially for
non-chop Short entries, while preserving opportunity recall, both sides,
dominant-chop avoidance, zero blows, and winner quality.

Stop if v19 creates a validation blow, universal WAIT, Long/Short collapse,
lower pass conversion, worse near-blow incidence, or no Q-gap/economic-quality
separation relative to v18.

## Causal and label contract

No new data or label is introduced. Reuse the authenticated pre-2025 exact
Entry target:

- completed decision bar and next-bar-open execution;
- one directional launch;
- continuation truth;
- fee-inclusive `$300` risk;
- `+2R` before `-1R` within 150 bars;
- stop-first same-bar collision;
- resolved label before the temporal boundary.

An A+ positive is an exact `ENTER_LONG_1` or `ENTER_SHORT_1` row. A failed
negative is an exact `WAIT` row with side-specific failed-confluence membership.
Episode outcome is never the label: not every trade in a pass is A+, and not
every trade in a timeout is bad.

Expansion and Regime channels may construct training-only membership and match
groups. They remain absent from policy observations, validation, and inference.

## Pairing contract

Form pairs only inside the already sampled recurrent batch after burn-in. Do
not change replay composition in v19.

Pairs match exactly on declared side: Long with Long, Short with Short.
Three-state Regime probabilities contribute continuous pair similarity weight,
so near-boundary market states remain comparable without an `argmax` class
boundary or probability cutoff.

No fixed account-headroom threshold defines or groups an A+ setup. Normalized
account state remains a causal policy input learned continuously by RL, but the
paired A+ target is strictly market-setup quality plus exact economics.

Continuous valid/failed confluence membership supplies the pair weight. Within
each present matched group, compute the weighted mean of every valid/failed
pair loss. Average Long and Short equally when present; within each side,
aggregate Regime contributions by their continuous similarity mass. Missing
pair mass contributes no fabricated target and no loss. Use
`pair_margin=0.25`; keep the existing Regime-selectivity aggregate
loss weight unchanged.

## Frozen matched boundary

V19 keeps v18's following identities unchanged:

- immutable Stage 1 warm-start parent;
- 2021-2024 training and 2025 teacher-free selection roles;
- Expansion and three-state Regime caches;
- exact Entry targets and class balance;
- replay schedule and recurrent burn-in;
- agent architecture, C51 support, optimizer, learning rate, seed, and budget;
- existing exact-action, chop, failed-confluence, and winner-management losses;
- one-contract risk, fees, fills, stop, MLL, and evaluator;
- 100 training episodes and 200 teacher-free selection episodes.

Only the paired A+ rank loss, its declared margin, diagnostics, semantics
version, experiment identity, and tests may differ.

## Confirmed public test seams

1. A pure public `paired_a_plus_rank_loss` seam validates the literal ranking
   equation, deterministic match groups, weights, and absent-pair behavior.
2. `RecurrentC51Agent.train_batch` proves the loss is wired into the executable
   recurrent learner, survives teacher-imitation dropout, and emits additive
   Long/Short pair counts and loss evidence.
3. The existing public teacher-free evaluator proves zero teacher lookups and
   unchanged policy input/output behavior.

Required TDD cases:

- a valid Long outranks a matched failed Long;
- a valid Short outranks a matched failed Short;
- Long never pairs with Short;
- near-boundary Regime probabilities weight pairs continuously without a cutoff;
- a failed non-chop Short is a negative, never a positive;
- dominant-chop exact WAIT remains negative supervision;
- absent positive or negative cohorts produce zero pairs and zero loss;
- teacher dropout removes imitation only, not exact or paired supervision;
- recurrent burn-in rows are never treated as learning pairs;
- teacher-free evaluation performs no teacher lookup.

## Diagnostics and acceptance

Training diagnostics must report pair count, effective pair mass, loss, and mean
good/bad Q advantage separately for Long, Short, and Regime group.

These training diagnostics prove mechanism activity, not promotion. If v19
reaches the 200-episode teacher-free selection, run the existing read-only
selectivity analysis before accepting or revising the candidate. Do not build
that post-run comparison if the frozen training-health gate already stops v19.

Teacher-free selection must report:

- exact Entry precision and opportunity recall by side;
- Q-advantage quantiles against `+2R-before-1R` truth;
- dominant-chop versus non-chop entry rates;
- pass, timeout, blow, and terminal near-blow rates;
- expectancy, average winner R, MFE capture, and trade frequency;
- per-market, side, and Regime slices.

Proceed only when v19 has zero validation blows, both sides remain active, its
Q-advantage ranking is economically monotonic, and it improves the first failed
v18 boundary without materially regressing pass rate, opportunity recall,
winner R, or dominant-chop avoidance. Otherwise preserve v18/Stage 1 evidence
and stop or revise exactly the first failed boundary.

## Deferred work

- No new A+/B/D taxonomy or manually weighted score.
- No dedicated A+ model head unless native Q advantage fails calibration.
- No replay oversampling, prioritized replay, PCGrad, extra episodes, margin
  sweep, Stage 2B recovery, or Trend teacher in this experiment.
- No production threshold or inference gate.

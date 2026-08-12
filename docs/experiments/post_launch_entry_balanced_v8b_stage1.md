# Balanced post-launch Entry Stage 1 (v8b)

## Status

`post-launch-entry-balanced-v8b-r1` is an active research run. This record
freezes the evidence observed through teacher-free chronological selection
episode 171 of 200. It is not a final selection decision, sealed confirmation,
or production promotion.

The exact training recovery checkpoint is:

`runs/historical_mask_expansion_regime_post_launch_entry_balanced_v8b/campaign-runs/post-launch-entry-balanced-v8b-r1/safety_foundation_1m/attempt-2/training-recovery.pt`

The frozen recipe is:

`config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json`

The implementation was committed as `c7794e4` (`Balance post-launch entry
supervision`) after 279 tests passed.

## Learning problem and repair

The prior post-launch Entry run had authenticated bar-1-through-bar-5 action
targets, but the target population was imbalanced: 394,793 WAIT, 40,353 Long,
and 39,001 Short examples. Unweighted action loss learned the majority class
and produced a universal-WAIT greedy policy. Training trades and passes were
then supplied largely by exploration and disappeared when selection set
epsilon to zero.

v8b changed only the Entry action-class balance and fresh run identity. It
derives inverse-frequency weights from the authenticated pre-2025 target
manifest using `N / (3 * class_count)` in fixed WAIT, Long, Short order:

- WAIT: `0.40033384583819875`
- Long: `3.9166604713404207`
- Short: `4.052434552960181`

The receipt binds the action order, exact counts, derived weights, and source
manifest identity. Deterministic tests prove that one agent can learn all five
WAIT contexts plus both Long and Short contexts, retain those rankings after
teacher removal and serialization, and fail closed on receipt or recovery
drift. A fixed WAIT weight such as 0.25 was rejected because weighted-mean
cross entropy depends on class ratios and the authenticated formula corrects
the observed imbalance without guessing.

Attempt 1 then localized a separate management failure. It reached 506,754
steps with zero passes and zero blows, but held only 2.60 bars on average and
closed winners prematurely. The campaign reasoning step made one bounded
revision: more management exploration near the learning boundary and a modest
large-winner credit. The revised attempt completed the full one-million-step
training budget.

## Training evidence

Attempt 2 completed 1,000,046 environment steps across 85 episodes:

- 14 passes, 0 blowouts, and 71 timeouts;
- 43.45% trade win rate and 0.734R average winner;
- 21.24-bar average hold, versus 2.60 bars at the failed boundary;
- 8.85% ratchet activation and 80.48% capture on trades reaching 2R MFE;
- positive learned HOLD-minus-CLOSE value gap of 0.0798;
- 11,375 trades with both Long and Short actions represented;
- no training short circuit.

Training fit is diagnostic only. It does not establish autonomous or temporal
generalization.

## Teacher-free chronological selection evidence

Selection uses chronological 2025 NQ episodes with epsilon zero. Expansion,
Regime, Entry action labels, retention anchors, and training-only auxiliary
heads are unavailable to the evaluated policy. The policy therefore acts only
from frozen FFM embeddings and its causal account and position-management
state.

At the frozen episode-171 snapshot:

- 42 passes, 0 blowouts, and 129 timeouts;
- observed pass rate: 24.56%;
- the policy traded autonomously rather than collapsing to universal WAIT;
- multiple passing episodes retained average winners above 2R, including
  2.338R, 2.762R, and 2.665R examples.

This is the first credible end-to-end evidence that the training-only teachers
and balanced post-launch labels transferred useful Entry and management
behavior into a teacher-free greedy policy. It does not isolate the individual
causal contribution of Expansion, Regime, class balance, or the management
revision; matched ablations are still required for attribution.

## What remains unresolved

The candidate is safe so far but not economically complete. At episode 171,
129 selections had timed out, including many outcomes near the loss floor.
Pass frequency, average terminal PnL, near-blow timeout rate, and stability over
the complete 200 episodes remain the current decision boundary. The aggregate
selection average winner and expectancy must come from the completed evaluator
receipt rather than selected successful episode examples.

The result also does not prove 2026 performance, cross-market OOS performance,
or production parity. The 2026 period remains sealed.

## Retained decisions

The complete intended progression is documented in
`docs/experiments/propevolve_teacher_curriculum_stages.md`.

1. Preserve this exact candidate and completed evaluation receipt as the Stage
   1 research baseline, even if a later gate requests revision.
2. Do not change its ratchet or winner-retention mechanics while diagnosing
   entry selectivity; those mechanics produced the improvement we want to
   retain.
3. The next matched curriculum stage should warm-start from the authenticated
   Stage 1 candidate and teach WAIT during high-confidence chop using temporary
   soft Regime supervision.
4. Keep Regime guidance probabilistic, decayed, randomly hidden, and absent
   from selection and deployment. Profitability remains authoritative.
5. Add temporary Long/Short Trend-readiness supervision only after the isolated
   chop-abstention stage is evaluated. Do not bundle the two changes and lose
   attribution.
6. Require teacher-free zero blowouts, reduced near-blow timeouts, nonzero pass
   rate, retained winner R, and positive expectancy before advancing.
7. Use one-million steps for mechanism screening, two million for confirmation,
   and five million with multiple seeds only after selecting a credible recipe.

## Finalization condition

When all 200 selection episodes finish, archive the exact candidate, complete
evaluation receipt, recipe, code identity, training diagnostic summary, and
checkpoint hash. If it passes the declared Stage 1 gates, register it as the
Stage 1 parent. If a gate fails, retain it as an immutable promising research
candidate and revise only the first failed boundary.

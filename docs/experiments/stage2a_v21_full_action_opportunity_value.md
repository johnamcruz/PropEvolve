# Stage 2A v21 full-action opportunity-value supervisor

## Identity and status

- Process: `ML-RIGOR-v1`
- Repository: PropEvolve
- Opened: 2026-08-18
- Status: `REJECTED / NON-EXECUTABLE — production code and recipe reverted`
- Parent experiment: Stage 2A v20 generic canonical A+
- Baseline code: `7aeb7a9bdb00a1f0fd0b44a069dd3df06d30af08`
- Baseline config SHA-256:
  `96dcb2fb70120d0363d60016199cea167b9d619d1d77bc19e9a456a5b3fcf6a5`

## Decision and hypothesis

The changed decision is the flat-state ranking among `WAIT`,
`ENTER_LONG_1`, and `ENTER_SHORT_1` on an authenticated Expansion-entry row.
The policy continues to observe only causal frozen market-context embeddings
plus normalized account and execution state. Expansion, Regime, exact Entry
outcomes, and the new opportunity-value vector are training labels only.

Hypothesis: on the same rows, seed, recurrent replay, optimizer, episode
budget, costs, risk, and teacher-free selection period as v20, supervising all
three action values on each eligible row will improve teacher-free Entry
precision and pass conversion without a blow, near-blow regression, universal
WAIT, or Long/Short collapse.

Reject the mechanism if it improves supervised agreement but not teacher-free
economics.

## R1 failure and bounded correction

R1 was stopped at training episode 37. Through episode 35 it had five passes,
zero blows, and thirteen terminal near-blow timeouts. Short recall improved,
but the latest eight-episode learner sample predicted Entry on 154 exact-WAIT
dominant-chop rows, including 121 false Shorts versus 33 false Longs. This was
not epsilon exploration: the diagnostic is reconstructed from greedy C51
confusion counts.

- R1 frozen config SHA-256:
  `403fd53e5de9a4fd1d3cd2b2e8ef969c7c8ec771b3aa6cba4a782168ec599c55`
- R2 corrected config SHA-256:
  `e73741140ad2bf72b60711f47632d75996a7e48097bfa1e54eb132a0d388d59a`

The first failed boundary was a training-objective conflict. On dominant-chop
rows with a winning forced side, the raw opportunity vector taught Entry while
the existing all-dominant-chop margin taught WAIT. R2 remains v21 and changes
only how the new opportunity KL consumes the already-authenticated continuous
Regime evidence:

```text
m = clamp(chop - max(transition, expansion_trend), 0, 1)
p_economic = softmax([T_WAIT, T_LONG, T_SHORT] / temperature)
p_training = (1 - m) * p_economic + m * [1, 0, 0]
```

At `m=0`, Long/Short economic supervision is unchanged. As chop dominance
increases, the training target moves continuously toward WAIT for both sides.
This is a training-only soft target, not an inference feature, threshold, hard
gate, relabeling of exact Entry truth, or change to teacher-free evaluation.
The config freezes this as
`post_launch_entry_opportunity_value_v2` with
`regime_conditioned_teacher_to_policy_kl_v2`.

R2 then failed the matched economic gate. Through episode 78 it produced seven
passes, seventy timeouts, one blow, and twenty-four terminal near-blow
timeouts. The retained e7/v19 proportional run produced eighteen passes and
zero blows over its matched first 78 episodes. Recent R2 dominant-chop exact
WAIT errors remained elevated and asymmetric: 211 false Shorts versus 39
false Longs over the final twenty analyzed episodes. Because the objective
failed to preserve the accepted A+ competence, both the v21 executable recipe
and the full-action supervision production path were reverted. This document
and the immutable run evidence remain diagnostic history only.

## Executable target contract

Reuse the existing `post_launch_entry_v1` engine without changing its timing:

```text
completed bar t -> action decision -> fill at open[t+1]
                 -> three-bar +0.5R-before-0.25R continuation
                 -> +2R-before-1R within 150 bars
```

Fees, one-contract `$300` risk, and adverse/stop-first same-bar ordering remain
unchanged. For every available candidate row, independently evaluate both
directions with that same continuation and economic definition:

```text
T_WAIT  =  0
T_LONG  = +2 if Long continuation and +2R-before-1R both succeed, else -1
T_SHORT = +2 if Short continuation and +2R-before-1R both succeed, else -1
```

The vector is a relative training preference, not a Bellman target in dollars.
If both directions qualify, both may be positive. If neither qualifies, WAIT
is best. Ambiguous overlapping launch events and unresolved temporal-boundary
rows remain unavailable. The existing categorical exact-action target is not
redefined.

## One-change training experiment

Add one KL loss from the policy's centered flat-action C51 expectations to the
softmax of the opportunity-value vector. Freeze temperature and loss weight
before a campaign; do not sweep them in this experiment. C51 TD, exact-action
classification and margin, paired A+, recurrent sequences and burn-in, replay
sampling, observations, network, optimizer, and risk logic remain unchanged.

The target vector is absent from observations, saved policy inference, and
teacher-free validation. Entry-supervision curriculum scaling may scale this
auxiliary loss, but Expansion/Regime imitation dropout must not remove it.

## TDD and smoke gate

Before a campaign, public tests must prove:

1. Long and Short values independently match literal path fixtures.
2. Failed sides remain below WAIT and a winning side remains above WAIT.
3. Both-side winners are represented without an arbitrary one-hot choice.
4. Next-open fills, fees, stop-first collisions, temporal censoring, and target
   manifest identity remain authenticated.
5. Replay preserves the vector through sampling and checkpoint round trips.
6. The KL loss decreases when policy ranking moves toward the teacher vector.
7. Teacher-imitation dropout does not remove the auxiliary supervision.
8. Teacher-free evaluation performs zero opportunity-value lookups.

Only after the tests and a bounded label/loss smoke pass may one clean matched
campaign run. The 2026 sealed period remains unopened.

## Matched acceptance

- zero teacher-free validation blows;
- no worse path-wise near-blow incidence than the matched v20 baseline;
- higher pass rate or pass conversion;
- improved Entry precision without unacceptable opportunity-recall loss;
- nonzero Long and Short participation with per-side precision and recall;
- monotonic teacher-free action advantage by held-out opportunity-value bin;
- zero target lookup during validation and unchanged observation schema.

## Explicitly out of scope

- EarnHFT's crypto inventory architecture or full historical-path DP;
- a 30-day perfect-information challenge oracle;
- PPO, multi-agent, replay, network, or risk-framework replacement;
- hard Expansion or Regime gates;
- Stage 2B drawdown/recovery changes;
- additional auxiliary objectives or hyperparameter search.

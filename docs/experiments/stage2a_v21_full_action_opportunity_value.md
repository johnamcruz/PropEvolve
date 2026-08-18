# Stage 2A v21 full-action opportunity-value supervisor

## Identity and status

- Process: `ML-RIGOR-v1`
- Repository: PropEvolve
- Opened: 2026-08-18
- Status: `exploration`
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

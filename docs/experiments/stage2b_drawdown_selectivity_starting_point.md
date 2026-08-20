# Stage 2B drawdown-selectivity starting point

## Status

This is a research starting point, not an authorized training recipe. Stage 2B
begins only after a Stage 2A checkpoint passes its frozen teacher-free selection
gate. The active Stage 2A run is unchanged.

The source comparison was performed against:

- algoTraderAI commit `7048522f6ed93849d19ef77726d7c1dc94788aeb`;
- PropEvolve commit `05d72188410621f95086c6f6b7522b51c27e3398`.

## Reusable lesson from algoTraderAI

algoTraderAI combines learned account-state conditioning with a deterministic
risk envelope:

1. The policy observes balance, drawdown, MLL headroom, session PnL, remaining
   time, and progress toward the pass target.
2. Recovery training samples both healthy and drawdown starting cushions so one
   policy experiences the transition from ordinary participation to recovery.
3. When headroom is thin, the environment permits only very high-quality
   entries, caps size, and bounds stop risk.
4. When cushion is restored, the policy can participate normally and protected
   trend amplification may resume.
5. Diagnostics separately measure whether the policy itself requests A+ entries
   more often than sub-A+ entries in deficit. A policy is not credited merely
   because a hard gate blocked its bad requests.

Relevant algoTraderAI seams are `rl/environments/prop_firm.py` (recovery-start
sampling, account observation, deficit selectivity, size/risk bounds),
`configs/sweep/combine_v11_pivot.json` (frozen recovery and protected-add
recipe), and `scripts/diagnose_rl.py` (A+ deficit-entry lift).

## PropEvolve mapping

PropEvolve already has the required architecture:

- `ObservationAssembler` exposes normalized realized/equity PnL, peak equity,
  MLL headroom, drawdown, position state, and remaining challenge time;
- `ActionMasker` preserves the one-contract minimum-headroom safety boundary;
- `RecoveryCurriculumSettings` provides a frozen recovery start and a bounded
  one-entry recovery permit;
- the recurrent policy remains the final WAIT/Long/Short decision-maker;
- Expansion and Regime supervision remains training-only and is absent from
  validation and inference.

The desired learned behavior is state-dependent:

- **Healthy headroom:** participate normally in valid Expansion + Regime
  opportunities and retain winners.
- **Moderate drawdown:** demand stronger learned confluence and reject marginal
  entries.
- **Near MLL:** request an entry only for the strongest learned A+ opportunity;
  otherwise WAIT.
- **All states:** keep one-contract sizing and the deterministic stop/MLL safety
  envelope. Higher aggression means opportunity participation and winner
  retention, not larger size or weaker entries.

Do not copy algoTraderAI's inference-time Pivot probability threshold into
PropEvolve. PropEvolve must remain teacher-free. Any Expansion/Regime signal may
shape training targets or margins, but the serialized policy must reproduce the
behavior from native market and account state alone.

## Smallest Stage 2B experiment

Warm-start from the immutable selected Stage 2A checkpoint. Preserve its model,
selection losses, replay, risk, execution, and ordinary challenge evaluation.
Change only the declared recovery curriculum and its state-conditioned learning
pressure.

The existing exact `$300`-headroom recovery snapshot remains the smallest first
matched experiment. A broader headroom curriculum modeled on algoTraderAI's
`$500-$3,000` sampling is a later ablation only if the exact-start experiment
shows brittle emergency-only behavior. Do not bundle both choices into the
first run.

## Required proof

Report teacher-free policy requests before action masking, separately from
executed actions, by:

- MLL-headroom bucket: low, middle, and healthy;
- Regime: dominant chop, transition, and expansion/trend;
- Expansion strength and Long/Short side;
- pass, timeout, recovery-success, survived-not-recovered, near-blow, and blow
  outcome.

The Stage 2B child must demonstrate:

- zero ordinary-selection and recovery-stress blows;
- lower near-blow incidence than its frozen Stage 2A parent;
- positive A+ entry lift over sub-A+ opportunities at low headroom in the
  policy's unmasked requests;
- increasing WAIT selectivity as headroom falls without universal WAIT;
- preserved Long/Short participation, Stage 2A pass rate, dominant-chop
  avoidance, expectancy, and winner R;
- no teacher lookup during validation or inference;
- no relaxation of one-contract, stop, fee, MLL, or execution rules.

Failure preserves the Stage 2A checkpoint unchanged. Recovery improvement that
depends only on a hard gate, increases ordinary near-blows, or destroys pass
conversion does not qualify.

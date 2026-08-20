# Stage 2B drawdown-selectivity starting point

## Status

This is a research starting point, not an authorized training recipe. Stage 2B
begins only after a Stage 2A checkpoint passes its frozen teacher-free selection
gate. The active Stage 2A run is unchanged.

## PropEvolve recovery premise

One recurrent policy observes causal market context together with realized PnL,
equity, drawdown, MLL headroom, session PnL, remaining time, and pass progress.
Recovery training must teach that same policy to rank WAIT, Long, and Short
economically in drawdown. Diagnostics must measure the policy's requested
actions directly so blocked or adjusted executions are never misreported as
learned selectivity.

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

PropEvolve must remain teacher-free. Any Expansion/Regime signal may shape
training targets or margins, but the serialized policy must reproduce the
behavior from native market and account state alone.

## Smallest Stage 2B experiment

Warm-start from the immutable selected Stage 2A checkpoint. Preserve its model,
selection losses, replay, risk, execution, and ordinary challenge evaluation.
Change only the declared recovery curriculum and its state-conditioned learning
pressure.

Use one frozen recovery-start contract per matched experiment. A broader
headroom curriculum is a later ablation only if the first exact-start experiment
shows brittle emergency-only behavior. Do not bundle both choices into one run.

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

# Recurrent value-collapse diagnosis (v8d-v8e)

## Frozen boundary

The compared runs used the same causal nine-market 2021-2024 training universe,
chronological 2025 NQ selection period, sealed 2026 period, Challenge economics,
FFM Mask representation, Expansion and Regime teachers, and recurrent C51
policy family. Teachers remained training-only and were excluded from temporal
selection.

## Evidence

The v8d policy demonstrated real training capability before collapsing. It
passed ES at episode 7 and ZB at episode 9, reaching approximately 121k steps
before the behavioral transition was visible. Episode 11 then timed out with
1,082 trades and only `0.192R` average winner size. Later episodes sustained
short holds, high voluntary-closing behavior, no additional passes, and weak
winner retention.

Batch diagnostics localized the first failed learning boundary. Sampled HOLD
transitions retained better immediate reward than CLOSE transitions, but the
learned HOLD-minus-CLOSE value gap changed from positive to negative around
episodes 9-11. The original recurrent TD update evaluated the successor state
from a reset hidden state, so current and successor values did not share the
same causal history. Correcting that defect was necessary but not sufficient:
one-step targets still propagated delayed winner-management value too slowly
relative to the immediately available flat action set after CLOSE.

The first attempted v8e/v8f restart was not a valid multi-step test. Although
its hypothesis declared an eight-step return, the serialized recipe omitted
`agent.n_step_return`, so configuration normalization silently selected the
backward-compatible one-step default. Episodes 1-15 reproduced v8d almost
exactly, including the Q-gap sign change and one-bar CLOSE churn. In every
updated episode, the reported n-step HOLD and CLOSE returns exactly equalled
their immediate rewards. This localized the repeated failure to experiment
activation rather than falsifying multi-step credit assignment.

The review also found that arbitrary replay slices did not reproduce the
behavior policy's periodic hidden-state resets. A deterministic test now proves
that observations preceding a recorded reset cannot affect post-reset learning.
Replay schema v2 stores both current- and successor-state reset lineage, while
the learner groups reset patterns without any device-to-CPU synchronization in
the update loop.

The authenticated episode-9 policy and episode-9-through-14 replay chronology
then reproduced the production failure exactly: the mean management Q gap was
`+0.051523`, `-0.042209`, `-0.073720`, `-0.074216`, `-0.070269`, and
`-0.074870`. Exact recurrent resets improved the final value only to
`-0.073961`, proving that hidden-state lineage was a correctness defect but not
the root cause.

The next boundary was replay and continual-policy retention. The sampler called
itself balanced while selecting `(ticker, outcome, side)` buckets uniformly;
nine timeout tickers plus one pass ticker therefore produced a 9:1 batch rather
than balancing the economic outcomes. Hierarchical outcome-first sampling fixes
that contract, but the authentic replay still ended at `-0.072843`. Removing
teacher gradients allowed eventual recovery to `+0.000870`; freezing the target
network improved the active-teacher result to `-0.007458`. Neither isolated
change prevented the intervening negative-Q behavior.

The demonstrated-pass competence anchor is the first repair that survives the
authentic chronology. After a pass, a frozen training-only copy of that policy
provides smooth value distillation only on management states sampled from pass
episodes. It preserves the demonstrated Q values rather than forcing HOLD.
With the original teachers and soft-target updates still active, the authentic
episode-9-through-14 Q gap remained bounded from `+0.080179` to `+0.081214`.
The anchor is checkpointed for exact recovery and discarded before temporal
selection and model packaging.

## Retained lessons and repair

1. Recurrent current-state and successor-state values must come from one
   contiguous causal GRU trace.
2. Delayed winner-management economics require a declared multi-step return;
   the one-step recipe is retained only as a matched control.
3. Collapse detection must run as soon as its evidence window is populated,
   independently of the later global pass/blow short-circuit boundary.
4. A policy that demonstrates a training pass is saved before subsequent replay
   updates as a rollback anchor, never as promotion evidence.
5. Exact restart requires the bounded replay population and sampler RNG in
   addition to policy, target network, optimizer, progress, and environment RNG.
6. Fine-tuning starts from an authenticated selected policy checkpoint while
   rebuilding optimizer, exploration, replay, and temporary teacher state for
   the new declared stage.
7. A named multi-step experiment must assert its effective serialized recipe;
   testing only the algorithm default does not prove the real run activated it.
8. Replayed recurrent learning needs a burn-in prefix so arbitrary sampled
   slices do not pretend their first learning state has zero causal history.
9. Replay recovery must accept the empty next-action mask on authentic terminal
   rows; the earlier validator incorrectly rejected every real terminal episode.
10. Periodic recovery is insufficient at a budget boundary. The final completed
    episode is now checkpointed even when it is outside the periodic interval.
11. Pass rollback checkpoints are immutable per episode. A latest-pass alias may
    advance, but a later pass cannot erase earlier pass-capable weights.
12. Outcome balance is hierarchical: economic outcome first, then ticker and
    side diversity. A wide timeout universe must not erase scarce pass evidence.
13. Training-only teacher gradients can amplify recurrent policy collapse even
    when their semantic losses decrease. Teacher-free ablations are required,
    but teacher removal alone is not a collapse repair.
14. A pass-capable policy may teach subsequent versions only on demonstrated
    pass states. Smooth value retention prevents forgetting without imposing a
    hard HOLD rule or becoming a validation-time dependency.
15. Permanent tests must include both a delayed-winner HOLD example and an
    economically superior CLOSE example, plus a chronological fixture where an
    unanchored control demonstrably collapses.

## Resume condition

The full unit and integration suite, deterministic collapse fixture, recovery
round trip, and authenticated replay falsifier must pass before a fresh run.
The old replay schema and collapsed recovery are intentionally not resumable.
Promote nothing until teacher-free chronological selection improves the declared
safety, winner-retention, pass-rate, and expectancy gates. Cross-market training
alone is not cross-market OOS evidence; NQ 2025 remains the current selection
claim and 2026 remains untouched.

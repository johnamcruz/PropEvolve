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

The v8e matched test added an eight-step recurrent return. It was stopped after
five episodes when the checkpoint-retention boundary was found, so its partial
one-pass result is not selection evidence and must not be interpreted as a
winner.

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

## Resume condition

Resume with a fresh run identity only after the full recovery round trip and
collapse rollback tests pass. Promote nothing until teacher-free chronological
selection improves the declared safety, winner-retention, pass-rate, and
expectancy gates. Cross-market training alone is not cross-market OOS evidence;
NQ 2025 remains the current selection claim and 2026 remains untouched.

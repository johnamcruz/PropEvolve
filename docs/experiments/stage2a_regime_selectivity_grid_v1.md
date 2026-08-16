# Stage 2A Regime-selectivity grid v1

## Decision

Test whether persistent-chop emphasis and teacher-guidance retention improve
teacher-free bad-trade avoidance over the immutable Stage 1 parent.

## Frozen comparison

The study compiles four cells from
`historical_mask_expansion_anchored_regime_stage2_v10.json`. It varies only:

- `regime_selectivity.persistent_chop_negative_emphasis`: `1.0`, `2.0`;
- `training.teacher_guidance_dropout_end`: `0.5`, `1.0`.

Every cell uses the same parent, seed, data and temporal roles, Expansion and
Regime artifacts, recurrent C51 architecture, replay, risk, costs, Entry
supervision, 100 training episodes, and 200 teacher-free selection episodes.
The period beginning in 2026 remains sealed.

## Compute and stop rules

All four configs are prepared before compute, but one MPS cell runs at a time.
The existing trainer may stop at episode 18 for its declared no-pass,
excess-blow, or collapse conditions. Its fixed policy-health and economic
futility evidence binds at episode 100. Selection stops after its first blow
or five consecutive no-trade episodes. A valid failed cell releases the slot;
an integrity or runtime blocker stops the queue.

## Acceptance and ranking

Only a cell that passes the existing Stage 2A selection and parent-retention
gates is eligible. Eligible cells rank lexicographically by zero/lower blow,
lower near-blow timeout rate, higher pass rate, higher expectancy, higher
average winner R, and higher two-R MFE capture. Only the winner may continue
to the existing 250- and 500-episode confirmation stages.

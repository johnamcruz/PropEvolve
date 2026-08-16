# Stage 2A constrained-TPE selection v1 — INVALID

## Decision

This study is retained only as reproducible failure evidence. It is invalid for
ranking because policy health evaluated one episode's Entry optimizer mass and
falsely pruned cumulatively balanced trials. Do not resume or use its SQLite
history for selection; use the separately identified v2 study.

## Frozen evidence boundary

Every trial started from the same authenticated Stage 1 parent and used the
same seed, 2021–2024 data roles, Expansion and three-state Regime artifacts,
recurrent C51 architecture, replay contract, costs, one-contract risk limits,
and 100-episode Stage 2A screen. Selection was greedy and teacher-free. The
period beginning in 2026 remained sealed.

## Search

The 24-trial SQLite study used four Stage 2A selection parameters:
large-winner bonus, Regime-selectivity loss, persistent-chop emphasis, and
teacher-guidance dropout end. Learning rate and policy-retention loss remained
fixed at `0.0001` and `10`. One MPS trial ran at a time.

## Objective and feasibility

The scalar objective was `100*pass_rate + 8*min(average_win_r,8)`. Feasibility
required zero teacher-free selection blows, at least 20% pass rate, at least
3R average winners, at least 38% trade win rate, nonnegative expectancy, at
least 70% two-R MFE capture, both directions, no validation short circuit, and
no worse than the frozen near-blow ceiling. These ranking results are invalid
because the training-prune evidence was defective.

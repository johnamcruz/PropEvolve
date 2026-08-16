# Stage 2A constrained-TPE selection v1

## Decision

Use Optuna's constrained TPE sampler to search the balance between Regime-led
chop avoidance, Expansion participation, autonomous Long/Short activity,
winner retention, prop-account safety, and pass consistency.

## Frozen evidence boundary

Every trial starts from the same authenticated Stage 1 parent and uses the
same seed, 2021–2024 data roles, Expansion and three-state Regime artifacts,
recurrent C51 architecture, replay contract, costs, one-contract risk limits,
and 100-episode Stage 2A screen. Selection is greedy and teacher-free. The
period beginning in 2026 remains sealed. The autonomous final 20% of training
is fixed and is not a search parameter.

## Search

The 24-trial SQLite study uses six training-only learning or selection
parameters: learning rate, policy-retention loss, large-winner bonus,
Regime-selectivity loss, persistent-chop emphasis, and teacher-guidance
dropout end. Entry supervision, ratchet behavior, account loss, position size,
MLL headroom, per-trade risk, data, labels, artifacts, architecture, and
temporal roles are fixed by the authenticated Stage 1 parent and are not
tunable.

One MPS trial runs at a time. The existing episode-18 training short circuit,
episode-100 policy-health/economic-futility checks, first-blow selection stop,
and five-no-trade selection stop remain authoritative. SQLite records every
complete, pruned, failed, and infeasible trial and supports bounded resume.

## Objective and feasibility

The scalar objective mirrors the established algoTraderRL approach:

`100*pass_rate + 8*min(average_win_r,8)`.

TPE may learn from every completed result, but a candidate is eligible only
when teacher-free selection has zero blows, at least 20% pass rate, at least
3R average winners, at least 38% trade win rate, nonnegative expectancy, at
least 70% two-R MFE capture, Long and Short entries, no validation short
circuit, and no worse than the frozen near-blow ceiling. Existing matched
parent-improvement and retention gates still apply. Only the highest-objective
feasible recipe may advance to the 200-episode confirmation campaign.

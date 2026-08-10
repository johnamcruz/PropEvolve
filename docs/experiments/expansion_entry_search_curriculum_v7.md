# Expansion Entry-Search Curriculum v7

## Decision

Test whether causal Expansion readiness can teach the final recurrent policy when
to begin evaluating Long and Short entries, while RL economics retains authority
over `ENTER`, `WAIT`, `HOLD`, and `CLOSE`.

## Frozen contract

- Training population: nine 3-minute markets, 2021-2024 inclusive.
- Selection: NQ during 2025.
- Sealed: 2026 and later; never loaded by training or selection.
- Representation: authenticated frozen Chronos2 Mask cache.
- Expansion supervision: authenticated four-channel, nine-market training cache.
- Challenge rules, costs, execution, risk, temporal splits, and cache lineage are
  unchanged from curriculum v6.
- Validation receives no Expansion teacher, teacher cache, or teacher head.

## Learning flow

```text
FFM recurrent market memory
  -> Expansion readiness supervision (training only)
  -> independent Long and Short entry values versus WAIT
  -> RL economic values for ENTER / WAIT / HOLD / CLOSE
  -> teacher head discarded
  -> teacher-free chronological validation
```

Replay reserves 25% of sequences for the strongest causal Expansion-entry
opportunities, 25% for highest MLL risk, and 50% for terminal outcomes. The
entry-search loss acts only while flat; it cannot train `HOLD` or `CLOSE`.

## Pivot readiness

Pivot is the next readiness source and will use the same training-only pattern.
It is intentionally disabled in v7 because the available Pivot artifact is an
NQ-only paired-timeframe POC and does not match the authenticated nine-market
3-minute population. No stale or mismatched Pivot teacher may be substituted.

## Falsifier

Revise or reject if post-warmup behavior still collapses toward one-bar holding,
if entries do not concentrate at higher Expansion readiness, or if 2R+ winner
retention and pass rate do not improve while validation blow rate remains zero.

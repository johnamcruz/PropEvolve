# Stage 2 v5 exact-episode curriculum

## Failed boundary

Stage 2 v4 used a 500,000-environment-step floor. Its long challenge windows exhausted that floor after approximately 40 completed episodes, so the nominal episode ceiling did not provide the intended balanced market exposure or autonomous consolidation. A training blow is evidence, not an automatic training failure; teacher-free selection still requires zero blows.

## Frozen hypothesis

An exact 200/300/500-episode ladder should expose the recurrent policy to enough balanced nine-market challenge windows to learn persistent-chop avoidance while retaining Stage 1 expansion timing. The only Regime action auxiliary remains the persistent-chop negative weight on authenticated exact-WAIT rows. It must never apply a soft-WAIT veto to positive Long or Short rows.

## Causal and model identity

- Immutable Stage 1 parent candidate: `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- Parent model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`
- Training: 2021-01-01 through 2024-12-31
- Inner teacher-free selection: calendar 2025
- Sealed confirmation: calendar 2026, untouched during this campaign
- Expansion semantic-teacher autonomy boundary: 80%
- Exact Entry consolidation autonomy boundary: 95%
- Markets: NQ, ES, GC, RTY, YM, CL, SI, ZB, and ZN

## Ladder

Each tier declares 200 teacher-free validation episodes and permits no recipe revision. Validation stops early only on decisive failure: the first blow violates the zero-blow gate, while five consecutive no-trade episodes establish universal-WAIT collapse. Otherwise all 200 episodes run. A tier must pass before its selected child becomes the warm-start parent of the next tier.

1. `persistent_chop_negative_200ep`: exactly 200 training episodes.
2. `persistent_chop_negative_300ep`: exactly 300 training episodes.
3. `persistent_chop_negative_500ep_full_coverage`: exactly 500 training episodes, with a deterministic full-data coverage receipt and 100% minimum per-market decision-row coverage.

These are additional training budgets at each selected-child stage: the 300-episode tier starts from the selected 200-episode child, and the 500-episode tier starts from the selected 300-episode child. Environment-step counts remain diagnostic telemetry only; they cannot stop training, drive curriculum schedules, or satisfy a stage gate. The stopped v4 trace averaged 11,281 decisions per episode, so the frozen ladder is expected to expose roughly 2.26M, 3.38M, and 5.64M decisions respectively. The final coverage receipt—not that projection—is authoritative for whether every eligible 2021–2024 decision row was actually visited.

All tiers require zero validation blows, at least 20% pass rate, at least 1.75R average winner, nonnegative expectancy, at least 70% 2R MFE capture, nonzero Long and Short entries, and near-blow timeout rate no worse than the frozen Stage 2 v2 boundary. The final tier additionally requires complete and balanced full-data coverage.

## Fail-fast training health

The first outcome/collapse short-circuit boundary is 18 completed episodes. While exact Entry supervision is active, every optimizer-bearing episode immediately requires WAIT, Long, and Short effective mass fractions between 0.30 and 0.36. While the Regime objective is active, positive-entry rows must have zero Regime soft-WAIT vetoes. These are implementation invariants, not noisy performance estimates. At episode 45 and each subsequent 45-episode interval, the fixed teacher-free policy-health probe requires recall of at least 0.35 WAIT, 0.30 Long, and 0.30 Short. Economic futility is evaluated from episode 45 onward and stops only when at least two of excessive near-blow timeouts, mean PnL at or below -$1,500, and expectancy at or below -0.15R fail together. Entry-mass checks cease after the 95% Entry-consolidation boundary because the remaining tail is deliberately auxiliary-free.

Every completed episode writes an authenticated policy-health receipt. A health failure checkpoints immediately even between periodic checkpoint intervals. The fixed teacher-free probe corpus is selected once, identity-bound to the run, and reused at later milestones so recall changes are comparable rather than resampled noise.

## Falsifier

Reject the recipe if any tier fails its frozen teacher-free economic gate, triggers a policy-health short circuit, loses either entry direction, violates Stage 1 parent retention, or cannot attest complete balanced coverage at the 500-episode tier. Do not revise this campaign in place.

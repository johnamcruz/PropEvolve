# Stage 2 v7 100-episode evidence horizon

## Identity and decision

Stage 2 v6 is stopped with valid negative evidence after 45 training episodes.
It did not crash and it did not run validation. Stage 2 v7 is a fresh
experiment rooted directly in the immutable authenticated Stage 1 parent; it
does not resume v6 or use v6 as a parent.

- Contract: `stage2_v7_100_episode_evidence_horizon_v1`
- Governance: `ML-RIGOR-v1`
- Stage 1 candidate: `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- Stage 1 model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`
- Recipe: `config/historical_mask_expansion_regime_stage2_v7.json`
- Output: `runs/historical_mask_expansion_regime_stage2_v7`
- Status: frozen exploration

## Observed v6 failure

At episode 45, v6 had 8 passes, 1 blow, and 36 timeouts. The blow occurred on
CL at episode 31 after repeated entries in dominant chop; it was not an
infrastructure failure or one terminal gap. The fixed teacher-free health probe
measured WAIT recall `5/32=0.15625`, Long recall `23/32=0.71875`, and Short
recall `25/32=0.78125`. Its dead-WAIT versus transition-positive WAIT margin
was positive but only `+0.001949`. The sampled optimizer rows were materially
more optimistic, so sampled recall remains diagnostic only.

The first failed boundary is Regime/action learning to fixed teacher-free action
generalization. Class mass, data identity, cache lineage, Long learning, and
Short learning passed. V6 did not establish whether the WAIT failure was a
transient 45-episode state or persisted after a complete curriculum.

## Falsifiable claim

With v6 data, labels, temporal roles, seed, architecture, optimizer, replay,
objectives, schedules, and economics unchanged, completing a 100-episode
curriculum before the first binding teacher-free policy-health decision should
raise fixed-corpus WAIT recall above the frozen health floor while preserving
Long, Short, and positive dead-chop association.

V7 falsifies “more training data is sufficient” if the fixed probe still fails
at episode 100. In that case, do not tune the learning rate and do not add more
episodes: reproduce the fixed-versus-sampled mismatch through the public
training/health seam and revise the action objective or replay consumer in a
new experiment.

## One matched change

Only the evidence horizon and declared episode ladder change from v6:

- First binding policy-health probe: episode 100, then every 100 episodes.
- Economic-futility evidence boundary: episode 100.
- Training tiers: 100, 250, and 500 episodes.

The early outcome/collapse boundary remains 18 episodes. Integrity faults,
non-finite optimization, zero positive-label fidelity, excessive blow rate, and
declared collapse may still stop before episode 100. A run is not guaranteed to
reach 100 if a genuine catastrophic invariant fails.

Everything else is frozen to v6:

- 2021–2024 training, 2025 teacher-free validation, sealed 2026 untouched.
- NQ, ES, GC, RTY, YM, CL, SI, ZB, and ZN.
- Exact WAIT/Long/Short labels and `persistent_chop_association_v2`.
- Same Expansion and Regime cache identities and fit-only centers.
- Same Stage 1 parent, seed `314159`, recurrent C51 agent, learning rate,
  sampling, risk, fees, fills, stops, targets, and simulator.
- Same teacher autonomy at 80% and exact-Entry autonomy at 95% of each tier.
- No recipe revisions and no search.

## Episode ladder and validation

1. `persistent_chop_association_100ep`: 100 training episodes.
2. `persistent_chop_association_250ep`: 250 additional episodes from the
   selected 100-episode child.
3. `persistent_chop_association_500ep_full_coverage`: 500 additional episodes
   from the selected 250-episode child plus authenticated full-data coverage.

Every tier receives up to 200 chronological, greedy, teacher-free validation
episodes. Validation stops immediately on one blow or after five consecutive
no-trade episodes. Otherwise all 200 episodes run.

The unchanged gates require zero validation blows, pass rate at least 20%,
average winner at least 1.75R, nonnegative expectancy, at least 70% 2R MFE
capture, both Long and Short entries, and the frozen near-blow ceiling. Every
tier must strictly improve pass rate and near-blow timeout rate over its selected
parent while retaining Stage 1 economics. The final tier also requires exact
balanced full-data coverage.

## Teacher-free learning evidence

At each binding health boundary, the fixed 32-row-per-action corpus requires:

- WAIT recall at least 35%; Long and Short recall at least 30%.
- Positive dead-WAIT minus transition-positive WAIT response.
- Positive transition-positive Long and Short responses.
- Zero positive-entry soft-WAIT veto and bounded optimizer class mass.

Final training evidence remains stricter: WAIT at least 50%, Long and Short at
least 40%, all three association responses positive, teachers and retention
discarded, and no teacher/exact-label state accessible to validation. Sampled
optimizer recall cannot substitute for the fixed teacher-free probe.

## Stop or proceed

- Stop immediately for lineage, temporal, cache, label, non-finite, or source
  parity faults.
- Stop under the retained early catastrophic/collapse contract.
- Stop at episode 100 if the fixed teacher-free health probe fails.
- Run validation only after training and final teacher-free gates pass.
- Advance to 250 and 500 only from a selected passing child.
- Never use the sealed 2026 period to repair or select v7.

## Expected artifacts

- `training-diagnostics.jsonl`
- `training-policy-health.jsonl`
- `training-policy-health-probe.pkl`
- `training-diagnostic-summary.json`
- `final-regime-probe.json`
- `validation-diagnostics.jsonl` only after training gates pass
- content-addressed candidate and evaluation receipts

## Approval

- G0: PASS — the user authorized the 100/250/500 episode ladder.
- G1: pending code/config tests, clean commit, parent/cache authentication, and
  one-shot launch safety audit.

# Stage 2 v6 persistent-chop association

## Decision and experiment identity

Stage 2 v5 is stopped with negative evidence after 42 of its first 200 training episodes. Stage 2 v6 is a fresh experiment identity rooted directly in the immutable authenticated Stage 1 parent. It is not a revision or continuation of v5, and it does not authorize restarting training.

- Contract: `stage2_v6_persistent_chop_association_v1`
- Governance: `ML-RIGOR-v1`
- Stage 1 parent candidate: `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- Parent model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`
- Recipe: `config/historical_mask_expansion_regime_stage2_v6.json`
- Fresh output root: `runs/historical_mask_expansion_regime_stage2_v6`
- Status: frozen; operational launch state is recorded by the campaign receipt

## v5 negative evidence and failed boundary

The frozen 42-episode v5 trace produced 5 passes, 0 blows, and 37 timeouts. Sixteen of the 37 timeouts were near-blows. Across 7,116 trades, win rate was 43.9%, average winner was 0.572R, average loser was -0.463R, expectancy was -0.0085R, and net PnL was -$18,180. Its optimizer diagnostics reported mean model WAIT probability 0.409 on persistent-dead rows versus 0.422 on transition-ready rows, a dead-minus-ready association of -0.0132. The association was negative in all 35 episodes in which the objective was active. V5 was stopped before its first fixed teacher-free policy-health probe, so these figures are training diagnostics rather than final-probe evidence.

The first failed boundary is therefore not data volume. The existing negative-only objective does not associate persistent-dead evidence with greater WAIT preference than authenticated transition-positive evidence. More episodes may reduce uncertainty, but the current trace does not support treating additional unchanged v5 training as the smallest causal fix.

## Frozen hypothesis

A zero-margin association term can make persistent-dead exact-WAIT rows prefer WAIT over Entry while making exact transition-positive Long and Short rows prefer the declared side over WAIT, without changing data, labels, temporal roles, seed, economics, model architecture, curriculum, or the existing 0.3 Regime loss weight.

Exact action labels remain the sole truth and gate every cohort. Causal Kaufman, trend, volatility-expansion, and side-specific Expansion evidence only weight eligible exact-label rows. They do not create pseudo-labels, infer direction, veto a positive Entry label, or define a hard inference rule.

## One matched change family

The public semantics is `persistent_chop_association_v2`. The serialized formula is:

`equal_present_group_mean(exact_wait_weighted_ce,exact_long_ce,exact_short_ce,zero_margin_dead_vs_transition_positive_wait_rank)`

The existing Regime group retains `loss_weight=0.3`, `q_temperature=1`, `persistent_chop_negative_emphasis=1`, and `side_balance=equal_long_short_v1`. There is no margin, temperature, threshold, auxiliary weight, new objective subobject, or revision path.

For an exact-label row, using only the already authenticated causal channels:

- `q = kaufman_efficiency * mean(trend_onset, trend_persistence, volatility_expansion_onset, volatility_high_persistence, volatility_percentile)`
- `c = chop_persistence`
- `e_s = sigmoid(logit(clamp(attempt_s * clean_s)) - logit(authenticated_fit_only_center_s))`
- persistent-dead membership: `d = 1[y=WAIT] * c * (1-q)`
- transition-positive membership by side: `r_s = 1[y=s] * c * q * e_s`, for exact `s` in Long or Short

Each cohort is normalized separately. The transition-positive model-WAIT mean is the equal-present-side mean of separately normalized Long and Short cohorts. The association group applies zero-margin ranking, and the total Regime objective is the equal-present-group mean of the existing exact WAIT, Long, and Short cross-entropy groups plus this association group.

## Frozen controls, data, and causality

Everything outside that objective family is copied exactly from the authenticated Stage 2 v5 recipe and remains matched to Stage 1:

- Training data: 2021-01-01 through 2024-12-31
- Inner teacher-free selection: calendar 2025
- Sealed confirmation: calendar 2026, untouched and unavailable for selection
- Markets: NQ, ES, GC, RTY, YM, CL, SI, ZB, and ZN
- Same cache, row eligibility, exact WAIT/Long/Short labels, splits, seed, teachers, observation, agent, optimizer, challenge rules, point values, fees, and execution economics
- Same authenticated fit-only side centers and receipt SHA-256
- Same 80% semantic-teacher and 95% exact-Entry consolidation boundaries
- Stage 1 is the immutable parent and retention control; v5 is negative diagnostic evidence, not v6's parent

Any drift in those fields invalidates the matched comparison and requires a new experiment identity.

## Exact episode ladder and validation

No tier permits a recipe revision. A tier must pass before its selected child can warm-start the next tier.

1. `persistent_chop_association_200ep`: exactly 200 training episodes.
2. `persistent_chop_association_300ep`: exactly 300 additional training episodes.
3. `persistent_chop_association_500ep_full_coverage`: exactly 500 additional training episodes and the existing deterministic full-data coverage receipt.

Each tier declares 200 teacher-free validation episodes. Validation ends early only on decisive failure: one blow violates the zero-blow gate, and five consecutive no-trade episodes establish universal-WAIT collapse. Otherwise all 200 validation episodes run. Environment-step counts remain diagnostic telemetry only.

The existing absolute economic gates are unchanged: zero blows, at least 20% pass rate, at least 1.75R average winner, nonnegative expectancy, at least 70% 2R MFE capture, nonzero Long and Short entries, and near-blow timeout rate no worse than the frozen Stage 2 v2 boundary. Every tier must also improve both outcomes against its selected parent: `selection.pass_rate` must be strictly higher and `selection.near_blow_timeout_rate` must be strictly lower. Both serialize `minimum_delta=0.0`; orchestration accepts only `improvement > minimum_delta`, so equality fails closed. The final tier additionally requires complete balanced full-data coverage and unchanged Stage 1 retention gates.

## Fail-fast and falsifier

The first outcome/collapse boundary remains 18 completed episodes, and the periodic fixed teacher-free policy-health probe remains at episode 45 and every 45 episodes thereafter. Existing entry-mass, recall, zero-positive-entry soft-WAIT-veto, and economic-futility invariants remain unchanged.

V6 additionally enables `require_positive_persistent_regime_association`. At every applicable health boundary, the fixed teacher-free probe uses the exact v2 compiler and authenticated fit-only centers and requires all three trained-objective association metrics to be strictly positive: `final_regime_probe_dead_wait_minus_transition_positive_wait`, `final_regime_probe_transition_positive_long_response`, and `final_regime_probe_transition_positive_short_response`. The final evaluation gates the same three metrics. The older exact-WAIT diagnostic `final_regime_probe_dead_wait_minus_transition_ready_wait` and optimizer-side `regime_selectivity_dead_wait_minus_transition_positive_model_wait` remain diagnostic telemetry only and cannot satisfy either gate. There is no serialized nonzero threshold: zero is the frozen falsification boundary.

Reject v6 immediately if the association is non-positive, if any exact positive-entry row receives an impermissible soft-WAIT veto, if a tier blows or collapses to no trade, if either side disappears, if an economic or Stage 1 retention gate fails, or if final full-data coverage cannot be authenticated. Do not revise v6 in place and do not use the sealed 2026 confirmation period to repair, select, or interpret it.

## Launch gate

This contract and config freeze the experiment only. Launch remains blocked until the association implementation, health metric, parser fail-closed tests, exact-label cohort tests, authenticated center receipt, and research-to-serialized parity all pass against the frozen recipe. Passing those checks permits a separate explicit launch decision; it does not restart training automatically.

# Stage 2 v8: hierarchical Entry and Regime association

## 1. Identity

- Experiment ID: `stage2_v8_hierarchical_entry_regime_v1`
- Process: `ML-RIGOR-v1`
- Repository: PropEvolve
- Date opened: 2026-08-13
- Status: implementation and G1 verification; launch authorized after G1 passes
- Recipe: `config/historical_mask_expansion_regime_stage2_v8.json`
- Parent: immutable Stage 1 candidate `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- Parent model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`
- Baseline: Stage 2 v7, with `equal_present_class_mean_v1`

## 2. Hypothesis

### Decision

Improve the recurrent policy's flat-state decision without changing its deployed interface: first learn whether the state is `WAIT` or aggregate `ENTER`, then learn `Long` or `Short` conditional on an authentic ENTER label.

### Causal information and mechanism

At each flat decision the policy sees only the existing causal recurrent observation, frozen FFM embedding, Expansion channels, Regime channels, and challenge state. The existing three-class reduction gave WAIT one class group while aggregate ENTER occupied two groups. V6 then learned Long and Short but generalized WAIT poorly. V8 rebalances two economically distinct tasks:

1. `entry_timing_loss`: equally balance exact WAIT against aggregate ENTER. Aggregate ENTER is the `logsumexp(Long, Short)` marginal of the unchanged three-action logits.
2. `entry_direction_loss`: equally balance Long against Short only on exact ENTER labels.

The training-only Entry objective is the sum of the active timing and conditional-direction losses; the direction term is zero when no authentic ENTER rows are present. Regime learning remains soft and training-only. RL reward, recurrent credit assignment, management actions, and challenge economics remain active.

### Falsifiable claim

Against the direct immutable Stage 1 parent and matched v7 recipe, the loss-only hierarchy should restore teacher-free WAIT competence and positive dead-chop versus transition-positive association while retaining Long/Short competence, trend capture, and useful opportunity frequency. It must produce zero validation blows, strictly improve pass rate and near-blow timeout rate over the parent, and retain the parent's existing economic metrics.

### Stop or revise

Stop at the first applicable boundary if hierarchical mass is absent or not exactly 0.5/0.5 within either task, the fixed teacher-free probe fails WAIT/Long/Short recall, persistent-Regime association is nonpositive, the campaign short-circuits, validation blows once, or five validation episodes produce no trades. Reject the candidate if strict parent pass-rate or near-blow improvement fails. A failure does not authorize a learning-rate, replay, architecture, threshold, or data change; each would require a new experiment identity.

## 3. Ownership

| Artifact | Owner | Immutable identity |
|---|---|---|
| Raw futures data and continuous contracts | PropEvolve local assets | `config/local-assets.json` and authenticated runtime manifests |
| Frozen FFM representation | Futures-Foundation-Model | encoder `1b8b7f001b0b4e501aa47ca90a3c2fd31d0b41dbd1d896e98ce084f6ed325710` |
| Expansion and Regime caches | PropEvolve | per-market manifests, fit rows strictly before 2025 |
| Exact Entry labels and challenge simulation | PropEvolve | frozen v7 contracts |
| Stage 1 recurrent policy | PropEvolve | candidate and model hashes above |
| V8 candidate, diagnostics, and evaluation | PropEvolve | fresh v8 output and campaign state roots |

Private strategy IP remains in PropEvolve; no artifact moves to the public FFM repository.

## 4. Executable decision contract

- Universe: NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN for training; NQ deployment slice; CL, SI, ZB, ZN remain training-only.
- Timeframe: 3-minute bars; timestamps and roll/session rules are authenticated by the existing asset/cache manifests.
- Lookback: 256 frozen-representation bars plus the existing 96-step replay sequence and 64-step recurrent horizon.
- Decision: existing flat-state recurrent action event.
- Execution: next-bar open at offsets 1 through 5 after launch evidence.
- Entry label: +2R before -1R within 150 bars; stop-first collision handling; $300 risk.
- Positioning: one contract, no overlapping position.
- Challenge: +$6,000 pass, -$3,000 trailing MLL blow, $500 ordinary entry headroom, existing fees by market.
- Live interface: unchanged recurrent C51 action set and three-action inference argmax.

```text
causal bars and state -> recurrent WAIT/Long/Short values -> next-bar action -> exact challenge simulator
```

## 5. Data and label identity

- Training: `[2021-01-01, 2025-01-01)`.
- Inner selection: `[2025-01-01, 2026-01-01)`; repeatedly inspected and therefore development evidence, not sealed proof.
- Sealed confirmation: `[2026-01-01, 2027-01-01)`; prohibited during this exploration and opened only after the complete recipe and gates are frozen.
- Representation cache: `cache/chronos2_mask_full_3min_pre2026_v2`, schema `ffm_frozen_representation_v2`, context 256, stride 1.
- Expansion cache: `cache/expansion_teacher_9market_3min_pre2025_v1`.
- Regime cache: `cache/regime_teacher_9market_3min_pre2025_v1`.
- Entry labels: existing `post_launch_entry_v1`; causal next-bar execution, stop-first ambiguity, symmetric Long/Short construction.
- Missing, nonfinite, misaligned, temporally overlapping, or identity-drifted rows fail closed before training.

## 6. Matched comparison

The only model-learning change from v7 is:

```text
agent.entry_action_loss_reduction:
  equal_present_class_mean_v1
  -> hierarchical_enter_wait_direction_v1
```

Frozen across v7 and v8: architecture, observation/action tensors, state-dict keys, parent, data, caches, teacher channels, labels, market set, temporal roles, seed 314159, learning rate 0.0001, replay capacities and sampling fractions, recurrent settings, risk, costs, reward, Regime association, teacher schedules, 100/250/500 episode tiers, and 200-episode teacher-free validation.

## 7. Stage 1 compatibility and regression boundary

V8 is a same-architecture loss experiment. It adds no head, parameter, action, scalar, threshold, serialized teacher, or inference branch. The Stage 1 checkpoint must load with the exact state keys and tensor shapes, and its Q-values, recurrent state, and greedy actions must be identical before the first optimizer update. Training writes only to a fresh v8 candidate; it never overwrites Stage 1. The parent model SHA-256 is authenticated before use.

Checkpoint compatibility does not guarantee behavioral retention. V8 is rejected if it loses Stage 1 trend-day Long/Short competence, pass rate, average winning R, expectancy, 2R MFE capture, or near-blow performance. Existing opportunity-frequency metrics (`trade_count`, `greedy_entry_count`, Long/Short entry counts and shares) remain diagnostics and anti-collapse evidence. No exact 3R opportunity metric currently exists, so v8 does not invent a 3R promotion gate; available MFE, realized-R, 2R capture, and large-win diagnostics must be reported.

The production G1 compatibility check loaded the authenticated archived Stage 1
model (`b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`)
into the actual v8 MPS configuration with zero optimizer updates. Shared online and
target tensors, recurrent features and final hidden state, Q-values, and greedy
actions were bit-exact on the authenticated 96-sequence fixed probe; the parent
file hash was unchanged and the parent remained teacher-free. This checks the
real checkpoint boundary, while the portable unit fixture keeps CI independent
of machine-local run artifacts.

## 8. Model and training

- Architecture: unchanged recurrent C51, hidden dimension 128, 51 atoms, value support [-3R, +3R].
- New objective mode: `hierarchical_enter_wait_direction_v1`.
- The frozen `entry_supervision.action_class_balance=inverse_frequency_v1` receipt is retained for exact label lineage and backward compatibility, but hierarchical mode does not consume its legacy three-class weights. It derives equal mass independently from the present WAIT/ENTER rows and, conditionally, the present Long/Short rows.
- Timing mass gate: WAIT 0.5, aggregate ENTER 0.5.
- Conditional direction mass gate: Long 0.5, Short 0.5.
- RL: unchanged 8-step return, gamma 0.997, recurrent burn-in 64.
- Optimizer: unchanged learning rate 0.0001, weight decay 0.00001, gradient clip 10, soft target tau 0.005.
- Replay: unchanged 500 episodes / 500,000 transitions; 32 updates per episode, 16 sequences per batch; terminal 0.50, safety 0.25, Entry opportunity 0.25.
- Regime: existing 18-channel training-only teacher plus persistent-chop association v2.
- Autonomy: Regime/Expansion guidance ends by 80%; Entry supervision ends by 95%; validation and produced policy are teacher-free.
- Episode ladder: 100, 250, 500; final tier requires full market coverage.
- Search: none. No revisions and no hyperparameter tuning are allowed inside v8.

## 9. Metrics and gates

Training integrity gates require nonempty rows and exact 0.5 mass for:

- `entry_timing_wait_*` and `entry_timing_enter_*`;
- `entry_direction_long_*` and `entry_direction_short_*`.

At episode 100 and each configured health boundary, the fixed authentic teacher-free corpus requires 32 rows per action, WAIT recall at least 0.35, Long and Short recall at least 0.30, and positive:

- `final_regime_probe_dead_wait_minus_transition_positive_wait`;
- `final_regime_probe_transition_positive_long_response`;
- `final_regime_probe_transition_positive_short_response`.

Selection requires zero blow, pass rate at least 0.20, average winning R at least 1.75, nonnegative expectancy, 2R MFE capture at least 0.70, both Long and Short entries, and near-blow timeout rate no greater than 0.6263636364. In addition, pass rate and near-blow timeout rate must each strictly improve over the direct parent. Parent retention permits zero regression in pass rate, average winning R, expectancy, 2R MFE capture, and near-blow timeout rate.

Required reporting slices: ticker, Long/Short, guidance versus autonomy, Regime channel errors, persistent dead chop, transition-ready chop, transition-positive Long/Short, episode outcome, headroom, signal frequency, MFE/MAE, realized R, and worst episode.

## 10. TDD evidence and execution plan

The v8 implementation began RED at public seams before production support:

1. Config failed because the v8 recipe and reduction were absent.
2. Health failed because hierarchical task mass was not represented or consumed.
3. Evaluation rejected the new reduction and lacked exact task-mass gates.
4. Training diagnostics did not propagate the learner's hierarchical metrics.

GREEN requires the focused config, training, health, orchestration, learner, Regime-contract, and full repository tests. G1 additionally requires authenticated parent/cache/data manifests, Stage 1 same-shape pre-update parity, teacher-free save/load, available disk/MPS resources, fresh v8 paths, and no live v7 mutation. The user authorized a Stage 2A v8 launch after these integrated tests, reviews, and G1 checks pass.

## 11. Expected artifacts

- `training-diagnostics.jsonl` with timing/direction losses and exact cohort ledgers.
- `training-policy-health.jsonl` with hierarchical mass and fixed-probe evidence.
- `training-diagnostic-summary.json` with Regime and hierarchical aggregates.
- `final-regime-probe.json` from teacher-free policy inference.
- Candidate contract authenticating parent, source code, configuration, caches, teacher-free state, and new reduction.
- Teacher-free selection receipts and, only after recipe freeze, sealed confirmation receipts.

## 12. Known risks and backlog

- The summed direction task can increase total auxiliary gradient on ENTER rows; the frozen scale is intentional and must be judged by matched economics, not changed inside v8.
- Repeated use of 2025 can overfit experiment design; it remains selection-only.
- The untouched 2026 period is the one-shot, teacher-free final overfit check
  across all nine markets. Seeing it does not authorize any recipe change; a
  failure rejects v8.
- Exact 0.5 optimizer mass proves objective construction, not economic learning; fixed teacher-free association and economics remain binding.
- A safer policy can fake improvement by not trading; both sides, signal-frequency diagnostics, pass improvement, and retention gates guard against this.
- Exact 3R opportunity capture is unavailable. Add it only as a separately tested diagnostic in a future experiment; do not revise v8.

## Approval

- G0: PASS for a bounded, matched, loss-only experiment.
- G1: IN PROGRESS until the integrated branch passes all tests and live artifact preflight.
- Launch: authorized after G1 passes; Stage 2B recovery remains out of scope.

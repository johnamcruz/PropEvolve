# Stage 2-v4 Short-collapse repair

> G0 research contract. Freeze this document before launching `stage2-v4`.
> Passing this screen selects a research candidate; it is not sealed confirmation
> or production promotion.

## 1. Identity and decision

- Experiment ID: `stage2-v4`
- Process: `ML-RIGOR-v1`
- Repository: `PropEvolve`
- Date opened: `2026-08-13`
- Status: selection; not launched when this contract was written
- Recipe: `config/historical_mask_expansion_regime_stage2_v4.json`
- Stage: `persistent_chop_negative_500k`
- Failed diagnostic baseline: completed `stage2-v2`
- Immutable economic parent: Stage 1 candidate
  `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- Parent evaluation:
  `c49852955655b705e376e057dfe2bf58784481175363b970bab063d8c42f981b`
- Parent model SHA-256:
  `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`

The decision being improved is the teacher-free recurrent policy's causal
`WAIT | ENTER_LONG_1 | ENTER_SHORT_1` decision while flat. v4 tests whether
removing contradictory positive-Entry Regime targets restores Short learning
without weakening WAIT, Long, Stage 1 safety, or winner retention.

## 2. Completed Stage 2-v2 failure evidence

The failed v2 candidate and evaluation are immutable negative evidence:

- candidate:
  `5825dd4b326af7d1f9dab622a621d62892696773fddec6df3359d058ecd63b84`
- evaluation:
  `8c3234e54783c99900aff70c5773f773b3aa911972aaec89ab476c662e193fd3`
- selection: 25 passes, 0 blowouts, and 175 timeouts in 200 episodes;
- pass rate: `0.125`;
- near-blow timeouts: 114 of 175, rate `0.6514285714285715`;
- trades: 10,695, with 3,305 winners, win rate
  `0.30902290790088827`;
- average winner: `2.086355582450871R`;
- mean terminal P&L: `-$789.2352000004711`;
- greedy Entry rate: `0.09548510360959583`;
- sampled best-Entry-minus-WAIT advantage: `-0.07022763296583233`;
- Long: 6,226 entries, win rate `0.3143270157404433`, expectancy
  `-0.03491061141452578R` per trade;
- Short: 4,469 entries, win rate `0.3016334750503468`, expectancy
  `-0.06909897814577692R` per trade.

The final teacher-discarded v2 probe used 32 rows per class. Recall was WAIT
`0.65625`, Long `0.4375`, and Short `0.1875`. Of 32 Short targets, the policy
predicted WAIT 25 times, Long once, and Short only six times.

This was not a missing-Short-data or hard-class-balance failure:

- authenticated Entry targets remained WAIT 394,793, Long 40,353, and Short
  39,001;
- sampled guided rows were WAIT 6,785, Long 3,175, and Short 3,103;
- the hard Entry reducer assigned exactly `288.00000858306885` weighted mass
  to each present class, or one third each.

The first failed boundary was the static Regime action auxiliary. On the same
positive Entry rows it produced these conflicting soft targets:

| Declared Entry side | Rows | Target WAIT mean | Declared-side mean | Soft-WAIT disagreements |
|---|---:|---:|---:|---:|
| Long | 3,175 | `0.4677441891910523` | `0.5322558188626146` | 1,380 (`0.4346456692913386`) |
| Short | 3,103 | `0.7235930097360516` | `0.2764069716603073` | 2,568 (`0.8275862068965517`) |

With Entry loss weight `0.3` averaged across three classes, each hard class
had a `0.10` objective budget. Regime loss weight `0.3` averaged across two
sides supplied `0.15` per side. The untrusted Regime veto was therefore 1.5x
the per-class hard objective and asymmetrically contradicted Short.

## 3. Hypothesis and falsifier

At a completed three-minute decision bar, replacing the positive-Entry Regime
soft veto with `persistent_chop_negative_weight_v1` weighting on exact WAIT
labels, then consolidating exact Entry labels independently to the 95% boundary,
should eliminate Regime/Entry conflict and late three-action forgetting. On
matched 2025 teacher-free selection, the resulting policy should retain all three actions,
restore final Short recall to at least 40%, improve the Stage 1 near-blow
timeout rate by at least one percentage point, and retain Stage 1 economics.

This is a test of objective compatibility, not proof of Regime-teacher quality.
There is currently no accepted evidence that the Regime scores economically
separate chop, predict profitable direction, or add causal OOS lift. Regime
score-response deltas remain diagnostics and are not binding gates.

The claim is falsified if positive Entry rows receive any Regime-induced WAIT
target, any final action recall misses its threshold, the 2025 policy loses a
direction, near-blow improvement misses its target, or any Stage 1 retention
gate regresses. No automatic revision is permitted; a failed v4 becomes
negative evidence and any next change requires a new experiment identity.

## 4. Frozen executable and label contract

- Universe: NQ, ES, GC, RTY, YM, CL, SI, ZB, and ZN for training; NQ is the
  deployment/chronological-selection market.
- Input: causal completed-bar frozen Mask embeddings plus causal account and
  position-management state; teachers and labels are training-only.
- Timeframe and decision cadence: three-minute bars.
- Entry execution: next-bar open.
- Entry action label: learn WAIT, Long, or Short at decision bars 1 through 5
  after authenticated Expansion onset (`fill_offsets = [1,2,3,4,5]`).
- Outcome label: `+2R` before `-1R` within 150 bars, with `stop_first`
  same-bar collision handling.
- Position size: one contract maximum; no overlapping position.
- Per-trade risk and stop: fixed `$300`, equal to `1R`.
- Prop economics: start at zero, pass at `+$6,000`, fail at `-$3,000`, 30
  sessions, trailing MLL lock, configured per-market point values and exact
  round-trip fees.
- Winner retention: ratchet activates at `2R`, permits `0.5R` giveback, and
  keeps a `2R` lock floor. Maximum reachable winners are not capped by this
  experiment.

Timeline:

```text
completed causal lookback -> decision bar 1..5 -> next-bar-open entry
-> +2R before -1R label horizon -> recurrent management/terminal outcome
```

Data, FFM cache identity, teacher caches, target construction, row eligibility,
action order, execution timing, costs, challenge rules, and metric code are
identical to v2 and the Stage 1 lineage. Unknown lineage, cache mismatch,
boundary-crossing rows, non-finite inputs, impossible labels, or simulator
timing disagreement stops the run immediately.

## 5. Temporal roles

| Role | Dates | Permitted use |
|---|---|---|
| Research/train | 2021-01-01 through 2024-12-31 | Fit the warm-started recurrent policy |
| Inner selection | N/A | No hyperparameter or architecture search in v4 |
| Chronological selection | 2025-01-01 through 2025-12-31 | One matched 200-episode teacher-free screen |
| Sealed confirmation | 2026-01-01 through 2026-12-31 | Untouched; prohibited for v4 training, selection, diagnosis, or revision |
| Shadow/live | N/A | Outside this research screen |

The 2025 selection period has already influenced Stage 1 and Stage 2 design, so
it is development/selection evidence, not sealed confirmation. No 2026 result
was inspected to design this experiment. Passing v4 does not authorize opening
the sealed period.

## 6. Two-boundary matched repair

v4 warm-starts the exact Stage 1 model, not the failed v2 model. The completed
v2 run and deterministic TDD reproduction established two distinct causes in
the same Short-collapse lifecycle, so v4 changes both predeclared boundaries:

1. Regime action supervision switches from positive-Entry softening to
   exact-WAIT-only persistent-chop weighting; and
2. exact Entry consolidation decays independently through 95% of training,
   after Regime/teacher guidance reaches zero at 80%, leaving the final 5%
   fully teacher- and label-free.

The objective switch is expressed atomically by three fields:

```json
{
  "formula": "exact_wait*(1+persistent_chop_negative_emphasis*chop_persistence*(1-kaufman_efficiency*mean(trend_onset,trend_persistence,volatility_expansion_onset,volatility_high_persistence,volatility_percentile)))",
  "semantics": "persistent_chop_negative_weight_v1",
  "persistent_chop_negative_emphasis": 1.0
}
```

For exact WAIT labels, the auxiliary weight is:

```text
1 + chop persistence * (1 - transition readiness)

transition readiness = Kaufman efficiency * mean(
  trend onset,
  trend persistence,
  volatility-expansion onset,
  high-volatility persistence,
  volatility percentile
)
```

Long and Short targets stay exact one-hot labels. Regime never softens them
toward WAIT, masks an action, or becomes a policy input. Equal-present-class
hard Entry reduction remains unchanged.

The independent consolidation boundary is:

```json
{"entry_supervision_autonomy_start_fraction": 0.95}
```

Entry loss decays linearly to zero at this boundary. It replays only the same
authenticated pre-2025 bar-1-through-5 action labels already used before 80%;
it adds no selection-period information. The final 5% has both Regime and
Entry loss scales exactly zero, and selection remains fully teacher- and
label-free.

Everything else is frozen: Stage 1 parent, rows, labels, split, seed `314159`,
replay capacity (500 episodes and 500,000 transitions), replay fractions,
sequence/burn-in/horizon, optimizer, action class weights, Regime loss weight
`0.3`, generic teacher dropout/autonomy boundary, 500,000-step campaign-stage budget,
200 selection episodes, risk, costs, reward, management, and evaluation code.
The global training recipe retains its one-million-step default; the matched
campaign stage resolves the same 500,000-step screen boundary used by v2.
There are zero allowed revisions.

## 7. Mechanism gates

Training fails closed unless all of these are true at the declared boundary:

1. no training or selection short circuit;
2. hard Entry WAIT, Long, and Short rows are each nonzero and each weighted-mass
   fraction is within `[0.32, 0.34]`;
3. positive Long and Short Regime-conflict rows are each nonzero, but each has
   target WAIT mean exactly `0`, declared-side target mean exactly `1`, and
   zero soft-WAIT disagreement rows;
4. exact-WAIT rows are nonzero, exact-WAIT weight mean is greater than `1`,
   and persistent-dead-chop and transition-ready WAIT weighted masses are each
   nonzero;
5. positive Long and Short rows and transition-positive Long and Short masses
   are each nonzero;
6. the final checkpoint is probed with no updates and no teachers on exactly 32
   unique authenticated pre-2025 rows for each class; WAIT recall is at least
   `0.50`, Long recall at least `0.40`, and Short recall at least `0.40`;
7. final diagnostic teacher and Entry-action loss scales are both exactly `0`;
8. checkpoint save/load and teacher-discard preserve the three-class behavior.

Persistent-dead-WAIT versus transition-ready-WAIT model deltas and
transition-positive Long/Short response deltas are recorded for diagnosis only.
Their signs cannot pass or fail v4 because current Regime scores have not
demonstrated economic separation.

## 8. Teacher-free economic gates

The primary v4 target is near-blow timeout rate
`<= 0.6263636363636363`, at least one percentage point better than the Stage 1
rate `0.6363636363636364`. Zero blowouts is mandatory.

The effective retention gates are the stricter combination of absolute recipe
floors and the immutable Stage 1 parent:

| Metric | Binding v4 gate |
|---|---:|
| Evaluated episodes | 200 |
| Blow rate | `0` |
| Pass rate | `>= 0.23` (46 of 200 parent rate) |
| Average winner R | `>= 1.952229673590538` |
| Net expectancy | `>= 0.001259368072760152R` per trade |
| 2R MFE capture ratio | `>= 0.7359998386429496` |
| Near-blow timeout rate | `<= 0.6263636363636363` |
| Long entries | `> 0` |
| Short entries | `> 0` |

Stage 1 mean terminal P&L `+$19.03219999958792`, trade win rate
`0.33449131513647645`, and all Regime-stratified responses are reported
diagnostics. They do not replace the binding gates above.

## 9. Stop, revise, and promotion rules

- Stop immediately for any lineage, causality, finite-value, split, cache,
  recovery/checkpoint, parent-hash, target, action-order, or execution-parity
  fault.
- Stop after the 500,000-step screen and complete teacher-free selection if any
  mechanism or economic gate fails. Do not rescue v4 with a blind Short class
  weight, logit bias, schedule change, Trend/Pivot teacher, recovery curriculum,
  sizing change, reward change, or a second seed.
- A diagnostic target such as 40% trade win rate, 2R average winner, 25% pass
  rate, or 50% near-blow rate is aspirational unless it is also a binding gate
  above.
- If v4 passes, archive it as a Stage 2 research candidate. Multi-seed/full
  confirmation, Trend guidance, deficit recovery, and sealed evaluation each
  require a separately frozen experiment.

## 10. Ownership, search, and known limitation

| Artifact/responsibility | Owner | Identity |
|---|---|---|
| Frozen Mask checkpoint and representation contract | FFM | existing authenticated encoder/cache SHA identities |
| Bar-1-through-5 Entry labels, RL policy, recipe, simulator, metrics, and candidate archive | PropEvolve | v4 recipe plus archived hashes/receipts |
| Live execution/risk integration | algoTraderAI | N/A; outside v4 |

Search is `N/A`: v4 is one predeclared two-boundary lifecycle repair with one
seed, one budget, no tuning, no pruning, and no reasoning revisions. The known
limitation is that exact-WAIT weighting may remove objective conflict without
providing useful economic chop separation. The matched teacher-free economics,
not training fit or Regime response, decide the result.

## 11. Commands and expected artifacts

Run only from a committed, pushed, clean `main` after the full test suite,
config validation, exact-parent preflight, and MPS behavior test pass:

```bash
.venv/bin/propevolve validate-config \
  --config config/historical_mask_expansion_regime_stage2_v4.json

.venv/bin/propevolve evolve \
  --config config/historical_mask_expansion_regime_stage2_v4.json \
  --run-id stage2-v4

.venv/bin/propevolve evolve-status \
  --config config/historical_mask_expansion_regime_stage2_v4.json \
  --run-id stage2-v4
```

Expected durable artifacts:

- `runs/historical_mask_expansion_regime_stage2_v4/ml-loop-state/stage2-v4/state.json`
- `runs/historical_mask_expansion_regime_stage2_v4/campaign-runs/stage2-v4/persistent_chop_negative_500k/attempt-1/training-diagnostics.jsonl`
- `runs/historical_mask_expansion_regime_stage2_v4/campaign-runs/stage2-v4/persistent_chop_negative_500k/attempt-1/training-diagnostic-summary.json`
- `runs/historical_mask_expansion_regime_stage2_v4/campaign-runs/stage2-v4/persistent_chop_negative_500k/attempt-1/final-regime-probe.json`
- `runs/historical_mask_expansion_regime_stage2_v4/campaign-runs/stage2-v4/persistent_chop_negative_500k/attempt-1/validation-diagnostics.jsonl`
- content-addressed candidate contract, recipe, model, manifest, and evaluation
  under `runs/historical_mask_expansion_regime_stage2_v4/archive/`

The launch receipt must record the committed code identity, normalized recipe,
exact Stage 1 parent/model hash, source/cache identities, recovery identity,
seed, and metric implementation. Logs belong on the local system volume when
the long run is supervised by `launchd`.

## 12. G0/G1 verdict

- G0: `PASS` — the decision, two-boundary repair, temporal roles, falsifier, mechanism
  evidence, economics, and artifact ownership are fixed.
- G1: `PASS` — the issue-reproduction tests fail on the old semantics and pass
  on v4 semantics; all 472 repository tests pass in the real host context,
  including both MPS smokes, and config validation plus the exact-parent
  contract check pass. Launch remains conditional on committing and pushing
  this exact revision to clean `main`.

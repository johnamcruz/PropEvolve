# PropEvolve teacher curriculum stages

## Status and purpose

This document freezes the intended curriculum after the balanced post-launch
Entry v8b result. Stage 1 is now an immutable archived parent; later stages are
research plans until their own teacher-free evaluation receipts pass.

The curriculum builds one teacher-free policy incrementally:

```text
Stage 1: Expansion Entry foundation
    -> Stage 2A: Regime-aware chop abstention
    -> Stage 2B: Regime-aware deficit recovery
    -> Stage 3: Long/Short Trend confluence
    -> Stage 4: challenge completion and pass conversion
    -> Stage 5: 5M multi-seed confirmation
    -> Stage 6: sealed 2026 confirmation and handoff
```

Every stage warm-starts from an immutable selected parent, introduces one
declared learning mechanism, and is evaluated without teachers. A child may
advance only when it preserves the parent's demonstrated competence and passes
its own incremental gate.

## Frozen contract shared by every stage

- One recurrent distributional Double-DQN policy is the final decision-maker.
- The policy observes causal FFM embeddings plus causal account and
  position-management state.
- Training covers the declared 2021-2024 roles across nine markets.
- Chronological 2025 NQ is inner selection and is never folded back into a
  revised training decision without a new experiment identity.
- 2026 remains sealed until the final recipe, checkpoint, and thresholds are
  frozen.
- The combine target is +$6,000 before the 30-session deadline or the -$3,000
  loss boundary.
- Initial risk remains $300 per trade, including fees, until a separately
  validated Pivot-edge risk experiment replaces it.
- Position size remains one. Ratchet activation, giveback, and lock mechanics
  remain frozen unless a stage explicitly tests winner management.
- Expansion, Regime, Trend, and Entry labels are training-only scaffolding.
  They are never policy observations, execution gates, or validation inputs.
- Teacher influence decays, is randomly hidden, reaches a declared autonomous
  tail, and is removed before checkpoint selection.
- Greedy validation uses epsilon zero. Exploration-supported passes are not
  competence evidence.
- Zero observed selection blowouts is a hard gate at every stage.
- Every candidate, parent, recipe, code revision, source manifest, checkpoint,
  and evaluator receipt is content-addressed and immutable.
- A failed child never overwrites its parent. Rollback always returns to the
  last selected teacher-free candidate.

## Stage 1 - Expansion Entry foundation

### Decision learned

Given causal market and account state, choose WAIT, Long, or Short during the
five sequential decision bars associated with a rapid Expansion opportunity,
then manage the position to preserve asymmetric winners.

### Training scaffolding

- Temporary Expansion and Regime semantic teachers.
- Authenticated bar-1-through-bar-5 WAIT/Long/Short action supervision.
- Inverse-frequency class balance derived from the exact pre-2025 target
  population so WAIT cannot erase the rarer directional examples.
- RL challenge economics remain authoritative.

### Current evidence

The archived v8b candidate eliminated universal WAIT and traded autonomously
in teacher-free chronological 2025 NQ selection. Its completed 200-episode
receipt recorded 46 passes, zero blowouts, and 154 timeouts: 23.0% pass rate,
33.4% trade win rate, 1.952R average winners, 0.736 two-R MFE capture, and
+$19.03 mean terminal PnL. Its first failed boundary was 98 near-blow timeouts,
or 63.6% of all timeouts.

The exact selected candidate, evaluation, and model identities are the Stage 2
warm-start contract. The checked-in v8b template is not a substitute for that
selected recipe because the selected child also carries its revised large-win
and management-exploration settings.

### Preserve

- Bar-1-through-bar-5 action semantics.
- Authenticated class-balance receipt.
- Long, Short, and WAIT capability.
- Current winner-retention and ratchet behavior.
- Exact selected checkpoint as the rollback parent.

## Stage 2 - Regime-aware safety and deficit recovery

Stage 2 has two ordered sub-stages. Stage 2A must demonstrate safer Entry
selection before Stage 2B teaches recovery near the loss floor. They must not
be bundled into one run because avoidance and recovery are different causal
learning problems.

### Stage 2A - Chop abstention

#### Hypothesis

Temporary soft Regime supervision can teach the policy to WAIT when chop risk
is high, reducing weak entries and near-blow timeouts without suppressing the
Expansion entries and large winners learned in Stage 1.

#### One allowed learning change

Add an explicit training-only Regime-to-action auxiliary objective that
increases WAIT preference smoothly with causal chop probability. Supportive
Regime context may reduce the abstention pressure but never forces Long or
Short. There is no hard Regime threshold or inference-time Regime gate.

The exact objective, coefficient, schedule, and pre-2025 teacher lineage must
be frozen in a new JSON recipe and covered by deterministic Long, Short, WAIT,
teacher-hiding, and teacher-removal tests before compute.

#### Matched comparison

Warm-start from the immutable Stage 1 candidate. Freeze all data, labels,
splits, risk, fees, Entry supervision, agent architecture, optimizer settings,
winner management, and ordinary evaluation economics. The Stage 2 code
revision adds the declared Regime-abstention mechanism plus generic masked
short-episode replay support required by the later recovery stage; both are
content-addressed and tested so they are inert on ordinary Stage 1 traces.
Change only the activated Regime objective and fresh experiment identity in
Stage 2A. Treat the archived Stage 1 result as the competence floor, not as a
claim that it was evaluated by byte-identical source code.

#### Required diagnostics

- authenticated pre-2025 positive-Entry rows stratified by dominant-chop versus
  non-chop context and low versus safe MLL headroom;
- mean soft-WAIT target and supervised-row count within each of those
  training-only strata;
- near-blow timeout rate;
- trade count, win rate, average winner R, expectancy, MFE capture, and pass
  rate;
- teacher-free greedy entry rate after the autonomous tail.

#### Advance gate

- zero teacher-free selection blowouts;
- no universal-WAIT or all-ENTER collapse;
- lower near-blow timeout rate than Stage 1;
- pass rate, expectancy, average winner R, and 2R MFE capture do not regress
  against Stage 1 beyond a predeclared tolerance;
- dominant-chop and low-headroom positive-Entry rows receive stronger soft-WAIT
  pressure than their matched non-chop and safe-headroom training rows;
- all teachers are absent from the serialized candidate and evaluator.

Failure preserves Stage 1 unchanged and revises only the first failed Regime
learning boundary.

### Stage 2B - Deficit recovery

#### Hypothesis

After the policy learns to avoid weak chop entries, a bounded one-shot recovery
curriculum can teach it to recover from exactly -$2,700 without increasing
ordinary-selection blowouts. Recovery should come from one selective Expansion
entry under learned Regime context, not from higher trade frequency or relaxed
risk limits.

#### One allowed learning change

Warm-start from the selected Stage 2A candidate and mix ordinary full-challenge
episodes with an authenticated, complete -$2,700 recovery snapshot. That
snapshot is flat with realized and equity PnL both -$2,700, peak equity $0, MLL
floor -$3,000, session PnL -$2,700, no passmark lock, and exactly $300 of MLL
headroom. It carries one challenge-lifetime Entry permit. WAIT does not consume
the permit, but the first Long or Short Entry does.

The recovery episode terminates when that first trade closes. Crossing to
-$2,500 restores the ordinary $500 Entry headroom and is `recovery_success`.
Closing below -$2,500 without touching the MLL floor is
`survived_not_recovered`. A full fee-inclusive -1R stop reaches -$3,000 and is a
blowout. If the agent never finds a valid opportunity, it may WAIT until the
ordinary challenge horizon and receives `wait_timeout` rather than artificial
credit for inactivity.

Position size remains one, risk remains $300 including fees, and the ordinary
$500 minimum-headroom guard remains unchanged. The permit is a recovery-only
training and stress-evaluation seam; it does not lower the ordinary guard or
create a second trade. The initial matched experiment uses this single exact
start rather than a ladder of starting balances so its evidence is attributable.

#### Matched comparison

Freeze the Stage 2A Regime objective and every Stage 1 Entry and management
contract. Change only the deterministic mixture of ordinary and -$2,700
recovery episodes under a fresh experiment identity. Evaluate both the
ordinary chronological 2025 NQ challenge distribution and a separately
declared teacher-free -$2,700 recovery stress set. Neither path may touch 2026.

#### Required diagnostics

- recovery-success, survived-not-recovered, wait-timeout, and blow rates from
  exactly -$2,700;
- mean terminal PnL and mean WAIT decisions before the one permitted Entry;
- recovery Entry count and one-Entry contract violations;
- training-only low-headroom selectivity targets and Long/Short action balance;
- ordinary zero-balance challenge performance to detect curriculum regression.

#### Advance gate

- zero blowouts on ordinary teacher-free chronological selection;
- zero one-Entry contract violations;
- recovery-stress blow rate does not worsen against the Stage 2A parent;
- recovery-success rate or mean terminal PnL improves against Stage 2A;
- the policy actually uses the recovery Entry permit on some episodes;
- near-blow timeout rate decreases without an increase in trade frequency being
  the sole mechanism; specifically, the child greedy-entry rate may exceed its
  selected parent by at most two percentage points;
- Stage 2A chop abstention, Stage 1 pass rate, and winner R do not materially
  regress;
- no risk, MLL, position-size, or execution guard is relaxed.

Failure preserves Stage 2A unchanged. If the policy learns reckless loss-floor
gambling, the recovery curriculum is falsified even when some episodes pass.

## Stage 3 - Long/Short Trend confluence

### Hypothesis

Temporary directional Trend-readiness supervision can help the policy select
the correct side, access more sustained moves, and retain 2R+ winners after
Expansion timing and chop abstention are already competent.

### One allowed learning change

Add the authenticated Long and Short Trend-readiness teacher as a soft
training-only auxiliary objective. It provides directional and persistence
context; it is not an entry signal, hard filter, or deployed dependency.

### Matched comparison

Warm-start from the selected Stage 2 candidate. Freeze Stage 1 Entry timing,
Stage 2 Regime abstention, risk, management, data, splits, and evaluation.
Change only the declared Trend auxiliary mechanism and fresh experiment
identity.

### Required diagnostics

- Long and Short performance separately;
- entry direction agreement by Trend-readiness bucket;
- average winner R, MFE, realized-MFE gap, hold duration, and ratchet capture;
- pass and timeout rates by market during training;
- teacher-free greedy side balance and temporal selection economics.

### Advance gate

- zero teacher-free selection blowouts;
- Stage 2 chop-abstention behavior is retained;
- both Long and Short remain active and economically credible;
- average winner R, 2R MFE capture, expectancy, or pass rate improves without
  a material regression in the other parent metrics;
- teacher removal, serialization, and greedy-action parity pass.

Failure preserves Stage 2 unchanged. Do not compensate by loosening safety or
adding another teacher in the same experiment.

## Stage 4 - Challenge completion and pass conversion

### Hypothesis

After Entry, Regime safety, deficit recovery, direction, and winner retention
are competent, a bounded completion curriculum can convert safe positive or
near-target timeouts into faster passes.

### Allowed focus

- warm-start the selected Stage 3 candidate;
- use a two-million-step confirmation budget;
- improve terminal credit, lead-giveback behavior, and pass-speed credit only
  through declared JSON revisions;
- preserve one-contract sizing, $300 initial risk, teacher-free deployment,
  and zero-blow selection.

### Advance gate

- zero teacher-free selection blowouts;
- near-blow timeout rate at or below 5%;
- pass rate at or above 25%;
- positive expectancy;
- average winner at or above 1.5R and no regression from the selected parent;
- improvement is stable across chronological slices rather than concentrated
  in one episode block.

Failure returns to the selected Stage 3 parent. Completion shaping must not
create WAIT collapse, excessive trading, compressed winner R, or loss-floor
gambling.

## Stage 5 - Five-million-step multi-seed confirmation

### Purpose

Estimate recipe stability after all learning choices are frozen. This stage
does not search or revise.

### Execution

- train eight declared seeds for five million environment steps each;
- run at most three seeds concurrently;
- use the same frozen recipe, temporal roles, costs, risk, and evaluator;
- preserve every seed checkpoint and receipt;
- rank zero blowouts first, followed by near-blow safety, 2R retention, pass
  rate, and expectancy.

### Confirmation gate

- all required seeds complete with no integrity failure;
- zero teacher-free selection blowouts;
- near-blow timeout rate at or below 5%;
- pass rate at or above 50%;
- trade win rate at or above 40%;
- average winner at or above 2R;
- positive expectancy;
- 2R MFE capture at or above 50%;
- balanced Long/Short and temporal evidence.

No failed confirmation result may trigger tuning on the same inspected rows.
Return to a declared research stage with a new experiment identity.

## Stage 6 - Sealed 2026 confirmation and handoff

### Purpose

Test the frozen champion once on untouched 2026 data, then package it for
future integration only if every gate passes.

### Rules

- no teacher caches, targets, calibration fitting, or recipe changes may use
  2026;
- spend the sealed period once after model, policy, thresholds, and seed
  selection are frozen;
- reproduce research, serialized-model, execution, fee, MLL, stop, ratchet,
  pass, blow, and termination parity on a golden temporal slice;
- retain the prior selected champion as the rollback target;
- a sealed failure is evidence, not permission to tune on 2026.

The generated PropEvolve policy remains a research artifact until sealed
evidence and integration parity authorize a separate production handoff.

## Campaign implementation note

The v8b JSON and archive remain unchanged. The fresh
`historical_mask_expansion_regime_stage2_selectivity_recovery_v1.json` recipe
authenticates the selected v8b candidate as its external parent and executes
only `regime_selectivity_1m -> deficit_recovery_1m`. It has fresh output and
campaign-state roots, no finalization stage, and no sealed evaluation step.
Stage 3 and later work require a new recipe after Stage 2 evidence is accepted.

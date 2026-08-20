# Immediate-Red Recovery Activation for PropEvolve

Status: read-only research decision. No code, configuration, process, run, or
launchd change was made.

Date: 2026-08-20

Audited snapshots:

- algoTraderAI: `7048522f6ed93849d19ef77726d7c1dc94788aeb`
- PropEvolve: `925d5781b36f9c1a699c6fe31754c96a191b6dee`

## Decision

PropEvolve should not switch to a second recovery model when P&L turns red.
It should keep the single frozen-V21-derived C51 policy and make **every
negative-P&L state eligible for training-only recovery-value supervision**.
At inference, the same policy continuously conditions on its existing realized
P&L, equity, MLL headroom, and drawdown observations. There is no teacher,
router, mode bit, thresholded action gate, or additional model.

In other words, `realized_pnl < 0` should define when recovery examples are
created and reported during training. It must not define which actions are
allowed at inference.

## What algoTraderAI actually does

### It does not dynamically route at breakeven

The current production strategy is explicitly single-arm and bundle-driven.
`_route()` always returns the one loaded model; old
`recovery_switch_threshold` arguments are ignored. Operationally changing to a
different bundle requires restarting the bot, not crossing an account-balance
threshold.

Evidence:

- `/Volumes/CRUZ SSD/algoTraderAI/strategies/strategy_mantis.py:511-516`
- `/Volumes/CRUZ SSD/algoTraderAI/strategies/strategy_mantis.py:571-577`
- `/Volumes/CRUZ SSD/algoTraderAI/strategies/strategy_mantis.py:823-835`

### Recovery learning comes from continuous account state and sampled starts

Training samples each episode's starting cushion uniformly from
`U[$500,$3,000]`. With a `$3,000` MLL, that is equivalent to training across
challenge P&L from `-$2,500` through `$0`. The same PPO policy sees normalized
balance, equity, drawdown-to-peak, distance to MLL, session P&L, time remaining,
and pass progress. There is no learned recovery on/off observation.

Evidence:

- `/Volumes/CRUZ SSD/algoTraderAI/configs/sweep/combine_v11_pivot.json:31-37`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:900-930`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:2077-2093`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:2148-2174`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/tests/test_combine_rules.py:337-355`

### Its immediate-red behavior is mostly hard enforcement

The active recipe sets `deficit_threshold = $3,000`. The implementation compares
this threshold to remaining MLL headroom, so any headroom below `$3,000`—any
loss from a fresh `$3,000` cushion—activates a hard setup-probability floor of
`0.90` and a one-contract cap. It also applies a 50% headroom risk cap, a minimum
headroom-to-risk ratio, stop-cost limits, daily-loss controls, and cooldowns.
This is execution enforcement, not proof that PPO learned recovery.

The lower-balance reward modifier is separate and starts only below `$2,500`
headroom, equivalent to worse than `-$500` P&L. The active recipe leaves its
breakeven milestone reward at zero. The environment supports the milestone only
for deficit-start episodes and continues the episode after breakeven.

Evidence:

- `/Volumes/CRUZ SSD/algoTraderAI/configs/sweep/combine_v11_pivot.json:86-147`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:357-363`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:384-416`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:454-472`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:1089-1102`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:1286-1309`
- `/Volumes/CRUZ SSD/algoTraderAI/rl/environments/prop_firm.py:1609-1650`
- `/Volumes/CRUZ SSD/algoTraderAI/strategies/strategy_mantis.py:837-939`

algoTraderAI separately measures the policy's raw deficit actions before
environment enforcement. That separation is important because safe executed
actions can otherwise conceal a policy that still requests bad entries.

Evidence:

- `/Volumes/CRUZ SSD/algoTraderAI/scripts/diagnose_rl.py:38-56`
- `/Volumes/CRUZ SSD/algoTraderAI/scripts/diagnose_rl.py:250-266`
- `/Volumes/CRUZ SSD/algoTraderAI/scripts/diagnose_rl.py:391-419`

## Current PropEvolve V22 boundary

PropEvolve already has the inputs required for continuous learned recovery. The
policy observation includes normalized realized P&L, equity, peak equity, MLL
headroom, drawdown, session/challenge time, and position state.

Evidence:

- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/observation.py:81-100`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/observation.py:103-159`

V22 currently differs from the desired immediate-red contract in two ways:

1. Every recovery-enabled main training episode is reset at exactly `-$2,000`.
   The recovery target is constructed only from that reset state once per
   configured episode. This teaches deep-red recovery, not the full path from
   the first negative dollar through `-$2,000`.
2. The environment's `recovery_active` state exists only because a special
   `ChallengeStartState` was supplied. An ordinary V21 episode that naturally
   crosses below `$0` does not dynamically acquire that flag. The policy still
   observes the negative account state, but the Stage 2B full-action supervisor
   is not automatically created there.

Evidence:

- `/Volumes/CRUZ SSD/PropEvolve/config/historical_mask_expansion_anchored_regime_stage2_v22_recovery_200ep.json:317-340`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/training.py:4772-4826`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/environment.py:473-510`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/environment.py:582-601`

The existing `ActionMasker` uses `recovery_active` only to keep native actions
available below the ordinary minimum-headroom boundary. That is not the learned
activation mechanism and should not be expanded into a recovery router or
selection gate.

Evidence:

- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/decision.py:26-62`

## P0: r8 cannot produce interpretable training evidence

At commit `925d578`, the V22 configuration and config validator require a
`-$2,000` start with `$1,000` MLL headroom. However,
`HistoricalChallengeEnv._validate_challenge_start_state()` still requires start
headroom to equal `per_trade_risk_dollars`, which remains `$300`.

Therefore the first environment reset must compare `$1,000 == $300` and raise
`ValueError("recovery headroom must equal per-trade risk")`. The r8 log inspected
during this audit was still preprocessing entry supervision and had not produced
an episode; stderr was empty only because it had not reached that reset.

Evidence:

- `/Volumes/CRUZ SSD/PropEvolve/config/historical_mask_expansion_anchored_regime_stage2_v22_recovery_200ep.json:150-170`
- `/Volumes/CRUZ SSD/PropEvolve/config/historical_mask_expansion_anchored_regime_stage2_v22_recovery_200ep.json:317-337`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/config.py:523-549`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/training.py:2289-2297`
- `/Volumes/CRUZ SSD/PropEvolve/src/propevolve/environment.py:513-527`

Decision: **stop r8 before further preprocessing**. It is not an economic
failure or a recovery-learning experiment; it is an executable-contract
failure. Its data cannot answer the immediate-red question.

## Smallest falsifiable next design

After repairing the start-state validator, run one matched Stage 2B experiment:

1. Keep the immutable V21 checkpoint, observations, action space, ordinary
   replay composition, A+ losses, optimizer, schedules, rewards, and public
   outcomes unchanged.
2. Keep ordinary main episodes starting at `$0`, exactly as V21, so normal A+
   behavior remains represented throughout fine-tuning.
3. In the separate recovery-value sidecar only, sample realized P&L continuously
   from `(-$2,000, $0)`—equivalently MLL headroom from `($1,000,$3,000)`—and
   build the existing same-state `WAIT/LONG/SHORT` economic target. This adapts
   algoTraderAI's useful start-state coverage while excluding the already
   falsified sub-`$1,000` headroom region.
4. Treat `realized_pnl < 0` as a training-target eligibility and diagnostic
   predicate only. Stop target rollouts at breakeven, blow, or timeout; the real
   episode continues after breakeven.
5. At validation and inference, run the same teacher-free C51 policy
   continuously. No recovery lookup, threshold, mode bit, action gate, or model
   switch is allowed.

This design teaches recovery immediately after the account becomes red while
also teaching progressively more conservative choices as headroom shrinks. The
same network learns that behavior from continuous account state; no artificial
activation boundary is needed at inference.

### Decisive TDD and evaluation proof

- At a flat state with `realized_pnl = -$1`, recovery supervision is eligible
  and all native `WAIT/LONG/SHORT` actions remain available.
- At `realized_pnl = $0`, the recovery loss is exactly zero and the V21 update,
  sampled ordinary sequence IDs, loss components, and gradients match the
  frozen fixture.
- Red states at fixed evaluation anchors `-$250`, `-$1,000`, and `-$2,000`
  each report supervisor-policy concurrence, requested-action Q margins,
  recovery rate, mean terminal P&L, and blow rate.
- Executed-action safety is never credited as learning; requested actions must
  improve.
- Teacher-free validation performs zero teacher or recovery-target lookups.
- Reject if any anchor blows, if recovery competence does not improve over V21,
  or if ordinary V21 A+ pass/winner competence regresses.

## Bottom line

algoTraderAI's useful lesson is not “switch recovery on at a threshold.” Its
useful lesson is to expose continuous account state and deliberately train over
the full red-account region. Its current immediate-red safety is substantially
hard-gated and must not be copied. PropEvolve should activate **training-only
economic recovery supervision** at every negative-P&L state and let the one
existing policy express that learned behavior continuously at inference.

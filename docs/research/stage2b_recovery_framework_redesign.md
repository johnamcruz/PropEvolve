# Stage 2B Recovery Framework Redesign

Status: research decision; no implementation, configuration, run, checkpoint, or launch change is authorized by this document.

Date: 2026-08-20

## Implementation amendment

The V22 campaign uses the frozen recovery start state for **every main training
episode**: realized and equity P&L begin at `-$2,700`, the MLL floor remains
`-$3,000`, recovery is reached at `$0`, and the same episode then continues
toward the ordinary `+$6,000` pass target. The recovery-disabled V21 path still
uses its ordinary start state. This amendment supersedes later wording that
describes V22 main episodes as ordinary-start episodes; it does not change the
V21 model, optimizer, replay sampler, A+ losses, public outcomes, or
teacher-free inference contract.

## Decision

Stage 2B should be rebuilt as one **additive, training-only, low-headroom full-action recovery-value objective** on top of the immutable V21 policy.

At a low-headroom state, the training supervisor must estimate the recovery value of all three native actions—`WAIT`, `ENTER_LONG`, and `ENTER_SHORT`—under the same causal prefix and economics. The existing V21 C51 policy is then taught to rank those three actions by recovery value. No new model or inference input is added. No teacher, target table, score threshold, action mask, or recovery gate exists at validation or inference.

This is the smallest mechanism that directly answers the required question:

> At this low-headroom state, which of WAIT, LONG, or SHORT has the best probability-adjusted economic path back to breakeven without blowing?

Everything that already teaches ordinary A+ entry selection remains exactly V21.

## Non-negotiable V21 boundary

The following are immutable:

- the frozen V21 checkpoint and checkpoint loader;
- policy observations and native `WAIT`/`LONG`/`SHORT` action space;
- ordinary 16-sequence replay composition and sequence identities;
- A+ Long/Short/WAIT supervision, paired population correction, losses, and margins;
- optimizer, learning-rate schedule, teacher schedule, and health gates;
- ordinary environment rewards and transitions;
- public episode outcomes: `pass`, `blow`, or `timeout`;
- teacher-free validation and inference.

Recovery has one orthogonal result field: `recovered` or `not_recovered`. Reaching breakeven does not terminate or rename the ordinary episode. The episode continues toward its normal `pass`, `blow`, or `timeout` outcome.

## Root cause of the rejected V22 design

The first V22 implementation did not merely add recovery learning. It competed with V21 learning inside the same replay batch.

V21 allocates its 16-sequence batch as eight terminal, four safety, and four entry-opportunity sequences. V22 added a `recovery_sequence_fraction` of 0.25 and reserved four of those same 16 positions for recovery episodes. Its sampler then split the entry anchors between ordinary and recovery cohorts and required winner/failure pairs within the recovery cohort. The practical change was:

| Exposure | V21 | Rejected V22 | Change |
|---|---:|---:|---:|
| Ordinary sequences per batch | 16 | 12 | -25% |
| Ordinary A+ entry sequences | 4 | 2 | -50% |
| Recovery sequences | 0 | 4 | Added by replacement |

That is why V22 could weaken the learned A+ behavior even though it loaded V21. At early recovery replay depth, valid same-cohort winner/failure pairs could also be absent, so the newly reserved slots did not guarantee useful recovery credit.

V22 additionally altered ordinary behavior by making the MLL-proximity penalty conditional on exposure and by changing campaign gates. Those are pipeline changes, not recovery-only teaching. They violate the requirement that V21 remain the control.

Evidence: the V21 baseline is commit [`827c492`](https://github.com/johnamcruz/PropEvolve/commit/827c492202de6a3ba2d3cedfb1b3dbe088c2ca03); the rejected Stage 2B addition is commit [`8244721`](https://github.com/johnamcruz/PropEvolve/commit/8244721ae6e438dad59d5fb530bee92f9c03d625). The latter changes replay, environment, decision, agent, training, configuration, and gates together, rather than adding one isolated recovery objective.

## What AlgoTraderAI actually teaches

AlgoTraderAI is the closest operational recovery reference, but its active solution is a hybrid of learned policy behavior and hard execution controls. Those two mechanisms must not be conflated.

### Exact active snapshot

The audited active recipe is `configs/sweep/combine_v11_pivot.json` at commit `7048522`:

| Boundary | Active behavior |
|---|---|
| Parent/training | Warm-starts a proven trial; PPO trains for 2,000,000 steps at learning rate `5e-5` on NQ/ES/GC/RTY/YM |
| Start-state curriculum | Samples starting cushion continuously from `U[$500, $3,000]`; each episode is 30 days |
| Policy account inputs | Normalized balance, equity, drawdown/headroom, session PnL, time/progress, position, and market/setup context |
| Training entry pressure | `risk_penalty` mode with coefficient `3.0` and margin `0.35`; it penalizes deficient requested entries but does not positively reward clean entries |
| Breakeven milestone | The environment supports a one-time milestone, but the active recipe sets `term_breakeven_reward = 0.0`; active recovery learning therefore does **not** depend on that bonus |
| Hard recovery envelope | Deficit proba floor `0.90`, deficit max size `1`, balance safety fraction `0.5`, minimum balance/risk ratio `1.5`, max stop cost `$600`, soft/hard daily loss `$750/$1,500`, loss-streak cooldown `2/65` |
| Public accounting | Episodes retain normal terminal outcomes; breakeven milestone, when enabled elsewhere, is internal and the episode continues |
| Evaluation boundary | 200 episodes at fixed `$3,000` cushion over 2021–2025, three seeds, fail-fast on blow; the file explicitly labels this in-sample selection and leaves 2026 held out |
| Diagnostics | Separately measures the raw policy's requested deficit-entry quality before environment enforcement |

This matters: the active recipe's fixed-$3,000 evaluation is not direct evidence of low-headroom recovery, and much of its active zero-blow envelope is hard enforcement. Its transferable lesson is the state-dependent curriculum and policy-first diagnostic seam—not its recovery result as a matched PropEvolve benchmark.

### Learned mechanisms worth transferring

1. **Account state is observable.** The policy observes normalized balance, drawdown/headroom, session PnL, time, position, and market evidence. PropEvolve already has the equivalent account-state seam; it does not need new observations.
2. **Recovery starts are explicit.** The active combine config samples recovery starts over a configured balance interval. This makes low-headroom states common enough to learn from rather than waiting for rare accidental drawdowns.
3. **Breakeven is a training milestone, not a public episode outcome.** AlgoTraderAI can award a one-time breakeven reward and then continue the episode. This maps cleanly to `recovered`/`not_recovered` plus unchanged `pass`/`blow`/`timeout`.
4. **The policy receives soft training pressure against poor entry context.** Its active `risk_penalty` mode penalizes an entry whose setup evidence is below its context margin. This is training pressure, not itself an inference mask.
5. **Diagnostics separate requested behavior from enforced behavior.** Its deficit diagnostics measure the policy's requested A+ preference before environment controls. That is essential proof that the policy learned recovery selectivity.

Primary local source at audited commit [`7048522`](https://github.com/johnamcruz/algoTraderAI/tree/7048522f6ed93849d19ef77726d7c1dc94788aeb): [active combine configuration](https://github.com/johnamcruz/algoTraderAI/blob/7048522f6ed93849d19ef77726d7c1dc94788aeb/configs/sweep/combine_v11_pivot.json), [start-state and environment implementation](https://github.com/johnamcruz/algoTraderAI/blob/7048522f6ed93849d19ef77726d7c1dc94788aeb/rl/environments/prop_firm.py), and [policy-first diagnostics](https://github.com/johnamcruz/algoTraderAI/blob/7048522f6ed93849d19ef77726d7c1dc94788aeb/scripts/diagnose_rl.py).

### Hard-gated mechanisms that must not transfer

AlgoTraderAI also blocks or clamps actions through:

- a deficit setup-probability floor;
- a soft-daily-loss entry block and cooldown;
- ADX/setup gates;
- deficit size caps;
- minimum balance-to-risk and maximum stop-cost checks;
- pyramid eligibility rules.

These can protect the account, but they do not prove the policy learned `WAIT`. The clearest example is `soft_daily_loss_blocks_entry`, which rejects a non-A+ requested entry before execution ([source](https://github.com/johnamcruz/algoTraderAI/blob/7048522f6ed93849d19ef77726d7c1dc94788aeb/utils/risk_policy.py#L21-L38)). PropEvolve must not import the Pivot probability, ADX, or other hand-coded inference gates. It must teach recovery economically and verify the raw requested action.

### Exact transfer boundary

| AlgoTraderAI mechanism | Learned or enforced? | PropEvolve decision |
|---|---|---|
| Balance/headroom observations | Learned input | Already present; keep unchanged |
| Low-headroom training starts | Curriculum | Use only in a separate recovery training stream |
| One-time breakeven credit | Training reward/credit | Transfer as `recovered`; episode continues |
| Entry-context penalty | Training-only learning signal | Replace fixed proba margin with full economic action values |
| Requested-vs-executed diagnostics | Proof seam | Transfer |
| Deficit proba/ADX thresholds | Hard gate | Do not copy |
| Daily-loss/cooldown blocks | Hard gate | Do not copy as evidence of learning |
| Size, stop-cost, pyramid clamps | Execution/risk envelope | Do not add in this Stage 2B experiment |
| PPO architecture and warm-start machinery | Different RL stack | Do not copy |

## What EarnHFT contributes

EarnHFT provides the more relevant *teaching form*. During training it constructs a future-informed value vector for every action available in the same state. The learner is trained with ordinary TD error plus a distributional matching loss between its action values and the supervisor's complete action-value vector. The testing environment does not expose that training Q table.

In the official implementation:

- [`demonstration.py`](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/tool/demonstration.py#L191-L288) computes recursive economic values for each current action;
- [`low_level_env.py`](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/env/low_level_env.py#L276-L331) exposes the value vector in the training environment;
- [`ddqn_pes_risk_aware.py`](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L307-L350) combines TD loss with KL matching to the supervisor distribution.

The useful idea is not EarnHFT's crypto action grid, hierarchy, or feed-forward DDQN. It is this: **teach the relative economic value of every action in the same state, then remove the supervisor at inference**. This is stronger recovery credit than labeling only the action that happened.

The [EarnHFT paper](https://arxiv.org/abs/2309.12891) motivates the full-action Q supervisor, but PropEvolve must compute it with its own next-bar fills, costs, stops, MLL rules, recurrent state, and 30-day recovery horizon.

## Recommended recovery-only objective

### 1. Recovery training states

Use the exact frozen Stage 2B start state already agreed for the first falsifying experiment: realized PnL of `-$2,700`, $300 above the `-$3,000` MLL floor. This is an experimental start distribution, not an inference rule and not a gate.

Recovery target rollouts use authenticated training windows only. Validation and sealed rows may never be used to construct targets.

### 2. Three counterfactual actions from one causal state

For every eligible low-headroom training state `s`, clone the exact simulator and recurrent prefix three times:

- branch 1 forces `WAIT` for the first decision;
- branch 2 forces `ENTER_LONG` for the first decision;
- branch 3 forces `ENTER_SHORT` for the first decision.

After that first action, every branch follows the same frozen V21 policy, teacher-free, with identical remaining market path and immutable execution economics. A branch ends for target construction at the first of:

- realized PnL reaches `0`: `recovered`;
- the account reaches MLL: `blow`;
- the ordinary 30-day horizon ends: `not_recovered`.

The actual campaign episode is not ended or renamed when it recovers. This early stop exists only while constructing the training target.

### 3. Recovery value

Define a bounded economic utility for each forced first action:

```text
G_rec(s, a) = +1                                      if recovered
              -1                                      if blow
              clip((terminal_pnl - start_pnl)
                   / (0 - start_pnl), -1, +1)         if not_recovered
```

For the frozen first experiment, `start_pnl = -2700`.

This produces the required credit without inventing an A+ score:

- a productive Long or Short that reaches breakeven ranks above WAIT;
- a trade that makes money briefly but later blows receives `-1`;
- WAIT ranks highest when both entries damage headroom;
- a timeout that preserves or improves headroom ranks above one that regresses;
- a slow recovery still counts as recovery; the ordinary V21 rewards and fixed horizon already encode time and opportunity cost.

All three branches share the same causal state, remaining market path, costs, and account state. The comparison is therefore apples-to-apples.

### 4. Distill the action ranking into the existing policy

Let `Q_theta(s)` be the existing V21 C51 expected values for `WAIT`, `LONG`, and `SHORT`. Convert both vectors to distributions with one frozen temperature `tau`:

```text
p_rec(a | s)   = softmax(G_rec(s, a) / tau)
p_theta(a | s) = softmax(Q_theta(s, a) / tau)

L_recovery = KL(p_rec || p_theta)
```

Apply `L_recovery` only to recovery-learning rows. It does not replace TD learning or any V21 A+ loss. It adds no head. It does not encode Expansion, Regime, ticker, or direction thresholds. The policy must infer the winning pattern from its existing causal market embeddings plus account state.

### 5. Separate recurrent recovery updates

Recovery experience must live in a separate recovery store. It must never consume, replace, rebalance, or relabel an ordinary V21 replay position.

Each recovery sequence must preserve the R2D2 invariants already expected by PropEvolve: episode-bounded sequences, recurrent burn-in, correct reset masks, and loss only after hidden-state reconstruction. The [R2D2 paper](https://openreview.net/pdf?id=r1lyTjAqYX) explains why replayed recurrent state and sequence boundaries matter. [R2D3](https://openreview.net/pdf?id=SygKyeHKDH) shows the value of keeping rare guided experience in a distinct replay population, but its stochastic demonstration replacement ratio must **not** be copied here because replacement is precisely what broke V22. Acme's [replay adders and tables](https://github.com/google-deepmind/acme/blob/master/docs/user/components.md) support a separate-sequence-table design.

The MVP uses at most one complete three-action target beside each ordinary V21 update. Its mean KL term is added to the exact V21 ordinary loss before the **same single backward/optimizer step**. If no valid target is available, the recovery term is exactly zero. Thus every ordinary batch, optimizer step count, scheduler advancement, and teacher-schedule advancement remains V21; only the declared recovery gradient is additive. No recovery slot is reserved in the ordinary batch and no second optimizer step is introduced.

## Why this teaches the desired behavior

| Low-headroom evidence and eventual economics | Full-action target |
|---|---|
| Long reaches breakeven; Short loses/blows; WAIT stalls | Rank LONG first |
| Short reaches breakeven; Long loses/blows; WAIT stalls | Rank SHORT first |
| Both sides damage headroom; WAIT preserves it | Rank WAIT first |
| Both sides conflict in chop and later lose | Rank WAIT first |
| Entry gains early but eventually blows | Rank it last, not as A+ |
| Entry does not recover but improves terminal PnL | Rank by bounded recovery progress |
| Existing V21 A+ setup also recovers safely | Reinforce that native action without a new setup score |

Expansion and Regime remain causal information available to the policy through the frozen V21 representation and ordinary supervision. The recovery objective says only which action was economically best in the same low-headroom state. It therefore teaches state-dependent selectivity without hard-coding what an A+ setup looks like.

## Compatibility matrix

| Contract | Immutable V21 | Proposed Stage 2B addition | Compatibility requirement |
|---|---|---|---|
| Checkpoint | Exact V21 | Load as initialization | Hash and tensors unchanged before first update |
| Observations | Existing causal market + account state | None added | Exact shape/order/normalization parity |
| Actions | WAIT/LONG/SHORT | Same three forced only for target construction | No new action or mask |
| Ordinary replay | Exact 16-sequence composition | Separate recovery store | Ordinary sample IDs/order byte-identical under same seed |
| A+ supervision | Existing Long/Short/WAIT losses and margins | Untouched | Exact ordinary loss and gradient parity |
| Optimizer/schedule | V21 | Same object/settings and same one step per ordinary update | No extra step, hyperparameter, or schedule change |
| Teacher schedule | V21 | Untouched | Recovery targets are economic training artifacts, not policy observations |
| Environment | V21 transitions/rewards | Counterfactual clones use exact environment | No ordinary reward-shaping change |
| Public outcomes | pass/blow/timeout | Unchanged | No fourth outcome |
| Recovery result | None | recovered/not_recovered | Separate diagnostic only |
| Health gates | V21 | Unchanged | No recovery-specific weakening or short circuit |
| Validation/inference | Teacher-free V21 policy | Exact same path | Zero supervisor/table lookup and zero hard gate |
| Diagnostics | Existing V21 metrics | Add raw recovery ranking evidence | Never credit executed masking as learning |

## Exact TDD seams

Implementation is not authorized until these tests exist and fail for the intended reason.

### A. Golden V21 immutability

1. With recovery disabled, the configuration identity, loaded checkpoint tensors, optimizer state, teacher schedule, health gates, environment rewards, and public outcomes exactly match the pre-Stage-2B V21 fixture.
2. Given the same seed and replay state, the ordinary sampler returns the same 16 sequence IDs in the same order and the same eight terminal/four safety/four entry composition.
3. The ordinary V21 loss scalar, per-component losses, and parameter gradients match the frozen fixture within the existing deterministic tolerance.
4. No recovery row can enter the ordinary replay store; no ordinary row is relabeled by recovery code.

### B. Full-action economic targets

1. Long recovers, Short blows, WAIT times out: `G_long > G_wait > G_short`.
2. Short recovers, Long blows, WAIT times out: `G_short > G_wait > G_long`.
3. Both entries lose while WAIT preserves headroom: `G_wait` is greatest.
4. An entry earns early profit and later blows: its final value is `-1`.
5. Two `not_recovered` branches are ordered monotonically by terminal realized PnL.
6. Fees, next-bar fills, gaps, stops, trailing MLL, and session-boundary balance rules affect all branches exactly as in V21.
7. All three branches share the same causal prefix and market future; target construction refuses validation/sealed rows or a lineage mismatch.

### C. Recurrent correctness

1. Recovery sequences never cross episode boundaries.
2. Burn-in reconstruction produces the same hidden state and Q values as uninterrupted collection for the same causal prefix.
3. Learning loss is applied only after burn-in.
4. Reset masks match collection, checkpoint reload, and teacher-free evaluation.

### D. Learning direction

1. A Long-best target increases `Q(LONG) - max(Q(WAIT), Q(SHORT))` after one controlled recovery update.
2. A Short-best target produces the mirror result.
3. A WAIT-best target increases `Q(WAIT) - max(Q(LONG), Q(SHORT))`.
4. Zero eligible recovery rows yields exactly zero recovery loss and gradient; the ordinary V21 update is unchanged.
5. Recovery gradients are absent from ordinary replay updates.

### E. Lifecycle and inference

1. First touch of realized PnL `>= 0` emits `recovered` once and the actual episode continues.
2. Blow or timeout before breakeven emits `not_recovered`.
3. Every actual episode ends as exactly one of `pass`, `blow`, or `timeout`.
4. Saved checkpoints contain no counterfactual target table, future label, or supervisor object.
5. Teacher-free validation performs zero recovery-target or Expansion/Regime-teacher lookups and applies no recovery action gate.

### F. Diagnostic proof

Report, by MLL headroom bucket, Regime state, Expansion quantile, and side:

- requested WAIT/LONG/SHORT counts before execution handling;
- executed actions separately;
- supervisor top action and policy top action;
- three-action top-1 concurrence and KL;
- Q margins for WAIT-vs-Long and WAIT-vs-Short;
- recovered/not_recovered;
- pass/blow/timeout and near-blow incidence.

The requested-action report is decisive. If executed actions look safe but raw policy requests do not improve, the model did not learn recovery.

## One experiment and one falsifier

Run one matched candidate from the exact V21 checkpoint, using the exact V21 ordinary data, seeds, schedule, gates, and 200-episode teacher-free validation. The only delta is the separate recovery store plus `L_recovery` on exact `-$2,700` recovery starts.

**Primary falsifier:** reject Stage 2B if it does not improve teacher-free `recovered` rate—or, when the baseline has zero recoveries, mean terminal recovery PnL—at **zero recovery blows**, while preserving every unchanged V21 ordinary gate and showing positive lift in raw requested-action agreement with the full-action recovery target.

Fail immediately, before economic interpretation, if any V21 checkpoint, observation, ordinary replay, A+ loss, margin, optimizer, teacher schedule, health gate, reward, or public-outcome identity drifts. Also fail if safety is produced by an inference mask/gate, if validation consults a supervisor, or if low-value/chop requested entries do not decline.

Do not tune ordinary V21 mechanics after failure. A failed result falsifies this one recovery objective at this target-construction seam; it does not authorize another broad V22 redesign.

## Constrained-RL boundary

Constrained Policy Optimization and CVaR methods are useful evidence that safety should be evaluated as an explicit constraint rather than hidden in average reward. However, [CPO](https://proceedings.mlr.press/v70/achiam17a.html) is a policy-gradient replacement, and [risk-constrained RL with CVaR](https://jmlr.org/papers/v18/15-636.html) introduces a different constrained optimization problem. Neither is the MVP for a frozen recurrent C51 policy. PropEvolve should retain exact MLL simulation and a zero-blow acceptance gate; it should not trade blow probability against average reward through a new Lagrangian.

## What not to build

- no new A+ quality head or recovery model;
- no Pivot probability, ADX, Regime, Expansion, or account-balance inference threshold;
- no recovery action mask, one-entry permit, router, or second controller;
- no replacement of V21 replay slots;
- no change to V21 rewards, A+ losses, margins, optimizer, teacher schedule, or health gates;
- no PPO, hierarchical EarnHFT, or constrained-RL migration;
- no random recovery-start range until the exact `-$2,700` MVP is falsified or accepted;
- no status beyond `recovered`/`not_recovered`, and no outcome beyond `pass`/`blow`/`timeout`;
- no claim of learning based only on executed actions.

## Bottom line

AlgoTraderAI proves that account-aware recovery can be trained and that breakeven should be an internal milestone, but most of its safety is enforced by hard gates. EarnHFT supplies the missing teaching pattern: compare the economic value of every action in the same state during training. R2D2 supplies the recurrent integrity requirements.

The correct PropEvolve synthesis is narrow: preserve V21 completely, create same-state WAIT/LONG/SHORT recovery values on training data, distill that ranking into the existing policy through a separate recovery-only recurrent update, and prove the raw teacher-free policy—not an action gate—learned to choose the safest productive A+ recovery action.

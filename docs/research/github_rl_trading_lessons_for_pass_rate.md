# GitHub RL trading lessons for PropEvolve pass rate

## Scope and decision

This is a primary-source, research-only audit. It does not change the active
run. The question is not whether another trading repository should replace
PropEvolve's recurrent C51/Double-DQN policy. It is whether one small,
evidence-backed training mechanism could improve teacher-free challenge pass
rate while preserving zero blows.

The strongest candidate is **EarnHFT-style training-only supervision over the
entire valid action vector**. PropEvolve already has exact action labels,
DQfD-style margins, paired A+ ranking, Expansion + Regime supervision,
recurrent sequence replay, and exact prop-firm economics. What it does not
clearly have is a dense relative target for `WAIT`, `ENTER_LONG_1`, and
`ENTER_SHORT_1` on the same row. That is the smallest material gap exposed by
this review.

The second candidate, reserved for Stage 2B, is one learned drawdown-risk
auxiliary/loss built from PropEvolve's exact headroom trajectory. DeepScalper
and FinRL-Meta support the general ideas of auxiliary risk learning and
incremental drawdown pressure, but neither proves prop-firm MLL control.

No reviewed project offers a better recurrent replay system than
PropEvolve's R2D2-like complete sequences with burn-in. No reviewed result
justifies a policy rewrite, multi-agent router, hard Regime gate, or FinRL
turbulence threshold.

## Current PropEvolve baseline

The existing system already has the important foundation:

- recurrent distributional Double-DQN/C51 action values;
- episode-safe sequence replay with recurrent burn-in;
- exact next-open, costed `+2R-before-1R` Entry labels;
- exact-action and DQfD-style action-margin supervision;
- side-conditioned Expansion + Regime confluence losses;
- paired A+ winner-versus-failure ranking;
- policy-retention anchoring for management actions;
- causal account/execution state and exact pass, blow, MLL, and timeout
  simulation;
- teacher-free validation with no teacher channels in policy observations.

Therefore, repository mechanisms based on transition replay, daily portfolio
allocation, long/cash-only actions, or threshold liquidation are weaker
matches than the current baseline.

## Primary-source audit

| Project | What the source actually does | Transferable lesson | Task mismatch / decision |
|---|---|---|---|
| [EarnHFT](https://github.com/TradeMaster-NTU/EarnHFT/tree/0e1e11a6d9aff70efb1807baa3416429568deb31) | Builds a future-informed action-value table by backward dynamic programming, places the full vector in training transitions, and adds KL distillation to TD loss. Evaluation uses the learned network without the table. | A full relative action-value supervisor can be more informative and sample-efficient than a one-hot action target while remaining teacher-free at inference. | Crypto HFT, long/cash inventory only, LOB execution, and uniform transition replay. Its DP objective is not PropEvolve's challenge objective and cannot be copied. |
| [Qlib / OPDS](https://github.com/microsoft/qlib/tree/79633dd9506ea689e5400dea0197717b5b3d74b7) | Qlib's RL stack primarily targets single-asset order execution. The OPDS paper distills a perfect-information oracle into a common executable policy. | Independently supports the training-only oracle, teacher-free execution boundary. | Forced order completion is not discretionary Entry/WAIT or prop-challenge control. Current Qlib's public OPDS example is PPO-based and does not reproduce the paper's oracle mechanism. |
| [DeepScalper](https://arxiv.org/abs/2201.09058) / [TradeMaster source](https://github.com/TradeMaster-NTU/TradeMaster/tree/1747cc18db3fe2639af12defc80e138c51a625c0) | The paper adds a training-only hindsight bonus for longer-horizon opportunity capture and an auxiliary future-volatility prediction loss; it reports return/Sharpe and stability gains in ablation. | Training-only continuation/risk targets can shape a trading representation without becoming inference inputs. | Its paper uses intraday futures/LOB inputs and a different reward/action problem. Volatility is not adverse excursion or MLL risk; the authors note it did not consistently improve downside-sensitive Sortino. Current TradeMaster's generic DQN omits the paper's risk auxiliary and uses transition replay. |
| [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta/tree/15405db81ef46790d430341166700a61ba51b70d) | A recent MACE environment subtracts a squared penalty only when drawdown reaches a new maximum. Other FinRL environments hard-liquidate above a turbulence threshold. | Penalizing worsening drawdown rather than all negative P&L is a plausible Stage 2B target. | Daily portfolio/market-impact settings do not model PropEvolve's MLL. Turbulence liquidation is a hard gate and conflicts with learned WAIT behavior. |
| [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster/tree/1747cc18db3fe2639af12defc80e138c51a625c0) | Provides standardized tasks, simulators, metrics, and generic PER/transition buffers. Its benchmark found tuned classic RL competitive with or better than several finance-specific algorithms. | Supports matched, all-else-equal evaluation and skepticism toward architecture novelty. | Its generic replay does not preserve recurrent sequences or burn-in. Do not replace PropEvolve replay with it. |
| [MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54) | Trains trend/volatility specialists, conditionally modulates their features, learns a soft mixture of six Q policies, and adds nearest-neighbor episodic-memory regression. | Continuous context modulation is worth remembering if Expansion/Regime supervision is active but fails to transfer. | It is long/cash crypto HFT, uses transition replay, adds multiple policies plus inference-time routing/memory, and is an architecture rewrite. Defer. |

### Source details that matter

EarnHFT computes an action-value table from future prices, execution costs,
and all inventory actions by backward induction
([source](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/tool/demonstration.py#L191-L288)).
Its training environment exposes that vector only as transition metadata
([source](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/env/low_level_env.py#L276-L331)).
The learner minimizes ordinary TD error plus KL divergence between the
network's full action vector and the demonstration vector
([source](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L307-L354)).
Its evaluation action path uses the network without the Q table
([source](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L397-L408)).
The [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/29384)
reports improved training efficiency and profitability, but its market,
inventory, and simulator remain materially different from PropEvolve.

The OPDS paper independently uses perfect-information oracle policy
distillation followed by ordinary executable policy inference
([AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/16083)).
This supports the boundary, not direct transfer of its order-execution loss.

DeepScalper's environment implements a future-movement training reward and
records future volatility
([source](https://github.com/TradeMaster-NTU/TradeMaster/blob/1747cc18db3fe2639af12defc80e138c51a625c0/trademaster/environments/algorithmic_trading/environment.py#L225-L263)).
The current generic DQN learner is ordinary transition-based Q-learning
([source](https://github.com/TradeMaster-NTU/TradeMaster/blob/1747cc18db3fe2639af12defc80e138c51a625c0/trademaster/agents/algorithmic_trading/dqn.py#L112-L157)),
so the paper—not that generic learner—is the evidence for the auxiliary.

FinRL-Meta's MACE reward penalizes only new increases in drawdown
([source](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/15405db81ef46790d430341166700a61ba51b70d/meta/env_market_impact/envs/env_mace_stock_trading.py#L430-L477)).
By contrast, FinRL's turbulence path overwrites policy decisions with forced
liquidation
([source](https://github.com/AI4Finance-Foundation/FinRL/blob/2334a5fe6d30629157f13c3b0319e1637e15e123/finrl/meta/env_stock_trading/env_stocktrading.py#L313-L340)).
The latter is explicitly not the PropEvolve design.

MacroHFT's context adapter directly scales and shifts normalized market
features
([source](https://github.com/ZONG0004/MacroHFT/blob/31e5ef41f93b2aea6e63e6ae2267675c997a8e54/model/net.py#L15-L56)),
but the complete system mixes six specialists
([source](https://github.com/ZONG0004/MacroHFT/blob/31e5ef41f93b2aea6e63e6ae2267675c997a8e54/RL/agent/high_level.py#L196-L244))
and uses uniform transition replay
([source](https://github.com/ZONG0004/MacroHFT/blob/31e5ef41f93b2aea6e63e6ae2267675c997a8e54/RL/util/replay_buffer.py#L49-L58)).
That cost is not justified before the smaller supervisor test.

## Experiment 1: PropEvolve full-action opportunity-value supervisor

### Why this is the first experiment

Current exact-action classification answers which action crossed the frozen
economic boundary. Margins answer whether the selected action outranks an
alternative by a minimum gap. Paired A+ learning answers whether a matched
winner outranks a failure. None necessarily tells the recurrent policy the
relative economic quality of **all three actions on the same causal row**.

EarnHFT's useful idea is not its long-only DP or replay. It is a dense,
training-only vector that makes every valid action a supervised comparison.
For PropEvolve this may improve:

- `WAIT` versus both directions on weak/conflicted rows;
- Long versus Short asymmetry without separate policy architectures;
- ranking of clean A+ opportunities above near-misses;
- sample efficiency while preserving the existing Bellman/C51 objective.

### Original assumptions versus PropEvolve

| EarnHFT assumption | PropEvolve requirement |
|---|---|
| Seconds-level crypto and five-level LOB | Bar-complete futures context with next-bar-open fills |
| Nonnegative inventory grid; bear behavior is selling to cash | Symmetric `WAIT`, `ENTER_LONG_1`, `ENTER_SHORT_1` decisions |
| Backward DP maximizes discounted cash value over a known chunk | Frozen `+2R-before-1R` Entry contract within 150 bars |
| No prop-challenge trailing MLL/pass state | 30-day challenge, costs, stop-first collision, session MLL trail, pass/fail termination |
| Uniform n-step transition replay | Complete recurrent sequences, burn-in, reset masks, episode boundaries |
| Q table is available only during training | Teacher artifact must never enter observations or evaluation |

### Do not begin with full challenge dynamic programming

Exact DP over the 30-day challenge is theoretically possible only after the
state includes market time, position, realized balance, session-boundary MLL,
pass lock, and every future management choice. Continuous balance/headroom
and path-dependent trailing-floor state make the table enormous. Discretizing
them would introduce new assumptions, and optimizing the whole challenge
would silently change the current Entry-label contract.

That is overkill for the first test. It also risks teaching an oracle policy
for the historical path rather than the generic Expansion + Regime Entry
signature.

### Smallest valid target semantics

Build a training-only **opportunity-value vector**, not a purported full
challenge Q function. Reuse the authenticated Entry-label engine and its exact
next-open, fees, one-contract `$300` risk, 150-bar horizon, and stop-first
collision semantics.

For each eligible flat row `t`:

```text
T_WAIT(t)  = 0
T_LONG(t)  = opportunity_value of forced ENTER_LONG_1 at t+1 open
T_SHORT(t) = opportunity_value of forced ENTER_SHORT_1 at t+1 open
```

The primary ordering remains frozen:

- a side that reaches `+2R` before `-1R` receives a positive value;
- every side that does not satisfy that event remains non-positive;
- `WAIT=0`, so a failed or unresolved side cannot be turned into an Entry
  merely by a secondary score;
- if both sides qualify, both may be positive rather than forcing an arbitrary
  one-hot side; if neither qualifies, WAIT remains best.

Within those primary classes only, a bounded secondary term may encode richer
quality already produced by the same simulation—for example time-to-`+2R`
for winners and bounded MFE/distance-to-target for non-winners. Its range must
be proven too small to reorder a winner below WAIT or a non-winner above WAIT.
This makes the vector richer than the categorical label without redefining
`+2R-before-1R`.

Treat `T` as relative teacher preference logits or normalized advantages, not
as Bellman Q values in dollars. Match the network's centered action advantages
to `T` with one training-only vector loss while leaving C51 TD, exact action,
paired A+, replay, economics, and architecture unchanged. EarnHFT uses KL over
softmaxed values; PropEvolve should compare KL versus a direct advantage loss
only during design, then freeze one before the matched run—do not sweep both
inside the same experiment.

### Proof required before a full run

The MVP should be a label/loss smoke on authenticated training rows, then one
matched campaign only if all of these pass:

1. `T_LONG>0` if and only if Long satisfies the frozen economic event;
2. `T_SHORT>0` if and only if Short satisfies it;
3. all non-winners remain `<=T_WAIT`;
4. Long and Short mirror correctly under side-canonical inputs;
5. same-bar target/stop collisions remain stop-first;
6. the next-open and fee calculations exactly match the existing label engine;
7. teacher targets are generated only from training rows and never cross a
   temporal boundary;
8. recurrent burn-in and sequence boundaries are unchanged;
9. the teacher vector is absent from the policy observation schema and saved
   inference contract;
10. validation installs a teacher callback that raises on any lookup and still
    completes with zero lookups;
11. the checkpoint produces identical action values for the same causal input
    whether teacher artifacts are present on disk or unavailable.

### Matched acceptance test

Compare against the accepted Stage 2A parent with the same data, seed,
episode budget, replay samples, network, optimizer, and evaluation periods.
The only change is the full-action opportunity-value loss. Require:

- higher teacher-free pass rate or pass conversion;
- zero validation blows and no worse near-blow incidence;
- better Entry precision without unacceptable opportunity-recall loss;
- both Long and Short remain active, with per-side precision/recall reported;
- monotonic teacher-free C51 advantage by held-out opportunity-value bins;
- no teacher lookups and unchanged inference observations.

If it improves supervised agreement but not teacher-free economics, reject it.
Do not respond by increasing its weight or adding more architecture in the
same experiment.

## Experiment 2: learned Stage 2B worsening-drawdown loss

Run this only after Stage 2A is frozen. The goal is not another market gate;
it is to make the same recurrent policy more selective when causal account
state shows limited headroom.

Use the existing exact recovery/headroom definition to label replayed actions:

- harmful: after the action, the episode reaches the frozen near-blow/blow
  condition before recovery or pass;
- productive: after the action, the episode recovers or passes without first
  worsening into that condition.

Apply one auxiliary action-ranking loss on those training rows: harmful Entry
actions must rank below WAIT; productive A+ Entry actions may rank above WAIT.
Keep it inside the existing recurrent learning portion after burn-in, keep
Long/Short diagnostics separate, and do not expose the label or a teacher
channel at inference. This is a task-specific adaptation of DeepScalper's
training-only risk auxiliary and FinRL-Meta's focus on *worsening* drawdown,
not a claim that either source solved MLL recovery.

The matched falsifier is direct: teacher-free validation must reduce terminal
near-blows and improve timeout-to-pass conversion without reducing Stage 2A
Entry precision, collapsing a side, or introducing a blow. If it merely makes
the policy WAIT universally, reject it.

## What not to adopt

- **FinRL turbulence liquidation:** hard threshold gating; it does not teach
  Expansion + Regime discrimination.
- **MacroHFT's specialist pool/router/memory:** multiple inference policies and
  transition replay are a large architecture change with no prop-firm or
  Long/Short proof.
- **EarnHFT's full historical-chunk DP:** wrong objective, long-only inventory,
  and no challenge state. Adapt only the dense training-supervision idea.
- **EarnHFT/TradeMaster transition replay:** would discard recurrent burn-in,
  reset parity, and complete-sequence guarantees.
- **DeepScalper's raw future-volatility target:** volatility is not adverse
  excursion, dominant chop, or MLL headroom.
- **Qlib order-execution policy:** forced liquidation/acquisition is not
  discretionary A+ Entry/WAIT selection.
- **A new recurrent algorithm, PPO migration, or generic risk-sensitive RL
  replacement:** none is the smallest falsifying test of the current gap.
- **Two changes in one campaign:** test the full-action supervisor first;
  preserve the risk loss for the separately frozen Stage 2B experiment.

## Recommendation

Study and prototype the **PropEvolve opportunity-value target** first. It is
the only reviewed mechanism that adds materially new learning information
without changing the recurrent policy, action space, replay system, economic
contract, or teacher-free inference. Do not copy EarnHFT's DP literally.

If the label/loss smoke proves the target is genuinely richer than the current
categorical and paired-A+ targets, run one matched Stage 2A ablation. Only after
an improved Stage 2A checkpoint is frozen should the worsening-drawdown loss
be tested as the Stage 2B recovery mechanism.


# PropEvolve

[![Tests](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/johnamcruz/PropEvolve/graph/badge.svg?branch=main)](https://codecov.io/gh/johnamcruz/PropEvolve)

PropEvolve is a self-improving trading agent that **learns, remembers, adapts,
and trades within prop-firm constraints**.

The agent learns the complete trading decision directly from causal, frozen
FFM/Chronos2 market-context embeddings plus normalized account and execution
state. Authenticated Expansion, Trend, and Regime models can provide temporary
training supervision. The final policy is native to PropEvolve: it does not
require those teachers, an external trading policy, or handcrafted trend
indicators at inference time.

The current objective is direct: build a system that learns to pass prop-firm
challenges consistently without blowing the account.

## Objective

One 30-day Monte Carlo episode represents one prop challenge attempt:

- profit target: **+$6,000**;
- maximum loss budget: **-$3,000**;
- goal: maximize pass probability without blowing the account;
- execution: decisions observed at a completed bar are filled at the next bar
  open, with intrabar MLL enforcement and round-trip costs;
- risk accounting: the MLL floor trails realized balance only at the 5:00 p.m.
  Central session boundary and locks permanently at starting balance once the
  account reaches the passmark.

The same normalized state works whether the account is expressed as `$0 →
$6k` with a `-$3k` floor or as a `$3k` cushion targeting `$9k` with a `$0`
floor.

## Architecture

PropEvolve exposes one deterministic trading-policy interface with two
configurable implementations:

```text
causal completed bars
        │
frozen FFM/Chronos2 embeddings ── normalized account, MLL and execution state
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                  TradingPolicy interface
                           │
             ┌─────────────┴─────────────┐
             │                           │
       C51 / R2D2 policy          reasoning-model policy
     recurrent Q distributions    quantized local backbone
      and bounded replay          + causal market projector
             │                    + 20-bar rolling context
             └─────────────┬─────────────┘
                           │
               legal Wait / Long / Short
                 or Hold / Close action
                           │
              deterministic prop-risk mask
                           │
             unchanged prop-challenge simulator
                           │
                  pass / blow / timeout

Expansion / Trend / Regime teachers ──► training labels and losses only
```

The C51/R2D2 implementation is the established recurrent baseline and remains
available as a fallback. The reasoning-model implementation is a challenger:
it uses a configurable MLX-LM backbone on Apple silicon, scores only legal
action completions, and can be selected without changing the environment or
execution contract. C51 Q values and reasoning-model log likelihoods share an
action interface but are not treated as interchangeable scores.

The policy uses one state-dependent discrete action set:

- flat: wait or enter one contract Long/Short;
- positioned: hold or close the one-contract position;
- unsafe or nonsensical actions are masked outside the model.

At every flat-state decision, the selected policy compares Wait, Long, and
Short from the same causal state. While positioned, it compares Hold and Close.
The deterministic prop-risk mask remains authoritative. The MVP does not
pyramid, average down, or size positions by model confidence.

The environment owns next-open fills, costs, intrabar stop and blow priority,
the EOD trailing MLL, passmark locking, and pass/blow/timeout termination. The
model learns when and how to trade; it cannot override those invariants.

## Training and inference

Authenticated teacher outputs are training-only supervision. They never enter
the deployed policy observation and are never required during teacher-free
validation or inference. Both implementations consume the same causal market,
account, execution, and legal-action contracts.

The recurrent curriculum teaches the policy to:

- enter Long when Long Expansion is strong, the Expansion-anchored Regime is
  ready/non-chop, and Short evidence does not dominate;
- mirror the rule for Short;
- wait when Expansion is weak, directions conflict, the exact economic setup
  failed, or persistent chop dominates.

Training results do not promote a model. Candidates are evaluated greedily and
teacher-free using pass rate, blow rate, near-blow incidence, expectancy,
Long/Short participation, Entry precision, opportunity recall, and winner
retention.

The reasoning challenger is trained from scratch in two explicit phases before
economic evaluation:

- supervised fine-tuning distills market context and learns the complete legal
  trading vocabulary—Long/Short/WAIT while flat and Hold/Close while
  positioned—from causal counterfactual outcomes, including 2R/3R/4R
  target-before-stop, MFE, MAE, and terminal-return targets;
- bounded reinforcement learning uses the unchanged prop environment to refine
  those decisions for the complete account path: reach the profit target while
  preserving MLL headroom and avoiding a blow.

These phases have separate responsibilities. SFT receives trade-quality and
trade-management labels only; pass, blow, near-blow, and timeout rewards are
excluded from its targets and prompt. Its teacher-free gate requires all five
actions to generalize, positive trade expectancy, at least the configured
40% win-rate floor, and a configured 3R average winner. RL may start only from
that saved five-action SFT parent and alone optimizes challenge economics.

The challenger is not promoted merely because supervised loss improves. It
must master all action boundaries, survive save/reload, generalize to unseen
2025 data, and improve teacher-free economics. The year 2026 remains sealed for
final confirmation. Until those gates pass, C51/R2D2 remains the fallback.

## Causal inputs and evidence

Historical development uses independent 3-minute streams for:

`NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN`

All inputs are causal and available at the completed decision bar. Training,
selection, and sealed confirmation periods are chronological. No teacher,
cache, replay row, threshold, or recipe revision may inspect the sealed period
before the final contract is frozen.

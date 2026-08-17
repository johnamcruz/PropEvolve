# PropEvolve

[![Tests](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/johnamcruz/PropEvolve/graph/badge.svg?branch=main)](https://codecov.io/gh/johnamcruz/PropEvolve)

PropEvolve is a self-improving RL trading agent that **learns, remembers,
adapts, and trades within prop-firm constraints**.

The agent learns the complete trading decision directly from causal, frozen
FFM/Chronos2 market-context embeddings plus normalized account and execution
state. During the current curriculum, an authenticated Expansion model acts as
a temporary training teacher. The final policy is native to PropEvolve: it does
not require that teacher, an external trading policy, or handcrafted trend
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

```text
Promoted frozen Mask Chronos2 checkpoint
                 │
causal 3-minute OHLCV windows
                 │
        frozen FFM embeddings
                 │
                 ├── normalized account / MLL / position state
                 └── normalized contract economics
                               │
                   recurrent distributional
                     Double-DQN challenger
                               │
              simultaneous action-return distributions
                    ┌──────────┴──────────┐
               Long hypotheses      Short hypotheses
                    └──────────┬──────────┘
                    Wait / Hold / Exit values
                               │
                  deterministic prop-risk mask
                               │
                    30-day challenge outcome
                               │
                   bounded balanced replay
                               │
                 offline challenger improvement
                               ^
                               |
             temporary Expansion teacher loss
              (training only; discarded by stage)
```

The policy uses one state-dependent discrete action set:

- flat: wait or enter one contract Long/Short;
- positioned: hold or close the one-contract position;
- unsafe or nonsensical actions are masked outside the model.

At every flat-state decision, the recurrent C51 policy scores Wait, Long, and
Short from the same causal forward pass. The stronger risk-adjusted action can
win while the deterministic prop-risk mask remains authoritative. The MVP does
not pyramid, average down, or size positions by model confidence.

The environment owns next-open fills, costs, intrabar stop and blow priority,
the EOD trailing MLL, passmark locking, and pass/blow/timeout termination. The
model learns when and how to trade; it cannot override those invariants.

## Training and inference

Authenticated teacher outputs are training-only supervision. They never enter
the deployed policy observation and are never required during teacher-free
validation or inference.

The current selection curriculum teaches the policy to:

- enter Long when Long Expansion is strong, the Expansion-anchored Regime is
  ready/non-chop, and Short evidence does not dominate;
- mirror the rule for Short;
- wait when Expansion is weak, directions conflict, the exact economic setup
  failed, or persistent chop dominates.

Training results do not promote a model. Candidates are evaluated greedily and
teacher-free using pass rate, blow rate, near-blow incidence, expectancy,
Long/Short participation, Entry precision, opportunity recall, and winner
retention.

## Causal inputs and evidence

Historical development uses independent 3-minute streams for:

`NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN`

All inputs are causal and available at the completed decision bar. Training,
selection, and sealed confirmation periods are chronological. No teacher,
cache, replay row, threshold, or recipe revision may inspect the sealed period
before the final contract is frozen.

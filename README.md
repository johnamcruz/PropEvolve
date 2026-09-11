# PropEvolve

[![Tests](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/johnamcruz/PropEvolve/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/johnamcruz/PropEvolve/graph/badge.svg?branch=main)](https://codecov.io/gh/johnamcruz/PropEvolve)

PropEvolve is a self-improving trading agent that **learns, remembers, adapts,
and trades within prop-firm constraints**.

The agent learns the complete trading decision directly from causal, frozen
FFM/Chronos2 market-context embeddings. Authenticated Expansion, Trend, and
Regime models provide temporary training supervision. The final policy is
native to PropEvolve: it does not require those teachers, an external trading
policy, or handcrafted trend indicators at inference time. Account and MLL
state belong to the later RL challenge-mastery stage, not the supervised
trade-mastery labels.

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

PropEvolve exposes one deterministic reasoning-policy interface:

```text
causal completed bars
        │
frozen FFM/Chronos2 embeddings ── normalized account, MLL and execution state
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                  reasoning-model policy
                  quantized local backbone
                  + causal market projector
                  + 20-bar rolling context
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

The reasoning model uses a configurable MLX-LM backbone on Apple silicon and
scores only legal action completions. Model and adapter selection is JSON-driven;
changing either does not change the environment or execution contract.

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
validation or inference. The deployed policy consumes causal FFM embeddings,
trade-management state, and legal actions.

Training results do not promote a model. Candidates are evaluated greedily and
teacher-free using pass rate, blow rate, near-blow incidence, expectancy,
Long/Short participation, Entry precision, opportunity recall, and winner
retention.

The reasoning policy is trained from scratch in two explicit phases before
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

The policy is not promoted merely because supervised loss improves. It
must master all action boundaries, survive save/reload, generalize to unseen
2025 data, and improve teacher-free economics. The year 2026 remains sealed for
final confirmation.

When broad frozen assessment reveals mistakes, the corrective campaign repeats
a bounded evidence loop:

1. assess the frozen adapter on fixed development rows;
2. select actual mistakes plus representative mastered-action anchors;
3. fine-tune briefly and save a new adapter;
4. reassess on the unchanged validation rows;
5. retain the adapter only when mistakes improve without erasing mastered
   Long, Short, Wait, Hold, or Close behavior.

Each round writes score, selection, training, and acceptance receipts. These
make the exact corrective examples and retained anchors auditable without
feeding validation mistakes back into training.

## Causal inputs and evidence

Historical development uses independent 3-minute streams for:

`NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN`

All inputs are causal and available at the completed decision bar. Training,
selection, and sealed confirmation periods are chronological. No teacher,
cache, replay row, threshold, or recipe revision may inspect the sealed period
before the final contract is frozen.

## Tests and coverage

The normal suite exercises the reasoning path without downloading a model:

```bash
python -m pytest -p no:cacheprovider
```

It covers causal source authentication, temporal sealing, label generation,
all five legal actions, cache-local dataset collection, targeted
mistake-and-anchor sampling, checkpoint save/reload contracts, RL environment
boundaries, teacher-free evaluation, and workflow resume behavior. Local MLX
numeric tests run when MLX is available; full model-compute tests remain
explicit opt-ins through their documented environment variables.

CI runs the portable suite on Linux and the complete MLX suite on an Apple
silicon runner. The MLX coverage job measures the complete `propevolve` package
and fails below 85% statement coverage; Linux is not used to label Apple-only
learner code as uncovered. Coverage is a regression guard, not the acceptance
criterion: model selection still requires the separate trade-mastery and
challenge-economics gates above.

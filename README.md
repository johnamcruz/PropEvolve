# PropEvolve

PropEvolve is a self-improving RL trading agent that **learns, remembers,
adapts, and trades within prop-firm constraints**.

The first model learns the complete trading decision directly from causal,
frozen FFM/Chronos2 market-context embeddings plus normalized account and
execution state. It does not inherit algoTraderAI's PPO policy, Mantis signals,
or handcrafted trend indicators.

## Objective

One 30-day Monte Carlo episode represents one prop challenge attempt:

- profit target: **+$6,000**;
- maximum loss budget: **-$3,000**;
- goal: maximize pass probability without blowing the account;
- execution: decisions observed at a completed bar are filled at the next bar
  open, with intrabar MLL enforcement and round-trip costs.

The same normalized state works whether the account is expressed as `$0 →
$6k` with a `-$3k` floor or as a `$3k` cushion targeting `$9k` with a `$0`
floor.

## Initial architecture

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
                    exact valid-action scoring
                               │
                  deterministic prop-risk mask
                               │
                    30-day challenge outcome
                               │
                   bounded balanced replay
                               │
                 offline challenger improvement
```

The policy uses one state-dependent discrete action set:

- flat: wait or enter long/short at an allowed size;
- positioned: hold, add, reduce, or close;
- unsafe or nonsensical actions are masked outside the model.

## Market population

Historical training uses nine independent 3-minute markets:

`NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN`

NQ is the initial validation and live-deployment market. CL, SI, ZB, and ZN
are training-only source markets: they broaden market-pattern experience but
cannot be selected by the live execution allowlist. ES, GC, RTY, and YM require
their own isolated promotion evidence before live use.

Each episode contains one market's causal embedding stream. The live agent does
not require all nine feeds simultaneously.

## Learning and memory

PropEvolve separates two loops:

1. The fast execution loop uses immutable champion weights and external risk
   controls.
2. The slow improvement loop stores completed historical or shadow episodes,
   balances replay across markets, outcomes, and sides, trains a challenger,
   and evaluates it chronologically.

Live observations initially run in shadow/paper mode. They are authenticated
and added to offline replay; the deployed model never rewrites itself intraday.
A challenger must pass temporal OOS, prop-gauntlet, shadow, and canary gates
before replacing the champion.

## FFM integration

PropEvolve imports `futures-foundation-model[foundation]` as a package and uses
its Chronos2 embedding implementation. FFM source is not copied into this
repository. OHLCV and the promoted Mask checkpoint are local symbolic links;
generated embedding caches and run artifacts remain on the SSD and are ignored
by Git.

Current promoted checkpoint identity:

```text
checkpoints/chronos2_mask_full/adapter_model.safetensors
sha256 a5d31166f7cd36b3eb7f7d1242dd07d65c3eddda94d8c478e77a2e11307c1104
```

Create or refresh the local links without copying assets:

```bash
propevolve setup-assets \
  --market-data "/Volumes/CRUZ SSD/algoTraderAI/data" \
  --checkpoint "/Volumes/CRUZ SSD/Futures-Foundation-Model/checkpoints/chronos2_mask_full"
```

## Running the historical POC

Install PropEvolve and its pinned FFM package integration:

```bash
python -m pip install -e '.[dev,ffm]'
```

Validate the frozen experiment contract:

```bash
propevolve validate-config --config config/historical_mask_v1.json
```

Build the nine frozen embedding caches:

```bash
propevolve build-cache --config config/historical_mask_v1.json
```

Then train the historical challenger and evaluate naturally on NQ 2025:

```bash
propevolve train --config config/historical_mask_v1.json
```

The recipe trains on 2021–2024, uses 2025 for development validation, and
leaves data from 2026 onward sealed. Running the commands is intentionally
deferred while the separate Expansion campaign owns the machine.

## Expansion, Pivot, and Trend teachers

The first matched baseline uses only frozen embeddings and account state. If
that context is insufficient, causal OOF Expansion, Pivot, or Trend predictions
may be used as **temporary training teachers**. Teacher outputs never enter the
student's deployed observation, their influence decays during training, and the
final teacher-free student must earn temporal OOS economic lift.

Permanent detector inputs are considered only if teacher-free distillation
fails and a matched ablation proves that the additional live dependency raises
pass rate or expectancy while reducing blow risk.

## Economics and evidence

Contract point values are explicit in the experiment JSON. Round-trip fees are
also explicit and currently follow the official
[TopstepX commission table](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees).
No market is allowed to train fee-free because a constant is missing.

Training loss is not promotion evidence. A candidate must ultimately report
per-market and chronological pass, blow, timeout, expectancy, drawdown, and
turnover evidence, with NQ evaluated independently.

## Tests

```bash
pytest
```

Tests cover asset identity and symbolic linking, causal cache timing, frozen FFM
delegation, observation normalization, action masking, challenge economics,
balanced replay, shadow-memory authentication, configuration, orchestration,
and the recurrent distributional agent.

The research synthesis behind the design is in
[`docs/research/stanford_cs329a_self_improving_rl_agent.md`](docs/research/stanford_cs329a_self_improving_rl_agent.md).

# PropEvolve

PropEvolve is a self-improving RL trading agent that **learns, remembers,
adapts, and trades within prop-firm constraints**.

The first model learns the complete trading decision directly from causal,
frozen FFM/Chronos2 market-context embeddings plus normalized account and
execution state. The policy is native to PropEvolve and does not depend on
external trading policies, signal models, or handcrafted trend indicators.

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

The promoted training recipe selects accelerators in the order CUDA, Apple
Metal (`mps`), then CPU. Replay storage and environment simulation remain on
CPU; batched network updates, recurrent inference, target-network updates, and
checkpoint resume are MPS-compatible. An explicitly requested unavailable
accelerator fails immediately; only `auto` may fall back to the next device.

## Market population

Historical training uses nine independent 3-minute markets:

`NQ, ES, GC, RTY, YM, CL, SI, ZB, ZN`

NQ is the initial validation and live-deployment market. CL, SI, ZB, and ZN
are training-only source markets: they broaden market-pattern experience but
cannot be selected by the live execution allowlist. ES, GC, RTY, and YM require
their own isolated promotion evidence before live use.

Each episode contains one market's causal embedding stream, but training draws
episodes from all nine markets in seeded, shuffled, balanced cycles into the
same shared policy and replay memory. Every complete nine-episode cycle covers
each market once, and experience learned on any market updates the one agent.
The markets are not presented as nine simultaneous feeds because live inference
observes and trades one market at a time; this preserves train-to-live input
parity while still teaching cross-market generalization.

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

PropEvolve imports
[`futures-foundation-model[foundation]`](https://github.com/johnamcruz/Futures-Foundation-Model)
as a package and uses its Chronos2 embedding implementation. FFM source is not
copied into this repository. OHLCV and the promoted Mask checkpoint are local
symbolic links; generated embedding caches and run artifacts remain in
configured local storage and are ignored by Git.

Current promoted checkpoint identity:

```text
checkpoints/chronos2_mask_full/adapter_model.safetensors
sha256 a5d31166f7cd36b3eb7f7d1242dd07d65c3eddda94d8c478e77a2e11307c1104
```

Create or refresh the local links without copying assets:

```bash
propevolve setup-assets \
  --market-data "/path/to/ohlcv/data" \
  --checkpoint "/path/to/chronos2_mask_full"
```

## Running the historical POC

Install PropEvolve and its pinned FFM package integration:

```bash
python -m pip install -e '.[dev,ffm]'
git config core.hooksPath .githooks
```

The tracked pre-commit hook runs the complete unit-test suite and blocks the
commit if any test fails. It uses `.venv/bin/python` when available; set
`PROPEVOLVE_PYTHON` to an explicit interpreter when the environment lives
elsewhere.

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

The development recipe trains on 2021–2024 and uses 2025 for selection. After
the complete recipe is frozen, a separate final-fit stage may retrain on
2021–2025. Data from 2026 onward is a one-time sealed evaluation set and must
never enter cache generation, replay, training, validation, reasoning packets,
candidate selection, or recipe revision.

Embedding caches physically censor bars by close-time availability before
Chronos encoding. Their authenticated manifest records
`research_end_exclusive = 2026-01-01T00:00:00+00:00` and
`sealed_holdout_touched = false`; legacy or boundary-crossing caches fail
closed. The promoted Mask checkpoint is also backed by the FFM 36-stream
preflight whose aligned data ends before that same holdout boundary.

## Offline self-improvement

Every completed training run becomes a content-addressed challenger bundle
under the configured output directory. A bundle contains immutable model
weights, its frozen contract, complete recipe, parent lineage, and hypothesis.
Running the trainer again creates another candidate; it never overwrites the
previous model.

PropEvolve's offline evolution layer provides six research interfaces:

1. An immutable candidate archive and reversible champion registry.
2. A bounded set of diverse elites selected by declared metrics such as pass
   rate, blow rate, expectancy, temporal stability, and side balance.
3. Executable multi-metric evaluation, with economic evidence—not model
   critique—as the source of truth.
4. A staged evaluator cascade that stops weak candidates before expensive
   walk-forward, prop-gauntlet, sealed, or shadow evaluation.
5. Authenticated reasoning packets containing the champion, selected prior
   candidates, exact evidence, failure taxonomy, and frozen contract.
6. Allowlisted JSON revisions that may tune declared model or training fields
   but cannot change data lineage, temporal splits, costs, prop rules, sealed
   periods, or deployment markets.

The shared ML Training Loop consumes these interfaces to diagnose evidence and
propose the next bounded challenger. It operates only between runs: deployed
champion weights remain frozen, and activation or rollback always records an
append-only receipt. Start or interruption-safely resume a campaign with:

```bash
propevolve evolve --config config/historical_mask_v1.json --run-id mask-v1
```

Inspect durable state without launching work with:

```bash
propevolve evolve-status --config config/historical_mask_v1.json --run-id mask-v1
```

Reasoning remains the campaign controller. The shared loop's optional
`SurrogateAdvisor` may later provide uncertainty-aware diagnostics and bounded
numerical proposals through Optuna or another Bayesian backend, but it cannot
select a proposal, mutate the plan, execute training, or waive a gate. No
surrogate is enabled by default. This boundary follows
[*Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch*](https://arxiv.org/abs/2608.00316).

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

Tests cover asset identity and symbolic linking, causal cache timing, physical
sealed-row censoring, temporal-role boundaries, frozen FFM
delegation, observation normalization, action masking, challenge economics,
balanced replay, shadow-memory authentication, configuration, orchestration,
the recurrent distributional agent, immutable model lineage, evaluator
cascades, diverse candidate selection, reasoning-packet authentication, and
frozen-contract recipe revision.

The research synthesis behind the design is in
[`docs/research/stanford_cs329a_self_improving_rl_agent.md`](docs/research/stanford_cs329a_self_improving_rl_agent.md).
The AlphaEvolve mapping and its trading-specific safety limits are documented
in
[`docs/research/alphaevolve_concepts_for_propevolve.md`](docs/research/alphaevolve_concepts_for_propevolve.md).

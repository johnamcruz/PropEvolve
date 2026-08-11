# PropEvolve

PropEvolve is a self-improving RL trading agent that **learns, remembers,
adapts, and trades within prop-firm constraints**.

The agent learns the complete trading decision directly from causal, frozen
FFM/Chronos2 market-context embeddings plus normalized account and execution
state. During the current curriculum, an authenticated Expansion model acts as
a temporary training teacher. The final policy is native to PropEvolve: it does
not require that teacher, an external trading policy, or handcrafted trend
indicators at inference time.

## Objective

One 30-day Monte Carlo episode represents one prop challenge attempt:

- profit target: **+$6,000**;
- maximum loss budget: **-$3,000**;
- goal: maximize pass probability without blowing the account;
- execution: decisions observed at a completed bar are filled at the next bar
  open, with intrabar MLL enforcement and round-trip costs.
- risk accounting: the MLL floor trails realized balance only at the 5:00 p.m.
  Central session boundary and locks permanently at starting balance once the
  account reaches the passmark.

The same normalized state works whether the account is expressed as `$0 →
$6k` with a `-$3k` floor or as a `$3k` cushion targeting `$9k` with a `$0`
floor.

## Current architecture

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

- flat: wait or enter one contract long/short;
- positioned: hold or close the one-contract position;
- unsafe or nonsensical actions are masked outside the model.

FFM supplies shared market-state and expansion-opportunity context; it is not
required to classify direction. At every flat-state decision, the recurrent
C51 policy scores the complete return distribution for Long and Short in the
same causal forward pass, alongside Wait. These are
independent action-value hypotheses learned from one shared recurrent state,
so the stronger risk-adjusted side can win while the external prop-risk mask
remains authoritative.

The first version deliberately excludes pyramiding, add/reduce actions,
conviction sizing, inherited signal gates, and PPO-specific behavior. The
algorithm-independent challenge contract retains cumulative balance, next-open
fills, fees, intrabar blow precedence, EOD trailing MLL, passmark locking,
pass/blow/timeout priority, and faster-pass reward. Terminal reward ratios are
preserved under a constant scale suitable for distributional value learning.

The completed matched baseline used only frozen FFM embeddings as market
context. The current experiment adds causal, pre-2025 Expansion targets as a
temporary distillation teacher. They shape the shared recurrent policy during
training but are omitted from validation and deployment observations. The
student must therefore internalize useful expansion semantics while learning
direction, timing, trade management, and abstention from challenge economics.
No teacher or specialist may inspect the 2025 selection period or sealed 2026
period.

The promoted training recipe selects accelerators in the order CUDA, Apple
Metal (`mps`), then CPU. Replay storage and environment simulation remain on
CPU; batched network updates, recurrent inference, target-network updates, and
checkpoint resume are MPS-compatible. An explicitly requested unavailable
accelerator fails immediately; only `auto` may fall back to the next device.

Runtime acceleration is evidence-gated rather than assumed. The JSON `runtime`
section controls mixed precision, Metal matmul, and optional `torch.compile`;
C51 projection and loss math remain FP32. Compare the exact four matched arms
before changing a training recipe:

```bash
propevolve benchmark-runtime \
  --config config/historical_mask_expansion_regime_curriculum_v8.json \
  --warmup-updates 5 \
  --measured-updates 20 \
  --output runs/runtime-benchmark.json
```

The benchmark runs eager FP32, FP16 autocast, FP16 plus Metal matmul, and FP16
plus `torch.compile` in separate processes. This ensures the Metal environment
switch is applied before MPS initializes. Compilation falls back to eager
execution if graph lowering is unsupported. Training runtime settings stay
frozen during a reasoning campaign. An arm is eligible only when its final
matched-update loss stays within the JSON-declared relative drift tolerance;
a compile arm that fell back to eager is reported but is not eligible.

The experiment JSON is the complete serialized recipe. Cache dimensions,
challenge economics, risk and reward behavior, action size, C51 support,
optimizer, target synchronization, replay, exploration, temporal splits, and
device must all be declared there. The loader fails closed when a required
field is absent; Python constructors do not silently supply training defaults.

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
copied into this repository. OHLCV, the promoted Mask checkpoint, and an
optional authenticated FFM embedding cache remain external; PropEvolve uses
local symbolic links and stores only its own manifests and run artifacts.

Current promoted checkpoint identity:

```text
checkpoints/chronos2_mask_full/adapter_model.safetensors
sha256 a5d31166f7cd36b3eb7f7d1242dd07d65c3eddda94d8c478e77a2e11307c1104
```

## Local setup and Mask cache generation

PropEvolve does not redistribute market data. Supply one UTC, bar-open OHLCV
CSV per market using this layout:

```text
/path/to/ohlcv/data/
├── NQ_3min.csv
├── ES_3min.csv
├── GC_3min.csv
├── RTY_3min.csv
├── YM_3min.csv
├── CL_3min.csv
├── SI_3min.csv
├── ZB_3min.csv
└── ZN_3min.csv
```

Every CSV must contain ordered, unique `datetime`, `open`, `high`, `low`,
`close`, and `volume` columns. `datetime` must be parseable as UTC bar-open
time. PropEvolve validates timestamps, OHLC geometry, volume, source hashes,
and close-time availability before encoding.

Clone PropEvolve and the public FFM repository, then install PropEvolve with
the pinned FFM integration:

```bash
git clone https://github.com/johnamcruz/PropEvolve.git
git clone https://github.com/johnamcruz/Futures-Foundation-Model.git
cd PropEvolve

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ffm]'
git config core.hooksPath .githooks
```

The promoted public checkpoint is available in the FFM checkout at
`checkpoints/chronos2_mask_full`. Register local assets using generic paths;
the command creates symbolic links and an ignored, hash-authenticated
`config/local-assets.json` rather than copying large files:

```bash
propevolve setup-assets \
  --workspace . \
  --market-data "/path/to/ohlcv/data" \
  --checkpoint "/path/to/Futures-Foundation-Model/checkpoints/chronos2_mask_full"
```

There are two supported cache paths.

### First-time native generation

Copy the active recipe to a local working recipe, do not commit that local
copy, and set `cache.format` to `native`. Keep the checkpoint, context length,
stride, temporal boundary, and all other recipe fields unchanged. Then
validate the recipe and build one market as a quick integrity check:

```bash
cp config/historical_mask_expansion_entry_search_curriculum_v7.json \
  config/local-experiment.json
# Edit config/local-experiment.json: set cache.format to "native".

propevolve validate-config --config config/local-experiment.json
propevolve build-cache \
  --config config/local-experiment.json \
  --ticker NQ
```

After the NQ build succeeds, generate the remaining declared markets. Existing
exact-identity caches are reported as `HIT` and are not rebuilt:

```bash
propevolve build-cache --config config/local-experiment.json
```

Native generation loads OHLCV through the installed FFM package and encodes it
with the promoted Chronos2 Mask checkpoint. The cache contains only causal
completed-bar embeddings strictly before the recipe's sealed boundary.

### Import an existing authenticated FFM cache

If an FFM representation cache already exists, register it during setup:

```bash
propevolve setup-assets \
  --workspace . \
  --market-data "/path/to/ohlcv/data" \
  --checkpoint "/path/to/Futures-Foundation-Model/checkpoints/chronos2_mask_full" \
  --embedding-cache "/path/to/ffm/representation_cache"
```

Keep `cache.format` as `ffm_frozen_representation_v2`, validate, and import:

```bash
propevolve validate-config \
  --config config/historical_mask_expansion_entry_search_curriculum_v7.json
propevolve build-cache \
  --config config/historical_mask_expansion_entry_search_curriculum_v7.json
```

This path authenticates the FFM checkpoint, source rows, row map, embedding
array, encoder identity, context length, stride, and pre-2026 boundary. It
symlinks the large embedding array instead of duplicating it.

Both paths write per-market manifests beneath the configured `cache_root`.
Cache reuse is allowed only when every authenticated identity matches; a stale,
partial, boundary-crossing, or differently encoded cache fails closed.

## Running the historical POC

The tracked pre-commit hook runs the complete unit-test suite and blocks the
commit if any test fails. It uses `.venv/bin/python` when available; set
`PROPEVOLVE_PYTHON` to an explicit interpreter when the environment lives
elsewhere.

Build the authenticated, training-only Expansion teacher targets if they are
not already present:

```bash
propevolve build-expansion-teacher-cache \
  --config config/expansion_teacher_cache_v1.json
```

Then start or resume the complete reasoning-guided curriculum:

```bash
propevolve evolve \
  --config config/historical_mask_expansion_entry_search_curriculum_v7.json \
  --run-id expansion-entry-curriculum-v7r1
```

The development recipe trains on 2021–2024 and uses 2025 for selection. After
the complete recipe is frozen, a separate final-fit stage may retrain on
2021–2025. Data from 2026 onward is a one-time sealed evaluation set and must
never enter cache generation, replay, training, validation, reasoning packets,
candidate selection, or recipe revision.

The terminal robustness gate opens 2026 exactly once after the policy, seeds,
economics, costs, and evaluator are frozen. It runs teacher-free,
non-overlapping 30-session prop challenges independently for every declared
market. A passing policy must achieve at least 50% aggregate and per-market
pass rate, zero aggregate and per-market blow rate, and positive net expectancy.
Timeouts remain non-passes. The receipt also exposes trade frequency,
Long/Short usage, win rate, average winner R, 2R winner retention, near-blow
frequency, terminal P&L, and every per-market result so aggregate performance
cannot conceal a failed market. Any teacher access, overlapping window, missing
market, incomplete 30-session episode, or inspected-row reuse invalidates the
confirmation rather than producing a score.

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
propevolve evolve \
  --config config/historical_mask_expansion_teacher_curriculum_v6.json \
  --run-id expansion-curriculum-v6r1
```

Inspect durable state without launching work with:

```bash
propevolve evolve-status \
  --config config/historical_mask_expansion_teacher_curriculum_v6.json \
  --run-id expansion-curriculum-v6r1
```

Reasoning remains the campaign controller. The shared loop's optional
`SurrogateAdvisor` may later provide uncertainty-aware diagnostics and bounded
numerical proposals through Optuna or another Bayesian backend, but it cannot
select a proposal, mutate the plan, execute training, or waive a gate. No
surrogate is enabled by default. This boundary follows
[*Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch*](https://arxiv.org/abs/2608.00316).

The reasoning proposer is also selectable without changing the reasoning
provider. Existing recipes default to `standard`. Set the following only for an
opt-in GEPA-style reflective checkpoint:

```json
{
  "campaign": {
    "reasoning": {
      "provider": "codex",
      "proposer": "gepa_reflective"
    }
  }
}
```

The reflective proposer adds a content-addressed Actionable Side Information
packet containing the matched parent, diverse candidate evidence, exact gate
failure, prior revisions, and experiment ledger. Codex still proposes exactly
one allowlisted JSON revision, while the existing evaluator and hard zero-blow
gate remain authoritative. It does not add GEPA as a runtime dependency or
permit reasoning to mutate the frozen research contract.

## Expansion, Regime, Pivot, and Trend teachers

The first matched baseline uses only frozen embeddings and account state. If
that context is insufficient, causal OOF Expansion, Pivot, or Trend predictions
may be used as **temporary training teachers**. Teacher outputs never enter the
student’s deployed observation. JSON-configured curricula gradually decay every
auxiliary teacher loss while deterministically hiding an increasing fraction of
teacher targets, so the student first learns the semantics and then must act
without dependable guidance. The final teacher-free student must earn temporal
OOS economic lift.

`teachers/` owns the common teacher manifest and separate `expansion/`,
`regime/`, and `trend/` adapters. Expansion, Regime, and Trend contain
authenticated nine-market, 3-minute checkpoints matching PropEvolve's imported
Chronos2 Mask caches. Trend contributes separate Long/Short launch likelihood
and conditional-quality confluence; it is never a trading signal or hard gate.
No Pivot teacher is currently approved. Every target cache must be regenerated
from its declared checkpoint for all nine markets before a teacher experiment
is enabled; unrelated or merely timestamp-compatible score buses are rejected.

After the matched baseline completes, generate the training-only Expansion
teacher caches with the existing Mask embeddings:

```bash
propevolve build-expansion-teacher-cache \
  --config config/expansion_teacher_cache_v1.json

propevolve build-regime-teacher-cache \
  --config config/regime_teacher_cache_v1.json

propevolve build-trend-teacher-cache \
  --config config/trend_teacher_cache_v1.json
```

The builder loads the model once on MPS, scores one ticker at a time, starts at
batch size 1024 with bounded memory fallbacks, writes each ticker atomically,
and treats authenticated completed tickers as resumable cache hits. It
physically excludes 2025 selection and the sealed 2026 period.

Permanent detector inputs are considered only if teacher-free distillation
fails and a matched ablation proves that the additional live dependency raises
pass rate or expectancy while reducing blow risk.

## Current staged curriculum

The active recipe teaches one economic competency at a time. A stage advances
only after its chronological selection evidence passes its declared gates:

```text
Stage 1: Safety foundation (1M environment steps)
  -> zero blowouts, limited near-blow timeouts, initial pass capability
Stage 2: Winner retention (1M environment steps)
  -> preserve Stage 1 safety and improve capture of trades reaching 2R+
Stage 3: Challenge completion (2M environment steps)
  -> preserve safety and retention while improving pass rate and expectancy
Stage 4: Frozen confirmation (5M steps x 8 seeds, at most 3 in parallel)
  -> no recipe revisions; require robust economic and side-balanced evidence
```

Stages 2–4 warm-start from the authenticated policy weights selected by the
preceding stage. They do not restart policy learning. Optimizer state, replay
memory, exploration state, and the temporary teacher head are deliberately
reset at each stage so the next lesson is learned from fresh experience without
discarding the policy's acquired representation and behavior.

RL progress is measured in environment steps rather than epochs because
episode lengths differ. Each 1M, 2M, or 5M stage is one declared evidence
window: training diagnostics detect collapse inside the window, and the policy
is evaluated without teachers on the chronological 2025 selection period only
after the window closes. Training episodes can preserve rollback anchors, but
they cannot select or promote a model. The 2026 period remains sealed.

Checkpoint roles are intentionally separate:

- `training-recovery.pt` atomically saves the exact latest restart state,
  including policy and target networks, optimizer, mixed-precision scaler,
  progress, environment RNG, complete bounded replay, and replay sampler RNG;
- `retained-pass-policies/` preserves every pre-update policy that demonstrated
  a training pass, while `retained-pass-policy.pt` atomically identifies the
  latest rollback anchor; neither is treated as OOS promotion evidence;
- every completed candidate and its teacher-free evaluation receipt is stored
  immutably in the campaign archive, and a selected candidate may warm-start a
  later fine-tuning stage while temporary teacher heads and optimization state
  are rebuilt from that stage's recipe.

When a revisable stage misses a gate, the authenticated diagnostic report is
sent to the configured reasoning proposer. It may change exactly one bounded,
allowlisted recipe revision relevant to that stage and rerun it. It cannot
change market data, cache or teacher lineage, temporal splits, costs, prop
rules, deployment markets, or the sealed holdout. The confirmation stage is
fully frozen and cannot reason around a failed gate.

The curriculum's promotion priorities are ordered deliberately:

1. zero blowouts;
2. fewer near-blow timeouts;
3. stronger retention of 2R-or-larger opportunities;
4. higher challenge pass rate and positive expectancy.

This is curriculum learning, not four unrelated searches. Each accepted stage
becomes the parent of the next candidate, and the final multi-seed gauntlet
tests the accumulated policy rather than selecting a new recipe.

## Economics and evidence

Contract point values are explicit in the experiment JSON. Round-trip fees are
also explicit and currently follow the official
[TopstepX commission table](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees).
No market is allowed to train fee-free because a constant is missing.

Training loss is not promotion evidence. A candidate must ultimately report
per-market and chronological pass, blow, timeout, expectancy, drawdown, and
turnover evidence, with NQ evaluated independently.

Closed-trade diagnostics also report MAE, MFE, MFE-to-realized-R gap, profit
capture, and the fraction of trades that reached at least 1R or 2R before
round-tripping to scratch or loss. These are split by PASS, TIMEOUT, BLOW, and
ticker and are included in the authenticated reasoning packet. They diagnose
whether entry selection or winner retention failed; they are not direct reward
targets because optimizing capture alone can select a short-horizon scalper.

Campaign stages may declare `parent_improvement_requirements`. With those
enabled, a zero-blow challenger can advance only when both pass rate and the
declared retained-winner metric strictly improve over its authenticated parent.
Reasoning chooses the mechanism family first; bounded numerical search is then
restricted to that family's JSON fields.

## Tests

```bash
pytest
```

Tests cover asset identity and symbolic linking, causal cache timing, physical
sealed-row censoring, temporal-role boundaries, frozen FFM
delegation, observation normalization, action masking, challenge economics,
golden-trajectory parity for fills, fees, MLL behavior, pass/blow/timeout, and
episode termination,
balanced replay, shadow-memory authentication, configuration, orchestration,
the recurrent distributional agent, immutable model lineage, evaluator
cascades, diverse candidate selection, reasoning-packet authentication, and
frozen-contract recipe revision.

The research synthesis behind the design is in
[`docs/research/stanford_cs329a_self_improving_rl_agent.md`](docs/research/stanford_cs329a_self_improving_rl_agent.md).
The AlphaEvolve mapping and its trading-specific safety limits are documented
in
[`docs/research/alphaevolve_concepts_for_propevolve.md`](docs/research/alphaevolve_concepts_for_propevolve.md).

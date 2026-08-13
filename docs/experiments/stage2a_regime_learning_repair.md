# Stage 2A Regime learning repair

## Status

This is a frozen two-screen research contract. It is ready to run but has not
produced evidence. Nothing in this document is a promotion claim.

The runnable recipe is:

`config/historical_mask_expansion_regime_stage2a_learning_repair_v1.json`

It has a new output root and campaign-state root. It cannot resume the failed
Stage 2 campaign or inherit one of its candidates.

## Failed boundary

The first Stage 2 attempt did not convert Regime supervision into better
teacher-free behavior. Its authenticated source labels were approximately
balanced by direction, but sampled positive supervision was not: Long examples
outnumbered Short examples by about 9.4 to 1, and Short recall fell from 77.94%
in Stage 1 to 13.77%. Static chop probability also failed to distinguish
passing episodes from near-blow timeouts.

Two failures must therefore be tested separately:

1. unequal Long and Short learning exposure;
2. static state semantics that conflate persistent dead chop with a
   transition-ready compression.

## Immutable parent and temporal contract

Both screens descend from the exact selected Stage 1 candidate:

- candidate: `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- evaluation: `c49852955655b705e376e057dfe2bf58784481175363b970bab063d8c42f981b`
- model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`

Stage 2A.1 warm-starts that exact model. Stage 2A.2 may start only from the
selected Stage 2A.1 candidate. Revisions within a stage do not silently adopt a
failed attempt as their parent.

Training remains 2021 through 2024, and chronological teacher-free selection
remains 2025. No row from the sealed period beginning in 2026 is returned to,
used by, or evaluated by this campaign. The shared loader still hashes and
parses each source CSV before applying its temporal filter, so this is a
modeling/evaluation exclusion rather than a claim that 2026 file bytes are
never read from storage.
Market data, frozen embeddings, teacher caches, Entry labels, costs, prop
economics, action space, risk, observation state, and winner-retention behavior
are unchanged from the Stage 1 lineage.

## Stage 2A.1: side-balanced mechanism screen

`regime_side_balance_500k` is a 500,000-step matched screen. It changes only
how the existing static Regime auxiliary receives direction examples:

- replay entry opportunities alternate authentic Long and Short anchors when
  both are available;
- Regime positive loss computes independent Long and Short means and averages
  the active sides equally;
- WAIT remains separate and is not included in the direction balance;
- Regime semantics remain `static_state_v1`;
- persistent-chop emphasis remains exactly zero.

The frozen side-balance identity is:

```json
{
  "schema": "equal_long_short_v1",
  "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"]
}
```

Training must contain nonzero Long and Short sampled rows, nonzero recall for
both directions, nonzero declared-side model response, both chop and non-chop
rows, and learned WAIT probability that is higher in dominant chop than in
non-chop. Teacher-free selection must complete, trade both directions, have no
blowouts, and match or exceed its selected Stage 1 parent on pass rate,
average winner R, expectancy, and two-R MFE capture without worsening the
near-blow timeout rate. The lower absolute gates remain fail-safe floors; they
cannot authorize regression from the authenticated parent.

This screen tests mechanism retention. It does not claim that static Regime
semantics improve economics.

The no-pass, blow-rate, and policy-collapse screen is evaluated only after the
declared 500,000-step evidence boundary. Early warm-start episodes remain
diagnostic evidence; they cannot terminate the screen before both directions
have had the contracted learning budget.

After the fully autonomous tail, each final teacher-discarded checkpoint is
probed without updates on exactly 32 unique authenticated pre-2025 replay rows
for each of WAIT, Long, and Short. Labels and Regime probabilities classify the
already-produced greedy Q values post hoc; they are never policy inputs. WAIT
recall must be at least 50%, and Long and Short recall must each be at least
40%. Stage 2A.1 additionally requires higher WAIT response on same-label
positive Entry rows in static chop than non-chop. This final-checkpoint gate,
not a whole-run optimizer average, is the evidence that learned behavior
survived teacher removal and late-training forgetting.

## Stage 2A.2: transition-aware persistent-chop screen

`persistent_chop_regime_500k` is another 500,000-step screen. It warm-starts
only the selected Stage 2A.1 candidate and atomically switches three fields:

- `formula`;
- `semantics = persistent_chop_negative_weight_v1`;
- `persistent_chop_negative_emphasis = 1.0`.

Partial switches are rejected. The effective resolved recipe, warm-start model,
source-module hashes, replay sampling contract, and training recovery identity
are archived together, so an old Stage 2 recovery file cannot be loaded.

The teacher remains auxiliary. It does not mask actions or enter trades. Exact
WAIT examples receive continuous additional weight when chop persistence is
high and transition readiness is low. Transition readiness is:

```text
Kaufman efficiency
* mean(
    trend onset,
    trend persistence,
    volatility-expansion onset,
    high-volatility persistence,
    volatility percentile
  )
```

This consumes the requested Regime channels without converting them into a
hard-coded trading rule. The same exact-WAIT label population is divided by
continuous membership into persistent-dead-chop and transition-ready evidence;
no arbitrary classification threshold is introduced.

## Required learning evidence

The persistent screen fails closed unless diagnostics establish all of the
following:

- nonzero exact-WAIT mass;
- nonzero same-label persistent-dead-chop and transition-ready WAIT mass;
- learned model WAIT probability is greater for persistent-dead-chop WAIT than
  for transition-ready WAIT;
- transition-positive Long and Short populations are both nonzero and receive
  nonzero declared-side model probability;
- sampled Long and Short Entry rows and recall are both nonzero;
- teacher-free selection enters both Long and Short at least once.

Comparing the same exact-WAIT label on both sides of the Regime gate avoids a
confounded comparison between WAIT labels and positive Entry labels.

The final Stage 2A.2 probe also requires positive response for both Long and
Short transition-ready rows, alongside higher WAIT response for persistent
dead-chop exact-WAIT rows than transition-ready exact-WAIT rows.

Stage 2A.2 must also improve teacher-free near-blow timeout rate over its
selected parent by more than one percentage point while matching or exceeding
that parent on pass rate, average winner R, expectancy, and two-R MFE capture,
with zero blowouts.

## Bounded evolution and falsifier

Each screen permits at most three reasoning revisions. Stage 2A.1 may revise
only Regime loss weight or Q temperature within its JSON bounds. Stage 2A.2 may
revise only Regime loss weight or persistent-chop emphasis within its JSON
bounds. Data, labels, teachers, temporal roles, economics, Stage 1 parent,
side-balance identity, and all inference behavior remain frozen.

The hypothesis is falsified if either direction is absent, if learned WAIT
ordering is not present without arbitrary thresholds, if teacher-free selection
loses either direction, if blow rate is nonzero, or if persistent-chop learning
does not produce the declared near-blow improvement while retaining Stage 1
economics. Deficit recovery, Trend, Pivot, sizing, and reward changes are out of
scope for this campaign.

## Launch after review

Use a new run identifier; do not reuse a failed Stage 2 run ID:

```bash
propevolve evolve \
  --config config/historical_mask_expansion_regime_stage2a_learning_repair_v1.json \
  --run-id stage2a-regime-learning-repair-v1
```

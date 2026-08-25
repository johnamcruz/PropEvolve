# PropEvolve pass-rate root-cause research

## Decision

V26 is not the root-cause correction. It changes the weight of the existing
local `+2R-before-1R` winner margin from `2.0` to `3.0`, but it does not change
the economic horizon learned by recurrent C51. The completed V25 training and
two teacher-free validations localize the first failed boundary to
**challenge-level temporal credit assignment**, not replay integrity, recurrent
state, teacher leakage, missing local winner/failure labels, or insufficient
replay calls.

The smallest compatible root-cause experiment is a **challenge-return
self-imitation ablation on the existing recurrent C51 target**, following the
Self-Imitation Advantage Learning (SAIL) principle. It must use an explicitly
challenge-horizon return contract; adding canonical SAIL while retaining the
current short effective horizon would not address the defect. This proposal
does not change the `WAIT`/Long/Short action space, add a policy/value head,
introduce a hard gate, or expose training labels at inference.

No primary source or completed PropEvolve run proves that this change will
produce a 60% pass rate. It is the smallest experiment that directly tests the
proven root cause. V26 should not consume a full campaign merely to test a
larger local margin.

## Frozen evidence

### The action and economic contract can support the target

A posthoc feasibility oracle used the existing authenticated `+2R before -1R`
entry labels, current next-open fills, one-contract sizing, fees, MLL, and
ratchet mechanics on the inspected 2025 NQ development period. It is not a
deployable policy or validation result, but it answers whether the existing
`WAIT`/Long/Short contract contains enough opportunity:

| Start PnL | Pass | Blow | Timeout | Near blow | Mean trades |
|---|---:|---:|---:|---:|---:|
| `$0` | 200/200 | 0/200 | 0/200 | 0/200 | `10.45` |
| `-$1,500` | 200/200 | 0/200 | 0/200 | 0/200 | `12.90` |

This does not estimate achievable policy performance. It rules out the claim
that the action space, opportunity frequency, or challenge mechanics make a
60% pass rate with zero blows impossible.

### Completed V25 outcomes

| Evaluation | Pass | Blow | Timeout | Near-blow timeout | Mean terminal PnL |
|---|---:|---:|---:|---:|---:|
| Training, start `-$1,500` | 16/200 | 4/200 | 180/200 | 136 | `-$1,607` |
| Standard teacher-free, start `$0` | 24/200 | 0/200 | 176/200 | 113 | `-$992` |
| Balance teacher-free, start `-$1,500` | 10/200 | 1/200 | 189/200 | 162 | `-$2,076` |

The frozen V21 standard teacher-free baseline produced 25/200 passes, zero
blows, 99 near-blow timeouts, and mean terminal PnL of approximately `-$560`.
V25 therefore did not retain the accepted baseline: it lost one pass, added
fourteen near-blows, and worsened mean PnL by about `$432` in the standard
evaluation. Its balance-stress result was worse still.

The economic signal exists in the data:

- V25 standard passes averaged `+$6,143`, `72.2` trades, and `+0.391R`
  expectancy per trade; standard timeouts averaged `-$1,965`, `47.6` trades,
  and `-0.286R` expectancy.
- Balance passes averaged `+$6,204`, `65.8` trades, and `+0.469R`
  expectancy; balance timeouts averaged `-$2,505`, `25.9` trades, and
  `-0.423R` expectancy.
- The training loop replayed 772 balance-pass sequences and 772 matched
  balance-outcome pairs. The failure is not a lack of replay calls.

Evidence:

- [V25 training diagnostics](../../runs/historical_mask_expansion_anchored_regime_stage2_v25_preserve_opportunity_200ep/campaign-runs/historical-mask-expansion-anchored-regime-stage2-v25-preserve-opportunity-200ep-r1/pcgrad_preserve_opportunity_200ep/attempt-1/training-diagnostics.jsonl)
- [V25 standard validation](../../runs/historical_mask_expansion_anchored_regime_stage2_v25_preserve_opportunity_200ep/campaign-runs/historical-mask-expansion-anchored-regime-stage2-v25-preserve-opportunity-200ep-r1/pcgrad_preserve_opportunity_200ep/attempt-1/validation-diagnostics.jsonl)
- [V25 balance validation](../../runs/historical_mask_expansion_anchored_regime_stage2_v25_preserve_opportunity_200ep/campaign-runs/historical-mask-expansion-anchored-regime-stage2-v25-preserve-opportunity-200ep-r1/pcgrad_preserve_opportunity_200ep/attempt-1/balance-validation-diagnostics.jsonl)
- [V21 teacher-free baseline](../../runs/historical_mask_expansion_anchored_regime_stage2_v21_paired_recurrent_aplus_200ep/post-short-circuit-validation-r1/validation-diagnostics.jsonl)

### Integrity and recurrent mechanics are ruled out

The episode-200 frozen audit authenticated all 52 replay shards, found zero
winner/failure label violations, placed all audited anchors at the learning
boundary after burn-in, and measured exact checkpoint round-trip Q parity and
full-versus-split recurrent parity within `4.77e-7`. The audited economic
population contained 252 Long winners, 1,269 Long failures, 245 Short winners,
and 1,324 Short failures. The data and recurrent-sequence boundary are usable.

The same audit showed the local decision-boundary symptom:

- `chop/WAIT` gradient norm: `4.542`;
- paired-winner norm: `1.457`;
- paired-failure norm: `0.390`;
- `chop/WAIT` cosine with paired winner: `-0.795`;
- `chop/WAIT` cosine with exact action: `-0.867`;
- one configured learning-rate step from each component changed zero of eight
  audited greedy actions.

PCGrad resolved the measured gradient conflict mechanically, but the complete
teacher-free economics regressed. Increasing only the winner multiplier in
V26 changes force, not the target horizon.

The enhanced frozen audit then confirmed both root-cause seams on all 52
authenticated shards:

- pass episode return averaged `+3.404`, versus `-1.182` for near-blow
  timeouts and `-1.993` for blows under the frozen challenge discount;
- 138 authenticated economic-winner rows incorrectly carried dominant-chop
  `WAIT` pressure, with membership mass `56.39`;
- restricting dominant-chop `WAIT` pressure to exact economic failures removes
  all winner mass while retaining 974 failure rows and mass `465.45`;
- a `0.05` challenge-return self-imitation weight produced a bounded mean
  audited winner bonus of `0.0717` and maximum `0.1201`; and
- 274 of 3,090 raw labeled returns exceeded the C51 support, so raw return
  replacement is rejected. Only the bounded bonus is added before the existing
  categorical projection, with newly clipped rows reported during training.

These are regression-suite checks in
[audit_frozen_checkpoint_batch.py](../../scripts/audit_frozen_checkpoint_batch.py),
not one-off campaign assumptions.

### The primary C51 objective is too short for the challenge

One challenge can contain `30 * 480 = 14,400` bar decisions. The current
recipe uses `gamma = 0.997` and an eight-step target
([V25 config](../../config/historical_mask_expansion_anchored_regime_stage2_v25_preserve_opportunity_200ep.json)).
At that discount:

| Distance | Remaining weight |
|---:|---:|
| 8 bars | `0.9763` |
| 231 bars | approximately `0.50` |
| 480 bars / one day | `0.2364` |
| 1,000 bars | `0.0496` |
| 5,000 bars | `2.99e-7` |
| 14,400 bars / 30 days | `1.62e-19` |

The environment correctly emits dense equity-change reward and terminal
pass/blow/timeout rewards
([environment.py](../../src/propevolve/environment.py#L628-L650)). The learner,
however, constructs its C51 targets using discounted eight-step rewards
([agent.py](../../src/propevolve/agent.py#L1733-L1775)). Consequently, the
local 150-bar `+2R-before-1R` entry target fits the learned horizon, while an
early decision's effect on the actual 30-day pass outcome is effectively
absent from its value target.

This explains the complete evidence without invoking a data bug:

1. The agent can improve local Expansion + Regime Entry/WAIT discrimination.
2. Pass episodes can have excellent win rate, winner size, and expectancy.
3. Most other episodes can still accumulate losses, reach low headroom, and
   time out because challenge success is not credited across the episode.
4. Increasing a local winner margin can make entries stronger, but it cannot
   teach which complete sequence of decisions converts to a pass.

### Existing balance contrast is still a local label

`sample_balance_outcome_contrast_pairs` selects a local `+2R` winner from a
pass episode and a local `+2R` failure from a near-blow episode
([replay.py](../../src/propevolve/replay.py#L2056-L2120)). It then feeds those
anchors to the same local margin:

```text
good = Q(correct side) - Q(WAIT)
bad  = Q(failed side) - Q(WAIT)
```

The loss independently raises the local winner, lowers the local failure, and
ranks winner over failure
([agent.py](../../src/propevolve/agent.py#L776-L804)). Episode outcome chooses
which pools supply examples, but episode return does not enter the target.
Thus the mechanism does not solve long-horizon pass credit. V26 raises the
winner coefficient inside this same target.

## Primary-source method comparison

### Recommended principle: SAIL-style challenge-return self-imitation

[Self-Imitation Advantage Learning](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p501.pdf)
was designed for off-policy Q learners with implicit policies. It modifies the
reward used by the existing TD target:

```text
r_tilde = r_t + alpha * (
    max(G_t, Q_target(s_t, a_t))
    - max_a Q_target(s_t, a)
)
```

The optimistic maximum prevents stale historical returns from lowering an
action that the current Q-function already values more highly. SAIL requires
neither a separate actor nor a value head and was evaluated with value-based
and distributional agents. This is a closer fit to recurrent C51 than replacing
the learner with an actor-critic. The authors specifically position it as
self-imitation for off-policy action-value learning
([paper abstract](https://arxiv.org/abs/2012.11989)).

Canonical SAIL uses the RL return `G_t`. Therefore PropEvolve must first freeze
a pass-aligned finite-horizon return definition. Feeding SAIL a return still
discounted by `0.997` would preserve the current defect. The adaptation must
also project the modified target into the existing C51 support and audit atom
saturation; [C51](https://proceedings.mlr.press/v70/bellemare17a.html) learns a
categorical return distribution, so silently clipping a newly enlarged return
would invalidate the experiment.

### Self-imitation, AWR, AWAC, and IQL

[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html)
reinforces past actions only when their observed returns exceed the current
value estimate. That is the right selective principle, but its published
algorithm is actor-critic.

[AWR](https://arxiv.org/abs/1910.00177) fits a value function and an
advantage-weighted policy; [AWAC](https://arxiv.org/abs/2006.09359) is also an
actor-critic method; and [IQL](https://openreview.net/pdf?id=68n2s9ZJWF8)
adds an expectile value function and extracts a policy with advantage-weighted
behavioral cloning. Their principles are relevant, but wholesale adoption
would add actor/value machinery and replace the current implicit C51 policy.
SAIL is the smaller value-based translation.

### Return-conditioned sequence models

[Decision Transformer](https://arxiv.org/abs/2106.01345) conditions a new
autoregressive policy on desired return, state, and action history. It changes
the architecture and inference contract. More importantly, research on
[the limits of return-conditioned supervised learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a2f65c9d2313b71005e600bd23393fe-Abstract-Conference.html)
shows that it needs stronger assumptions than dynamic-programming methods and
can fail to stitch useful behavior from suboptimal trajectories. Naively
cloning complete passing episodes can also reward accidental or harmful actions
that happened to coexist with later winners.

### CQL and IQL-style offline conservatism

[CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
addresses overestimated out-of-distribution actions in a static offline
dataset by learning a conservative lower-bound Q-function. PropEvolve has a
small fully exercised action space, simulator interaction, authenticated
counterexamples, and a demonstrated excess-WAIT problem. Additional generic
pessimism is aimed at the wrong boundary and could worsen opportunity recall.

### DQfD, R2D2, and R2D3

[DQfD](https://arxiv.org/abs/1704.03732) combines TD learning with supervised
large-margin demonstration loss. PropEvolve already implements that local
action-margin principle. Another coefficient increase does not add
challenge-level return credit.

[R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) supports the current choices
of complete recurrent sequences, burn-in, n-step learning, and sequence
priority based on maximum-plus-mean TD error. PropEvolve's parity audit shows
that its recurrent mechanics are sound. R2D2-style sequence priority may
improve sample efficiency later, but it cannot correct the economic objective
by itself.

[R2D3](https://openreview.net/pdf?id=SygKyeHKDH) mixes recurrent demonstration
and agent replay for sparse-reward tasks and shows that the demonstration
ratio is a sensitive hyperparameter; lower ratios often worked better. This
argues for a sparse self-imitation cadence, not flooding replay with passes.

### RUDDER

[RUDDER](https://papers.nips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html)
directly addresses delayed rewards by learning a return-decomposition model
and redistributing delayed reward to causally important events. It supports
the diagnosis, but adds another recurrent predictor and attribution system.
That is not the smallest correction and violates the current no-new-model
boundary.

## One falsifiable experiment

### Completed precondition: frozen return-to-go audit

The frozen V25 checkpoint and authenticated replay were used to measure, at
exact flat-action anchors:

1. challenge-horizon return-to-go and `Q(chosen action) - max Q(other actions)`;
2. pass, nonnegative-timeout, near-blow-timeout, and blow cohorts separately;
3. Long and Short separately;
4. matched normalized Expansion, Regime, account/headroom, and recurrent
   context;
5. rank correlation of current Q advantage with local `+2R` truth versus
   challenge return-to-go; and
6. the C51 categorical projection saturation rate under the proposed target.

The hypothesis is confirmed only if actions in successful challenge paths have
materially higher challenge returns than matched failures while current Q
advantages track local `+2R` labels more strongly than challenge outcome. This
is the missing evidence needed to distinguish temporal credit failure from an
unlearnable state representation.

### Candidate: challenge-return SAIL ablation

Use the frozen V21 parent and hold the network, action space, recurrent state,
burn-in, replay population, exact Entry labels, chop/WAIT safety losses,
optimizer, episode schedule, data, costs, and teacher-free evaluator fixed.

Change one mechanism:

- remove the V25/V26 PCGrad and fixed `3x` winner-weight experiment as the
  tested variable;
- define one config-driven finite-horizon challenge return `G_t` on the exact
  pass/blow/timeout reward contract;
- add a bounded SAIL bonus to the existing categorical C51 target at the
  existing sparse balance-replay cadence;
- use `max(G_t, Q_target(s_t,a_t))` so stale passes cannot lower a currently
  better action;
- report bonus rows, magnitude, return cohort, side, headroom, greedy-action
  changes, and categorical-support clipping;
- perform zero teacher/return lookup during validation or inference.

The return horizon is part of the frozen hypothesis, not a later tuning knob.
One defensible audit value is the per-bar discount whose half-life equals the
30-day challenge:

```text
gamma_challenge = 0.5 ** (1 / (episode_days * bars_per_day))
                = 0.999951866... for 30 * 480 bars
```

This retains half of a terminal outcome at the earliest decision instead of
`1.62e-19`. It is still conservative relative to the formally undiscounted
finite-horizon pass objective. The frozen projection audit must reject this
value before training if it causes material C51 support saturation.

### Acceptance and falsification

Minimum teacher-free evidence:

- standard validation beats V21's 25/200 passes;
- standard near-blows are below 99/200;
- balance validation beats V25's 10/200 passes;
- balance near-blows are below 162/200;
- zero blows in both evaluations;
- Long and Short remain active;
- pass-return SAIL advantage is positive for successful matched actions and
  absent for stale/non-improving actions; and
- standard Entry/chop behavior does not regress.

Reject the mechanism if the audit shows no challenge-return/Q mismatch, if
categorical targets saturate, if the bonus reinforces actions that matched
failures disprove, or if teacher-free pass and near-blow economics do not
improve. If the return target is correct but learning remains too slow, the
next isolated ablation is R2D2 maximum-plus-mean sequence priority—not another
winner-weight increase.

## Lifecycle decision

- Severity: `PERFORMANCE_FAILURE / P1_NO_PROMOTE`.
- First failed boundary: economic-policy temporal credit assignment.
- V25: valid, completed, and falsified; do not promote.
- V26: tests a local symptom, not the first failed boundary; do not treat it as
  the pass-rate solution.
- Authorized next experiment: the frozen return-to-go audit followed, only if
  confirmed, by one challenge-return SAIL ablation.
- Not authorized by this evidence: new A+ head, actor-critic rewrite, CQL/IQL
  migration, Decision Transformer, hard gate, Trend teacher, larger network,
  higher replay volume, or another fixed local margin increase.

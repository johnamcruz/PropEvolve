# PropEvolve Stage 2B Recovery Failure Research

Status: research decision only. This document does not authorize a code,
configuration, checkpoint, validation, or launch change.

Date: 2026-08-22

## Research question

Why can the Stage 2B recovery supervisor show high same-state
`WAIT`/`LONG`/`SHORT` target concurrence while the learned policy still blows,
fails to recover from negative realized PnL, or recovers to breakeven and then
relapses?

The required behavior is:

1. start with negative realized PnL;
2. select safe, productive A+ Expansion + Regime opportunities until realized
   PnL reaches breakeven;
3. hand control to the frozen V21 policy at breakeven;
4. retain breakeven without a relapse and continue toward the ordinary
   `+$6,000` target;
5. validate with the native recurrent policy, epsilon zero, and no teacher at
   inference.

The frozen constraints are unchanged: no new model head, no hard entry gate,
no PPO or constrained-policy migration, no inference-time teacher, and no
relabeling of genuine V21/recovery winners because of a later failure.

## Executive conclusion

The evidence points to a **target-policy and target-state problem**, not a lack
of epochs, replay volume, model capacity, or a missing action margin.

The current full-action target forces one action at a selected recovery state,
then follows the frozen V21 policy for every subsequent negative-PnL decision.
The deployed system does something different: it follows the learned recovery
policy while realized PnL is negative and switches to frozen V21 only at
breakeven. The supervisor therefore estimates

```text
forced action -> frozen V21 continuation
```

while execution uses

```text
forced action -> recovery-policy continuation while PnL < 0
              -> frozen V21 continuation while PnL >= 0
```

High agreement with the first ranking does not establish good economic behavior
under the second policy. More training against that target can strengthen the
wrong fixed point.

Three additional implementation details make concurrence look stronger than
the economic supervision actually is:

- target selection equates `outcome == pass` with recovery success, so a
  timeout that successfully reached breakeven is selected as a failure cohort;
- target selection prefers the smallest-headroom, latest anchor, where all
  three actions may already blow and therefore contain no useful decision
  boundary;
- top-1 concurrence uses ordinary `argmax`; tied targets choose the first action
  (`WAIT`) even when every branch has the same value. An all-actions-blow vector
  can therefore record a concurrence hit without teaching a safe action.

The smallest correct next step is not another loss-weight or margin change. It
is a frozen offline audit of target validity, followed by one TDD correction:
evaluate forced first actions under a frozen snapshot of the **actual composite
continuation policy**, generate gradients only from economically discriminative
states, and keep post-breakeven retention credit at the handoff boundary instead
of blaming pre-breakeven winners.

## Local evidence from the stopped matched runs

The local evidence is strong enough to reject another unmodified training run.
The run identifiers below are local experiment identities, not promotion
claims.

### Recovery is occurring, but retention is not

Run `r8` stopped after 69 training episodes with 5 passes, 44 blows, and 20
timeouts. Twenty-five episodes reached breakeven at least once. Of those 25:

- only 2 remained nonnegative through the end of the episode;
- 23 relapsed below zero after first recovery;
- all 5 passes relapsed at least once before eventually passing;
- 6 relapsed but still finished as positive timeouts;
- 12 relapsed and finished as negative timeouts.

All 44 blows occurred before a first recovery handoff. This separates two real
problems that must not share a mislabeled target:

1. **pre-recovery safety and competence** — reach zero without blowing;
2. **post-recovery retention** — do not return negative after reaching zero.

The current recovery status records the first event but does not measure the
second. Consequently, a high recovery-success count can coexist with severe
giveback.

### Replay volume is already high enough to falsify “more replay”

The relevant replay loops executed hundreds of times:

| Run | Successful-recovery replays | Post-recovery contrast pairs | Newly promoted retained episodes | Newly promoted contrast episodes |
|---|---:|---:|---:|---:|
| `r8` | 248 | 248 | 2 | 11 |
| `r9` | 392 | 392 | 1 | 15 |
| `r10` | 208 | 208 | 1 | 6 |

The learner was not starved of replay calls. It repeatedly rehearsed a very
small and partly misclassified target population. Increasing replay cadence
would amplify that target error.

### Episode outcome is misclassifying recovery targets

Target anchors were marked successful only when the public episode outcome was
`pass`:

| Run | Target anchors | Preferred success | Preferred failure | No preference |
|---|---:|---:|---:|---:|
| `r8` | 69 | 6 | 53 | 10 |
| `r9` | 105 | 6 | 74 | 25 |
| `r10` | 59 | 3 | 41 | 15 |

In `r8`, every pass was assigned a success target, while all 20 timeouts were
assigned failure targets—even though all 20 had reached breakeven and 8
finished positive. This directly demonstrates that `outcome == pass` is not a
valid proxy for the Stage 2B recovery boundary.

### Most full-action targets contain no action decision boundary

The stopped `r8` checkpoint contained 65 stored full-action recovery targets:

- 48 targets (74%) had all three action values equal;
- 36 were the all-actions-blow vector `[-1, -1, -1]`;
- only 12 had one unique best action;
- unique best actions were `WAIT: 3`, `LONG: 1`, and `SHORT: 8`.

The same failure persisted in the later runs:

| Run | Stored targets | All-action ties | All-actions-blow | Unique best |
|---|---:|---:|---:|---:|
| `r8` | 65 | 48 | 36 | 12 |
| `r9` | 100 | 74 | 60 | 20 |
| `r10` | 55 | 43 | 39 | 8 |

On a tie, the action margin contributes no ranking information. The KL target
teaches equal action values, and ordinary `argmax` nevertheless reports `WAIT`
as top-1 because `WAIT` is action index zero. Therefore the observed high
top-1 concurrence is not proof that recovery discrimination was learned.

These four observations jointly reject the hypotheses that the immediate fix
is more epochs, higher replay volume, a larger margin, or more model capacity.

## What the current code actually teaches

### 1. Execution uses a composite policy

`RecoveryHandoffPolicy` routes negative realized PnL to the recovery policy and
nonnegative realized PnL to frozen V21. It reconstructs the newly active
policy's recurrent state from the causal prefix at a handoff
([`src/propevolve/recovery.py`](../../src/propevolve/recovery.py),
`RecoveryHandoffPolicy`). This is the policy whose economics matter.

### 2. Target rollouts use only frozen V21 after the forced action

`build_recovery_value_target` receives one `policy`, forces `WAIT`, `LONG`, or
`SHORT`, and then calls that same policy for every continuation decision until
breakeven, blow, or horizon
([`src/propevolve/recovery.py`](../../src/propevolve/recovery.py),
`build_recovery_value_target`). Training passes `recovery_value_policy`, which
is the frozen checkpoint, not the evolving recovery policy
([`src/propevolve/training.py`](../../src/propevolve/training.py), recovery
target construction).

This is a counterfactual rollout-policy mismatch. The teacher answers which
first action is best if frozen V21 controls the rest of the recovery, but the
candidate is evaluated under its own recovery decisions until zero.

This distinction is fundamental in off-policy return estimation. Retrace was
derived specifically because uncorrected multi-step returns can target the wrong
policy when behavior and target policies differ; its trace coefficients account
for that discrepancy while retaining useful near-policy returns
([Munos et al., 2016](https://papers.nips.cc/paper_files/paper/2016/hash/c3992e9a68c5ae12bd18488bc579b30d-Abstract.html)).
PropEvolve does not need to import Retrace to fix this deterministic simulator
seam, but it must evaluate the policy it will actually execute.

### 3. Recovery success is selected using the final pass outcome

The training seam calls:

```python
select_recovery_target_prefix(..., prefer_success=outcome == "pass")
```

A recovered episode that finishes as a timeout is therefore treated as a
failure for anchor preference even though Stage 2B's recovery boundary was
successfully crossed. `pass`/`blow`/`timeout` and
`recovered`/`not_recovered` are intentionally separate contracts; target-state
selection currently collapses them.

### 4. The anchor is biased toward the latest, lowest-headroom state

`select_recovery_target_prefix` ranks candidates by minimum headroom and then
latest index. This tends to select decisions close to MLL. If all forced actions
blow from such a late state, the vector is `[-1, -1, -1]`: a valid record of an
unrecoverable state, but not a useful action-ranking target. The decision that
caused the account to enter that unrecoverable region occurred earlier.

The correct training anchor is a state where the same causal prefix and market
path produce different economic outcomes across feasible actions. That is a
counterfactual decision boundary, not a hard-coded Expansion, Regime, balance,
or ticker threshold.

### 5. Top-1 concurrence is not tie-aware

`recovery_value_top1_concurrence` compares two `argmax` indices directly. With
the native action order `WAIT`, `LONG`, `SHORT`, an equal-valued target resolves
to `WAIT`. By contrast, `recovery_action_margin` correctly returns zero when a
target lacks one unique winner. The diagnostic and the learning seam therefore
use different notions of a valid best action.

Concurrence must be reported only on discriminative targets. At minimum, the
target audit needs:

- unique-best-action fraction;
- all-actions-blow, all-actions-recover, and all-actions-equal fractions;
- top-two economic-value gap and target entropy;
- `WAIT`-best, `LONG`-best, and `SHORT`-best counts;
- concurrence conditional on a unique target winner;
- realized outcome under the actual composite continuation.

Without these, high concurrence can mean that the model agrees on many
uninformative ties rather than that it learned recovery.

### 6. The environment records first recovery, not retained recovery

When the account first reaches breakeven, the environment sets recovery status
to `recovered` and clears the recovery threshold. It does not change that status
if the account subsequently returns negative. The stress evaluator also rejects
a blown episode carrying `recovered` status instead of representing the valid
historical sequence “recovered, relapsed, then blew.”

Public outcomes must remain exactly `pass`, `blow`, and `timeout`, and public
recovery status can remain `recovered` or `not_recovered`. The missing facts are
orthogonal training and diagnostic fields:

- first breakeven index;
- first negative relapse index;
- relapse count;
- minimum realized PnL after first breakeven;
- retained through terminal;
- recovered then blown.

Without those fields, “recovery without giveback” cannot be measured directly
or used as an authenticated training boundary.

### 7. The configured supervision-start schedule is not integrated

The recovery curriculum defines a deterministic `supervision_start_state`
schedule over its configured negative starting PnLs, but target construction
always passes the single fixed curriculum start state. The schedule is not used
at the training call site. This is a config-to-execution parity defect.

It is not the first fix: changing starting-state diversity while continuation
and target semantics are wrong would confound the diagnosis. The audit should
record the actual target start state, and the dormant schedule should be either
integrated in a later matched experiment or removed from the declared recipe.

### 8. The scalar auxiliary discards the existing C51 downside distribution

PropEvolve's policy represents a categorical distribution of returns, but the
recovery auxiliary converts each action distribution to its expectation and
matches a softmax over three scalar utilities. C51 was introduced because a
return distribution contains information that an expected value discards
([Bellemare, Dabney, and Munos, 2017](https://proceedings.mlr.press/v70/bellemare17a)).

This does not prove that a new distributional loss is required. It does prove
that top-1 expected-Q concurrence cannot establish zero blow risk. Exact blow
incidence must remain an independent acceptance constraint. A later
distributional ablation is justified only after continuation, anchor, and
recurrent parity are correct.

## What the primary methods contribute—and what they do not

### EarnHFT: full-action economic supervision

EarnHFT backward-computes an optimal value for every current position and
candidate action on the same recorded future market path, including execution
costs. It adds a decaying KL loss between the learner's complete action vector
and this training-only Q teacher, while retaining ordinary DDQN TD learning
([EarnHFT paper, Section 4.1 and Algorithms 1–2](https://arxiv.org/abs/2309.12891),
[`demonstration.py`](https://github.com/TradeMaster-NTU/EarnHFT/blob/main/EarnHFT_Algorithm/tool/demonstration.py),
[`ddqn_pes_risk_aware.py`](https://github.com/TradeMaster-NTU/EarnHFT/blob/main/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py)).

The reusable idea is same-state full-action economic ranking without a teacher
at inference. EarnHFT does **not** solve PropEvolve's handoff problem: its
teacher uses an oracle-optimal continuation, its deployed networks are
feed-forward, and it does not model an MLL floor, negative-PnL recovery,
recurrent hidden-state switching, or post-breakeven relapse. The paper itself
notes that learner deviations change the position-dependent supervision and can
compound deviation; it adds optimal-actor transitions to reduce that local
trap. EarnHFT's oracle is therefore a training regularizer, not evidence that a
different deployed continuation will realize the same return.

### DQfD: demonstration pressure must remain grounded by TD economics

DQfD combines a supervised large-margin action loss with one-step TD, n-step
TD, regularization, agent experience, and prioritized replay. The authors state
that unobserved actions are otherwise ungrounded and can propagate unrealistic
values; the TD losses keep the value function Bellman-consistent
([Hester et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/download/11757/11616)).

The implication is narrow: an action margin can sharpen a valid teacher ranking,
but cannot repair a teacher generated under the wrong continuation policy or at
an unrecoverable state. The failed full-ordering experiment is consistent with
that limitation. It increased agreement pressure without fixing target
semantics.

### R2D2: recurrent parity is part of target correctness

R2D2 stores episode-bounded sequences, never crosses episode boundaries, and
uses stored recurrent state plus burn-in because parameter drift makes old
hidden states stale. It computes loss only after reconstructing a useful
current recurrent state
([Kapturowski et al., 2019](https://openreview.net/references/pdf?id=H1SpI6cKm)).

PropEvolve already preserves much of this structure. The unresolved requirement
is parity at the recovery-to-normal switch: for one exact causal prefix, the
Q-values used during live collection, replay learning, target construction,
checkpoint reload, and teacher-free validation must agree within declared
numeric tolerance for both the recovery policy and frozen V21. If they do not,
an economically correct target can still be attached to the wrong recurrent
state.

### R2D3 and prioritized replay: rare guidance helps only at the right dosage

R2D3 maintains separate agent and demonstration replay buffers and reports that
demo ratio has a dramatic effect; small but nonzero ratios worked best in its
tasks. It also reports failures on its most memory-intensive tasks and names
recurrent-state handling as a possible cause
([Paine et al., 2020](https://arxiv.org/abs/1909.01387)).

Prioritized Experience Replay warns that nonuniform sampling both reduces
diversity and introduces bias; it uses stochastic prioritization and
importance-sampling correction
([Schaul et al., 2016](https://arxiv.org/abs/1511.05952)). R2D2, however,
empirically found no benefit from importance weighting in its benchmark and
omitted it. Therefore the primary literature does not justify blindly adding or
removing importance correction. PropEvolve first needs to report the recovery
store's population distribution, sampled distribution, target age, and
per-cohort gradient mass. If balanced sampling is intended to estimate a
population objective, correct it; if it is deliberate rare guidance, declare
and ablate the dosage.

The current recovery store balances by the side and A+ economic label attached
to the anchor, not by recovery-target informativeness or actual recovery
success. This can keep stale or nondiscriminative targets active even as the
candidate changes.

### Safe and constrained RL: zero blow is a constraint, not a reward tradeoff

Constrained Policy Optimization formalizes reward maximization subject to a
separate expected-cost constraint and seeks near-constraint satisfaction during
updates
([Achiam et al., 2017](https://proceedings.mlr.press/v70/achiam17a.html)).
Risk-constrained RL similarly treats tail events through chance or CVaR
constraints rather than hiding them inside average return
([Chow et al., 2018](https://www.jmlr.org/papers/v18/15-636.html)).

Those algorithms are policy-gradient replacements and are outside the frozen
V21 design. The applicable lesson is evaluative: do not let a higher pass rate
or average PnL compensate for any validation blow. Preserve exact MLL simulation
and the zero-blow promotion gate as a separate hard acceptance condition.

## Correct credit assignment across the handoff

The episode must be decomposed by policy responsibility.

| Failure mode | Controlling policy | Correct learning boundary |
|---|---|---|
| Blows before ever reaching breakeven | Recovery policy | Same-state recovery action and recovery-policy continuation while PnL is negative |
| Reaches breakeven, then V21 returns negative | Frozen V21 at the positive decision; recovery policy after the first negative reactivation | Handoff parity diagnostic at zero; new recovery target only after the policy re-enters negative PnL |
| Reaches breakeven and stays nonnegative but times out | Recovery succeeded; ordinary V21 did not pass in remaining horizon | Valid recovery success, ordinary pass-conversion evidence |
| Reaches breakeven, retains it, and passes | Both stages succeeded | Retained checkpoint and promotion evidence |

A pre-breakeven entry that genuinely produces recovery must never be relabeled
`WAIT` solely because frozen V21 later relapses. The recovery policy did its job.
Likewise, a positive-side action selected by frozen V21 cannot be improved by a
loss applied only to the recovery candidate. Under the frozen-V21 constraint,
post-breakeven improvement can come only from:

1. fixing recurrent-state or environment parity at the handoff;
2. training the recovery candidate at the first negative state after a relapse;
3. selecting a different already-frozen V21 parent in a new research decision.

The third option is outside the current Stage 2B experiment.

### Constraint imposed by the exact-zero frozen-V21 handoff

There is one unavoidable control boundary: once realized PnL reaches zero, the
recovery policy no longer chooses actions. Therefore the recovery candidate
alone cannot guarantee that the account will never print a negative realized
PnL again if frozen V21 subsequently takes a losing trade.

Within the current contract, “recovery without giveback” must be pursued in
three precise ways:

1. teach the recovery policy to reach zero safely and in a favorable causal
   market context rather than merely touch zero from a crisis state;
2. verify that the exact handoff reconstructs frozen V21 without recurrent or
   observation drift;
3. if V21 produces a loss, reactivate recovery at the first negative state and
   train that state as a new recovery decision boundary.

A retention-aware auxiliary may compare the longer composite outcomes of
otherwise valid recovery choices, but it must not erase the genuine Expansion
entry label of an action that successfully recovered the account. The native
entry objective and the longer-horizon recovery-retention objective are two
economic labels on the same action, not substitutes for one another.

If the literal acceptance rule is “after first breakeven, realized PnL may
never become negative again,” and parity-correct frozen V21 violates it, then
one contract must change: either V21 must be fine-tuned for post-recovery
capital retention or the handoff must use a positive safety buffer/hysteresis.
That is a separate experiment requiring an explicit decision; it must not be
smuggled into the current Stage 2B correction.

## Highest-confidence correction

### Gate 0: offline target-validity audit before another training run

Use the accepted/reverted baseline and authenticated training rows. Do not
update weights.

For the exact same causal anchors, construct all three forced first-action
branches under two continuations:

1. current teacher continuation: frozen V21 at every subsequent decision;
2. deployed continuation: a frozen snapshot of the recovery candidate while
   PnL is negative, then frozen V21 at nonnegative PnL, with the same handoff
   reconstruction as inference.

Report:

- top-action and safe/unsafe outcome disagreement between the continuations;
- unique-best, tie, all-blow, all-recover, and action-value-gap distributions;
- target composition by actual `recovered` status and public outcome;
- anchor headroom, time remaining, side, ticker, and target age;
- live-versus-reconstructed Q parity at the anchor and first handoff;
- pre-recovery blow versus recovered-then-relapsed counts.

If the continuations disagree economically, the current teacher is falsified.
If recurrent parity fails, fix parity before target design. If most targets are
nondiscriminative or selected after the decisive mistake, fix anchor selection
before any new loss.

### Gate 1: one TDD target correction

Only after Gate 0 localizes the first failed boundary:

1. Snapshot the current recovery policy for target construction. It is a
   training target policy, not a new model or inference dependency.
2. Force exactly one first action from one authenticated causal state.
3. Continue greedily with the frozen recovery snapshot while realized PnL is
   negative and frozen V21 while it is nonnegative. Use the identical handoff
   and recurrent reconstruction as validation.
4. For the pre-recovery objective, stop at first breakeven, blow, or ordinary
   horizon. Do not attach later V21 relapse to the pre-recovery action label.
5. Select anchors by the Stage 2B boundary (`recovered`, `not_recovered`, blow),
   not by `outcome == pass`.
6. Generate a learning target only where feasible actions have meaningful
   economic spread. Keep all-equal/all-blow/all-recover rows as diagnostics, not
   KL or margin examples.
7. Keep ordinary V21 replay, TD loss, A+ supervision, optimizer step, and sparse
   recovery cadence unchanged.
8. Refresh the target-policy snapshot only at an explicit frozen cadence and
   record its identity on every target so target age and drift are measurable.

This is fitted policy iteration on the existing recurrent C51 policy: evaluate
first actions under a frozen version of the policy that will actually continue,
then improve that same policy. It keeps the teacher training-only and does not
add an inference component.

### Gate 2: retention boundary, without corrupting recovery labels

Retained-versus-relapsed episodes remain useful, but their role is narrower than
previously assumed:

- use them to diagnose the handoff and to identify the first negative
  reactivation after a relapse;
- preserve genuine pre-breakeven winners and V21 A+ labels unchanged;
- build any new recovery target at that reactivated negative state, where the
  recovery policy again controls the action;
- do not apply a recovery-candidate gradient to a positive state controlled by
  frozen V21.

If frozen V21 itself causes a relapse from a recurrent state that is proven
parity-correct, Stage 2B cannot eliminate that positive-state decision without
changing the frozen parent. The experiment must report that boundary honestly
rather than move blame backward.

## Required TDD seams

Before training, tests must prove:

1. The target continuation uses recovery below zero and V21 at or above zero.
2. Only the first action is forced; all branches share identical causal origin,
   market path, economics, and recurrent prefix.
3. A recovered timeout is recovery success even though its public outcome is
   `timeout`.
4. An all-equal target does not count toward top-1 concurrence and contributes
   no action-ranking auxiliary gradient.
5. A genuine pre-recovery economic winner remains an entry winner when V21
   later relapses.
6. A recovered-then-relapsed episode is anchored at the handoff for diagnostics
   and at the first negative reactivation for recovery learning—not at the old
   recovery winner.
7. Live collection, target rollout, replay reconstruction, checkpoint reload,
   and teacher-free evaluation produce matching Q-values for the same causal
   prefix and policy identity.
8. Recurrent sequences never cross an episode boundary; burn-in rows receive no
   loss; the target action lies after burn-in.
9. Target identities bind the recovery snapshot, V21 checkpoint, data/cache,
   simulator economics, causal prefix, and configuration.
10. Validation performs zero teacher lookups and exposes no target values as
    observations.
11. The complete unchanged V21 test suite still passes.

## One matched experiment

After the audit and TDD correction, run one matched candidate against the clean
reverted baseline. Freeze data, starting state, episode windows, seeds, V21
checkpoint, ordinary replay, optimizer, teacher schedule, and validation.

Primary acceptance:

- zero teacher-free validation blows;
- improved recovery rate and terminal recovery PnL from the negative start;
- no regression in frozen V21 normal-start behavior;
- lower pre-recovery blow incidence;
- retained breakeven after handoff;
- pass-rate lift is reported but cannot compensate for a blow.

Mechanism acceptance:

- high unique-target concurrence under the deployed composite continuation;
- low teacher-versus-deployed continuation disagreement;
- recurrent parity at the anchor and handoff;
- reduced raw requested unsafe entries, not merely altered executed actions;
- target age, tie rate, and sampled cohort mass remain bounded and visible.

Reject immediately if the target audit fails, V21 drifts, safety comes from an
inference gate, or a later relapse changes a genuine pre-recovery winner label.

## What not to do next

- Do not run 200 or 450 more episodes against the current frozen-V21
  continuation target.
- Do not increase replay volume, model capacity, learning rate, loss weight, or
  action margin before target validity is proven.
- Do not turn retained-versus-relapsed episodes into pre-recovery `WAIT` labels.
- Do not interpret raw top-1 concurrence without excluding ties.
- Do not add an A+ head, recovery head, Pivot/Trend gate, or inference teacher.
- Do not migrate from recurrent C51 to PPO, CPO, or another framework to solve
  this localized target error.
- Do not let pass rate or average PnL offset a validation blow.

## Decision

The next implementation should be a target-validity audit, not another training
recipe. The highest-confidence root cause is that Stage 2B currently learns a
same-state ranking under the wrong continuation policy, from anchors that can be
misclassified or too late, while its concurrence diagnostic counts
nondiscriminative ties.

Fix those three boundaries first. Then the existing recurrent C51 policy can
learn the intended rule without a new head or inference dependency:

> While negative, wait for or take the native action whose actual composite
> continuation has the best recovery economics and does not blow; after
> breakeven, hand off cleanly to frozen V21; if V21 later relapses, train the
> recovery policy from the first state where it again controls the account.

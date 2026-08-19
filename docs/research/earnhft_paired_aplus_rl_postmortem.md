# EarnHFT and PropEvolve paired A+ RL postmortem

## Decision

Do not restore the rejected v21 full-action opportunity-value supervisor and
do not add an A+ model head.  Preserve the e7/v19 recurrent C51 policy and its
exact `+2R-before-1R` Entry contract.

The smallest supported next mechanism is **true paired recurrent replay**:
replace the existing positive-only Entry opportunity slots with same-side
winner/failure sequence pairs selected by continuous Expansion and Regime
context similarity.  Apply the existing paired Q-advantage loss at the two
authenticated learning anchors while both sequences continue to receive their
ordinary C51 TD updates.

This mechanism is implemented behind the explicit
`paired_recurrent_a_plus_expansion_regime_contrastive_v7` training semantics
and `paired_recurrent_long_short_v1` replay contract for one matched ablation.
It is not yet an accepted or promoted Stage 2A policy; teacher-free economics
remain the deciding evidence.

## Scope boundary: retain recurrent RL

PropEvolve should not adopt EarnHFT's policy pool, minute-level router, or
feed-forward transition replay.  Those components address second-level HFT
trajectory length and market-drift routing, not PropEvolve's narrower problem
of learning which Expansion setups deserve `LONG` or `SHORT` instead of
`WAIT`.

The transferable unit is one training principle: compare the native action
values economically while ordinary RL remains in charge.  In PropEvolve that
principle must operate on its existing recurrent C51 learner:

- sample complete episode-bounded sequences;
- reconstruct recurrent state through the unchanged burn-in prefix;
- identify a causal decision anchor inside the learning portion;
- apply ordinary distributional TD to both complete sequences;
- compare `Q(side) - Q(WAIT)` only at the authenticated winner/failure
  anchors; and
- remove all Expansion, Regime, and exact-outcome supervision at evaluation.

No new model, policy head, router, hard entry gate, or inference dependency is
part of this proposal.

## What EarnHFT actually implements

EarnHFT constructs a backward dynamic-programming table indexed by time,
previous inventory, and next inventory.  Its learner receives the full vector
for one state, combines ordinary TD error with KL divergence from that vector,
and evaluates without the table:

- [DP table construction](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/tool/demonstration.py#L191-L288)
- [TD plus KL learner](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L307-L354)
- [teacher-free action selection](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L356-L387)

The useful principle is dense relative action comparison combined with TD.
The table itself is not portable to PropEvolve: EarnHFT's actions are inventory
levels in one known price path, whereas PropEvolve decides discretionary
`WAIT`/Long/Short under a path-dependent prop challenge.

The implementation adds four details that matter when interpreting the idea:

1. The Q-teacher is indexed by the *same timestamp and current inventory* and
   gives values for every next-inventory action.  Its action comparisons are
   internally consistent; they are not outcomes spliced from unrelated rows.
2. The learner uses both optimal-actor trajectories and epsilon-greedy
   trajectories.  Both receive ordinary n-step TD and the decaying Q-teacher
   KL term.  The teacher therefore accelerates one RL policy; it does not
   replace RL.
3. The published low-level action space is a finite long-only inventory grid,
   not PropEvolve's `WAIT`/Long/Short entry decision.  EarnHFT does not solve
   PropEvolve's side symmetry or dominant-chop problem for us.
4. The paper explicitly reports that the profit-only optimal supervisor makes
   EarnHFT relatively aggressive and can weaken risk performance.  Copying its
   dense profit target without PropEvolve's chop and challenge-risk objectives
   would move in the wrong direction.

Primary sources: [EarnHFT paper](https://ojs.aaai.org/index.php/AAAI/article/download/29384/30614),
[low-level training loop](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/agent/low_level/ddqn_pes_risk_aware.py#L410-L597),
and [n-step replay](https://github.com/TradeMaster-NTU/EarnHFT/blob/0e1e11a6d9aff70efb1807baa3416429568deb31/EarnHFT_Algorithm/RL/util/replay_buffer_DQN.py#L256-L424).

EarnHFT's later hierarchy is intentionally out of scope.  It trains many
low-level agents using different return-conditioned chunk preferences, selects
profitable specialists on labelled validation regimes, and trains a separate
minute-level DDQN router over that fixed pool.  This is a multi-policy
non-stationarity solution, not an A+ setup-learning mechanism for one recurrent
policy.

## Why the v21 adaptation failed

V21 generated forced Long and Short continuation/economic outcomes for every
eligible event row and trained KL against `[WAIT, Long, Short]`.  That was not
one internally consistent challenge-state value table.  It could reward a
forced side on a row where the existing dominant-chop objective required
`WAIT`, creating opposing gradients.

The matched result rejected the mechanism: through episode 78, v21 R2 had
seven passes, one recorded blow in the frozen document (the retained JSONL
later contains two by episode 80), and 24 terminal near-blow timeouts.  The
matched e7/v19 run had 18 passes and zero blows through episode 78.  V21 also
retained asymmetric dominant-chop false entries, especially Short.

This is the exact falsifier declared in the research contract: supervised
agreement improved while teacher-free economics regressed.

## What v19 paired A+ currently does

V19 is RL from demonstrations: C51 TD remains primary and the exact Entry,
chop, failed-confluence, and paired losses directly shape the same native
`Q(WAIT)`, `Q(LONG)`, and `Q(SHORT)` values.

The paired loss is:

```text
good_advantage = Q(correct side on winner) - Q(WAIT)
bad_advantage  = Q(same side on failure)   - Q(WAIT)
loss = softplus(margin + bad_advantage - good_advantage)
```

However, the executable `paired_a_plus_rank_loss` forms a Cartesian product of
all valid and failed rows sharing original side and one Regime component in the
already sampled batch.  Expansion probabilities affect valid/failed membership
mass, but they do not make two rows context-neighbors.  Therefore the code does
not guarantee the documented comparison of a winner with a similar failure.

The replay sampler makes this worse: four of 16 sequence slots are fixed exact
positive Entry anchors.  Failed-confluence anchors are not co-sampled as their
matched controls.  A nonzero aggregate pair count proves loss activity, not
that the policy saw apples-to-apples A+ contrasts.

## What v20 established

V20 R1 required the same ticker and original side.  It reduced paired mass by
about 77 percent and was rejected.  The generic-canonical follow-up removed the
ticker gate but also mixed original Long and Short failures, added headroom
similarity, and greatly increased pair mass.  At episode 26 it had five passes
versus six for matched v19, while sampled Short precision was 28.8 percent.
It was incomplete and did not prove economic lift.

The valid idea and the invalid bundle were never isolated.  The untested
middle is:

- preserve original side: Long winner versus Long failure, Short versus Short;
- share one formula across sides, but do not cross-pair them;
- allow cross-ticker candidates without a ticker gate;
- match continuously on fit-normalized candidate/opposite Expansion geometry
  and the complete three-state Regime vector;
- exclude account headroom from A+ identity; it remains a causal RL input;
- sample the two complete recurrent sequences together with unchanged burn-in.

## Primary-source RL constraints

- DQfD combines TD and a large-margin demonstration loss because either alone
  is insufficient; PropEvolve already has both mechanisms.  A larger margin
  does not repair unrelated or missing pairs.
- R2D2 learns from complete recurrent sequences after burn-in and prioritizes
  sequences using maximum-plus-mean TD error.  PropEvolve already preserves
  sequence boundaries and burn-in; do not replace them with transition replay.
- R2D3 shows that demonstration mixtures can be useful at rare dosages.  This
  argues against blindly increasing fixed teacher/opportunity replay mass.

Sources:

- [DQfD paper](https://research.google/pubs/deep-q-learning-from-demonstrations/)
- [Acme R2D2 learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py)
- [R2D3 paper](https://openreview.net/pdf?id=SygKyeHKDH)

## Smallest valid next experiment

### Fixed-batch audit before training

From one frozen e7/v19 checkpoint and deterministic authenticated recurrent
batches:

1. Build same-side positive/failed *recurrent sequence* pairs from pre-2025
   exact Entry rows.
2. Use continuous Expansion plus full Regime similarity without hard cutoffs;
   these select comparable training examples but never become policy gates.
3. Preserve episode boundaries and identical burn-in/learning masks, and place
   both decision anchors inside the learning portion.
4. Compare current Cartesian pairing with true co-sampled recurrent pairs.
5. Record pair coverage by ticker and side, effective pair mass, good/bad Q
   advantages, TD and auxiliary gradient norms/cosines, and greedy action
   changes.
6. Reject the mechanism before a campaign if it collapses either side, drives
   universal WAIT, or opposes TD/exact-action gradients persistently.

### Matched campaign only if the audit passes

Keep the e7/v19 parent, network, labels, optimizer, seed, costs, risk, episode
budget, and teacher-free evaluator unchanged.  Change only the Entry replay
slots from positive-only anchors to balanced same-side winner/failure pairs and
apply the existing paired loss at their anchors.  Do not add full-action KL,
an A+ head, a hard gate, new margins, or Stage 2B recovery.

Acceptance remains economic: zero validation blows, fewer near-blows, higher
pass conversion than v19, both sides active, improved Entry precision without
destroying opportunity recall, and monotonic native Q advantage versus exact
economic truth.  Training loss or pair mass alone cannot promote the result.

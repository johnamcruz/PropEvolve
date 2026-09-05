# V40 retention preflight — no sweep launch until evidence passes

User-authorized correction, 2026-09-04. Preserve Expansion/Regime/Trend labels,
execution, safety, C51 and recurrent architecture. Preserve management retention.
Use real frozen training replay for diagnosis; inspected validation is development
evidence, never untouched confirmation. No change to the 60% pass / zero observed
blow / low near-blow objective.

## Reproduced failure and controls

Trial 32 checkpoint at episode 45, 43 authenticated shards, two economic pairs
(one per side), 128 repeated updates then 32 fixed mixed batches of 16 sequences.
Identical source checkpoint and draw across matched arms; CPU/PyTorch.
Teacher imitation is zero, exact entry scale one, healthy-entry retention enabled.

| Arm | Long winner minus WAIT after mix | Short winner minus WAIT after mix | All six margins pass |
| --- | ---: | ---: | --- |
| Original retention | -0.004703 | 0.780625 | No |
| All retention disabled (diagnostic only) | 1.404901 | 0.871635 | Yes |
| Only healthy-entry retention disabled | 1.331276 | 0.852503 | Yes |
| Exclude contradictory labelled anchor rows only | -0.049492 | 0.405112 | No |
| Retain only verified compatible entry rows | 0.594864 | 0.698147 | Yes |

All arms can acquire the selected boundaries. The first narrow correction fails
preservation and is rejected. Management-only retention passing localizes the
interference to entry retention, not inability to learn these examples.

## Current candidate

When exact entry supervision is active, retain old entry behavior only on valid
economically labelled rows where the anchor satisfies the configured action
margin. Wrong, weak and unlabelled anchor entry rankings are not verified
competence. Existing entry objectives teach the replacement. Management
retention and the legacy unsupervised path remain unchanged. Labels are authority,
not example recency. No new observations, model, loss weight or hard entry gate.

TDD: old code retained all eight contradictory rows for each WAIT/Long/Short
target; corrected code excludes them. A second red test reproduces unlabelled
mixed-row retention. Tests cover correct, incorrect and insufficient-margin
anchors on PyTorch and MLX, checkpoint reload and matched real optimizer updates.

## Decisive checks

- Repeat the matched Trial 32 diagnostic after the candidate.
- Exercise additional real preserved Trial 33 checkpoint/replay with identical
  bounded budget, autonomous teacher scale zero, without claiming this equals
  its original intermediate curriculum scale.
- Check each winner against WAIT and opposite side and each failure against WAIT;
  report acquisition, mixed replay and any loss of learned margins.
- Check teacher-free stripping, recurrent parity, exact label/anchor authentication.
- Run recovery, audit, economic replay E2E, learner, training, replay checkpoint,
  environment/execution and MLX regression suites.
- Commit/push and V40 launch only after proof. If a check fails, report it and
  revise the bounded experiment, not silently weaken acceptance.

Local results live under `runs/diagnostics/`: `v39-frozen-acquisition-r1`,
`v39-management-retention-r1`, `v39-compatible-retention-r1`,
`v39-verified-retention-r1`, and `v39-verified-retention-trial033`.
No private checkpoint or replay data belongs in Git. Earlier compacted trial
summaries cannot replace missing original recurrent tensors. Small real-cohort
learnability is not proof of 60% pass probability or zero future losses.

## Matched Trial 32 candidate result

The verified-only candidate acquired all six margins at update 28 and preserved
them at every one of the 32 mixed updates. Final WAIT-minus-failure margins were
1.016066 Long and 2.741893 Short. Winner-minus-opposite margins were 2.715606 Long
and 2.767320 Short. Teacher-free stripping Q drift was zero. The initial Long
failure ranking was already wrong, so failure rejection cannot be described as
correct from initialization; inspect its correction trajectory separately.

This passes the original preservation regression, not the overall launch gate.

## Additional Trial 33 regression: failed

All 20 shards authenticated with zero economic-label violations and contiguous
recurrent sources. The candidate acquired all six margins at update 35. After
128 acquisition updates, winner-over-WAIT was Long 5.154964, Short 2.938918.
After 32 ordinary mixed updates these became Long -0.643173, Short -2.234854.
The configured target margin for this trial is 0.40. Failure rejection remained
correct throughout; teacher-free Q drift was zero. Mixed-replay margin loss
started at update 6. This is a real preservation failure, not a serialization or
teacher-removal discrepancy.

Decision: NO V40 LAUNCH and no commit/push as a completed fix yet. Run the
management-only retention control on Trial 33 to distinguish residual retention
interference from other mixed-replay effects. These stress tests use ordinary
sampled batches, not full campaign continuations with all sparse replay additions;
do not misrepresent them as a reproduction of every production scheduling path.

Affected E2E/learner/training/replay/execution/MLX regression suite: 317 passed on
the revised implementation. Recovery and audit tests also pass after adapting
the retained-anchor fixture to use verified failure competence rather than random
unlabelled predictions. Passing controlled tests is not a substitute for the
failed additional real-data stress test.

Trial 33 management-only control also failed mixed preservation (Long -1.588084,
Short -1.843343). This residual is not explained by entry retention alone.
Next bounded diagnostic reads the existing trial's violation-replay cadence and
candidate counts directly from its JSON training configuration. It reuses the
production violation selector, appending selected sequences at the declared
cadence, with fixed candidate draws across matched arms. This tests whether the
existing sparse replay path protects learned boundaries; it does not add a new
production replay mechanism or claim all campaign replay additions are included.
The audit-to-production-call E2E comparison for this path passed after a red test.

The first scheduled diagnostic failed while serializing the final report due to
an audit-local variable-name collision. A complete CLI test using real replay
serialization and actual learner updates reproduced that exact exception and
passed after correction. The rerun is preserved separately at
`runs/diagnostics/v39-scheduled-retention-trial033-r2`. No inference about its
learning result is permitted until a complete report exists.

Current learner source Git blob: `e7e8cb480848263c363b08ea486da88c2aec034a`.
The base commit remains `9f881af`; the learner correction is not yet committed.

## Scheduled-replay result and handoff

The completed Trial 33 rerun exercised eight violation-replay updates using the
configured period 4, 16 candidate pairs per side, and one selected pair per side.
It still failed: Long winner minus WAIT -0.531298, Short -2.531158. Both failure
rejections remained correct (WAIT minus Long failure 2.724310, Short 3.313187).
All margins were acquired before the mixed phase, but preservation again failed
starting at mixed update 6. No teacher-free Q drift occurred.

REVISE. The old-anchor correction passes the original cohort but is not a complete
preservation solution. Code-level coverage is 375 passing tests across the
affected suites (317 broader regressions plus 50 recovery and 8 audit tests).
Do not commit/push as a completed fix or start V40 under the user's conditional
approval. The next narrow diagnostic should record per-update exposure of the
previously acquired winner/failure anchors inside the learning portion, selected
violation pairs, and the loss/update that reverses each winner. Determine whether
rehearsal omits learned examples or present examples receive conflicting updates
before changing replay semantics or model capacity. Existing mixed tests remain
bounded source-pool tests, not whole-campaign economic confirmation.

## Exposure and candidate coverage follow-up

`v39-exposure-trial033` reproduces the same scheduled failure. Each acquired
winner appears in the learning portion only once over 32 mixed updates. Neither
is selected by the eight scheduled violation draws. Long first loses its margin
at update 6; Short subsequently loses it too. A source-row exposure count does
not assert identical full recurrent history when the row appears at a different
position in another sequence.

Candidate generation had a separate deterministic coverage defect: every call
used the most recent N winners. A production-replay regression with four winners
per side and two candidates per side reproduced permanent exclusion of the
older half. Candidates now use the existing checkpointed replay RNG, sampling
without replacement within a draw when enough examples exist. Requested bounded
counts, side balance, economic pairs, new-episode admission, eviction and exact
checkpoint resume are tested. No new replay state or schedule is introduced.

The audit now accepts an earlier report as explicit pair selectors, validates
checkpoint lineage and source economic labels, and reuses those exact witnesses.
This prevents sampler changes from silently changing the regression cohort.
Full CLI E2E reproduction verifies identical acquisition values on frozen pairs.
Latest replay/checkpoint/audit suite: 75 passed.

`v39-candidate-coverage-trial033` has exactly the same witnesses and acquisition
values as the failed exposure run. It still fails after mixed replay: Long minus
WAIT -0.416113; Short minus WAIT -2.635382. Both failures remain below WAIT.
Coverage correction is valid but not sufficient. V40 remains blocked. The next
matched diagnostic isolates all old-policy retention from ordinary/scheduled
replay interference; management retention is not disabled in production.

## Present-example erasure: reproduced optimizer boundary defect

`v39-retention-control-trial033`: disabling all retention still fails after
mixed replay (Long -1.170198, Short -2.125729 versus WAIT). This rejects removal
of management retention as a sufficient correction.

`v39-rehearsal-control-trial033` deliberately appends the same acquired pairs to
every mixed batch, preserving unique pair identities and unchanged production
losses. Long still drops from 4.94296 at update 4 to -0.08509 at update 5, despite
one explicit paired occurrence in the learning portion of that update. Short
finishes at -1.258788. Mere exposure is therefore not sufficient either.

The post-optimizer guard compared group-mean margins. A learned individual could
be erased while its group average remained acceptable. A production replay and
optimizer E2E regression reproduced this independently: a protected individual
margin fell below its 0.25 target to 0.05133. The correction retains vector
margins and checks every previously satisfied row. Unsatisfied rows are not hard
constraints; gradient objectives and projection groups remain unchanged.

The new regression passes on actual PyTorch and MLX learners. Existing tests for
progress with unsatisfied boundaries and protecting previously satisfied groups
also pass. Real frozen-pair rerun is under `v39-individual-guard-trial033`.
Do not infer its result until the report completes. A rehearsal control is not
the production cadence; ordinary scheduled replay must also be tested afterward.

Current candidate learner source blob after this correction:
`2a82628b39e37be8d27f8e6d3f9c0e85a9a462c8`.

The broad per-row candidate is rejected: `v39-individual-guard-trial033`
failed acquisition (neither winner beat WAIT after 128 updates), with 121
backtracked updates. Small regressions alone missed this real-data stall.
Adding the nearest satisfied individual to every projection group removed the
backtracking but still failed Short acquisition in
`v39-individual-projection-trial033` (Long 4.800980, Short -0.542916 after
acquisition). This broader variant is also not accepted.

The current narrowed correction applies individual protection to authenticated
economic pair groups, retaining aggregate protection for ordinary exact-action
groups. Mixed pair groups also include the nearest satisfied individual in the
direction projection. All previously satisfied paired rows are checked after
AdamW, not just the mean. This isolates the demonstrated pair-erasure defect
without changing the protection semantics of other exact-action rows.
`v39-pair-guard-trial033` is the matched real-data test. Focused regressions pass
on PyTorch and MLX, including teacher stripping, checkpoint reload, and identical
native greedy actions/Q values. Normal scheduled replay still needs separate
proof; the diagnostic-only repeated-witness control is not a new production
replay path.

## Automatic rehearsal: still under test

The narrowed pair guard passes the deliberately repeated-witness control:
all six boundaries survive all 32 mixed updates; final winner-minus-WAIT
is Long 4.929010 and Short 2.938190, with zero teacher-stripping drift.
Normal scheduled replay still fails (Long -0.242353, Short -2.625740), so
protecting only examples present in a batch is insufficient.

The production candidate adds opt-in `training.paired_a_plus_mastery_capacity_per_side`
(JSON default zero). Replay admits a source pair only after the actual learner
ranks its winner above WAIT and the opposite side, and WAIT above its failure,
by the configured margin. It stores bounded source references, reconstructs the
existing recurrent sequences, and reuses the same losses/optimizer. Already-present
pairs are not duplicated. Sources evicted from replay are removed from memory;
external pass libraries retain their existing ownership. The memory is training-only.

Production learner E2E first exposed duplicate exposure favoring the first learned
side. Removing that duplication allows both sides to be promoted and rehearsed
when ordinary batches omit them. Configuration parsing, default-off behavior,
full/sharded/lightweight replay restoration are exercised separately.

The first real-data automatic-memory test, `v40-automatic-mastery-trial033`,
uses capacity two per side. Acquisition remains identical, but preservation
still fails: final Long -0.662326, Short 0.480995 against a 0.40 margin. All
failure boundaries remain correct and teacher-stripping drift is zero.
Thirteen promotions exceed the four-pair bounded memory. A capacity-eight
matched test records exact resident sources to separate eviction from optimizer
interference. This is not yet an accepted fix or a sweep launch decision.

The capacity-eight matched result (`v40-automatic-mastery-trial033-cap8`)
passes every mixed update: final Long winner-minus-WAIT 4.115774, Short 1.244224;
WAIT-minus-failure 2.763122 Long and 3.190906 Short. Teacher-stripping Q drift is
zero. Both original witness sources remain resident while additional pairs are
automatically admitted. Final memory contains nine pairs (18 source sequences),
not copied episode arrays. There were 440 rehearsal-sequence exposures across
32 mixed updates; this additional learner work is intentional and must not be
described as free or as a performance optimization.

The second checkpoint (`v40-automatic-mastery-trial032`) also passes every one
of 32 mixed updates: Long 2.311461 and Short 1.323245 above WAIT; both exceed
its 0.25 target. Final failure margins are 1.837372 Long and 3.214915 Short,
with zero teacher-stripping drift. Its initial Long failure was wrong and was
corrected; failure rejection was not correct from initialization.

Full repository regression: 969 passed, 65 warnings. Native MLX production
rehearsal and enabled/default-off historical campaign resume tests pass. An
earlier concurrent-edit test run reported source-identity drift; holding source
files unchanged for the complete rerun passes. Never edit source during an
identity-sensitive regression run.

Remaining launch gate: native MLX frozen Trial 33, 128 acquisition plus 128
mixed updates, capacity eight per side, with resident-source and boundary
trajectories. This extended test checks later forgetting and memory turnover;
do not launch V40 merely because the shorter CPU tests passed.

## Extended production-learner gate: passed

Both backends passed all six boundaries at every one of 128 mixed updates
following 128 acquisition updates on the identical Trial 33 source witnesses.
The configured margin is 0.40.

| Backend | Long minus WAIT | Short minus WAIT | WAIT minus failed Long | WAIT minus failed Short | Teacher stripping Q drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native MLX | 0.958488 | 1.167270 | 2.217463 | 3.327773 | 0 |
| CPU/PyTorch | 2.371624 | 1.574463 | 1.835086 | 3.333944 | 0 |

Winner-minus-opposite margins also pass: MLX Long 3.578298, Short 2.773866;
CPU Long 4.363251, Short 2.645650. Failure rejection is correct throughout this
Trial 33 test. MLX admitted 15 additional pairs during mixed replay; CPU admitted
13. Each had 11 mixed updates with backtracking, rather than a stopped learner.
Floating-point trajectories and violation selection diverge; exact cross-backend
weight equality is not claimed. The behavioral acceptance boundaries pass on both.

Evidence: `v40-native-mlx-mastery-trial033` and
`v40-long-mastery-trial033-cpu` under the local diagnostics directory.
The audit no longer calls a backtrack count of 12 a rejected update: that inference
was unsupported. TDD reproduced the reporting error; all nine audit E2E tests pass
after removing the inferred field. Historical reports containing `rejected_updates`
must not use it as a measured rejection count.

V40 preserves V39's search space, objective, stages, 100 trials, three workers,
grouped multivariate TPE, MLX backend and prefetch zero. The new rehearsal capacity
is fixed at eight per side; the default remains zero for existing configurations.
No checkpoint/replay artifacts or private machine paths are included in Git.

These checks establish acquisition and preservation on the recorded regression
cohorts, not a guaranteed 60% pass rate, global perfect classification, or zero
future losses. V40 screening and subsequent frozen confirmation must establish
economic improvement. Previously inspected validation remains development evidence.

Commit preflight also reproduced a test-fixture defect: adding a second valid
V2 sweep config prevented test collection because the fixture required exactly
one JSON file. Fixtures now allow retained contracts to coexist, and a parametrized
test compiles each through real configuration validation, including effective
defaults and the explicit mastery setting. All 37 V2 sweep tests pass. No production
scheduler change or deletion of the preserved V39 configuration was required.

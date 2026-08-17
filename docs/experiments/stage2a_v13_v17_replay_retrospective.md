# Stage 2A v13-v18 selection and replay retrospective

## Status

This is the durable experiment record for the Stage 2A v13-v17 sequence and
the abandoned v6 constrained-TPE launch. The large run directories for v13-v16
and TPE v6 were deleted on 2026-08-17 after this evidence was recovered from
their tracked recipes and local launchd logs. They are not promotion candidates
and must not be resumed.

V17 was stopped at episode 40 after its diagnostics exposed a P1 mismatch
between the declared dominant-chop behavior and the learned loss/probe
boundary. V18 is the authorized matched correction. The frozen Stage 1 parent
remains on disk because v18 authenticates and warm-starts it.

## Product decision under test

Stage 2A must learn teacher-free Entry selection from the Expansion and
three-state Regime relationship:

- enter Long when Long Expansion is strong, Regime is transition/non-chop,
  and Short evidence does not dominate;
- enter Short under the mirrored conditions;
- wait on weak Expansion, conflicting sides, failed economic setups, or
  dominant persistent chop;
- retain both Long and Short participation while producing zero validation
  blows and reducing near-blow timeouts.

Expansion and Regime outputs are training-only supervision. They are not policy
observations during selection.

## Frozen lineage and evaluation boundary

All experiments in this sequence descend from the same authenticated Stage 1
parent:

- candidate: `1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0`
- evaluation: `c49852955655b705e376e057dfe2bf58784481175363b970bab063d8c42f981b`
- model SHA-256: `b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584`

The matched boundary remained 2021-2024 training, 2025 teacher-free selection,
and sealed 2026. Each campaign used 100 training episodes followed by a planned
200 teacher-free selection episodes. Training behavior is diagnostic; only the
teacher-free selection gate can promote a candidate.

## Experiment sequence

| Experiment | One intended change | Observed evidence | Decision |
| --- | --- | --- | --- |
| v13 exact-action margin | Add a DQfD-inspired `entry_action_margin=0.25`. | Manually stopped at episode 70: 6 passes, 2 blows, 62 timeouts; both blows were CL. The training mean balance had fallen to -$943.80 by episode 70. | STOP. A global action margin did not remove unsafe entries. |
| v14 chop/failed-confluence margins | Keep the exact-action behavior and add `chop_wait_margin=0.25` plus `failed_confluence_margin=0.25`. | Completed 100 episodes: 16% pass, 1% blow, 40.96% near-blow timeout rate, -0.0051R expectancy, 0.691R average winner, 80.86% 2R MFE capture. Final probe recalls were WAIT 56.25%, Long 25%, Short 50%; aggregate dominant-chop greedy-entry rate was 63.40%. The campaign stopped at the training short-circuit gate. | STOP. Winner retention remained reasonable, but safety and Long recall failed. |
| v15 fixed hard-WAIT replay | Reserve 2/16 sequences for hard-WAIT (`regime_wait_sequence_fraction=0.125`), reducing terminal replay to 6/16; keep margins unchanged. | Completed 100 episodes: 10% pass, 0% blow, 40% near-blow timeout rate, -0.0141R expectancy, 0.710R average winner, 80.87% 2R MFE capture. Final probe recalls were WAIT 93.75%, Long 21.88%, Short 9.38%. The 32-row final probe had zero dominant-chop greedy entries, but the aggregate diagnostic rate was still 82.67%. | STOP. Safety improved, but hard-WAIT exposure caused decisive opportunity/side collapse; the small final probe alone overstated chop rejection. |
| v16 balanced hard-WAIT replay | Reduce hard-WAIT to 1/16 and restore terminal replay from 6/16 to 7/16; keep safety and opportunity replay unchanged. | Manually stopped at episode 40: 3 passes, 0 blows, 37 timeouts. It had not produced terminal teacher-free evidence. | STOP/INCOMPLETE. Lowering the fixed quota was insufficient evidence and remained aggressive relative to rare-guidance research. |
| v17 sparse hard-WAIT replay | Preserve the ordinary 8-terminal/4-safety/4-opportunity batch on seven updates; on every eighth update use 7-terminal/4-safety/1-hard-WAIT/4-opportunity. Effective hard-WAIT dosage is 1/128 sequences. | Stopped at episode 40: 5 passes, 0 blows, 35 timeouts, 15 near-blow timeouts, -0.0093R expectancy, and 0.620R average winner. Of 6,527 trades, 2,946 (45.1%) occurred in dominant chop; Long and Short dominant-chop expectancy were both negative. The loss applied its chop margin only to rows already labeled WAIT, and the final gate checked only that same subset. | STOP/P1. The experiment did not train or test the declared all-dominant-chop WAIT behavior. |
| v18 all-dominant-chop learned margin | Keep v17 data, parent, replay, exact economic labels, margins, network, seed, and budgets; apply continuous dominance-weighted WAIT-margin pressure to every dominant-chop action example and test every dominant-chop row teacher-free. | Pending. | Run the matched 100-episode training and 200-episode teacher-free selection. |
| constrained-TPE v6 | Search large-win bonus, Regime loss, chop emphasis, and teacher dropout. | Only trial 0 started with values `0.1`, `0.3`, `2.0`, and `0.5`; it produced no terminal trial evidence before being abandoned. | INVALID/STOP. Do not infer parameter rankings from this study. |

The v14 and v15 rates above are training diagnostics, not causal OOS claims.
The v13 and v16 counts are partial-run observations and are not comparable to a
completed 100/200 campaign.

## What the sequence established

1. **The architecture is not the first failed boundary.** PropEvolve already
   stores complete recurrent sequences, uses burn-in and n-step Q-learning,
   preserves reset handling, and evaluates without teacher observations. The
   observed failure is more consistent with replay composition and competing
   learning pressures than with a need to replace recurrent RL.
2. **Margins and replay solve different problems.** DQfD-style margins can
   improve action separation on a sampled row, but they cannot correct an
   overrepresented cohort. v14 did not eliminate unsafe behavior; v15 eliminated
   observed training blows while collapsing Long/Short recall.
3. **Hard-WAIT replay has a real safety-opportunity tradeoff.** Moving from no
   fixed hard-WAIT cohort in v14 to 2/16 in v15 increased final WAIT recall from
   56.25% to 93.75% and removed training blows, but reduced Long recall from 25%
   to 21.88%, Short recall from 50% to 9.38%, and pass rate from 16% to 10%.
   This is why v17 changes dosage only.
4. **Training passes cannot select the model.** v14 produced several strong
   individual passes and 80.86% 2R MFE capture, yet still had a blow, negative
   expectancy, weak Long recall, and a failed gate. v15 had zero training blows
   but still failed through opportunity collapse.
5. **CL blows are useful training evidence, not permission for live CL.** Both
   partial v13 blows occurred on CL. CL remains training-only in this curriculum;
   the result still falsifies zero-blow learning for that recipe.
6. **The old TPE sweep answered nothing.** A started trial without a terminal
   teacher-free evaluation is not a ranking observation. Mechanism repair must
   precede another numerical sweep.

## External recurrent-RL evidence applied to the replay sequence

- PropEvolve's sequence, burn-in, n-step, reset, and teacher-free mechanics are
  directionally consistent with [DeepMind Acme R2D2](https://github.com/google-deepmind/acme/tree/master/acme/agents/jax/r2d2).
- [R2D3](https://openreview.net/pdf?id=SygKyeHKDH) evaluated rare demonstration
  mixtures from 1/32 through 1/256, with smaller ratios often performing better.
  Relative to that evidence, v16's fixed 1/16 hard-WAIT quota was aggressive;
  v17's deterministic 1/128 mixture is the smallest matched dosage correction.
- R2D2-style prioritized replay eventually warrants maximum-plus-mean TD-error
  priority with importance-sampling correction. PropEvolve currently
  oversamples named cohorts without an equivalent population correction. This
  is a later ablation, not part of v17.
- Recurrent-state parity remains a mandatory audit: collection, replay burn-in,
  checkpoint reload, and teacher-free evaluation should produce matching Q
  values for the same causal prefix. See the
  [TorchRL recurrent DQN tutorial](https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/dqn_with_rnn.py).
- DQfD-style action margins remain a valid ranking mechanism, but a larger
  margin cannot repair replay overrepresentation. See
  [Deep Q-learning from Demonstrations](https://research.google/pubs/deep-q-learning-from-demonstrations/).
- Paired failed-versus-valid recurrent replay is plausible but is not treated as
  an established standard. It remains a second ablation after v17 resolves the
  dosage question.
- Migrating to recurrent PPO is not justified by this evidence and would remove
  the replay and margin mechanisms currently under test.

## v18 falsifier and next decision

V18 is the only authorized active test. Keep margin values, losses, network,
data, seed, economics, episode budgets, parent, replay composition, and
teacher-free evaluator unchanged.

- **PROCEED** only if the complete teacher-free selection has zero blows, keeps
  both Long and Short active, meets the frozen economic gates, and improves or
  retains parent pass/near-blow behavior.
- **REVISE** if recurrent parity passes and safety improves without opportunity
  collapse, but the candidate narrowly misses a declared gate.
- **STOP** if sparse hard-WAIT still collapses side recall or pass participation,
  or if teacher-free validation blows.
- If v18 fails, audit recurrent-state parity and post-burn-in cohort mass first.
  Test importance-corrected prioritization or paired contrastive replay only
  when that audit identifies the corresponding failure. Do not increase margins
  or training episodes merely because training fit is incomplete.

## Evidence identities retained after run cleanup

| Evidence | SHA-256 |
| --- | --- |
| v13 recipe | `42f487aaa8f392afec1871e43e979350b27d2f734a4eaf3956bc86b62800bfae` |
| v13 stdout | `efdf4a43ed2a2d5e0355bb1a9986b4f9d3069eb7743e9044d73cd7f58690cd8c` |
| v14 recipe | `1cfc63c0add69b33bb5fcf12c81dbcb0cbd2853da1292ce67dcad0960eb483ce` |
| v14 stdout | `7ec655b2bdc3d171535d6b308d36706c56e898534b783e8b18406aec792cdcc2` |
| v15 recipe | `95c985dff7834e30de7441d5479f43c4e389e456c46d665fb96e41a8e5b18941` |
| v15 stdout | `edb92eebc2a5e4f3d8fc16d4cf0126433f17c094143034321f6d09a16868ceba` |
| v16 recipe | `200c1f3849e41c296ac70a884d3eeee12f8fadfa5f232ec58bc69ecd48d446d5` |
| v16 stdout | `a131b8bca63153b4696cf6b4a02b9e5192813ac1f7bf877a4a564c39f504aa9a` |
| TPE v6 recipe | `c6717af310a193d8338eaa812011d3511660602e7cb3423bcc842a6a7aee62b3` |
| TPE v6 stdout | `86f36bebf81b93f60b20a35c8f0a1b3c3426b6dc349ad2de69c12ba9ef6c5633` |
| TPE v6 stderr | `d5682db6d69e0b64d1843776d813f601275a650decf198f1408439dd047b2102` |
| v17 recipe at launch | `75f261799d7d10c4a438084e04a800e2962b3a2d25a0505803bbb34eb378000a` |
| v17 stdout | `4153f8865a8db052383481695b01567b95a7e86182adb6a9adf86112ac889ed4` |

The tracked recipes and this document are the durable experiment record. The
deleted model/replay/SQLite directories were failed or incomplete research
outputs and are not needed to reproduce the experiment definitions.

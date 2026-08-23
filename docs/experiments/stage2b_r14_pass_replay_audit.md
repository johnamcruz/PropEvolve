# Stage 2B R14 pass-replay audit

Date: 2026-08-23

## Decision

R14 was stopped after episode 105. Its frozen V21 parent, current checkpoint,
diagnostics, retained pass policy, and exact episode-70 trajectory are retained.
No new training run was launched.

The next replay input must be a bounded, deduplicated library of complete pass
trajectories. Policy snapshots under `retained-pass-policies/` are not replay
examples by themselves: they preserve network state, while the causal sequence
of observations, actions, rewards, recurrent resets, and economic labels must
be exported from replay state.

## Current-contract anchor

The only direct R14 pass is:

- replay identity: `historical-69-1787517489071640000`
- ticker and primary side: GC Short
- starting realized PnL: -$2,000
- terminal PnL: +$6,036.28
- trades: 307
- win rate: 50.49%
- average winner: +0.795R
- expectancy: +0.087R per trade
- 2R MFE capture ratio: 0.802
- source resume identity: `5172de30af8d18c23d44e5186087ba38c94f2f9a4af09a1b6c3dc57145e5ae00`

It is exported as
`runs/recovery-pass-replay/v22-r14-unified-balance-ep70-pass-v1.pt` with SHA-256
`53fecb977b6950c4714c1b45fdde463d28cd0c8302248c808a5f15cb65f65bca`.
The artifact contains one complete pass and is 95,820,013 bytes.

## Replay classification

### Keep

- `v21-healthy-passes-v1.pt`: 30 zero-start healthy passes. Keep as the V21
  nonnegative/A+ behavior anchor; it is not a substitute for full -$2,000
  recovery.
- `v22-r14-unified-balance-ep70-pass-v1.pt`: exact current-contract R14 pass.
- R14 `training-recovery.pt`, diagnostics, and retained pass policy: frozen
  source evidence and resumable state.

### Salvage into a bounded recovery-pass library

- `v22-r1-recent-recovery-passes-v1.pt`: four -$2,000-to-pass trajectories.
  The set is Short-heavy (SI and GC), so it must not be loaded alone.
- `v22-r14-recovery-passes-v1.pt`: twelve -$2,000-to-pass trajectories with
  both sides across ZN, RTY, CL, SI, ZB, ES, GC, and NQ. Despite its filename,
  its source identity is `039e91a13a3d9bf863e913c0e68db0caf5c7b15234480c636de1c1ee191dd9e8`,
  not the stopped R14 source. Treat it as historical off-policy evidence.

Recommended initial bounded set: current GC Short plus four historical Long
and three historical Short passes, preserving side and market diversity. Do
not outcome-balance the single current pass to half of every batch.

The audited eight-trajectory seed is:

| Source | Replay identity | Market | Side | Terminal PnL | Reward sum |
| --- | --- | --- | --- | ---: | ---: |
| current R14 | `historical-69-1787517489071640000` | GC | Short | $6,036.28 | 4.2197 |
| historical | `historical-17-1787375298578640000` | SI | Short | $7,333.80 | 5.0313 |
| historical | `historical-27-1787302769300575000` | ZN | Long | $6,011.21 | 3.7599 |
| historical | `historical-87-1787308287502256000` | CL | Long | $6,676.88 | 4.3018 |
| historical | `historical-111-1787311459553028000` | ES | Short | $6,019.02 | 4.0828 |
| historical | `historical-146-1787317250040856000` | SI | Long | $6,017.36 | 4.5016 |
| historical | `historical-169-1787320833658768000` | NQ | Short | $6,221.28 | 4.1335 |
| historical | `historical-195-1787325735632839000` | RTY | Long | $6,009.98 | 3.5024 |

All eight start at -$2,000, finish above the $6,000 target, contain finite
observations and rewards, preserve the recurrent reset cadence, and use only
valid actions. The seed is balanced four Long/four Short across seven markets.
They are bound in
`runs/recovery-pass-replay/v22-r14-unified-balance-best8-pass-replay-v1.pt`,
SHA-256
`b52afd254a30bf9d93a0807016ed6c36fa8676bc5774ef86dc69c8b3b641a22a`.
The artifact is 339,823,453 bytes. The next config adds one pass sequence every
eight learner updates, equivalent to four added sequences per 32-update
episode, and automatically promotes new passes while keeping the library at
eight examples.

### Rejected and deleted

- `v22-post-recovery-contrast-v1.pt`, SHA-256
  `98f133c7172b59008df2bc8c93e19b80edd420361d2c0426d3d2129754587a2b`:
  abandoned recovery-to-V21 handoff semantics.
- `v22-r2-exact-post-recovery-contrast-v2.pt`, SHA-256
  `e8d1a583a48f3d351a83f0e9e722b15adf12ea6e52132cd9311830b73da018fb`:
  abandoned handoff/relapse contrast semantics.
- R13 `training-recovery.pt`, SHA-256
  `c3605027a9e52cd14f078caaa44d83ec3ca55084cb36cdd7866001b39bb3fcdb`:
  eight-episode old recovery-mode checkpoint with zero passes and seven blows.

No live config, source file, test, or run state referenced these three files.
Deletion reclaimed 3,361,354,059 bytes (about 3.13 GiB). Compact diagnostics
remain for research comparison.

## Failure found in R14

Ordinary replay balances observed outcomes. Once episode 70 became the only
pass, it could occupy roughly half of a 16-sequence batch. Post-pass diagnostics
then became GC/Short-specific and dominant-chop entry behavior worsened. The
pass itself earned 84.6% of its profit outside dominant chop, so repeating the
entire trajectory at that mass teaches substantial incidental chop behavior as
well as the desired recovery path.

The next run should therefore use sparse, bounded pass rehearsal with side and
market diversity. It should not reintroduce the abandoned recovery-mode
handoff, alter V21 inference, or load all historical passes at ordinary
outcome-balanced mass.

# Stage 2A v21 paired recurrent population correction

## Decision

The r6 campaign was stopped after episode 85 because its training evidence
falsified the intended A+ selection behavior.  The recurrent pair construction
was valid, causal, side-specific, and economically labeled, but its balanced
one-winner/one-failure sampling omitted the population correction required by
stratified replay.

The only mechanism change for the next matched v21 run is to retain each
continuous-context winner/failure pair while weighting its two absolute action
anchors back to the current replay population.  The relative winner-over-
failure term remains unweighted.  Entry labels, margins, replay strata,
network, optimizer, risk, episode budget, and teacher-free evaluator remain
unchanged.

## Preserved r6 evidence

- Run: `stage2a-paired-recurrent-aplus-v21-200ep-r6-20260819`
- Code: `d56cf47`
- Analysis receipt:
  `pass-timeout-v2-through-episode-000085-0d49102a1dc8.json`
- Outcome: 10 passes, 73 timeouts, 2 blows, and 29 terminal near-blow
  timeouts through episode 85.
- Both blows were CL.  Their 36 Short trades lost `-28.141R`; the 13
  dominant-chop Short trades lost `-21.107R` at `-1.624R/trade`.
- Short recall rose from `4.23%` in episodes 1-20 to `65.69%` in episodes
  61-85, while Short precision remained near `20%`.
- Dominant-chop Entry rose from `2.92%` to `19.92%`; exact-WAIT recall fell
  from `98.36%` to `88.57%`.
- The sampled Short prediction mass was 2.27 times its exact target mass
  through episode 85 (`274.125 / 120.65625`).

The recovery checkpoint contained 46 retained episodes and the following
distinct economic anchor population:

| Side | Winners | Failures | Natural winner share |
| --- | ---: | ---: | ---: |
| Long | 176 | 726 | 19.51% |
| Short | 140 | 658 | 17.54% |

The paired sampler presented both sides as 50% winner and 50% failure.  It
therefore inflated absolute winner exposure by about 2.6 times for Long and
2.9 times for Short.  The observed failure is an action-prior distortion, not
a missing-pair, recurrent-boundary, or label-timing failure.

## Corrected objective

For one side with `W` winner anchors and `F` failure anchors in the retained
replay population:

```text
winner_weight = 2W / (W + F)
failure_weight = 2F / (W + F)

relative = softplus(pair_margin + bad_advantage - good_advantage)
winner   = winner_weight * softplus(action_margin - good_advantage)
failure  = failure_weight * softplus(action_margin + bad_advantage)

pair_loss = mean(relative, winner, failure)
```

The weights sum to two, so the three-term loss scale is preserved.  Balanced
pair sampling still protects Long and Short learning coverage, while the
absolute ENTER-versus-WAIT pressure again reflects the economic population.
For the preserved r6 pool, the implied weights are approximately Long
`0.390/1.610` and Short `0.351/1.649` for winner/failure respectively.

## TDD boundary

Public tests cover fee-inclusive +2R-before−1R winners and non-winners for
both sides, stop-first failure, pair atomicity after recurrent burn-in,
balanced and one-sided replay populations, deterministic checkpoint-resumed
sampling, both-side winner/failure gradients, invalid population evidence,
and diagnostic publication of the applied weights.  Teacher-free inference
still performs no teacher or label lookup.

## Matched falsifier

The next clean v21 run must improve Short precision and prevent the rising
dominant-chop Entry trajectory without collapsing Short recall or the valid
non-chop opportunities seen in v19/v20.  Training evidence alone cannot
promote the policy: the fixed teacher-free evaluator still requires zero
blows and the frozen economic gates.

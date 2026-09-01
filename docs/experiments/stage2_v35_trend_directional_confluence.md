# Stage 2 V35 Trend Directional Confluence

## Frozen question

Does training-only Trend direction improve the teacher-free policy's economic
ranking of authenticated Expansion opportunities without weakening the V34
Regime chop filter or changing the action, execution, replay, and inference
contracts?

Expansion remains the opportunity and entry source. Regime remains
non-directional chop context. Trend may only add directional learning pressure
to an existing authenticated economic pair; it may not create, mask, execute,
or relabel an entry.

## Causal Trend evidence

The packaged Trend teacher predicts next-bar directional Trend starts through
Long/Short launch probability and conditional quality. To represent a Trend
that has started and remains relevant, V35 carries the recent directional score
forward with a bounded, causal rolling maximum. No future row is visible.

Directional score:

`launch_probability * conditional_quality`

The declared confirmation window is 50 completed three-minute bars. Exact
Long/Short ties add no Trend pressure.

## Economic authority

- Aligned authenticated Long winner: reinforce Long above WAIT and Short.
- Aligned authenticated Short winner: reinforce Short above WAIT and Long.
- Countertrend failed Long: reinforce WAIT above Long.
- Countertrend failed Short: reinforce WAIT above Short.
- Profitable countertrend winner: preserve the winner label; Trend adds no
  contradictory pressure.
- Aligned failure, ambiguous Trend, or a row without an authenticated economic
  pair: Trend adds no pressure.

The existing +2R-before-1R label remains authoritative.

## Matched V35 test

The Optuna search retains the verified V34 search family and adds one categorical
switch: `trend_start_confluence.enabled = false | true`. Both arms load the same
11 training-label channels, while generic Trend imitation remains zero. This
makes the disabled arm a matched control and keeps teacher-free selection
identical.

## Required evidence

Training telemetry records Trend opportunity/safety loss, active rows,
directional dominance mass, aligned Long/Short winner rows, and countertrend
Long/Short failure rows.

The frozen final teacher-free probe records:

- aligned Long winner `Q(Long) - Q(WAIT)`;
- aligned Short winner `Q(Short) - Q(WAIT)`;
- countertrend failed Long `Q(WAIT) - Q(Long)`;
- countertrend failed Short `Q(WAIT) - Q(Short)`;
- row counts and continuous Trend-dominance mass for every cohort.

V35 is useful only if the enabled arm improves these rankings and downstream
pass economics without increasing blows, near-blows, or dominant-chop entries.
The target remains at least 60% teacher-free pass rate, zero blows, no more than
25% near-blow timeouts, preserved 2R capture, and active Long and Short sides.

## Falsifier

Reject Trend confluence if the enabled arm does not beat its disabled control on
teacher-free economic rankings and pass economics, or if it weakens safety. Do
not add Trend observations, hard entry gates, a new inference model, or broader
ENTER activation in response.

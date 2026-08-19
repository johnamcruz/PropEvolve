# Stage 2A v20 generic canonical A+ correction

## Status

`REJECTED / NON-EXECUTABLE — production code and recipe reverted to e7/v19`

The first v20 run was stopped after episode 24. It showed lower dominant-chop
entry but weaker economics and about 77% less paired A+ mass than v19 because
pair eligibility required the same ticker and original direction. Its evidence
remains immutable under its original run identity and is not a restart parent.

The generic-canonical follow-up remained incomplete. It produced five passes
and zero blows through episode 26, but did not establish incremental economic
lift over the accepted e7/v19 baseline. The executable v20 recipe and its
production path were removed during the e7 rollback; this document is retained
only as historical research evidence.

## Decision

Keep the v20 recipe family and change only its paired A+ comparison contract.
The recurrent policy must learn one generic candidate-side Expansion signature
across markets and original directions while remaining teacher-free at
validation and inference.

For every training-only exact Entry candidate, canonicalize:

```text
candidate Expansion = proposed-side attempt and clean-retention probabilities
opposite Expansion  = opposite-side attempt and clean-retention probabilities
candidate advantage = Q(proposed side) - Q(WAIT)
```

A Long and a Short example with the same candidate/opposite geometry therefore
share one ranking rule. They do not share an action value: the raw causal
embedding and separate `Q(LONG)` and `Q(SHORT)` outputs preserve asymmetric
Long and Short behavior. Ticker identity does not gate or weight pairs. Exact
`+2R-before-1R` winners are positives; context-similar failed confluence rows
are negatives. Expansion geometry, the complete three-state Regime vector, and
account-headroom similarity weight comparisons continuously without cutoffs.
Headroom is a matched control, not an A+ ingredient.

```text
loss = softplus(
    pair_margin
    + failed_candidate_Q_minus_WAIT
    - winning_candidate_Q_minus_WAIT
)
```

## Frozen boundary

The immutable Stage 1 parent, authenticated pre-2025 teachers and exact Entry
labels, recurrent architecture, replay, burn-in, optimizer, seed, margins,
loss weights, risk, costs, execution, 200 training episodes, and 200
teacher-free validation episodes remain unchanged. No teacher or ticker
identity becomes a policy observation. Stage 2B recovery remains deferred.

## Falsifier

Reject v20 if teacher-free validation produces a blow, universal WAIT, side
collapse, no monotonic winner-versus-failure Q separation, or no improvement in
pass conversion and terminal near-blow timeouts over the preserved v19/v20-r1
evidence. Mechanism diagnostics must show nonzero generic pair mass and
separation for both original Long and Short examples; training-loss improvement
alone is not acceptance.

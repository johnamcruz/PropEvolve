"""The compact search policy: find a trend-following policy that passes without blowing.

This is the SEARCH half of the plan. The reasoning model cannot run thousands of 14,400
bar episodes at ~0.3s a decision, so a small network over the 54-field evidence contract
does the searching and the reasoning model is taught the result. It trades in the real
HistoricalChallengeEnv -- same $6,000 target, same $3,000 floor, same fees, same ratchet
and the same risk controls -- so what it learns is transferable rather than an artefact
of a parallel simulator.

Two properties are pinned here because getting either wrong fails silently. Illegal
actions must be masked, not merely discouraged, or the policy wastes probability mass on
moves the environment refuses and its advantage estimates are computed against decisions
it never made. And the objective must treat a blow as categorically worse than any
number of timeouts, matching the lexicographic gate the acceptance criteria already use
(maximum_blow_rate 0.0); a blend lets a policy rationally buy pass rate with blow-ups.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from propevolve.reasoning_policy.compact_rl import (
    ActorCritic, challenge_objective, masked_categorical)
from propevolve.reasoning_policy.evidence import EVIDENCE_FIELDS


def test_the_network_reads_the_evidence_contract_width():
    net = ActorCritic(len(EVIDENCE_FIELDS), n_actions=5)
    obs = torch.zeros(3, len(EVIDENCE_FIELDS))
    logits, value = net(obs)
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_illegal_actions_get_zero_probability():
    """Masking, not discouraging: the environment refuses these outright."""
    logits = torch.tensor([[1.0, 5.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[True, False, True, False, False]])
    dist = masked_categorical(logits, mask)
    probs = dist.probs[0]
    assert probs[1].item() == pytest.approx(0.0, abs=1e-8)
    assert probs[3].item() == pytest.approx(0.0, abs=1e-8)
    assert probs[0].item() + probs[2].item() == pytest.approx(1.0, abs=1e-6)


def test_masking_still_ranks_the_legal_actions():
    logits = torch.tensor([[3.0, 9.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, True, True, False]])
    probs = masked_categorical(logits, mask).probs[0]
    assert probs[0] > probs[2] > probs[3]


def test_a_fully_masked_row_is_refused():
    """No legal action means the rollout is broken, not that any move is fine."""
    logits = torch.zeros(1, 5)
    with pytest.raises(ValueError):
        masked_categorical(logits, torch.zeros(1, 5, dtype=torch.bool))


# ───────────────────────── the objective
def test_any_blow_ranks_below_every_zero_blow_policy():
    """Lexicographic, matching maximum_blow_rate 0.0 in the acceptance criteria."""
    blowing = challenge_objective({"pass": 0.90, "blow": 0.01, "timeout": 0.09})
    worst_safe = challenge_objective({"pass": 0.0, "blow": 0.0, "timeout": 1.0})
    assert blowing < worst_safe


def test_among_zero_blow_policies_more_passes_wins():
    better = challenge_objective({"pass": 0.55, "blow": 0.0, "timeout": 0.45})
    worse = challenge_objective({"pass": 0.34, "blow": 0.0, "timeout": 0.66})
    assert better > worse


def test_within_the_blow_region_fewer_blows_is_still_better():
    """Keep a gradient so the optimiser can climb OUT of the blow region."""
    fewer = challenge_objective({"pass": 0.30, "blow": 0.05, "timeout": 0.65})
    more = challenge_objective({"pass": 0.30, "blow": 0.40, "timeout": 0.30})
    assert fewer > more


def test_the_rule_baseline_scores_below_a_safe_but_dull_policy():
    """The frozen rule: 41.7% pass but 46.2% blow. It must not outrank a zero-blow policy."""
    rule = challenge_objective({"pass": 0.417, "blow": 0.462, "timeout": 0.122})
    dull = challenge_objective({"pass": 0.238, "blow": 0.0, "timeout": 0.762})
    assert dull > rule

"""Simulator economic receipts -> configured screening, never invented outcomes."""
import pytest
from propevolve.decision import Action
from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.evaluation import evaluate_policy
from propevolve.reasoning_policy.selection import assess_candidate
from test_reasoning_challenger_e2e import environment
from test_reasoning_collection_evaluation_e2e import sources


@pytest.mark.parametrize("entry,outcome", [(Action.ENTER_LONG_1, "pass"),
    (Action.ENTER_SHORT_1, "blow"), (Action.WAIT, "timeout")])
def test_screening_uses_simulator_results_and_reports_specialist_dependence(entry, outcome):
    class Policy:
        def decide(self, context, legal_actions):
            return (Action.HOLD if Action.HOLD in legal_actions else entry), {}
    events = []
    report = evaluate_policy(Policy(), environment(), episodes=[{"ticker": "NQ", "start": 0}],
        context_config=ContextConfig(2, ("account.realized_pnl_norm",)), sources=sources(),
        max_steps=8, near_blow_headroom_fraction=0.1, on_decision=events.append)
    assert report["episodes"][0]["outcome"] == outcome
    assert events[0]["requested_action"] == entry.name
    assert report["action_counts"][entry.name] >= 1
    criteria = {"minimum_episodes": 1, "minimum_pass_rate": 0.6,
        "maximum_blow_rate": 0, "maximum_near_blow_rate": 0.1,
        "near_blow_headroom_fraction": 0.1, "require_teacher_free": True,
        "minimum_long_entries": 0, "minimum_short_entries": 0}
    decision = assess_candidate(report, criteria)
    assert "teacher_dependence" in decision["failures"]
    assert decision["promoted"] is False
    if outcome == "blow":
        assert "blow_rate" in decision["failures"]
    if outcome == "timeout":
        assert "pass_rate" in decision["failures"]

"""Simulator economic receipts -> configured screening, never invented outcomes."""
import pytest
from propevolve.decision import Action
from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.evaluation import evaluate_policy
from propevolve.reasoning_policy.selection import assess_candidate, assess_trade_mastery
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


def test_trade_mastery_gate_requires_three_r_winners_without_using_pass_rate():
    actions = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")
    report = {
        "teacher_free": True,
        "per_action": {name: {"count": 20, "accuracy": 0.8,
                               "mean_target_advantage": 0.25}
                       for name in actions},
        "execution": {"win_rate": 0.45, "average_win_r": 2.9,
                      "expectancy_r": 0.2, "two_r_mfe_capture_ratio": 0.6},
    }
    criteria = {"required_actions": list(actions), "minimum_examples_per_action": 20,
                "minimum_accuracy_per_action": 0.5,
                "minimum_mean_action_advantage": 0.0,
                "minimum_win_rate": 0.4, "minimum_average_win_r": 3.0,
                "minimum_expectancy_r": 0.0,
                "minimum_two_r_mfe_capture_ratio": 0.5,
                "require_teacher_free": True}

    rejected = assess_trade_mastery(report, criteria)
    assert rejected["failures"] == ["average_win_r"]
    report["execution"]["average_win_r"] = 3.1
    assert assess_trade_mastery(report, criteria)["verdict"] == "REVIEW_CANDIDATE"
    assert "pass_rate" not in report

"""Fixed evidence -> real rollout runner -> before/after and coverage receipts."""
import json
import numpy as np

from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.learning_audit import (
    score_labeled_examples,
    summarize_trade_mastery,
)
from propevolve.reasoning_policy.rl import train_rl
from test_reasoning_challenger_e2e import environment
from test_reasoning_collection_evaluation_e2e import sources
from test_reasoning_rl_e2e import ScriptedRuntime


def test_frozen_audit_respects_actual_legal_actions_instead_of_inventing_alternatives():
    record = {"source_id": "masked", "completed_at_ns": 1,
        "messages": [{"role": "user", "content": json.dumps({"legal_actions": ["WAIT"]})},
                     {"role": "assistant", "content": "WAIT"}]}
    report = score_labeled_examples(ScriptedRuntime("WAIT"), [record])[0]
    assert report["scores"] == {"WAIT": 0.0}
    assert report["target_advantage"] is None


def test_trade_mastery_report_is_separate_from_challenge_economics():
    scored = [
        {"target": name, "correct": correct, "target_advantage": advantage}
        for name, correct, advantage in (
            ("WAIT", True, 0.4), ("ENTER_LONG_1", True, 0.3),
            ("ENTER_SHORT_1", False, -0.2), ("HOLD", True, 0.1),
            ("CLOSE", True, 0.2),
        )
    ]
    report = summarize_trade_mastery(scored)

    assert set(report["per_action"]) == {
        "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE"}
    assert report["macro_accuracy"] == 0.8
    assert report["worst_action_advantage"] == -0.2
    assert not ({"pass_rate", "blow_rate", "near_blow_rate"} & report.keys())


def test_real_rollout_reports_frozen_ranking_before_and_after_update():
    runtime = ScriptedRuntime("ENTER_LONG_1")
    record = {"source_id": "fixed", "completed_at_ns": 1,
        "messages": [{"role": "user", "content": json.dumps({"legal_actions":
            ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]})},
            {"role": "assistant", "content": "ENTER_LONG_1"}]}
    class ExternalUpdater:
        def update(self, rows, rng):
            # Deliberate regression in the external model tests diagnostic sensitivity.
            runtime.entry = "WAIT"
            return {"sampled_update_rows": 1,
                    "sampled_action_mass": {"ENTER_LONG_1": 1}}
    reports = train_rl(runtime, environment(), learner=ExternalUpdater(),
        episodes=[{"ticker": "NQ", "start": 0}],
        context_config=ContextConfig(2, ("account.realized_pnl_norm",)), sources=sources(),
        config={"seed": 7, "groups": 1, "group_size": 2, "max_steps": 8,
                "advantage_scale": 1, "checkpoint_every_groups": 1},
        diagnostic_records=[record])
    report = reports[0]
    assert report["audit_before"][0]["correct"] is True
    assert report["audit_after"][0]["correct"] is False
    assert report["available_action_mass"]["ENTER_LONG_1"] == 2
    assert report["available_action_mass"]["HOLD"] > 0
    assert report["update_coverage"]["HOLD"] == 0
    assert report["ranking_regressions"] == 1

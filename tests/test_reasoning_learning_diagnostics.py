"""Fixed evidence -> real rollout runner -> before/after and coverage receipts."""
import json
import numpy as np
import pytest

from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.learning_audit import (
    main as learning_audit_main,
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


def test_learning_audit_rejects_corrupt_legal_actions_and_nonfinite_scores():
    malformed = {"source_id": "bad", "completed_at_ns": 1,
        "messages": [{"role": "user", "content": json.dumps({"legal_actions":
            ["WAIT", "WAIT"]})}, {"role": "assistant", "content": "WAIT"}]}
    with pytest.raises(ValueError, match="conflicts with legal actions"):
        score_labeled_examples(ScriptedRuntime("WAIT"), [malformed])

    class NonFinitePolicy:
        def completion_scores(self, messages, choices, **kwargs):
            return {choice: (float("nan") if choice == "WAIT" else 0.0)
                    for choice in choices}
    valid = {"source_id": "bad-score", "completed_at_ns": 1,
        "messages": [{"role": "user", "content": "{}"},
                     {"role": "assistant", "content": "WAIT"}]}
    with pytest.raises(ValueError, match="invalid frozen audit scores"):
        score_labeled_examples(NonFinitePolicy(), [valid])
    with pytest.raises(ValueError, match="needs labeled examples"):
        score_labeled_examples(ScriptedRuntime("WAIT"), [])


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.__setitem__(0, {**rows[0], "target_advantage": None}),
    lambda rows: rows.__setitem__(0, {**rows[0], "correct": 1}),
])
def test_trade_mastery_summary_requires_all_five_competing_action_boundaries(mutation):
    rows = [{"target": name, "correct": True, "target_advantage": 0.1}
            for name in ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")]
    mutation(rows)
    with pytest.raises(ValueError, match="trade-mastery audit"):
        summarize_trade_mastery(rows)


def test_learning_audit_cli_scores_an_authenticated_record_without_training(
        tmp_path, monkeypatch, capsys):
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "source_id": "row-1", "completed_at_ns": 1,
        "messages": [
            {"role": "user", "content": json.dumps({"legal_actions": ["WAIT"]})},
            {"role": "assistant", "content": "WAIT"},
        ],
    }) + "\n")
    monkeypatch.setattr(
        "propevolve.reasoning_policy.learning_audit.MLXActionPolicy.from_config",
        lambda path: ScriptedRuntime("WAIT"),
    )

    learning_audit_main([
        "--config", "model.json", "--records", str(records), "--limit", "1",
    ])

    report = json.loads(capsys.readouterr().out)
    assert report[0]["source_id"] == "row-1"
    assert report[0]["correct"] is True
def test_frozen_action_assessment_strips_teacher_training_queries_only():
    from propevolve.reasoning_policy.learning_audit import (
        TeacherFreeAssessmentRows, teacher_free_assessment_config)

    original = [{"tokens": [1, 2], "error_selected_distillation": {
        "tokens": [3, 4], "market_targets": {"probabilities": [.8]}}}]
    frozen = TeacherFreeAssessmentRows(original)

    assert frozen[0] == {"tokens": [1, 2]}
    assert "error_selected_distillation" in original[0]
    configured = teacher_free_assessment_config({
        "action_supervision": {"enabled": True},
        "error_selected_distillation": {"loss_weight": .25},
        "market_distillation": None,
    })
    assert configured["action_supervision"] == {"enabled": True}
    assert configured["error_selected_distillation"] is None

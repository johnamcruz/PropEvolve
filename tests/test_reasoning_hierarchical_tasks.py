"""Public trade-label -> hierarchical decision-task contract."""

from propevolve.decision import Action
from propevolve.reasoning_policy.dataset import supervised_record
from propevolve.reasoning_policy.decision_tasks import hierarchical_task_records
from propevolve.reasoning_policy.decision_schema import legal_completion_names
from propevolve.reasoning_policy.supervision import hierarchical_action_objective
from propevolve.reasoning_policy.labels import ActionLabels, ActionOutcome
from test_reasoning_challenger_e2e import environment
from propevolve.reasoning_policy.context import ContextConfig, RollingContext


def _outcome(value, *, end=10):
    return ActionOutcome("fixture", value * 300.0, value, 3000.0, 3, end)


def _record(values):
    history = RollingContext(ContextConfig(2, ("trade.open",)))
    history.append(1, {"trade.open": 0.0})
    labels = ActionLabels(
        observation=environment().reset(options={"ticker": "NQ", "start": 0})[0],
        outcomes={action: _outcome(value) for action, value in values.items()},
    )
    return supervised_record(
        history.snapshot(), labels, source_id="fixture", continuation_id="fixture",
        target_temperature=1.0,
    )


def test_long_winner_becomes_enter_then_long_without_hold_close_competition():
    record = _record({
        Action.WAIT: _outcome(0.0).reward_to_go,
        Action.ENTER_LONG_1: _outcome(2.0).reward_to_go,
        Action.ENTER_SHORT_1: _outcome(-1.0).reward_to_go,
    })

    tasks = hierarchical_task_records(record)

    assert [(row["decision_task"], row["target_name"], row["names"])
            for row in tasks] == [
        ("entry", "ENTER", ["WAIT", "ENTER"]),
        ("direction", "LONG", ["LONG", "SHORT"]),
    ]
    assert tasks[0]["values"] == [0.0, 2.0]
    assert tasks[1]["values"] == [2.0, -1.0]
    assert all("HOLD" not in row["names"] and "CLOSE" not in row["names"]
               for row in tasks)


def test_short_winner_becomes_enter_then_short():
    record = _record({
        Action.WAIT: 0.0,
        Action.ENTER_LONG_1: -1.0,
        Action.ENTER_SHORT_1: 3.0,
    })

    tasks = hierarchical_task_records(record)

    assert [(row["decision_task"], row["target_name"])
            for row in tasks] == [("entry", "ENTER"), ("direction", "SHORT")]
    assert tasks[0]["values"] == [0.0, 3.0]
    assert tasks[1]["values"] == [-1.0, 3.0]


def test_wait_or_directional_conflict_never_creates_a_direction_task():
    failed = _record({
        Action.WAIT: 0.0,
        Action.ENTER_LONG_1: -1.0,
        Action.ENTER_SHORT_1: -1.0,
    })
    conflict = _record({
        Action.WAIT: 2.25,
        Action.ENTER_LONG_1: 2.0,
        Action.ENTER_SHORT_1: 2.0,
    })

    for record in (failed, conflict):
        tasks = hierarchical_task_records(record)
        assert [(row["decision_task"], row["target_name"])
                for row in tasks] == [("entry", "WAIT")]


def test_positioned_state_produces_only_hold_close_task():
    record = _record({Action.HOLD: 1.2, Action.CLOSE: 0.4})

    tasks = hierarchical_task_records(record)

    assert [(row["decision_task"], row["target_name"], row["names"])
            for row in tasks] == [
        ("management", "HOLD", ["HOLD", "CLOSE"]),
    ]
    assert tasks[0]["values"] == [1.2, 0.4]


def _objective(scores, probabilities, values, task_code):
    return float(hierarchical_action_objective(
        scores, probabilities, values,
        {"soft_target_weight": 0.5, "ranking_weight": 2.0, "margin": 0.1},
        task_code=task_code, xp=__import__("numpy"),
    ))


def test_flat_winner_loss_is_entry_gate_plus_conditional_direction():
    # WAIT, LONG, SHORT: economically authenticated Long winner.
    initial = _objective([0.0, 0.0, 0.0], [.1, .8, .1], [0.0, 2.0, -1.0], 0)
    correct = _objective([0.0, 1.0, -1.0], [.1, .8, .1], [0.0, 2.0, -1.0], 0)
    wrong_side = _objective([0.0, -1.0, 1.0], [.1, .8, .1], [0.0, 2.0, -1.0], 0)

    assert correct < initial < wrong_side


def test_short_winner_loss_learns_enter_then_short():
    initial = _objective([0.0, 0.0, 0.0], [.1, .1, .8], [0.0, -1.0, 2.0], 0)
    correct = _objective([0.0, -1.0, 1.0], [.1, .1, .8], [0.0, -1.0, 2.0], 0)
    wrong_side = _objective([0.0, 1.0, -1.0], [.1, .1, .8], [0.0, -1.0, 2.0], 0)

    assert correct < initial < wrong_side


def test_wait_loss_does_not_train_an_arbitrary_direction():
    # WAIT is economically correct; swapping equally failed side scores cannot
    # change the objective because the direction task is ineligible.
    long_high = _objective([1.0, 0.5, -0.5], [.8, .1, .1], [0.0, -1.0, -1.0], 0)
    short_high = _objective([1.0, -0.5, 0.5], [.8, .1, .1], [0.0, -1.0, -1.0], 0)
    assert long_high == short_high


def test_wait_must_beat_both_long_and_short():
    tied = _objective([0.0, 0.0, 0.0], [.8, .1, .1], [0.0, -1.0, -1.0], 0)
    safe = _objective([1.0, -1.0, -2.0], [.8, .1, .1], [0.0, -1.0, -1.0], 0)
    unsafe_long = _objective([0.0, 1.0, -2.0], [.8, .1, .1], [0.0, -1.0, -1.0], 0)
    unsafe_short = _objective([0.0, -2.0, 1.0], [.8, .1, .1], [0.0, -1.0, -1.0], 0)

    assert safe < tied < unsafe_long
    assert unsafe_long == unsafe_short


def test_long_winner_entry_gate_cannot_be_satisfied_by_wrong_short_score():
    from propevolve.reasoning_policy.supervised_trainer import hierarchical_boundary_metrics

    row = {"target_name": "ENTER_LONG_1", "action_targets": {
        "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
        "values": [0.0, 2.0, -1.0]}}
    metrics = hierarchical_boundary_metrics([row], [[0.0, -1.0, 2.0]])

    assert metrics["per_task"]["entry.ENTER"]["mean_target_advantage"] == -1.0
    assert metrics["per_task"]["direction.LONG"]["mean_target_advantage"] == -3.0


def test_positioned_loss_is_only_hold_close_binary():
    tied = _objective([0.0, 0.0], [.8, .2], [1.2, 0.4], 1)
    correct = _objective([1.0, -1.0], [.8, .2], [1.2, 0.4], 1)
    assert correct < tied


def test_hierarchical_metrics_report_each_binary_boundary_and_reconstructed_actions():
    from propevolve.reasoning_policy.supervised_trainer import hierarchical_boundary_metrics

    rows = [
        {"target_name": "ENTER_LONG_1", "action_targets": {
            "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "values": [0.0, 2.0, -1.0]}},
        {"target_name": "WAIT", "action_targets": {
            "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "values": [0.0, -1.0, -1.0]}},
        {"target_name": "ENTER_SHORT_1", "action_targets": {
            "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "values": [0.0, -1.0, 2.0]}},
        {"target_name": "HOLD", "action_targets": {
            "names": ["HOLD", "CLOSE"], "values": [1.0, 0.0]}},
        {"target_name": "CLOSE", "action_targets": {
            "names": ["HOLD", "CLOSE"], "values": [-1.0, 0.0]}},
    ]
    metrics = hierarchical_boundary_metrics(
        rows, [[0, 2, -1], [2, 0, 0], [0, -1, 2], [2, 0], [0, 2]], margin=.1)

    assert set(metrics["per_task"]) == {
        "entry.ENTER", "entry.WAIT", "direction.LONG", "direction.SHORT",
        "management.HOLD", "management.CLOSE",
    }
    assert metrics["worst_task_advantage"] == 2.0
    assert metrics["task_macro_accuracy"] == 1.0


def test_action_completion_schema_rejects_nontrading_answers():
    import pytest

    with pytest.raises(ValueError, match="not an action-supervision answer"):
        legal_completion_names("PASS")

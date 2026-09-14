"""Public score reports distinguish acquisition, retention and ambiguity."""
import pytest

from propevolve.reasoning_policy.decisive_learning import decision_evidence, compare_learning


def test_enter_correct_wrong_direction_is_not_an_entry_failure():
    row = {"target_name": "ENTER_LONG_1", "action_targets": {
        "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"], "values": [0., 2., -1.]}}
    evidence = decision_evidence(row, [-1., -2., 0.], ambiguity_r=.01)
    assert evidence["ENTER"]["correct"] is True
    assert evidence["LONG"]["correct"] is False
    assert evidence["LONG"]["margin"] == -2.


def test_tied_management_economics_are_ambiguous_not_a_clear_failure():
    row = {"target_name": "HOLD", "action_targets": {
        "names": ["HOLD", "CLOSE"], "values": [1., 1.]}}
    evidence = decision_evidence(row, [-1., 0.], ambiguity_r=.01)
    assert evidence["HOLD"]["ambiguous"] is True
    result = compare_learning([evidence], [evidence])
    assert result["applicable_clear_boundaries"] == 0
    assert result["all_clear_correct"] is False  # no vacuous PASS


def test_newly_acquired_boundary_is_protected_on_the_next_comparison():
    old = [{"LONG": {"margin": -.2, "correct": False, "ambiguous": False}}]
    learned = [{"LONG": {"margin": .1, "correct": True, "ambiguous": False}}]
    assert compare_learning(old, learned)["acquired"] == 1
    lost = compare_learning(learned, old)
    assert lost["forgotten"] == 1
    assert lost["all_clear_correct"] is False


def test_nonfinite_scores_are_rejected():
    row = {"target_name": "WAIT", "action_targets": {
        "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"], "values": [0., -1., -1.]}}
    with pytest.raises(ValueError):
        decision_evidence(row, [0., float("nan"), -1.], ambiguity_r=.01)


@pytest.mark.parametrize("target,values,scores,expected", [
    ("ENTER_LONG_1", [0.,2.,-1.], [0.,1.,-1.], {"ENTER": True, "LONG": True}),
    ("ENTER_LONG_1", [0.,2.,-1.], [2.,1.,0.], {"ENTER": False, "LONG": True}),
    ("ENTER_LONG_1", [0.,2.,-1.], [2.,0.,1.], {"ENTER": False, "LONG": False}),
    ("ENTER_SHORT_1", [0.,-1.,2.], [0.,-1.,1.], {"ENTER": True, "SHORT": True}),
    ("WAIT", [0.,-1.,-1.], [1.,0.,0.], {"WAIT": True}),
    ("HOLD", [2.,1.], [1.,0.], {"HOLD": True}),
    ("CLOSE", [0.,1.], [0.,1.], {"CLOSE": True}),
])
def test_each_legal_decision_is_audited_without_inapplicable_tasks(target, values, scores, expected):
    names = ["HOLD", "CLOSE"] if len(values) == 2 else ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    row = {"target_name": target, "action_targets": {"names": names, "values": values}}
    evidence = decision_evidence(row, scores, ambiguity_r=.01)
    assert {name: value["correct"] for name, value in evidence.items()} == expected


def test_real_simulator_prefix_never_reports_an_incomplete_window_as_a_pass():
    from test_reasoning_challenger_e2e import environment
    from propevolve.reasoning_policy.context import ContextConfig
    from propevolve.reasoning_policy.decisive_learning import simulate_prefix
    from propevolve.decision import Action

    class WaitingPolicy:
        def decide(self, context, legal_actions):
            return Action.WAIT, {action.name: float(action == Action.WAIT) for action in legal_actions}

    trace = []
    result = simulate_prefix(WaitingPolicy(), environment(),
        options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        max_steps=2, on_decision=trace.append)
    assert result["complete"] is False
    assert result["outcome"] is None
    assert result["steps"] == 2
    assert result["action_counts"] == {"WAIT": 2}
    assert result["closed_trades"] == 0
    assert len(trace) == 2


def test_basic_generalization_pool_is_chronological_balanced_and_deduplicated():
    from propevolve.reasoning_policy.decisive_learning import generalization_indices
    rows = [
        {"ticker": ticker, "target": action, "source_id": f"{ticker}-{action}-{i}",
         "completed_at_ns": 100+i, "label_end_ns": 110+i}
        for ticker in ("NQ", "ES") for action in ("WAIT", "HOLD") for i in range(3)
    ]
    selected = generalization_indices(rows, tickers=["NQ", "ES"], actions=["WAIT", "HOLD"],
        rows_per_group=2, start_ns=100, end_ns=120, training_end_ns=100, seed=17)
    assert len(selected) == 8
    assert len(set(selected)) == 8
    for ticker in ("NQ", "ES"):
        for action in ("WAIT", "HOLD"):
            assert sum(rows[i]["ticker"] == ticker and rows[i]["target"] == action for i in selected) == 2
    with pytest.raises(ValueError, match="chronological"):
        generalization_indices(rows, tickers=["NQ"], actions=["WAIT"], rows_per_group=1,
            start_ns=99, end_ns=120, training_end_ns=100, seed=17)


def test_exported_diagnostic_policy_can_be_read_by_the_real_assessment_interface(tmp_path):
    import json
    from pathlib import Path
    from propevolve.reasoning_policy.decisive_learning import evaluation_recipe
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    root = Path(__file__).resolve().parents[1]
    config = read_sft_config(root / 'config/reasoning/error_selected_distillation_sft.json', root=root)
    config['mastered_anchor_retention'] = {'loss_weight': 1., 'temperature': 1.}
    recipe = evaluation_recipe(config, adapter_path=str(tmp_path / 'adapter'))
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(recipe))
    result = read_sft_config(path, root=root)
    assert result['mastered_anchor_retention'] is None
    assert result['adapter_path'] == str(tmp_path / 'adapter')
    assert result['projector'] == config['projector']

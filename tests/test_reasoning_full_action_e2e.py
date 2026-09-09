"""Same-state simulator targets remain intact through the production objective."""
import numpy as np
import pytest
from propevolve.reasoning_policy.supervision import (
    action_completion_scores,
    action_objective,
    action_targets,
    completion_objective,
    mean_completion_scores,
)
from propevolve.reasoning_policy.dataset import supervised_record
from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.labels import label_actions
from test_reasoning_challenger_e2e import environment, passive_factory


@pytest.mark.parametrize("direction,winning_index", [(1, 1), (-1, 2)])
def test_full_action_loss_rewards_winner_and_rejects_failure_from_same_simulator_state(direction, winning_index):
    env = environment(direction)
    labels = label_actions(env, reset_options={"ticker": "NQ", "start": 0}, prefix=(),
        continuation_factory=passive_factory, max_steps=8)
    history = RollingContext(ContextConfig(2, ("balance",)))
    history.append(1, {"balance": 0})
    record = supervised_record(history.snapshot(), labels, source_id="test", continuation_id="fixed",
        target_temperature=1)
    targets = action_targets(record)
    assert targets["names"] == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    scores = np.zeros(3)
    config = {"soft_target_weight": 1., "ranking_weight": 1., "margin": .25}
    def loss(x):
        return action_objective(x, np.array(targets["probabilities"]),
            np.array(targets["values"]), config, xp=np)
    old = loss(scores)
    scores[winning_index] += .01
    assert loss(scores) < old
    scores[:] = 0
    scores[3 - winning_index] += .01
    assert loss(scores) > old
    assert record["targets"]["outcomes"][targets["names"][winning_index]]["outcome"] == "pass"


def test_equal_economic_values_do_not_manufacture_a_directional_margin():
    config = {"soft_target_weight": 0., "ranking_weight": 1., "margin": .25}
    assert action_objective(np.array([0., 5., -5.]), np.ones(3)/3,
        np.ones(3), config, xp=np) == 0


def test_action_score_is_not_biased_by_completion_token_count():
    token_log_probs = np.array([
        [-2., -2., 0., 0., 0.],
        [-2., -2., -2., -2., -2.],
    ])
    mask = np.array([
        [True, True, False, False, False],
        [True, True, True, True, True],
    ])
    np.testing.assert_allclose(mean_completion_scores(token_log_probs, mask, xp=np), [-2., -2.])


def test_mean_completion_objective_is_not_divided_by_token_count_twice():
    scores = mean_completion_scores(
        np.asarray([[-2.0, -2.0], [-4.0, -4.0]]),
        np.asarray([[True, True], [True, True]]), xp=np)
    assert completion_objective(scores, np.asarray([True, True]), xp=np) == pytest.approx(3.0)


def test_action_credit_uses_the_legal_action_token_not_shared_eos_formatting():
    token_log_probs = np.array([
        [-0.1, -9.0, 0.0],
        [-0.5, -0.01, 0.0],
    ])
    mask = np.array([
        [True, True, False],
        [True, True, False],
    ])
    np.testing.assert_allclose(
        action_completion_scores(token_log_probs, mask, xp=np), [-0.1, -0.5])

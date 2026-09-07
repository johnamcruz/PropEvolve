"""Same-state simulator targets remain intact through the production objective."""
import numpy as np
import pytest
from propevolve.reasoning_policy.supervision import action_objective, action_targets
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

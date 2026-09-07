"""Outcome uncertainty must not turn a losing context into an immutable entry."""
import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.decision import Action
from propevolve.replay import Transition


@pytest.mark.parametrize('reduction', [
    'equal_present_class_mean_v1', 'population_weighted_mean_v1',
])
def test_uncertain_outcomes_learn_economic_selection_and_survive_reload(tmp_path, reduction):
    """25% of +0.2/-0.1 loses money; 60% is profitable, on either side."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
        batch, observations = [], []
        for side in flat[1:]:
            for quality, winners in ((-1., 5), (1., 12)):
                obs = np.asarray([float(side), quality, 1.], np.float32)
                observations.append(obs)
                for i in range(20):
                    win = i < winners
                    batch.append((Transition(
                        observation=obs, action=side, reward=.2 if win else -.1,
                        next_observation=np.zeros(3, np.float32), terminated=True,
                        valid_actions=flat, next_valid_actions=(),
                        entry_action_target=side if win else Action.WAIT,
                    ),))
                batch.append((Transition(
                    observation=obs, action=Action.WAIT, reward=0.,
                    next_observation=np.zeros(3, np.float32), terminated=True,
                    valid_actions=flat, next_valid_actions=(),
                ),))
        agent = RecurrentC51Agent(
            3, hidden_dim=16, atoms=51, value_min=-3., value_max=3.,
            learning_rate=.003, weight_decay=.00001, gradient_clip=10.,
            gamma=.997, n_step_return=1, recurrent_burn_in=0,
            device='cpu', seed=314159, target_sync_updates=1000,
            entry_action_loss_weight=.9, entry_action_opportunity_loss_multiplier=2.,
            entry_action_loss_reduction=reduction,
            entry_action_class_weights=(1., 1., 1.), entry_action_margin=.25,
            auxiliary_gradient_conflict_mode='pcgrad_preserve_paired_boundaries_v4',
        )
        for _ in range(200):
            agent.train_batch(batch, teacher_weight_scale=0., entry_action_weight_scale=1.)
        expected = [Action.WAIT, Action.ENTER_LONG_1, Action.WAIT, Action.ENTER_SHORT_1]
        before = [agent.select_action(obs, hidden=None, valid_actions=flat,
                  epsilon=0., return_action_values=True) for obs in observations]
        assert [row[0] for row in before] == expected
        checkpoint = agent.save(tmp_path/'policy.pt', manifest={})
        restored, _ = RecurrentC51Agent.load(checkpoint, device='cpu')
        restored.discard_teacher()
        restored.discard_retention_anchor()
        restored.assert_teacher_free()
        after = [restored.select_action(obs, hidden=None, valid_actions=flat,
                 epsilon=0., return_action_values=True) for obs in observations]
        assert [row[0] for row in after] == expected
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(old[2], new[2])
    finally:
        torch.set_num_threads(previous_threads)

"""Entry preferences must not make abandoning a winner economically attractive."""
import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.decision import Action as A
from propevolve.replay import Transition


def test_resume_rejects_missing_economic_learning_state(tmp_path):
    agent = RecurrentC51Agent(2, hidden_dim=4, atoms=11,
        value_min=-3., value_max=3., gamma=.99, learning_rate=.001,
        weight_decay=0., gradient_clip=10., device='cpu', seed=1,
        target_sync_updates=10, economic_target_mode='td_only')
    path = agent.save(tmp_path / 'damaged.pt', manifest={})
    payload = torch.load(path, weights_only=False)
    payload['updates'] = 1
    payload.pop('economic_critic')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='economic critic state'):
        RecurrentC51Agent.load(path, device='cpu')


def test_curriculum_warm_start_preserves_economic_weights_with_fresh_optimizer(tmp_path):
    agent = RecurrentC51Agent(2, hidden_dim=4, atoms=11,
        value_min=-3., value_max=3., gamma=.99, learning_rate=.001,
        weight_decay=0., gradient_clip=10., device='cpu', seed=1,
        target_sync_updates=10, economic_target_mode='td_only')
    obs = np.ones(2, np.float32)
    agent.train_batch([(Transition(observation=obs, action=A.WAIT, reward=0.,
        next_observation=obs, terminated=True, valid_actions=(A.WAIT,),
        next_valid_actions=()),)])
    original = agent.save(tmp_path / 'parent.pt', manifest={})
    payload = torch.load(original, weights_only=False)
    warmed, _ = RecurrentC51Agent.warm_start(original,
        config=dict(payload['config'], device='cpu'))
    produced = torch.load(warmed.save(tmp_path / 'child.pt', manifest={}), weights_only=False)
    assert produced['economic_critic'] is not None
    for name in ('online', 'target'):
        for key, value in payload['economic_critic'][name].items():
            torch.testing.assert_close(produced['economic_critic'][name][key], value, atol=0., rtol=0.)
    assert produced['economic_critic']['updates'] == 0
    assert not produced['economic_critic']['optimizer']['state']


@pytest.mark.parametrize('winner', [A.ENTER_LONG_1, A.ENTER_SHORT_1])
@pytest.mark.parametrize('backend', ['pytorch', 'mlx'])
def test_supervised_entry_preserves_profitable_holding_after_reload(tmp_path, winner, backend):
    if backend == 'mlx':
        pytest.importorskip('mlx.core')
        if not torch.backends.mps.is_available():
            pytest.skip('MLX requires Apple Silicon')
    device = 'mps' if backend == 'mlx' else 'cpu'
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flat = (A.WAIT, A.ENTER_LONG_1, A.ENTER_SHORT_1)
        held = (A.HOLD, A.CLOSE)
        failure = A.ENTER_SHORT_1 if winner == A.ENTER_LONG_1 else A.ENTER_LONG_1
        f, h = np.eye(2, dtype=np.float32)
        def row(obs, action, reward, nxt, valid, next_valid, done, label):
            return (Transition(observation=obs, action=action, reward=reward,
                next_observation=nxt, terminated=done, valid_actions=valid,
                next_valid_actions=next_valid, entry_action_target=label,
                recurrent_reset=True, next_recurrent_reset=True),)
        batch = [row(f, A.WAIT, 0., f, flat, flat, False, winner),
                 row(f, winner, -.01, h, flat, held, False, winner),
                 row(f, failure, -.1, f, flat, (), True, winner),
                 row(h, A.HOLD, .21, f, held, (), True, None),
                 row(h, A.CLOSE, 0., f, held, flat, False, None)] * 8
        agent = RecurrentC51Agent(2, hidden_dim=16, atoms=51,
            value_min=-3., value_max=3., gamma=.99, learning_rate=.001,
            weight_decay=0., gradient_clip=10., device=device, seed=314159,
            learner_backend=backend,
            target_sync_updates=10, target_update_mode='soft', target_soft_tau=.05,
            entry_action_loss_weight=.9, entry_action_margin=.6,
            entry_action_opportunity_loss_multiplier=2., economic_target_mode='td_only')
        for _ in range(1000):
            agent.train_batch(batch, teacher_weight_scale=0.)
        def decisions(policy):
            return [policy.select_action(obs, hidden=None, valid_actions=valid,
                epsilon=0., return_action_values=True) for obs, valid in ((f, flat), (h, held))]
        before = decisions(agent)
        assert [item[0] for item in before] == [winner, A.HOLD]
        # Independent Bellman solution: HOLD=.21, CLOSE=.195921.
        np.testing.assert_allclose(before[1][2][[int(A.HOLD), int(A.CLOSE)]],
            [.21, .195921], atol=.015, rtol=0.)
        path = agent.save(tmp_path / 'learner.pt', manifest={})
        restored, _ = RecurrentC51Agent.load(path, device=device)
        for old, new in zip(before, decisions(restored), strict=True):
            np.testing.assert_array_equal(old[2], new[2])
        # Continuing training must restore the economic learner, not bootstrap
        # from the supervised policy after a checkpoint round trip.
        for _ in range(10):
            agent.train_batch(batch, teacher_weight_scale=0.)
            restored.train_batch(batch, teacher_weight_scale=0.)
        for old, new in zip(decisions(agent), decisions(restored), strict=True):
            np.testing.assert_array_equal(old[2], new[2])
        restored.discard_teacher()
        restored.discard_retention_anchor()
        restored.assert_teacher_free()
        assert [item[0] for item in decisions(restored)] == [winner, A.HOLD]
        # A new curriculum stage resets optimizers but must keep the calibrated
        # critic instead of initializing it from supervised preference values.
        settings = dict(torch.load(path, weights_only=False)['config'], device=device)
        warmed, _ = RecurrentC51Agent.warm_start(path, config=settings)
        warmed.discard_retention_anchor()
        for _ in range(25):
            warmed.train_batch(batch, teacher_weight_scale=0.)
        assert [item[0] for item in decisions(warmed)] == [winner, A.HOLD]
        np.testing.assert_allclose(decisions(warmed)[1][2][[int(A.HOLD), int(A.CLOSE)]],
            [.21, .195921], atol=.015, rtol=0.)
    finally:
        torch.set_num_threads(previous_threads)

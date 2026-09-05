from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import textwrap
import torch

pytest.importorskip("mlx.core")

from propevolve.agent import RecurrentC51Agent  # noqa: E402
from propevolve import mlx_backend  # noqa: E402


def test_completed_mlx_backward_releases_request_inputs_while_output_remains_valid():
    """A completed training update must not pin the last batch in its worker."""
    program = textwrap.dedent('''
        import gc
        import weakref
        import torch
        from propevolve.agent import RecurrentC51Agent
        from propevolve.mlx_backend import mlx_memory_metrics, shutdown_mlx_backend
        agent = RecurrentC51Agent(6, hidden_dim=8, atoms=11,
            value_min=-3., value_max=3., gamma=.997, learning_rate=1e-4,
            weight_decay=1e-5, gradient_clip=10., target_sync_updates=250,
            device="mps", seed=37, learner_backend="mlx")
        values = torch.ones(2, 7, 6, device="mps")
        reference = weakref.ref(values)
        result, hidden = agent.online(values)
        expected = result.cpu().clone()
        result.sum().backward()
        agent.online.zero_grad(set_to_none=True)
        del values, hidden
        mlx_memory_metrics()  # complete a subsequent request before inspecting ownership
        gc.collect()
        assert reference() is None, "completed backward retained its input batch"
        torch.testing.assert_close(result.cpu(), expected)
        shutdown_mlx_backend()
    ''')
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mlx_worker_applies_runtime_cache_budget_in_real_process():
    program = textwrap.dedent('''
        import torch
        from propevolve.config import configure_runtime_environment
        from propevolve.agent import RecurrentC51Agent
        from propevolve.mlx_backend import mlx_memory_metrics, shutdown_mlx_backend
        configure_runtime_environment({"mps_prefer_metal": False,
            "mps_fast_math": False, "mlx_cache_limit_bytes": 262144})
        settings = dict(hidden_dim=8, atoms=11, value_min=-3., value_max=3.,
            gamma=.997, learning_rate=1e-4, weight_decay=1e-5,
            gradient_clip=10., target_sync_updates=250, device="mps", seed=37)
        baseline = RecurrentC51Agent(6, **settings)
        actual = RecurrentC51Agent(6, learner_backend="mlx", **settings)
        values = torch.ones(2, 7, 6, device="mps")
        expected, _ = baseline.online(values)
        for _ in range(3):
            result, _ = actual.online(values)
            torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-4)
            result.sum().backward()
            actual.online.zero_grad(set_to_none=True)
        metrics = mlx_memory_metrics()
        assert metrics["configured_cache_limit_bytes"] == 262144, metrics
        assert metrics["cache_memory_bytes"] <= 262144, metrics
        shutdown_mlx_backend()
    ''')
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_completed_update_reclaims_unused_mps_cache_without_changing_live_values():
    program = textwrap.dedent('''
        import os
        import torch
        from propevolve.agent import RecurrentC51Agent
        from propevolve.mlx_backend import shutdown_mlx_backend
        os.environ['PROPEVOLVE_MPS_CACHE_CLEAR_THRESHOLD_BYTES'] = '0'
        agent = RecurrentC51Agent(6, hidden_dim=8, atoms=11,
            value_min=-3., value_max=3., gamma=.997, learning_rate=1e-4,
            weight_decay=1e-5, gradient_clip=10., target_sync_updates=250,
            device="mps", seed=37, learner_backend="mlx")
        values = torch.ones(2, 7, 6, device='mps')
        result, _ = agent.online(values)
        result.sum().backward()
        expected = result.detach().cpu().clone()
        expected_gradients = [p.grad.cpu().clone() for p in agent.online.parameters()]
        temporary = torch.ones(4 * 1024 * 1024, device='mps')
        torch.mps.synchronize()
        del temporary
        before = torch.mps.driver_allocated_memory()
        assert agent.release_runtime_cache()
        after = torch.mps.driver_allocated_memory()
        assert after < before - 8 * 1024 * 1024, (before, after)
        torch.testing.assert_close(result.detach().cpu(), expected)
        for parameter, gradient in zip(agent.online.parameters(), expected_gradients):
            torch.testing.assert_close(parameter.grad.cpu(), gradient)
        agent.online.zero_grad(set_to_none=True)
        agent.online(values)[0].sum().backward()
        for parameter, gradient in zip(agent.online.parameters(), expected_gradients):
            torch.testing.assert_close(parameter.grad.cpu(), gradient)
        shutdown_mlx_backend()
    ''')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True,
                            text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mlx_projects_the_complete_recurrent_input_sequence_once() -> None:
    mx = mlx_backend._mlx_core()
    encoded = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    weight = mx.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=mx.float32,
    )
    bias = mx.array([0.5, -0.5, 1.0, -1.0, 2.0, -2.0], dtype=mx.float32)

    projected = mlx_backend._mlx_recurrent_input_projection(
        encoded,
        weight,
        bias,
    )
    mx.eval(projected)

    np.testing.assert_allclose(
        np.asarray(projected),
        np.array(
            [
                [
                    [1.5, 1.5, 4.0, 0.0, 4.0, 2.0],
                    [3.5, 3.5, 8.0, 0.0, 8.0, 6.0],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_mlx_reuses_shared_recurrent_reset_and_burn_in_contract() -> None:
    settings = {
        "hidden_dim": 8,
        "atoms": 11,
        "value_min": -3.0,
        "value_max": 3.0,
        "gamma": 0.997,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "gradient_clip": 10.0,
        "target_sync_updates": 250,
        "device": "mps",
        "seed": 37,
    }
    torch_agent = RecurrentC51Agent(6, **settings)
    mlx_agent = RecurrentC51Agent(6, learner_backend="mlx", **settings)
    observations = np.random.default_rng(41).normal(
        size=(3, 7, 6)
    ).astype(np.float32)
    resets = (
        (True, False, False, False, True, False, False),
        (True, False, False, False, True, False, False),
        (True, False, True, False, False, False, False),
    )
    values = torch.from_numpy(observations).to("mps")

    expected_recurrent, expected_hidden = (
        RecurrentC51Agent._recurrent_features_with_resets(
            torch_agent.online,
            values,
            resets,
        )
    )
    actual_recurrent, actual_hidden = (
        RecurrentC51Agent._recurrent_features_with_resets(
            mlx_agent.online,
            values,
            resets,
        )
    )
    expected_logits = torch_agent.online.distribution_logits(
        expected_recurrent
    )
    actual_logits = mlx_agent.online.distribution_logits(actual_recurrent)

    torch.testing.assert_close(
        actual_recurrent, expected_recurrent, atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        actual_hidden, expected_hidden, atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        actual_logits, expected_logits, atol=2e-5, rtol=2e-4
    )


@pytest.mark.parametrize('resets', [
    ((False,) * 7,) * 3,
    ((True,) * 7,) * 3,
    ((True, False, False, False, False, False, True),
     (False, False, True, True, False, False, False),
     (False, True, False, False, True, False, False)),
])
def test_mlx_reset_mask_preserves_hidden_input_and_parameter_gradients(resets):
    settings = dict(hidden_dim=8, atoms=11, value_min=-3., value_max=3.,
        gamma=.997, learning_rate=1e-4, weight_decay=1e-5,
        gradient_clip=10., target_sync_updates=250, device='mps', seed=37)
    baseline = RecurrentC51Agent(6, **{**settings, 'device': 'cpu'})
    baseline.online.double()
    actual = RecurrentC51Agent(6, learner_backend='mlx', **settings)
    source = torch.arange(126, dtype=torch.float32).reshape(3, 7, 6) / 100
    results = []
    for agent in (baseline, actual):
        dtype = next(agent.online.parameters()).dtype
        inputs = source.to(device=agent.device, dtype=dtype).detach().requires_grad_()
        hidden = torch.ones(1, 3, 8, device=agent.device, dtype=dtype, requires_grad=True)
        recurrent, final = agent._recurrent_features_with_resets(
            agent.online, inputs, resets, hidden)
        (recurrent.square().sum() + final.square().sum()).backward()
        results.append((recurrent.detach(), final.detach(), inputs.grad,
                        torch.zeros_like(hidden) if hidden.grad is None else hidden.grad,
                        tuple(p.grad for p in agent.online.parameters() if p.grad is not None)))
    for index, (expected, observed) in enumerate(zip(results[0][:-1], results[1][:-1])):
        # Near-constant rows amplify FP32 LayerNorm input-gradient roundoff;
        # the independent FP64 reference keeps this absolute error bounded.
        tolerance = 1e-4 if index == 2 else 3e-5
        torch.testing.assert_close(observed.cpu().double(), expected.cpu().double(), atol=tolerance, rtol=5e-4)
    assert len(results[0][-1]) == len(results[1][-1])
    for expected, observed in zip(results[0][-1], results[1][-1]):
        torch.testing.assert_close(observed.cpu().double(), expected.cpu().double(), atol=3e-5, rtol=5e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_mlx_compiled_autograd_worker_exits_cleanly() -> None:
    """Reproduce the MLX compile-cache crash during Python finalization."""
    program = textwrap.dedent(
        """
        import numpy as np

        from propevolve.agent import RecurrentC51Agent
        from propevolve.decision import Action
        from propevolve.replay import Transition

        agent = RecurrentC51Agent(
            4,
            hidden_dim=8,
            atoms=11,
            value_min=-3.0,
            value_max=3.0,
            gamma=0.997,
            learning_rate=1e-4,
            weight_decay=1e-5,
            gradient_clip=10.0,
            target_sync_updates=250,
            device="mps",
            seed=73,
            learner_backend="mlx",
        )
        observations = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
        sequence = tuple(
            Transition(
                observation=observations[index],
                action=Action.WAIT,
                reward=0.1,
                next_observation=observations[index + 1],
                terminated=index == 3,
                valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
                next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
            )
            for index in range(4)
        )
        for _ in range(4):
            agent.train_batch((sequence, sequence))
        print("clean-mlx-worker", flush=True)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean-mlx-worker"

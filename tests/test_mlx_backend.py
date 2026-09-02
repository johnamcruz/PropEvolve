from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import textwrap
import torch

pytest.importorskip("mlx.core")

from propevolve.agent import RecurrentC51Agent  # noqa: E402


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

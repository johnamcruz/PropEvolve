from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.audit_frozen_checkpoint_batch import (
    challenge_outcome_cohort,
    discounted_returns_to_go,
    make_replay,
)


def test_discounted_challenge_returns_preserve_full_episode_credit() -> None:
    returns = discounted_returns_to_go(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        discount=0.5,
    )

    np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])


def test_challenge_outcome_cohorts_separate_pass_and_safety_failures() -> None:
    assert challenge_outcome_cohort("pass", 6_100.0, -2_250.0) == "pass"
    assert challenge_outcome_cohort("blow", -3_000.0, -2_250.0) == "blow"
    assert (
        challenge_outcome_cohort("timeout", -2_700.0, -2_250.0)
        == "near_blow_timeout"
    )
    assert (
        challenge_outcome_cohort("timeout", 100.0, -2_250.0)
        == "nonnegative_timeout"
    )


def test_frozen_audit_rebuilds_exact_replay_contract() -> None:
    contract = {
        "capacity_episodes": 8,
        "capacity_transitions": 128,
        "sequence_length": 6,
        "recurrent_burn_in": 2,
        "n_step_return": 2,
        "terminal_sequence_fraction": 0.25,
        "safety_sequence_fraction": 0.25,
        "entry_opportunity_sequence_fraction": 0.5,
        "regime_wait_sequence_fraction": 0.0,
        "regime_wait_sequence_update_period": 8,
        "entry_opportunity_side_balance": "paired_recurrent_long_short_v1",
        "paired_a_plus_population_weighting": "equal_pair_mass_v1",
    }

    replay = make_replay(contract, 13, [])

    assert replay.paired_a_plus_population_weighting == "equal_pair_mass_v1"
    assert replay.paired_a_plus_context_matching == (
        "static_expansion_regime_v1"
    )


def test_frozen_batch_audit_cli_is_path_driven() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "audit_frozen_checkpoint_batch.py"),
            "--help",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--attempt-dir" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--replay-root" in result.stdout
    assert "--output" in result.stdout
    assert "--near-blow-pnl" in result.stdout
    assert "--pair-count" in result.stdout
    assert "--challenge-return-discount" in result.stdout
    assert "--challenge-return-weight" in result.stdout
    assert "--optimizer-overfit-updates" in result.stdout
    assert "--paired-context-matching" in result.stdout
    assert "--violation-prioritized-pairs-per-side" in result.stdout

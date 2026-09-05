from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.audit_frozen_checkpoint_batch import (
    causal_feature_families,
    challenge_return_credit_coverage,
    challenge_outcome_cohort,
    discounted_returns_to_go,
    make_replay,
)
from propevolve.decision import Action
from propevolve.replay import Transition


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


def test_causal_feature_audit_uses_expansion_regime_and_trend_lifecycle() -> None:
    context = np.asarray((0.9, 0.8, 0.2, 0.1, 0.1, 0.7, 0.2), np.float32)
    targets = np.zeros((6, 11), np.float32)
    targets[:, :4] = np.asarray((0.9, 0.8, 0.2, 0.1), np.float32)
    targets[:, 7:11] = np.asarray((0.8, 0.2, 0.7, 0.1), np.float32)
    targets[-1, 0] = 1.0
    targets[-1, 7] = 0.9

    families = causal_feature_families(
        context,
        targets,
        5,
        Action.ENTER_LONG_1,
    )

    assert set(families) == {
        "static_expansion",
        "regime",
        "static_expansion_regime",
        "expansion_lifecycle",
        "trend_lifecycle",
        "combined_causal_lifecycle",
    }
    assert families["combined_causal_lifecycle"].shape == (27,)


def test_challenge_credit_audit_distinguishes_burn_in_from_learnable_rows() -> None:
    transitions = tuple(
        Transition(
            observation=np.asarray((index,), np.float32),
            action=(Action.WAIT, Action.ENTER_LONG_1, Action.HOLD)[index],
            reward=0.0,
            next_observation=np.asarray((index + 1,), np.float32),
            terminated=False,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.HOLD),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.HOLD),
            challenge_return_to_go=1.0,
        )
        for index in range(3)
    )

    coverage = challenge_return_credit_coverage(
        (transitions,),
        recurrent_burn_in=1,
        n_step_return=1,
    )

    assert coverage == {
        "sequences": 1,
        "credited_rows": 3,
        "learnable_credited_rows": 2,
        "discarded_credited_rows": 1,
        "actions": {"WAIT": 1, "ENTER_LONG_1": 1, "HOLD": 1},
    }
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


def test_audit_update_matches_explicit_production_call_after_reload(tmp_path):
    import torch
    from test_economic_replay_e2e import _agent, _economic_episode, _replay
    from scripts.audit_frozen_checkpoint_batch import actual_configured_update, qtrace
    from propevolve.agent import RecurrentC51Agent

    replay = _replay(seed=91)
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        for win in (True, False):
            replay.add(_economic_episode(
                episode_id=f"{side.name}-{win}", side=side, economic_win=win,
                context=(0.9, 0.8, 0.2, 0.1, 0.1, 0.7, 0.2),
                offset=1.0 + int(side) + int(win),
            ))
    sequences = replay.sample_paired_a_plus_candidate_pairs(1)
    agent = _agent()
    agent.policy_retention_loss_weight = 10.0
    # Give the saved anchor verified competence on the failure rows. Random
    # unlabelled entry predictions are no longer eligible for retention.
    with torch.no_grad():
        agent.online.output.bias.view(len(Action), agent.atoms)[int(Action.WAIT), -1] = 6.0
    agent.retain_policy()
    with torch.no_grad():
        next(agent.online.parameters()).add_(0.1)
    checkpoint = tmp_path / "arbitrary.pt"
    agent.save(checkpoint, manifest={})
    direct, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    before = qtrace(direct, sequences)[0][:, 0, :3].detach()
    call = dict(teacher_weight_scale=0.0, entry_action_weight_scale=1.0,
                retain_nonnegative_entry_policy=True)
    direct.train_batch(sequences, **call)
    expected = qtrace(direct, sequences)[0][:, 0, :3].detach()
    report = actual_configured_update(checkpoint, sequences, before, train_kwargs=call)
    assert report["train_call"] == call
    assert report["healthy_entry_policy_retention_rows"] > 0
    assert report["mean_abs_q_change"] == float((expected - before).abs().mean())

    from scripts.audit_frozen_checkpoint_batch import configured_optimizer_overfit_probe
    direct, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    mixed = (replay.sample(2),)
    for batch in (sequences, sequences, *mixed):
        direct.train_batch(batch, **call)
    expected = qtrace(direct, sequences)[0][:, 0, :3].detach()
    probe = configured_optimizer_overfit_probe(
        checkpoint, sequences, updates=2, train_kwargs=call, mixed_batches=mixed,
    )
    assert probe["train_call"] == call
    # Backtracking is observable; it does not prove an update was rejected.
    assert "rejected_updates" not in probe
    assert [x["phase"] for x in probe["trajectory"]] == [
        "before", "acquisition", "acquisition", "mixed",
    ]
    np.testing.assert_allclose(probe["q_after"], expected.numpy(), atol=1e-7, rtol=0)
    assert len(probe["trajectory"][-1]["boundary_rows"]) == 6
    assert probe["teacher_free_max_abs_q_drift"] == 0.0
    assert all(row["healthy_entry_retention_rows"] > 0
               for row in probe["trajectory"] if row["phase"] == "acquisition")
    ablation = configured_optimizer_overfit_probe(
        checkpoint, sequences, updates=2, train_kwargs=call,
        mixed_batches=mixed, retention_weight=0.0,
    )
    assert all(row["healthy_entry_retention_rows"] == 0
               for row in ablation["trajectory"][1:])
    assert not np.allclose(ablation["q_after"], probe["q_after"])

    from propevolve.training import _prioritize_paired_a_plus_violations
    candidates = replay.sample_paired_a_plus_candidate_pairs(1, pair_id_start=1000)
    direct, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    direct.train_batch(sequences, **call)
    selected, _ = _prioritize_paired_a_plus_violations(direct, candidates, pairs_per_side=1)
    direct.train_batch(tuple(mixed[0]) + selected, **call)
    expected = qtrace(direct, sequences)[0][:, 0, :3].detach()
    scheduled = configured_optimizer_overfit_probe(
        checkpoint, sequences, updates=1, train_kwargs=call, mixed_batches=mixed,
        mixed_candidate_batches=(candidates,), mixed_pairs_per_side=1,
    )
    np.testing.assert_allclose(scheduled["q_after"], expected.numpy(), atol=1e-7, rtol=0)
    assert scheduled["mixed_violation_updates"] == 1
    assert all(row["learning_occurrences"] >= 1
               for row in scheduled["trajectory"][1]["witness_exposure"])
    assert "rl_loss" in scheduled["trajectory"][-1]["learner_metrics"]
    from dataclasses import replace
    direct, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    direct.train_batch(sequences, **call)
    pair_offset = 1 + max(t.paired_a_plus_pair_id for s in mixed[0] for t in s
                          if t.paired_a_plus_pair_id is not None)
    rehearsal = tuple(tuple(replace(t, paired_a_plus_pair_id=t.paired_a_plus_pair_id + pair_offset)
                            if t.paired_a_plus_pair_id is not None else t for t in s)
                      for s in sequences)
    direct.train_batch(tuple(mixed[0]) + rehearsal, **call)
    expected = qtrace(direct, sequences)[0][:, 0, :3].detach()
    rehearse = configured_optimizer_overfit_probe(
        checkpoint, sequences, updates=1, train_kwargs=call, mixed_batches=mixed,
        mixed_rehearsal_period=1,
    )
    np.testing.assert_allclose(rehearse["q_after"], expected.numpy(), atol=1e-7, rtol=0)
    assert rehearse["mixed_rehearsal_updates"] == 1


def test_frozen_probe_uses_requested_mlx_production_backend(tmp_path):
    pytest.importorskip("mlx.core")
    from test_economic_replay_e2e import _agent, _economic_episode, _replay
    from propevolve.agent import RecurrentC51Agent
    from scripts.audit_frozen_checkpoint_batch import configured_optimizer_overfit_probe, qtrace

    replay = _replay(seed=91)
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        for win in (True, False):
            replay.add(_economic_episode(
                episode_id=f"{side.name}-{win}", side=side, economic_win=win,
                context=(0.9, 0.8, 0.2, 0.1, 0.1, 0.7, 0.2),
                offset=1.0 + int(side) + int(win),
            ))
    seqs = replay.sample_paired_a_plus_candidate_pairs(1)
    checkpoint = tmp_path / "native.pt"
    _agent().save(checkpoint, manifest={})
    backend = dict(device="mps", learner_backend_override="mlx")
    direct, _ = RecurrentC51Agent.load(checkpoint, **backend)
    call = dict(teacher_weight_scale=0.0, entry_action_weight_scale=1.0,
                retain_nonnegative_entry_policy=True)
    direct.train_batch(seqs, **call)
    direct.train_batch(seqs, **call)
    expected = qtrace(direct, seqs)[0][:, 0, :3].detach().cpu().numpy()
    report = configured_optimizer_overfit_probe(
        checkpoint, seqs, updates=2, train_kwargs=call, **backend,
    )
    assert report["learner_backend"] == "mlx"
    np.testing.assert_allclose(report["q_after"], expected, atol=1e-6, rtol=0)


def test_frozen_audit_cli_serializes_scheduled_replay_result(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from test_economic_replay_e2e import _agent, _economic_episode, _replay
    from propevolve.replay import ReplayCheckpointStore

    replay = _replay(seed=91)
    for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
        for win in (True, False):
            replay.add(_economic_episode(
                episode_id=f"{side.name}-{win}", side=side, economic_win=win,
                context=(0.9, 0.8, 0.2, 0.1, 0.1, 0.7, 0.2),
                offset=1.0 + int(side) + int(win),
            ))
    store = ReplayCheckpointStore(tmp_path / "replay")
    checkpoint = tmp_path / "model.pt"
    _agent().save(checkpoint, manifest={
        "replay_checkpoint": store.persist(replay),
        "progress": {"completed_episodes": 4}, "resume_identity": "test",
    })
    call = tmp_path / "call.json"
    call.write_text(json.dumps(dict(teacher_weight_scale=0., entry_action_weight_scale=1.,
                                    retain_nonnegative_entry_policy=True)))
    config = tmp_path / "anything.json"
    config.write_text(json.dumps({"training": {
        "paired_a_plus_violation_replay_update_period": 1,
        "paired_a_plus_violation_candidate_pairs_per_side": 1,
        "paired_a_plus_violation_pairs_per_side": 1,
    }}))
    output = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, "scripts/audit_frozen_checkpoint_batch.py",
        "--checkpoint", str(checkpoint), "--replay-root", str(store.root),
        "--output", str(output), "--pair-count", "2",
        "--train-call-config", str(call), "--optimizer-overfit-updates", "1",
        "--mixed-updates", "1", "--mixed-batch-sequences", "2",
        "--mixed-training-config", str(config),
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["replay_authentication"]["verified_shards"] == 4
    assert report["configured_optimizer_overfit_probe"]["mixed_violation_updates"] == 1
    assert report["candidate_pool"]["long_winner"] > 0
    repeated = subprocess.run([
        sys.executable, "scripts/audit_frozen_checkpoint_batch.py",
        "--checkpoint", str(checkpoint), "--replay-root", str(store.root),
        "--output", str(tmp_path / "repeated.json"), "--pair-count", "2",
        "--reference-audit", str(output),
        "--train-call-config", str(call), "--optimizer-overfit-updates", "1",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    matched = json.loads((tmp_path / "repeated.json").read_text())
    assert matched["audited_pairs"] == report["audited_pairs"]
    assert matched["configured_optimizer_overfit_probe"]["margins_after_acquisition"] == report["configured_optimizer_overfit_probe"]["margins_after_acquisition"]

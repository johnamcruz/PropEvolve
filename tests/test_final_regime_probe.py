from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.balance_aware_regime_selectivity import (
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
)
from propevolve.decision import Action
from propevolve.final_regime_probe import evaluate_final_regime_probe
from propevolve.replay import (
    BalancedSequenceReplay,
    Episode,
    Transition,
    final_regime_probe_row_identity,
)
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS
from propevolve.training import _training_evaluation_gates


CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)
FLAT_ACTIONS = (
    Action.WAIT,
    Action.ENTER_LONG_1,
    Action.ENTER_SHORT_1,
)


def _teacher_row(
    *,
    chop: float,
    trend: float,
    chop_persistence: float,
    transition_readiness: float,
) -> np.ndarray:
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    updates = {
        "structure_chop_probability": chop,
        "structure_neutral_probability": 0.1,
        "structure_trend_probability": trend,
        "structure_chop_persistence_probability": chop_persistence,
        "structure_trend_onset_probability": transition_readiness,
        "structure_trend_persistence_probability": transition_readiness,
        "volatility_expansion_onset_probability": transition_readiness,
        "volatility_high_persistence_probability": transition_readiness,
        "kaufman_efficiency": transition_readiness,
        "volatility_percentile": transition_readiness,
    }
    for channel, value in updates.items():
        values[CHANNELS.index(channel)] = value
    return values


def _transition(
    code: float,
    target: Action | None,
    teacher: np.ndarray,
    *,
    reset: bool = False,
    source_decision_index: int,
) -> Transition:
    observation = np.asarray((code, 0.0, 0.0), dtype=np.float32)
    return Transition(
        observation=observation,
        action=Action.WAIT,
        reward=0.0,
        next_observation=observation.copy(),
        terminated=False,
        valid_actions=FLAT_ACTIONS,
        next_valid_actions=FLAT_ACTIONS,
        recurrent_reset=reset,
        next_recurrent_reset=False,
        teacher_target=teacher,
        entry_action_target=target,
        regime_selectivity_headroom_fraction=(
            1.0 if target is not None else None
        ),
        training_valid=True,
        source_decision_index=source_decision_index,
    )


def _episode(
    episode_id: str,
    code: float,
    target: Action,
    teacher: np.ndarray,
    source_decision_index: int,
) -> Episode:
    # The anchor is at index one, after one authentic recurrent burn-in row.
    first = _transition(
        -99.0,
        None,
        teacher,
        reset=True,
        source_decision_index=max(0, source_decision_index - 1),
    )
    anchor = _transition(
        code,
        target,
        teacher,
        source_decision_index=source_decision_index,
    )
    final_observation = np.asarray((code + 0.25, 0.0, 0.0), np.float32)
    anchor = Transition(
        **{
            **anchor.__dict__,
            "next_observation": final_observation,
            "next_valid_actions": (Action.WAIT,),
            "terminated": True,
        }
    )
    first = Transition(**{**first.__dict__, "next_observation": anchor.observation})
    return Episode(
        episode_id=episode_id,
        ticker="NQ",
        outcome="timeout",
        primary_side="flat",
        ended_at_ns=10,
        transitions=(first, anchor),
    )


def _probe_replay() -> BalancedSequenceReplay:
    replay = BalancedSequenceReplay(
        capacity_episodes=128,
        sequence_length=2,
        recurrent_burn_in=1,
        seed=47,
    )
    rows = []
    for index in range(16):
        rows.append((
            f"wait-dead-{index}",
            0.0,
            Action.WAIT,
            _teacher_row(
                chop=0.9,
                trend=0.05,
                chop_persistence=0.95,
                transition_readiness=0.0,
            ),
        ))
        rows.append((
            f"wait-ready-{index}",
            1.0,
            Action.WAIT,
            _teacher_row(
                chop=0.05,
                trend=0.9,
                chop_persistence=0.95,
                transition_readiness=0.95,
            ),
        ))
    for index in range(32):
        regime = (
            {"chop": 0.9, "trend": 0.05, "transition_readiness": 0.0}
            if index < 16
            else {"chop": 0.05, "trend": 0.9, "transition_readiness": 0.95}
        )
        ready = _teacher_row(
            chop_persistence=0.95,
            **regime,
        )
        rows.append((f"long-ready-{index}", 2.0, Action.ENTER_LONG_1, ready))
        rows.append((f"short-ready-{index}", 3.0, Action.ENTER_SHORT_1, ready))
    for source_index, (episode_id, code, target, teacher) in enumerate(rows, 100):
        replay.add(_episode(
            episode_id,
            code,
            target,
            teacher,
            source_decision_index=source_index,
        ))
    return replay


class ScriptedFinalPolicy:
    def __init__(self, values_by_code: dict[int, tuple[float, float, float]]) -> None:
        self.values_by_code = values_by_code
        self.calls = 0

    def greedy_sequence_action_values(self, sequences):
        self.calls += 1
        action_rows = []
        value_rows = []
        for sequence in sequences:
            sequence_actions = []
            sequence_values = []
            for transition in sequence:
                code = int(transition.observation[0])
                flat_values = self.values_by_code.get(code, (1.0, 0.0, 0.0))
                values = np.full(len(Action), -np.inf, dtype=np.float64)
                values[:3] = flat_values
                sequence_values.append(values)
                sequence_actions.append(int(np.argmax(values)))
            action_rows.append(sequence_actions)
            value_rows.append(sequence_values)
        return np.asarray(action_rows), np.asarray(value_rows)


def _evaluate(policy: ScriptedFinalPolicy, replay: BalancedSequenceReplay):
    samples = replay.final_regime_probe_sequences(samples_per_action=32)
    return evaluate_final_regime_probe(
        policy,
        samples,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )


def _aggregate_learning_exposure() -> dict[str, float]:
    return {
        "latest_teacher_weight_scale": 0.0,
        "latest_entry_action_weight_scale": 0.0,
        "sampled_entry_action_long_rows": 100.0,
        "sampled_entry_action_short_rows": 100.0,
        "sampled_entry_action_long_recall": 0.8,
        "sampled_entry_action_short_recall": 0.8,
        "regime_selectivity_positive_long_rows": 100.0,
        "regime_selectivity_positive_short_rows": 100.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 50.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 50.0,
        "regime_entry_conflict_long_rows": 100.0,
        "regime_entry_conflict_short_rows": 100.0,
        "regime_entry_conflict_long_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_short_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_long_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_short_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_long_soft_wait_disagreement_rows": 0.0,
        "regime_entry_conflict_short_soft_wait_disagreement_rows": 0.0,
        "regime_selectivity_exact_wait_rows": 100.0,
        "regime_selectivity_exact_wait_weight_mean": 1.5,
        "regime_selectivity_persistent_dead_chop_weight_sum": 50.0,
        "regime_selectivity_transition_ready_weight_sum": 50.0,
        "regime_selectivity_transition_positive_long_rows": 50.0,
        "regime_selectivity_transition_positive_short_rows": 50.0,
        "regime_selectivity_transition_positive_long_"
        "declared_side_probability_sum": 25.0,
        "regime_selectivity_transition_positive_short_"
        "declared_side_probability_sum": 25.0,
    }


def test_final_probe_is_fixed_balanced_and_does_not_advance_replay_rng() -> None:
    replay = _probe_replay()
    before = copy.deepcopy(replay.state_dict()["random_state"])

    first = replay.final_regime_probe_sequences(samples_per_action=32)
    second = replay.final_regime_probe_sequences(samples_per_action=32)

    assert replay.state_dict()["random_state"] == before
    assert [row.row_identity_sha256 for row in first] == [
        row.row_identity_sha256 for row in second
    ]
    assert [row.target_action for row in first].count(Action.WAIT) == 32
    assert [row.target_action for row in first].count(Action.ENTER_LONG_1) == 32
    assert [row.target_action for row in first].count(Action.ENTER_SHORT_1) == 32
    assert all(row.sequence[1].training_valid for row in first)
    assert all(row.sequence[1].entry_action_target == row.target_action for row in first)
    competent = ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    })
    report = evaluate_final_regime_probe(
        competent,
        first,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )
    assert len(report.rows) == 96
    assert all(row["source_decision_index"] >= 100 for row in report.rows)


def test_final_probe_fails_closed_when_any_class_has_fewer_than_32_rows() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=2,
        recurrent_burn_in=1,
        seed=47,
    )
    teacher = _teacher_row(
        chop=0.05,
        trend=0.9,
        chop_persistence=0.95,
        transition_readiness=0.95,
    )
    replay.add(_episode(
        "one-long",
        2.0,
        Action.ENTER_LONG_1,
        teacher,
        source_decision_index=100,
    ))

    with pytest.raises(ValueError, match="exact balanced authentic rows"):
        replay.final_regime_probe_sequences(samples_per_action=32)


def test_probe_row_selection_is_independent_of_wall_clock_episode_identity() -> None:
    teacher = _teacher_row(
        chop=0.05,
        trend=0.9,
        chop_persistence=0.95,
        transition_readiness=0.95,
    )
    first = _episode(
        "historical-1-111111",
        2.0,
        Action.ENTER_LONG_1,
        teacher,
        source_decision_index=1234,
    )
    second = _episode(
        "historical-1-999999",
        2.0,
        Action.ENTER_LONG_1,
        teacher,
        source_decision_index=1234,
    )
    replay_a = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=2,
        recurrent_burn_in=1,
        seed=1,
    )
    replay_b = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=2,
        recurrent_burn_in=1,
        seed=999,
    )
    replay_a.add(first)
    replay_b.add(second)

    assert final_regime_probe_row_identity(
        ticker="NQ",
        source_decision_index=1234,
        target_action=Action.ENTER_LONG_1,
        observation=first.transitions[1].observation,
        teacher_target=first.transitions[1].teacher_target,
    ) == final_regime_probe_row_identity(
        ticker="NQ",
        source_decision_index=1234,
        target_action=Action.ENTER_LONG_1,
        observation=second.transitions[1].observation,
        teacher_target=second.transitions[1].teacher_target,
    )

    duplicate_replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=2,
        recurrent_burn_in=1,
        seed=9,
    )
    duplicate_replay.add(first)
    duplicate_replay.add(second)
    with pytest.raises(ValueError, match="exact balanced authentic rows"):
        duplicate_replay.final_regime_probe_sequences(samples_per_action=2)


def test_final_probe_rejects_early_good_diagnostics_when_final_policy_collapsed() -> None:
    replay = _probe_replay()
    collapsed = ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (3.0, 0.0, 0.0),
        2: (3.0, 0.0, 0.0),
        3: (3.0, 0.0, 0.0),
    })
    report = _evaluate(collapsed, replay)
    # These whole-run aggregates describe an early competent period. They must
    # not let the final collapsed checkpoint pass the mechanism gate.
    early_good_optimizer_diagnostics = _aggregate_learning_exposure()
    metrics = {
        "short_circuited": 0.0,
        **early_good_optimizer_diagnostics,
        **report.metrics,
    }
    gates = _training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=(
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
    )

    assert report.metrics["final_regime_probe_long_recall"] == 0.0
    assert report.metrics["final_regime_probe_short_recall"] == 0.0
    assert not all(gate.passes(metrics) for gate in gates)

    competent = ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    })
    competent_report = _evaluate(competent, replay)
    assert competent_report.sample_identity_sha256 == report.sample_identity_sha256


def test_final_probe_accepts_competent_final_policy_and_reports_regime_contrast() -> None:
    replay = _probe_replay()
    competent = ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    })

    report = _evaluate(competent, replay)
    gates = _training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=(
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
    )
    metrics = {
        "short_circuited": 0.0,
        **_aggregate_learning_exposure(),
        **report.metrics,
    }

    assert report.schema == "propevolve_final_regime_probe_v1"
    assert report.source_period == ("2021-01-01", "2025-01-01")
    assert report.sample_identity_sha256
    assert report.metrics["final_regime_probe_wait_rows"] == 32.0
    assert report.metrics["final_regime_probe_long_rows"] == 32.0
    assert report.metrics["final_regime_probe_short_rows"] == 32.0
    assert report.metrics["final_regime_probe_wait_recall"] == 1.0
    assert report.metrics["final_regime_probe_long_recall"] == 1.0
    assert report.metrics["final_regime_probe_short_recall"] == 1.0
    assert report.metrics[
        "final_regime_probe_dead_wait_minus_transition_ready_wait"
    ] > 0.0
    assert report.metrics[
        "final_regime_probe_transition_positive_long_response"
    ] > 0.0
    assert report.metrics[
        "final_regime_probe_transition_positive_short_response"
    ] > 0.0
    assert all(gate.passes(metrics) for gate in gates)


def test_final_probe_cannot_hide_missing_side_balanced_learning_exposure() -> None:
    report = _evaluate(ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    }), _probe_replay())
    exposure = _aggregate_learning_exposure()
    exposure["sampled_entry_action_short_rows"] = 0.0
    metrics = {"short_circuited": 0.0, **exposure, **report.metrics}
    gates = _training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=(
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
    )

    assert not all(gate.passes(metrics) for gate in gates)


def test_final_probe_side_recall_uses_the_frozen_40_percent_floor() -> None:
    report = _evaluate(ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    }), _probe_replay())
    metrics = {
        "short_circuited": 0.0,
        **_aggregate_learning_exposure(),
        **report.metrics,
    }
    gates = _training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=(
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
    )

    metrics["final_regime_probe_short_recall"] = 1.0 / 32.0
    assert not all(gate.passes(metrics) for gate in gates)

    metrics["final_regime_probe_short_recall"] = 12.0 / 32.0
    assert not all(gate.passes(metrics) for gate in gates)
    metrics["final_regime_probe_short_recall"] = 13.0 / 32.0
    assert all(gate.passes(metrics) for gate in gates)


def test_static_probe_contrast_uses_positive_rows_not_wait_rows() -> None:
    replay = _probe_replay()
    # WAIT behavior differs by its observation code, but positive rows have
    # identical WAIT-vs-entry response across static chop/nonchop strata.
    policy = ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (0.0, 3.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    })

    report = _evaluate(policy, replay)

    assert report.metrics["final_regime_probe_dominant_chop_rows"] == 32.0
    assert report.metrics["final_regime_probe_nonchop_rows"] == 32.0
    assert report.metrics["final_regime_probe_chop_minus_nonchop_wait"] == pytest.approx(
        0.0
    )


def test_static_probe_detects_positive_entry_wait_suppression_in_chop() -> None:
    replay = _probe_replay()

    class RegimeAwarePolicy(ScriptedFinalPolicy):
        def greedy_sequence_action_values(self, sequences):
            actions, values = super().greedy_sequence_action_values(sequences)
            for row_index, sequence in enumerate(sequences):
                teacher = sequence[1].teacher_target
                assert teacher is not None
                chop = teacher[CHANNELS.index("structure_chop_probability")]
                target = sequence[1].entry_action_target
                if target in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1} and chop > 0.5:
                    values[row_index, 1, int(Action.WAIT)] = 3.0
                    values[row_index, 1, int(target)] = 0.0
                    actions[row_index, 1] = int(Action.WAIT)
            return actions, values

    policy = RegimeAwarePolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    })

    report = _evaluate(policy, replay)

    assert report.metrics["final_regime_probe_chop_minus_nonchop_wait"] > 0.0


def test_final_probe_rows_explain_predictions_with_regime_channels_and_strata() -> None:
    report = _evaluate(ScriptedFinalPolicy({
        0: (3.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 3.0, 0.0),
        3: (0.0, 0.0, 3.0),
    }), _probe_replay())

    row = report.rows[0]
    assert row["target_action"] in {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
    assert row["greedy_action"] in {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
    assert row["correct"] == (row["target_action"] == row["greedy_action"])
    assert row["flat_action_probabilities"].keys() == {
        "WAIT",
        "ENTER_LONG_1",
        "ENTER_SHORT_1",
    }
    assert set(row["regime_channels"]) == set(REGIME_CHANNELS)
    assert row["headroom_fraction"] == 1.0
    assert row["headroom_stratum"] == "safe_headroom_ge_0_75"
    assert row["static_regime_stratum"] in {"dominant_chop", "nonchop"}
    assert row["persistent_regime_strata"].keys() == {
        "persistent_dead_chop_membership",
        "transition_ready_membership",
        "transition_positive_long_membership",
        "transition_positive_short_membership",
    }
    confusion_total = sum(
        report.metrics[
            f"final_regime_probe_target_{target}_predicted_{prediction}_rows"
        ]
        for target in ("wait", "long", "short")
        for prediction in ("wait", "long", "short")
    )
    assert confusion_total == 96.0


def _real_agent(seed: int = 71) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        3,
        hidden_dim=16,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        recurrent_burn_in=1,
        device="cpu",
        seed=seed,
        teacher_channels=len(CHANNELS),
        teacher_channel_names=CHANNELS,
        teacher_loss_weight=1e-6,
        regime_selectivity_loss_weight=1.0,
        regime_selectivity_expansion_centers=(0.1, 0.1),
        regime_selectivity_side_balance="equal_long_short_v1",
        regime_selectivity_semantics=(
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
        regime_selectivity_persistent_chop_negative_emphasis=2.0,
    )


def test_real_policy_probe_preserves_model_optimizer_rng_and_teacher_free_parity(
    tmp_path: Path,
) -> None:
    replay = _probe_replay()
    samples = replay.final_regime_probe_sequences(samples_per_action=32)
    agent = _real_agent()
    model_before = {
        key: value.detach().clone() for key, value in agent.online.state_dict().items()
    }
    optimizer_before = copy.deepcopy(agent.optimizer.state_dict())
    rng_before = copy.deepcopy(agent._rng.bit_generator.state)

    before = evaluate_final_regime_probe(
        agent,
        samples,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )

    assert all(
        torch.equal(model_before[key], value)
        for key, value in agent.online.state_dict().items()
    )
    assert agent.optimizer.state_dict() == optimizer_before
    assert agent._rng.bit_generator.state == rng_before

    resumable = agent.save(tmp_path / "with-teacher.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(resumable, device="cpu")
    restored_report = evaluate_final_regime_probe(
        restored,
        samples,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )
    assert restored_report.metrics == pytest.approx(before.metrics)

    agent.discard_teacher()
    discarded = evaluate_final_regime_probe(
        agent,
        samples,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )
    assert discarded.metrics == pytest.approx(before.metrics)
    teacher_free_path = agent.save(tmp_path / "teacher-free.pt", manifest={})
    teacher_free, _ = RecurrentC51Agent.load(teacher_free_path, device="cpu")
    teacher_free_report = evaluate_final_regime_probe(
        teacher_free,
        samples,
        teacher_channel_names=CHANNELS,
        q_temperature=1.0,
        source_period=("2021-01-01", "2025-01-01"),
    )
    assert teacher_free_report.metrics == pytest.approx(before.metrics)

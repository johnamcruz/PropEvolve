from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.replay import BalancedSequenceReplay, Episode, Transition


def _episode(ticker: str, outcome: str, side: str, offset: int) -> Episode:
    transitions = tuple(
        Transition(
            observation=np.array([offset + i], np.float32),
            action=Action.WAIT,
            reward=float(i),
            next_observation=np.array([offset + i + 1], np.float32),
            terminated=i == 5,
            valid_actions=(Action.WAIT,),
            next_valid_actions=() if i == 5 else (Action.WAIT,),
        )
        for i in range(6)
    )
    return Episode(
        episode_id=f"{ticker}-{outcome}-{offset}",
        ticker=ticker,
        outcome=outcome,
        primary_side=side,
        ended_at_ns=offset,
        transitions=transitions,
    )


def test_replay_is_bounded_and_samples_whole_causal_sequences() -> None:
    replay = BalancedSequenceReplay(capacity_episodes=3, sequence_length=4, seed=7)
    for i in range(5):
        replay.add(_episode("NQ", "pass" if i % 2 else "blow", "long", i))

    assert len(replay) == 3
    batch = replay.sample(3)
    assert len(batch) == 3
    for sequence in batch:
        assert len(sequence) == 4
        values = [int(item.observation[0]) for item in sequence]
        assert values == list(range(values[0], values[0] + 4))


def test_replay_balances_outcome_buckets_before_reusing_them() -> None:
    replay = BalancedSequenceReplay(capacity_episodes=20, sequence_length=2, seed=11)
    for i in range(8):
        replay.add(_episode("NQ", "pass", "long", i * 10))
    replay.add(_episode("ES", "blow", "short", 100))

    sampled = replay.sample_episodes(2)

    assert {episode.outcome for episode in sampled} == {"pass", "blow"}


def test_replay_balances_outcomes_across_a_multi_market_population() -> None:
    """Many timeout tickers must not drown out scarce pass competence."""
    replay = BalancedSequenceReplay(capacity_episodes=20, sequence_length=2, seed=43)
    for index, ticker in enumerate(("NQ", "ES", "YM", "RTY", "CL", "GC", "SI", "ZN", "ZB")):
        replay.add(_episode(ticker, "timeout", "long", index * 10))
    replay.add(_episode("NQ", "pass", "long", 1000))

    sampled = replay.sample_episodes(10)
    outcomes = Counter(episode.outcome for episode in sampled)

    assert outcomes == {"pass": 5, "timeout": 5}


def test_replay_anchors_declared_fraction_of_sequences_at_terminal_outcomes() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=20,
        sequence_length=3,
        terminal_sequence_fraction=1.0,
        seed=13,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    replay.add(_episode("ES", "blow", "short", 100))

    batch = replay.sample(2)

    assert all(sequence[-1].terminated for sequence in batch)
    assert {
        tuple(int(item.observation[0]) for item in sequence)
        for sequence in batch
    } == {(3, 4, 5), (103, 104, 105)}


def test_replay_anchors_declared_fraction_at_highest_mll_risk() -> None:
    episode = _episode("NQ", "timeout", "long", 0)
    safety_priorities = (0.0, 0.1, 0.2, 0.9, 0.1, 0.0)
    prioritized = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(**{**item.__dict__, "safety_priority": priority})
            for item, priority in zip(
                episode.transitions, safety_priorities, strict=True
            )
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=3,
        terminal_sequence_fraction=0.0,
        safety_sequence_fraction=1.0,
        seed=23,
    )
    replay.add(prioritized)

    sequence = replay.sample(1)[0]

    assert tuple(int(item.observation[0]) for item in sequence) == (1, 2, 3)
    assert sequence[-1].safety_priority == pytest.approx(0.9)


def test_replay_anchors_declared_fraction_at_expansion_entry_opportunity() -> None:
    episode = _episode("NQ", "timeout", "long", 0)
    entry_priorities = (0.0, 0.1, 0.2, 0.95, 0.1, 0.0)
    prioritized = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(**{**item.__dict__, "entry_opportunity_priority": priority})
            for item, priority in zip(
                episode.transitions, entry_priorities, strict=True
            )
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=3,
        entry_opportunity_sequence_fraction=1.0,
        seed=31,
    )
    replay.add(prioritized)

    sequence = replay.sample(1)[0]

    assert tuple(int(item.observation[0]) for item in sequence) == (1, 2, 3)
    assert sequence[-1].entry_opportunity_priority == pytest.approx(0.95)


def test_replay_caps_compact_storage_by_transition_budget() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=20,
        capacity_transitions=12,
        sequence_length=3,
        seed=17,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    replay.add(_episode("ES", "timeout", "short", 100))
    replay.add(_episode("GC", "timeout", "long", 200))

    assert replay.transition_count <= 12
    assert len(replay) == 2
    assert len(replay.sample(2)) == 2


def test_replay_preserves_optional_training_only_teacher_targets() -> None:
    episode = _episode("NQ", "pass", "long", 0)
    taught = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(
                **{
                    **item.__dict__,
                    "teacher_target": np.array([0.8, 0.7, 0.2, 0.1], np.float32),
                }
            )
            for item in episode.transitions
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        capacity_transitions=12,
        sequence_length=3,
        seed=19,
    )

    replay.add(taught)
    sampled = replay.sample(1)[0]

    assert all(item.teacher_target is not None for item in sampled)
    np.testing.assert_allclose(sampled[0].teacher_target, [0.8, 0.7, 0.2, 0.1])


def test_replay_eviction_preserves_rare_terminal_outcomes() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=3,
        capacity_transitions=18,
        sequence_length=3,
        seed=29,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    replay.add(_episode("ES", "timeout", "long", 100))
    replay.add(_episode("GC", "timeout", "long", 200))
    replay.add(_episode("CL", "timeout", "long", 300))

    retained = replay.sample_episodes(20)

    assert "pass" in {episode.outcome for episode in retained}


def test_replay_checkpoint_round_trip_preserves_memory_and_sampling_state() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        capacity_transitions=24,
        sequence_length=3,
        terminal_sequence_fraction=0.25,
        safety_sequence_fraction=0.25,
        entry_opportunity_sequence_fraction=0.25,
        seed=37,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    replay.add(_episode("ES", "timeout", "short", 100))
    replay.add(_episode("GC", "blow", "long", 200))
    state = replay.state_dict()
    restored = BalancedSequenceReplay(
        capacity_episodes=4,
        capacity_transitions=24,
        sequence_length=3,
        terminal_sequence_fraction=0.25,
        safety_sequence_fraction=0.25,
        entry_opportunity_sequence_fraction=0.25,
        seed=999,
    )

    restored.load_state_dict(state)

    assert len(restored) == len(replay)
    assert restored.transition_count == replay.transition_count
    expected = replay.sample(8)
    actual = restored.sample(8)
    for expected_sequence, actual_sequence in zip(expected, actual, strict=True):
        assert [item.action for item in actual_sequence] == [
            item.action for item in expected_sequence
        ]
        np.testing.assert_array_equal(
            [item.observation for item in actual_sequence],
            [item.observation for item in expected_sequence],
        )


def test_replay_round_trip_preserves_behavior_recurrent_reset_lineage() -> None:
    original = _episode("NQ", "pass", "long", 0)
    marked = Episode(
        episode_id=original.episode_id,
        ticker=original.ticker,
        outcome=original.outcome,
        primary_side=original.primary_side,
        ended_at_ns=original.ended_at_ns,
        transitions=tuple(
            Transition(
                **{
                    **item.__dict__,
                    "recurrent_reset": index in {0, 3},
                    "next_recurrent_reset": index + 1 in {0, 3},
                }
            )
            for index, item in enumerate(original.transitions)
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=41,
    )
    replay.add(marked)
    restored = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=99,
    )

    restored.load_state_dict(replay.state_dict())
    sampled = restored.sample(1)[0]

    assert [item.recurrent_reset for item in sampled] == [
        True, False, False, True, False, False
    ]
    assert [item.next_recurrent_reset for item in sampled] == [
        False, False, True, False, False, False
    ]


def test_replay_marks_only_demonstrated_pass_episodes_as_competence() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=47,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    replay.add(_episode("ES", "timeout", "short", 100))

    sampled = replay.sample(2)
    competence_by_origin = {
        int(sequence[0].observation[0]): all(
            transition.competence_anchor for transition in sequence
        )
        for sequence in sampled
    }

    assert competence_by_origin == {0: True, 100: False}

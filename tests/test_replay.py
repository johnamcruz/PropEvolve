from __future__ import annotations

import numpy as np

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
            next_valid_actions=(Action.WAIT,),
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

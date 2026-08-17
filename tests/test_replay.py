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


def test_replay_entry_anchor_keeps_wait_wait_enter_context() -> None:
    episode = _episode("NQ", "timeout", "long", 0)
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    targets = (
        None,
        Action.WAIT,
        Action.WAIT,
        Action.ENTER_LONG_1,
        None,
        None,
    )
    prioritized = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(**{
                **item.__dict__,
                "valid_actions": flat_actions,
                "next_valid_actions": () if item.terminated else flat_actions,
                "entry_action_target": target,
                "entry_opportunity_priority": float(
                    target in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
                ),
            })
            for item, target in zip(episode.transitions, targets, strict=True)
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=3,
        entry_opportunity_sequence_fraction=1.0,
        seed=37,
    )
    replay.add(prioritized)

    sequence = replay.sample(1)[0]

    assert tuple(item.entry_action_target for item in sequence) == (
        Action.WAIT,
        Action.WAIT,
        Action.ENTER_LONG_1,
    )


def _entry_opportunity_episode(
    *,
    episode_id: str,
    side: Action,
    offset: int,
) -> Episode:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    transitions = tuple(
        Transition(
            observation=np.array([offset + index], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([offset + index + 1], np.float32),
            terminated=index == 11,
            valid_actions=flat_actions,
            next_valid_actions=() if index == 11 else flat_actions,
            entry_action_target=(
                Action.WAIT if index in {3, 4} else side if index == 5 else None
            ),
            entry_opportunity_priority=1.0 if index == 5 else 0.0,
        )
        for index in range(12)
    )
    return Episode(
        episode_id=episode_id,
        ticker="NQ",
        outcome="timeout",
        primary_side="long" if side == Action.ENTER_LONG_1 else "short",
        ended_at_ns=offset,
        transitions=transitions,
    )


def test_replay_side_balances_entry_anchors_inside_the_learnable_window() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=8,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=41,
    )
    for index in range(4):
        replay.add(_entry_opportunity_episode(
            episode_id=f"long-{index}",
            side=Action.ENTER_LONG_1,
            offset=index * 100,
        ))
    replay.add(_entry_opportunity_episode(
        episode_id="short-0",
        side=Action.ENTER_SHORT_1,
        offset=1000,
    ))

    sampled = replay.sample(4)

    anchored_sides = []
    for sequence in sampled:
        learnable = sequence[2:5]
        anchors = [
            (index, row.entry_action_target)
            for index, row in enumerate(sequence)
            if row.entry_action_target in {
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            }
        ]
        assert len(anchors) == 1
        anchor_index, anchor_side = anchors[0]
        assert sequence[anchor_index] in learnable
        assert sequence[anchor_index - 2].entry_action_target == Action.WAIT
        assert sequence[anchor_index - 1].entry_action_target == Action.WAIT
        anchored_sides.append(anchor_side)
    assert Counter(anchored_sides) == {
        Action.ENTER_LONG_1: 2,
        Action.ENTER_SHORT_1: 2,
    }


def test_replay_side_balance_falls_back_to_the_only_authentic_side() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=43,
    )
    replay.add(_entry_opportunity_episode(
        episode_id="long-only",
        side=Action.ENTER_LONG_1,
        offset=0,
    ))

    sampled = replay.sample(3)

    for sequence in sampled:
        authentic_targets = [
            row.entry_action_target
            for row in sequence[2:5]
            if row.entry_action_target is not None
        ]
        assert authentic_targets == [Action.ENTER_LONG_1]
        assert all(
            row.entry_action_target != Action.ENTER_SHORT_1
            for row in sequence
        )


def test_replay_side_balance_splits_an_odd_entry_stratum_by_at_most_one() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=45,
    )
    replay.add(_entry_opportunity_episode(
        episode_id="long",
        side=Action.ENTER_LONG_1,
        offset=0,
    ))
    replay.add(_entry_opportunity_episode(
        episode_id="short",
        side=Action.ENTER_SHORT_1,
        offset=100,
    ))

    sampled = replay.sample(5)

    counts = Counter(
        row.entry_action_target
        for sequence in sampled
        for row in sequence[2:5]
        if row.entry_action_target in {
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
    )
    assert sum(counts.values()) == 5
    assert abs(counts[Action.ENTER_LONG_1] - counts[Action.ENTER_SHORT_1]) == 1


def test_replay_side_balance_never_fabricates_a_missing_entry_target() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=3,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        seed=46,
    )
    replay.add(_episode("NQ", "timeout", "long", 0))

    sampled = replay.sample(2)

    assert all(
        row.entry_action_target is None
        for sequence in sampled
        for row in sequence
    )


def test_replay_sampling_reuses_precomputed_entry_anchor_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=64,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=46,
    )
    for index in range(32):
        replay.add(_entry_opportunity_episode(
            episode_id=f"episode-{index}",
            side=(
                Action.ENTER_LONG_1
                if index % 2 == 0
                else Action.ENTER_SHORT_1
            ),
            offset=index * 100,
        ))

    calls = 0
    original_flatnonzero = np.flatnonzero

    def counted_flatnonzero(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_flatnonzero(values)

    monkeypatch.setattr(np, "flatnonzero", counted_flatnonzero)

    for _ in range(20):
        replay.sample(16)

    assert calls == 0


def test_replay_samples_uniformly_across_precomputed_entry_events() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    def episode_with_anchors(
        episode_id: str,
        offset: int,
        anchors: set[int],
    ) -> Episode:
        transitions = tuple(
            Transition(
                observation=np.array([offset + index], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([offset + index + 1], np.float32),
                terminated=index == 19,
                valid_actions=flat_actions,
                next_valid_actions=() if index == 19 else flat_actions,
                entry_action_target=(
                    Action.ENTER_LONG_1 if index in anchors else None
                ),
                entry_opportunity_priority=float(index in anchors),
            )
            for index in range(20)
        )
        return Episode(
            episode_id=episode_id,
            ticker="NQ",
            outcome="timeout",
            primary_side="long",
            ended_at_ns=offset,
            transitions=transitions,
        )

    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=47,
    )
    replay.add(episode_with_anchors("three-events", 0, {3, 8, 13}))
    replay.add(episode_with_anchors("one-event", 100, {3}))

    sampled = replay.sample(400)

    first_episode_count = sum(
        int(row.observation[0]) < 100
        for sequence in sampled
        for row in sequence
        if row.entry_action_target == Action.ENTER_LONG_1
    )
    assert 270 <= first_episode_count <= 330


def test_replay_side_balance_uses_only_retained_episode_anchors() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=46,
    )
    replay.add(_entry_opportunity_episode(
        episode_id="evicted-long",
        side=Action.ENTER_LONG_1,
        offset=0,
    ))
    replay.add(_entry_opportunity_episode(
        episode_id="retained-short",
        side=Action.ENTER_SHORT_1,
        offset=100,
    ))
    replay.add(_entry_opportunity_episode(
        episode_id="retained-long",
        side=Action.ENTER_LONG_1,
        offset=200,
    ))

    sampled = replay.sample(4)

    anchor_observations = {
        int(row.observation[0])
        for sequence in sampled
        for row in sequence
        if row.entry_action_target in {
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
    }
    assert anchor_observations == {105, 205}


def test_replay_checkpoint_versions_the_entry_side_balance_contract() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=47,
    )
    replay.add(_entry_opportunity_episode(
        episode_id="long",
        side=Action.ENTER_LONG_1,
        offset=0,
    ))
    replay.add(_entry_opportunity_episode(
        episode_id="short",
        side=Action.ENTER_SHORT_1,
        offset=100,
    ))

    state = replay.state_dict()

    assert state["schema_version"] == 7
    assert state["contract"]["entry_opportunity_side_balance"] == (
        "equal_long_short_v1"
    )
    restored = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="equal_long_short_v1",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=999,
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["contract"] == state["contract"]
    expected = replay.sample(5)
    actual = restored.sample(5)
    for expected_sequence, actual_sequence in zip(expected, actual, strict=True):
        assert [row.entry_action_target for row in actual_sequence] == [
            row.entry_action_target for row in expected_sequence
        ]
        np.testing.assert_array_equal(
            [row.observation for row in actual_sequence],
            [row.observation for row in expected_sequence],
        )

    stale = dict(state)
    stale["schema_version"] = 5
    with pytest.raises(ValueError, match="schema is unsupported"):
        restored.load_state_dict(stale)

    drifted = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=6,
        entry_opportunity_sequence_fraction=1.0,
        entry_opportunity_side_balance="none",
        recurrent_burn_in=2,
        n_step_return=2,
        seed=999,
    )
    with pytest.raises(ValueError, match="contract drifted"):
        drifted.load_state_dict(state)


def test_replay_round_trip_preserves_teacher_imitation_visibility() -> None:
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    episode = Episode(
        episode_id="dropout-row",
        ticker="NQ",
        outcome="timeout",
        primary_side="flat",
        ended_at_ns=1,
        transitions=(Transition(
            observation=np.zeros(1, np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.ones(1, np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=np.ones(7, np.float32),
            teacher_imitation_visible=False,
            entry_action_target=Action.WAIT,
            regime_selectivity_headroom_fraction=1.0,
        ),),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=1,
        sequence_length=1,
        seed=53,
    )
    replay.add(episode)
    restored = BalancedSequenceReplay(
        capacity_episodes=1,
        sequence_length=1,
        seed=53,
    )

    restored.load_state_dict(replay.state_dict())

    row = restored.sample(1)[0][0]
    assert row.teacher_target is not None
    assert row.entry_action_target == Action.WAIT
    assert row.teacher_imitation_visible is False


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


def test_replay_preserves_explicit_entry_action_targets_and_censoring() -> None:
    episode = _episode("NQ", "pass", "long", 0)
    targets = (
        Action.WAIT,
        Action.WAIT,
        Action.ENTER_LONG_1,
        None,
        None,
        None,
    )
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    taught = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(**{
                **item.__dict__,
                "valid_actions": flat_actions,
                "next_valid_actions": () if item.terminated else flat_actions,
                "entry_action_target": target,
            })
            for item, target in zip(episode.transitions, targets, strict=True)
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=23,
    )

    replay.add(taught)
    sampled = replay.sample(1)[0]

    assert tuple(item.entry_action_target for item in sampled) == targets


def test_replay_preserves_training_only_decision_headroom_for_regime_selectivity() -> None:
    episode = _episode("NQ", "timeout", "long", 0)
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    taught = Episode(
        episode_id=episode.episode_id,
        ticker=episode.ticker,
        outcome=episode.outcome,
        primary_side=episode.primary_side,
        ended_at_ns=episode.ended_at_ns,
        transitions=tuple(
            Transition(**{
                **item.__dict__,
                "valid_actions": flat_actions,
                "next_valid_actions": () if item.terminated else flat_actions,
                "teacher_target": np.full(22, 0.1, np.float32),
                "entry_action_target": Action.WAIT,
                "regime_selectivity_headroom_fraction": 0.1 + index * 0.1,
            })
            for index, item in enumerate(episode.transitions)
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=23,
    )

    replay.add(taught)
    restored = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=23,
    )
    restored.load_state_dict(replay.state_dict())
    sampled = restored.sample(1)[0]

    assert tuple(
        item.regime_selectivity_headroom_fraction for item in sampled
    ) == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))


def test_replay_rejects_stale_v3_checkpoint_schema() -> None:
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=29,
    )
    replay.add(_episode("NQ", "pass", "long", 0))
    state = replay.state_dict()
    state["schema_version"] = 3

    restored = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=6,
        seed=29,
    )
    with pytest.raises(ValueError, match="schema is unsupported"):
        restored.load_state_dict(state)


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


def test_replay_preserves_short_recovery_trace_with_explicit_invalid_padding() -> None:
    original = _episode("NQ", "recovery_success", "long", 0)
    short = Episode(
        episode_id="NQ-recovery-short",
        ticker="NQ",
        outcome="recovery_success",
        primary_side="long",
        ended_at_ns=2,
        transitions=original.transitions[:1] + (
            Transition(**{
                **original.transitions[1].__dict__,
                "terminated": True,
                "next_valid_actions": (),
            }),
        ),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=96,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=53,
    )

    replay.add(short)
    sequence = replay.sample(1)[0]

    assert len(sequence) == 96
    assert [index for index, row in enumerate(sequence) if row.training_valid] == [64, 65]
    assert sequence[64].recurrent_reset is True
    assert sequence[63].next_recurrent_reset is True
    assert [row.reward for row in sequence if row.training_valid] == [0.0, 1.0]
    assert sum(row.terminated for row in sequence if row.training_valid) == 1
    assert all(not row.competence_anchor for row in sequence if not row.training_valid)


def test_short_recovery_replay_checkpoint_round_trip_is_exact_and_versioned() -> None:
    original = _episode("NQ", "recovery_success", "long", 0)
    short = Episode(
        episode_id="NQ-recovery-short",
        ticker="NQ",
        outcome="recovery_success",
        primary_side="long",
        ended_at_ns=2,
        transitions=original.transitions[:2],
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=96,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=59,
    )
    replay.add(short)
    state = replay.state_dict()
    restored = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=96,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=999,
    )

    assert state["schema_version"] == 7
    restored.load_state_dict(state)
    expected = replay.sample(1)[0]
    actual = restored.sample(1)[0]
    assert [row.training_valid for row in actual] == [
        row.training_valid for row in expected
    ]
    assert [row.action for row in actual] == [row.action for row in expected]
    np.testing.assert_array_equal(
        [row.observation for row in actual],
        [row.observation for row in expected],
    )

    state["schema_version"] = 5
    with pytest.raises(ValueError, match="schema is unsupported"):
        restored.load_state_dict(state)


def test_short_recovery_entry_and_terminal_strata_keep_both_boundaries_learnable() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    management_actions = (Action.HOLD, Action.CLOSE)
    transitions = []
    for index in range(71):
        entered = index == 5
        terminal = index == 70
        transitions.append(Transition(
            observation=np.array([float(index)], np.float32),
            action=(
                Action.ENTER_LONG_1 if entered
                else Action.CLOSE if terminal
                else Action.WAIT if index < 5
                else Action.HOLD
            ),
            reward=1.0 if terminal else 0.0,
            next_observation=np.array([float(index + 1)], np.float32),
            terminated=terminal,
            valid_actions=flat_actions if index <= 5 else management_actions,
            next_valid_actions=(
                () if terminal
                else management_actions if index >= 5
                else flat_actions
            ),
            entry_action_target=Action.ENTER_LONG_1 if entered else None,
            entry_opportunity_priority=1.0 if entered else 0.0,
        ))
    episode = Episode(
        episode_id="NQ-recovery-71",
        ticker="NQ",
        outcome="recovery_success",
        primary_side="long",
        ended_at_ns=71,
        transitions=tuple(transitions),
    )
    replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=96,
        terminal_sequence_fraction=0.5,
        entry_opportunity_sequence_fraction=0.5,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=61,
    )
    replay.add(episode)

    sampled = replay.sample(2)
    entry_sequence = next(
        sequence
        for sequence in sampled
        if any(
            index >= 64
            and row.entry_action_target == Action.ENTER_LONG_1
            for index, row in enumerate(sequence)
        )
    )
    terminal_sequence = next(
        sequence for sequence in sampled if any(row.terminated for row in sequence)
    )
    entry_index = next(
        index
        for index, row in enumerate(entry_sequence)
        if row.entry_action_target == Action.ENTER_LONG_1
    )
    terminal_index = next(
        index for index, row in enumerate(terminal_sequence) if row.terminated
    )

    assert entry_index == 64
    learning_indices = range(64, 96 - 8 + 1)
    assert terminal_index == 88
    assert entry_index in learning_indices
    assert terminal_index in learning_indices
    assert entry_sequence[entry_index].training_valid is True
    assert terminal_sequence[terminal_index].training_valid is True
    first_real_index = entry_index - 5
    assert entry_sequence[first_real_index].recurrent_reset is True
    assert entry_sequence[first_real_index - 1].next_recurrent_reset is True
    np.testing.assert_array_equal(
        entry_sequence[first_real_index - 1].next_observation,
        entry_sequence[first_real_index].observation,
    )
    assert all(
        not row.terminated
        for row in entry_sequence
        if not row.training_valid
    )
    assert all(
        not row.terminated
        for row in terminal_sequence
        if not row.training_valid
    )

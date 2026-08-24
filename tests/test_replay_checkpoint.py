from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.replay import (
    BalancedSequenceReplay,
    Episode,
    ReplayCheckpointStore,
    Transition,
)


def _episode(offset: int) -> Episode:
    return Episode(
        episode_id=f"NQ-timeout-{offset}",
        ticker="NQ",
        outcome="timeout",
        primary_side="long",
        ended_at_ns=offset,
        transitions=tuple(
            Transition(
                observation=np.array([offset + index], np.float32),
                action=Action.WAIT,
                reward=float(index),
                next_observation=np.array([offset + index + 1], np.float32),
                terminated=index == 5,
                valid_actions=(Action.WAIT,),
                next_valid_actions=() if index == 5 else (Action.WAIT,),
            )
            for index in range(6)
        ),
    )


def _replay(*, seed: int, capacity: int = 3) -> BalancedSequenceReplay:
    return BalancedSequenceReplay(
        capacity_episodes=capacity,
        sequence_length=3,
        seed=seed,
    )


def _sample_values(replay: BalancedSequenceReplay) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(item.observation[0]) for item in sequence)
        for sequence in replay.sample(2)
    )


def test_replay_checkpoint_writes_each_episode_once_and_restores_exactly(
    tmp_path: Path,
) -> None:
    replay = _replay(seed=17)
    replay.add(_episode(0))
    replay.add(_episode(100))
    store = ReplayCheckpointStore(tmp_path / "training-replay")

    first = store.persist(replay)
    shard_paths = tuple(
        store.root / item["path"] for item in first["episodes"]
    )
    first_mtimes = tuple(path.stat().st_mtime_ns for path in shard_paths)

    second = store.persist(replay)

    assert second == first
    assert tuple(path.stat().st_mtime_ns for path in shard_paths) == first_mtimes
    restored = _replay(seed=999)
    ReplayCheckpointStore(store.root).restore(restored, second)
    assert _sample_values(restored) == _sample_values(replay)


def test_replay_checkpoint_appends_new_episode_and_prunes_only_after_commit(
    tmp_path: Path,
) -> None:
    replay = _replay(seed=23, capacity=2)
    store = ReplayCheckpointStore(tmp_path / "training-replay")
    replay.add(_episode(0))
    replay.add(_episode(100))
    original = store.persist(replay)
    original_paths = {
        item["episode_id"]: store.root / item["path"]
        for item in original["episodes"]
    }

    replay.add(_episode(200))
    current = store.persist(replay)

    assert original_paths["NQ-timeout-0"].is_file()
    assert original_paths["NQ-timeout-100"].is_file()
    store.prune(current)
    assert not original_paths["NQ-timeout-0"].exists()
    assert original_paths["NQ-timeout-100"].is_file()
    assert len(tuple((store.root / "episodes").glob("*.pkl"))) == 2


def test_replay_checkpoint_fails_closed_on_tampered_shard(tmp_path: Path) -> None:
    replay = _replay(seed=29)
    replay.add(_episode(0))
    store = ReplayCheckpointStore(tmp_path / "training-replay")
    descriptor = store.persist(replay)
    shard = store.root / descriptor["episodes"][0]["path"]
    shard.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="replay checkpoint shard identity"):
        ReplayCheckpointStore(store.root).restore(
            _replay(seed=29), descriptor
        )

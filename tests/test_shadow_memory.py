from __future__ import annotations

from pathlib import Path

import numpy as np

from propevolve.decision import Action
from propevolve.replay import Episode, Transition
from propevolve.shadow import ShadowEpisodeStore


def test_shadow_episode_round_trip_preserves_causal_training_record(tmp_path: Path) -> None:
    transition = Transition(
        observation=np.array([1.0, 2.0], np.float32),
        action=Action.ENTER_LONG_1,
        reward=0.25,
        next_observation=np.array([2.0, 3.0], np.float32),
        terminated=True,
        valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        next_valid_actions=(),
    )
    episode = Episode(
        episode_id="paper-NQ-1",
        ticker="NQ",
        outcome="pass",
        primary_side="long",
        ended_at_ns=123,
        transitions=(transition,),
    )
    store = ShadowEpisodeStore(tmp_path)

    receipt = store.append(episode, policy_sha256="champion", checkpoint_sha256="mask")
    restored = store.load(receipt)

    assert receipt["schema"] == "propevolve_shadow_episode_v1"
    assert restored.episode_id == episode.episode_id
    assert restored.ticker == "NQ"
    assert restored.transitions[0].action == Action.ENTER_LONG_1
    np.testing.assert_array_equal(restored.transitions[0].observation, transition.observation)


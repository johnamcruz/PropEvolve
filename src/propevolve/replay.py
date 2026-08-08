"""Bounded balanced replay of causal recurrent trading sequences."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import random

import numpy as np

from .decision import Action


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: Action
    reward: float
    next_observation: np.ndarray
    terminated: bool
    valid_actions: tuple[Action, ...]
    next_valid_actions: tuple[Action, ...]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    ticker: str
    outcome: str
    primary_side: str
    ended_at_ns: int
    transitions: tuple[Transition, ...]

    @property
    def bucket(self) -> tuple[str, str, str]:
        return self.ticker, self.outcome, self.primary_side


class BalancedSequenceReplay:
    """Retain recent episodes while sampling scarce outcome buckets fairly."""

    def __init__(
        self,
        capacity_episodes: int,
        sequence_length: int,
        *,
        seed: int = 0,
    ) -> None:
        if capacity_episodes < 1 or sequence_length < 1:
            raise ValueError("replay capacity and sequence length must be positive")
        self.capacity = int(capacity_episodes)
        self.sequence_length = int(sequence_length)
        self._episodes: OrderedDict[str, Episode] = OrderedDict()
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        if len(episode.transitions) < self.sequence_length:
            raise ValueError("episode is shorter than the replay sequence length")
        self._episodes.pop(episode.episode_id, None)
        self._episodes[episode.episode_id] = episode
        while len(self._episodes) > self.capacity:
            self._episodes.popitem(last=False)

    def sample_episodes(self, count: int) -> tuple[Episode, ...]:
        if count < 1 or not self._episodes:
            raise ValueError("cannot sample an empty or nonpositive replay batch")
        buckets: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
        for episode in self._episodes.values():
            buckets[episode.bucket].append(episode)
        keys = list(buckets)
        self._random.shuffle(keys)
        selected: list[Episode] = []
        while len(selected) < count:
            for key in keys:
                selected.append(self._random.choice(buckets[key]))
                if len(selected) == count:
                    break
        return tuple(selected)

    def sample(self, count: int) -> tuple[tuple[Transition, ...], ...]:
        sequences = []
        for episode in self.sample_episodes(count):
            last_start = len(episode.transitions) - self.sequence_length
            start = self._random.randint(0, last_start)
            sequences.append(episode.transitions[start:start + self.sequence_length])
        return tuple(sequences)


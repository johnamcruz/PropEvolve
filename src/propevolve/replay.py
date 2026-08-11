"""Bounded balanced replay of causal recurrent trading sequences."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import random
from typing import Mapping

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
    teacher_target: np.ndarray | None = None
    safety_priority: float = 0.0
    entry_opportunity_priority: float = 0.0


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


@dataclass(frozen=True)
class _StoredEpisode:
    episode_id: str
    ticker: str
    outcome: str
    primary_side: str
    ended_at_ns: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    valid_masks: np.ndarray
    next_valid_masks: np.ndarray
    teacher_targets: np.ndarray | None
    safety_priorities: np.ndarray
    entry_opportunity_priorities: np.ndarray

    @property
    def bucket(self) -> tuple[str, str, str]:
        return self.ticker, self.outcome, self.primary_side

    @property
    def transition_count(self) -> int:
        return len(self.actions)

    @classmethod
    def from_episode(cls, episode: Episode) -> "_StoredEpisode":
        transitions = episode.transitions
        if any(
            not np.isfinite(item.safety_priority) or item.safety_priority < 0
            for item in transitions
        ):
            raise ValueError("replay safety priority is invalid")
        if any(
            not np.isfinite(item.entry_opportunity_priority)
            or item.entry_opportunity_priority < 0
            for item in transitions
        ):
            raise ValueError("replay entry opportunity priority is invalid")
        for current, following in zip(transitions, transitions[1:]):
            if not np.array_equal(current.next_observation, following.observation):
                raise ValueError("replay episode observations are not contiguous")
        observations = np.stack([
            *(item.observation for item in transitions),
            transitions[-1].next_observation,
        ]).astype(np.float32, copy=False)
        teacher_widths = {
            int(np.asarray(item.teacher_target).size)
            for item in transitions
            if item.teacher_target is not None
        }
        if len(teacher_widths) > 1 or 0 in teacher_widths:
            raise ValueError("replay teacher targets have inconsistent widths")
        teacher_targets = None
        if teacher_widths:
            width = next(iter(teacher_widths))
            teacher_targets = np.full((len(transitions), width), np.nan, np.float32)
            for index, item in enumerate(transitions):
                if item.teacher_target is not None:
                    target = np.asarray(item.teacher_target, dtype=np.float32).reshape(-1)
                    if target.size != width or not np.isfinite(target).all():
                        raise ValueError("replay teacher target is invalid")
                    teacher_targets[index] = target
        return cls(
            episode_id=episode.episode_id,
            ticker=episode.ticker,
            outcome=episode.outcome,
            primary_side=episode.primary_side,
            ended_at_ns=episode.ended_at_ns,
            observations=observations,
            actions=np.asarray([int(item.action) for item in transitions], np.int8),
            rewards=np.asarray([item.reward for item in transitions], np.float32),
            terminated=np.asarray([item.terminated for item in transitions], np.bool_),
            valid_masks=np.asarray([
                [action in item.valid_actions for action in Action]
                for item in transitions
            ], np.bool_),
            next_valid_masks=np.asarray([
                [action in item.next_valid_actions for action in Action]
                for item in transitions
            ], np.bool_),
            teacher_targets=teacher_targets,
            safety_priorities=np.asarray(
                [item.safety_priority for item in transitions], np.float32
            ),
            entry_opportunity_priorities=np.asarray(
                [item.entry_opportunity_priority for item in transitions], np.float32
            ),
        )

    def sequence(self, start: int, length: int) -> tuple[Transition, ...]:
        stop = start + length
        return tuple(
            Transition(
                observation=self.observations[index],
                action=Action(int(self.actions[index])),
                reward=float(self.rewards[index]),
                next_observation=self.observations[index + 1],
                terminated=bool(self.terminated[index]),
                valid_actions=tuple(
                    action for action in Action if self.valid_masks[index, int(action)]
                ),
                next_valid_actions=tuple(
                    action
                    for action in Action
                    if self.next_valid_masks[index, int(action)]
                ),
                teacher_target=(
                    None
                    if self.teacher_targets is None
                    or not np.isfinite(self.teacher_targets[index]).all()
                    else self.teacher_targets[index]
                ),
                safety_priority=float(self.safety_priorities[index]),
                entry_opportunity_priority=float(
                    self.entry_opportunity_priorities[index]
                ),
            )
            for index in range(start, stop)
        )


class BalancedSequenceReplay:
    """Retain recent episodes while sampling scarce outcome buckets fairly."""

    def __init__(
        self,
        capacity_episodes: int,
        sequence_length: int,
        *,
        capacity_transitions: int | None = None,
        terminal_sequence_fraction: float = 0.0,
        safety_sequence_fraction: float = 0.0,
        entry_opportunity_sequence_fraction: float = 0.0,
        seed: int,
    ) -> None:
        if capacity_episodes < 1 or sequence_length < 1:
            raise ValueError("replay capacity and sequence length must be positive")
        if not 0.0 <= terminal_sequence_fraction <= 1.0:
            raise ValueError("terminal sequence fraction must be between zero and one")
        if (
            not 0.0 <= safety_sequence_fraction <= 1.0
            or not 0.0 <= entry_opportunity_sequence_fraction <= 1.0
            or terminal_sequence_fraction + safety_sequence_fraction
            + entry_opportunity_sequence_fraction > 1.0
        ):
            raise ValueError("replay sequence fractions are invalid")
        self.capacity = int(capacity_episodes)
        if capacity_transitions is not None and capacity_transitions < sequence_length:
            raise ValueError("replay transition capacity is smaller than one sequence")
        self.capacity_transitions = (
            None if capacity_transitions is None else int(capacity_transitions)
        )
        self.sequence_length = int(sequence_length)
        self.terminal_sequence_fraction = float(terminal_sequence_fraction)
        self.safety_sequence_fraction = float(safety_sequence_fraction)
        self.entry_opportunity_sequence_fraction = float(
            entry_opportunity_sequence_fraction
        )
        self._episodes: OrderedDict[str, _StoredEpisode] = OrderedDict()
        self._transition_count = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def state_dict(self) -> dict[str, object]:
        """Return the complete resumable replay state, including sampler RNG."""
        return {
            "schema_version": 1,
            "contract": {
                "capacity_episodes": self.capacity,
                "capacity_transitions": self.capacity_transitions,
                "sequence_length": self.sequence_length,
                "terminal_sequence_fraction": self.terminal_sequence_fraction,
                "safety_sequence_fraction": self.safety_sequence_fraction,
                "entry_opportunity_sequence_fraction": (
                    self.entry_opportunity_sequence_fraction
                ),
            },
            "random_state": self._random.getstate(),
            "episodes": [
                {
                    field: getattr(episode, field)
                    for field in _StoredEpisode.__dataclass_fields__
                }
                for episode in self._episodes.values()
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore replay exactly and fail closed if its sampling contract drifted."""
        if state.get("schema_version") != 1:
            raise ValueError("replay checkpoint schema is unsupported")
        expected_contract = {
            "capacity_episodes": self.capacity,
            "capacity_transitions": self.capacity_transitions,
            "sequence_length": self.sequence_length,
            "terminal_sequence_fraction": self.terminal_sequence_fraction,
            "safety_sequence_fraction": self.safety_sequence_fraction,
            "entry_opportunity_sequence_fraction": (
                self.entry_opportunity_sequence_fraction
            ),
        }
        if state.get("contract") != expected_contract:
            raise ValueError("replay checkpoint contract drifted")
        payloads = state.get("episodes")
        if not isinstance(payloads, list):
            raise ValueError("replay checkpoint episodes are invalid")
        restored: OrderedDict[str, _StoredEpisode] = OrderedDict()
        transition_count = 0
        action_count = len(Action)
        for payload in payloads:
            if not isinstance(payload, Mapping):
                raise ValueError("replay checkpoint episode is invalid")
            try:
                observations = np.asarray(payload["observations"], dtype=np.float32)
                actions = np.asarray(payload["actions"], dtype=np.int8)
                rewards = np.asarray(payload["rewards"], dtype=np.float32)
                terminated = np.asarray(payload["terminated"], dtype=np.bool_)
                valid_masks = np.asarray(payload["valid_masks"], dtype=np.bool_)
                next_valid_masks = np.asarray(
                    payload["next_valid_masks"], dtype=np.bool_
                )
                safety_priorities = np.asarray(
                    payload["safety_priorities"], dtype=np.float32
                )
                entry_priorities = np.asarray(
                    payload["entry_opportunity_priorities"], dtype=np.float32
                )
                raw_teacher_targets = payload["teacher_targets"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("replay checkpoint episode is malformed") from error
            count = int(actions.size)
            teacher_targets = (
                None
                if raw_teacher_targets is None
                else np.asarray(raw_teacher_targets, dtype=np.float32)
            )
            if (
                count < self.sequence_length
                or actions.shape != (count,)
                or rewards.shape != (count,)
                or terminated.shape != (count,)
                or observations.ndim != 2
                or observations.shape[0] != count + 1
                or valid_masks.shape != (count, action_count)
                or next_valid_masks.shape != (count, action_count)
                or safety_priorities.shape != (count,)
                or entry_priorities.shape != (count,)
                or (teacher_targets is not None and teacher_targets.shape[0] != count)
                or not np.isfinite(observations).all()
                or not np.isfinite(rewards).all()
                or not np.isfinite(safety_priorities).all()
                or not np.isfinite(entry_priorities).all()
                or (safety_priorities < 0).any()
                or (entry_priorities < 0).any()
                or (actions < 0).any()
                or (actions >= action_count).any()
                or not valid_masks.any(axis=1).all()
                or not np.array_equal(
                    next_valid_masks.any(axis=1),
                    ~terminated,
                )
            ):
                raise ValueError("replay checkpoint episode arrays are invalid")
            if teacher_targets is not None:
                if teacher_targets.ndim != 2 or teacher_targets.shape[1] < 1:
                    raise ValueError("replay checkpoint teacher targets are invalid")
                valid_teacher_rows = np.isfinite(teacher_targets).all(axis=1)
                missing_teacher_rows = np.isnan(teacher_targets).all(axis=1)
                if not np.logical_or(valid_teacher_rows, missing_teacher_rows).all():
                    raise ValueError("replay checkpoint teacher targets are invalid")
            episode_id = str(payload.get("episode_id", ""))
            if not episode_id or episode_id in restored:
                raise ValueError("replay checkpoint episode identity is invalid")
            episode = _StoredEpisode(
                episode_id=episode_id,
                ticker=str(payload.get("ticker", "")),
                outcome=str(payload.get("outcome", "")),
                primary_side=str(payload.get("primary_side", "")),
                ended_at_ns=int(payload.get("ended_at_ns", 0)),
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                valid_masks=valid_masks,
                next_valid_masks=next_valid_masks,
                teacher_targets=teacher_targets,
                safety_priorities=safety_priorities,
                entry_opportunity_priorities=entry_priorities,
            )
            if not episode.ticker or not episode.outcome or not episode.primary_side:
                raise ValueError("replay checkpoint episode metadata is invalid")
            restored[episode_id] = episode
            transition_count += count
        if len(restored) > self.capacity or (
            self.capacity_transitions is not None
            and transition_count > self.capacity_transitions
        ):
            raise ValueError("replay checkpoint exceeds its declared capacity")
        try:
            self._random.setstate(state["random_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay checkpoint RNG state is invalid") from error
        self._episodes = restored
        self._transition_count = transition_count

    def add(self, episode: Episode) -> None:
        if len(episode.transitions) < self.sequence_length:
            raise ValueError("episode is shorter than the replay sequence length")
        stored = _StoredEpisode.from_episode(episode)
        replaced = self._episodes.pop(episode.episode_id, None)
        if replaced is not None:
            self._transition_count -= replaced.transition_count
        self._episodes[episode.episode_id] = stored
        self._transition_count += stored.transition_count
        while len(self._episodes) > 1 and (
            len(self._episodes) > self.capacity
            or (
                self.capacity_transitions is not None
                and self._transition_count > self.capacity_transitions
            )
        ):
            outcome_side_counts: dict[tuple[str, str], int] = defaultdict(int)
            for retained in self._episodes.values():
                outcome_side_counts[(retained.outcome, retained.primary_side)] += 1
            largest = max(outcome_side_counts.values())
            removable_groups = {
                key for key, count in outcome_side_counts.items() if count == largest
            }
            remove_id = next(
                episode_id
                for episode_id, retained in self._episodes.items()
                if (retained.outcome, retained.primary_side) in removable_groups
            )
            removed = self._episodes.pop(remove_id)
            self._transition_count -= removed.transition_count

    def sample_episodes(self, count: int) -> tuple[_StoredEpisode, ...]:
        if count < 1 or not self._episodes:
            raise ValueError("cannot sample an empty or nonpositive replay batch")
        buckets: dict[tuple[str, str, str], list[_StoredEpisode]] = defaultdict(list)
        for episode in self._episodes.values():
            buckets[episode.bucket].append(episode)
        keys = list(buckets)
        self._random.shuffle(keys)
        selected: list[_StoredEpisode] = []
        while len(selected) < count:
            for key in keys:
                selected.append(self._random.choice(buckets[key]))
                if len(selected) == count:
                    break
        return tuple(selected)

    def sample(self, count: int) -> tuple[tuple[Transition, ...], ...]:
        sequences = []
        terminal_count = round(count * self.terminal_sequence_fraction)
        safety_count = round(count * self.safety_sequence_fraction)
        entry_count = round(count * self.entry_opportunity_sequence_fraction)
        for index, episode in enumerate(self.sample_episodes(count)):
            last_start = episode.transition_count - self.sequence_length
            if index < terminal_count:
                start = last_start
            elif index < terminal_count + safety_count:
                critical = int(np.argmax(episode.safety_priorities))
                start = max(0, min(last_start, critical - self.sequence_length + 1))
            elif index < terminal_count + safety_count + entry_count:
                critical = int(np.argmax(episode.entry_opportunity_priorities))
                start = max(0, min(last_start, critical - self.sequence_length + 1))
            else:
                start = self._random.randint(0, last_start)
            sequences.append(episode.sequence(start, self.sequence_length))
        return tuple(sequences)

"""Bounded balanced replay of causal recurrent trading sequences."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
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
    recurrent_reset: bool = False
    next_recurrent_reset: bool = False
    competence_anchor: bool = False
    teacher_target: np.ndarray | None = None
    entry_action_target: Action | None = None
    regime_selectivity_headroom_fraction: float | None = None
    safety_priority: float = 0.0
    entry_opportunity_priority: float = 0.0
    training_valid: bool = True
    source_decision_index: int | None = None


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
class FinalRegimeProbeSequence:
    """One immutable authentic replay row plus its recurrent context."""

    episode_id: str
    ticker: str
    anchor_index: int
    sequence_anchor_index: int
    source_decision_index: int
    target_action: Action
    row_identity_sha256: str
    sequence: tuple[Transition, ...]


def final_regime_probe_row_identity(
    *,
    ticker: str,
    source_decision_index: int,
    target_action: Action,
    observation: np.ndarray,
    teacher_target: np.ndarray,
) -> str:
    """Content-address one stable market row without wall-clock identity."""
    if (
        not ticker
        or isinstance(source_decision_index, bool)
        or int(source_decision_index) < 0
    ):
        raise ValueError("final Regime probe source identity is invalid")
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    teacher_target = np.asarray(teacher_target, dtype=np.float32).reshape(-1)
    if (
        observation.size < 1
        or teacher_target.size < 1
        or not np.isfinite(observation).all()
        or not np.isfinite(teacher_target).all()
    ):
        raise ValueError("final Regime probe source row is invalid")
    hasher = hashlib.sha256()
    hasher.update(b"propevolve-final-regime-probe-row-v1\0")
    hasher.update(ticker.encode("utf-8"))
    hasher.update(int(source_decision_index).to_bytes(8, "big", signed=False))
    hasher.update(int(Action(target_action)).to_bytes(2, "big", signed=False))
    hasher.update(observation.tobytes())
    hasher.update(teacher_target.tobytes())
    return hasher.hexdigest()


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
    recurrent_resets: np.ndarray
    next_recurrent_resets: np.ndarray
    teacher_targets: np.ndarray | None
    entry_action_targets: np.ndarray
    entry_long_anchor_indices: np.ndarray
    entry_short_anchor_indices: np.ndarray
    regime_selectivity_headroom_fractions: np.ndarray
    safety_priorities: np.ndarray
    entry_opportunity_priorities: np.ndarray
    source_decision_indices: np.ndarray

    @property
    def bucket(self) -> tuple[str, str, str]:
        return self.ticker, self.outcome, self.primary_side

    @property
    def transition_count(self) -> int:
        return len(self.actions)

    @classmethod
    def from_episode(cls, episode: Episode) -> "_StoredEpisode":
        transitions = episode.transitions
        if not transitions or not all(item.training_valid is True for item in transitions):
            raise ValueError("replay episodes must contain authentic transitions only")
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
        if any(
            item.source_decision_index is not None
            and (
                isinstance(item.source_decision_index, bool)
                or int(item.source_decision_index) < 0
            )
            for item in transitions
        ):
            raise ValueError("replay source decision index is invalid")
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
        entry_action_targets = np.full(len(transitions), -1, dtype=np.int8)
        regime_selectivity_headroom_fractions = np.full(
            len(transitions), np.nan, dtype=np.float32
        )
        flat_actions = {
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
        for index, item in enumerate(transitions):
            if item.entry_action_target is None:
                continue
            try:
                target = Action(item.entry_action_target)
            except (TypeError, ValueError) as error:
                raise ValueError("replay entry target is invalid") from error
            if target not in flat_actions or not flat_actions.issubset(
                item.valid_actions
            ):
                raise ValueError("replay entry target is invalid")
            entry_action_targets[index] = int(target)
        for index, item in enumerate(transitions):
            if item.regime_selectivity_headroom_fraction is None:
                continue
            headroom = float(item.regime_selectivity_headroom_fraction)
            if not np.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
                raise ValueError("replay Regime-selectivity headroom is invalid")
            if (
                item.teacher_target is None
                or item.entry_action_target is None
                or not flat_actions.issubset(item.valid_actions)
            ):
                raise ValueError("replay Regime-selectivity row is invalid")
            regime_selectivity_headroom_fractions[index] = headroom
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
            recurrent_resets=np.asarray(
                [item.recurrent_reset for item in transitions], np.bool_
            ),
            next_recurrent_resets=np.asarray(
                [item.next_recurrent_reset for item in transitions], np.bool_
            ),
            teacher_targets=teacher_targets,
            entry_action_targets=entry_action_targets,
            entry_long_anchor_indices=np.flatnonzero(
                entry_action_targets == int(Action.ENTER_LONG_1)
            ).astype(np.int32, copy=False),
            entry_short_anchor_indices=np.flatnonzero(
                entry_action_targets == int(Action.ENTER_SHORT_1)
            ).astype(np.int32, copy=False),
            regime_selectivity_headroom_fractions=(
                regime_selectivity_headroom_fractions
            ),
            safety_priorities=np.asarray(
                [item.safety_priority for item in transitions], np.float32
            ),
            entry_opportunity_priorities=np.asarray(
                [item.entry_opportunity_priority for item in transitions], np.float32
            ),
            source_decision_indices=np.asarray(
                [
                    -1
                    if item.source_decision_index is None
                    else int(item.source_decision_index)
                    for item in transitions
                ],
                np.int64,
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
                recurrent_reset=bool(self.recurrent_resets[index]),
                next_recurrent_reset=bool(self.next_recurrent_resets[index]),
                competence_anchor=self.outcome in {"pass", "recovery_success"},
                teacher_target=(
                    None
                    if self.teacher_targets is None
                    or not np.isfinite(self.teacher_targets[index]).all()
                    else self.teacher_targets[index]
                ),
                entry_action_target=(
                    None
                    if self.entry_action_targets[index] < 0
                    else Action(int(self.entry_action_targets[index]))
                ),
                regime_selectivity_headroom_fraction=(
                    None
                    if not np.isfinite(
                        self.regime_selectivity_headroom_fractions[index]
                    )
                    else float(
                        self.regime_selectivity_headroom_fractions[index]
                    )
                ),
                safety_priority=float(self.safety_priorities[index]),
                entry_opportunity_priority=float(
                    self.entry_opportunity_priorities[index]
                ),
                training_valid=True,
                source_decision_index=(
                    None
                    if self.source_decision_indices[index] < 0
                    else int(self.source_decision_indices[index])
                ),
            )
            for index in range(start, stop)
        )

    @staticmethod
    def _padding_transition(
        observation: np.ndarray,
        *,
        next_recurrent_reset: bool = False,
    ) -> Transition:
        value = np.zeros_like(observation, dtype=np.float32)
        next_value = (
            np.asarray(observation, dtype=np.float32).copy()
            if next_recurrent_reset
            else value.copy()
        )
        return Transition(
            observation=value,
            action=Action.WAIT,
            reward=0.0,
            next_observation=next_value,
            terminated=False,
            valid_actions=(Action.WAIT,),
            next_valid_actions=(Action.WAIT,),
            recurrent_reset=True,
            next_recurrent_reset=next_recurrent_reset,
            competence_anchor=False,
            training_valid=False,
            source_decision_index=None,
        )

    def padded_sequence(
        self,
        *,
        length: int,
        recurrent_burn_in: int,
        n_step_return: int = 1,
        anchor: str = "terminal",
    ) -> tuple[Transition, ...]:
        if self.transition_count >= length:
            raise ValueError("only short replay episodes may be padded")
        last_learning_index = length - n_step_return
        if not recurrent_burn_in <= last_learning_index < length:
            raise ValueError("padded replay learning contract is invalid")
        if anchor == "terminal":
            maximum_real_count = last_learning_index + 1
            source_start = max(0, self.transition_count - maximum_real_count)
            source_stop = self.transition_count
            selected_count = source_stop - source_start
            real_start = min(
                recurrent_burn_in,
                last_learning_index - selected_count + 1,
            )
        elif anchor in {"safety", "entry"}:
            priorities = (
                self.safety_priorities
                if anchor == "safety"
                else self.entry_opportunity_priorities
            )
            anchor_index = int(np.argmax(priorities))
            source_start = max(0, anchor_index - recurrent_burn_in)
            context_before_anchor = anchor_index - source_start
            real_start = recurrent_burn_in - context_before_anchor
            source_stop = min(
                self.transition_count,
                source_start + length - real_start,
            )
        else:
            raise ValueError("padded replay anchor is invalid")
        if real_start < 0 or source_start >= source_stop:
            raise ValueError("padded replay cannot place its authentic anchor")
        real = list(self.sequence(source_start, source_stop - source_start))
        pre = [
            self._padding_transition(
                real[0].observation,
                next_recurrent_reset=index == real_start - 1,
            )
            for index in range(real_start)
        ]
        real[0] = Transition(**{**real[0].__dict__, "recurrent_reset": True})
        post_count = length - real_start - len(real)
        post = [
            self._padding_transition(real[-1].next_observation)
            for _ in range(post_count)
        ]
        return tuple((*pre, *real, *post))

    def entry_anchored_sequence(
        self,
        *,
        anchor_index: int,
        length: int,
        recurrent_burn_in: int,
        n_step_return: int,
    ) -> tuple[Transition, ...]:
        """Place an authentic entry target at the first learnable timestep."""
        last_learning_index = length - n_step_return
        if (
            not 0 <= anchor_index < self.transition_count
            or not recurrent_burn_in <= last_learning_index < length
            or self.entry_action_targets[anchor_index]
            not in {int(Action.ENTER_LONG_1), int(Action.ENTER_SHORT_1)}
        ):
            raise ValueError("entry replay anchor is invalid")
        source_start = max(0, anchor_index - recurrent_burn_in)
        context_before_anchor = anchor_index - source_start
        real_start = recurrent_burn_in - context_before_anchor
        source_stop = min(
            self.transition_count,
            source_start + length - real_start,
        )
        real = list(self.sequence(source_start, source_stop - source_start))
        pre = [
            self._padding_transition(
                real[0].observation,
                next_recurrent_reset=index == real_start - 1,
            )
            for index in range(real_start)
        ]
        if pre:
            real[0] = Transition(**{**real[0].__dict__, "recurrent_reset": True})
        post = [
            self._padding_transition(real[-1].next_observation)
            for _ in range(length - real_start - len(real))
        ]
        return tuple((*pre, *real, *post))

    def target_anchored_sequence(
        self,
        *,
        anchor_index: int,
        length: int,
        recurrent_burn_in: int,
        n_step_return: int,
    ) -> tuple[Transition, ...]:
        """Place any exact flat-action target at the first learnable row."""
        if (
            not 0 <= anchor_index < self.transition_count
            or self.entry_action_targets[anchor_index]
            not in {
                int(Action.WAIT),
                int(Action.ENTER_LONG_1),
                int(Action.ENTER_SHORT_1),
            }
        ):
            raise ValueError("Regime probe anchor is invalid")
        target = int(self.entry_action_targets[anchor_index])
        if target in {int(Action.ENTER_LONG_1), int(Action.ENTER_SHORT_1)}:
            return self.entry_anchored_sequence(
                anchor_index=anchor_index,
                length=length,
                recurrent_burn_in=recurrent_burn_in,
                n_step_return=n_step_return,
            )
        last_learning_index = length - n_step_return
        if not recurrent_burn_in <= last_learning_index < length:
            raise ValueError("Regime probe learning contract is invalid")
        source_start = max(0, anchor_index - recurrent_burn_in)
        context_before_anchor = anchor_index - source_start
        real_start = recurrent_burn_in - context_before_anchor
        source_stop = min(
            self.transition_count,
            source_start + length - real_start,
        )
        real = list(self.sequence(source_start, source_stop - source_start))
        pre = [
            self._padding_transition(
                real[0].observation,
                next_recurrent_reset=index == real_start - 1,
            )
            for index in range(real_start)
        ]
        if pre:
            real[0] = Transition(**{**real[0].__dict__, "recurrent_reset": True})
        post = [
            self._padding_transition(real[-1].next_observation)
            for _ in range(length - real_start - len(real))
        ]
        return tuple((*pre, *real, *post))


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
        entry_opportunity_side_balance: str = "none",
        recurrent_burn_in: int = 0,
        n_step_return: int = 1,
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
        if (
            isinstance(recurrent_burn_in, bool)
            or int(recurrent_burn_in) < 0
            or isinstance(n_step_return, bool)
            or int(n_step_return) < 1
            or int(recurrent_burn_in) >= self.sequence_length
        ):
            raise ValueError("replay recurrent learning contract is invalid")
        self.recurrent_burn_in = int(recurrent_burn_in)
        self.n_step_return = int(n_step_return)
        self.terminal_sequence_fraction = float(terminal_sequence_fraction)
        self.safety_sequence_fraction = float(safety_sequence_fraction)
        self.entry_opportunity_sequence_fraction = float(
            entry_opportunity_sequence_fraction
        )
        if entry_opportunity_side_balance not in {
            "none",
            "equal_long_short_v1",
        }:
            raise ValueError("replay entry opportunity side balance is invalid")
        self.entry_opportunity_side_balance = entry_opportunity_side_balance
        self._episodes: OrderedDict[str, _StoredEpisode] = OrderedDict()
        self._transition_count = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def final_regime_probe_sequences(
        self,
        *,
        samples_per_action: int,
    ) -> tuple[FinalRegimeProbeSequence, ...]:
        """Return a fixed class-balanced probe without advancing replay RNG.

        Every anchor is an authentic training row already admitted by replay.
        SHA ranking makes the bounded sample invariant to sampler RNG state and
        stable across repeated evaluation of the same replay contents.
        """
        if (
            isinstance(samples_per_action, bool)
            or not isinstance(samples_per_action, int)
            or samples_per_action < 1
        ):
            raise ValueError("Regime probe sample count must be positive")
        classes = (
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        )
        candidates: dict[
            Action,
            list[tuple[str, _StoredEpisode, int, str]],
        ] = {action: [] for action in classes}
        for episode in self._episodes.values():
            if episode.teacher_targets is None:
                continue
            for anchor_index, raw_target in enumerate(
                episode.entry_action_targets.tolist()
            ):
                if raw_target not in {int(action) for action in classes}:
                    continue
                if (
                    not np.isfinite(episode.teacher_targets[anchor_index]).all()
                    or not np.isfinite(
                        episode.regime_selectivity_headroom_fractions[anchor_index]
                    )
                    or episode.source_decision_indices[anchor_index] < 0
                ):
                    continue
                target = Action(int(raw_target))
                row_identity = final_regime_probe_row_identity(
                    ticker=episode.ticker,
                    source_decision_index=int(
                        episode.source_decision_indices[anchor_index]
                    ),
                    target_action=target,
                    observation=episode.observations[anchor_index],
                    teacher_target=episode.teacher_targets[anchor_index],
                )
                rank = hashlib.sha256(
                    b"propevolve-final-regime-probe-order-v1\0"
                    + row_identity.encode("ascii")
                ).hexdigest()
                candidates[target].append(
                    (rank, episode, anchor_index, row_identity)
                )

        selected: list[FinalRegimeProbeSequence] = []
        for target in classes:
            ranked = []
            seen_identities = set()
            for candidate in sorted(candidates[target], key=lambda item: item[0]):
                row_identity = candidate[3]
                if row_identity not in seen_identities:
                    ranked.append(candidate)
                    seen_identities.add(row_identity)
            if len(ranked) < samples_per_action:
                raise ValueError(
                    "final Regime probe lacks exact balanced authentic rows"
                )
            for _, episode, anchor_index, row_identity in ranked[:samples_per_action]:
                sequence = episode.target_anchored_sequence(
                    anchor_index=anchor_index,
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                )
                anchor = sequence[self.recurrent_burn_in]
                if (
                    not anchor.training_valid
                    or anchor.entry_action_target != target
                    or anchor.teacher_target is None
                ):
                    raise ValueError("Regime probe anchor lost authentic lineage")
                selected.append(FinalRegimeProbeSequence(
                    episode_id=episode.episode_id,
                    ticker=episode.ticker,
                    anchor_index=anchor_index,
                    sequence_anchor_index=self.recurrent_burn_in,
                    source_decision_index=int(
                        episode.source_decision_indices[anchor_index]
                    ),
                    target_action=target,
                    row_identity_sha256=row_identity,
                    sequence=sequence,
                ))
        return tuple(selected)

    def state_dict(self) -> dict[str, object]:
        """Return the complete resumable replay state, including sampler RNG."""
        return {
            "schema_version": 6,
            "contract": {
                "capacity_episodes": self.capacity,
                "capacity_transitions": self.capacity_transitions,
                "sequence_length": self.sequence_length,
                "recurrent_burn_in": self.recurrent_burn_in,
                "n_step_return": self.n_step_return,
                "terminal_sequence_fraction": self.terminal_sequence_fraction,
                "safety_sequence_fraction": self.safety_sequence_fraction,
                "entry_opportunity_sequence_fraction": (
                    self.entry_opportunity_sequence_fraction
                ),
                "entry_opportunity_side_balance": (
                    self.entry_opportunity_side_balance
                ),
            },
            "random_state": self._random.getstate(),
            "episodes": [
                {
                    field: getattr(episode, field)
                    for field in _StoredEpisode.__dataclass_fields__
                    if field not in {
                        "entry_long_anchor_indices",
                        "entry_short_anchor_indices",
                    }
                }
                for episode in self._episodes.values()
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore replay exactly and fail closed if its sampling contract drifted."""
        if state.get("schema_version") != 6:
            raise ValueError("replay checkpoint schema is unsupported")
        expected_contract = {
            "capacity_episodes": self.capacity,
            "capacity_transitions": self.capacity_transitions,
            "sequence_length": self.sequence_length,
            "recurrent_burn_in": self.recurrent_burn_in,
            "n_step_return": self.n_step_return,
            "terminal_sequence_fraction": self.terminal_sequence_fraction,
            "safety_sequence_fraction": self.safety_sequence_fraction,
            "entry_opportunity_sequence_fraction": (
                self.entry_opportunity_sequence_fraction
            ),
            "entry_opportunity_side_balance": (
                self.entry_opportunity_side_balance
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
                recurrent_resets = np.asarray(
                    payload["recurrent_resets"], dtype=np.bool_
                )
                next_recurrent_resets = np.asarray(
                    payload["next_recurrent_resets"], dtype=np.bool_
                )
                safety_priorities = np.asarray(
                    payload["safety_priorities"], dtype=np.float32
                )
                entry_priorities = np.asarray(
                    payload["entry_opportunity_priorities"], dtype=np.float32
                )
                raw_teacher_targets = payload["teacher_targets"]
                raw_entry_action_targets = payload["entry_action_targets"]
                raw_regime_selectivity_headroom = payload[
                    "regime_selectivity_headroom_fractions"
                ]
                source_decision_indices = np.asarray(
                    payload.get(
                        "source_decision_indices",
                        np.full(actions.size, -1, dtype=np.int64),
                    ),
                    dtype=np.int64,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("replay checkpoint episode is malformed") from error
            count = int(actions.size)
            teacher_targets = (
                None
                if raw_teacher_targets is None
                else np.asarray(raw_teacher_targets, dtype=np.float32)
            )
            entry_action_targets = np.asarray(raw_entry_action_targets, dtype=np.int8)
            regime_selectivity_headroom = np.asarray(
                raw_regime_selectivity_headroom, dtype=np.float32
            )
            if (
                count < 1
                or actions.shape != (count,)
                or rewards.shape != (count,)
                or terminated.shape != (count,)
                or observations.ndim != 2
                or observations.shape[0] != count + 1
                or valid_masks.shape != (count, action_count)
                or next_valid_masks.shape != (count, action_count)
                or recurrent_resets.shape != (count,)
                or next_recurrent_resets.shape != (count,)
                or entry_action_targets.shape != (count,)
                or regime_selectivity_headroom.shape != (count,)
                or source_decision_indices.shape != (count,)
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
                or (entry_action_targets < -1).any()
                or (entry_action_targets > int(Action.ENTER_SHORT_1)).any()
                or (source_decision_indices < -1).any()
                or not np.logical_or(
                    np.isnan(regime_selectivity_headroom),
                    (
                        (regime_selectivity_headroom >= 0.0)
                        & (regime_selectivity_headroom <= 1.0)
                    ),
                ).all()
                or not valid_masks[
                    entry_action_targets >= 0,
                    :int(Action.ENTER_SHORT_1) + 1,
                ].all()
                or not valid_masks.any(axis=1).all()
                or not np.array_equal(
                    next_valid_masks.any(axis=1),
                    ~terminated,
                )
                or not np.array_equal(
                    next_recurrent_resets[:-1], recurrent_resets[1:]
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
                recurrent_resets=recurrent_resets,
                next_recurrent_resets=next_recurrent_resets,
                teacher_targets=teacher_targets,
                entry_action_targets=entry_action_targets,
                entry_long_anchor_indices=np.flatnonzero(
                    entry_action_targets == int(Action.ENTER_LONG_1)
                ).astype(np.int32, copy=False),
                entry_short_anchor_indices=np.flatnonzero(
                    entry_action_targets == int(Action.ENTER_SHORT_1)
                ).astype(np.int32, copy=False),
                regime_selectivity_headroom_fractions=(
                    regime_selectivity_headroom
                ),
                safety_priorities=safety_priorities,
                entry_opportunity_priorities=entry_priorities,
                source_decision_indices=source_decision_indices,
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
        # Outcome is the competence boundary: scarce passes or blows must not
        # disappear merely because timeout episodes span more tickers. Balance
        # outcomes first, then rotate ticker/side buckets within each outcome.
        keys_by_outcome: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for key in buckets:
            keys_by_outcome[key[1]].append(key)
        outcomes = list(keys_by_outcome)
        self._random.shuffle(outcomes)
        for keys in keys_by_outcome.values():
            self._random.shuffle(keys)
        cursors = {outcome: 0 for outcome in outcomes}
        selected: list[_StoredEpisode] = []
        while len(selected) < count:
            for outcome in outcomes:
                keys = keys_by_outcome[outcome]
                key = keys[cursors[outcome] % len(keys)]
                cursors[outcome] += 1
                selected.append(self._random.choice(buckets[key]))
                if len(selected) == count:
                    break
        return tuple(selected)

    def sample(self, count: int) -> tuple[tuple[Transition, ...], ...]:
        sequences = []
        terminal_count = round(count * self.terminal_sequence_fraction)
        safety_count = round(count * self.safety_sequence_fraction)
        entry_count = round(count * self.entry_opportunity_sequence_fraction)
        sampled_episodes = list(self.sample_episodes(count))
        balanced_entry_anchors: dict[int, tuple[_StoredEpisode, int]] = {}
        if self.entry_opportunity_side_balance == "equal_long_short_v1":
            opportunity_episodes: dict[Action, list[_StoredEpisode]] = defaultdict(list)
            cumulative_counts: dict[Action, list[int]] = defaultdict(list)
            for stored in self._episodes.values():
                for side, anchors in (
                    (Action.ENTER_LONG_1, stored.entry_long_anchor_indices),
                    (Action.ENTER_SHORT_1, stored.entry_short_anchor_indices),
                ):
                    if not anchors.size:
                        continue
                    opportunity_episodes[side].append(stored)
                    previous = (
                        cumulative_counts[side][-1]
                        if cumulative_counts[side]
                        else 0
                    )
                    cumulative_counts[side].append(previous + int(anchors.size))
            available_sides = [
                side
                for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
                if cumulative_counts[side]
            ]
            if len(available_sides) == 2:
                self._random.shuffle(available_sides)
            entry_start = terminal_count + safety_count
            for offset in range(entry_count):
                if not available_sides:
                    break
                side = available_sides[offset % len(available_sides)]
                anchor_offset = self._random.randrange(cumulative_counts[side][-1])
                episode_offset = bisect_right(
                    cumulative_counts[side], anchor_offset
                )
                previous_count = (
                    cumulative_counts[side][episode_offset - 1]
                    if episode_offset
                    else 0
                )
                episode = opportunity_episodes[side][episode_offset]
                anchor_indices = (
                    episode.entry_long_anchor_indices
                    if side == Action.ENTER_LONG_1
                    else episode.entry_short_anchor_indices
                )
                anchor_index = int(anchor_indices[anchor_offset - previous_count])
                balanced_entry_anchors[entry_start + offset] = (
                    episode,
                    anchor_index,
                )
        for index, episode in enumerate(sampled_episodes):
            balanced_entry_anchor = balanced_entry_anchors.get(index)
            if balanced_entry_anchor is not None:
                episode, anchor_index = balanced_entry_anchor
                sequences.append(episode.entry_anchored_sequence(
                    anchor_index=anchor_index,
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                ))
                continue
            if episode.transition_count < self.sequence_length:
                if index < terminal_count:
                    anchor = "terminal"
                elif index < terminal_count + safety_count:
                    anchor = "safety"
                elif index < terminal_count + safety_count + entry_count:
                    anchor = "entry"
                else:
                    # Recovery traces always end at a real economic boundary;
                    # default short samples retain that boundary rather than
                    # silently burying it beyond the learner's n-step window.
                    anchor = "terminal"
                sequences.append(episode.padded_sequence(
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                    anchor=anchor,
                ))
                continue
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

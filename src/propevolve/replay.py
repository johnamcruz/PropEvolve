"""Bounded balanced replay of causal recurrent trading sequences."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
import hashlib
import random
from typing import Mapping

import numpy as np

from .decision import Action


PAIRED_A_PLUS_CONTEXT_WIDTH = 7


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
    teacher_imitation_visible: bool = True
    entry_action_target: Action | None = None
    regime_selectivity_headroom_fraction: float | None = None
    recovery_active: bool = False
    safety_priority: float = 0.0
    entry_opportunity_priority: float = 0.0
    regime_wait_priority: float = 0.0
    training_valid: bool = True
    source_decision_index: int | None = None
    paired_a_plus_context: np.ndarray | None = None
    paired_a_plus_side: Action | None = None
    paired_a_plus_economic_win: bool | None = None
    paired_a_plus_pair_id: int | None = None
    paired_a_plus_pair_side: Action | None = None
    paired_a_plus_population_weight: float | None = None


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
    teacher_imitation_visible: np.ndarray
    entry_action_targets: np.ndarray
    entry_long_anchor_indices: np.ndarray
    entry_short_anchor_indices: np.ndarray
    regime_selectivity_headroom_fractions: np.ndarray
    recovery_active: np.ndarray
    safety_priorities: np.ndarray
    entry_opportunity_priorities: np.ndarray
    regime_wait_priorities: np.ndarray
    regime_wait_anchor_indices: np.ndarray
    regime_wait_anchor_cumulative_priorities: np.ndarray
    regime_wait_priority_sum: float
    source_decision_indices: np.ndarray
    paired_a_plus_contexts: np.ndarray
    paired_a_plus_sides: np.ndarray
    paired_a_plus_economic_wins: np.ndarray
    paired_a_plus_long_winner_anchor_indices: np.ndarray
    paired_a_plus_long_failure_anchor_indices: np.ndarray
    paired_a_plus_short_winner_anchor_indices: np.ndarray
    paired_a_plus_short_failure_anchor_indices: np.ndarray

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
            not np.isfinite(item.regime_wait_priority)
            or item.regime_wait_priority < 0
            for item in transitions
        ):
            raise ValueError("replay Regime WAIT priority is invalid")
        if any(
            item.source_decision_index is not None
            and (
                isinstance(item.source_decision_index, bool)
                or int(item.source_decision_index) < 0
            )
            for item in transitions
        ):
            raise ValueError("replay source decision index is invalid")
        paired_a_plus_contexts = np.full(
            (len(transitions), PAIRED_A_PLUS_CONTEXT_WIDTH),
            np.nan,
            dtype=np.float32,
        )
        paired_a_plus_sides = np.full(len(transitions), -1, dtype=np.int8)
        paired_a_plus_economic_wins = np.full(
            len(transitions), -1, dtype=np.int8
        )
        flat_actions = {
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
        for index, item in enumerate(transitions):
            evidence = (
                item.paired_a_plus_context,
                item.paired_a_plus_side,
                item.paired_a_plus_economic_win,
            )
            if all(value is None for value in evidence):
                continue
            if any(value is None for value in evidence):
                raise ValueError("replay paired A+ evidence is incomplete")
            context = np.asarray(
                item.paired_a_plus_context, dtype=np.float32
            ).reshape(-1)
            try:
                side = Action(item.paired_a_plus_side)
            except (TypeError, ValueError) as error:
                raise ValueError("replay paired A+ side is invalid") from error
            economic_win = item.paired_a_plus_economic_win
            if (
                context.shape != (PAIRED_A_PLUS_CONTEXT_WIDTH,)
                or not np.isfinite(context).all()
                or (context < 0.0).any()
                or (context > 1.0).any()
                or side not in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
                or type(economic_win) is not bool
                or item.teacher_target is None
                or item.regime_selectivity_headroom_fraction is None
                or not flat_actions.issubset(item.valid_actions)
                or (
                    economic_win
                    and item.entry_action_target != side
                )
                or (
                    not economic_win
                    and item.entry_action_target != Action.WAIT
                )
            ):
                raise ValueError(
                    "replay paired A+ economic outcome or learner evidence "
                    "is invalid"
                )
            paired_a_plus_contexts[index] = context
            paired_a_plus_sides[index] = int(side)
            paired_a_plus_economic_wins[index] = int(economic_win)
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
        regime_wait_priorities = np.asarray(
            [item.regime_wait_priority for item in transitions], np.float32
        )
        regime_wait_anchor_indices = np.flatnonzero(
            regime_wait_priorities > 0.0
        ).astype(np.int32, copy=False)
        if any(
            item.regime_wait_priority > 0.0
            and (
                item.entry_action_target != Action.WAIT
                or item.teacher_target is None
                or not flat_actions.issubset(item.valid_actions)
            )
            for item in transitions
        ):
            raise ValueError("replay Regime WAIT priority lacks exact evidence")
        regime_wait_anchor_cumulative_priorities = np.cumsum(
            regime_wait_priorities[regime_wait_anchor_indices], dtype=np.float64
        )
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
            teacher_imitation_visible=np.asarray(
                [item.teacher_imitation_visible for item in transitions],
                np.bool_,
            ),
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
            recovery_active=np.asarray(
                [item.recovery_active for item in transitions], np.bool_
            ),
            safety_priorities=np.asarray(
                [item.safety_priority for item in transitions], np.float32
            ),
            entry_opportunity_priorities=np.asarray(
                [item.entry_opportunity_priority for item in transitions], np.float32
            ),
            regime_wait_priorities=regime_wait_priorities,
            regime_wait_anchor_indices=regime_wait_anchor_indices,
            regime_wait_anchor_cumulative_priorities=(
                regime_wait_anchor_cumulative_priorities
            ),
            regime_wait_priority_sum=(
                0.0
                if not regime_wait_anchor_cumulative_priorities.size
                else float(regime_wait_anchor_cumulative_priorities[-1])
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
            paired_a_plus_contexts=paired_a_plus_contexts,
            paired_a_plus_sides=paired_a_plus_sides,
            paired_a_plus_economic_wins=paired_a_plus_economic_wins,
            paired_a_plus_long_winner_anchor_indices=np.flatnonzero(
                (paired_a_plus_sides == int(Action.ENTER_LONG_1))
                & (paired_a_plus_economic_wins == 1)
            ).astype(np.int32, copy=False),
            paired_a_plus_long_failure_anchor_indices=np.flatnonzero(
                (paired_a_plus_sides == int(Action.ENTER_LONG_1))
                & (paired_a_plus_economic_wins == 0)
            ).astype(np.int32, copy=False),
            paired_a_plus_short_winner_anchor_indices=np.flatnonzero(
                (paired_a_plus_sides == int(Action.ENTER_SHORT_1))
                & (paired_a_plus_economic_wins == 1)
            ).astype(np.int32, copy=False),
            paired_a_plus_short_failure_anchor_indices=np.flatnonzero(
                (paired_a_plus_sides == int(Action.ENTER_SHORT_1))
                & (paired_a_plus_economic_wins == 0)
            ).astype(np.int32, copy=False),
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
                competence_anchor=self.outcome == "pass",
                teacher_target=(
                    None
                    if self.teacher_targets is None
                    or not np.isfinite(self.teacher_targets[index]).all()
                    else self.teacher_targets[index]
                ),
                teacher_imitation_visible=bool(
                    self.teacher_imitation_visible[index]
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
                recovery_active=bool(self.recovery_active[index]),
                safety_priority=float(self.safety_priorities[index]),
                entry_opportunity_priority=float(
                    self.entry_opportunity_priorities[index]
                ),
                regime_wait_priority=float(self.regime_wait_priorities[index]),
                training_valid=True,
                source_decision_index=(
                    None
                    if self.source_decision_indices[index] < 0
                    else int(self.source_decision_indices[index])
                ),
                paired_a_plus_context=(
                    None
                    if not np.isfinite(
                        self.paired_a_plus_contexts[index]
                    ).all()
                    else self.paired_a_plus_contexts[index]
                ),
                paired_a_plus_side=(
                    None
                    if self.paired_a_plus_sides[index] < 0
                    else Action(int(self.paired_a_plus_sides[index]))
                ),
                paired_a_plus_economic_win=(
                    None
                    if self.paired_a_plus_economic_wins[index] < 0
                    else bool(self.paired_a_plus_economic_wins[index])
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

    def recovery_anchored_sequence(
        self,
        *,
        anchor_index: int,
        length: int,
        recurrent_burn_in: int,
        n_step_return: int,
    ) -> tuple[Transition, ...]:
        """Place the last red row before breakeven at the first learnable row."""
        last_learning_index = length - n_step_return
        if (
            not 0 <= anchor_index < self.transition_count - 1
            or not recurrent_burn_in <= last_learning_index < length
            or not bool(self.recovery_active[anchor_index])
            or bool(self.recovery_active[anchor_index + 1])
        ):
            raise ValueError("successful recovery replay anchor is invalid")
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
            real[0] = replace(real[0], recurrent_reset=True)
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
        regime_wait_sequence_fraction: float = 0.0,
        regime_wait_sequence_update_period: int = 0,
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
            or not 0.0 <= regime_wait_sequence_fraction <= 1.0
            or terminal_sequence_fraction + safety_sequence_fraction
            + entry_opportunity_sequence_fraction
            + regime_wait_sequence_fraction > 1.0
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
        self.regime_wait_sequence_fraction = float(regime_wait_sequence_fraction)
        if (
            isinstance(regime_wait_sequence_update_period, bool)
            or not isinstance(regime_wait_sequence_update_period, int)
            or regime_wait_sequence_update_period < 0
            or (
                regime_wait_sequence_update_period > 0
                and self.regime_wait_sequence_fraction > 0.0
            )
            or (
                regime_wait_sequence_update_period > 0
                and self.terminal_sequence_fraction <= 0.0
            )
        ):
            raise ValueError("replay Regime WAIT update schedule is invalid")
        self.regime_wait_sequence_update_period = int(
            regime_wait_sequence_update_period
        )
        if entry_opportunity_side_balance not in {
            "none",
            "equal_long_short_v1",
            "paired_recurrent_long_short_v1",
        }:
            raise ValueError("replay entry opportunity side balance is invalid")
        self.entry_opportunity_side_balance = entry_opportunity_side_balance
        self._episodes: OrderedDict[str, _StoredEpisode] = OrderedDict()
        self._transition_count = 0
        self._random = random.Random(seed)
        self._sample_calls = 0

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
            "schema_version": 11,
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
                "regime_wait_sequence_fraction": self.regime_wait_sequence_fraction,
                "regime_wait_sequence_update_period": (
                    self.regime_wait_sequence_update_period
                ),
                "entry_opportunity_side_balance": (
                    self.entry_opportunity_side_balance
                ),
            },
            "random_state": self._random.getstate(),
            "sample_calls": self._sample_calls,
            "episodes": [
                {
                    field: getattr(episode, field)
                    for field in _StoredEpisode.__dataclass_fields__
                    if field not in {
                        "entry_long_anchor_indices",
                        "entry_short_anchor_indices",
                        "regime_wait_anchor_indices",
                        "regime_wait_anchor_cumulative_priorities",
                        "regime_wait_priority_sum",
                        "paired_a_plus_long_winner_anchor_indices",
                        "paired_a_plus_long_failure_anchor_indices",
                        "paired_a_plus_short_winner_anchor_indices",
                        "paired_a_plus_short_failure_anchor_indices",
                    }
                }
                for episode in self._episodes.values()
            ],
        }

    def sampler_state_dict(self) -> dict[str, object]:
        """Return resumable sampler metadata without duplicating episodes."""
        return {
            "schema_version": 1,
            "random_state": self._random.getstate(),
            "sample_calls": self._sample_calls,
        }

    def load_sampler_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore sampler metadata after immutable episodes are loaded."""
        if set(state) != {"schema_version", "random_state", "sample_calls"}:
            raise ValueError("replay sampler state is invalid")
        sample_calls = state["sample_calls"]
        if (
            state["schema_version"] != 1
            or isinstance(sample_calls, bool)
            or not isinstance(sample_calls, int)
            or sample_calls < 0
        ):
            raise ValueError("replay sampler state is invalid")
        try:
            self._random.setstate(state["random_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("replay sampler state is invalid") from error
        self._sample_calls = sample_calls

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore replay exactly and fail closed if its sampling contract drifted."""
        schema_version = state.get("schema_version")
        if schema_version not in {7, 8, 9, 10, 11}:
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
            "regime_wait_sequence_fraction": self.regime_wait_sequence_fraction,
            "regime_wait_sequence_update_period": (
                self.regime_wait_sequence_update_period
            ),
            "entry_opportunity_side_balance": (
                self.entry_opportunity_side_balance
            ),
        }
        if schema_version in {7, 8}:
            if self.regime_wait_sequence_update_period != 0:
                raise ValueError("replay checkpoint contract drifted")
            expected_contract.pop("regime_wait_sequence_update_period")
        if schema_version == 7:
            if self.regime_wait_sequence_fraction != 0.0:
                raise ValueError("replay checkpoint contract drifted")
            expected_contract.pop("regime_wait_sequence_fraction")
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
                regime_wait_priorities = np.asarray(
                    payload.get(
                        "regime_wait_priorities",
                        np.zeros(actions.size, dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                raw_teacher_targets = payload["teacher_targets"]
                teacher_imitation_visible = np.asarray(
                    payload["teacher_imitation_visible"], dtype=np.bool_
                )
                raw_entry_action_targets = payload["entry_action_targets"]
                raw_regime_selectivity_headroom = payload[
                    "regime_selectivity_headroom_fractions"
                ]
                recovery_active = np.asarray(
                    payload.get(
                        "recovery_active",
                        np.zeros(actions.size, dtype=np.bool_),
                    ),
                    dtype=np.bool_,
                )
                source_decision_indices = np.asarray(
                    payload.get(
                        "source_decision_indices",
                        np.full(actions.size, -1, dtype=np.int64),
                    ),
                    dtype=np.int64,
                )
                paired_a_plus_contexts = np.asarray(
                    payload.get(
                        "paired_a_plus_contexts",
                        np.full(
                            (actions.size, PAIRED_A_PLUS_CONTEXT_WIDTH),
                            np.nan,
                            dtype=np.float32,
                        ),
                    ),
                    dtype=np.float32,
                )
                paired_a_plus_sides = np.asarray(
                    payload.get(
                        "paired_a_plus_sides",
                        np.full(actions.size, -1, dtype=np.int8),
                    ),
                    dtype=np.int8,
                )
                paired_a_plus_economic_wins = np.asarray(
                    payload.get(
                        "paired_a_plus_economic_wins",
                        np.full(actions.size, -1, dtype=np.int8),
                    ),
                    dtype=np.int8,
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
                or teacher_imitation_visible.shape != (count,)
                or regime_selectivity_headroom.shape != (count,)
                or recovery_active.shape != (count,)
                or source_decision_indices.shape != (count,)
                or safety_priorities.shape != (count,)
                or entry_priorities.shape != (count,)
                or regime_wait_priorities.shape != (count,)
                or paired_a_plus_contexts.shape
                != (count, PAIRED_A_PLUS_CONTEXT_WIDTH)
                or paired_a_plus_sides.shape != (count,)
                or paired_a_plus_economic_wins.shape != (count,)
                or (teacher_targets is not None and teacher_targets.shape[0] != count)
                or not np.isfinite(observations).all()
                or not np.isfinite(rewards).all()
                or not np.isfinite(safety_priorities).all()
                or not np.isfinite(entry_priorities).all()
                or not np.isfinite(regime_wait_priorities).all()
                or (safety_priorities < 0).any()
                or (entry_priorities < 0).any()
                or (regime_wait_priorities < 0).any()
                or (actions < 0).any()
                or (actions >= action_count).any()
                or (entry_action_targets < -1).any()
                or (entry_action_targets > int(Action.ENTER_SHORT_1)).any()
                or (source_decision_indices < -1).any()
                or (paired_a_plus_sides < -1).any()
                or (paired_a_plus_sides > int(Action.ENTER_SHORT_1)).any()
                or (paired_a_plus_economic_wins < -1).any()
                or (paired_a_plus_economic_wins > 1).any()
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
            paired_rows = paired_a_plus_sides >= 0
            missing_paired_rows = paired_a_plus_sides < 0
            if (
                not np.array_equal(
                    paired_rows, paired_a_plus_economic_wins >= 0
                )
                or not np.isfinite(paired_a_plus_contexts[paired_rows]).all()
                or (paired_a_plus_contexts[paired_rows] < 0.0).any()
                or (paired_a_plus_contexts[paired_rows] > 1.0).any()
                or not np.isnan(
                    paired_a_plus_contexts[missing_paired_rows]
                ).all()
                or np.any(
                    paired_rows
                    & (paired_a_plus_sides == int(Action.WAIT))
                )
                or np.any(
                    paired_rows
                    & (paired_a_plus_economic_wins == 1)
                    & (entry_action_targets != paired_a_plus_sides)
                )
                or np.any(
                    paired_rows
                    & (paired_a_plus_economic_wins == 0)
                    & (entry_action_targets != int(Action.WAIT))
                )
            ):
                raise ValueError("replay checkpoint paired A+ evidence is invalid")
                valid_teacher_rows = np.isfinite(teacher_targets).all(axis=1)
                missing_teacher_rows = np.isnan(teacher_targets).all(axis=1)
                if not np.logical_or(valid_teacher_rows, missing_teacher_rows).all():
                    raise ValueError("replay checkpoint teacher targets are invalid")
            episode_id = str(payload.get("episode_id", ""))
            if not episode_id or episode_id in restored:
                raise ValueError("replay checkpoint episode identity is invalid")
            positive_regime_wait = regime_wait_priorities > 0.0
            if np.any(
                positive_regime_wait
                & (entry_action_targets != int(Action.WAIT))
            ) or (
                np.any(positive_regime_wait)
                and (
                    teacher_targets is None
                    or np.any(
                        positive_regime_wait
                        & ~np.isfinite(teacher_targets).all(axis=1)
                    )
                )
            ):
                raise ValueError("replay checkpoint Regime WAIT evidence is invalid")
            regime_wait_anchor_indices = np.flatnonzero(
                positive_regime_wait
            ).astype(np.int32, copy=False)
            regime_wait_anchor_cumulative_priorities = np.cumsum(
                regime_wait_priorities[regime_wait_anchor_indices],
                dtype=np.float64,
            )
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
                teacher_imitation_visible=teacher_imitation_visible,
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
                recovery_active=recovery_active,
                safety_priorities=safety_priorities,
                entry_opportunity_priorities=entry_priorities,
                regime_wait_priorities=regime_wait_priorities,
                regime_wait_anchor_indices=regime_wait_anchor_indices,
                regime_wait_anchor_cumulative_priorities=(
                    regime_wait_anchor_cumulative_priorities
                ),
                regime_wait_priority_sum=(
                    0.0
                    if not regime_wait_anchor_cumulative_priorities.size
                    else float(regime_wait_anchor_cumulative_priorities[-1])
                ),
                source_decision_indices=source_decision_indices,
                paired_a_plus_contexts=paired_a_plus_contexts,
                paired_a_plus_sides=paired_a_plus_sides,
                paired_a_plus_economic_wins=paired_a_plus_economic_wins,
                paired_a_plus_long_winner_anchor_indices=np.flatnonzero(
                    (paired_a_plus_sides == int(Action.ENTER_LONG_1))
                    & (paired_a_plus_economic_wins == 1)
                ).astype(np.int32, copy=False),
                paired_a_plus_long_failure_anchor_indices=np.flatnonzero(
                    (paired_a_plus_sides == int(Action.ENTER_LONG_1))
                    & (paired_a_plus_economic_wins == 0)
                ).astype(np.int32, copy=False),
                paired_a_plus_short_winner_anchor_indices=np.flatnonzero(
                    (paired_a_plus_sides == int(Action.ENTER_SHORT_1))
                    & (paired_a_plus_economic_wins == 1)
                ).astype(np.int32, copy=False),
                paired_a_plus_short_failure_anchor_indices=np.flatnonzero(
                    (paired_a_plus_sides == int(Action.ENTER_SHORT_1))
                    & (paired_a_plus_economic_wins == 0)
                ).astype(np.int32, copy=False),
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
        sample_calls = state.get("sample_calls", 0)
        if (
            isinstance(sample_calls, bool)
            or not isinstance(sample_calls, int)
            or sample_calls < 0
            or (schema_version in {9, 10} and "sample_calls" not in state)
        ):
            raise ValueError("replay checkpoint sample schedule is invalid")
        self._episodes = restored
        self._transition_count = transition_count
        self._sample_calls = sample_calls

    def add(self, episode: Episode) -> None:
        self._retain_stored_episode(_StoredEpisode.from_episode(episode))

    def _retain_stored_episode(self, stored: _StoredEpisode) -> None:
        replaced = self._episodes.pop(stored.episode_id, None)
        if replaced is not None:
            self._transition_count -= replaced.transition_count
        self._episodes[stored.episode_id] = stored
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

    def absorb_recent_successful_recoveries(
        self,
        source: "BalancedSequenceReplay",
        *,
        max_examples: int = 8,
    ) -> int:
        """Share newest retained recoveries from ordinary replay.

        Stored episodes are immutable, so sharing them avoids copying their
        observation arrays or serializing a second growing replay corpus.
        Re-running this after checkpoint restore reconstructs the same online
        priority pool from the already-resumable ordinary replay.
        """
        if not isinstance(source, BalancedSequenceReplay) or max_examples < 1:
            raise ValueError("recent recovery replay request is invalid")
        if (
            self.sequence_length != source.sequence_length
            or self.recurrent_burn_in != source.recurrent_burn_in
            or self.n_step_return != source.n_step_return
        ):
            raise ValueError("recent recovery replay recurrent contract drifted")
        candidates = sorted(
            (
                episode
                for episode in source._episodes.values()
                if episode.outcome in {"pass", "timeout"}
                and episode.transition_count >= 2
                and self._first_recovery_boundary(episode) is not None
            ),
            key=lambda episode: (-episode.ended_at_ns, episode.episode_id),
        )[:max_examples]
        added = 0
        # Retain oldest-to-newest so a tight target capacity preserves the
        # newest pass if admission requires eviction.
        for episode in reversed(candidates):
            if episode.episode_id not in self._episodes:
                added += 1
            self._retain_stored_episode(episode)
        return added

    def absorb_recent_passes(
        self,
        source: "BalancedSequenceReplay",
        *,
        max_examples: int = 8,
    ) -> int:
        """Share newest complete passes for a single-policy curriculum."""
        if not isinstance(source, BalancedSequenceReplay) or max_examples < 1:
            raise ValueError("recent balance pass replay request is invalid")
        if (
            self.sequence_length != source.sequence_length
            or self.recurrent_burn_in != source.recurrent_burn_in
            or self.n_step_return != source.n_step_return
        ):
            raise ValueError("recent balance pass recurrent contract drifted")
        candidates = sorted(
            (
                episode
                for episode in source._episodes.values()
                if episode.outcome == "pass"
            ),
            key=lambda episode: (-episode.ended_at_ns, episode.episode_id),
        )[:max_examples]
        added = 0
        for episode in reversed(candidates):
            if episode.episode_id not in self._episodes:
                added += 1
            self._retain_stored_episode(episode)
        return added

    def absorb_recent_post_recovery_contrasts(
        self,
        source: "BalancedSequenceReplay",
        *,
        max_examples: int = 8,
    ) -> int:
        """Share newest retained and relapsed recovery examples by side."""
        if not isinstance(source, BalancedSequenceReplay) or max_examples < 1:
            raise ValueError("recent post-recovery contrast request is invalid")
        if (
            self.sequence_length != source.sequence_length
            or self.recurrent_burn_in != source.recurrent_burn_in
            or self.n_step_return != source.n_step_return
        ):
            raise ValueError("recent post-recovery recurrent contract drifted")
        buckets: dict[tuple[str, Action], list[_StoredEpisode]] = defaultdict(list)
        for episode in source._episodes.values():
            if episode.transition_count < 2:
                continue
            boundaries = np.flatnonzero(
                episode.recovery_active[:-1]
                & ~episode.recovery_active[1:]
            )
            if not boundaries.size:
                continue
            retained_boundary = self._retained_recovery_boundary(episode)
            if retained_boundary is not None:
                for side, winner_indices in (
                    (
                        Action.ENTER_LONG_1,
                        episode.paired_a_plus_long_winner_anchor_indices,
                    ),
                    (
                        Action.ENTER_SHORT_1,
                        episode.paired_a_plus_short_winner_anchor_indices,
                    ),
                ):
                    if np.any(
                        (winner_indices <= retained_boundary)
                        & episode.recovery_active[winner_indices]
                    ):
                        buckets[("retained", side)].append(episode)
            if not bool(episode.recovery_active[-1]):
                continue
            healthy_start = int(boundaries[0]) + 1
            relapse_offsets = np.flatnonzero(
                episode.recovery_active[healthy_start:]
            )
            if not relapse_offsets.size:
                continue
            relapse_index = healthy_start + int(relapse_offsets[0])
            for side, failure_indices in (
                (
                    Action.ENTER_LONG_1,
                    episode.paired_a_plus_long_failure_anchor_indices,
                ),
                (
                    Action.ENTER_SHORT_1,
                    episode.paired_a_plus_short_failure_anchor_indices,
                ),
            ):
                if np.any(
                    (failure_indices >= relapse_index)
                    & episode.recovery_active[failure_indices]
                ):
                    buckets[("relapsed", side)].append(episode)
        selected = {
            episode.episode_id: episode
            for candidates in buckets.values()
            for episode in sorted(
                candidates,
                key=lambda item: (-item.ended_at_ns, item.episode_id),
            )[:max_examples]
        }
        added = 0
        for episode in sorted(
            selected.values(),
            key=lambda item: (item.ended_at_ns, item.episode_id),
        ):
            if episode.episode_id not in self._episodes:
                added += 1
            self._retain_stored_episode(episode)
        return added

    def absorb_recent_healthy_passes(
        self,
        source: "BalancedSequenceReplay",
        *,
        max_examples: int = 8,
    ) -> int:
        """Share newest passes that contain usable post-recovery flat rows."""
        if not isinstance(source, BalancedSequenceReplay) or max_examples < 1:
            raise ValueError("recent healthy pass replay request is invalid")
        if (
            self.sequence_length != source.sequence_length
            or self.recurrent_burn_in != source.recurrent_burn_in
            or self.n_step_return != source.n_step_return
        ):
            raise ValueError("recent healthy pass replay recurrent contract drifted")
        flat_action_count = int(Action.ENTER_SHORT_1) + 1
        candidates = sorted(
            (
                episode
                for episode in source._episodes.values()
                if episode.outcome == "pass"
                and np.any(
                    ~episode.recovery_active
                    & (episode.entry_action_targets >= int(Action.WAIT))
                    & episode.valid_masks[:, :flat_action_count].all(axis=1)
                    & (episode.valid_masks.sum(axis=1) == flat_action_count)
                )
            ),
            key=lambda episode: (-episode.ended_at_ns, episode.episode_id),
        )[:max_examples]
        added = 0
        for episode in reversed(candidates):
            if episode.episode_id not in self._episodes:
                added += 1
            self._retain_stored_episode(episode)
        return added

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
        fixed_regime_wait_count = round(
            count * self.regime_wait_sequence_fraction
        )
        entry_count = round(count * self.entry_opportunity_sequence_fraction)
        sampled_episodes = list(self.sample_episodes(count))
        self._sample_calls += 1
        scheduled_regime_wait_count = int(
            self.regime_wait_sequence_update_period > 0
            and self._sample_calls % self.regime_wait_sequence_update_period == 0
        )
        if scheduled_regime_wait_count > terminal_count:
            raise ValueError("Regime WAIT schedule lacks a terminal sequence slot")
        terminal_count -= scheduled_regime_wait_count
        regime_wait_count = fixed_regime_wait_count + scheduled_regime_wait_count
        regime_wait_anchors: dict[int, tuple[_StoredEpisode, int]] = {}
        regime_wait_episodes = [
            stored
            for stored in self._episodes.values()
            if stored.regime_wait_priority_sum > 0.0
        ]
        cumulative_regime_wait_priorities: list[float] = []
        for stored in regime_wait_episodes:
            previous = (
                cumulative_regime_wait_priorities[-1]
                if cumulative_regime_wait_priorities
                else 0.0
            )
            cumulative_regime_wait_priorities.append(
                previous + stored.regime_wait_priority_sum
            )
        if scheduled_regime_wait_count and not cumulative_regime_wait_priorities:
            terminal_count += scheduled_regime_wait_count
            regime_wait_count -= scheduled_regime_wait_count
        regime_wait_start = terminal_count + safety_count
        for offset in range(regime_wait_count):
            if not cumulative_regime_wait_priorities:
                break
            selected_priority = (
                self._random.random() * cumulative_regime_wait_priorities[-1]
            )
            episode_offset = bisect_right(
                cumulative_regime_wait_priorities, selected_priority
            )
            episode = regime_wait_episodes[episode_offset]
            previous_priority = (
                cumulative_regime_wait_priorities[episode_offset - 1]
                if episode_offset
                else 0.0
            )
            local_priority = selected_priority - previous_priority
            anchor_offset = int(
                np.searchsorted(
                    episode.regime_wait_anchor_cumulative_priorities,
                    local_priority,
                    side="right",
                )
            )
            anchor_offset = min(
                anchor_offset, episode.regime_wait_anchor_indices.size - 1
            )
            regime_wait_anchors[regime_wait_start + offset] = (
                episode,
                int(episode.regime_wait_anchor_indices[anchor_offset]),
            )
        balanced_entry_anchors: dict[int, tuple[_StoredEpisode, int]] = {}
        paired_entry_anchors: dict[
            int, tuple[_StoredEpisode, int, int, Action, float]
        ] = {}
        if (
            self.entry_opportunity_side_balance
            == "paired_recurrent_long_short_v1"
        ):
            if entry_count % 2:
                raise ValueError(
                    "paired recurrent replay requires an even entry stratum"
                )
            winners: dict[
                Action, list[tuple[_StoredEpisode, int]]
            ] = defaultdict(list)
            failures: dict[
                Action, list[tuple[_StoredEpisode, int]]
            ] = defaultdict(list)
            for stored in self._episodes.values():
                for side, winner_indices, failure_indices in (
                    (
                        Action.ENTER_LONG_1,
                        stored.paired_a_plus_long_winner_anchor_indices,
                        stored.paired_a_plus_long_failure_anchor_indices,
                    ),
                    (
                        Action.ENTER_SHORT_1,
                        stored.paired_a_plus_short_winner_anchor_indices,
                        stored.paired_a_plus_short_failure_anchor_indices,
                    ),
                ):
                    winners[side].extend(
                        (stored, int(index)) for index in winner_indices
                    )
                    failures[side].extend(
                        (stored, int(index)) for index in failure_indices
                    )
            available_sides = [
                side
                for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
                if winners[side] and failures[side]
            ]
            if len(available_sides) == 2:
                self._random.shuffle(available_sides)
            entry_start = terminal_count + safety_count + regime_wait_count
            for pair_offset in range(entry_count // 2):
                if not available_sides:
                    break
                side = available_sides[pair_offset % len(available_sides)]
                winner_episode, winner_index = self._random.choice(
                    winners[side]
                )
                winner_context = self._paired_context_for_side(
                    winner_episode.paired_a_plus_contexts[winner_index],
                    side,
                )
                failure_episode, failure_index = min(
                    failures[side],
                    key=lambda candidate: self._paired_failure_rank(
                        candidate,
                        side=side,
                        winner_context=winner_context,
                    ),
                )
                # Pair identity is consumed only within this sampled batch.
                # Keep it backend-safe: MPS truncates large int64 values in
                # unique/equality kernels, so a sample-call prefix can corrupt
                # otherwise valid winner/failure pairs.
                pair_id = pair_offset
                pair_start = entry_start + pair_offset * 2
                population_count = len(winners[side]) + len(failures[side])
                winner_population_weight = (
                    2.0 * len(winners[side]) / population_count
                )
                failure_population_weight = (
                    2.0 * len(failures[side]) / population_count
                )
                paired_entry_anchors[pair_start] = (
                    winner_episode,
                    winner_index,
                    pair_id,
                    side,
                    winner_population_weight,
                )
                paired_entry_anchors[pair_start + 1] = (
                    failure_episode,
                    failure_index,
                    pair_id,
                    side,
                    failure_population_weight,
                )
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
            entry_start = terminal_count + safety_count + regime_wait_count
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
            regime_wait_anchor = regime_wait_anchors.get(index)
            if regime_wait_anchor is not None:
                episode, anchor_index = regime_wait_anchor
                sequences.append(episode.target_anchored_sequence(
                    anchor_index=anchor_index,
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                ))
                continue
            paired_entry_anchor = paired_entry_anchors.get(index)
            if paired_entry_anchor is not None:
                (
                    episode,
                    anchor_index,
                    pair_id,
                    pair_side,
                    population_weight,
                ) = paired_entry_anchor
                sequence = list(episode.target_anchored_sequence(
                    anchor_index=anchor_index,
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                ))
                sequence[self.recurrent_burn_in] = replace(
                    sequence[self.recurrent_burn_in],
                    paired_a_plus_pair_id=pair_id,
                    paired_a_plus_pair_side=pair_side,
                    paired_a_plus_population_weight=population_weight,
                )
                sequences.append(tuple(sequence))
                continue
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
                elif index < terminal_count + safety_count + regime_wait_count:
                    anchor = "terminal"
                elif index < (
                    terminal_count + safety_count + regime_wait_count + entry_count
                ):
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
            elif index < terminal_count + safety_count + regime_wait_count:
                critical = int(np.argmax(episode.regime_wait_priorities))
                start = max(0, min(last_start, critical - self.sequence_length + 1))
            elif index < (
                terminal_count + safety_count + regime_wait_count + entry_count
            ):
                critical = int(np.argmax(episode.entry_opportunity_priorities))
                start = max(0, min(last_start, critical - self.sequence_length + 1))
            else:
                start = self._random.randint(0, last_start)
            sequences.append(episode.sequence(start, self.sequence_length))
        return tuple(sequences)

    def sample_successful_recovery_sequences(
        self,
        count: int,
        *,
        max_examples: int = 8,
    ) -> tuple[tuple[Transition, ...], ...]:
        """Sample successful traces at the first red-to-breakeven boundary."""
        if count < 1 or max_examples < 1:
            raise ValueError("successful recovery replay count must be positive")
        candidates: list[tuple[int, float, str, _StoredEpisode, int]] = []
        for episode in self._episodes.values():
            if (
                episode.outcome not in {"pass", "timeout"}
                or episode.transition_count < 2
            ):
                continue
            boundary_index = self._first_recovery_boundary(episode)
            if boundary_index is not None:
                candidates.append((
                    episode.ended_at_ns,
                    float(np.sum(episode.rewards, dtype=np.float64)),
                    episode.episode_id,
                    episode,
                    boundary_index,
                ))
        if not candidates:
            return ()
        candidates = sorted(
            candidates,
            key=lambda item: (-item[0], -item[1], item[2]),
        )[:max_examples]
        start = self._sample_calls % len(candidates)
        self._sample_calls += count
        return tuple(
            episode.recovery_anchored_sequence(
                anchor_index=anchor_index,
                length=self.sequence_length,
                recurrent_burn_in=self.recurrent_burn_in,
                n_step_return=self.n_step_return,
            )
            for _, _, _, episode, anchor_index in (
                candidates[(start + offset) % len(candidates)]
                for offset in range(count)
            )
        )

    def sample_healthy_pass_sequences(
        self,
        count: int,
        *,
        max_examples: int = 8,
    ) -> tuple[tuple[Transition, ...], ...]:
        """Sample pass traces with a healthy flat row in the learning window."""
        if count < 1 or max_examples < 1:
            raise ValueError("healthy pass replay count must be positive")
        candidates: list[
            tuple[int, float, str, _StoredEpisode, np.ndarray]
        ] = []
        flat_action_count = int(Action.ENTER_SHORT_1) + 1
        for episode in self._episodes.values():
            if episode.outcome != "pass":
                continue
            healthy_flat_rows = np.flatnonzero(
                ~episode.recovery_active
                & (episode.entry_action_targets >= int(Action.WAIT))
                & episode.valid_masks[:, :flat_action_count].all(axis=1)
                & (episode.valid_masks.sum(axis=1) == flat_action_count)
            )
            if healthy_flat_rows.size:
                candidates.append((
                    episode.ended_at_ns,
                    float(np.sum(episode.rewards, dtype=np.float64)),
                    episode.episode_id,
                    episode,
                    healthy_flat_rows,
                ))
        if not candidates:
            return ()
        candidates = sorted(
            candidates,
            key=lambda item: (-item[0], -item[1], item[2]),
        )[:max_examples]
        starts = self._sample_calls
        self._sample_calls += count
        sequences = []
        for offset in range(count):
            candidate_index = (starts + offset) % len(candidates)
            _, _, _, episode, healthy_flat_rows = candidates[candidate_index]
            cycle = (starts + offset) // len(candidates)
            anchor_index = int(healthy_flat_rows[cycle % len(healthy_flat_rows)])
            sequences.append(episode.target_anchored_sequence(
                anchor_index=anchor_index,
                length=self.sequence_length,
                recurrent_burn_in=self.recurrent_burn_in,
                n_step_return=self.n_step_return,
            ))
        return tuple(sequences)

    def sample_balance_pass_entry_sequences(
        self,
        count: int,
        *,
        max_examples: int = 8,
    ) -> tuple[tuple[Transition, ...], ...]:
        """Replay the earliest economic winner from complete pass traces.

        Balance curricula begin with limited MLL headroom.  Random windows
        from a complete pass mostly rehearse WAIT or position-management rows
        and can omit the entry that made the recovery possible.  Anchor the
        authenticated +2R-before-1R winner at the first learnable row instead,
        retaining its causal recurrent prefix as burn-in.
        """
        if count < 1 or max_examples < 1:
            raise ValueError("balance pass entry replay count must be positive")
        candidates: dict[
            Action, list[tuple[int, float, str, _StoredEpisode, int]]
        ] = defaultdict(list)
        for episode in self._episodes.values():
            if episode.outcome != "pass":
                continue
            score = float(np.sum(episode.rewards, dtype=np.float64))
            for side, winner_indices in (
                (
                    Action.ENTER_LONG_1,
                    episode.paired_a_plus_long_winner_anchor_indices,
                ),
                (
                    Action.ENTER_SHORT_1,
                    episode.paired_a_plus_short_winner_anchor_indices,
                ),
            ):
                if winner_indices.size:
                    candidates[side].append((
                        episode.ended_at_ns,
                        score,
                        episode.episode_id,
                        episode,
                        int(winner_indices[0]),
                    ))
        for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
            candidates[side] = sorted(
                candidates[side],
                key=lambda item: (-item[0], -item[1], item[2], item[4]),
            )[:max_examples]
        available_sides = [
            side
            for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
            if candidates[side]
        ]
        if not available_sides:
            return ()
        start = self._sample_calls
        self._sample_calls += count
        sequences = []
        for offset in range(count):
            sample_index = start + offset
            side = available_sides[sample_index % len(available_sides)]
            side_cycle = sample_index // len(available_sides)
            _, _, _, episode, anchor_index = candidates[side][
                side_cycle % len(candidates[side])
            ]
            sequences.append(episode.target_anchored_sequence(
                anchor_index=anchor_index,
                length=self.sequence_length,
                recurrent_burn_in=self.recurrent_burn_in,
                n_step_return=self.n_step_return,
            ))
        return tuple(sequences)

    def sample_post_recovery_contrast_pairs(
        self,
        count: int,
        *,
        max_examples: int = 8,
        pair_id_start: int = 0,
    ) -> tuple[tuple[Transition, ...], ...]:
        """Pair recovery winners with failures after negative reactivation."""
        if (
            count < 1
            or max_examples < 1
            or isinstance(pair_id_start, bool)
            or pair_id_start < 0
        ):
            raise ValueError("post-recovery contrast request is invalid")
        retained: dict[
            Action, list[tuple[int, float, str, _StoredEpisode, int]]
        ] = defaultdict(list)
        givebacks: dict[
            Action, list[tuple[int, float, str, _StoredEpisode, int]]
        ] = defaultdict(list)
        for episode in self._episodes.values():
            if episode.transition_count < 2:
                continue
            boundaries = np.flatnonzero(
                episode.recovery_active[:-1]
                & ~episode.recovery_active[1:]
            )
            if not boundaries.size:
                continue
            score = float(np.sum(episode.rewards, dtype=np.float64))
            retained_boundary = self._retained_recovery_boundary(episode)
            if retained_boundary is not None:
                for side, winner_indices in (
                    (
                        Action.ENTER_LONG_1,
                        episode.paired_a_plus_long_winner_anchor_indices,
                    ),
                    (
                        Action.ENTER_SHORT_1,
                        episode.paired_a_plus_short_winner_anchor_indices,
                    ),
                ):
                    for anchor_index in winner_indices:
                        if (
                            int(anchor_index) <= retained_boundary
                            and bool(episode.recovery_active[int(anchor_index)])
                        ):
                            retained[side].append((
                                episode.ended_at_ns,
                                score,
                                episode.episode_id,
                                episode,
                                int(anchor_index),
                            ))
            if not bool(episode.recovery_active[-1]):
                continue
            healthy_start = int(boundaries[0]) + 1
            relapse_offsets = np.flatnonzero(
                episode.recovery_active[healthy_start:]
            )
            if not relapse_offsets.size:
                continue
            relapse_index = healthy_start + int(relapse_offsets[0])
            for side, failure_indices in (
                (
                    Action.ENTER_LONG_1,
                    episode.paired_a_plus_long_failure_anchor_indices,
                ),
                (
                    Action.ENTER_SHORT_1,
                    episode.paired_a_plus_short_failure_anchor_indices,
                ),
            ):
                for anchor_index in failure_indices:
                    anchor_index = int(anchor_index)
                    if (
                        anchor_index >= relapse_index
                        and bool(episode.recovery_active[anchor_index])
                    ):
                        givebacks[side].append((
                            episode.ended_at_ns,
                            score,
                            episode.episode_id,
                            episode,
                            anchor_index,
                        ))
        for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1):
            retained[side] = sorted(
                retained[side],
                key=lambda item: (-item[0], -item[1], item[2], item[4]),
            )[:max_examples]
            givebacks[side] = sorted(
                givebacks[side],
                key=lambda item: (-item[0], item[1], item[2], item[4]),
            )[:max_examples]
        available_sides = [
            side
            for side in (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
            if retained[side] and givebacks[side]
        ]
        if not available_sides:
            return ()
        start = self._sample_calls
        self._sample_calls += count
        sequences: list[tuple[Transition, ...]] = []
        for offset in range(count):
            side = available_sides[(start + offset) % len(available_sides)]
            winner = retained[side][
                ((start + offset) // len(available_sides)) % len(retained[side])
            ]
            _, _, _, winner_episode, winner_index = winner
            winner_context = self._paired_context_for_side(
                winner_episode.paired_a_plus_contexts[winner_index], side
            )
            failure = min(
                givebacks[side],
                key=lambda item: self._paired_failure_rank(
                    (item[3], item[4]),
                    side=side,
                    winner_context=winner_context,
                ),
            )
            _, _, _, failure_episode, failure_index = failure
            pair_id = pair_id_start + offset
            population_count = len(retained[side]) + len(givebacks[side])
            winner_weight = 2.0 * len(retained[side]) / population_count
            failure_weight = 2.0 * len(givebacks[side]) / population_count
            for episode, anchor_index, population_weight in (
                (winner_episode, winner_index, winner_weight),
                (failure_episode, failure_index, failure_weight),
            ):
                sequence = list(episode.target_anchored_sequence(
                    anchor_index=anchor_index,
                    length=self.sequence_length,
                    recurrent_burn_in=self.recurrent_burn_in,
                    n_step_return=self.n_step_return,
                ))
                sequence[self.recurrent_burn_in] = replace(
                    sequence[self.recurrent_burn_in],
                    paired_a_plus_pair_id=pair_id,
                    paired_a_plus_pair_side=side,
                    paired_a_plus_population_weight=population_weight,
                )
                sequences.append(tuple(sequence))
        return tuple(sequences)

    @staticmethod
    def _first_recovery_boundary(episode: _StoredEpisode) -> int | None:
        """Return the first recovery-controlled red-to-breakeven boundary."""
        if episode.transition_count < 2:
            return None
        boundaries = np.flatnonzero(
            episode.recovery_active[:-1]
            & ~episode.recovery_active[1:]
        )
        return None if not boundaries.size else int(boundaries[0])

    @staticmethod
    def _retained_recovery_boundary(episode: _StoredEpisode) -> int | None:
        """Return the first red-to-breakeven boundary when never relapsed."""
        boundary = BalancedSequenceReplay._first_recovery_boundary(episode)
        if boundary is None:
            return None
        if bool(episode.recovery_active[boundary + 1:].any()):
            return None
        return boundary

    @staticmethod
    def _paired_context_for_side(
        context: np.ndarray,
        side: Action,
    ) -> np.ndarray:
        """Canonicalize direction while preserving continuous market geometry."""
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (PAIRED_A_PLUS_CONTEXT_WIDTH,):
            raise ValueError("paired A+ context is invalid")
        if side == Action.ENTER_LONG_1:
            return context
        if side == Action.ENTER_SHORT_1:
            return context[[2, 3, 0, 1, 4, 5, 6]]
        raise ValueError("paired A+ side is invalid")

    @classmethod
    def _paired_failure_rank(
        cls,
        candidate: tuple[_StoredEpisode, int],
        *,
        side: Action,
        winner_context: np.ndarray,
    ) -> tuple[float, str, int]:
        episode, anchor_index = candidate
        failure_context = cls._paired_context_for_side(
            episode.paired_a_plus_contexts[anchor_index], side
        )
        distance = float(np.square(winner_context - failure_context).sum())
        return distance, episode.episode_id, anchor_index

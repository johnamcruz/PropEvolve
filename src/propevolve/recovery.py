"""Training-only full-action recovery values for the native policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from .decision import Action
from .replay import Transition


RECOVERY_ACTIONS = (
    Action.WAIT,
    Action.ENTER_LONG_1,
    Action.ENTER_SHORT_1,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RecoveryTargetAudit:
    """Validity counts for one recovery action-value population."""

    total: int
    discriminative: int
    ambiguous: int
    all_blow: int
    all_recover: int
    wait_best: int
    long_best: int
    short_best: int

    @property
    def valid_for_training(self) -> bool:
        return self.total > 0 and self.ambiguous == 0


def audit_recovery_action_values(
    rows: Sequence[Sequence[float]],
) -> RecoveryTargetAudit:
    """Fail-closed summary used before recovery targets can train the policy."""
    total = 0
    ambiguous = 0
    all_blow = 0
    all_recover = 0
    best_counts = [0, 0, 0]
    for row in rows:
        values = np.asarray(tuple(row), dtype=np.float64)
        if values.shape != (len(RECOVERY_ACTIONS),) or not np.isfinite(values).all():
            raise ValueError("recovery target audit row is invalid")
        total += 1
        all_blow += int(bool(np.all(values == -1.0)))
        all_recover += int(bool(np.all(values == 1.0)))
        winners = np.flatnonzero(values == values.max())
        if len(winners) != 1:
            ambiguous += 1
        else:
            best_counts[int(winners[0])] += 1
    return RecoveryTargetAudit(
        total=total,
        discriminative=total - ambiguous,
        ambiguous=ambiguous,
        all_blow=all_blow,
        all_recover=all_recover,
        wait_best=best_counts[0],
        long_best=best_counts[1],
        short_best=best_counts[2],
    )


@dataclass(frozen=True)
class RecoveryBranchResult:
    """Terminal economics after forcing one first action."""

    outcome: str
    terminal_pnl: float
    recovered: bool
    maximum_pnl: float | None = None
    retained: bool | None = None
    minimum_post_recovery_pnl: float | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"pass", "blow", "timeout"}:
            raise ValueError("recovery branch outcome is invalid")
        if type(self.recovered) is not bool or not math.isfinite(self.terminal_pnl):
            raise ValueError("recovery branch economics are invalid")
        retained = self.recovered if self.retained is None else self.retained
        if type(retained) is not bool:
            raise ValueError("recovery branch retention is invalid")
        if self.maximum_pnl is not None and not math.isfinite(self.maximum_pnl):
            raise ValueError("recovery branch maximum PnL is invalid")
        if (
            self.minimum_post_recovery_pnl is not None
            and not math.isfinite(self.minimum_post_recovery_pnl)
        ):
            raise ValueError("recovery branch minimum retained PnL is invalid")
        if retained and not self.recovered:
            raise ValueError("an unrecovered branch cannot retain recovery")
        if self.outcome == "blow" and retained:
            raise ValueError("a blown branch cannot retain recovery")
        object.__setattr__(self, "retained", retained)


@dataclass(frozen=True)
class RecoveryValueTarget:
    """One same-state WAIT/Long/Short economic supervision row."""

    observation: np.ndarray
    action_values: tuple[float, float, float]
    source_identity_sha256: str
    recurrent_observations: np.ndarray | None = None
    recurrent_resets: tuple[bool, ...] | None = None
    anchor_action: Action | None = None
    anchor_economic_success: bool | None = None

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32)
        values = tuple(float(value) for value in self.action_values)
        recurrent_observations = self.recurrent_observations
        recurrent_resets = self.recurrent_resets
        anchor_action = self.anchor_action
        anchor_economic_success = self.anchor_economic_success
        if anchor_action is not None:
            anchor_action = Action(anchor_action)
        if recurrent_observations is None:
            recurrent_resets = None
        else:
            recurrent_observations = np.asarray(
                recurrent_observations, dtype=np.float32
            )
            if recurrent_resets is None:
                raise ValueError("recovery recurrent resets are missing")
            recurrent_resets = tuple(recurrent_resets)
        if (
            observation.ndim != 1
            or observation.size == 0
            or not np.isfinite(observation).all()
            or len(values) != len(RECOVERY_ACTIONS)
            or not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)
            or _SHA256.fullmatch(self.source_identity_sha256) is None
            or recurrent_observations is not None
            and (
                recurrent_observations.ndim != 2
                or recurrent_observations.shape[0] < 1
                or recurrent_observations.shape[1:] != observation.shape
                or not np.isfinite(recurrent_observations).all()
                or len(recurrent_resets) != recurrent_observations.shape[0]
                or not all(type(value) is bool for value in recurrent_resets)
                or not recurrent_resets[0]
                or not np.array_equal(recurrent_observations[-1], observation)
            )
            or (anchor_action is None) != (anchor_economic_success is None)
            or anchor_action is not None
            and (
                anchor_action not in {
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                }
                or type(anchor_economic_success) is not bool
            )
        ):
            raise ValueError("recovery value target is invalid")
        observation = observation.copy()
        observation.setflags(write=False)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "action_values", values)
        if recurrent_observations is not None:
            recurrent_observations = recurrent_observations.copy()
            recurrent_observations.setflags(write=False)
        object.__setattr__(self, "recurrent_observations", recurrent_observations)
        object.__setattr__(self, "recurrent_resets", recurrent_resets)
        object.__setattr__(self, "anchor_action", anchor_action)

    @property
    def best_action(self) -> Action:
        values = np.asarray(self.action_values)
        winners = np.flatnonzero(values == values.max())
        if len(winners) != 1:
            raise ValueError("recovery value target has no unique best action")
        return RECOVERY_ACTIONS[int(winners[0])]

    @property
    def is_discriminative(self) -> bool:
        values = np.sort(np.asarray(self.action_values, dtype=np.float64))
        return bool(values[-1] > values[-2])

    @property
    def identity_sha256(self) -> str:
        version = 1 if self.recurrent_observations is None else 2
        digest = hashlib.sha256(
            f"propevolve-recovery-value-target-v{version}\0".encode("ascii")
        )
        digest.update(self.source_identity_sha256.encode("ascii"))
        digest.update(str(self.observation.shape).encode("ascii"))
        digest.update(self.observation.tobytes(order="C"))
        digest.update(np.asarray(self.action_values, np.float64).tobytes(order="C"))
        if self.recurrent_observations is not None:
            digest.update(str(self.recurrent_observations.shape).encode("ascii"))
            digest.update(self.recurrent_observations.tobytes(order="C"))
            digest.update(bytes(self.recurrent_resets))
        if self.anchor_action is not None:
            digest.update(bytes((int(self.anchor_action),)))
            digest.update(bytes((int(self.anchor_economic_success),)))
        return digest.hexdigest()


class RecoveryPolicy(Protocol):
    def select_action(
        self,
        observation: np.ndarray,
        *,
        hidden: object | None,
        valid_actions: tuple[Action, ...],
        epsilon: float,
        return_action_values: bool = False,
    ) -> tuple[Action, object | None, np.ndarray | None]: ...


class RecoveryHandoffPolicy:
    """Route negative PnL to recovery and nonnegative PnL to frozen V21.

    Only the active policy runs on ordinary decisions.  When the economic
    state changes, the newly active recurrent policy reconstructs its hidden
    state from the causal prefix collected since the last recurrent reset.
    """

    def __init__(
        self,
        recovery_policy: RecoveryPolicy,
        *,
        normal_policy: RecoveryPolicy,
    ) -> None:
        if recovery_policy is normal_policy:
            raise ValueError("recovery and normal policies must be distinct")
        self.recovery_policy = recovery_policy
        self.normal_policy = normal_policy
        self.reset()

    def reset(self) -> None:
        """Reset the causal prefix at an episode or recurrent boundary."""
        self._active_state: str | None = None
        self._active_hidden: object | None = None
        self._prefix: list[tuple[np.ndarray, tuple[Action, ...]]] = []

    def assert_teacher_free(self) -> None:
        """Fail closed unless both inference policies are teacher-free."""
        for policy in (self.recovery_policy, self.normal_policy):
            check = getattr(policy, "assert_teacher_free", None)
            if check is not None:
                check()

    def _reconstruct_hidden(self, policy: RecoveryPolicy) -> object | None:
        hidden = None
        for observation, valid_actions in self._prefix:
            _, hidden, _ = policy.select_action(
                observation,
                hidden=hidden,
                valid_actions=valid_actions,
                epsilon=0.0,
                return_action_values=False,
            )
        return hidden

    def select_action(
        self,
        observation: np.ndarray,
        *,
        valid_actions: tuple[Action, ...],
        realized_pnl: float,
        recovery_epsilon: float,
        return_action_values: bool = False,
    ) -> tuple[Action, np.ndarray | None, str]:
        """Select from recovery below zero and frozen V21 otherwise."""
        if (
            isinstance(realized_pnl, bool)
            or not math.isfinite(float(realized_pnl))
            or not 0.0 <= float(recovery_epsilon) <= 1.0
        ):
            raise ValueError("recovery handoff state is invalid")
        state = "recovery" if float(realized_pnl) < 0.0 else "normal"
        policy = (
            self.recovery_policy
            if state == "recovery"
            else self.normal_policy
        )
        if state != self._active_state:
            self._active_hidden = self._reconstruct_hidden(policy)
            self._active_state = state
        epsilon = float(recovery_epsilon) if state == "recovery" else 0.0
        action, self._active_hidden, action_values = policy.select_action(
            observation,
            hidden=self._active_hidden,
            valid_actions=valid_actions,
            epsilon=epsilon,
            return_action_values=return_action_values,
        )
        prefix_observation = np.asarray(observation, dtype=np.float32).copy()
        self._prefix.append((prefix_observation, tuple(valid_actions)))
        return Action(action), action_values, state


class RecoveryEnvironment(Protocol):
    def reset(self, *, options: Mapping[str, object]) -> tuple[np.ndarray, dict]: ...

    def step(
        self, action: Action
    ) -> tuple[np.ndarray, float, bool, bool, dict]: ...


class RecoveryValueStore:
    """Small deterministic store that is independent of ordinary replay."""

    def __init__(self, *, capacity: int, seed: int) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("recovery value store contract is invalid")
        self.capacity = capacity
        self.seed = seed
        self._targets: list[RecoveryValueTarget] = []
        self._rng = np.random.default_rng(seed)
        self._balanced_sample_count = 0

    def __len__(self) -> int:
        return len(self._targets)

    def add(self, target: RecoveryValueTarget) -> bool:
        if not isinstance(target, RecoveryValueTarget):
            raise TypeError("recovery store accepts only RecoveryValueTarget")
        if not target.is_discriminative:
            return False
        self._targets.append(target)
        if len(self._targets) > self.capacity:
            del self._targets[0]
        return True

    def sample(self) -> RecoveryValueTarget:
        if not self._targets:
            raise ValueError("recovery value store is empty")
        return self._targets[int(self._rng.integers(len(self._targets)))]

    def sample_balanced(self) -> RecoveryValueTarget:
        """Sample one target while balancing side and success when available."""
        if not self._targets:
            raise ValueError("recovery value store is empty")
        groups: dict[tuple[Action, bool], list[RecoveryValueTarget]] = {}
        for target in self._targets:
            if target.anchor_action is None:
                continue
            key = (target.anchor_action, bool(target.anchor_economic_success))
            groups.setdefault(key, []).append(target)
        if not groups:
            return self.sample()
        keys = sorted(groups, key=lambda item: (int(item[0]), item[1]))
        key = keys[self._balanced_sample_count % len(keys)]
        self._balanced_sample_count += 1
        group = groups[key]
        return group[int(self._rng.integers(len(group)))]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "propevolve_recovery_value_store_v1",
            "capacity": self.capacity,
            "seed": self.seed,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "balanced_sample_count": self._balanced_sample_count,
            "targets": [
                {
                    "observation": target.observation.copy(),
                    "action_values": target.action_values,
                    "source_identity_sha256": target.source_identity_sha256,
                    "recurrent_observations": (
                        None
                        if target.recurrent_observations is None
                        else target.recurrent_observations.copy()
                    ),
                    "recurrent_resets": target.recurrent_resets,
                    "anchor_action": (
                        None
                        if target.anchor_action is None
                        else int(target.anchor_action)
                    ),
                    "anchor_economic_success": (
                        target.anchor_economic_success
                    ),
                    "identity_sha256": target.identity_sha256,
                }
                for target in self._targets
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            state.get("schema") != "propevolve_recovery_value_store_v1"
            or state.get("capacity") != self.capacity
            or state.get("seed") != self.seed
            or not isinstance(state.get("targets"), list)
            or not isinstance(state.get("rng_state"), Mapping)
        ):
            raise ValueError("recovery value store checkpoint drifted")
        targets: list[RecoveryValueTarget] = []
        for payload in state["targets"]:
            if not isinstance(payload, Mapping):
                raise ValueError("recovery value store target is invalid")
            target = RecoveryValueTarget(
                observation=np.asarray(payload.get("observation"), np.float32),
                action_values=tuple(payload.get("action_values", ())),
                source_identity_sha256=str(payload.get("source_identity_sha256", "")),
                recurrent_observations=(
                    None
                    if payload.get("recurrent_observations") is None
                    else np.asarray(payload["recurrent_observations"], np.float32)
                ),
                recurrent_resets=(
                    None
                    if payload.get("recurrent_resets") is None
                    else tuple(payload["recurrent_resets"])
                ),
                anchor_action=(
                    None
                    if payload.get("anchor_action") is None
                    else Action(int(payload["anchor_action"]))
                ),
                anchor_economic_success=payload.get(
                    "anchor_economic_success"
                ),
            )
            if payload.get("identity_sha256") != target.identity_sha256:
                raise ValueError("recovery value store target identity drifted")
            if target.is_discriminative:
                targets.append(target)
        if len(targets) > self.capacity:
            raise ValueError("recovery value store exceeds capacity")
        self._targets = targets
        self._rng.bit_generator.state = copy.deepcopy(dict(state["rng_state"]))
        balanced_sample_count = state.get("balanced_sample_count", 0)
        if (
            isinstance(balanced_sample_count, bool)
            or not isinstance(balanced_sample_count, int)
            or balanced_sample_count < 0
        ):
            raise ValueError("recovery value store sample state drifted")
        self._balanced_sample_count = balanced_sample_count


def recovery_value_kl(
    policy_q_values: torch.Tensor,
    recovery_values: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Match the native action ranking to same-state recovery economics."""
    if (
        policy_q_values.shape != recovery_values.shape
        or policy_q_values.shape[-1:] != (len(RECOVERY_ACTIONS),)
        or not torch.is_floating_point(policy_q_values)
        or not torch.is_floating_point(recovery_values)
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("recovery value KL contract is invalid")
    unique_winner = (
        recovery_values
        == recovery_values.max(dim=-1, keepdim=True).values
    ).sum(dim=-1) == 1
    if not bool(unique_winner.any().item()):
        return policy_q_values.sum() * 0.0
    target_probabilities = torch.softmax(
        recovery_values / float(temperature), dim=-1
    )
    policy_log_probabilities = torch.log_softmax(
        policy_q_values / float(temperature), dim=-1
    )
    target_log_probabilities = target_probabilities.clamp_min(
        torch.finfo(target_probabilities.dtype).tiny
    ).log()
    row_losses = (
        target_probabilities
        * (target_log_probabilities - policy_log_probabilities)
    ).sum(dim=-1)
    return row_losses[unique_winner].mean()


def recovery_action_margin(
    policy_q_values: torch.Tensor,
    recovery_values: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Require the unique best recovery action to outrank alternatives."""
    if (
        policy_q_values.shape != recovery_values.shape
        or policy_q_values.shape != (len(RECOVERY_ACTIONS),)
        or not torch.is_floating_point(policy_q_values)
        or not torch.is_floating_point(recovery_values)
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("recovery action margin contract is invalid")
    maximum = recovery_values.max()
    winners = recovery_values == maximum
    if int(winners.sum().item()) != 1:
        return policy_q_values.sum() * 0.0
    winner_index = int(winners.nonzero(as_tuple=False)[0].item())
    alternatives = torch.cat((
        policy_q_values[:winner_index],
        policy_q_values[winner_index + 1:],
    ))
    return torch.relu(
        float(margin) + alternatives.max() - policy_q_values[winner_index]
    )


def recovery_action_values(
    *,
    observation: np.ndarray,
    branches: Mapping[Action, RecoveryBranchResult],
    start_pnl: float,
    recovery_success_pnl: float,
    source_role: str,
    source_identity_sha256: str,
    recurrent_observations: np.ndarray | None = None,
    recurrent_resets: tuple[bool, ...] | None = None,
    anchor_action: Action | None = None,
    anchor_economic_success: bool | None = None,
) -> RecoveryValueTarget:
    """Convert three authenticated counterfactual branches to bounded utility."""
    if source_role != "training":
        raise ValueError("recovery values require the training role")
    if _SHA256.fullmatch(source_identity_sha256) is None:
        raise ValueError("recovery source identity is invalid")
    if (
        isinstance(start_pnl, bool)
        or isinstance(recovery_success_pnl, bool)
        or not math.isfinite(float(start_pnl))
        or not math.isfinite(float(recovery_success_pnl))
        or float(start_pnl) >= float(recovery_success_pnl)
    ):
        raise ValueError("recovery value range is invalid")
    if set(branches) != set(RECOVERY_ACTIONS) or not all(
        isinstance(branches[action], RecoveryBranchResult)
        for action in RECOVERY_ACTIONS
    ):
        raise ValueError("recovery branches must cover WAIT, Long, and Short")

    distance = float(recovery_success_pnl) - float(start_pnl)
    values: list[float] = []
    for action in RECOVERY_ACTIONS:
        branch = branches[action]
        if branch.outcome == "blow":
            value = -1.0
        elif branch.recovered:
            if branch.terminal_pnl < float(recovery_success_pnl):
                # A branch may record a later diagnostic relapse, but the
                # recovery action's credit ended when it first reached the
                # breakeven handoff. Do not assign V21's later loss backward.
                if branch.minimum_post_recovery_pnl is None:
                    raise ValueError("recovered branch did not reach recovery PnL")
            value = 1.0
        else:
            value = float(np.clip(
                (branch.terminal_pnl - float(start_pnl)) / distance,
                -1.0,
                1.0,
            ))
        values.append(value)
    return RecoveryValueTarget(
        observation=observation,
        action_values=tuple(values),
        source_identity_sha256=source_identity_sha256,
        recurrent_observations=recurrent_observations,
        recurrent_resets=recurrent_resets,
        anchor_action=anchor_action,
        anchor_economic_success=anchor_economic_success,
    )


def select_recovery_target_prefix(
    transitions: Sequence[Transition],
    *,
    recovery_succeeded: bool,
) -> tuple[Transition, ...] | None:
    """Select the earliest matching recovery decision and its causal prefix."""
    if type(recovery_succeeded) is not bool:
        raise ValueError("recovery target preference is invalid")
    rows = tuple(transitions)
    segment_start = 0
    for index in range(1, len(rows)):
        if rows[index].recovery_active and not rows[index - 1].recovery_active:
            segment_start = index
    candidates = tuple(
        (index, transition)
        for index, transition in enumerate(rows)
        if index >= segment_start
        and transition.recovery_active
        and set(transition.valid_actions) == set(RECOVERY_ACTIONS)
        and transition.action in {
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
        and transition.paired_a_plus_side is transition.action
        and type(transition.paired_a_plus_economic_win) is bool
    )
    preferred = tuple(
        item
        for item in candidates
        if item[1].paired_a_plus_economic_win is recovery_succeeded
    )
    fallback = tuple(
        (index, transition)
        for index, transition in enumerate(rows)
        if index >= segment_start
        and transition.recovery_active
        and set(transition.valid_actions) == set(RECOVERY_ACTIONS)
    )
    pool = preferred or candidates or fallback
    if not pool:
        return None

    anchor_index, _ = pool[0]
    return rows[: anchor_index + 1]


def build_recovery_value_target(
    recovery_policy: RecoveryPolicy,
    environment: RecoveryEnvironment,
    *,
    normal_policy: RecoveryPolicy,
    reset_options: Mapping[str, object],
    recurrent_horizon: int,
    start_pnl: float,
    recovery_success_pnl: float,
    source_role: str,
    source_identity_sha256: str,
    causal_prefix: Sequence[Transition] | None = None,
) -> RecoveryValueTarget:
    """Roll out all native flat actions from one authenticated recovery state."""
    if (
        isinstance(recurrent_horizon, bool)
        or not isinstance(recurrent_horizon, int)
        or recurrent_horizon < 1
        or not {"ticker", "start", "challenge_start_state"} <= set(reset_options)
    ):
        raise ValueError("recovery rollout contract is invalid")
    prefix = tuple(causal_prefix or ())
    if prefix and (
        not all(isinstance(transition, Transition) for transition in prefix)
        or not prefix[-1].recovery_active
        or set(prefix[-1].valid_actions) != set(RECOVERY_ACTIONS)
    ):
        raise ValueError("recovery causal prefix is invalid")
    recurrent_prefix = prefix
    if prefix:
        reset_indices = tuple(
            index
            for index, transition in enumerate(prefix)
            if transition.recurrent_reset
        )
        if not reset_indices:
            raise ValueError("recovery causal prefix has no recurrent boundary")
        recurrent_prefix = prefix[reset_indices[-1]:]
    recurrent_observations = (
        None
        if not recurrent_prefix
        else np.stack(
            [transition.observation for transition in recurrent_prefix]
        ).astype(np.float32, copy=False)
    )
    recurrent_resets = (
        None
        if not recurrent_prefix
        else tuple(
            bool(transition.recurrent_reset)
            for transition in recurrent_prefix
        )
    )
    branches: dict[Action, RecoveryBranchResult] = {}
    shared_observation: np.ndarray | None = None
    shared_origin: tuple[object, object] | None = None
    for forced_action in RECOVERY_ACTIONS:
        handoff = RecoveryHandoffPolicy(
            recovery_policy,
            normal_policy=normal_policy,
        )
        observation, reset_info = environment.reset(options=dict(reset_options))
        observation = np.asarray(observation, np.float32)
        origin = (reset_info.get("ticker"), reset_info.get("start"))
        valid = tuple(Action(value) for value in reset_info.get("valid_actions", ()))
        anchor_pnl = float(
            reset_info.get("realized_pnl", reset_info.get("equity_pnl", start_pnl))
        )
        for index, transition in enumerate(prefix):
            if not np.array_equal(observation, transition.observation):
                raise ValueError("recovery causal prefix observation drifted")
            valid = tuple(Action(value) for value in transition.valid_actions)
            if transition.recurrent_reset:
                handoff.reset()
            handoff.select_action(
                observation,
                valid_actions=valid,
                realized_pnl=anchor_pnl,
                recovery_epsilon=0.0,
            )
            if index == len(prefix) - 1:
                break
            next_observation, _, terminated, truncated, info = environment.step(
                transition.action
            )
            if terminated or truncated:
                raise ValueError("recovery causal prefix ended before its anchor")
            if not np.array_equal(next_observation, transition.next_observation):
                raise ValueError("recovery causal prefix transition drifted")
            observation = np.asarray(next_observation, np.float32)
            anchor_pnl = float(
                info.get("realized_pnl", info.get("equity_pnl", anchor_pnl))
            )
        if shared_observation is None:
            shared_observation = observation.copy()
            shared_origin = origin
        elif not np.array_equal(observation, shared_observation) or origin != shared_origin:
            raise ValueError("recovery branches do not share one causal state")
        if forced_action not in valid:
            raise ValueError("forced recovery action is not executable")

        # Advance the frozen recurrent state on the shared observation, but
        # discard its choice because only the first action is counterfactual.
        if not prefix:
            handoff.select_action(
                observation,
                valid_actions=valid,
                realized_pnl=anchor_pnl,
                recovery_epsilon=0.0,
            )
        next_observation, _, terminated, truncated, info = environment.step(
            forced_action
        )
        if truncated:
            raise ValueError("recovery target rollout cannot be truncated")
        step_index = 1
        maximum_pnl = float(info.get("equity_pnl", anchor_pnl))
        while True:
            outcome = info.get("outcome")
            realized_pnl = float(info.get("realized_pnl", info.get("equity_pnl")))
            maximum_pnl = max(maximum_pnl, float(info.get("equity_pnl", realized_pnl)))
            blown = terminated and outcome == "blow"
            recovered = bool(
                not blown and realized_pnl >= float(recovery_success_pnl)
            )
            # Recovery owns decisions only while realized PnL is negative.
            # Stop its counterfactual return at the breakeven handoff so a
            # later frozen-V21 loss cannot relabel the recovery action.
            if recovered or terminated:
                if outcome is not None and outcome not in {"pass", "blow", "timeout"}:
                    raise ValueError("recovery rollout produced an invalid outcome")
                branches[forced_action] = RecoveryBranchResult(
                    outcome="timeout" if outcome is None else str(outcome),
                    terminal_pnl=realized_pnl,
                    recovered=recovered,
                    maximum_pnl=maximum_pnl,
                    retained=recovered,
                    minimum_post_recovery_pnl=(
                        realized_pnl if recovered else None
                    ),
                )
                break
            if step_index % recurrent_horizon == 0:
                handoff.reset()
            valid = tuple(Action(value) for value in info.get("valid_actions", ()))
            if not valid:
                raise ValueError("recovery rollout has no valid continuation action")
            action, _, _ = handoff.select_action(
                np.asarray(next_observation, np.float32),
                valid_actions=valid,
                realized_pnl=realized_pnl,
                recovery_epsilon=0.0,
            )
            next_observation, _, terminated, truncated, info = environment.step(action)
            if truncated:
                raise ValueError("recovery target rollout cannot be truncated")
            step_index += 1
    assert shared_observation is not None
    anchor_action: Action | None = None
    anchor_economic_success: bool | None = None
    if (
        prefix
        and prefix[-1].action in {
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        }
        and type(prefix[-1].paired_a_plus_economic_win) is bool
    ):
        anchor_action = prefix[-1].action
        anchor_economic_success = prefix[-1].paired_a_plus_economic_win
    return recovery_action_values(
        observation=shared_observation,
        branches=branches,
        start_pnl=anchor_pnl if prefix else start_pnl,
        recovery_success_pnl=recovery_success_pnl,
        source_role=source_role,
        source_identity_sha256=source_identity_sha256,
        recurrent_observations=recurrent_observations,
        recurrent_resets=recurrent_resets,
        anchor_action=anchor_action,
        anchor_economic_success=anchor_economic_success,
    )


__all__ = [
    "RECOVERY_ACTIONS",
    "RecoveryTargetAudit",
    "RecoveryBranchResult",
    "RecoveryValueStore",
    "RecoveryValueTarget",
    "build_recovery_value_target",
    "audit_recovery_action_values",
    "recovery_action_values",
    "recovery_action_margin",
    "recovery_value_kl",
    "select_recovery_target_prefix",
]

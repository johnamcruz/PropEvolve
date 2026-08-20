"""Training-only full-action recovery values for the native policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Mapping, Protocol

import numpy as np
import torch

from .decision import Action


RECOVERY_ACTIONS = (
    Action.WAIT,
    Action.ENTER_LONG_1,
    Action.ENTER_SHORT_1,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RecoveryBranchResult:
    """Terminal economics after forcing one first action."""

    outcome: str
    terminal_pnl: float
    recovered: bool
    maximum_pnl: float | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"pass", "blow", "timeout"}:
            raise ValueError("recovery branch outcome is invalid")
        if type(self.recovered) is not bool or not math.isfinite(self.terminal_pnl):
            raise ValueError("recovery branch economics are invalid")
        if self.maximum_pnl is not None and not math.isfinite(self.maximum_pnl):
            raise ValueError("recovery branch maximum PnL is invalid")
        if self.outcome == "blow" and self.recovered:
            raise ValueError("a blown recovery branch cannot be recovered")


@dataclass(frozen=True)
class RecoveryValueTarget:
    """One same-state WAIT/Long/Short economic supervision row."""

    observation: np.ndarray
    action_values: tuple[float, float, float]
    source_identity_sha256: str

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32)
        values = tuple(float(value) for value in self.action_values)
        if (
            observation.ndim != 1
            or observation.size == 0
            or not np.isfinite(observation).all()
            or len(values) != len(RECOVERY_ACTIONS)
            or not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)
            or _SHA256.fullmatch(self.source_identity_sha256) is None
        ):
            raise ValueError("recovery value target is invalid")
        observation = observation.copy()
        observation.setflags(write=False)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "action_values", values)

    @property
    def best_action(self) -> Action:
        return RECOVERY_ACTIONS[int(np.argmax(np.asarray(self.action_values)))]

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"propevolve-recovery-value-target-v1\0")
        digest.update(self.source_identity_sha256.encode("ascii"))
        digest.update(str(self.observation.shape).encode("ascii"))
        digest.update(self.observation.tobytes(order="C"))
        digest.update(np.asarray(self.action_values, np.float64).tobytes(order="C"))
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

    def __len__(self) -> int:
        return len(self._targets)

    def add(self, target: RecoveryValueTarget) -> None:
        if not isinstance(target, RecoveryValueTarget):
            raise TypeError("recovery store accepts only RecoveryValueTarget")
        self._targets.append(target)
        if len(self._targets) > self.capacity:
            del self._targets[0]

    def sample(self) -> RecoveryValueTarget:
        if not self._targets:
            raise ValueError("recovery value store is empty")
        return self._targets[int(self._rng.integers(len(self._targets)))]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "propevolve_recovery_value_store_v1",
            "capacity": self.capacity,
            "seed": self.seed,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "targets": [
                {
                    "observation": target.observation.copy(),
                    "action_values": target.action_values,
                    "source_identity_sha256": target.source_identity_sha256,
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
            )
            if payload.get("identity_sha256") != target.identity_sha256:
                raise ValueError("recovery value store target identity drifted")
            targets.append(target)
        if len(targets) > self.capacity:
            raise ValueError("recovery value store exceeds capacity")
        self._targets = targets
        self._rng.bit_generator.state = copy.deepcopy(dict(state["rng_state"]))


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
    target_probabilities = torch.softmax(
        recovery_values / float(temperature), dim=-1
    )
    policy_log_probabilities = torch.log_softmax(
        policy_q_values / float(temperature), dim=-1
    )
    target_log_probabilities = target_probabilities.clamp_min(
        torch.finfo(target_probabilities.dtype).tiny
    ).log()
    return (
        target_probabilities
        * (target_log_probabilities - policy_log_probabilities)
    ).sum(dim=-1).mean()


def recovery_action_values(
    *,
    observation: np.ndarray,
    branches: Mapping[Action, RecoveryBranchResult],
    start_pnl: float,
    recovery_success_pnl: float,
    source_role: str,
    source_identity_sha256: str,
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
        if branch.recovered:
            if branch.terminal_pnl < float(recovery_success_pnl):
                raise ValueError("recovered branch did not reach recovery PnL")
            value = 1.0
        elif branch.outcome == "blow":
            value = -1.0
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
    )


def build_recovery_value_target(
    policy: RecoveryPolicy,
    environment: RecoveryEnvironment,
    *,
    reset_options: Mapping[str, object],
    recurrent_horizon: int,
    start_pnl: float,
    recovery_success_pnl: float,
    source_role: str,
    source_identity_sha256: str,
) -> RecoveryValueTarget:
    """Roll out all native flat actions from one authenticated recovery state."""
    if (
        isinstance(recurrent_horizon, bool)
        or not isinstance(recurrent_horizon, int)
        or recurrent_horizon < 1
        or not {"ticker", "start", "challenge_start_state"} <= set(reset_options)
    ):
        raise ValueError("recovery rollout contract is invalid")
    branches: dict[Action, RecoveryBranchResult] = {}
    shared_observation: np.ndarray | None = None
    shared_origin: tuple[object, object] | None = None
    for forced_action in RECOVERY_ACTIONS:
        observation, reset_info = environment.reset(options=dict(reset_options))
        observation = np.asarray(observation, np.float32)
        origin = (reset_info.get("ticker"), reset_info.get("start"))
        if shared_observation is None:
            shared_observation = observation.copy()
            shared_origin = origin
        elif not np.array_equal(observation, shared_observation) or origin != shared_origin:
            raise ValueError("recovery branches do not share one causal state")
        valid = tuple(Action(value) for value in reset_info.get("valid_actions", ()))
        if forced_action not in valid:
            raise ValueError("forced recovery action is not executable")

        # Advance the frozen recurrent state on the shared observation, but
        # discard its choice because only the first action is counterfactual.
        _, hidden, _ = policy.select_action(
            observation,
            hidden=None,
            valid_actions=valid,
            epsilon=0.0,
        )
        next_observation, _, terminated, truncated, info = environment.step(
            forced_action
        )
        if truncated:
            raise ValueError("recovery target rollout cannot be truncated")
        step_index = 1
        maximum_pnl = float(info.get("equity_pnl", start_pnl))
        while True:
            outcome = info.get("outcome")
            realized_pnl = float(info.get("realized_pnl", info.get("equity_pnl")))
            maximum_pnl = max(maximum_pnl, float(info.get("equity_pnl", realized_pnl)))
            blown = terminated and outcome == "blow"
            recovered = not blown and realized_pnl >= float(recovery_success_pnl)
            if recovered or terminated:
                if outcome is not None and outcome not in {"pass", "blow", "timeout"}:
                    raise ValueError("recovery rollout produced an invalid outcome")
                branches[forced_action] = RecoveryBranchResult(
                    outcome="timeout" if outcome is None else str(outcome),
                    terminal_pnl=realized_pnl,
                    recovered=recovered,
                    maximum_pnl=maximum_pnl,
                )
                break
            if step_index % recurrent_horizon == 0:
                hidden = None
            valid = tuple(Action(value) for value in info.get("valid_actions", ()))
            if not valid:
                raise ValueError("recovery rollout has no valid continuation action")
            action, hidden, _ = policy.select_action(
                np.asarray(next_observation, np.float32),
                hidden=hidden,
                valid_actions=valid,
                epsilon=0.0,
            )
            next_observation, _, terminated, truncated, info = environment.step(action)
            if truncated:
                raise ValueError("recovery target rollout cannot be truncated")
            step_index += 1
    assert shared_observation is not None
    return recovery_action_values(
        observation=shared_observation,
        branches=branches,
        start_pnl=start_pnl,
        recovery_success_pnl=recovery_success_pnl,
        source_role=source_role,
        source_identity_sha256=source_identity_sha256,
    )


__all__ = [
    "RECOVERY_ACTIONS",
    "RecoveryBranchResult",
    "RecoveryValueStore",
    "RecoveryValueTarget",
    "build_recovery_value_target",
    "recovery_action_values",
    "recovery_value_kl",
]

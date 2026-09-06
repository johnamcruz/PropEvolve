"""Same-state economic labels, computed only by the production simulator.

These are realized outcomes under a declared continuation policy, not oracle
pass probabilities. Reconstructing a bounded causal prefix through reset/step
avoids an incomplete snapshot implementation and shares immutable market data.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..decision import Action
from ..environment import HistoricalChallengeEnv


def label_entry_opportunity(
    market, *, decision: int, role_end: int, horizon: int, risk_dollars: float,
    point_value: float, round_trip_fee: float, target_r: float, stop_r: float,
) -> tuple[bool, bool] | None:
    """Reuse the accepted next-open barrier semantics without changing C51.

None is censored, never a failed setup. True means net target before adverse;
False means adverse first or target not reached within the complete horizon.
These are setup labels, not account-aware action prescriptions.
    """
    from ..entry_supervision import _target_before_adverse

    if (type(decision) is not int or type(role_end) is not int
            or type(horizon) is not int or decision < 0 or horizon < 1
            or not decision < role_end <= len(market.close)):
        raise ValueError("invalid label row or temporal role boundary")
    numbers = [risk_dollars, point_value, round_trip_fee, target_r, stop_r]
    if (not np.isfinite(numbers).all() or min(risk_dollars, point_value, target_r, stop_r) <= 0
            or round_trip_fee < 0 or round_trip_fee >= risk_dollars * stop_r):
        raise ValueError("invalid economic label contract")
    if decision + 1 + horizon > role_end:
        return None
    return tuple(_target_before_adverse(
        market.open, market.high, market.low, decision=decision, role_end=role_end,
        side=side, risk_dollars=risk_dollars, point_value=point_value,
        round_trip_fee=round_trip_fee, target_r=target_r, adverse_r=stop_r,
        horizon=horizon,
    ) for side in ("long", "short"))


@dataclass(frozen=True)
class ActionOutcome:
    outcome: str
    terminal_pnl: float
    reward_to_go: float
    minimum_mll_headroom: float
    steps: int
    outcome_end_ns: int


@dataclass(frozen=True)
class ActionLabels:
    observation: np.ndarray
    outcomes: Mapping[Action, ActionOutcome]


def label_actions(
    environment: HistoricalChallengeEnv, *, reset_options: dict,
    prefix: Sequence[Action], continuation_factory: Callable,
    max_steps: int,
) -> ActionLabels:
    """Force each legal action from one reconstructed state, then finish it.

The factory must return a fresh causal deterministic policy per branch. It is
advanced on the prefix for memory parity; its suggested actions are ignored
there. Explicit ticker/start prevent accidental comparison of different states.
    """
    if "ticker" not in reset_options or "start" not in reset_options:
        raise ValueError("labels require explicit ticker and start")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    def reconstruct():
        branch = HistoricalChallengeEnv(
            environment.markets, tick_values=environment.tick_values,
            round_trip_fees=environment.round_trip_fees, spec=environment.spec,
            observation_spec=environment._assembler.trade_management, seed=0,
        )
        policy = continuation_factory()
        observation, info = branch.reset(options=dict(reset_options))
        for action in prefix:
            policy(observation, info)
            observation, _, terminated, truncated, info = branch.step(action)
            if terminated or truncated:
                raise ValueError("prefix reaches a terminal state")
        return branch, policy, observation, info

    _, _, anchor, initial = reconstruct()
    outcomes = {}
    for action in initial["valid_actions"]:
        branch, policy, observation, info = reconstruct()
        if not np.array_equal(anchor, observation):
            raise ValueError("counterfactual anchor mismatch")
        total = 0.0
        for step in range(max_steps):
            suggested = policy(observation, info)
            observation, reward, terminated, truncated, info = branch.step(
                action if step == 0 else suggested
            )
            total += reward
            if terminated or truncated:
                if info["outcome"] not in {"pass", "blow", "timeout"}:
                    raise ValueError("invalid terminal outcome")
                outcomes[Action(action)] = ActionOutcome(
                    outcome=info["outcome"], terminal_pnl=float(info["realized_pnl"]),
                    reward_to_go=float(total),
                    minimum_mll_headroom=float(info["minimum_mll_headroom"]),
                    steps=step + 1,
                    outcome_end_ns=int(np.datetime64(info["timestamp"], "ns").astype(np.int64)),
                )
                break
        else:
            raise ValueError("incomplete counterfactual rollout; increase max_steps")
    return ActionLabels(anchor.copy(), outcomes)

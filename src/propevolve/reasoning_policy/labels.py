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


def label_future_excursions(market, *, decision, role_end, horizon, risk_dollars,
                            point_value, round_trip_fee):
    """Full-horizon market targets, not claims of an executed stop-managed trade.

Entry reference is the next open. Gross extrema use the declared dollar risk
budget as denominator; terminal net R includes one round-trip fee. The path may
cross a stop before its MFE: target-before-stop remains a separate label.
    """
    if (type(decision) is not int or type(horizon) is not int or type(role_end) is not int
            or decision < 0 or horizon < 1 or not decision < role_end <= len(market.close)):
        raise ValueError("invalid excursion horizon")
    if (not np.isfinite([risk_dollars, point_value, round_trip_fee]).all()
            or risk_dollars <= 0 or point_value <= 0 or round_trip_fee < 0):
        raise ValueError("invalid excursion economics")
    first, end = decision + 1, decision + 1 + horizon
    if end > role_end:
        return None
    entry = float(market.open[first])
    high, low, close = float(np.max(market.high[first:end])), float(np.min(market.low[first:end])), float(market.close[end - 1])
    if not np.isfinite([entry, high, low, close]).all():
        raise ValueError("nonfinite excursion source")
    scale = point_value / risk_dollars
    return {side: {
        "mfe_r_gross": max(0.0, favorable) * scale,
        "mae_r_gross": max(0.0, adverse) * scale,
        "terminal_r_net": (sign * (close - entry) * point_value - round_trip_fee) / risk_dollars,
    } for side, sign, favorable, adverse in (
        ("long", 1, high - entry, entry - low), ("short", -1, entry - low, high - entry))}


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


def label_market_actions(
    market, *, decision, role_end, observation, risk_dollars, point_value,
    round_trip_fee, minimum_mll_headroom, horizon, target_rs, stop_r, utilities,
):
    """Scratch-policy flat action ordering from one future economic barrier.

    Future paths are labels only.  They never enter the causal prompt.  This is
    an entry warm start; the unchanged challenge environment teaches sequential
    Hold/Close and account-aware behavior during RL.
    """
    required = {"winner", "failure", "wait", "missed_opportunity", "conflict_margin"}
    if set(utilities) != required or not np.isfinite(list(utilities.values())).all():
        raise ValueError("invalid market action utility contract")
    if not (utilities["winner"] > utilities["wait"] > utilities["failure"]
            and utilities["conflict_margin"] > 0
            and utilities["wait"] > utilities["missed_opportunity"] > utilities["failure"]):
        raise ValueError("market action utilities violate the decision boundary")
    target_rs = tuple(float(value) for value in target_rs)
    if (not target_rs or tuple(sorted(set(target_rs))) != target_rs
            or not np.isfinite(target_rs).all() or target_rs[0] <= 0):
        raise ValueError("target R grid must be finite, positive and increasing")
    grid = {target: label_entry_opportunity(
        market, decision=decision, role_end=role_end, horizon=horizon,
        risk_dollars=risk_dollars, point_value=point_value,
        round_trip_fee=round_trip_fee, target_r=target, stop_r=stop_r,
    ) for target in target_rs}
    excursions = label_future_excursions(
        market, decision=decision, role_end=role_end, horizon=horizon,
        risk_dollars=risk_dollars, point_value=point_value,
        round_trip_fee=round_trip_fee,
    )
    if any(value is None for value in grid.values()) or excursions is None:
        raise ValueError("market action label is censored by its temporal role")
    achieved = [max((target for target, result in grid.items() if result[side]), default=0.0)
                for side in (0, 1)]
    side_values = [utilities["failure"] if target == 0 else
                   utilities["winner"] + target - target_rs[0] for target in achieved]
    if achieved[0] > 0 and achieved[0] == achieved[1]:
        wait_value = max(side_values) + utilities["conflict_margin"]
    elif max(achieved) > 0:
        wait_value = utilities["missed_opportunity"]
    else:
        wait_value = utilities["wait"]
    values = {
        Action.WAIT: wait_value,
        Action.ENTER_LONG_1: side_values[0],
        Action.ENTER_SHORT_1: side_values[1],
    }
    end_ns = int(market.timestamps[decision + horizon].astype("datetime64[ns]").astype(np.int64))
    outcomes = {}
    for action, value in values.items():
        side = "long" if action is Action.ENTER_LONG_1 else "short" if action is Action.ENTER_SHORT_1 else None
        terminal_pnl = 0.0 if side is None else excursions[side]["terminal_r_net"] * risk_dollars
        outcomes[action] = ActionOutcome(
            outcome=("wait" if side is None else
                     f"target_{achieved[0 if side == 'long' else 1]:g}r_before_stop"
                     if achieved[0 if side == "long" else 1] else "failed_target"),
            terminal_pnl=float(terminal_pnl), reward_to_go=float(value),
            minimum_mll_headroom=float(minimum_mll_headroom), steps=horizon,
            outcome_end_ns=end_ns,
        )
    return ActionLabels(np.asarray(observation).copy(), outcomes)


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

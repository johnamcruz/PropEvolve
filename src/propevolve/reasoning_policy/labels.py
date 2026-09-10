"""Same-state economic labels, computed only by the production simulator.

These are realized outcomes under a declared continuation policy, not oracle
pass probabilities. Reconstructing a bounded causal prefix through reset/step
avoids an incomplete snapshot implementation and shares immutable market data.
"""

from __future__ import annotations

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


def label_position_actions(
    market, *, decision: int, role_end: int, entry_index: int, side: Action,
    observation, risk_dollars: float, point_value: float, round_trip_fee: float,
    minimum_mll_headroom: float, horizon: int, stop_r: float,
    minimum_improvement_r: float,
) -> ActionLabels:
    """Rank HOLD versus next-open CLOSE from one causal positioned state.

    The prompt receives only the completed-bar observation, including MFE/MAE
    accumulated so far. Future opens and adverse excursions are label-only. HOLD
    receives the best later executable net R that remains reachable before the
    declared stop; the configurable improvement margin prevents immaterial extra
    exposure from beating CLOSE. This is trade-exit supervision, not a challenge
    pass/blow objective.
    """
    if (type(decision) is not int or type(role_end) is not int
            or type(entry_index) is not int or type(horizon) is not int
            or not 0 <= entry_index <= decision < role_end <= len(market.close)
            or horizon < 2):
        raise ValueError("invalid positioned label timeline")
    try:
        side = Action(side)
    except (TypeError, ValueError) as error:
        raise ValueError("position side must be a Long or Short entry") from error
    if side not in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}:
        raise ValueError("position side must be a Long or Short entry")
    numbers = np.asarray([
        risk_dollars, point_value, round_trip_fee, minimum_mll_headroom,
        stop_r, minimum_improvement_r,
    ], dtype=float)
    if (not np.isfinite(numbers).all() or risk_dollars <= 0 or point_value <= 0
            or round_trip_fee < 0 or minimum_mll_headroom < 0 or stop_r <= 0
            or minimum_improvement_r < 0
            or round_trip_fee >= risk_dollars * stop_r):
        raise ValueError("invalid positioned label economics")
    immediate_exit = decision + 1
    final_exit = min(role_end - 1, decision + horizon)
    if immediate_exit >= role_end or immediate_exit + 1 > final_exit:
        raise ValueError("positioned label is censored by its temporal role")
    entry = float(market.open[entry_index])
    sign = 1.0 if side is Action.ENTER_LONG_1 else -1.0
    fee_r = round_trip_fee / risk_dollars

    def exit_r(index):
        return sign * (float(market.open[index]) - entry) * point_value / risk_dollars - fee_r

    close_r = exit_r(immediate_exit)
    adverse_points = (stop_r * risk_dollars - round_trip_fee) / point_value
    bars = np.arange(immediate_exit, final_exit, dtype=int)
    adverse = (entry - np.asarray(market.low[bars], dtype=float)
               if sign > 0 else np.asarray(market.high[bars], dtype=float) - entry)
    if (not np.isfinite([entry, close_r]).all() or not np.isfinite(adverse).all()
            or adverse_points <= 0):
        raise ValueError("nonfinite positioned label source")
    stop_hits = np.flatnonzero(adverse >= adverse_points)
    first_stop = final_exit if not len(stop_hits) else immediate_exit + int(stop_hits[0])
    reachable_exits = range(immediate_exit + 1, min(final_exit, first_stop) + 1)
    candidates = [(index, exit_r(index)) for index in reachable_exits]
    if candidates:
        hold_end, hold_r = max(candidates, key=lambda item: item[1])
        hold_outcome = "continued_to_better_exit"
    else:
        hold_end, hold_r = first_stop, -float(stop_r)
        hold_outcome = "stopped_before_later_exit"
    if not np.isfinite(hold_r):
        raise ValueError("nonfinite positioned action value")
    outcomes = {
        Action.HOLD: ActionOutcome(
            outcome=hold_outcome,
            terminal_pnl=float(hold_r * risk_dollars),
            reward_to_go=float(hold_r - minimum_improvement_r),
            minimum_mll_headroom=float(minimum_mll_headroom),
            steps=int(hold_end - decision),
            outcome_end_ns=int(market.timestamps[hold_end].astype("datetime64[ns]").astype(np.int64)),
        ),
        Action.CLOSE: ActionOutcome(
            outcome="close_next_open",
            terminal_pnl=float(close_r * risk_dollars),
            reward_to_go=float(close_r),
            minimum_mll_headroom=float(minimum_mll_headroom),
            steps=1,
            outcome_end_ns=int(market.timestamps[immediate_exit].astype("datetime64[ns]").astype(np.int64)),
        ),
    }
    return ActionLabels(np.asarray(observation).copy(), outcomes)


def label_entry_opportunity(
    market, *, decision: int, role_end: int, horizon: int, risk_dollars: float,
    point_value: float, round_trip_fee: float, target_r: float, stop_r: float,
) -> tuple[bool, bool] | None:
    """Apply the reasoning policy's next-open economic barrier semantics.

None is censored, never a failed setup. True means net target before adverse;
False means adverse first or target not reached within the complete horizon.
These are setup labels, not account-aware action prescriptions.
    """
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


def _target_before_adverse(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    decision: int,
    role_end: int,
    side: str,
    risk_dollars: float,
    point_value: float,
    round_trip_fee: float,
    target_r: float,
    adverse_r: float,
    horizon: int,
) -> bool:
    """Return whether a net target is reached before the adverse barrier.

    Entries fill at the next open. Because OHLC bars do not expose intrabar
    ordering, an adverse and favorable touch on the same bar is conservatively
    classified as a failure.
    """
    fill_index = decision + 1
    stop = fill_index + horizon
    if fill_index >= role_end or stop > role_end:
        return False
    favorable_points = (target_r * risk_dollars + round_trip_fee) / point_value
    adverse_points = (adverse_r * risk_dollars - round_trip_fee) / point_value
    if not adverse_points > 0.0:
        raise ValueError("entry supervision adverse distance must be positive")
    entry = float(open_[fill_index])
    for index in range(fill_index, stop):
        if side == "long":
            favorable = float(high[index]) - entry
            adverse = entry - float(low[index])
        elif side == "short":
            favorable = entry - float(low[index])
            adverse = float(high[index]) - entry
        else:
            raise ValueError("side must be long or short")
        if adverse >= adverse_points:
            return False
        if favorable >= favorable_points:
            return True
    return False


def classify_market_action_rows(
    market, *, role_end, risk_dollars, point_value, round_trip_fee,
    horizon, target_rs, stop_r, chunk_size=16384,
):
    """Classify every uncensored flat state with the scalar barrier contract.

    This is the exhaustive-corpus counterpart of ``label_market_actions``.
    It returns Action integer values and -1 for the censored tail. OHLC ties
    remain adverse-first, exactly like the scalar barrier implementation.
    """
    if (type(role_end) is not int or type(horizon) is not int or type(chunk_size) is not int
            or not 1 <= role_end <= len(market.close) or horizon < 1 or chunk_size < 1):
        raise ValueError("invalid full-history label bounds")
    numbers = np.asarray([risk_dollars, point_value, round_trip_fee, stop_r], dtype=float)
    targets = np.asarray(tuple(target_rs), dtype=float)
    if (not np.isfinite(numbers).all() or risk_dollars <= 0 or point_value <= 0
            or round_trip_fee < 0 or stop_r <= 0 or targets.ndim != 1 or not len(targets)
            or not np.isfinite(targets).all() or (targets <= 0).any()
            or not np.all(np.diff(targets) > 0)):
        raise ValueError("invalid full-history economic contract")
    adverse_points = (stop_r * risk_dollars - round_trip_fee) / point_value
    if adverse_points <= 0:
        raise ValueError("entry supervision adverse distance must be positive")
    target_points = (targets * risk_dollars + round_trip_fee) / point_value
    eligible = role_end - horizon
    result = np.full(role_end, -1, dtype=np.int8)
    if eligible <= 0:
        return result
    from numpy.lib.stride_tricks import sliding_window_view
    high_windows = sliding_window_view(np.asarray(market.high[1:role_end]), horizon)
    low_windows = sliding_window_view(np.asarray(market.low[1:role_end]), horizon)
    entries = np.asarray(market.open[1:eligible + 1])
    steps = np.arange(horizon)[None, :]

    for start in range(0, eligible, chunk_size):
        end = min(start + chunk_size, eligible)
        entry = entries[start:end, None]
        high = high_windows[start:end]
        low = low_windows[start:end]
        achieved = []
        for favorable, adverse in ((high - entry, entry - low),
                                    (entry - low, high - entry)):
            adverse_hits = adverse >= adverse_points
            first_adverse = np.argmax(adverse_hits, axis=1)
            first_adverse = np.where(adverse_hits.any(axis=1), first_adverse, horizon)
            before_adverse = steps < first_adverse[:, None]
            maximum = np.max(np.where(before_adverse, favorable, -np.inf), axis=1)
            maximum = np.maximum(maximum, 0.0)
            levels = (maximum[:, None] >= target_points[None, :]).sum(axis=1)
            achieved.append(levels)
        long_levels, short_levels = achieved
        chosen = np.full(end - start, int(Action.WAIT), dtype=np.int8)
        chosen[long_levels > short_levels] = int(Action.ENTER_LONG_1)
        chosen[short_levels > long_levels] = int(Action.ENTER_SHORT_1)
        result[start:end] = chosen
    return result


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

    Future paths are labels only and never enter the causal prompt. This is the
    SFT entry boundary: the smallest configured target (normally 2R) establishes
    a valid setup, while larger achieved targets increase its economic value.
    Positioned HOLD/CLOSE labels are produced separately by
    ``label_position_actions``; challenge pass/blow behavior belongs to RL.
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

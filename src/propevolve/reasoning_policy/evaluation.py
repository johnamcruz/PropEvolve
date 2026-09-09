"""Challenger evaluation with unchanged environment economics and specialist inputs."""

import numpy as np
from ..policy import TradingPolicy, PolicyInput

from .context import RollingContext
from .inputs import observe_context


def evaluate_policy(policy, environment, *, episodes, context_config, sources, max_steps,
                    near_blow_headroom_fraction=None, on_decision=None):
    """Evaluate explicit episode starts, with no teacher-free claim.

Returns economic receipts, not C51-specific action-value diagnostics. The caller
must use exactly the same starts/environment settings for the baseline. A
resource cap is an error, never a fabricated economic timeout.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if near_blow_headroom_fraction is not None and not 0 <= near_blow_headroom_fraction <= 1:
        raise ValueError("near-blow diagnostic fraction must be in [0, 1]")
    receipts = []
    shared_policy = isinstance(policy, TradingPolicy)
    uses_specialists = getattr(policy, "requires_specialists", True)
    needs_context = policy.requires_context if shared_policy else True
    if needs_context and (context_config.input_mode == "specialists") != uses_specialists:
        raise ValueError("evaluation policy and context input mode differ")
    for options in episodes:
        if "ticker" not in options or "start" not in options:
            raise ValueError("evaluation requires explicit ticker/start")
        observation, info = environment.reset(options=options)
        market = environment.markets[options["ticker"]]
        context = RollingContext(context_config) if needs_context else None
        if shared_policy:
            policy.reset()
        row = options["start"]
        total_reward = 0.0
        action_counts = {}
        for step in range(max_steps):
            if context is not None:
                observe_context(context, environment, observation, ticker=options["ticker"], row=row, sources=sources)
            if shared_policy:
                decision = policy.decide(PolicyInput(observation, tuple(info["valid_actions"]),
                    None if context is None else context.snapshot()))
                action, scores, score_type = decision.action, decision.scores, decision.score_type
            else:
                action, scores = policy.decide(context.snapshot(), info["valid_actions"])
                score_type = "log_likelihood"
            if action not in info["valid_actions"]:
                raise ValueError("policy requested an illegal action")
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if on_decision is not None:
                on_decision({"ticker": options["ticker"], "episode_start": options["start"],
                    "decision_index": row, "timestamp": str(market.timestamps[row]),
                    "legal_actions": [item.name for item in info["valid_actions"]],
                    "requested_action": action.name, "action_scores": scores,
                    "score_type": score_type,
                    **({"action_log_likelihoods": scores} if score_type == "log_likelihood" else {}),
                    "account": environment.causal_trade_context()})
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            row = int(info["fill_index"])
            if terminated or truncated:
                receipts.append({
                    **info, "reward": total_reward, "steps": step + 1,
                    "start": options["start"], "specialist_inputs_used": uses_specialists,
                    "teacher_free": not uses_specialists,
                    "action_counts": action_counts,
                    "near_blow": None if near_blow_headroom_fraction is None else (
                        info["minimum_mll_headroom"] <= near_blow_headroom_fraction * environment.spec.max_loss),
                })
                break
        else:
            raise ValueError("incomplete evaluation episode")
    if not receipts:
        raise ValueError("evaluation requires episodes")
    return {
        "episodes": receipts,
        "pass_rate": sum(row["outcome"] == "pass" for row in receipts) / len(receipts),
        "blow_rate": sum(row["outcome"] == "blow" for row in receipts) / len(receipts),
        "timeout_rate": sum(row["outcome"] == "timeout" for row in receipts) / len(receipts),
        "positive_timeout_rate": sum(row["outcome"] == "timeout" and row["realized_pnl"] > 0 for row in receipts) / len(receipts),
        "near_blow_rate": None if near_blow_headroom_fraction is None else sum(row["near_blow"] for row in receipts) / len(receipts),
        "near_blow_headroom_fraction": near_blow_headroom_fraction,
        "mean_terminal_pnl": float(np.mean([row["realized_pnl"] for row in receipts])),
        "median_terminal_pnl": float(np.median([row["realized_pnl"] for row in receipts])),
        "mean_trades": float(np.mean([row["trade_count"] for row in receipts])),
        "action_counts": {name: sum(row["action_counts"].get(name, 0) for row in receipts)
                          for name in {key for row in receipts for key in row["action_counts"]}},
        "teacher_free": not uses_specialists,
    }

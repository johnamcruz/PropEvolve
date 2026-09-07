"""Challenger evaluation with unchanged environment economics and specialist inputs."""

import numpy as np

from .context import RollingContext
from .inputs import specialist_account_fields


def evaluate_policy(policy, environment, *, episodes, context_config, sources, max_steps):
    """Evaluate explicit episode starts, with no teacher-free claim.

Returns economic receipts, not C51-specific action-value diagnostics. The caller
must use exactly the same starts/environment settings for the baseline. A
resource cap is an error, never a fabricated economic timeout.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    receipts = []
    for options in episodes:
        if "ticker" not in options or "start" not in options:
            raise ValueError("evaluation requires explicit ticker/start")
        observation, info = environment.reset(options=options)
        market = environment.markets[options["ticker"]]
        context = RollingContext(context_config)
        row = options["start"]
        total_reward = 0.0
        for step in range(max_steps):
            fields = specialist_account_fields(
                observation, embedding_dim=market.embeddings.shape[1],
                ticker=options["ticker"], row=row, sources=sources,
            )
            fields.update(environment.causal_trade_context())
            if not set(context_config.fields).issubset(fields):
                raise ValueError("configured input unavailable during evaluation")
            context.append(
                int(market.timestamps[row].astype("datetime64[ns]").astype(np.int64)),
                {key: fields[key] for key in context_config.fields},
            )
            action, scores = policy.decide(context.snapshot(), info["valid_actions"])
            if action not in info["valid_actions"]:
                raise ValueError("policy requested an illegal action")
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            row = int(info["fill_index"])
            if terminated or truncated:
                receipts.append({
                    **info, "reward": total_reward, "steps": step + 1,
                    "start": options["start"], "specialist_inputs_used": True,
                    "teacher_free": False,
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
        "mean_terminal_pnl": float(np.mean([row["realized_pnl"] for row in receipts])),
        "teacher_free": False,
    }

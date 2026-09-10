"""Bounded on-policy adapter learning from complete simulator episodes.

Actor-only clipped policy gradient, not C51 and not a claim of full PPO/GRPO.
Repeated identical episode starts supply independent sampled trajectories and
a leave-one-out return baseline. No critic or second large reference model.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..decision import Action
from .context import RollingContext
from .dataset import context_messages, embedding_payload
from .inputs import observe_context


# The trade-mastery adapter deliberately excludes challenge objectives during
# SFT.  RL must receive the complete causal challenge state so it can learn how
# to use that trading skill inside the prop rules.
CHALLENGE_MASTERY_FIELDS = (
    "account.realized_pnl_norm",
    "account.equity_pnl_norm",
    "account.peak_equity_pnl_norm",
    "account.mll_headroom_norm",
    "account.drawdown_norm",
    "account.position_side",
    "account.position_size_norm",
    "account.unrealized_pnl_norm",
    "account.session_remaining",
    "account.challenge_remaining",
    "account.point_value_norm",
    "account.round_trip_fee_norm",
    "trade.open",
    "trade.position_side",
    "trade.risk_available",
    "trade.mfe_r_so_far",
    "trade.mae_r_so_far",
    "trade.current_r",
    "trade.giveback_r",
    "trade.hold_bars",
    "challenge.profit_target_dollars",
    "challenge.max_loss_dollars",
    "challenge.realized_pnl_dollars",
    "challenge.equity_pnl_dollars",
    "challenge.target_remaining_dollars",
    "challenge.mll_floor_dollars",
    "challenge.headroom_dollars",
)


def require_challenge_mastery_context(context_config):
    """Fail before RL when the policy cannot observe the prop objective."""
    fields = tuple(getattr(context_config, "fields", ()))
    missing = tuple(field for field in CHALLENGE_MASTERY_FIELDS if field not in fields)
    if missing:
        raise ValueError(
            "challenge-mastery context is missing: " + ", ".join(missing)
        )
    if getattr(context_config, "input_mode", None) != "embeddings":
        raise ValueError("challenge-mastery RL requires teacher-free embeddings")
    return context_config


@dataclass(frozen=True)
class RLDecision:
    messages: list
    actions: tuple[str, ...]
    selected: int
    old_log_probs: tuple[float, ...]
    reward: float
    market_context: dict | None = None


def rollout(policy, environment, *, options, context_config, sources, rng, max_steps):
    """Store CPU causal prompts and detached sampling probabilities only."""
    observation, info = environment.reset(options=options)
    market = environment.markets[options["ticker"]]
    context = RollingContext(context_config)
    row = options["start"]
    decisions = []
    for _ in range(max_steps):
        observe_context(context, environment, observation, ticker=options["ticker"], row=row, sources=sources)
        actions = tuple(sorted((Action(a) for a in info["valid_actions"]), key=int))
        names = tuple(action.name for action in actions)
        messages = context_messages(context.snapshot(), actions)
        market_context = embedding_payload(context.snapshot()) or None
        scores = policy.completion_scores(messages, names, **({"market_context": market_context} if market_context else {}))
        logits = np.asarray([scores[name] for name in names], dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite rollout logits")
        logits -= logits.max()
        log_probs = logits - np.log(np.exp(logits).sum())
        selected = int(rng.choice(len(actions), p=np.exp(log_probs)))
        observation, reward, terminated, truncated, info = environment.step(actions[selected])
        decisions.append(RLDecision(messages, names, selected, tuple(log_probs), float(reward), market_context))
        row = int(info["fill_index"])
        if terminated or truncated:
            if info["outcome"] not in {"pass", "blow", "timeout"}:
                raise ValueError("invalid RL episode outcome")
            return decisions, info
    raise ValueError("RL rollout incomplete; resource cap is not an economic timeout")


def training_rows(trajectories, *, advantage_scale):
    """Complete undiscounted return-to-go minus independent same-start baseline."""
    if len(trajectories) < 2 or not np.isfinite(advantage_scale) or advantage_scale <= 0:
        raise ValueError("RL needs at least two complete trajectories and positive scale")
    totals = [sum(row.reward for row in trajectory) for trajectory in trajectories]
    if not np.isfinite(totals).all() or any(not x for x in trajectories):
        raise ValueError("invalid episode rewards")
    rows = []
    for i, trajectory in enumerate(trajectories):
        baseline = sum(value for j, value in enumerate(totals) if j != i) / (len(totals) - 1)
        remaining = 0.0
        episode_rows = []
        for decision in reversed(trajectory):
            remaining += decision.reward
            episode_rows.append((decision, (remaining - baseline) / advantage_scale))
        rows.extend(reversed(episode_rows))
    return rows


def clipped_action_loss(log_probs, old_log_probs, selected, advantage, *, clip_epsilon,
                        kl_weight, entropy_weight, xp):
    """Same loss for CPU reference tests and differentiable MLX production."""
    ratio = xp.exp(log_probs[selected] - old_log_probs[selected])
    surrogate = xp.minimum(ratio * advantage,
                           xp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    kl = (xp.exp(old_log_probs) * (old_log_probs - log_probs)).sum()
    entropy = -(xp.exp(log_probs) * log_probs).sum()
    return -surrogate + kl_weight * kl - entropy_weight * entropy


def read_rl_config(path):
    from .model_config import read_recipe
    config = read_recipe(path)
    for name in ("groups", "group_size", "max_steps", "epochs", "minibatch_size", "max_update_rows"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["group_size"] < 2:
        raise ValueError("group_size must be at least two")
    for name in ("learning_rate", "advantage_scale", "max_grad_norm", "clip_epsilon"):
        if isinstance(config[name], bool) or not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if config["clip_epsilon"] >= 1:
        raise ValueError("clip_epsilon must be less than one")
    for name in ("kl_weight", "entropy_weight", "weight_decay"):
        if isinstance(config[name], bool) or not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (type(config["checkpoint_every_groups"]) is not int or config["checkpoint_every_groups"] < 1
            or not config["checkpoint_root"]):
        raise ValueError("RL checkpoint cadence and root are required")
    if type(config["checkpoint_keep"]) is not int or config["checkpoint_keep"] < 1:
        raise ValueError("checkpoint_keep must be positive")
    return config


class MLXAdapterLearner:
    """Reuse the policy's actual token scores; update only loaded LoRA tensors."""

    def __init__(self, policy, config):
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        self.policy, self.config = policy, config
        mx.random.seed(config["seed"])
        self.optimizer = optim.AdamW(learning_rate=config["learning_rate"],
                                    weight_decay=config["weight_decay"])
        # Loaded adapters must be the only trainable leaves.
        policy.model.freeze()
        found = 0
        for _, module in policy.model.named_modules():
            keys = [key for key in ("lora_a", "lora_b") if hasattr(module, key)]
            if keys:
                module.unfreeze(keys=keys, recurse=False)
                found += 1
        leaves = tree_flatten(policy.model.trainable_parameters())
        if not found or any(name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"} for name, _ in leaves):
            raise ValueError("RL requires trainable LoRA adapters and a frozen base")
        if policy.input_mode == "embeddings":
            policy.model.market_projector.unfreeze()
        policy.model.eval()  # deterministic scoring; eval does not disable gradients

    def update(self, rows, rng):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map, tree_flatten
        from .policy import sequence_scores
        config = self.config
        if not rows:
            raise ValueError("RL update needs rollout decisions")
        indices = rng.choice(len(rows), size=min(len(rows), config["max_update_rows"]), replace=False)
        selected_rows = [rows[int(i)] for i in indices]
        losses = []
        gradient_norms = []
        def loss(model, tokens, old, selected, advantage):
            scores = sequence_scores(model, tokens)
            log_probs = scores - mx.logsumexp(scores)
            return clipped_action_loss(log_probs, old, selected, advantage,
                clip_epsilon=config["clip_epsilon"], kl_weight=config["kl_weight"],
                entropy_weight=config["entropy_weight"], xp=mx)
        grad_fn = nn.value_and_grad(self.policy.model, loss)
        for _ in range(config["epochs"]):
            rng.shuffle(selected_rows)
            for start in range(0, len(selected_rows), config["minibatch_size"]):
                batch = selected_rows[start:start + config["minibatch_size"]]
                accumulated = None
                for decision, advantage in batch:
                    tokens = self.policy.tokenize_completions(decision.messages, decision.actions,
                        market_context=decision.market_context)
                    value, grads = grad_fn(self.policy.model, tokens, mx.array(decision.old_log_probs),
                                           decision.selected, advantage)
                    mx.eval(value, grads)
                    losses.append(float(value.item()))
                    accumulated = grads if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grads)
                    mx.eval(accumulated)
                accumulated = tree_map(lambda g: g / len(batch), accumulated)
                leaves = [g for _, g in tree_flatten(accumulated)]
                norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in leaves))
                mx.eval(norm)
                if not np.isfinite(float(norm.item())) or not np.isfinite(losses).all():
                    raise ValueError("nonfinite RL gradient or loss")
                gradient_norms.append(float(norm.item()))
                factor = mx.minimum(1.0, config["max_grad_norm"] / mx.maximum(norm, mx.array(1e-12)))
                self.optimizer.update(self.policy.model, tree_map(lambda g: g * factor, accumulated))
                mx.eval(self.policy.model.parameters(), self.optimizer.state)
                mx.clear_cache()
        by_action = {}
        for decision, advantage in selected_rows:
            by_action.setdefault(decision.actions[decision.selected], []).append(float(advantage))
        return {"mean_loss": float(np.mean(losses)), "sampled_update_rows": len(selected_rows),
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "sampled_action_mass": {name: len(values) for name, values in by_action.items()},
                "mean_advantage_by_action": {name: float(np.mean(values)) for name, values in by_action.items()}}

    def save(self, destination, parent_adapter, metadata, *, runtime=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        import os
        import shutil
        import tempfile
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("RL output exists; refusing to overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".reasoning-rl-", dir=destination.parent))
        try:
            from .projector import export_policy_weights
            export_policy_weights(self.policy.model, temporary)
            shutil.copyfile(Path(parent_adapter) / "adapter_config.json", temporary / "adapter_config.json")
            (temporary / "rl_receipt.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
            if runtime is not None:
                from .checkpoints import save_training_state
                save_training_state(temporary, optimizer=self.optimizer, runtime=runtime)
                from .checkpoints import seal_checkpoint
                seal_checkpoint(temporary)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def train_rl(policy, environment, *, learner, episodes, context_config, sources, config,
             resume_state=None, checkpoint=None, diagnostic_records=None):
    """One policy version per complete same-start group; never update mid-rollout."""
    if not episodes:
        raise ValueError("RL training needs explicit training episodes")
    rng = np.random.default_rng(config["seed"])
    metrics = [] if resume_state is None else list(resume_state["metrics"])
    start_group = 0 if resume_state is None else resume_state["next_group"]
    if type(start_group) is not int or not 0 <= start_group <= config["groups"] or len(metrics) != start_group:
        raise ValueError("invalid RL resume group")
    if resume_state is not None:
        rng.bit_generator.state = resume_state["numpy_rng"]
    for group in range(start_group, config["groups"]):
        from .learning_audit import score_labeled_examples
        before = None if diagnostic_records is None else score_labeled_examples(policy, diagnostic_records)
        options = episodes[group % len(episodes)]
        trajectories, outcomes = [], []
        for _ in range(config["group_size"]):
            decisions, terminal = rollout(policy, environment, options=options,
                context_config=context_config, sources=sources, rng=rng, max_steps=config["max_steps"])
            trajectories.append(decisions)
            outcomes.append({"outcome": terminal["outcome"], "realized_pnl": terminal["realized_pnl"],
                             "minimum_mll_headroom": terminal["minimum_mll_headroom"]})
        rows = training_rows(trajectories, advantage_scale=config["advantage_scale"])
        report = {"group": group, "episodes": outcomes, "rollout_decisions": len(rows),
                  **learner.update(rows, rng)}
        from collections import Counter
        available = Counter(decision.actions[decision.selected] for decision, _ in rows)
        report["available_action_mass"] = dict(available)
        selected = report.get("sampled_action_mass", {})
        report["update_coverage"] = {name: selected.get(name, 0) / count for name, count in available.items()}
        if before is not None:
            after = score_labeled_examples(policy, diagnostic_records)
            report.update(audit_before=before, audit_after=after,
                ranking_regressions=sum(a["correct"] and not b["correct"] for a, b in zip(before, after)))
            reference = metrics[0].get("audit_before", before) if metrics else before
            report["regressions_from_initial_policy"] = sum(
                a["correct"] and not b["correct"] for a, b in zip(reference, after))
        print(json.dumps(report, allow_nan=False), flush=True)
        metrics.append(report)
        if checkpoint is not None and ((group + 1) % config["checkpoint_every_groups"] == 0
                                       or group + 1 == config["groups"]):
            checkpoint({"next_group": group + 1, "numpy_rng": rng.bit_generator.state,
                        "metrics": metrics})
    return metrics

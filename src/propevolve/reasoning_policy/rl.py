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
from .dataset import context_messages
from .inputs import specialist_account_fields


@dataclass(frozen=True)
class RLDecision:
    messages: list
    actions: tuple[str, ...]
    selected: int
    old_log_probs: tuple[float, ...]
    reward: float


def rollout(policy, environment, *, options, context_config, sources, rng, max_steps):
    """Store CPU causal prompts and detached sampling probabilities only."""
    observation, info = environment.reset(options=options)
    market = environment.markets[options["ticker"]]
    context = RollingContext(context_config)
    row = options["start"]
    decisions = []
    for _ in range(max_steps):
        fields = specialist_account_fields(observation, embedding_dim=market.embeddings.shape[1],
                                            ticker=options["ticker"], row=row, sources=sources)
        fields.update(environment.causal_trade_context())
        context.append(int(market.timestamps[row].astype("datetime64[ns]").astype(np.int64)),
                       {key: fields[key] for key in context_config.fields})
        actions = tuple(sorted((Action(a) for a in info["valid_actions"]), key=int))
        names = tuple(action.name for action in actions)
        messages = context_messages(context.snapshot(), actions)
        scores = policy.completion_scores(messages, names)
        logits = np.asarray([scores[name] for name in names], dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite rollout logits")
        logits -= logits.max()
        log_probs = logits - np.log(np.exp(logits).sum())
        selected = int(rng.choice(len(actions), p=np.exp(log_probs)))
        observation, reward, terminated, truncated, info = environment.step(actions[selected])
        decisions.append(RLDecision(messages, names, selected, tuple(log_probs), float(reward)))
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
    config = json.loads(Path(path).read_text())
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
    return config


class MLXAdapterLearner:
    """Reuse the policy's actual token scores; update only loaded LoRA tensors."""

    def __init__(self, policy, config):
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        self.policy, self.config = policy, config
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
                    tokens = self.policy.tokenize_completions(decision.messages, decision.actions)
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
                factor = mx.minimum(1.0, config["max_grad_norm"] / mx.maximum(norm, mx.array(1e-12)))
                self.optimizer.update(self.policy.model, tree_map(lambda g: g * factor, accumulated))
                mx.eval(self.policy.model.parameters(), self.optimizer.state)
                mx.clear_cache()
        return {"mean_loss": float(np.mean(losses)), "sampled_update_rows": len(selected_rows)}

    def save(self, destination, parent_adapter, metadata):
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
            mx.save_safetensors(str(temporary / "adapters.safetensors"),
                                dict(tree_flatten(self.policy.model.trainable_parameters())))
            shutil.copyfile(Path(parent_adapter) / "adapter_config.json", temporary / "adapter_config.json")
            (temporary / "rl_receipt.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def train_rl(policy, environment, *, learner, episodes, context_config, sources, config):
    """One policy version per complete same-start group; never update mid-rollout."""
    if not episodes:
        raise ValueError("RL training needs explicit training episodes")
    rng = np.random.default_rng(config["seed"])
    metrics = []
    for group in range(config["groups"]):
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
        print(json.dumps(report, allow_nan=False), flush=True)
        metrics.append(report)
    return metrics

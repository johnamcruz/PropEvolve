"""Compact search policy over the 54-field evidence contract.

The shipped artifact is a teacher-free reasoning model. This module is the SEARCH that
finds what it should be taught. A challenge episode is 30 days x 480 bars and the
reasoning model answers in ~0.3s a decision, so it cannot run the thousands of episodes
needed to discover a profitable trend-following policy; a small network over the same
evidence runs an episode in seconds and can.

It trades the real HistoricalChallengeEnv -- same $6,000 target, same $3,000 floor, same
fees, ratchet, daily loss limit and loss-streak cooldown -- so what it finds transfers
rather than being an artefact of a parallel simulator. Every teacher is visible here
(expansion, trend, regime, volume) precisely so RL can discover which combination rides
a trend safely; the student then reproduces those decisions without them.

No new dependency: torch is already present, and PropEvolve's discipline is hash-bound
receipts and deterministic replay, which an external RL framework would sit awkwardly
against.
"""
from __future__ import annotations

import numpy as np

_NEG_INF = -1e9

ACTION_ORDER = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")


def masked_categorical(logits, mask):
    """A distribution over LEGAL actions only.

    Masking rather than penalising matters: the environment refuses illegal actions
    outright, so probability mass spent on them is mass the policy never gets to use,
    and the advantage would be computed against decisions it could not have made.
    """
    import torch
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every row needs at least one legal action")
    masked = logits.masked_fill(~mask, _NEG_INF)
    return torch.distributions.Categorical(logits=masked)


class ActorCritic:
    """Separate policy and value trunks, as the shipped PPO baselines use."""

    def __init__(self, n_features: int, n_actions: int, hidden=(256, 256, 128), seed=17,
                 action_bias=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        def trunk(out):
            layers, last = [], n_features
            for h in hidden:
                layers += [nn.Linear(last, h), nn.Tanh()]
                last = h
            layers += [nn.Linear(last, out)]
            return nn.Sequential(*layers)

        self.policy = trunk(n_actions)
        self.value = trunk(1)
        self.n_actions = n_actions
        # Start biased toward HOLD. A uniform policy is a coin-flip exit, which cuts a
        # 2R winner that needed to run to 10R -- and this rule only pays because 23.9%
        # of trades carry $3,018 average winners. A random policy holding 90% of the
        # time returned +$1,214 to +$3,941 an episode while a uniform one returned
        # around zero, so uniform init makes PPO spend its budget rediscovering that
        # holding beats flipping. The bias is a starting point, not a constraint: the
        # policy can and should learn when to close.
        if action_bias:
            import torch as _t
            with _t.no_grad():
                final = self.policy[-1]
                for name, value in action_bias.items():
                    final.bias[ACTION_ORDER.index(name)] += float(value)

    def __call__(self, obs):
        return self.policy(obs), self.value(obs).squeeze(-1)

    def parameters(self):
        return list(self.policy.parameters()) + list(self.value.parameters())

    def state_dict(self):
        return {"policy": self.policy.state_dict(), "value": self.value.state_dict()}

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])


def challenge_objective(rates: dict) -> float:
    """Score a policy: zero blow first, then pass rate. Never a blend.

    Encodes the goal literally -- reach $6,000, and do not hit the $3,000 floor at all.
    A blend lets a policy rationally buy pass rate with blow-ups, which is exactly the
    behaviour worth making categorically unavailable, and it is what the acceptance
    criteria already demand with maximum_blow_rate 0.0.

    Any blow scores below -1.0, strictly under the worst possible zero-blow policy
    (which scores >= 0). Inside the blow region the score still improves as blows fall,
    so the optimiser can climb OUT rather than facing a flat wall.
    """
    blow = float(rates.get("blow", 0.0))
    passed = float(rates.get("pass", 0.0))
    if not (0.0 <= blow <= 1.0 and 0.0 <= passed <= 1.0):
        raise ValueError("challenge rates must be probabilities")
    if blow > 0.0:
        return -1.0 - blow
    return passed


def evidence_from_environment(environment, sources, *, ticker, row, observation,
                              names=None):
    """The causal evidence vector for one bar, via the shared observation path."""
    from .evidence import EVIDENCE_FIELDS, evidence_vector
    from .inputs import specialist_account_fields
    market = environment.markets[ticker]
    fields = specialist_account_fields(
        observation, embedding_dim=market.embeddings.shape[1], ticker=ticker, row=row,
        sources=sources, require_specialists=True,
        setup_dim=environment.setup_signals.output_dim)
    fields.update(environment.causal_trade_context())
    return evidence_vector(fields, names or EVIDENCE_FIELDS)


def compute_gae(rewards, values, *, last_value=0.0, gamma=1.0, lam=1.0):
    """Generalised advantage estimation with a LEARNED value baseline.

    The direct-RL run diverged on a leave-one-out group mean: with four episodes an
    action's advantage reflected which episode it landed in, not whether it was good,
    so every advantage flipped sign between groups. A per-state value baseline removes
    that coupling -- each decision is judged against what was expected FROM THAT STATE.

    gamma defaults to 1: a challenge is an episodic, undiscounted problem. Reaching
    $6,000 on day 29 is worth what it is worth on day 3, and the speed preference is
    already expressed by terminal_pass_speed_reward_per_day rather than by discounting.
    """
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nxt = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(net, optimizer, obs, mask, actions, old_log_probs, advantages, returns, *,
               clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=1.0):
    """One clipped-surrogate step over a minibatch."""
    import torch
    import torch.nn as nn
    logits, values = net(obs)
    dist = masked_categorical(logits, mask)
    log_probs = dist.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)
    std = advantages.std()
    normed = (advantages - advantages.mean()) / (std + 1e-8) if std > 1e-8 else advantages
    surrogate = torch.minimum(ratio * normed,
                              torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * normed)
    policy_loss = -surrogate.mean()
    value_loss = nn.functional.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    optimizer.step()
    return {"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()), "grad_norm": float(grad_norm)}


def rollout_episode(net, environment, sources, *, ticker, options, rng, torch_rng=None,
                    greedy=False, max_steps=20000, drawdown_penalty=2.5,
                    drawdown_floor_fraction=0.25):
    """One challenge episode. Forced bars are stepped without consulting the policy.

    Gated and flat on a bar the rule did not trigger, every entry is declined and the
    step resolves to WAIT whatever the policy says, so asking costs a forward pass and
    changes nothing. Reward earned on those bars accrues to the decision that led into
    them, keeping the undiscounted return exactly what the environment paid.
    """
    import torch
    from ..decision import Action
    from .rl import decision_is_forced
    obs, info = environment.reset(options=options)
    row = int(options["start"]) if options and "start" in options else environment._index
    index = {name: i for i, name in enumerate(ACTION_ORDER)}
    states, masks, actions, logps, values, rewards = [], [], [], [], [], []
    pending = 0.0
    for _ in range(max_steps):
        if decision_is_forced(environment):
            obs, reward, term, trunc, info = environment.step(Action.WAIT)
            pending += float(reward)
            if term or trunc:
                break
            continue
        vector = evidence_from_environment(environment, sources, ticker=ticker,
                                           row=environment._index, observation=obs)
        legal = [Action(a).name for a in info["valid_actions"]]
        mask = torch.zeros(1, len(ACTION_ORDER), dtype=torch.bool)
        for name in legal:
            mask[0, index[name]] = True
        state = torch.from_numpy(vector).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(state)
            dist = masked_categorical(logits, mask)
            choice = dist.probs.argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(choice)
        obs, reward, term, trunc, info = environment.step(Action[ACTION_ORDER[int(choice)]])
        states.append(vector); masks.append(mask[0].numpy())
        actions.append(int(choice)); logps.append(float(logp))
        values.append(float(value)); rewards.append(float(reward) + pending)
        pending = 0.0
        if term or trunc:
            break
    if pending and rewards:
        rewards[-1] += pending
    outcome = info.get("outcome")
    # Terminal drawdown penalty. The env's mll_proximity_penalty charges for TIME spent
    # near the floor, but near-blow is about the MINIMUM reached: a brief dive to 10%
    # cushion and back costs ~0.01 against a P&L signal of ~0.3, so it is effectively
    # free. Measured at 0.001 the learned policy held 47.5% near-blow across iterations
    # 5 and 10 while pass moved 50% -> 37.5%, i.e. the per-bar term taught nothing.
    # This charges the excursion once, scaled to matter against the ~2.25 a pass earns.
    if rewards and drawdown_penalty:
        floor = drawdown_floor_fraction * environment.spec.max_loss
        shortfall = max(0.0, (floor - float(environment._minimum_mll_headroom)) / floor)
        rewards[-1] -= drawdown_penalty * shortfall ** 2
    return {"states": states, "masks": masks, "actions": actions, "log_probs": logps,
            "values": values, "rewards": rewards, "outcome": outcome,
            "realized_pnl": float(environment._account.realized_pnl),
            "min_headroom": float(environment._minimum_mll_headroom)}

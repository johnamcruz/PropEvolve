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

    def __init__(self, n_features: int, n_actions: int, hidden=(256, 256, 128), seed=17):
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

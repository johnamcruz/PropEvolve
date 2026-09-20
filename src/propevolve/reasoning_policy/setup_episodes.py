"""SFT anchors taken from the Expansion + order-flow setup, not from a label census.

PropEvolve normally selects its supervised anchors with an exhaustive barrier-label
census: every bar is labelled Wait/Long/Short from its forward 2R-before-stop outcome and
anchors are sampled stratified across all tickers. That teaches "what would have worked",
which is a different lesson from the one being tested here.

This module teaches the algoTraderAI setup instead. Anchors sit on the bars where the
frozen research rule fired — an Expansion rising edge armed the window and 45-minute
order-flow persistence crossed its threshold — and the taught action is the side the flow
imbalance implies. It is NQ-only by construction, because that is the only market with
order flow.

Two properties worth stating plainly, because they decide what the comparison means:

* The anchors are the SAME events algoTraderAI's PPO sees as gates, so the two systems are
  learning on identical opportunities and their pass rates are comparable.
* The ACTION is not dictated by the rule. Anchors say where to study; PropEvolve's
  economics say what the right action was. Teaching the rule's own verdict would cap the
  policy at imitating a rule we already have, and would fight the error-selected
  distillation, whose whole purpose is to train on the cases the current policy gets
  wrong. ``setup_action`` is recorded for reporting only.

Declined bars are sampled too, so the policy sees both sides of the decision instead of
only ever being shown entries. They carry NO expected action: PropEvolve labels from the
forward outcome, and asserting "the rule waited" would contradict the collector whenever
a declined setup would in fact have worked. So the anchors choose WHERE the policy
studies and the economics choose WHAT the right action was there — which is also what
leaves room to beat the rule rather than merely imitate it.
"""

from __future__ import annotations

import numpy as np

from ..decision import Action
from ..setup_signals import CHANNEL_NAMES

_TRIGGER = CHANNEL_NAMES.index("setup_trigger")
_SIDE = CHANNEL_NAMES.index("setup_side")
_AVAILABLE = CHANNEL_NAMES.index("setup_available")
_ARMED = CHANNEL_NAMES.index("expansion_armed")


def _eligible_range(environment, ticker: str, warmup: int) -> np.ndarray:
    """Bars that can start an episode: past warm-up and with a full episode ahead."""
    market = environment.markets[ticker]
    session_keys = environment._session_keys[ticker]
    unique_sessions = np.unique(session_keys)
    if len(unique_sessions) <= environment.spec.episode_days:
        raise ValueError(f"{ticker} cannot fit {environment.spec.episode_days} trading days")
    last_start_session = unique_sessions[-environment.spec.episode_days]
    maximum_start = min(
        len(market.close) - 2,
        int(np.searchsorted(session_keys, last_start_session, side="right") - 1),
    )
    eligible = np.zeros(len(market.close), dtype=bool)
    eligible[warmup:maximum_start + 1] = True
    return eligible


def setup_episode_specs(config, environment, role: str) -> list[dict]:
    """Anchors on the rule's triggers, plus the bars it declined.

    ``setup_action_sampling[role]`` takes ``per_action`` and ``seed``; ``wait_pool``
    chooses which declined bars are eligible for Wait anchors:

        armed   only bars inside an armed Expansion window that the flow declined —
                the hard negatives, where timing was right and the side was not
        any     any available bar without a trigger
    """
    sampling = config["setup_action_sampling"][role]
    if not isinstance(sampling, dict) or not {"per_action", "seed"}.issubset(sampling):
        raise ValueError("setup action sampling requires per_action and seed")
    unexpected = set(sampling) - {"per_action", "seed", "wait_pool"}
    if unexpected:
        raise ValueError(f"unsupported setup action sampling keys {sorted(unexpected)}")
    wait_pool = sampling.get("wait_pool", "armed")
    if wait_pool not in {"armed", "any"}:
        raise ValueError("wait_pool must be 'armed' or 'any'")
    per_action = int(sampling["per_action"])
    if per_action < 1:
        raise ValueError("per_action must be positive")
    warmup = int(config.get("collection_warmup_steps", 0))

    rng = np.random.default_rng(int(sampling["seed"]))
    specs: list[dict] = []
    for ticker in config["tickers"][role]:
        market = environment.markets[ticker]
        if market.setup_channels is None:
            raise ValueError(
                f"{ticker} has no setup channels; setup anchors need the research bundle")
        channels = np.asarray(market.setup_channels)
        eligible = _eligible_range(environment, ticker, warmup)
        available = channels[:, _AVAILABLE] > 0.0
        triggered = (channels[:, _TRIGGER] > 0.0) & available & eligible
        sides = np.sign(channels[:, _SIDE])

        pools = {
            Action.ENTER_LONG_1: np.flatnonzero(triggered & (sides > 0)),
            Action.ENTER_SHORT_1: np.flatnonzero(triggered & (sides < 0)),
        }
        declined = available & eligible & ~triggered
        if wait_pool == "armed":
            declined &= channels[:, _ARMED] > 0.0
        pools[Action.WAIT] = np.flatnonzero(declined)

        for action, rows in pools.items():
            if len(rows) == 0:
                continue
            take = min(per_action, len(rows))
            chosen = rng.choice(rows, size=take, replace=False) if take < len(rows) else rows
            for row in np.sort(np.asarray(chosen)):
                # No expected_action. The anchors choose WHERE the policy studies; the
                # economics choose WHAT the right action was there. Asserting the rule's
                # own verdict would cap the policy at imitating a rule we already have,
                # and would contradict the collector wherever the rule was wrong — which
                # is precisely the population the error-selected distillation needs.
                specs.append({"ticker": ticker, "start": int(row) - warmup,
                              "setup_action": int(action)})
    specs.sort(key=lambda item: (item["ticker"], item["start"]))
    if not specs:
        raise ValueError("the setup produced no anchors for this role")
    return specs


def setup_anchor_census(environment, ticker: str) -> dict:
    """Counts behind the anchors, for the run report."""
    market = environment.markets[ticker]
    if market.setup_channels is None:
        return {"ticker": ticker, "available": 0, "triggers": 0}
    channels = np.asarray(market.setup_channels)
    available = channels[:, _AVAILABLE] > 0.0
    triggered = (channels[:, _TRIGGER] > 0.0) & available
    sides = np.sign(channels[:, _SIDE])
    return {
        "ticker": ticker,
        "bars": int(len(channels)),
        "available": int(available.sum()),
        "armed": int(((channels[:, _ARMED] > 0.0) & available).sum()),
        "triggers": int(triggered.sum()),
        "long_triggers": int((triggered & (sides > 0)).sum()),
        "short_triggers": int((triggered & (sides < 0)).sum()),
    }

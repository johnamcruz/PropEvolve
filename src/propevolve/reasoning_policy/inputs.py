"""Explicit inference-time specialist inputs; no change to training-only teachers."""

import numpy as np

from ..observation import ObservationAssembler
from ..setup_signals import CHANNEL_NAMES as SETUP_CHANNEL_NAMES


ACCOUNT_FIELDS = (
    "realized_pnl_norm", "equity_pnl_norm", "peak_equity_pnl_norm",
    "mll_headroom_norm", "drawdown_norm", "position_side", "position_size_norm",
    "unrealized_pnl_norm", "session_remaining", "challenge_remaining",
    "point_value_norm", "round_trip_fee_norm",
)
MANAGEMENT_FIELDS = (
    "current_r_norm", "peak_favorable_r_norm", "giveback_r_norm",
    "hold_fraction", "ratchet_active", "protected_r_norm",
)


def specialist_account_fields(
    observation, *, embedding_dim: int, ticker: str, row: int, sources,
    require_specialists=True, setup_dim: int = 0,
) -> dict[str, float]:
    """Read only the exact aligned score row, plus production-normalized state.

Sources implement the existing target(ticker,row) protocol, with kind/channels.
This adapter deliberately declares specialist dependence at inference time.
Missing specialist history is an error, not a fabricated zero probability.
    """
    observation = np.asarray(observation)
    if observation.ndim != 1 or not np.isfinite(observation).all():
        raise ValueError("observation must be a finite vector")
    tail = observation[embedding_dim:]
    # The Expansion + order-flow channels are appended last by ObservationAssembler.
    # They are split off here and exposed as NAMED fields, because a reasoning policy
    # reads named prompt fields, not raw vector positions — leaving them in the account
    # slice would both break its width contract and hide the setup from the model.
    setup_dim = int(setup_dim)
    if setup_dim:
        if len(tail) <= setup_dim:
            raise ValueError("observation is too short to carry setup channels")
        account, setup_values = tail[:-setup_dim], tail[-setup_dim:]
    else:
        account, setup_values = tail, np.empty(0)
    if len(account) not in {ObservationAssembler.ACCOUNT_DIM, ObservationAssembler.ACCOUNT_DIM + 6}:
        raise ValueError("account observation contract mismatch")
    names = ACCOUNT_FIELDS + (MANAGEMENT_FIELDS if len(account) > len(ACCOUNT_FIELDS) else ())
    fields = {f"account.{name}": float(value) for name, value in zip(names, account)}
    if setup_dim:
        if setup_dim != len(SETUP_CHANNEL_NAMES):
            raise ValueError("setup channel contract mismatch")
        fields.update({f"setup.{name}": float(value)
                       for name, value in zip(SETUP_CHANNEL_NAMES, setup_values)})
    if not require_specialists:
        return fields
    kinds = [source.kind for source in sources]
    if (len(kinds) != len(set(kinds)) or not {"expansion", "trend", "regime"}.issubset(kinds)
            or set(kinds) - {"expansion", "trend", "regime", "volume"}):
        raise ValueError("challenger requires Expansion, Trend, Regime and optional Volume")
    for source in sources:
        values = source.targets.target(ticker, row)
        if values is None:
            raise ValueError(f"unavailable {source.kind} evidence")
        values = np.asarray(values)
        bounds = np.asarray(getattr(source, "bounds", [(0.0, 1.0)] * len(source.channels)))
        if (values.shape != (len(source.channels),) or bounds.shape != (len(source.channels), 2)
                or not np.isfinite(values).all()
                or (values < bounds[:, 0]).any() or (values > bounds[:, 1]).any()):
            raise ValueError(f"invalid {source.kind} values")
        for channel, value in zip(source.channels, values):
            fields[f"{source.kind}.{channel}"] = float(value)
    return fields


def observe_context(history, environment, observation, *, ticker, row, sources):
    """One causal observation path shared by collection, RL and evaluation."""
    market = environment.markets[ticker]
    use_teachers = history.config.input_mode == "specialists"
    fields = specialist_account_fields(observation, embedding_dim=market.embeddings.shape[1],
        ticker=ticker, row=row, sources=sources, require_specialists=use_teachers,
        setup_dim=environment.setup_signals.output_dim)
    fields.update(environment.causal_trade_context())
    if history.config.volatility_lookback is not None:
        fields.update(trade_r_context(market, row=row,
            risk_dollars=environment.spec.per_trade_risk_dollars,
            point_value=environment.tick_values[ticker],
            round_trip_fee=environment.round_trip_fees[ticker],
            lookback=history.config.volatility_lookback))
    if not set(history.config.fields).issubset(fields):
        raise ValueError("configured input unavailable from causal observation")
    history.append(int(market.timestamps[row].astype("datetime64[ns]").astype(np.int64)),
        {key: fields[key] for key in history.config.fields},
        embedding=observation[:market.embeddings.shape[1]] if not use_teachers else None)


def trade_r_context(market, *, row, risk_dollars, point_value, round_trip_fee, lookback):
    """Completed-bar arithmetic mean true range / prospective dollar R.

    Uses a full lookback plus its preceding close, never a future fill or label.
    R is the configured dollar risk (including costs), not gross stop distance.
    Cost is known even when volatility history is unavailable. No fitted scaler,
    challenge state, clipping, or position-dependent denominator is used.
    """
    if (type(row) is not int or not 0 <= row < len(market.close)
            or type(lookback) is not int or lookback < 1
            or risk_dollars is None):
        raise ValueError("invalid causal R context contract")
    economic = np.asarray([risk_dollars, point_value, round_trip_fee], dtype=float)
    if (not np.isfinite(economic).all() or risk_dollars <= 0 or point_value <= 0
            or not 0 <= round_trip_fee < risk_dollars):
        raise ValueError("invalid causal R context economics")
    result = {"trade.volatility_r": 0., "trade.cost_r": float(round_trip_fee / risk_dollars),
              "trade.volatility_available": float(row >= lookback)}
    if row < lookback:
        return result
    start = row - lookback + 1
    high = np.asarray(market.high[start:row + 1], dtype=float)
    low = np.asarray(market.low[start:row + 1], dtype=float)
    previous = np.asarray(market.close[start - 1:row], dtype=float)
    if not np.isfinite([high, low, previous]).all() or (high < low).any():
        raise ValueError("invalid completed bars for R context")
    true_range = np.maximum(high - low, np.maximum(abs(high - previous), abs(low - previous)))
    result["trade.volatility_r"] = float(true_range.mean() * point_value / risk_dollars)
    return result

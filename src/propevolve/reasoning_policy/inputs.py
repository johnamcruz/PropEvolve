"""Explicit inference-time specialist inputs; no change to training-only teachers."""

import numpy as np

from ..observation import ObservationAssembler


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
    observation, *, embedding_dim: int, ticker: str, row: int, sources, require_specialists=True,
) -> dict[str, float]:
    """Read only the exact aligned score row, plus production-normalized state.

Sources implement the existing target(ticker,row) protocol, with kind/channels.
This adapter deliberately declares specialist dependence at inference time.
Missing specialist history is an error, not a fabricated zero probability.
    """
    observation = np.asarray(observation)
    if observation.ndim != 1 or not np.isfinite(observation).all():
        raise ValueError("observation must be a finite vector")
    account = observation[embedding_dim:]
    if len(account) not in {ObservationAssembler.ACCOUNT_DIM, ObservationAssembler.ACCOUNT_DIM + 6}:
        raise ValueError("account observation contract mismatch")
    names = ACCOUNT_FIELDS + (MANAGEMENT_FIELDS if len(account) > len(ACCOUNT_FIELDS) else ())
    fields = {f"account.{name}": float(value) for name, value in zip(names, account)}
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
        ticker=ticker, row=row, sources=sources, require_specialists=use_teachers)
    fields.update(environment.causal_trade_context())
    if not set(history.config.fields).issubset(fields):
        raise ValueError("configured input unavailable from causal observation")
    history.append(int(market.timestamps[row].astype("datetime64[ns]").astype(np.int64)),
        {key: fields[key] for key in history.config.fields},
        embedding=observation[:market.embeddings.shape[1]] if not use_teachers else None)

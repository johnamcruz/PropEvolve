"""Reasoning-only market, teacher, simulator, and temporal source contract."""

from datetime import date
import json
import math
from pathlib import Path


_SOURCE_FIELDS = {
    "schema", "assets", "tickers", "timeframe_minutes", "cache_root",
    "teachers", "observation", "challenge", "point_values",
    "round_trip_fees", "temporal",
}
_COMMON_TEACHER_FIELDS = {
    "kind", "cache_root", "channels", "loss_weight",
    "entry_search_loss_weight",
}
_TEMPORAL_FIELDS = {
    "train_start", "train_end", "validation_start", "validation_end",
    "sealed_start",
}


def _finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and value >= 0)


def load_source_recipe(path):
    """Load the complete shared source without legacy learner configuration."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load reasoning market source: {error}") from error
    if (not isinstance(payload, dict) or set(payload) != _SOURCE_FIELDS
            or payload.get("schema") != "propevolve_reasoning_market_source_v1"):
        raise ValueError("invalid reasoning market source fields")
    tickers = payload["tickers"]
    if (not isinstance(tickers, list) or not tickers
            or any(not isinstance(item, str) or not item for item in tickers)
            or len(tickers) != len(set(tickers))
            or type(payload["timeframe_minutes"]) is not int
            or payload["timeframe_minutes"] < 1
            or any(not isinstance(payload[name], str) or not payload[name]
                   for name in ("assets", "cache_root"))):
        raise ValueError("invalid reasoning market source identity")
    markets = set(tickers)
    if (set(payload["point_values"]) != markets
            or set(payload["round_trip_fees"]) != markets
            or any(not _finite_nonnegative(value) or value == 0
                   for values in (payload["point_values"], payload["round_trip_fees"])
                   for value in values.values())):
        raise ValueError("reasoning market economics do not match tickers")
    teachers = payload["teachers"]
    if (not isinstance(teachers, list)
            or [item.get("kind") for item in teachers] != ["expansion", "regime", "trend"]):
        raise ValueError("reasoning source requires Expansion, Regime, and Trend")
    for teacher in teachers:
        expected = _COMMON_TEACHER_FIELDS | (
            {"entry_search_objective"} if teacher["kind"] == "expansion" else set())
        if (set(teacher) != expected or not isinstance(teacher["cache_root"], str)
                or not teacher["cache_root"] or not isinstance(teacher["channels"], list)
                or not teacher["channels"]
                or len(teacher["channels"]) != len(set(teacher["channels"]))
                or any(not isinstance(channel, str) or not channel
                       for channel in teacher["channels"])
                or not _finite_nonnegative(teacher["loss_weight"])
                or not _finite_nonnegative(teacher["entry_search_loss_weight"])
                or (teacher["kind"] == "expansion"
                    and teacher["entry_search_objective"] != "raw_probability")):
            raise ValueError("invalid reasoning teacher source")
    temporal = payload["temporal"]
    if not isinstance(temporal, dict) or set(temporal) != _TEMPORAL_FIELDS:
        raise ValueError("invalid reasoning temporal source")
    try:
        boundaries = [date.fromisoformat(temporal[name]) for name in (
            "train_start", "train_end", "validation_start", "validation_end",
            "sealed_start")]
    except (TypeError, ValueError) as error:
        raise ValueError("invalid reasoning temporal source") from error
    if not all(left <= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("reasoning temporal roles overlap")
    from ..environment import ChallengeSpec
    from ..observation import TradeManagementObservationSpec
    try:
        ChallengeSpec(**payload["challenge"])
        TradeManagementObservationSpec.from_config(payload["observation"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid shared simulator source") from error
    return payload


__all__ = ["load_source_recipe"]

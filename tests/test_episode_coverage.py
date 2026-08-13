from __future__ import annotations

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import (
    ChallengeSpec,
    FullDataEpisodeCoverageSpec,
    HistoricalChallengeEnv,
    MarketSeries,
)


def _spec() -> ChallengeSpec:
    return ChallengeSpec(
        profit_target=6_000.0,
        max_loss=3_000.0,
        episode_days=2,
        bars_per_day=2,
        max_position_size=1,
        minimum_mll_headroom=250.0,
        trailing_mll_lock=True,
        terminal_pass_reward=250.0,
        terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0,
        terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0,
    )


def _market(ticker: str, *, offset: float = 0.0) -> MarketSeries:
    timestamps = np.array([
        np.datetime64("2024-01-01T23:00")
        + session * np.timedelta64(1, "D")
        + bar * np.timedelta64(3, "m")
        for session in range(6)
        for bar in range(2)
    ])
    close = np.arange(len(timestamps), dtype=np.float32) + 100.0 + offset
    return MarketSeries(
        ticker=ticker,
        timestamps=timestamps,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        embeddings=np.zeros((len(close), 2), dtype=np.float32),
    )


def _finish_with_wait(env: HistoricalChallengeEnv) -> None:
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(Action.WAIT)


def test_declared_full_data_mode_covers_each_market_deterministically() -> None:
    env = HistoricalChallengeEnv(
        {"AA": _market("AA"), "BB": _market("BB", offset=20.0)},
        round_trip_fees={"AA": 0.0, "BB": 0.0},
        tick_values={"AA": 1.0, "BB": 1.0},
        spec=_spec(),
        seed=91,
        episode_coverage=FullDataEpisodeCoverageSpec(episode_budget=10),
    )

    starts = {"AA": [], "BB": []}
    for _ in range(5):
        for ticker in ("AA", "BB"):
            _, reset = env.reset(options={"ticker": ticker})
            starts[ticker].append(reset["start"])
            _finish_with_wait(env)

    receipt = env.episode_coverage_receipt(require_complete=True)

    assert starts == {"AA": [0, 3, 5, 7, 9], "BB": [0, 3, 5, 7, 9]}
    assert receipt["schema"] == "propevolve_episode_coverage_receipt_v1"
    assert receipt["episode_budget"] == 10
    assert receipt["episodes_consumed"] == 10
    for ticker in ("AA", "BB"):
        market = receipt["markets"][ticker]
        assert market["eligible_decision_rows"] == 11
        assert market["covered_decision_rows"] == 11
        assert market["coverage_fraction"] == 1.0
        assert market["first_eligible_index"] == 0
        assert market["last_eligible_index"] == 10
        assert market["first_covered_index"] == 0
        assert market["last_covered_index"] == 10
        assert len(market["row_map_identity_sha256"]) == 64
        assert len(market["identity_sha256"]) == 64
    assert len(receipt["identity_sha256"]) == 64


def test_full_data_coverage_resume_preserves_exact_row_and_start_identity() -> None:
    settings = dict(
        markets={"AA": _market("AA"), "BB": _market("BB", offset=20.0)},
        round_trip_fees={"AA": 0.0, "BB": 0.0},
        tick_values={"AA": 1.0, "BB": 1.0},
        spec=_spec(),
        seed=91,
        episode_coverage=FullDataEpisodeCoverageSpec(episode_budget=10),
    )
    uninterrupted = HistoricalChallengeEnv(**settings)
    resumed = HistoricalChallengeEnv(**settings)

    for ticker in ("AA", "BB", "AA", "BB"):
        uninterrupted.reset(options={"ticker": ticker})
        _finish_with_wait(uninterrupted)
    resumed.restore_rng_state(uninterrupted.rng_state())

    for _ in range(3):
        for ticker in ("AA", "BB"):
            uninterrupted.reset(options={"ticker": ticker})
            _finish_with_wait(uninterrupted)
            resumed.reset(options={"ticker": ticker})
            _finish_with_wait(resumed)

    assert resumed.episode_coverage_receipt(require_complete=True) == (
        uninterrupted.episode_coverage_receipt(require_complete=True)
    )


def test_full_data_coverage_rejects_budget_that_cannot_cover_every_market_row() -> None:
    with pytest.raises(
        ValueError,
        match="episode coverage budget cannot cover every eligible row",
    ):
        HistoricalChallengeEnv(
            {"AA": _market("AA"), "BB": _market("BB", offset=20.0)},
            round_trip_fees={"AA": 0.0, "BB": 0.0},
            tick_values={"AA": 1.0, "BB": 1.0},
            spec=_spec(),
            seed=91,
            episode_coverage=FullDataEpisodeCoverageSpec(episode_budget=8),
        )


def test_ordinary_environment_keeps_seeded_random_episode_starts() -> None:
    env = HistoricalChallengeEnv(
        {"AA": _market("AA")},
        round_trip_fees={"AA": 0.0},
        tick_values={"AA": 1.0},
        spec=_spec(),
        seed=91,
    )

    starts = []
    for _ in range(5):
        _, reset = env.reset(options={"ticker": "AA"})
        starts.append(reset["start"])
        _finish_with_wait(env)

    assert starts == [7, 2, 3, 9, 1]
    with pytest.raises(
        RuntimeError,
        match="deterministic episode coverage is not configured",
    ):
        env.episode_coverage_receipt()


def test_full_data_coverage_fails_closed_on_unbalanced_market_schedule() -> None:
    env = HistoricalChallengeEnv(
        {"AA": _market("AA"), "BB": _market("BB", offset=20.0)},
        round_trip_fees={"AA": 0.0, "BB": 0.0},
        tick_values={"AA": 1.0, "BB": 1.0},
        spec=_spec(),
        seed=91,
        episode_coverage=FullDataEpisodeCoverageSpec(episode_budget=10),
    )
    env.reset(options={"ticker": "AA"})
    _finish_with_wait(env)

    with pytest.raises(
        RuntimeError,
        match="episode coverage ticker schedule is not balanced",
    ):
        env.reset(options={"ticker": "AA"})


def test_full_data_coverage_spec_parses_only_the_declared_schema() -> None:
    assert FullDataEpisodeCoverageSpec.from_config({
        "schema": "full_data_episode_coverage_v1",
        "episode_budget": 500,
    }) == FullDataEpisodeCoverageSpec(episode_budget=500)

    with pytest.raises(ValueError, match="coverage mode contract is invalid"):
        FullDataEpisodeCoverageSpec.from_config({
            "schema": "full_data_episode_coverage_v1",
            "episode_budget": 500,
            "unbounded": True,
        })

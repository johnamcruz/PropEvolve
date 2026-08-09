from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import (
    ChallengeSpec,
    HistoricalChallengeEnv,
    MarketSeries,
    PropChallengeAccount,
    _cme_session_keys,
)


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "golden_prop_trajectories_v1.json").read_text()
)


def test_golden_fixture_is_bound_to_one_reference_revision() -> None:
    assert FIXTURE["schema"] == "propevolve_golden_prop_trajectories_v1"
    revision = FIXTURE["reference_revision"]
    assert len(revision) == 40
    assert set(revision) <= set("0123456789abcdef")


@pytest.mark.parametrize("trajectory", FIXTURE["account_trajectories"], ids=lambda row: row["name"])
def test_golden_account_trajectory(trajectory: dict) -> None:
    account = PropChallengeAccount(
        max_loss=trajectory["max_loss"],
        profit_target=trajectory["profit_target"],
        trailing_mll_lock=True,
    )
    for event in trajectory["events"]:
        if event["kind"] == "realize":
            account.realize(event["net_pnl"])
        elif event["kind"] == "session_close":
            account.close_session()
        else:  # pragma: no cover - fixture schema guard
            raise AssertionError(f"unknown fixture event {event['kind']}")

    expected = trajectory["expected"]
    assert account.realized_pnl == pytest.approx(expected["realized_pnl"])
    assert account.mll_floor_pnl == pytest.approx(expected["mll_floor_pnl"])
    assert account.mll_headroom() == pytest.approx(expected["mll_headroom"])
    assert account.passmark_locked is expected["passmark_locked"]
    assert account.outcome() == expected["outcome"]


@pytest.mark.parametrize("trajectory", FIXTURE["execution_trajectories"], ids=lambda row: row["name"])
def test_golden_execution_trajectory(trajectory: dict) -> None:
    length = len(trajectory["close"])
    market = MarketSeries(
        ticker="NQ",
        timestamps=np.asarray(
            [value.removesuffix("Z") for value in trajectory["timestamps"]],
            dtype="datetime64[ns]",
        ),
        open=np.asarray(trajectory["open"], dtype=np.float32),
        high=np.asarray(trajectory["high"], dtype=np.float32),
        low=np.asarray(trajectory["low"], dtype=np.float32),
        close=np.asarray(trajectory["close"], dtype=np.float32),
        embeddings=np.zeros((length, 2), dtype=np.float32),
    )
    episode_days = len(np.unique(_cme_session_keys(market.timestamps)))
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": trajectory["point_value"]},
        round_trip_fees={"NQ": trajectory["round_trip_fee"]},
        spec=ChallengeSpec(
            profit_target=trajectory["profit_target"],
            max_loss=trajectory["max_loss"],
            episode_days=episode_days,
            bars_per_day=length - 1,
            max_position_size=1,
            minimum_mll_headroom=0.0,
            trailing_mll_lock=True,
            terminal_pass_reward=250.0,
            terminal_blow_reward=-1_500.0,
            terminal_timeout_reward=-2.0,
            terminal_pass_speed_reward_per_day=20.0,
            reward_scale=1_000.0,
        ),
        seed=7,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    for action_name, expected in zip(
        trajectory["actions"], trajectory["expected_steps"], strict=True
    ):
        _, reward, terminated, truncated, info = env.step(Action[action_name])
        assert not truncated
        assert terminated is expected["terminated"]
        assert reward == pytest.approx(expected["reward"])
        for field in (
            "fill_index",
            "fill_price",
            "fees_paid",
            "worst_equity_pnl",
            "equity_pnl",
            "mll_floor_pnl",
            "mll_headroom",
        ):
            if field in expected:
                assert info[field] == pytest.approx(expected[field])
        assert info["outcome"] == expected["outcome"]

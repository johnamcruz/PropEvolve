from __future__ import annotations

from copy import deepcopy

import pytest

from propevolve import evaluate_sealed_confirmation


TICKERS = ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")


def _contract() -> dict:
    return {
        "start": "2026-01-01",
        "end": "2027-01-01",
        "episode_sessions": 30,
        "window_mode": "non_overlapping",
        "tickers": list(TICKERS),
        "minimum_episodes_per_ticker": 2,
        "teacher_free": True,
        "minimum_pass_rate": 0.5,
        "minimum_per_market_pass_rate": 0.5,
        "maximum_blow_rate": 0.0,
        "minimum_expectancy_r": 0.0,
    }


def _episodes() -> list[dict]:
    rows = []
    for ticker in TICKERS:
        rows.extend((
            {
                "ticker": ticker,
                "start_session": "2026-01-01",
                "end_session": "2026-02-11",
                "session_count": 30,
                "outcome": "pass",
                "terminal_pnl": 6000.0,
                "trade_count": 20,
                "win_count": 9,
                "winning_r_sum": 18.0,
                "trade_r_sum": 5.0,
                "long_trade_count": 10,
                "short_trade_count": 10,
                "two_r_eligible_count": 4,
                "two_r_capture_sum": 3.0,
                "near_blow": False,
                "teacher_accessed": False,
            },
            {
                "ticker": ticker,
                "start_session": "2026-02-12",
                "end_session": "2026-03-25",
                "session_count": 30,
                "outcome": "timeout",
                "terminal_pnl": 1000.0,
                "trade_count": 10,
                "win_count": 4,
                "winning_r_sum": 7.0,
                "trade_r_sum": 1.0,
                "long_trade_count": 5,
                "short_trade_count": 5,
                "two_r_eligible_count": 2,
                "two_r_capture_sum": 1.0,
                "near_blow": False,
                "teacher_accessed": False,
            },
        ))
    return rows


def test_teacher_free_non_overlapping_2026_challenges_can_pass() -> None:
    report = evaluate_sealed_confirmation(_contract(), _episodes())

    assert report["status"] == "PASS"
    assert report["metrics"]["pass_rate"] == 0.5
    assert report["metrics"]["blow_rate"] == 0.0
    assert report["metrics"]["trade_win_rate"] == pytest.approx(13 / 30)
    assert report["metrics"]["average_win_r"] == pytest.approx(25 / 13)
    assert report["metrics"]["expectancy_r"] == pytest.approx(6 / 30)
    assert report["metrics"]["long_trade_fraction"] == 0.5
    assert report["metrics"]["two_r_mfe_capture_ratio"] == pytest.approx(4 / 6)
    assert set(report["per_market"]) == set(TICKERS)


def test_any_blow_fails_the_final_confirmation() -> None:
    episodes = _episodes()
    episodes[0] = {**episodes[0], "outcome": "blow", "terminal_pnl": -3000.0}

    report = evaluate_sealed_confirmation(_contract(), episodes)

    assert report["status"] == "FAIL"
    assert any(item["metric"] == "blow_rate" for item in report["failures"])


def test_final_confirmation_rejects_teacher_access_or_overlapping_windows() -> None:
    teacher_rows = _episodes()
    teacher_rows[0] = {**teacher_rows[0], "teacher_accessed": True}
    with pytest.raises(ValueError, match="teacher-free"):
        evaluate_sealed_confirmation(_contract(), teacher_rows)

    overlapping = deepcopy(_episodes())
    overlapping[1]["start_session"] = "2026-02-01"
    with pytest.raises(ValueError, match="overlap"):
        evaluate_sealed_confirmation(_contract(), overlapping)

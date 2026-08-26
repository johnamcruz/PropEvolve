from __future__ import annotations

import json
from pathlib import Path

from propevolve.pass_timeout_analysis import analyze_pass_timeout_diagnostics


def _confusion(
    *,
    wait: tuple[int, int, int],
    long: tuple[int, int, int],
    short: tuple[int, int, int],
) -> dict[str, dict[str, int]]:
    return {
        "wait": {"wait": wait[0], "long": wait[1], "short": wait[2]},
        "long": {"wait": long[0], "long": long[1], "short": long[2]},
        "short": {"wait": short[0], "long": short[1], "short": short[2]},
    }


def _record(
    episode: int,
    *,
    outcome: str,
    ticker: str,
    terminal_pnl: float,
    near_blow: bool,
    dominant: dict[str, dict[str, int]],
    nonchop: dict[str, dict[str, int]],
    economics: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "episode": episode,
        "ticker": ticker,
        "outcome": outcome,
        "terminal_pnl": terminal_pnl,
        "near_blow_timeout": near_blow,
        "trade_count": 10 * episode,
        "win_rate": 0.5 if outcome == "pass" else 0.25,
        "avg_win_r": 2.0 if outcome == "pass" else 0.5,
        "expectancy_r": 0.25 if outcome == "pass" else -0.2,
        "two_r_mfe_capture_ratio": 0.8 if outcome == "pass" else 0.4,
        "gave_it_all_back_rate": 0.1 if outcome == "pass" else 0.5,
        "sampled_entry_action_target_counts": {
            "WAIT": 6,
            "ENTER_LONG_1": 2,
            "ENTER_SHORT_1": 2,
        },
        "sampled_entry_action_prediction_counts": {
            "WAIT": 7,
            "ENTER_LONG_1": 2,
            "ENTER_SHORT_1": 1,
        },
        "sampled_entry_action_correct_counts": {
            "WAIT": 5,
            "ENTER_LONG_1": 2,
            "ENTER_SHORT_1": 1,
        },
        "regime_selectivity": {
            "dominant_chop": {"confusion": dominant},
            "nonchop": {"confusion": nonchop},
        },
        "regime_trade_economics": {
            "groups": economics,
        },
        "challenge_return_self_imitation": {
            "rows": 2,
            "bonus_sum": 0.3,
            "bonus_mean": 0.15,
            "added_clip_rows": 0,
            "actions": {
                "WAIT": {"rows": 1, "bonus_sum": 0.1, "bonus_mean": 0.1},
                "ENTER_LONG_1": {
                    "rows": 1,
                    "bonus_sum": 0.2,
                    "bonus_mean": 0.2,
                },
                "ENTER_SHORT_1": {
                    "rows": 0,
                    "bonus_sum": 0.0,
                    "bonus_mean": None,
                },
            },
        },
    }


def test_pass_timeout_analysis_compares_sides_and_true_chop_entries(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "training-diagnostics.jsonl"
    records = (
        _record(
            1,
            outcome="pass",
            ticker="NQ",
            terminal_pnl=6_100.0,
            near_blow=False,
            dominant=_confusion(
                wait=(4, 1, 0),
                long=(1, 1, 0),
                short=(1, 0, 1),
            ),
            nonchop=_confusion(
                wait=(3, 0, 0),
                long=(0, 2, 0),
                short=(0, 0, 2),
            ),
            economics=[
                {
                    "episode_outcome": "pass",
                    "static_regime": "nonchop",
                    "headroom_stratum": "safe_headroom_ge_0_75",
                    "side": "long",
                    "trades": 2,
                    "wins": 1,
                    "realized_r_sum": 3.0,
                    "mfe_r_sum": 5.0,
                    "mae_r_sum": 1.0,
                    "initial_stop_count": 0,
                },
                {
                    "episode_outcome": "pass",
                    "static_regime": "dominant_chop",
                    "headroom_stratum": "safe_headroom_ge_0_75",
                    "side": "short",
                    "trades": 1,
                    "wins": 0,
                    "realized_r_sum": -0.5,
                    "mfe_r_sum": 0.2,
                    "mae_r_sum": 0.8,
                    "initial_stop_count": 1,
                },
            ],
        ),
        _record(
            2,
            outcome="timeout",
            ticker="CL",
            terminal_pnl=-2_400.0,
            near_blow=True,
            dominant=_confusion(
                wait=(3, 1, 1),
                long=(1, 1, 0),
                short=(1, 0, 1),
            ),
            nonchop=_confusion(
                wait=(2, 1, 0),
                long=(0, 1, 0),
                short=(0, 0, 1),
            ),
            economics=[
                {
                    "episode_outcome": "timeout",
                    "static_regime": "dominant_chop",
                    "headroom_stratum": "low_headroom_le_0_25",
                    "side": "short",
                    "trades": 2,
                    "wins": 0,
                    "realized_r_sum": -2.0,
                    "mfe_r_sum": 0.5,
                    "mae_r_sum": 2.5,
                    "initial_stop_count": 2,
                }
            ],
        ),
    )
    diagnostics.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    report = analyze_pass_timeout_diagnostics(
        diagnostics,
        recent_window=1,
        output_dir=tmp_path / "analysis",
    )

    assert report["episodes"] == {
        "through_episode": 2,
        "count": 2,
        "passes": 1,
        "timeouts": 1,
        "blows": 0,
        "terminal_near_blow_timeouts": 1,
        "pass_rate": 0.5,
        "blow_rate": 0.0,
    }
    assert report["risk"] == {
        "terminal_near_blow_timeouts": 1,
        "episodes_with_low_headroom_entries": 1,
        "passes_with_low_headroom_entries": 0,
        "timeouts_with_low_headroom_entries": 1,
        "blows_with_low_headroom_entries": 0,
        "low_headroom_entry_trades": 2,
        "coverage_note": (
            "terminal_near_blow_timeouts counts only flagged terminal timeouts; "
            "low-headroom entries are a broader entry-time risk indicator, not "
            "a path-wise minimum-headroom measurement"
        ),
    }
    assert report["selection"]["overall"]["actions"]["long"] == {
        "targets": 4,
        "predictions": 4,
        "correct": 4,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert report["selection"]["overall"]["actions"]["short"]["recall"] == 0.5
    assert report["selection"]["overall"]["dominant_chop"] == {
        "rows": 18,
        "predicted_wait": 11,
        "predicted_long": 4,
        "predicted_short": 3,
        "entry_rate": 7 / 18,
        "wait_rate": 11 / 18,
        "exact_wait_recall": 7 / 10,
    }
    assert report["selection"]["recent"]["episode_range"] == [2, 2]
    assert report["challenge_return_self_imitation"] == {
        "rows": 4,
        "bonus_sum": 0.6,
        "bonus_mean": 0.15,
        "added_clip_rows": 0,
        "actions": {
            "WAIT": {"rows": 2, "bonus_sum": 0.2, "bonus_mean": 0.1},
            "ENTER_LONG_1": {
                "rows": 2,
                "bonus_sum": 0.4,
                "bonus_mean": 0.2,
            },
            "ENTER_SHORT_1": {
                "rows": 0,
                "bonus_sum": 0.0,
                "bonus_mean": None,
            },
        },
    }
    assert report["outcome_comparison"]["pass"]["nonchop"]["realized_r_sum"] == 3.0
    assert report["outcome_comparison"]["timeout"]["dominant_chop"][
        "realized_r_sum"
    ] == -2.0
    assert report["passes"][0]["nonchop_realized_r_share"] == 1.2
    output = Path(report["output_path"])
    assert output.parent == tmp_path / "analysis"
    assert json.loads(output.read_text())["identity_sha256"] == report[
        "identity_sha256"
    ]
    assert analyze_pass_timeout_diagnostics(
        diagnostics,
        recent_window=1,
        output_dir=tmp_path / "analysis",
    ) == report

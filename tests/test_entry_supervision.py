from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

import propevolve.entry_supervision as entry_supervision_module
from propevolve.decision import Action
from propevolve.balance_aware_regime_selectivity import (
    EXPANSION_CHANNELS,
    REGIME_TEACHER_CHANNELS,
)
from propevolve.entry_supervision import (
    build_entry_action_targets,
    build_post_launch_entry_supervision,
)
from propevolve.environment import MarketSeries
from propevolve.training import _paired_a_plus_transition_evidence


def _entry_spec() -> dict[str, object]:
    return {
        "schema": "post_launch_entry_v1",
        "training_only": True,
        "decision_count": 5,
        "fill_offsets": [1, 2, 3, 4, 5],
        "execution": "next_bar_open",
        "risk_dollars": 300,
        "launch": {"favorable_r": 0.5, "adverse_r": 0.25, "horizon_bars": 3},
        "continuation": {
            "favorable_r": 0.5,
            "adverse_r": 0.25,
            "horizon_bars": 3,
        },
        "target_r": 2.0,
        "stop_r": 1.0,
        "horizon_bars": 150,
        "collision": "stop_first",
        "loss_weight": 0.3,
    }


def _build_targets(market: MarketSeries | None = None):
    return build_entry_action_targets(
        {"NQ": market or _market()},
        _entry_spec(),
        point_values={"NQ": 148.0},
        round_trip_fees={"NQ": 4.0},
        training_end_exclusive="2025-01-01",
    )


def _market(
    *,
    rows: int = 160,
    launch_collision: bool = False,
    economic_stop_collision: bool = False,
    economic_high: float = 104.1,
) -> MarketSeries:
    open_ = np.full(rows, 100.0, dtype=np.float64)
    high = np.full(rows, 100.2, dtype=np.float64)
    low = np.full(rows, 99.8, dtype=np.float64)
    close = np.full(rows, 100.0, dtype=np.float64)
    # The decision bar is intentionally far from the fill.  A correct labeler
    # references open[1], never close[0].
    open_[0] = high[0] = low[0] = close[0] = 200.0
    high[1] = 101.1
    if launch_collision:
        low[1] = 99.4
    if rows > 4:
        high[4] = economic_high
        if economic_stop_collision:
            low[4] = 100.0 - 296.0 / 148.0
    timestamps = (
        np.datetime64("2024-01-01T00:00")
        + np.arange(rows) * np.timedelta64(3, "m")
    ).astype("datetime64[ns]")
    return MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        embeddings=np.zeros((rows, 1), dtype=np.float32),
    )


def _market_from_paths(high: np.ndarray, low: np.ndarray) -> MarketSeries:
    rows = len(high)
    timestamps = (
        np.datetime64("2024-01-01T00:00")
        + np.arange(rows) * np.timedelta64(3, "m")
    ).astype("datetime64[ns]")
    open_ = np.full(rows, 100.0, dtype=np.float64)
    close = np.full(rows, 100.0, dtype=np.float64)
    open_[0] = close[0] = 200.0
    high = np.asarray(high, dtype=np.float64).copy()
    low = np.asarray(low, dtype=np.float64).copy()
    high[0] = max(high[0], 200.0)
    low[0] = min(low[0], 200.0)
    return MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        embeddings=np.zeros((rows, 1), dtype=np.float32),
    )


def _short_market(*, rows: int = 160, economic_low: float = 95.9) -> MarketSeries:
    high = np.full(rows, 100.2)
    low = np.full(rows, 99.8)
    low[1] = 98.9
    low[4] = economic_low
    return _market_from_paths(high, low)


def _fill_five_market(*, rows: int = 160) -> MarketSeries:
    high = np.full(rows, 100.2)
    low = np.full(rows, 99.8)
    high[1] = 101.1
    # Stop-first collisions make fills 1-4 uneconomic without creating a
    # competing Short launch.  Fill 5 begins at index 5 and wins afterward.
    high[4] = 105.0
    low[4] = 97.9
    high[6] = 101.1
    high[7] = 104.1
    return _market_from_paths(high, low)


def _ambiguous_market(*, rows: int = 160) -> MarketSeries:
    high = np.full(rows, 100.2)
    low = np.full(rows, 99.8)
    high[1] = 101.1
    # A Short launch begins two decisions later, overlapping the Long event's
    # five-state window without making the first launch bar itself ambiguous.
    high[3] = 100.2
    low[3] = 98.9
    return _market_from_paths(high, low)


def _retimestamp(market: MarketSeries, *, start: str) -> MarketSeries:
    timestamps = (
        np.datetime64(start)
        + np.arange(len(market.timestamps)) * np.timedelta64(3, "m")
    ).astype("datetime64[ns]")
    return MarketSeries(
        ticker=market.ticker,
        timestamps=timestamps,
        open=market.open,
        high=market.high,
        low=market.low,
        close=market.close,
        embeddings=market.embeddings,
    )


def _reticker(market: MarketSeries, *, ticker: str) -> MarketSeries:
    return MarketSeries(
        ticker=ticker,
        timestamps=market.timestamps,
        open=market.open,
        high=market.high,
        low=market.low,
        close=market.close,
        embeddings=market.embeddings,
    )


def test_earliest_long_economic_good_is_enter_and_later_states_are_censored() -> None:
    supervision = build_post_launch_entry_supervision(
        long_launch=True,
        short_launch=False,
        long_economic_good=(False, False, True, True, False),
        short_economic_good=(False, False, False, False, False),
    )

    assert supervision.side == "long"
    assert supervision.status == "enter"
    assert supervision.enter_fill_offset == 3
    assert [
        (
            state.decision_offset,
            state.fill_offset,
            state.action,
            state.available,
            state.censored,
            state.unavailable_reason,
        )
        for state in supervision.candidates
    ] == [
        (0, 1, "WAIT", True, False, None),
        (1, 2, "WAIT", True, False, None),
        (2, 3, "ENTER_LONG_1", True, False, None),
        (3, 4, None, False, True, "after_entry"),
        (4, 5, None, False, True, "after_entry"),
    ]


def test_short_side_uses_its_independent_economic_targets() -> None:
    supervision = build_post_launch_entry_supervision(
        long_launch=False,
        short_launch=True,
        long_economic_good=(False, False, False, False, False),
        short_economic_good=(False, False, False, False, True),
    )

    assert supervision.side == "short"
    assert supervision.status == "enter"
    assert supervision.enter_fill_offset == 5
    assert [state.action for state in supervision.candidates] == [
        "WAIT",
        "WAIT",
        "WAIT",
        "WAIT",
        "ENTER_SHORT_1",
    ]
    assert [state.fill_offset for state in supervision.candidates] == [1, 2, 3, 4, 5]


def test_no_good_entry_keeps_all_five_states_available_as_wait() -> None:
    supervision = build_post_launch_entry_supervision(
        long_launch=True,
        short_launch=False,
        long_economic_good=(False, False, False, False, False),
        short_economic_good=(False, False, False, False, False),
    )

    assert supervision.side == "long"
    assert supervision.status == "abstain"
    assert supervision.enter_fill_offset is None
    assert [state.action for state in supervision.candidates] == ["WAIT"] * 5
    assert all(state.available for state in supervision.candidates)
    assert not any(state.censored for state in supervision.candidates)
    assert not any(
        state.unavailable_reason is not None for state in supervision.candidates
    )


def test_overlapping_long_and_short_launches_are_unavailable_not_direction_selected() -> None:
    supervision = build_post_launch_entry_supervision(
        long_launch=True,
        short_launch=True,
        long_economic_good=(True, False, False, False, False),
        short_economic_good=(False, True, False, False, False),
    )

    assert supervision.side == "ambiguous"
    assert supervision.status == "ambiguous"
    assert supervision.enter_fill_offset is None
    assert all(state.action is None for state in supervision.candidates)
    assert not any(state.available for state in supervision.candidates)
    assert not any(state.censored for state in supervision.candidates)
    assert all(
        state.unavailable_reason == "ambiguous_side"
        for state in supervision.candidates
    )


def test_post_launch_builder_rejects_an_event_without_a_launch_side() -> None:
    with pytest.raises(ValueError, match="at least one launch side"):
        build_post_launch_entry_supervision(
            long_launch=False,
            short_launch=False,
            long_economic_good=(False, False, False, False, False),
            short_economic_good=(False, False, False, False, False),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("long_economic_good", (False,) * 4, "exactly 5 bool values"),
        ("short_economic_good", (False,) * 6, "exactly 5 bool values"),
        (
            "long_economic_good",
            (False, False, 1, False, False),
            "exactly 5 bool values",
        ),
        (
            "short_economic_good",
            (False, False, "false", False, False),
            "exactly 5 bool values",
        ),
    ],
)
def test_post_launch_builder_requires_exactly_five_strict_bool_targets(
    field: str,
    value: tuple[object, ...],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "long_launch": True,
        "short_launch": False,
        "long_economic_good": (False,) * 5,
        "short_economic_good": (False,) * 5,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        build_post_launch_entry_supervision(**arguments)


def test_post_launch_builder_requires_strict_bool_launch_flags() -> None:
    with pytest.raises(TypeError, match="launch flags must be bool"):
        build_post_launch_entry_supervision(
            long_launch=1,
            short_launch=False,
            long_economic_good=(False,) * 5,
            short_economic_good=(False,) * 5,
        )


def test_economic_builder_uses_next_open_and_fee_inclusive_300_dollar_risk() -> None:
    targets = _build_targets()

    assert targets.target("NQ", 0) == Action.ENTER_LONG_1
    metadata = targets.metadata("NQ", 0)
    assert metadata is not None
    assert metadata.side == "long"
    assert metadata.event_anchor_rows == (0,)
    assert metadata.candidate_count == 5
    assert metadata.candidate_decision_offset == 0
    assert metadata.fill_offset == 1
    assert metadata.continuation is True
    assert metadata.economic_win is True
    assert metadata.economic_good is True
    assert targets.target("NQ", 1) is None
    after_entry = targets.metadata("NQ", 1)
    assert after_entry.unavailable_reason == "after_entry"
    assert after_entry.available is False
    assert after_entry.censored is True
    # Censoring still excludes the row from learning. The outcome remains
    # available only to the post-policy Stage 1 timing audit so it can separate
    # a late but valid entry from entry after invalidation.
    assert after_entry.continuation is True
    assert after_entry.economic_win is True
    assert after_entry.economic_good is True

    assert isinstance(targets.manifest, MappingProxyType)
    assert targets.manifest["schema"] == "post_launch_entry_v1"
    assert targets.manifest["training_end_exclusive"] == "2025-01-01"
    assert targets.manifest["point_values"] == {"NQ": 148.0}
    assert targets.manifest["round_trip_fees"] == {"NQ": 4.0}
    assert targets.manifest["label_semantics"]["fresh_lookback_bars"] == 5
    assert targets.manifest["label_semantics"]["action_order"] == (
        "WAIT",
        "ENTER_LONG_1",
        "ENTER_SHORT_1",
    )
    assert targets.manifest["identity_sha256"]


def test_manifest_reports_exact_action_target_counts_per_market_and_aggregate() -> None:
    # One Long winner, one Short winner, and one launch with no executable
    # economic entry provide a literal 5 WAIT : 1 LONG : 1 SHORT fixture.
    markets = {
        "NQ": _market(),
        "ES": _reticker(_short_market(), ticker="ES"),
        "YM": _reticker(
            _market(economic_high=100.0 + 600.0 / 148.0),
            ticker="YM",
        ),
    }
    targets = build_entry_action_targets(
        markets,
        _entry_spec(),
        point_values={ticker: 148.0 for ticker in markets},
        round_trip_fees={ticker: 4.0 for ticker in markets},
        training_end_exclusive="2025-01-01",
    )

    expected_aggregate = {
        "WAIT": 5,
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 1,
    }
    assert targets.manifest["action_target_counts"] == expected_aggregate
    assert targets.manifest["markets"]["NQ"]["action_target_counts"] == {
        "WAIT": 0,
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 0,
    }
    assert targets.manifest["markets"]["ES"]["action_target_counts"] == {
        "WAIT": 0,
        "ENTER_LONG_1": 0,
        "ENTER_SHORT_1": 1,
    }
    assert targets.manifest["markets"]["YM"]["action_target_counts"] == {
        "WAIT": 5,
        "ENTER_LONG_1": 0,
        "ENTER_SHORT_1": 0,
    }
    assert sum(expected_aggregate.values()) == sum(
        sum(market["action_target_counts"].values())
        for market in targets.manifest["markets"].values()
    )
    receipt = targets.balance_receipt()
    assert receipt["source_manifest_identity_sha256"] == targets.manifest[
        "identity_sha256"
    ]
    assert receipt["target_counts"] == expected_aggregate
    assert receipt["class_weights"] == {
        "WAIT": pytest.approx(7.0 / 15.0),
        "ENTER_LONG_1": pytest.approx(7.0 / 3.0),
        "ENTER_SHORT_1": pytest.approx(7.0 / 3.0),
    }
    assert len(receipt["identity_sha256"]) == 64


def test_inverse_frequency_entry_weights_use_authenticated_action_counts() -> None:
    counts = {
        "WAIT": 4,
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 1,
    }

    weights = entry_supervision_module.inverse_frequency_entry_action_class_weights(
        counts
    )

    # N / (3 * class_count) preserves expected sample weight one while giving
    # exact inverse-frequency contribution in WAIT/LONG/SHORT order.
    assert weights == (0.5, 2.0, 2.0)


@pytest.mark.parametrize(
    "zero_action",
    ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"),
)
def test_inverse_frequency_entry_weights_fail_closed_on_a_zero_class(
    zero_action: str,
) -> None:
    counts = {
        "WAIT": 10,
        "ENTER_LONG_1": 2,
        "ENTER_SHORT_1": 2,
    }
    counts[zero_action] = 0

    with pytest.raises(ValueError, match="strictly positive"):
        entry_supervision_module.inverse_frequency_entry_action_class_weights(
            counts
        )


def test_gross_two_r_without_round_trip_fee_does_not_satisfy_net_two_r() -> None:
    gross_two_r_high = 100.0 + 600.0 / 148.0
    targets = _build_targets(_market(economic_high=gross_two_r_high))

    assert targets.target("NQ", 0) == Action.WAIT
    metadata = targets.metadata("NQ", 0)
    assert metadata.continuation is True
    assert metadata.economic_win is False
    assert metadata.economic_good is False
    assert [targets.target("NQ", row) for row in range(5)] == [Action.WAIT] * 5


def test_net_two_r_including_round_trip_fee_satisfies_economic_target() -> None:
    fee_inclusive_two_r_high = 100.0 + 604.0 / 148.0
    targets = _build_targets(_market(economic_high=fee_inclusive_two_r_high))

    assert targets.target("NQ", 0) == Action.ENTER_LONG_1
    assert targets.metadata("NQ", 0).economic_win is True


def test_exact_two_r_label_builds_paired_winner_and_non_hit_training_evidence() -> None:
    channels = (*EXPANSION_CHANNELS, *REGIME_TEACHER_CHANNELS)
    context = np.asarray((0.85, 0.10, 0.15, 0.05, 0.10, 0.35, 0.55), dtype=np.float32)
    winner_targets = _build_targets(
        _market(economic_high=100.0 + 604.0 / 148.0)
    )
    horizon_non_hit_targets = _build_targets(
        _market(economic_high=100.0 + 600.0 / 148.0)
    )
    stopped_non_hit_targets = _build_targets(
        _market(
            economic_high=100.0 + 604.0 / 148.0,
            economic_stop_collision=True,
        )
    )
    short_winner_targets = _build_targets(_short_market())
    short_horizon_non_hit_targets = _build_targets(
        _short_market(economic_low=96.0)
    )

    winner = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=channels,
        entry_action_target=winner_targets.target("NQ", 0),
        metadata=winner_targets.metadata("NQ", 0),
    )
    horizon_non_hit = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=channels,
        entry_action_target=horizon_non_hit_targets.target("NQ", 0),
        metadata=horizon_non_hit_targets.metadata("NQ", 0),
    )
    stopped_non_hit = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=channels,
        entry_action_target=stopped_non_hit_targets.target("NQ", 0),
        metadata=stopped_non_hit_targets.metadata("NQ", 0),
    )
    short_winner = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=channels,
        entry_action_target=short_winner_targets.target("NQ", 0),
        metadata=short_winner_targets.metadata("NQ", 0),
    )
    short_horizon_non_hit = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=channels,
        entry_action_target=short_horizon_non_hit_targets.target("NQ", 0),
        metadata=short_horizon_non_hit_targets.metadata("NQ", 0),
    )

    np.testing.assert_array_equal(winner[0], context)
    assert winner[1:] == (Action.ENTER_LONG_1, True)
    np.testing.assert_array_equal(horizon_non_hit[0], context)
    assert horizon_non_hit[1:] == (Action.ENTER_LONG_1, False)
    np.testing.assert_array_equal(stopped_non_hit[0], context)
    assert stopped_non_hit[1:] == (Action.ENTER_LONG_1, False)
    np.testing.assert_array_equal(short_winner[0], context)
    assert short_winner[1:] == (Action.ENTER_SHORT_1, True)
    np.testing.assert_array_equal(short_horizon_non_hit[0], context)
    assert short_horizon_non_hit[1:] == (Action.ENTER_SHORT_1, False)


def test_economic_stop_wins_when_two_r_and_minus_one_r_touch_same_bar() -> None:
    targets = _build_targets(
        _market(economic_high=100.0 + 604.0 / 148.0, economic_stop_collision=True)
    )

    assert targets.target("NQ", 0) == Action.WAIT
    assert targets.metadata("NQ", 0).continuation is True
    assert targets.metadata("NQ", 0).economic_win is False


def test_launch_adverse_touch_wins_over_favorable_touch_on_same_bar() -> None:
    targets = _build_targets(_market(launch_collision=True))

    assert targets.target("NQ", 0) is None
    assert targets.metadata("NQ", 0) is None


def test_short_labels_are_a_mirrored_independent_target_family() -> None:
    targets = _build_targets(_short_market())

    assert targets.target("NQ", 0) == Action.ENTER_SHORT_1
    metadata = targets.metadata("NQ", 0)
    assert metadata.side == "short"
    assert metadata.fill_offset == 1
    assert metadata.continuation is True
    assert metadata.economic_win is True


def test_full_builder_can_wait_until_exactly_the_fifth_fill() -> None:
    targets = _build_targets(_fill_five_market())

    assert [targets.target("NQ", row) for row in range(5)] == [
        Action.WAIT,
        Action.WAIT,
        Action.WAIT,
        Action.WAIT,
        Action.ENTER_LONG_1,
    ]
    assert targets.metadata("NQ", 4).candidate_decision_offset == 4
    assert targets.metadata("NQ", 4).fill_offset == 5


def test_split_end_event_with_incomplete_economic_path_is_unavailable() -> None:
    targets = _build_targets(_market(rows=154))

    assert targets.target("NQ", 0) is None
    metadata = targets.metadata("NQ", 0)
    assert metadata is not None
    assert metadata.available is False
    assert metadata.unavailable_reason == "unresolved_split_end"


def test_outcome_path_may_not_cross_the_2025_training_boundary() -> None:
    market = _retimestamp(_market(), start="2024-12-31T16:18")
    targets = _build_targets(market)

    # Row 154 opens exactly at 2025-01-01 and is physically excluded.  The
    # anchor therefore lacks a complete five-fill plus 150-bar outcome path.
    assert market.timestamps[154] == np.datetime64("2025-01-01T00:00")
    assert targets.manifest["markets"]["NQ"]["training_rows"] == 154
    assert targets.target("NQ", 0) is None
    assert targets.metadata("NQ", 0).unavailable_reason == "unresolved_split_end"
    assert targets.target("NQ", 154) is None
    assert targets.metadata("NQ", 154) is None


def test_overlapping_opposite_side_events_are_masked_by_full_builder() -> None:
    targets = _build_targets(_ambiguous_market())

    for row in range(5):
        assert targets.target("NQ", row) is None
        metadata = targets.metadata("NQ", row)
        assert metadata is not None
        assert metadata.side == "ambiguous"
        assert metadata.unavailable_reason == "ambiguous_side"


def test_target_storage_is_compact_read_only_and_metadata_is_sparse() -> None:
    targets = _build_targets()
    encoded = targets._targets["NQ"]

    assert encoded.dtype == np.int8
    assert encoded.nbytes == len(encoded)
    assert encoded.flags.writeable is False
    assert isinstance(targets._metadata["NQ"], MappingProxyType)
    assert len(targets._metadata["NQ"]) < len(encoded)
    assert targets.manifest["storage"] == {
        "target_dtype": "int8",
        "unavailable_sentinel": -1,
        "metadata": "sparse_mapping_by_supervised_row",
    }


def test_builder_rejects_contract_drift_and_runtime_economic_drift() -> None:
    drifted = _entry_spec()
    drifted["fill_offsets"] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="fill_offsets contract drifted"):
        build_entry_action_targets(
            {"NQ": _market()},
            drifted,
            point_values={"NQ": 148.0},
            round_trip_fees={"NQ": 4.0},
            training_end_exclusive="2025-01-01",
        )


def test_builder_uses_alternate_configured_entry_recipe() -> None:
    specification = _entry_spec()
    specification.update({
        "decision_count": 3,
        "fill_offsets": [1, 3, 5],
        "risk_dollars": 450.0,
        "launch": {
            "favorable_r": 0.6,
            "adverse_r": 0.3,
            "horizon_bars": 4,
        },
        "continuation": {
            "favorable_r": 0.75,
            "adverse_r": 0.35,
            "horizon_bars": 5,
        },
        "target_r": 3.0,
        "stop_r": 1.25,
        "horizon_bars": 180,
    })

    targets = build_entry_action_targets(
        {"NQ": _market(rows=220)},
        specification,
        point_values={"NQ": 148.0},
        round_trip_fees={"NQ": 4.0},
        training_end_exclusive="2025-02-01",
    )

    assert targets.manifest["contract"]["decision_count"] == 3
    assert targets.manifest["contract"]["fill_offsets"] == (1, 3, 5)
    assert targets.manifest["contract"]["target_r"] == 3.0
    assert targets.manifest["training_end_exclusive"] == "2025-02-01"

    with pytest.raises(ValueError, match="point_values must exactly match markets"):
        build_entry_action_targets(
            {"NQ": _market()},
            _entry_spec(),
            point_values={},
            round_trip_fees={"NQ": 4.0},
            training_end_exclusive="2025-01-01",
        )


def test_builder_reports_each_market_start_and_authenticated_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = _build_targets()

    output = capsys.readouterr().out
    assert "[entry-supervision] START ticker=NQ" in output
    assert "[entry-supervision] COMPLETE ticker=NQ" in output
    assert f"identity={targets.manifest['markets']['NQ']['targets_sha256']}" in output
    assert "enter_targets=" in output
    assert "wait_targets=" in output

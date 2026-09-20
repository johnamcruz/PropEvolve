"""Expansion + order-flow setup channels: alignment, causality and availability.

These channels are the only thing that tells the policy what the setup is, so an
off-by-one or a silently imputed value would be indistinguishable from the policy
learning something. Each test pins one way that could go wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.setup_signals import (
    CHANNEL_NAMES,
    SetupSignalSpec,
    entry_side_from_channels,
    load_setup_channels,
)

STAMPS = pd.date_range("2024-03-05 14:30", periods=6, freq="3min")


def _bundle(tmp_path, **overrides):
    frame = pd.DataFrame({
        "bar_open_utc": STAMPS,
        "expansion_pooled_score": [0.10, 0.90, 0.40, 0.55, 0.20, 0.30],
        "armed": [False, True, True, True, False, False],
        "persistence_45": [0.00, 0.30, -0.25, 0.10, 0.05, -0.40],
        "trigger": [False, True, False, True, False, False],
        "side": [0, 1, 0, 1, 0, 0],
    })
    for key, value in overrides.items():
        frame[key] = value
    path = tmp_path / "signals.parquet"
    frame.to_parquet(path, index=False)
    return path


def _index(name):
    return CHANNEL_NAMES.index(name)


# ─────────────────────────────── spec
def test_the_spec_is_off_by_default_so_existing_runs_are_unchanged():
    spec = SetupSignalSpec()
    assert spec.state == "off" and spec.gate_entries is False and spec.output_dim == 0


def test_enabling_the_signal_adds_one_channel_per_ingredient_plus_availability():
    assert SetupSignalSpec(state="expansion_flow_v1").output_dim == len(CHANNEL_NAMES)
    assert CHANNEL_NAMES[-1] == "setup_available"


def test_gating_without_the_signal_is_rejected():
    with pytest.raises(ValueError):
        SetupSignalSpec(state="off", gate_entries=True)
    with pytest.raises(ValueError):
        SetupSignalSpec(state="nonsense")


def test_the_spec_round_trips_through_config():
    spec = SetupSignalSpec.from_config({"state": "expansion_flow_v1", "gate_entries": True})
    assert spec.state == "expansion_flow_v1" and spec.gate_entries is True
    assert SetupSignalSpec.from_config(None).state == "off"


# ─────────────────────────────── alignment
def test_channels_align_to_the_bar_that_produced_them(tmp_path):
    out = load_setup_channels(_bundle(tmp_path), STAMPS.to_numpy())
    assert out.shape == (6, len(CHANNEL_NAMES))
    assert out[1, _index("expansion_score")] == pytest.approx(0.90)
    assert out[1, _index("expansion_armed")] == pytest.approx(1.0)
    assert out[1, _index("setup_trigger")] == pytest.approx(1.0)
    assert out[1, _index("setup_side")] == pytest.approx(1.0)
    assert out[0, _index("setup_trigger")] == pytest.approx(0.0)


def test_flow_persistence_is_scaled_into_the_unit_range_without_clipping(tmp_path):
    out = load_setup_channels(_bundle(tmp_path), STAMPS.to_numpy())
    assert out[1, _index("flow_persistence")] == pytest.approx(0.60)     # 0.30 x 2
    assert out[2, _index("flow_persistence")] == pytest.approx(-0.50)    # -0.25 x 2
    assert np.abs(out[:, _index("flow_persistence")]).max() <= 1.0


def test_a_bar_absent_from_the_bundle_is_zero_and_flagged_unavailable(tmp_path):
    extra = STAMPS.append(pd.DatetimeIndex(["2024-03-05 14:48"]))
    out = load_setup_channels(_bundle(tmp_path), extra.to_numpy())
    assert out[-1, _index("setup_available")] == 0.0
    assert np.all(out[-1, :-1] == 0.0)
    assert out[1, _index("setup_available")] == 1.0


def test_an_unscored_expansion_bar_is_unavailable_rather_than_scored_zero(tmp_path):
    """The bundle encodes 'no score' as a negative sentinel; treating it as 0.0 would
    read as a confident 'no expansion' instead of 'we do not know'."""
    path = _bundle(tmp_path, expansion_pooled_score=[-1.0, 0.9, 0.4, 0.55, 0.2, 0.3])
    out = load_setup_channels(path, STAMPS.to_numpy())
    assert out[0, _index("setup_available")] == 0.0
    assert out[0, _index("expansion_score")] == 0.0
    assert out[1, _index("setup_available")] == 1.0


def test_a_missing_flow_value_makes_the_row_unavailable(tmp_path):
    """Timing without a side is not this setup, so the row must not look usable."""
    path = _bundle(tmp_path, persistence_45=[0.0, np.nan, -0.25, 0.10, 0.05, -0.40])
    out = load_setup_channels(path, STAMPS.to_numpy())
    assert out[1, _index("setup_available")] == 0.0
    assert np.all(out[1, :-1] == 0.0)
    assert out[3, _index("setup_available")] == 1.0


def test_a_bundle_covering_only_part_of_the_window_is_not_smeared(tmp_path):
    """searchsorted must not attach the nearest row to bars the bundle never covered."""
    short = pd.DataFrame({
        "bar_open_utc": STAMPS[:2], "expansion_pooled_score": [0.1, 0.9],
        "armed": [False, True], "persistence_45": [0.0, 0.3],
        "trigger": [False, True], "side": [0, 1]})
    path = tmp_path / "short.parquet"
    short.to_parquet(path, index=False)
    out = load_setup_channels(path, STAMPS.to_numpy())
    assert out[1, _index("setup_available")] == 1.0
    assert np.all(out[2:, _index("setup_available")] == 0.0)


def test_everything_returned_is_finite_float32(tmp_path):
    out = load_setup_channels(_bundle(tmp_path), STAMPS.to_numpy())
    assert out.dtype == np.float32 and np.isfinite(out).all()


# ─────────────────────────────── guards
def test_a_missing_or_malformed_bundle_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_setup_channels(tmp_path / "nope.parquet", STAMPS.to_numpy())
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"bar_open_utc": STAMPS, "armed": [False] * 6}).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_setup_channels(bad, STAMPS.to_numpy())


def test_unsorted_or_empty_timestamps_are_rejected(tmp_path):
    path = _bundle(tmp_path)
    with pytest.raises(ValueError):
        load_setup_channels(path, STAMPS.to_numpy()[::-1])
    with pytest.raises(ValueError):
        load_setup_channels(path, np.array([], dtype="datetime64[ns]"))


# ─────────────────────────────── the rule's side, for the optional gate
def test_the_rule_side_is_only_read_on_an_available_triggered_bar(tmp_path):
    out = load_setup_channels(_bundle(tmp_path), STAMPS.to_numpy())
    assert entry_side_from_channels(out[1]) == 1       # triggered long
    assert entry_side_from_channels(out[2]) == 0       # armed but no trigger
    assert entry_side_from_channels(out[0]) == 0       # nothing


def test_a_short_setup_reads_as_minus_one(tmp_path):
    path = _bundle(tmp_path, trigger=[False, True, False, False, False, False],
                   side=[0, -1, 0, 0, 0, 0], persistence_45=[0.0, -0.3, 0.0, 0.0, 0.0, 0.0])
    out = load_setup_channels(path, STAMPS.to_numpy())
    assert entry_side_from_channels(out[1]) == -1


def test_a_trigger_on_an_unavailable_bar_yields_no_side(tmp_path):
    path = _bundle(tmp_path, expansion_pooled_score=[-1.0, -1.0, 0.4, 0.55, 0.2, 0.3])
    out = load_setup_channels(path, STAMPS.to_numpy())
    assert entry_side_from_channels(out[1]) == 0


def test_a_wrong_width_row_is_rejected():
    with pytest.raises(ValueError):
        entry_side_from_channels(np.zeros(3))


def test_underscore_keys_are_notes_but_a_misspelled_setting_still_raises():
    """Configs carry their rationale inline; a typo must not be silently ignored."""
    spec = SetupSignalSpec.from_config(
        {"state": "expansion_flow_v1", "_gate_note": "why the gate is off"})
    assert spec.state == "expansion_flow_v1"
    with pytest.raises(TypeError):
        SetupSignalSpec.from_config({"state": "expansion_flow_v1", "gate_entires": True})

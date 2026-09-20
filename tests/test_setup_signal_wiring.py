"""The setup channels must reach the policy on the RIGHT bar, or not at all.

Three ways this silently breaks, each pinned here: the feature leaking into runs that did
not ask for it, the channels arriving on the wrong bar because this cache is indexed by
bar CLOSE while the research bundle is indexed by bar OPEN, and a market being accepted
with channels missing or the wrong width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries
from propevolve.observation import AccountState, ObservationAssembler
from propevolve.setup_signals import CHANNEL_NAMES, SetupSignalSpec

WIDTH = len(CHANNEL_NAMES)


def _market(n=40, channels=None, ticker="NQ"):
    # hourly bars so a short series still spans several CME sessions (the env needs
    # episode_days distinct sessions); the channel wiring is timeframe-agnostic
    stamps = pd.date_range("2024-03-05 14:33", periods=n, freq="1h").to_numpy("datetime64[ns]")
    close = np.full(n, 100.0, dtype=np.float32)
    return MarketSeries(
        ticker=ticker, timestamps=stamps,
        open=close.copy(), high=close + 1, low=close - 1, close=close,
        embeddings=np.zeros((n, 4), dtype=np.float32),
        setup_channels=channels,
    )


# ─────────────────────────────── the assembler
def test_the_observation_grows_by_exactly_the_channel_count():
    off = ObservationAssembler(4, max_loss=3000, profit_target=6000)
    on = ObservationAssembler(4, max_loss=3000, profit_target=6000,
                              setup_signals=SetupSignalSpec(state="expansion_flow_v1"))
    assert on.output_dim - off.output_dim == WIDTH


def test_the_channels_are_appended_verbatim_and_last():
    on = ObservationAssembler(4, max_loss=3000, profit_target=6000,
                              setup_signals=SetupSignalSpec(state="expansion_flow_v1"))
    row = np.arange(WIDTH, dtype=np.float32) / 10.0
    out = on.assemble(np.zeros(4, np.float32), AccountState(), row)
    assert np.allclose(out[-WIDTH:], row)


def test_enabling_the_channels_without_supplying_them_fails_loudly():
    on = ObservationAssembler(4, max_loss=3000, profit_target=6000,
                              setup_signals=SetupSignalSpec(state="expansion_flow_v1"))
    with pytest.raises(ValueError, match="none were supplied"):
        on.assemble(np.zeros(4, np.float32), AccountState())
    with pytest.raises(ValueError, match="setup shape"):
        on.assemble(np.zeros(4, np.float32), AccountState(), np.zeros(WIDTH - 1, np.float32))


def test_supplying_channels_when_the_feature_is_off_is_rejected():
    """Silently ignoring them would make a misconfigured run look like a working one."""
    off = ObservationAssembler(4, max_loss=3000, profit_target=6000)
    with pytest.raises(ValueError, match="feature is off"):
        off.assemble(np.zeros(4, np.float32), AccountState(), np.zeros(WIDTH, np.float32))


def test_non_finite_channels_are_rejected():
    on = ObservationAssembler(4, max_loss=3000, profit_target=6000,
                              setup_signals=SetupSignalSpec(state="expansion_flow_v1"))
    bad = np.zeros(WIDTH, np.float32); bad[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        on.assemble(np.zeros(4, np.float32), AccountState(), bad)


# ─────────────────────────────── MarketSeries
def test_a_market_without_channels_is_unchanged():
    assert _market().setup_channels is None


def test_a_market_rejects_channels_of_the_wrong_shape():
    with pytest.raises(ValueError, match="one row per bar"):
        _market(channels=np.zeros((39, WIDTH), dtype=np.float32))
    with pytest.raises(ValueError, match="one row per bar"):
        _market(channels=np.zeros((40, WIDTH - 1), dtype=np.float32))


def test_a_market_rejects_non_finite_channels():
    channels = np.zeros((40, WIDTH), dtype=np.float32)
    channels[3, 0] = np.inf
    with pytest.raises(ValueError, match="setup channels must be finite"):
        _market(channels=channels)


# ─────────────────────────────── the environment
def _spec():
    return ChallengeSpec(
        profit_target=6_000.0, max_loss=3_000.0, episode_days=2, bars_per_day=10,
        max_position_size=1, minimum_mll_headroom=500.0, trailing_mll_lock=True,
        terminal_pass_reward=250.0, terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0, terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0, per_trade_risk_dollars=500.0,
        ratchet_activation_r=2.0, ratchet_giveback_r=0.5, ratchet_lock_floor_r=2.0)


def _env(markets, setup_signals=None):
    return HistoricalChallengeEnv(
        markets, tick_values={"NQ": 20.0}, spec=_spec(),
        round_trip_fees={"NQ": 3.84}, seed=1, setup_signals=setup_signals)


def test_the_environment_refuses_to_enable_channels_a_market_does_not_have():
    with pytest.raises(ValueError, match="a market has none"):
        _env({"NQ": _market()}, SetupSignalSpec(state="expansion_flow_v1"))


def test_the_environment_serves_the_current_bar_s_channels():
    channels = np.zeros((40, WIDTH), dtype=np.float32)
    channels[:, 0] = np.arange(40) / 100.0          # a per-bar fingerprint
    env = _env({"NQ": _market(channels=channels)},
               SetupSignalSpec(state="expansion_flow_v1"))
    observation, _ = env.reset()
    index = env._index
    assert observation[-WIDTH] == pytest.approx(channels[index, 0])


def test_the_default_environment_is_byte_identical_to_before():
    plain = _env({"NQ": _market()})
    observation, _ = plain.reset()
    assert len(observation) == plain._assembler.output_dim
    assert plain.setup_signals.output_dim == 0


# ─────────────────────────────── the clock, which is the dangerous part
def test_the_loader_shifts_the_bundle_from_bar_open_to_bar_close(tmp_path, monkeypatch):
    """The research bundle is indexed by bar OPEN and this cache by bar CLOSE. A missing
    shift would attach every signal to the bar before it and look like noise."""
    from propevolve.setup_signals import align_setup_channels, read_setup_bundle

    opens = pd.date_range("2024-03-05 14:30", periods=4, freq="3min")
    pd.DataFrame({
        "bar_open_utc": opens,
        "expansion_pooled_score": [0.1, 0.9, 0.2, 0.3],
        "armed": [False, True, False, False],
        "persistence_45": [0.0, 0.3, 0.0, 0.0],
        "trigger": [False, True, False, False],
        "side": [0, 1, 0, 0],
    }).to_parquet(tmp_path / "signals.parquet", index=False)

    bundle_opens, channels = read_setup_bundle(tmp_path / "signals.parquet")
    closes = bundle_opens + np.timedelta64(3, "m")
    target = (opens + pd.Timedelta(minutes=3)).to_numpy("datetime64[ns]")
    shifted = align_setup_channels(closes, channels, target)
    assert shifted[1, CHANNEL_NAMES.index("setup_trigger")] == 1.0

    unshifted = align_setup_channels(bundle_opens, channels, target)
    assert unshifted[1, CHANNEL_NAMES.index("setup_trigger")] == 0.0   # the bug it prevents
    assert unshifted[:, CHANNEL_NAMES.index("setup_available")].sum() < \
        shifted[:, CHANNEL_NAMES.index("setup_available")].sum()


def test_setup_fields_are_legal_embedding_mode_inputs_but_teacher_fields_are_not(tmp_path):
    """The embedding-mode guard exists to keep TEACHER probabilities out of the inputs.
    The setup channels are frozen upstream market context, available identically at
    inference and never a distillation target, so they belong with account/trade."""
    import json as _json
    from propevolve.reasoning_policy.context import ContextConfig

    def _write(fields):
        path = tmp_path / f"ctx_{abs(hash(tuple(fields)))}.json"
        path.write_text(_json.dumps({"context_steps": 4, "input_mode": "embeddings",
                                     "fields": list(fields)}))
        return path

    ContextConfig.load(_write(["trade.open", "setup.flow_persistence"]))
    with pytest.raises(ValueError, match="teacher fields cannot be inputs"):
        ContextConfig.load(_write(["trade.open", "expansion.long_attempt_probability"]))

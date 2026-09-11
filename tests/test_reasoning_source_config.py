import json

import pytest


def source_payload():
    return {
        "schema": "propevolve_reasoning_market_source_v1",
        "assets": "config/local-assets.json",
        "tickers": ["NQ"],
        "timeframe_minutes": 3,
        "cache_root": "cache/embeddings",
        "teachers": [
            {"kind": "expansion", "cache_root": "cache/expansion",
             "channels": ["long", "short"], "loss_weight": .2,
             "entry_search_loss_weight": 0.,
             "entry_search_objective": "raw_probability"},
            {"kind": "regime", "cache_root": "cache/regime",
             "channels": ["chop"], "loss_weight": .1,
             "entry_search_loss_weight": 0.},
            {"kind": "trend", "cache_root": "cache/trend",
             "channels": ["long", "short"], "loss_weight": 0.,
             "entry_search_loss_weight": 0.},
        ],
        "observation": {"management_state": "off"},
        "challenge": {
            "profit_target": 6000., "max_loss": 3000., "episode_days": 30,
            "bars_per_day": 480, "max_position_size": 1,
            "minimum_mll_headroom": 500., "trailing_mll_lock": True,
            "terminal_pass_reward": 250., "terminal_blow_reward": -1500.,
            "terminal_timeout_reward": -2.,
            "terminal_pass_speed_reward_per_day": 20., "reward_scale": 1000.,
        },
        "point_values": {"NQ": 20.}, "round_trip_fees": {"NQ": 3.84},
        "temporal": {"train_start": "2021-01-01", "train_end": "2025-01-01",
                     "validation_start": "2025-01-01",
                     "validation_end": "2026-01-01", "sealed_start": "2026-01-01"},
    }


def test_reasoning_source_accepts_only_shared_market_and_simulator_contract(tmp_path):
    from propevolve.reasoning_policy.source_config import load_source_recipe
    path = tmp_path / "source.json"
    payload = source_payload()
    path.write_text(json.dumps(payload))
    assert load_source_recipe(path) == payload
    payload["agent"] = {"hidden_dim": 128}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reasoning market source fields"):
        load_source_recipe(path)


def test_reasoning_source_rejects_missing_or_misaligned_market_economics(tmp_path):
    from propevolve.reasoning_policy.source_config import load_source_recipe
    path = tmp_path / "source.json"
    payload = source_payload()
    payload["round_trip_fees"] = {"ES": 3.84}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="market economics"):
        load_source_recipe(path)


def test_reasoning_source_accepts_volume_only_as_fourth_training_teacher(tmp_path):
    from propevolve.reasoning_policy.source_config import load_source_recipe
    payload = source_payload()
    payload["teachers"].append({
        "kind": "volume",
        "cache_root": "cache/volume",
        "channels": ["long_participation", "short_participation"],
        "loss_weight": 0.1,
        "entry_search_loss_weight": 0.0,
    })
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload))

    assert [row["kind"] for row in load_source_recipe(path)["teachers"]] == [
        "expansion", "regime", "trend", "volume"
    ]

    payload["teachers"][2], payload["teachers"][3] = (
        payload["teachers"][3], payload["teachers"][2]
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Expansion, Regime, Trend"):
        load_source_recipe(path)

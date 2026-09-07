from test_reasoning_challenger_e2e import environment

import numpy as np

from propevolve.reasoning_policy.job import collection_source_for_role, sample_episode_specs


def test_episode_sampling_is_explicit_balanced_and_deterministic():
    first = sample_episode_specs(environment(), tickers=("NQ",), count=4, seed=19)
    second = sample_episode_specs(environment(), tickers=("NQ",), count=4, seed=19)
    assert first == second
    assert len(first) == len({(item["ticker"], item["start"]) for item in first}) == 4
    assert all(item["ticker"] == "NQ" and type(item["start"]) is int for item in first)


def test_explicit_episode_list_takes_precedence_over_sampling():
    explicit = [{"ticker": "NQ", "start": 0}]
    assert sample_episode_specs(environment(), tickers=("NQ",), count=1, seed=19,
                                explicit=explicit) == explicit


def test_market_distillation_roles_are_carved_only_from_training_reserve():
    source = {"temporal": {"train_start": "2021-01-01", "train_end": "2025-01-01",
                           "validation_start": "2025-01-01", "validation_end": "2026-01-01"}}
    ns = lambda value: int(np.datetime64(value, "ns").astype(np.int64))
    config = {"dataset_temporal": {"train_start": "2021-01-01", "train_end": "2024-01-01",
                                    "validation_start": "2024-01-01", "validation_end": "2025-01-01"}}
    bounded, role = collection_source_for_role(
        config, source, "valid", {"train": [ns("2021-01-01"), ns("2025-01-01")]})
    assert role == [ns("2024-01-01"), ns("2025-01-01")]
    assert bounded["temporal"]["validation_start"] == "2024-01-01"
    assert source["temporal"]["validation_start"] == "2025-01-01"

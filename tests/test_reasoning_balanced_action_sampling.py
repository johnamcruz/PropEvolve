from collections import Counter

import numpy as np

from propevolve.reasoning_policy.supervised_trainer import balanced_action_order


def test_each_optimizer_window_balances_long_short_and_wait():
    rows = ([{"target_name": "ENTER_LONG_1"}] * 5
            + [{"target_name": "ENTER_SHORT_1"}] * 11
            + [{"target_name": "WAIT"}] * 7)
    order = balanced_action_order(rows, count=72, rng=np.random.default_rng(17))
    labels = [rows[index]["target_name"] for index in order]
    assert Counter(labels) == {"ENTER_LONG_1": 24, "ENTER_SHORT_1": 24, "WAIT": 24}
    for start in range(0, len(labels), 8):
        counts = Counter(labels[start:start + 8])
        assert max(counts.values()) - min(counts.values()) <= 1


def test_each_repeated_epoch_ends_on_a_complete_three_action_window():
    rows = ([{"target_name": "ENTER_LONG_1"}] * 19
            + [{"target_name": "ENTER_SHORT_1"}] * 24
            + [{"target_name": "WAIT"}] * 21)
    first = balanced_action_order(rows, count=len(rows), rng=np.random.default_rng(17))
    second = balanced_action_order(rows, count=len(rows), rng=np.random.default_rng(18))
    labels = [rows[index]["target_name"] for index in np.concatenate([first, second])]
    assert len(first) == len(second) == 63
    for start in range(0, len(labels), 3):
        assert Counter(labels[start:start + 3]) == {
            "ENTER_LONG_1": 1, "ENTER_SHORT_1": 1, "WAIT": 1,
        }


def test_balanced_sampler_rejects_missing_action_class():
    rows = [{"target_name": "WAIT"}, {"target_name": "ENTER_LONG_1"}]
    try:
        balanced_action_order(rows, count=8, rng=np.random.default_rng(1))
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("missing action class was accepted")


def test_indexed_sampler_balances_actions_inside_cache_local_ticker_blocks():
    rows = []
    for ticker in ("NQ", "ES"):
        for target in ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"):
            rows.extend({
                "target_name": target,
                "market_embedding_reference": {"ticker": ticker, "row": row,
                                                 "available_count": 20},
            } for row in range(4))

    order = balanced_action_order(rows, count=len(rows), rng=np.random.default_rng(23))
    ordered = [rows[index] for index in order]

    for start in range(0, len(ordered), 3):
        group = ordered[start:start + 3]
        assert {item["target_name"] for item in group} == {
            "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
        assert len({item["market_embedding_reference"]["ticker"] for item in group}) == 1
    switches = sum(
        ordered[index]["market_embedding_reference"]["ticker"]
        != ordered[index - 1]["market_embedding_reference"]["ticker"]
        for index in range(1, len(ordered))
    )
    assert switches == 1


def test_indexed_trade_mastery_repeats_sparse_hold_close_without_diluting_them():
    rows = []
    counts = {
        "WAIT": 4, "ENTER_LONG_1": 4, "ENTER_SHORT_1": 4,
        "HOLD": 1, "CLOSE": 2,
    }
    for target, count in counts.items():
        rows.extend({
            "target_name": target,
            "market_embedding_reference": {
                "ticker": "NQ", "row": row, "available_count": 20,
            },
        } for row in range(count))
    order = balanced_action_order(rows, count=len(rows), rng=np.random.default_rng(29))
    labels = [rows[index]["target_name"] for index in order]
    assert len(labels) == 15
    assert Counter(labels) == {
        "WAIT": 3, "ENTER_LONG_1": 3, "ENTER_SHORT_1": 3,
        "HOLD": 3, "CLOSE": 3,
    }
    for start in range(0, len(labels), 5):
        assert set(labels[start:start + 5]) == set(counts)

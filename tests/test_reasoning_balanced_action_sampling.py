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

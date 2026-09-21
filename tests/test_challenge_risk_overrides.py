"""Risk controls must reach the RL environment without rewriting the source contract.

source_environment.json is shared by many configs and source_identity is a hash of it,
so adding the daily loss limit and loss-streak cooldown there would invalidate the
lineage of every dataset already built against it -- including the trigger dataset this
run depends on. The controls therefore arrive as an explicit job-level override.

The override is deliberately narrow. It may only add the two risk controls. Letting it
reach profit_target, max_loss, episode_days or the reward terms would let a run quietly
redefine the challenge it claims to be passing, which is precisely the comparison the
whole exercise rests on.
"""
from __future__ import annotations

import pytest

from propevolve.reasoning_policy.job import apply_challenge_risk_overrides

BASE = {
    "profit_target": 6000, "max_loss": 3000, "episode_days": 30, "bars_per_day": 480,
    "max_position_size": 1, "minimum_mll_headroom": 500, "trailing_mll_lock": True,
    "terminal_pass_reward": 250, "terminal_blow_reward": -1500,
    "terminal_timeout_reward": -2, "terminal_pass_speed_reward_per_day": 20,
    "reward_scale": 1000, "per_trade_risk_dollars": 300,
}


def test_no_override_leaves_the_contract_untouched():
    out = apply_challenge_risk_overrides(BASE, None)
    assert out == BASE
    assert out is not BASE, "must not hand back the caller's dict to mutate"


def test_the_two_risk_controls_are_applied():
    out = apply_challenge_risk_overrides(BASE, {
        "daily_loss_limit_dollars": 750.0,
        "loss_streak_cooldown_trades": 2,
        "loss_streak_cooldown_bars": 65})
    assert out["daily_loss_limit_dollars"] == 750.0
    assert out["loss_streak_cooldown_trades"] == 2
    assert out["loss_streak_cooldown_bars"] == 65


def test_the_rest_of_the_contract_is_preserved():
    out = apply_challenge_risk_overrides(BASE, {"daily_loss_limit_dollars": 750.0})
    for key, value in BASE.items():
        assert out[key] == value


@pytest.mark.parametrize("field", [
    "profit_target", "max_loss", "episode_days", "terminal_blow_reward",
    "terminal_pass_reward", "per_trade_risk_dollars", "reward_scale",
])
def test_redefining_the_challenge_itself_is_refused(field):
    """A run must not be able to move the goalposts it is measured against."""
    with pytest.raises(ValueError, match="risk"):
        apply_challenge_risk_overrides(BASE, {field: 1})


def test_an_unknown_key_is_refused():
    with pytest.raises(ValueError, match="risk"):
        apply_challenge_risk_overrides(BASE, {"not_a_control": 1})


def test_a_non_mapping_override_is_refused():
    with pytest.raises(ValueError):
        apply_challenge_risk_overrides(BASE, [("daily_loss_limit_dollars", 750.0)])


def test_the_override_still_has_to_satisfy_the_spec():
    """Validation stays with ChallengeSpec; the override cannot smuggle a bad value."""
    from propevolve.environment import ChallengeSpec
    out = apply_challenge_risk_overrides(BASE, {
        "loss_streak_cooldown_trades": 2, "loss_streak_cooldown_bars": 0})
    with pytest.raises(ValueError):
        ChallengeSpec(**out)

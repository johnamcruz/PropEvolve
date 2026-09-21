"""One ordered evidence contract for the compact RL policy and the reasoning model.

A challenge episode is 30 days x 480 bars, and the reasoning model answers in roughly
0.3s a decision, so it cannot afford the thousands of episodes needed to SEARCH for a
profitable trend-following policy. A small policy over these fields can: it runs in
seconds an episode. So RL searches here, with every teacher visible, to discover which
combination of expansion, trend, regime, volume and order flow actually rides a trend
without blowing the account -- and the reasoning model is then taught the result.

The student runs TEACHER-FREE, which is what `require_challenge_mastery_context` and the
`require_teacher_free` acceptance criterion demand. That is workable because the teachers
were themselves trained from the same Chronos-2 embeddings the reasoning model reads, so
it is being asked to infer what a teacher-informed policy would do from the
representation those teachers came from -- not to guess something unknowable.
`TEACHER_PREFIXES` names that half explicitly so the split is never implied.

ORDER IS A CONTRACT. A trained policy's weights bind to these positions; reordering the
list would keep every run working while feeding each input to the wrong neuron.
"""
from __future__ import annotations

import numpy as np

# Teacher evidence. Visible to the searching RL policy, never to the shipped student.
TEACHER_PREFIXES: tuple[str, ...] = ("expansion", "trend", "regime", "volume")

_EXPANSION = (
    "expansion.long_attempt_probability",
    "expansion.long_clean_retained_given_attempt_probability",
    "expansion.short_attempt_probability",
    "expansion.short_clean_retained_given_attempt_probability",
)
_TREND = (
    "trend.long_conditional_quality",
    "trend.long_launch_probability",
    "trend.short_conditional_quality",
    "trend.short_launch_probability",
)
_REGIME = (
    "regime.chop_end_transition_probability",
    "regime.chop_no_trend_probability",
    "regime.expansion_trend_probability",
)
_VOLUME = (
    "volume.long_clean_given_participation_probability",
    "volume.long_participation_probability",
    "volume.short_clean_given_participation_probability",
    "volume.short_participation_probability",
)
# Expansion + order-flow setup: timing, side and whether the frozen rule fired.
_SETUP = (
    "setup.expansion_armed",
    "setup.expansion_score",
    "setup.flow_persistence",
    "setup.setup_available",
    "setup.setup_side",
    "setup.setup_trigger",
)
# What is being held right now — the state that made management learnable at 0.92.
_TRADE = (
    "trade.current_r",
    "trade.giveback_r",
    "trade.hold_bars",
    "trade.mae_r_so_far",
    "trade.mfe_r_so_far",
    "trade.open",
    "trade.position_side",
    "trade.risk_available",
)
# Risk budget. A static threshold on headroom alone already takes blow-ups from 46.2%
# to 1.1%, so this block is where the safety half of the policy has to come from.
_ACCOUNT = (
    "account.challenge_remaining",
    "account.current_r_norm",
    "account.drawdown_norm",
    "account.equity_pnl_norm",
    "account.giveback_r_norm",
    "account.hold_fraction",
    "account.mll_headroom_norm",
    "account.peak_equity_pnl_norm",
    "account.peak_favorable_r_norm",
    "account.point_value_norm",
    "account.position_side",
    "account.position_size_norm",
    "account.protected_r_norm",
    "account.ratchet_active",
    "account.realized_pnl_norm",
    "account.round_trip_fee_norm",
    "account.session_remaining",
    "account.unrealized_pnl_norm",
)
_CHALLENGE = (
    "challenge.equity_pnl_dollars",
    "challenge.headroom_dollars",
    "challenge.max_loss_dollars",
    "challenge.mll_floor_dollars",
    "challenge.profit_target_dollars",
    "challenge.realized_pnl_dollars",
    "challenge.target_remaining_dollars",
)

EVIDENCE_FIELDS: tuple[str, ...] = (
    _EXPANSION + _TREND + _REGIME + _VOLUME + _SETUP + _TRADE + _ACCOUNT + _CHALLENGE
)

TEACHER_FIELDS: tuple[str, ...] = tuple(
    f for f in EVIDENCE_FIELDS if f.split(".")[0] in TEACHER_PREFIXES)
STUDENT_FIELDS: tuple[str, ...] = tuple(
    f for f in EVIDENCE_FIELDS if f.split(".")[0] not in TEACHER_PREFIXES)


def evidence_vector(fields: dict, names: tuple[str, ...] = EVIDENCE_FIELDS) -> np.ndarray:
    """Pack named causal evidence into the fixed-order vector a policy was trained on.

    Missing values are refused rather than zero-filled: a fabricated zero reads as a real
    probability and would degrade the policy silently, which is the failure mode
    `specialist_account_fields` already refuses upstream for the same reason.
    """
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError("evidence is missing: " + ", ".join(missing[:8]))
    values = np.asarray([fields[name] for name in names], dtype=np.float32)
    if not np.isfinite(values).all():
        bad = [name for name, value in zip(names, values) if not np.isfinite(value)]
        raise ValueError("evidence must be finite: " + ", ".join(bad[:8]))
    return values

from __future__ import annotations

import numpy as np
import pytest

from propevolve.decision import (
    Action,
    ActionMasker,
    PositionSide,
)
from propevolve.observation import (
    AccountState,
    ObservationAssembler,
    TradeManagementObservationSpec,
)


def test_observation_is_frozen_embedding_plus_normalized_account_state() -> None:
    assembler = ObservationAssembler(embedding_dim=4, max_loss=3_000, profit_target=6_000)
    account = AccountState(
        realized_pnl=1_500,
        equity_pnl=1_200,
        peak_equity_pnl=1_800,
        position_side=PositionSide.LONG,
        position_size=2,
        max_position_size=4,
        unrealized_pnl=-300,
        session_remaining=0.25,
        challenge_remaining=0.50,
        point_value=20.0,
        round_trip_fee=3.78,
        mll_headroom=4_200,
    )

    value = assembler.assemble(np.array([1, 2, 3, 4], np.float32), account)

    np.testing.assert_array_equal(value[:4], np.array([1, 2, 3, 4], np.float32))
    np.testing.assert_allclose(
        value[4:],
        np.array([
            0.25, 0.20, 0.30, 1.40, 0.20, 1.0, 0.5, -0.1, 0.25, 0.50,
            20.0 / 3000.0, 3.78 / 3000.0,
        ], np.float32),
    )


def test_observation_rejects_nonfinite_embeddings() -> None:
    assembler = ObservationAssembler(
        embedding_dim=2, max_loss=3_000, profit_target=6_000
    )
    account = AccountState()
    with pytest.raises(ValueError, match="finite"):
        assembler.assemble(np.array([1.0, np.nan], np.float32), account)


def test_observation_exposes_normalized_causal_trade_management_state() -> None:
    assembler = ObservationAssembler(
        embedding_dim=2,
        max_loss=3_000,
        profit_target=6_000,
        trade_management=TradeManagementObservationSpec.entry_risk_v1(
            r_scale=10.0,
            hold_horizon_bars=120,
        ),
    )
    account = AccountState(
        current_r=1.5,
        peak_favorable_r=3.0,
        giveback_r=1.5,
        hold_bars=60,
        ratchet_active=True,
        protected_r=2.25,
    )

    value = assembler.assemble(np.asarray([1.0, 2.0], np.float32), account)

    assert assembler.output_dim == 20
    np.testing.assert_allclose(
        value[-6:],
        np.asarray([0.15, 0.30, 0.15, 0.50, 1.0, 0.225], np.float32),
    )


def test_risk_mask_exposes_only_state_valid_actions() -> None:
    masker = ActionMasker(
        max_position_size=1, max_loss=3_000, minimum_mll_headroom=250
    )

    flat = AccountState(equity_pnl=0, position_side=PositionSide.FLAT)
    assert masker.valid_actions(flat) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    long = AccountState(
        equity_pnl=-2_900,
        position_side=PositionSide.LONG,
        position_size=2,
        max_position_size=3,
    )
    assert masker.valid_actions(long) == (Action.HOLD, Action.CLOSE)


def test_configured_guard_allows_one_more_recovery_attempt_without_using_last_100(
) -> None:
    masker = ActionMasker(
        max_position_size=1,
        max_loss=3_000,
        minimum_mll_headroom=200,
    )
    recoverable = AccountState(
        equity_pnl=-2_600,
        mll_headroom=400,
        position_side=PositionSide.FLAT,
    )
    last_buffer = AccountState(
        equity_pnl=-2_900,
        mll_headroom=100,
        position_side=PositionSide.FLAT,
    )

    assert masker.valid_actions(recoverable) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert masker.valid_actions(last_buffer) == (Action.WAIT,)


def test_recovery_mode_keeps_wait_long_and_short_available_until_terminal() -> None:
    masker = ActionMasker(
        max_position_size=1,
        max_loss=3_000,
        minimum_mll_headroom=500,
    )
    near_mll = AccountState(
        equity_pnl=-2_700,
        mll_headroom=300,
        position_side=PositionSide.FLAT,
    )

    assert masker.valid_actions(near_mll) == (Action.WAIT,)
    assert masker.valid_actions(
        near_mll,
        recovery_active=True,
    ) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert masker.valid_actions(
        AccountState(
            equity_pnl=-2_600,
            mll_headroom=400,
            position_side=PositionSide.FLAT,
        ),
        recovery_active=True,
    ) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

import pytest

from propevolve.reasoning_policy.label_reference import barrier_result


def test_label_audit_gate_requires_examined_rows_and_no_discrepancies():
    from propevolve.reasoning_policy.label_reference import label_audit_passed
    assert label_audit_passed({'train/NQ': {'counts': {'WAIT': 1}, 'issues': {}}})
    assert not label_audit_passed({'train/NQ': {
        'counts': {'WAIT': 1}, 'issues': {'teacher_alignment_volume': 1}}})
    assert not label_audit_passed({})
    assert not label_audit_passed({'train/NQ': {'counts': {}, 'issues': {}}})


def test_management_execution_audit_replays_actual_fills_and_rejects_tampered_return():
    import copy
    import numpy as np
    from propevolve.decision import Action
    from propevolve.reasoning_policy.labels import label_position_continuation
    from propevolve.reasoning_policy.label_reference import verify_management_execution
    from test_reasoning_challenger_e2e import environment, passive_factory
    env = environment()
    from dataclasses import replace
    from propevolve.environment import HistoricalChallengeEnv
    env = HistoricalChallengeEnv(env.markets, tick_values=env.tick_values,
        round_trip_fees=env.round_trip_fees, spec=replace(env.spec, per_trade_risk_dollars=300.,
            ratchet_activation_r=2., ratchet_giveback_r=.5), seed=0)
    market = env.markets['NQ']
    labels = label_position_continuation(env, reset_options={'ticker': 'NQ', 'start': 0},
        prefix=(Action.ENTER_LONG_1,), continuation_factory=passive_factory,
        max_steps=3, minimum_improvement_r=.1)
    from dataclasses import asdict
    row = {'ticker': 'NQ', 'completed_at_ns': int(market.timestamps[1].astype('datetime64[ns]').astype(np.int64)),
        'targets': {'outcomes': {a.name: asdict(v) for a,v in labels.outcomes.items()},
                    'management_evidence': labels.management_evidence,
                    'position_entry': {'action': 'ENTER_LONG_1', 'execution': 'bar_open',
                        'completed_at_ns': int(market.timestamps[1].astype('datetime64[ns]').astype(np.int64))}}}
    options = dict(reset_options={'ticker': 'NQ', 'start': 0}, horizon=3,
                   minimum_improvement_r=.1, tolerance=1e-5)
    assert verify_management_execution(env, row, **options) == []
    wrong = copy.deepcopy(row)
    wrong['targets']['outcomes']['HOLD']['terminal_pnl'] += 100.
    assert 'HOLD.terminal_pnl' in verify_management_execution(env, wrong, **options)


@pytest.mark.parametrize("side,high,low,expected", [
    (1, [102.125], [99.5], True),
    (-1, [100.5], [97.875], True),
    (1, [103.], [98.], False),
    (1, [101.], [99.5], False),
])
def test_independent_barrier_reference(side, high, low, expected):
    assert barrier_result(100., high, low, side=side, risk=1., point_value=1.,
                          fee=.125, target=2., stop=1.) is expected


@pytest.mark.parametrize('action,sign', [('ENTER_LONG_1', 1), ('ENTER_SHORT_1', -1)])
def test_management_entry_reference_without_flat_label(action, sign):
    from propevolve.reasoning_policy.label_reference import management_entry
    row = {'completed_at_ns': 30, 'targets': {'position_entry': {
        'action': action, 'completed_at_ns': 20, 'execution': 'bar_open'}}}
    assert management_entry(row, None, {10: 0, 20: 1, 30: 2}) == (1, sign)
    row['targets']['position_entry']['completed_at_ns'] = 40
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, None, {10: 0, 20: 1, 30: 2, 40: 3})


def test_management_entry_reference_rejects_wait_parent_and_disagreement():
    from propevolve.reasoning_policy.label_reference import management_entry
    parent = {'completed_at_ns': 10, 'messages': [{'content': 'WAIT'}]}
    row = {'completed_at_ns': 30, 'targets': {}}
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, parent, {10: 0, 20: 1, 30: 2})
    parent['messages'][-1]['content'] = 'ENTER_LONG_1'
    assert management_entry(row, parent, {10: 0, 20: 1, 30: 2}) == (1, 1)
    row['targets']['position_entry'] = {'action': 'ENTER_SHORT_1',
                                      'completed_at_ns': 20, 'execution': 'bar_open'}
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, parent, {10: 0, 20: 1, 30: 2})

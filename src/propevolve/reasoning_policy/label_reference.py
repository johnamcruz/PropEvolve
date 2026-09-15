"""Independent scalar economic reference; never used to generate training labels."""


def management_entry(row, parent, lookup):
    """Resolve actual entry execution, never infer direction from a WAIT label."""
    sides = {'ENTER_LONG_1': 1, 'ENTER_SHORT_1': -1}
    explicit = row['targets'].get('position_entry')
    try:
        inherited = None
        if parent is not None and parent['messages'][-1]['content'] in sides:
            inherited = (lookup[parent['completed_at_ns']] + 1,
                         sides[parent['messages'][-1]['content']])
        if explicit is not None:
            if explicit['execution'] != 'bar_open':
                raise ValueError('invalid entry lineage execution')
            resolved = (lookup[explicit['completed_at_ns']], sides[explicit['action']])
            if inherited is not None and resolved != inherited:
                raise ValueError('conflicting entry lineage')
        elif inherited is not None:
            resolved = inherited
        else:
            raise ValueError('missing entry lineage')
        if not 0 <= resolved[0] <= lookup[row['completed_at_ns']]:
            raise ValueError('future entry lineage')
        return resolved
    except (KeyError, TypeError) as error:
        raise ValueError('invalid entry lineage') from error


def barrier_result(entry, high, low, *, side, risk, point_value, fee, target, stop):
    """Conservative OHLC target/stop ordering, expressed directly in net dollars."""
    for hi, lo in zip(high, low):
        worst_price = lo if side == 1 else hi
        best_price = hi if side == 1 else lo
        worst_net = side * (float(worst_price) - entry) * point_value - fee
        best_net = side * (float(best_price) - entry) * point_value - fee
        if worst_net <= -stop * risk:
            return False
        if best_net >= target * risk:
            return True
    return False


def verify_management_execution(environment, row, *, reset_options, horizon,
                                minimum_improvement_r, tolerance):
    """Audit passive-continuation labels by replaying public simulator steps.

    Does not call the label generator. The declared reference is WAIT before
    entry, HOLD after entry, then bounded continuation versus immediate CLOSE.
    This audits simulator parity, not whether passive continuation is optimal.
    """
    import numpy as np
    from ..decision import Action
    from ..environment import HistoricalChallengeEnv
    if horizon < 2 or tolerance <= 0:
        raise ValueError('invalid management replay audit contract')
    times = environment.markets[row['ticker']].timestamps.astype('datetime64[ns]').astype(np.int64)
    lookup = {int(value): index for index, value in enumerate(times)}
    entry, sign = management_entry(row, None, lookup)
    decision = lookup[row['completed_at_ns']]
    if not reset_options['start'] < entry <= decision:
        raise ValueError('management replay starts after entry')
    issues = []
    risk = environment.spec.per_trade_risk_dollars
    for first in (Action.HOLD, Action.CLOSE):
        env = HistoricalChallengeEnv(environment.markets, tick_values=environment.tick_values,
            round_trip_fees=environment.round_trip_fees, spec=environment.spec,
            observation_spec=environment._assembler.trade_management, seed=0)
        env.reset(options=reset_options)
        for index in range(reset_options['start'], decision):
            action = (Action.WAIT if index < entry-1 else
                      (Action.ENTER_LONG_1 if sign > 0 else Action.ENTER_SHORT_1)
                      if index == entry-1 else Action.HOLD)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or (index >= entry-1 and env.closed_trade_receipts()):
                raise ValueError('management replay does not reach a live anchor')
        receipt = None
        for step in range(horizon):
            action = first if step == 0 else Action.CLOSE if step == horizon-1 else Action.HOLD
            _, _, terminated, truncated, info = env.step(action)
            if env.closed_trade_receipts():
                receipt = env.closed_trade_receipts()[-1]
                break
            if terminated or truncated:
                raise ValueError('management audit censored by challenge')
        if receipt is None or receipt['exit_reason'] not in {'initial_stop', 'ratchet_stop', 'voluntary_close'}:
            raise ValueError('management audit lacks an executable trade exit')
        expected = row['targets']['outcomes'][first.name]
        economic = {'terminal_pnl': receipt['pnl'],
                    'reward_to_go': receipt['pnl']/risk - (minimum_improvement_r if first == Action.HOLD else 0.)}
        for key, value in economic.items():
            scale = risk if key == 'terminal_pnl' else 1.
            if not np.isfinite(expected[key]) or abs(expected[key]-value) > tolerance*scale:
                issues.append(f'{first.name}.{key}')
        if expected['outcome'] != receipt['exit_reason']:
            issues.append(f'{first.name}.outcome')
        exit_ns = int(np.datetime64(receipt['exit_timestamp'], 'ns').astype(np.int64))
        if expected['outcome_end_ns'] != exit_ns or expected['steps'] != step+1:
            issues.append(f'{first.name}.exit_timing')
        evidence = row['targets']['management_evidence'][first.name]
        for key in ('side', 'entry_timestamp', 'exit_timestamp', 'hold_bars', 'exit_reason', 'ratchet_activated'):
            if evidence[key] != receipt[key]:
                issues.append(f'{first.name}.{key}')
        for key in ('mfe_r', 'mae_r'):
            if not np.isfinite(evidence[key]) or abs(evidence[key]-receipt[key]) > tolerance:
                issues.append(f'{first.name}.{key}')
    return issues

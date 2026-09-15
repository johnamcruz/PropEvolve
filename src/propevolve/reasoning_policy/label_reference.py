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

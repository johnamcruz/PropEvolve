"""Independent scalar economic reference; never used to generate training labels."""


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

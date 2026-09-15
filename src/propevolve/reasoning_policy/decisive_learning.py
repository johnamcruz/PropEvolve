"""Explicit per-example acquisition/retention evidence, never a promotion score."""
import math


def require_initial_score_parity(reference, initial, indices, *, tolerance):
    """Fail before updates if an input ablation changes the frozen control."""
    if reference['indices'] != indices:
        raise ValueError('initial comparison requires identical rows')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid initial score parity tolerance')
    maximum = 0.0
    for role in ('train', 'valid'):
        old, new = reference['before'][role]['scores'], initial[role]['scores']
        if len(old) != len(new):
            raise ValueError('initial score parity shape mismatch')
        for left, right in zip(old, new):
            if len(left) != len(right) or not all(math.isfinite(x) for x in (*left, *right)):
                raise ValueError('invalid initial score parity evidence')
            maximum = max(maximum, max((abs(a-b) for a, b in zip(left, right)), default=0.0))
        if 'boundaries' in initial[role] and 'boundaries' in reference['before'][role]:
            changes = compare_learning(reference['before'][role]['boundaries'], initial[role]['boundaries'])
            if changes['acquired'] or changes['forgotten']:
                raise ValueError('initial score parity changed a decision boundary')
    if maximum > tolerance:
        raise ValueError(f'initial score parity exceeded tolerance: {maximum}')
    return maximum


def evaluation_recipe(config, *, adapter_path):
    """Export an evaluable policy, not the diagnostic's in-memory sampler state."""
    return {**config, 'adapter_path': adapter_path, 'targeted_sampling': None,
            'mastered_anchor_retention': None, 'initial_validation_receipt': None,
            'checkpoint_acceptance': None}


def decision_evidence(row, scores, *, ambiguity_r):
    targets = row["action_targets"]
    names, values = targets["names"], targets["values"]
    if (len(names) != len(scores) or len(names) != len(values)
            or len(set(names)) != len(names) or ambiguity_r < 0
            or not all(math.isfinite(float(v)) for v in (*values, *scores, ambiguity_r))):
        raise ValueError("invalid decisive boundary evidence")
    economics, predicted = dict(zip(names, values)), dict(zip(names, scores))
    target = row["target_name"]
    if target not in economics or economics[target] != max(values):
        raise ValueError("target is not an economic winner")
    result = {}

    def add(name, margin, gap):
        result[name] = {"margin": float(margin), "correct": bool(margin > 0),
                        "ambiguous": bool(gap <= ambiguity_r)}

    if set(names) == {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}:
        sides = ("ENTER_LONG_1", "ENTER_SHORT_1")
        enter_value = max(economics[n] for n in sides)
        enter_score = max(predicted[n] for n in sides)
        if target == "WAIT":
            add("WAIT", predicted["WAIT"] - enter_score, economics["WAIT"] - enter_value)
        else:
            add("ENTER", enter_score - predicted["WAIT"], enter_value - economics["WAIT"])
            other = sides[1] if target == sides[0] else sides[0]
            add("LONG" if target == sides[0] else "SHORT",
                predicted[target] - predicted[other], economics[target] - economics[other])
    elif set(names) == {"HOLD", "CLOSE"}:
        other = "CLOSE" if target == "HOLD" else "HOLD"
        add(target, predicted[target] - predicted[other], economics[target] - economics[other])
    else:
        raise ValueError("unsupported decisive legal actions")
    return result


def compare_learning(before, after):
    if len(before) != len(after):
        raise ValueError("decisive comparisons require identical rows")
    acquired = forgotten = clear = correct = 0
    for old, new in zip(before, after):
        if set(old) != set(new):
            raise ValueError("decisive comparisons require identical boundaries")
        for name, initial in old.items():
            current = new[name]
            if initial["ambiguous"] != current["ambiguous"]:
                raise ValueError("economic labels changed during comparison")
            if current["ambiguous"]:
                continue
            clear += 1
            correct += current["correct"]
            acquired += not initial["correct"] and current["correct"]
            forgotten += initial["correct"] and not current["correct"]
    return {"applicable_clear_boundaries": clear, "correct": correct,
            "acquired": acquired, "forgotten": forgotten,
            "all_clear_correct": bool(clear and correct == clear)}


def generalization_indices(rows, *, tickers, actions, rows_per_group,
                           start_ns, end_ns, training_end_ns, seed):
    """Choose fixed chronological strata without looking at model predictions."""
    import numpy as np
    if not training_end_ns <= start_ns < end_ns:
        raise ValueError("generalization must use a chronological reserve")
    if (type(rows_per_group) is not int or rows_per_group < 1
            or not tickers or not actions
            or len(set(tickers)) != len(tickers) or len(set(actions)) != len(actions)):
        raise ValueError("invalid generalization strata")
    groups = {(ticker, action): [] for ticker in tickers for action in actions}
    for index, row in enumerate(rows):
        key = row['ticker'], row['target']
        if key in groups and start_ns <= row['completed_at_ns'] <= row['label_end_ns'] < end_ns:
            groups[key].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for key, pool in groups.items():
        seen, chosen = set(), []
        for index in rng.permutation(pool):
            source_id = rows[int(index)]['source_id']
            if source_id not in seen:
                seen.add(source_id)
                chosen.append(int(index))
            if len(chosen) == rows_per_group:
                break
        if len(chosen) != rows_per_group:
            raise ValueError(f"insufficient independent generalization rows for {key}")
        selected.extend(chosen)
    return selected


def simulate_prefix(policy, environment, *, options, context_config, max_steps, on_decision):
    """Measure real teacher-free decisions without inventing a terminal outcome.

    This diagnostic is deliberately not the full challenge evaluator. A bounded
    prefix can expose poor entry/exit behavior, but cannot estimate pass rate.
    """
    from .context import RollingContext
    from .inputs import observe_context
    if context_config.input_mode != "embeddings" or type(max_steps) is not int or max_steps < 1:
        raise ValueError("prefix requires a teacher-free context and positive budget")
    context = RollingContext(context_config)
    observation, info = environment.reset(options=options)
    row, counts, complete = options['start'], {}, False
    for step in range(max_steps):
        observe_context(context, environment, observation,
                        ticker=options['ticker'], row=row, sources=())
        action, scores = policy.decide(context.snapshot(), info['valid_actions'])
        if action not in info['valid_actions']:
            raise ValueError("prefix policy produced illegal action")
        if not all(math.isfinite(float(v)) for v in scores.values()):
            raise ValueError("prefix policy produced nonfinite scores")
        counts[action.name] = counts.get(action.name, 0) + 1
        on_decision({'index': row, 'action': action.name, 'scores': scores,
                     'ticker': options['ticker'], 'trade_state': environment.causal_trade_context()})
        observation, _, terminated, truncated, info = environment.step(action)
        row = int(info['fill_index'])
        if terminated or truncated:
            complete = True
            break
    return {'complete': complete, 'outcome': info.get('outcome') if complete else None,
            'steps': step + 1, 'teacher_free': True, 'action_counts': counts,
            'closed_trades': int(info['trade_count']),
            'realized_pnl': float(info['realized_pnl']),
            'minimum_mll_headroom': float(info['minimum_mll_headroom']),
            'win_count': int(info.get('win_count', 0)),
            'average_win_r': float(info.get('avg_win_r', 0.)),
            'expectancy_r': float(info.get('expectancy_r', 0.)),
            'average_mfe_r': float(info.get('avg_mfe_r', 0.)),
            'average_mae_r': float(info.get('avg_mae_r', 0.))}
def fixed_diagnostic_indices(rows, indices, *, minimum_gap):
    """Honor an audited JSON cohort without resampling its chronological control."""
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int or i < 0 or i >= len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('invalid fixed diagnostic indices')
    for i in indices:
        row = rows[i]
        targets = row['action_targets']
        values = dict(zip(targets['names'], targets['values']))
        gap = values[row['target_name']] - max(
            value for name, value in values.items() if name != row['target_name'])
        if not gap >= minimum_gap:
            raise ValueError('fixed diagnostic row fails economic gap')
    return list(indices)

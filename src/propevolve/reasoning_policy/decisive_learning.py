"""Explicit per-example acquisition/retention evidence, never a promotion score."""
import math


def retained_margin_progress(before, after, *, forgotten, minimum_gain):
    """Diagnostic continuation only; never a model-promotion decision."""
    return (all(math.isfinite(x) for x in (before, after, minimum_gain))
            and minimum_gain > 0 and forgotten == 0 and after-before >= minimum_gain)


def precise_gram(rows, *, block_size):
    """Accumulate a small double-precision Gram without doubling the gradient bank."""
    import numpy as np
    rows = np.asarray(rows)
    if rows.ndim != 2 or block_size < 1 or not np.isfinite(rows).all():
        raise ValueError('invalid gradient bank')
    gram = np.zeros((len(rows), len(rows)), dtype=np.float64)
    for start in range(0, rows.shape[1], block_size):
        block = rows[:, start:start+block_size].astype(np.float64)
        gram += block @ block.T
    return gram


def differentiable_boundary_margin(scores, names, boundary, *, xp):
    """Same hierarchical decision as assessment, retaining array autodiff."""
    values = dict(zip(names, scores))
    if boundary in {'ENTER', 'WAIT'}:
        gap = xp.maximum(values['ENTER_LONG_1'], values['ENTER_SHORT_1']) - values['WAIT']
        return gap if boundary == 'ENTER' else -gap
    if boundary in {'LONG', 'SHORT'}:
        gap = values['ENTER_LONG_1'] - values['ENTER_SHORT_1']
        return gap if boundary == 'LONG' else -gap
    if boundary in {'HOLD', 'CLOSE'}:
        gap = values['HOLD'] - values['CLOSE']
        return gap if boundary == 'HOLD' else -gap
    raise ValueError('unsupported decision boundary')


def constraint_multipliers(gram, residual, *, tolerance, maximum_cycles):
    """Dual coordinate projection for G d >= target; diagnostic, not a learner."""
    import numpy as np
    gram, residual = np.asarray(gram, dtype=float), np.asarray(residual, dtype=float)
    if (gram.shape != (len(residual), len(residual)) or not len(residual)
            or not np.isfinite(gram).all() or not np.isfinite(residual).all()
            or not np.all(np.diag(gram) > 0) or tolerance <= 0 or maximum_cycles < 1):
        raise ValueError('invalid margin constraints')
    multipliers = np.zeros(len(residual))
    for _ in range(maximum_cycles):
        previous = multipliers.copy()
        for i in range(len(residual)):
            multipliers[i] = max(0., multipliers[i] +
                (residual[i] - gram[i] @ multipliers) / gram[i, i])
        if (np.max(residual - gram @ multipliers) <= tolerance
                and np.max(np.abs(multipliers - previous)) <= tolerance):
            return multipliers
    raise ValueError('margin constraints did not converge')


def fractional_snapshot(before, after, *, fraction):
    """Interpolate one saved displacement; never advance or alter optimizer state."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('diagnostic fraction must be finite and within [0, 1]')
    endpoint = component_snapshot(before, after, component='both')
    if fraction == 0:
        return dict(before)
    if fraction == 1:
        return endpoint
    return {name: value + fraction * (endpoint[name] - value)
            for name, value in before.items()}


def component_snapshot(before, after, *, component):
    """Isolate a saved actual update, without modifying either source snapshot."""
    if component not in {'lora', 'projector', 'both'} or before.keys() != after.keys():
        raise ValueError('component comparison requires identical parameter identities')
    result = {}
    for name, value in before.items():
        group = ('projector' if name.startswith('market_projector.') else
                 'lora' if name.endswith(('.lora_a', '.lora_b')) else None)
        if group is None or value.shape != after[name].shape:
            raise ValueError('unsupported or mismatched diagnostic parameter')
        result[name] = after[name] if component in {group, 'both'} else value
    return result


def project_retention_displacement(before, proposed, gradient, *, component='lora'):
    """Diagnostic one-sided component step projection, not a new optimizer.

    Retention is minimized, so a positive gradient dot displacement is harmful
    locally. Preserve other components and native optimizer state; finite action ranks
    must still be checked after this first-order correction.
    """
    import mlx.core as mx
    result = component_snapshot(before, proposed, component='both')
    if before.keys() != gradient.keys():
        raise ValueError('retention gradient identity differs')
    if component not in {'lora', 'projector'}:
        raise ValueError('retention projection requires a named component')
    names = [n for n in before if (n.startswith('market_projector.')) == (component == 'projector')]
    if not names:
        raise ValueError('retention projection requires component parameters')
    delta = {n: proposed[n].astype(mx.float32) - before[n].astype(mx.float32) for n in names}
    dot = float(sum(mx.sum(gradient[n].astype(mx.float32)*delta[n]) for n in names).item())
    norm2 = float(sum(mx.sum(gradient[n].astype(mx.float32)**2) for n in names).item())
    if not math.isfinite(dot) or not math.isfinite(norm2):
        raise ValueError('nonfinite retention displacement')
    coefficient = max(dot, 0.) / norm2 if norm2 > 0. else 0.
    if coefficient:
        for n in names:
            result[n] = (proposed[n].astype(mx.float32)
                         - coefficient * gradient[n].astype(mx.float32)).astype(proposed[n].dtype)
    after_dot = float(sum(mx.sum(gradient[n].astype(mx.float32)
        * (result[n].astype(mx.float32)-before[n].astype(mx.float32))) for n in names).item())
    return result, {'dot_before': dot, 'dot_after': after_dot, 'coefficient': coefficient,
                    'retention_gradient_norm': norm2**0.5}


def mean_batch_gradient(model, batches, *, loss, weight_by_rows):
    """Stream a diagnostic gradient without advancing the native optimizer.

    Row weighting matches a complete-panel mean, including a partial tail.
    Equal batch weighting matches MLX-LM's native accumulation convention.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    derivative = nn.value_and_grad(model, loss)
    total_value, total_gradient, mass = None, None, 0
    for batch in batches:
        value, gradient = derivative(model, *batch)
        weight = int(batch[0].shape[0]) if weight_by_rows else 1
        weighted = tree_map(lambda g: weight * g, gradient)
        total_gradient = (weighted if total_gradient is None else
                          tree_map(lambda a, b: a + b, total_gradient, weighted))
        total_value = weight * value if total_value is None else total_value + weight * value
        mass += weight
        mx.eval(total_value, total_gradient)
    if not mass:
        raise ValueError('gradient panel must not be empty')
    return total_value / mass, tree_map(lambda g: g / mass, total_gradient)


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
def update_progress(before, after):
    """Report plasticity on old mistakes separately from retained decisions."""
    result = compare_learning(before, after)
    changes = [new[name]['margin'] - value['margin']
               for old, new in zip(before, after) for name, value in old.items()
               if not value['ambiguous'] and not value['correct']]
    return {**result, 'mistake_count': len(changes),
            'mistake_margin_change': sum(changes) / len(changes) if changes else None}

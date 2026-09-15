"""Read-only gradient geometry; never changes gradient values or optimization."""
from itertools import combinations
import numpy as np


def gradient_geometry(gradients, update=None):
    if not gradients:
        raise ValueError('no objective gradients')
    keys = set(next(iter(gradients.values())))
    if any(set(g) != keys for g in gradients.values()) or (update is not None and set(update) != keys):
        raise ValueError('gradient trees differ')
    groups = {'lora': [], 'projector': []}
    for name in sorted(keys):
        if name.startswith('market_projector.'):
            groups['projector'].append(name)
        elif name.endswith(('.lora_a', '.lora_b')):
            groups['lora'].append(name)
        else:
            raise ValueError(f'unexpected trainable parameter: {name}')
    result = {}
    for group, names in groups.items():
        def dot(a, b):
            total = 0.
            for n in names:
                left, right = np.asarray(a[n], dtype=np.float64), np.asarray(b[n], dtype=np.float64)
                if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise ValueError('invalid gradient tensor')
                total += float(np.sum(left * right))
            return total
        norms = {n: dot(g, g) ** 0.5 for n, g in gradients.items()}
        cosines = {}
        for a, b in combinations(gradients, 2):
            denominator = norms[a] * norms[b]
            cosines[f'{a}|{b}'] = None if denominator == 0 else dot(gradients[a], gradients[b]) / denominator
        result[group] = {'norms': norms, 'cosines': cosines}
        if update is not None:
            result[group]['update_norm'] = dot(update, update) ** 0.5
            result[group]['gradient_dot_update'] = {n: dot(g, update) for n, g in gradients.items()}
    return result

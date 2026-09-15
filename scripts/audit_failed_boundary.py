"""Saved finite-step and single-row tangent diagnosis; no optimizer updates."""
import argparse
import copy
import json
from itertools import product
from functools import partial
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import (
    component_snapshot, decision_evidence, fractional_snapshot, mean_batch_gradient, update_progress,
)
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    _batch_outputs, batch_objective_loss, configure_trainable_components,
    evaluate_action_validation, tensor_batches,
)
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    path = Path(plan['source_report'])
    source = json.loads(path.read_text())
    measured = json.loads(Path(plan['measurement_report']).read_text())
    original_path = Path(measured['source_report'])
    original = json.loads(original_path.read_text())
    if (source['measurement_sha256'] != file_digest(plan['measurement_report'])
            or source['source_sha256'] != file_digest(original_path)):
        raise ValueError('source identity changed')
    prior = original['plan']
    config = read_sft_config(prior['learner_config'], root=Path.cwd())
    verify_mlx_view(prior['view_config'], prior['view'], root=Path.cwd())
    if file_digest(Path(config['data']) / 'manifest.json') != original['dataset_sha256']:
        raise ValueError('data identity changed')
    states = []
    expected = []
    for offset in (plan['before_offset'], plan['after_offset']):
        step = next(s for s in source['updates'] if s['offset'] == offset)
        state = path.parent / f'update-{offset+1:02d}' / 'state'
        receipt = json.loads((state / 'receipt.json').read_text())['receipt']
        if receipt['assessment'] != step['candidate'] or receipt['config_sha256'] != source['config_sha256']:
            raise ValueError('saved endpoint receipt differs')
        states.append(state / 'weights.safetensors')
        expected.append(step['candidate'])
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'config_sha256': file_digest(args.config),
        'source_sha256': file_digest(path), 'weight_hashes': {str(p): file_digest(p) for p in states},
        'fractions': [], 'gradients': {}, 'promotion': 'NOT_AUTHORIZED'}
    atomic_json(output / 'report.json', report)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_memory_limit(int(plan['memory_gb']*1024**3))
    mx.set_cache_limit(int(plan['cache_mb']*1024**2))
    policy = MLXActionPolicy.from_config(original_path.parent / 'policy.json')
    model = policy.model
    configure_trainable_components(model, config['trainable_components'])
    before, after = [mx.load(str(p)) for p in states]
    dataset = PreparedDataset(prior['view'], 'train')
    rows = [dataset[i] for i in original['indices']['train']]
    for component, fraction in product(plan.get('components', ['both']), plan['fractions']):
        endpoint = component_snapshot(before, after, component=component)
        model.load_weights(list(fractional_snapshot(before, endpoint, fraction=fraction).items()), strict=False)
        model.eval()
        scores = {}
        evaluate_action_validation(model, rows, config, on_scored=lambda i,s: scores.__setitem__(i,s))
        ordered = [scores[i] for i in range(len(rows))]
        evidence = [decision_evidence(r,s,ambiguity_r=prior['ambiguity_r']) for r,s in zip(rows,ordered)]
        delta = None
        if fraction == 0 or (fraction == 1 and component == 'both'):
            delta = max(abs(a-b) for x,y in zip(ordered,expected[int(fraction)]['scores']) for a,b in zip(x,y))
            if delta > prior['reload_tolerance']:
                raise ValueError(f'endpoint parity failed: {delta}')
        record = {'component': component, 'fraction': fraction, 'scores': ordered, 'boundaries': evidence,
                  'progress': update_progress(expected[0]['boundaries'], evidence), 'endpoint_delta': delta}
        report['fractions'].append(record)
        atomic_json(output / 'report.json', report)
        print('[boundary-audit]', component, fraction, json.dumps(record['progress']), 'target', evidence[plan['row']], flush=True)
    if not plan.get('measure_gradients', True):
        report['status'] = 'COMPLETE_DIAGNOSTIC'
        atomic_json(output / 'report.json', report)
        return
    model.load_weights(list(before.items()), strict=False)
    model.eval()
    row = copy.copy(rows[plan['row']])
    prefix = json.loads((original_path.parent / 'probe-prefix-state' / 'receipt.json').read_text())['receipt']
    frozen = dict(zip(prefix['selected_indices'], prefix['frozen_retention']))
    row['mastered_anchor_retention'] = frozen[original['indices']['train'][plan['row']]]
    config['mastered_anchor_retention'] = {**prior['retention'],
        'supervision_weight': prior['retention_switch']['supervision_weight']}
    batch = next(tensor_batches([row], 1, config['max_seq_length'], include_partial=True))
    if set(expected[0]['boundaries'][plan['row']]) != {'CLOSE'}:
        raise ValueError('this diagnostic requires a CLOSE boundary')
    evaluation_config = {**config, 'mastered_anchor_retention': None,
                         'error_selected_distillation': None, 'market_distillation': None}

    def margin_loss(m, *tensors):
        scores = _batch_outputs(m, *tensors[:10], config=evaluation_config)[2]
        return scores[0, 1] - scores[0, 0]  # CLOSE minus HOLD, maximize

    for name, loss in (
        ('retention_loss', partial(batch_objective_loss, config=config, objective='retention')),
        ('close_margin', margin_loss),
    ):
        value, gradient = mean_batch_gradient(model, [batch], loss=loss, weight_by_rows=True)
        flat = dict(tree_flatten(gradient))
        groups = {}
        for group in ('lora', 'projector'):
            names = [n for n in before if n.startswith('market_projector.') == (group == 'projector')]
            dot = sum(mx.sum(flat[n].astype(mx.float32)*(after[n].astype(mx.float32)-before[n].astype(mx.float32))) for n in names)
            norm = mx.sqrt(sum(mx.sum(flat[n].astype(mx.float32)**2) for n in names))
            groups[group] = {'dot_displacement': float(dot.item()), 'norm': float(norm.item())}
        report['gradients'][name] = {'value': float(value.item()), 'groups': groups}
        atomic_json(output / 'report.json', report)
        print('[boundary-audit]', name, json.dumps(report['gradients'][name]), flush=True)
    if plan.get('measure_complete_retention', False):
        anchors = []
        for index, retention in frozen.items():
            if any(retention['boundaries'].values()):
                anchor = copy.copy(dataset[index])
                anchor['mastered_anchor_retention'] = retention
                anchors.append(anchor)
        value, gradient = mean_batch_gradient(model,
            tensor_batches(anchors, prior['complete_anchor_probe']['anchor_batch_size'],
                           config['max_seq_length'], include_partial=True),
            loss=partial(batch_objective_loss, config=config, objective='retention'),
            weight_by_rows=True)
        flat = dict(tree_flatten(gradient))
        groups = {}
        for group in ('lora', 'projector'):
            names = [n for n in before if n.startswith('market_projector.') == (group == 'projector')]
            dot = sum(mx.sum(flat[n].astype(mx.float32)*(after[n].astype(mx.float32)-before[n].astype(mx.float32))) for n in names)
            groups[group] = {'dot_displacement': float(dot.item())}
        report['gradients']['complete_retention'] = {'value': float(value.item()),
            'anchor_rows': len(anchors), 'groups': groups}
        print('[boundary-audit] complete_retention', json.dumps(report['gradients']['complete_retention']), flush=True)
    report['status'] = 'COMPLETE_DIAGNOSTIC'
    atomic_json(output / 'report.json', report)


if __name__ == '__main__':
    main()

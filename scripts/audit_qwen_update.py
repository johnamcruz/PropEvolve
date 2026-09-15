"""Attribute a saved real optimizer displacement; never train or promote a policy."""
import argparse
import copy
import json
from functools import partial
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import (
    component_snapshot, decision_evidence, mean_batch_gradient, update_progress,
)
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    batch_objective_loss, configure_trainable_components, evaluate_action_validation,
    tensor_batches,
)
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    source_path = Path(plan['source_report'])
    source = json.loads(source_path.read_text())
    if source['status'] != 'COMPLETE_DIAGNOSTIC' or not source['reload_parity']:
        raise ValueError('source diagnostic must be complete with reload parity')
    old_plan = source['plan']
    config = read_sft_config(old_plan['learner_config'], root=Path.cwd())
    verify_mlx_view(old_plan['view_config'], old_plan['view'], root=Path.cwd())
    if (file_digest(Path(old_plan['view']) / 'view_manifest.json') != source['view_sha256']
            or file_digest(Path(config['data']) / 'manifest.json') != source['dataset_sha256']):
        raise ValueError('source data identity changed')
    prefix = source_path.parent / 'probe-prefix-state'
    receipt = json.loads((prefix / 'receipt.json').read_text())['receipt']
    for key in ('plan_sha256', 'view_sha256', 'dataset_sha256'):
        if receipt[key] != source[key]:
            raise ValueError('prefix checkpoint identity changed')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'source_sha256': file_digest(source_path),
              'config_sha256': file_digest(args.config), 'variants': {},
              'limitation': 'Saved update attribution only; no promotion or economic proof.'}
    atomic_json(output / 'report.json', result)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    policy = MLXActionPolicy.from_config(source_path.parent / 'policy.json')
    model = policy.model
    configure_trainable_components(model, config['trainable_components'])
    before = mx.load(str(prefix / 'weights.safetensors'))
    candidate = dict(tree_flatten(model.trainable_parameters()))
    control = mx.load(str(source_path.parent / 'control.safetensors'))
    component_snapshot(before, candidate, component='both')
    component_snapshot(before, control, component='both')
    result['weight_hashes'] = {str(p): file_digest(p) for p in (
        prefix / 'weights.safetensors', source_path.parent / 'control.safetensors',
        source_path.parent / 'adapter' / 'adapters.safetensors',
        source_path.parent / 'adapter' / 'projector.safetensors')}
    dataset = PreparedDataset(old_plan['view'], 'train')
    rows = [dataset[i] for i in source['indices']['train']]
    reference = source['complete_anchor_probe']

    def assess(weights):
        model.load_weights(list(weights.items()), strict=False)
        model.eval()
        scores = {}
        evaluate_action_validation(model, rows, config,
            on_scored=lambda i, s: scores.__setitem__(i, s))
        ordered = [scores[i] for i in range(len(rows))]
        return {'scores': ordered, 'boundaries': [decision_evidence(row, score,
            ambiguity_r=old_plan['ambiguity_r']) for row, score in zip(rows, ordered)]}

    def parity(actual, expected):
        if len(actual['scores']) != len(expected['scores']):
            raise ValueError('score row count changed')
        delta = max(abs(a-b) for x,y in zip(actual['scores'], expected['scores']) for a,b in zip(x,y))
        if delta > old_plan['reload_tolerance']:
            raise ValueError(f'saved update score parity failed: {delta}')
        return delta

    baseline = assess(before)
    result['prefix_delta'] = parity(baseline, reference['before'])
    result['before'] = baseline
    for name, snapshot in (('control', control), ('candidate', candidate)):
        for component in ('both', 'lora', 'projector'):
            evidence = assess(component_snapshot(before, snapshot, component=component))
            value = {'evidence': evidence,
                     'progress': update_progress(baseline['boundaries'], evidence['boundaries'])}
            if component == 'both':
                value['score_parity_delta'] = parity(evidence, reference[name])
            result['variants'][name + '_' + component] = value
            atomic_json(output / 'report.json', result)
            print('[update-audit]', name, component, json.dumps(value['progress']), flush=True)

    # Gradients of the actual per-task production objectives at the same prefix.
    # Dot product with the saved Adam displacement is a local approximation,
    # not proof about the finite update; component inference above is decisive.
    model.load_weights(list(before.items()), strict=False)
    model.eval()
    config['mastered_anchor_retention'] = {**old_plan['retention'],
        'supervision_weight': old_plan['retention_switch']['supervision_weight']}
    selected = {}
    for index, retention in zip(receipt['selected_indices'], receipt['frozen_retention']):
        row = copy.copy(dataset[index])
        row['mastered_anchor_retention'] = retention
        selected[index] = row
    batches = [next(tensor_batches([selected[i] for i in ids], len(ids),
        config['max_seq_length'], include_partial=True))
        for ids in reference['next_microbatch_indices']]
    result['objective_gradients'] = {}
    for objective in ('entry', 'direction', 'management', 'teacher', 'retention'):
        value, gradient = mean_batch_gradient(model, batches,
            loss=partial(batch_objective_loss, config=config, objective=objective), weight_by_rows=False)
        flat = dict(tree_flatten(gradient))
        record = {'loss': float(value.item()), 'groups': {}}
        for group in ('lora', 'projector'):
            names = [n for n in before if (n.startswith('market_projector.')) == (group == 'projector')]
            norm = mx.sqrt(sum(mx.sum(flat[n].astype(mx.float32)**2) for n in names))
            record['groups'][group] = {'gradient_norm': float(norm.item()), 'updates': {}}
            for name, snapshot in (('control', control), ('candidate', candidate)):
                delta = {n: snapshot[n].astype(mx.float32)-before[n].astype(mx.float32) for n in names}
                dot = sum(mx.sum(flat[n].astype(mx.float32)*delta[n]) for n in names)
                size = mx.sqrt(sum(mx.sum(delta[n]**2) for n in names))
                record['groups'][group]['updates'][name] = {
                    'gradient_dot_actual_displacement': float(dot.item()),
                    'displacement_norm': float(size.item())}
        result['objective_gradients'][objective] = record
        atomic_json(output / 'report.json', result)
        print('[update-audit] gradient', objective, json.dumps(record), flush=True)
    result['status'] = 'COMPLETE_DIAGNOSTIC'
    atomic_json(output / 'report.json', result)


if __name__ == '__main__':
    main()

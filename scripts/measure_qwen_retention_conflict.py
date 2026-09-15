"""Measure and cache actual Qwen acquisition/retention gradients at a saved prefix."""
import argparse
import copy
import json
from functools import partial
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import mean_batch_gradient
from propevolve.reasoning_policy.gradient_audit import gradient_geometry
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    batch_loss, batch_objective_loss, build_optimizer, configure_trainable_components, tensor_batches,
)
from propevolve.reasoning_policy.training_checkpoint import load_training_state
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    source_path = Path(plan['source_report'])
    source = json.loads(source_path.read_text())
    if source['status'] != 'COMPLETE_DIAGNOSTIC' or not source['reload_parity']:
        raise ValueError('require completed authenticated source')
    prior = source['plan']
    config = read_sft_config(prior['learner_config'], root=Path.cwd())
    verify_mlx_view(prior['view_config'], prior['view'], root=Path.cwd())
    if file_digest(Path(config['data']) / 'manifest.json') != source['dataset_sha256']:
        raise ValueError('dataset identity changed')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'source_report': str(source_path),
        'source_sha256': file_digest(source_path), 'config_sha256': file_digest(args.config),
        'gradient_artifacts': {}}
    atomic_json(output / 'report.json', report)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    policy = MLXActionPolicy.from_config(source_path.parent / 'policy.json')
    model = policy.model
    configure_trainable_components(model, config['trainable_components'])
    optimizer = build_optimizer(config)
    prefix = source_path.parent / 'probe-prefix-state'
    receipt = load_training_state(prefix, model, optimizer)
    for key in ('plan_sha256', 'view_sha256', 'dataset_sha256'):
        if receipt[key] != source[key]:
            raise ValueError('prefix identity changed')
    report['prefix_hashes'] = {name: file_digest(prefix / name) for name in
        ('receipt.json', 'weights.safetensors', 'state.safetensors')}
    config['mastered_anchor_retention'] = {**prior['retention'],
        'supervision_weight': prior['retention_switch']['supervision_weight']}
    dataset = PreparedDataset(prior['view'], 'train')
    selected = {}
    for index, retention in zip(receipt['selected_indices'], receipt['frozen_retention']):
        row = copy.copy(dataset[index])
        row['mastered_anchor_retention'] = retention
        selected[index] = row
    ids = source['complete_anchor_probe']['next_microbatch_indices']
    batches = [next(tensor_batches([selected[i] for i in group], len(group),
        config['max_seq_length'], include_partial=True)) for group in ids]
    anchors = [r for r in selected.values() if any(r['mastered_anchor_retention']['boundaries'].values())]
    anchor_batches = list(tensor_batches(anchors, prior['complete_anchor_probe']['anchor_batch_size'],
        config['max_seq_length'], include_partial=True))
    retention = partial(batch_objective_loss, config=config, objective='retention')
    acquisition = lambda m, *batch: batch_loss(m, *batch, config=config)[0] - retention(m, *batch)
    gradients = {}
    model.train()
    for name, loss, panel, row_mean in (
        ('acquisition', acquisition, batches, False),
        ('sampled_retention', retention, batches, False),
        ('complete_retention', retention, anchor_batches, True),
    ):
        value, gradient = mean_batch_gradient(model, panel, loss=loss, weight_by_rows=row_mean)
        flat = dict(tree_flatten(gradient))
        path = output / (name + '.safetensors')
        mx.save_safetensors(str(path), flat)
        gradients[name] = flat
        report['gradient_artifacts'][name] = {'path': str(path), 'sha256': file_digest(path),
            'loss': float(value.item()), 'microbatches': len(panel), 'row_mean': row_mean}
        atomic_json(output / 'report.json', report)
        print('[conflict] gradient=' + name + ' loss=' + str(float(value.item())), flush=True)
    report['geometry'] = gradient_geometry(gradients)
    report['status'] = 'COMPLETE_DIAGNOSTIC'
    atomic_json(output / 'report.json', report)
    print('[conflict] ' + json.dumps(report['geometry']), flush=True)


if __name__ == '__main__':
    main()

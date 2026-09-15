"""Measure production-Qwen task interference and actual update damage; no promotion."""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import decision_evidence, compare_learning
from propevolve.reasoning_policy.gradient_audit import gradient_geometry
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    batch_loss, batch_objective_loss, configure_trainable_components, build_optimizer,
    tensor_batches, evaluate_action_validation,
)
from propevolve.reasoning_policy.targeted_subset import _mastered_boundaries
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    root = Path.cwd()
    config = read_sft_config(plan['learner_config'], root=root)
    parent = read_sft_config(plan['parent_config'], root=root)
    verify_mlx_view(plan['parent_config'], plan['view'], root=root)
    manifest_path = Path(config['data']) / 'manifest.json'
    manifest_digest = file_digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    references = {name: json.loads(Path(plan[name]).read_text())
                  for name in ('learning_report', 'generalization_report', 'label_audit')}
    if (references['learning_report']['dataset_sha256'] != manifest_digest
            or references['generalization_report']['dataset_sha256'] != manifest_digest
            or references['label_audit']['dataset_manifest_sha256'] != manifest_digest
            or any(r['issues'] for r in references['label_audit']['reports'].values())):
        raise ValueError('source audit/row identity differs')
    if references['learning_report']['view_sha256'] != file_digest(Path(plan['view']) / 'view_manifest.json'):
        raise ValueError('prepared view identity differs')
    if manifest['splits']['train'][1] > manifest['splits']['valid'][0] or manifest['splits']['valid'][1] > manifest['sealed_start_ns']:
        raise ValueError('invalid chronological diagnostic roles')
    if type(plan['optimizer_steps']) is not int or plan['optimizer_steps'] < 1:
        raise ValueError('invalid diagnostic budget')
    destination = Path(plan['output'])
    destination.mkdir(parents=True, exist_ok=False)
    log = (destination / 'controller.log').open('x')
    def event(message):
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()
    report = {'schema': 'qwen_gradient_audit_v1', 'status': 'RUNNING', 'plan': plan,
        'plan_sha256': file_digest(args.config), 'dataset_sha256': manifest_digest,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p: file_digest(p) for p in (
            'src/propevolve/reasoning_policy/supervised_trainer.py',
            'src/propevolve/reasoning_policy/supervision.py', __file__)},
        'parent_weights': {n: file_digest(Path(parent['adapter_path']) / n)
            for n in ('adapters.safetensors', 'projector.safetensors')},
        'learner_config': config, 'updates': [],
        'limitations': ['Fresh optimizer state; not a resumed campaign trajectory.',
            '2024 is previously inspected development data, not final holdout.',
            'Management rows remain winner-entry conditioned.',
            'Small diagnostic cannot prove generalization or profitability.']}
    atomic_json(destination / 'report.json', report)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map, tree_flatten
        from mlx_lm.tuner.trainer import grad_checkpoint
        mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
        mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
        mx.random.seed(plan['seed'])
        datasets = {role: PreparedDataset(plan['view'], role) for role in ('train', 'valid')}
        indices = {'train': references['learning_report']['indices']['train'],
                   'valid': references['generalization_report']['indices']}
        rows = {role: [datasets[role][i] for i in idx] for role, idx in indices.items()}
        report['indices'] = indices
        config['mastered_anchor_retention'] = plan['retention']
        config['seed'] = plan['seed']
        event(f'[gradient-audit] loading actual frozen Qwen; training={len(rows["train"])} chronological={len(rows["valid"])}')
        policy = MLXActionPolicy.from_config(plan['parent_config'], root=root)
        model = policy.model
        configure_trainable_components(model, config['trainable_components'])
        if config['grad_checkpoint']:
            grad_checkpoint(model.layers[0])
        optimizer = build_optimizer(config)
        def assess(role):
            model.eval()
            scores = {}
            metrics = evaluate_action_validation(model, rows[role], config,
                on_scored=lambda i, s: scores.__setitem__(i, s))
            ordered = [scores[i] for i in range(len(rows[role]))]
            evidence = [decision_evidence(r, s, ambiguity_r=plan['ambiguity_r'])
                        for r, s in zip(rows[role], ordered)]
            return {'scores': ordered, 'boundaries': evidence, 'metrics': metrics}
        report['before'] = {}
        for role in rows:
            event(f'[gradient-audit] baseline_assessment={role} rows={len(rows[role])}')
            report['before'][role] = assess(role)
            atomic_json(destination / 'report.json', report)
        selected = []
        for row, scores in zip(rows['train'], report['before']['train']['scores']):
            item = copy.copy(row)
            item['mastered_anchor_retention'] = {'scores': scores, 'boundaries':
                _mastered_boundaries(row, dict(zip(row['action_targets']['names'], scores)))}
            selected.append(item)
        batches = list(tensor_batches(selected, config['batch_size'], config['max_seq_length'],
                                     include_partial=True, seed=plan['seed']))
        if len(batches) != config['grad_accumulation_steps']:
            raise ValueError('audited row block must match one production accumulation window')
        objectives = ('entry', 'direction', 'management', 'teacher', 'retention')
        previous = copy.deepcopy(report['before'])
        for step in range(1, plan['optimizer_steps'] + 1):
            start = time.monotonic()
            model.train()
            gradients, losses = {}, {}
            def accumulate(name):
                fn = (lambda m, *b: batch_loss(m, *b, config=config)[0]) if name == 'total' else (
                    lambda m, *b: batch_objective_loss(m, *b, config=config, objective=name))
                value_grad = nn.value_and_grad(model, fn)
                combined, loss_sum = None, 0.
                for batch in batches:
                    value, grad = value_grad(model, *batch)
                    mx.eval(value, grad)
                    loss_sum += float(value.item())
                    combined = grad if combined is None else tree_map(lambda a, b: a + b, combined, grad)
                    mx.eval(combined)
                    mx.clear_cache()
                averaged = tree_map(lambda a: a / len(batches), combined)
                mx.eval(averaged)
                return loss_sum / len(batches), averaged
            # These are diagnostic backward passes; no parameters change until total update.
            for name in (*objectives, 'total'):
                event(f'[gradient-audit] update={step}/{plan["optimizer_steps"]} objective={name}')
                losses[name], gradients[name] = accumulate(name)
            flat = {n: dict(tree_flatten(g)) for n, g in gradients.items()}
            summed = {k: sum(flat[n][k] for n in objectives) for k in flat['total']}
            squared_error = sum(float(mx.sum((summed[k].astype(mx.float32) - v.astype(mx.float32)) ** 2).item())
                                for k, v in flat['total'].items())
            squared_norm = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for v in flat['total'].values())
            relative_error = (squared_error / max(squared_norm, 1e-30)) ** .5
            if relative_error > plan['gradient_relative_tolerance']:
                raise ValueError(f'decomposed gradient differs from production: {relative_error}')
            before = dict(tree_flatten(model.trainable_parameters()))
            optimizer.update(model, gradients['total'])
            mx.eval(model.parameters(), optimizer.state)
            after = dict(tree_flatten(model.trainable_parameters()))
            delta = {k: after[k].astype(mx.float32) - before[k].astype(mx.float32) for k in before}
            geometry = gradient_geometry(flat, delta)
            result = {'step': step, 'losses': losses, 'gradient_relative_error': relative_error,
                      'geometry': geometry, 'panels': {}}
            del gradients, flat, summed, before, after, delta
            mx.clear_cache()
            for role in rows:
                event(f'[gradient-audit] update={step} reassessment={role}')
                current = assess(role)
                result['panels'][role] = {'evidence': current,
                    'vs_previous': compare_learning(previous[role]['boundaries'], current['boundaries']),
                    'vs_parent': compare_learning(report['before'][role]['boundaries'], current['boundaries'])}
                previous[role] = current
                event('[gradient-audit] ' + json.dumps({'step': step, 'role': role,
                    **result['panels'][role]['vs_parent']}))
            result['seconds'] = time.monotonic() - start
            report['updates'].append(result)
            atomic_json(destination / 'report.json', report)
        report['peak_memory_gb'] = mx.get_peak_memory() / 1024**3
        report['status'] = 'COMPLETE_DIAGNOSTIC'
        event('[gradient-audit] complete; no adapter promoted or parent modified')
    except Exception as error:
        report['status'] = 'FAILED'
        report['error'] = f'{type(error).__name__}: {error}'
        event('[gradient-audit] failed ' + report['error'])
        raise
    finally:
        atomic_json(destination / 'report.json', report)
        log.close()


if __name__ == '__main__':
    main()

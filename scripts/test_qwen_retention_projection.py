"""Bounded actual-Adam LoRA retention projection; no campaign promotion."""
import argparse
import copy
import json
from functools import partial
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import (
    decision_evidence, mean_batch_gradient, project_retention_displacement, update_progress,
)
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    batch_loss, batch_objective_loss, build_optimizer, configure_trainable_components,
    evaluate_action_validation, tensor_batches,
)
from propevolve.reasoning_policy.training_checkpoint import load_training_state, save_training_state
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    measured = json.loads(Path(plan['measurement_report']).read_text())
    if measured['status'] != 'COMPLETE_DIAGNOSTIC':
        raise ValueError('require completed gradient measurement')
    source_path = Path(measured['source_report'])
    if file_digest(source_path) != measured['source_sha256']:
        raise ValueError('source report changed')
    source = json.loads(source_path.read_text())
    prior = source['plan']
    config = read_sft_config(prior['learner_config'], root=Path.cwd())
    verify_mlx_view(prior['view_config'], prior['view'], root=Path.cwd())
    if file_digest(Path(config['data']) / 'manifest.json') != source['dataset_sha256']:
        raise ValueError('dataset changed')
    prefix = source_path.parent / 'probe-prefix-state'
    for name, digest in measured['prefix_hashes'].items():
        if file_digest(prefix / name) != digest:
            raise ValueError('prefix state changed')
    gradient_receipt = measured['gradient_artifacts']['complete_retention']
    if file_digest(gradient_receipt['path']) != gradient_receipt['sha256']:
        raise ValueError('cached gradient changed')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'config_sha256': file_digest(args.config),
        'measurement_sha256': file_digest(plan['measurement_report']),
        'source_sha256': measured['source_sha256'], 'updates': [],
        'chronological': 'NOT_RUN', 'promotion': 'NOT_AUTHORIZED'}
    atomic_json(output / 'report.json', report)
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.trainer import train, TrainingArgs
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    policy = MLXActionPolicy.from_config(source_path.parent / 'policy.json')
    model = policy.model
    configure_trainable_components(model, config['trainable_components'])
    optimizer = build_optimizer(config)
    receipt = load_training_state(prefix, model, optimizer)
    config['mastered_anchor_retention'] = {**prior['retention'],
        'supervision_weight': prior['retention_switch']['supervision_weight']}
    dataset = PreparedDataset(prior['view'], 'train')
    rows = [dataset[i] for i in source['indices']['train']]
    selected = []
    for i, retention in zip(receipt['selected_indices'], receipt['frozen_retention']):
        row = copy.copy(dataset[i])
        row['mastered_anchor_retention'] = retention
        selected.append(row)
    anchors = [r for r in selected if any(r['mastered_anchor_retention']['boundaries'].values())]
    retention_loss = partial(batch_objective_loss, config=config, objective='retention')

    def assess(panel):
        model.eval()
        scores = {}
        evaluate_action_validation(model, panel, config,
            on_scored=lambda i, s: scores.__setitem__(i, s))
        ordered = [scores[i] for i in range(len(panel))]
        return {'scores': ordered, 'boundaries': [decision_evidence(r, s,
            ambiguity_r=prior['ambiguity_r']) for r, s in zip(panel, ordered)]}

    def require_parity(actual, expected):
        if len(actual['scores']) != len(expected['scores']):
            raise ValueError('parity row mismatch')
        delta = max(abs(a-b) for left,right in zip(actual['scores'], expected['scores'])
                    for a,b in zip(left,right))
        if delta > prior['reload_tolerance']:
            raise ValueError(f'native control parity failed: {delta}')
        return delta

    initial = assess(rows)
    report['prefix_parity'] = require_parity(initial, receipt['assessment'])
    report['before'] = initial
    previous = initial
    accumulation = config['grad_accumulation_steps']
    start_offset = 0
    if plan.get('resume_report'):
        resume_path = Path(plan['resume_report'])
        resume = json.loads(resume_path.read_text())
        if resume['source_sha256'] != measured['source_sha256']:
            raise ValueError('resume source identity differs')
        last = resume['updates'][-1]
        start_offset = last['offset'] + 1
        resume_state = resume_path.parent / f'update-{start_offset:02d}' / 'state'
        restored = load_training_state(resume_state, model, optimizer)
        if restored['config_sha256'] != resume['config_sha256'] or restored['offset'] != last['offset']:
            raise ValueError('resume learner receipt differs')
        previous = assess(rows)
        report['resume_parity'] = require_parity(previous, restored['assessment'])
        report['resume_sha256'] = file_digest(resume_path)
        report['resumed_offset'] = start_offset
        if start_offset >= plan['maximum_updates']:
            raise ValueError('resume is already at diagnostic limit')
    for offset in range(start_offset, plan['maximum_updates']):
        stage = output / f'update-{offset + 1:02d}'
        stage.mkdir()
        before = dict(tree_flatten(model.trainable_parameters()))
        model.train()
        if offset == 0:
            retention_gradient = mx.load(gradient_receipt['path'])
        else:
            _, gradient = mean_batch_gradient(model,
                tensor_batches(anchors, prior['complete_anchor_probe']['anchor_batch_size'],
                    config['max_seq_length'], include_partial=True),
                loss=retention_loss, weight_by_rows=True)
            retention_gradient = dict(tree_flatten(gradient))
        draws = []
        native = TrainingArgs(batch_size=config['batch_size'], iters=accumulation,
            val_batches=0, steps_per_report=accumulation, steps_per_eval=accumulation+1,
            steps_per_save=accumulation, adapter_file=str(stage / 'native.safetensors'),
            max_seq_length=config['max_seq_length'],
            grad_checkpoint=config['grad_checkpoint'] and offset == start_offset,
            grad_accumulation_steps=accumulation, clear_cache_threshold=config['clear_cache_threshold'])
        train(model, optimizer, selected, None, args=native,
            loss=partial(batch_loss, config=config),
            iterate_batches=partial(tensor_batches, seed=prior['seed'], include_partial=True,
                skip_batches=receipt['iteration'] + offset*accumulation,
                sampling_strategy=prior['sampling_strategy'],
                on_selected=lambda ids: draws.append([receipt['selected_indices'][i] for i in ids])))
        proposed = dict(tree_flatten(model.trainable_parameters()))
        native_evidence = assess(rows)
        native_delta = None
        if offset == 0:
            if draws != source['complete_anchor_probe']['next_microbatch_indices']:
                raise ValueError('native next batches differ')
            native_delta = require_parity(native_evidence, source['complete_anchor_probe']['control'])
        corrected, projection = project_retention_displacement(before, proposed, retention_gradient)
        model.load_weights(list(corrected.items()), strict=False)
        current = assess(rows)
        progress = update_progress(previous['boundaries'], current['boundaries'])
        cumulative = update_progress(initial['boundaries'], current['boundaries'])
        protected = update_progress(source['before']['train']['boundaries'], current['boundaries'])
        passed = (progress['forgotten'] <= plan['maximum_forgotten']
            and cumulative['forgotten'] <= plan['maximum_forgotten']
            and protected['forgotten'] <= plan['maximum_forgotten']
            and cumulative['acquired'] >= plan['minimum_acquired'])
        step = {'offset': offset, 'native_parity': native_delta, 'batches': draws,
            'projection': projection, 'native': native_evidence, 'candidate': current,
            'progress': progress, 'cumulative': cumulative, 'vs_original_parent': protected,
            'passed': passed}
        report['updates'].append(step)
        atomic_json(output / 'report.json', report)
        print('[retention-step]', json.dumps({k:v for k,v in step.items()
            if k not in ('native','candidate','batches')}), flush=True)
        save_training_state(stage / 'state', model, optimizer, {'offset': offset,
            'assessment': current, 'config_sha256': report['config_sha256']})
        if not passed:
            report['status'] = 'FALSIFIED'
            break
        previous = current
    else:
        if cumulative['acquired'] < plan.get('final_minimum_acquired', plan['minimum_acquired']):
            report['status'] = 'FALSIFIED_NO_ACQUISITION'
        else:
            report['status'] = 'PASSED_BOUNDED_TRAINING'
            valid = PreparedDataset(prior['view'], 'valid')
            report['chronological'] = assess([valid[i] for i in source['indices']['valid']])
    # Confirm this actual endpoint survives ordinary model/optimizer reload.
    load_training_state(stage / 'state', model, optimizer)
    report['reload_delta'] = require_parity(assess(rows), current)
    atomic_json(output / 'report.json', report)
    print('[retention-step] status=' + report['status'], flush=True)


if __name__ == '__main__':
    main()

"""Config-driven real-Qwen diagnostic using native production loss/batches/optimizer.

No mock network and no campaign launch. Writes explicit acquisition, sequential
retention and chronological-development evidence, including failures.
"""
import argparse
import copy
import gc
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

from propevolve.reasoning_policy.decisive_learning import decision_evidence, compare_learning, evaluation_recipe, require_initial_score_parity, fixed_diagnostic_indices, update_progress
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import read_sft_config, PreparedDataset, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.projector import export_policy_weights
from propevolve.reasoning_policy.supervised_trainer import (
    configure_trainable_components, build_optimizer, tensor_batches, batch_loss,
    evaluate_action_validation,
    batch_objective_loss,
)
from propevolve.reasoning_policy.targeted_subset import _mastered_boundaries
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    destination = Path(plan['output'])
    root = Path.cwd()
    config = read_sft_config(plan['learner_config'], root=root)
    parent = read_sft_config(plan['parent_config'], root=root)
    config['resume_adapter_file'] = str(Path(parent['adapter_path']) / 'adapters.safetensors')
    config['resume_adapter_requirements'] = None
    manifest = Path(config['data']) / 'manifest.json'
    audit = json.loads(Path(plan['label_audit']).read_text())
    if audit['dataset_manifest_sha256'] != file_digest(manifest):
        raise ValueError('audited corpus identity differs')
    if any(r['issues'] for r in audit['reports'].values()):
        raise ValueError('label audit has unresolved integrity faults')
    verify_mlx_view(plan.get('view_config', plan['parent_config']), plan['view'], root=root)
    destination.mkdir(parents=True, exist_ok=False)
    class Tee:
        def __init__(self, original, file):
            self.original, self.file = original, file
        def write(self, value):
            self.original.write(value)
            self.file.write(value)
            self.file.flush()
        def flush(self):
            self.original.flush()
            self.file.flush()
    log = (destination / 'controller.log').open('x')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    datasets = {role: PreparedDataset(plan['view'], role) for role in ('train', 'valid')}
    rng = np.random.default_rng(plan['seed'])
    indices, rows = {}, {}
    required = {'WAIT', 'ENTER_LONG_1', 'ENTER_SHORT_1', 'HOLD', 'CLOSE'}
    for role, dataset in datasets.items():
        if 'fixed_indices' in plan:
            indices[role] = fixed_diagnostic_indices(dataset.rows, plan['fixed_indices'][role],
                minimum_gap=plan['minimum_economic_gap'])
            rows[role] = [dataset[i] for i in indices[role]]
            continue
        groups = {name: [] for name in required}
        for i, row in enumerate(dataset.rows):
            economics = row['action_targets']
            values = dict(zip(economics['names'], economics['values']))
            target = row['target_name']
            gap = values[target] - max(v for n, v in values.items() if n != target)
            if gap >= plan['minimum_economic_gap']:
                groups[target].append(i)
        indices[role] = []
        for name in sorted(groups):
            pool = groups[name]
            if len(pool) < plan['rows_per_action']:
                raise ValueError('insufficient independently audited clear action rows')
            indices[role].extend(map(int, rng.choice(pool, plan['rows_per_action'], replace=False)))
        rows[role] = [dataset[i] for i in indices[role]]
    config['seed'] = plan['seed']
    config['mastered_anchor_retention'] = plan['retention']
    report = {'schema': 'qwen_decisive_learning_v1', 'plan': plan,
        'plan_sha256': file_digest(args.config),
        'view_sha256': file_digest(Path(plan['view']) / 'view_manifest.json'),
        'dataset_sha256': file_digest(manifest), 'indices': indices,
        'splits': json.loads(manifest.read_text())['splits'],
        'status': 'RUNNING', 'stages': [], 'simulator_transfer': 'NOT_RUN',
        'limitations': ['Clear-label diagnostic only; ambiguous rows audited separately.',
            '2024 is development, not untouched final validation.',
            'Management coverage and entry conditioning must be interpreted using dataset lineage.']}
    atomic_json(destination / 'report.json', report)
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import train, TrainingArgs
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    mx.random.seed(plan['seed'])
    print('[decisive] loading actual Qwen frozen parent', flush=True)
    policy = MLXActionPolicy.from_config(plan['parent_config'], root=root)
    model = policy.model
    if config['projector'] != parent['projector']:
        from propevolve.reasoning_policy.projector import (
            state_extension_of, attach_projector, restore_projector)
        if not state_extension_of(parent['projector'], config['projector']):
            raise ValueError('diagnostic only permits an appended causal state extension')
        rng_state = mx.random.state
        attach_projector(model, config['projector'])
        restore_projector(model, parent['adapter_path'], allow_state_extension=True)
        mx.random.state = rng_state
        policy.projector_config = config['projector']
    configure_trainable_components(model, config['trainable_components'])
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

    previous = assess('train')
    report['before'] = {'train': previous, 'valid': assess('valid')}
    if plan.get('initial_score_reference'):
        reference_path = Path(plan['initial_score_reference'])
        delta = require_initial_score_parity(json.loads(reference_path.read_text()),
            report['before'], report['indices'], tolerance=plan['reload_tolerance'])
        report['initial_score_parity'] = {'maximum_delta': delta,
            'reference_sha256': file_digest(reference_path)}
    atomic_json(destination / 'report.json', report)
    print('[decisive] baseline=' + json.dumps(compare_learning(
        previous['boundaries'], previous['boundaries'])), flush=True)
    for phase in plan['phases']:
        # New correct boundaries become anchors at the next phase; old mistakes
        # are never protected. Within the phase the reference stays frozen.
        selected = []
        selected_indices = []
        for row_index, (row, scores) in enumerate(zip(rows['train'], previous['scores'])):
            if row['target_name'] not in phase['actions']:
                continue
            item = copy.copy(row)
            names = row['action_targets']['names']
            item['mastered_anchor_retention'] = {
                'scores': scores,
                'boundaries': _mastered_boundaries(row, dict(zip(names, scores)))}
            selected.append(item)
            selected_indices.append(indices['train'][row_index])
        stage = destination / phase['name']
        stage.mkdir()
        accumulation = config['grad_accumulation_steps']
        iterations = phase['optimizer_steps'] * accumulation
        probe = plan.get('endpoint_probe')
        snapshot = None
        snapshot_evidence = None
        iteration_offset = 0
        switch = plan.get('retention_switch')
        phase_draws = []
        def record_draw(local_indices):
            phase_draws.append([selected_indices[i] for i in local_indices])
        report.setdefault('sampling', {})[phase['name']] = {
            'selected_indices': selected_indices, 'microbatch_indices': phase_draws}
        class Callback:
            def on_train_loss_report(self, event):
                nonlocal previous, snapshot, snapshot_evidence
                event = {**event, 'iteration': event['iteration'] + iteration_offset}
                current = assess('train')
                result = {'phase': phase['name'], 'iteration': event['iteration'],
                    'train_loss': event['train_loss'], 'evidence': current,
                    'vs_previous': compare_learning(previous['boundaries'], current['boundaries']),
                    'vs_parent': compare_learning(report['before']['train']['boundaries'], current['boundaries'])}
                report['stages'].append(result)
                previous = current
                update = event['iteration'] // accumulation
                if switch and switch.get('reference_report') and update <= switch['after_updates']:
                    reference = json.loads(Path(switch['reference_report']).read_text())
                    expected = reference['stages'][update-1]['evidence']['scores']
                    delta = max(abs(a-b) for left,right in zip(expected, current['scores'])
                                for a,b in zip(left,right))
                    if delta > plan['reload_tolerance']:
                        raise ValueError(f'unchanged optimizer prefix differs: {delta}')
                    result['prefix_score_delta'] = delta
                if probe and update == probe['update'] - 1:
                    snapshot = tree_map(lambda x: x, model.trainable_parameters())
                    mx.eval(snapshot)
                    snapshot_evidence = current
                if probe and update == probe['update']:
                    if snapshot is None:
                        raise ValueError('endpoint probe requires a preceding optimizer snapshot')
                    reference = json.loads(Path(probe['reference_report']).read_text())
                    for expected, actual in (
                        (reference['stages'][update-2]['evidence'], snapshot_evidence),
                        (reference['stages'][update-1]['evidence'], current),
                    ):
                        delta = max(abs(a-b) for left,right in zip(expected['scores'], actual['scores'])
                                    for a,b in zip(left,right))
                        if delta > plan['reload_tolerance']:
                            raise ValueError(f'optimizer-prefix reproduction differs: {delta}')
                    proposal = tree_map(lambda x: x, model.trainable_parameters())
                    mx.eval(proposal)
                    report['endpoint_probe'] = {'update': update, 'endpoints': [],
                        'reference_sha256': file_digest(probe['reference_report']),
                        'before': snapshot_evidence,
                        'optimizer_state': 'one native proposal; evaluation does not advance moments'}
                    try:
                        for scale in probe['scales']:
                            if not 0 <= scale <= 1:
                                raise ValueError('endpoint scale must lie in [0,1]')
                            weights = (proposal if scale == 1 else snapshot if scale == 0 else
                                tree_map(lambda old,new: old + scale*(new-old), snapshot, proposal))
                            model.update(weights)
                            mx.eval(model.parameters())
                            endpoint = {'scale': scale, 'train': assess('train'), 'valid': assess('valid')}
                            endpoint['progress'] = update_progress(
                                snapshot_evidence['boundaries'], endpoint['train']['boundaries'])
                            report['endpoint_probe']['endpoints'].append(endpoint)
                            atomic_json(destination / 'report.json', report)
                            print('[endpoint] ' + json.dumps({'scale': scale, **endpoint['progress']}), flush=True)
                    finally:
                        model.update(proposal)
                        mx.eval(model.parameters())
                atomic_json(destination / 'report.json', report)
                print('[decisive] ' + json.dumps({k: v for k, v in result.items() if k != 'evidence'}), flush=True)
                model.train()
            def on_val_loss_report(self, event):
                pass
        segments = [iterations]
        if switch:
            prefix = switch['after_updates'] * accumulation
            if not 0 < prefix < iterations or probe:
                raise ValueError('retention switch requires a nonempty unchanged prefix and suffix')
            segments = [prefix, iterations - prefix]
        for segment_index, segment_iterations in enumerate(segments):
            segment_config = config
            if segment_index:
                segment_config = {**config, 'mastered_anchor_retention': {
                    **config['mastered_anchor_retention'],
                    'supervision_weight': switch['supervision_weight']}}
                from propevolve.reasoning_policy.targeted_subset import validate_mastered_anchor_retention
                validate_mastered_anchor_retention(segment_config['mastered_anchor_retention'])
            native = TrainingArgs(batch_size=config['batch_size'], iters=segment_iterations,
                val_batches=0, steps_per_report=accumulation, steps_per_eval=segment_iterations,
                steps_per_save=segment_iterations, adapter_file=str(stage / 'native.safetensors'),
                max_seq_length=config['max_seq_length'],
                grad_checkpoint=config['grad_checkpoint'] and segment_index == 0,
                grad_accumulation_steps=accumulation, clear_cache_threshold=config['clear_cache_threshold'])
            train(model, optimizer, selected, None, args=native,
                loss=partial(batch_loss, config=segment_config),
                iterate_batches=partial(tensor_batches, seed=config['seed'], include_partial=True,
                    skip_batches=iteration_offset,
                    on_selected=record_draw,
                    cycle_rows=plan.get('sampling_cycle_rows'),
                    sampling_strategy=plan.get('sampling_strategy', 'random')),
                training_callback=Callback())
            iteration_offset += segment_iterations
    if plan.get('complete_anchor_probe'):
        from itertools import islice
        from propevolve.reasoning_policy.decisive_learning import mean_batch_gradient
        probe = plan['complete_anchor_probe']
        if probe['update'] != phase['optimizer_steps'] + 1:
            raise ValueError('complete anchor probe must immediately follow the frozen prefix')
        reference = json.loads(Path(probe['reference_report']).read_text())
        before_probe = assess('train')
        def require_probe_parity(actual, expected):
            delta = max(abs(a-b) for left,right in zip(actual['scores'], expected['scores'])
                        for a,b in zip(left,right))
            if delta > plan['reload_tolerance']:
                raise ValueError(f'complete-anchor native proposal reproduction differs: {delta}')
            return delta
        prefix_delta = require_probe_parity(before_probe,
            reference['stages'][probe['update']-2]['evidence'])
        saved_weights = tree_map(lambda x: x, model.trainable_parameters())
        saved_optimizer = tree_map(lambda x: x, optimizer.state)
        saved_random = list(mx.random.state)
        mx.eval(saved_weights, saved_optimizer, saved_random)
        from propevolve.reasoning_policy.training_checkpoint import save_training_state
        save_training_state(destination / 'probe-prefix-state', model, optimizer, {
            'plan_sha256': report['plan_sha256'], 'view_sha256': report['view_sha256'],
            'dataset_sha256': report['dataset_sha256'], 'iteration': iterations,
            'frozen_retention': [row['mastered_anchor_retention'] for row in selected],
            'selected_indices': selected_indices, 'assessment': before_probe})
        next_draws = []
        next_batches = list(islice(tensor_batches(selected, config['batch_size'],
            config['max_seq_length'], loop=True, seed=config['seed'], include_partial=True,
            sampling_strategy=plan.get('sampling_strategy', 'random'), skip_batches=iterations,
            on_selected=lambda ids: next_draws.append([selected_indices[i] for i in ids])),
            accumulation))
        model.train()
        original_loss = lambda m, *batch: batch_loss(m, *batch, config=segment_config)[0]
        from dataclasses import replace
        control_args = replace(native, iters=accumulation,
            steps_per_report=accumulation, steps_per_save=accumulation,
            grad_checkpoint=False, adapter_file=str(destination / 'control.safetensors'))
        train(model, optimizer, selected, None, args=control_args,
            loss=partial(batch_loss, config=segment_config),
            iterate_batches=partial(tensor_batches, seed=config['seed'], include_partial=True,
                skip_batches=iterations, sampling_strategy=plan.get('sampling_strategy', 'random')))
        control = assess('train')
        control_delta = require_probe_parity(control,
            reference['stages'][probe['update']-1]['evidence'])
        model.update(saved_weights)
        optimizer.state = saved_optimizer
        mx.random.state = saved_random
        mx.eval(model.parameters(), optimizer.state, mx.random.state)
        model.train()
        retention_loss = partial(batch_objective_loss,
            config=segment_config, objective='retention')
        acquisition_loss = lambda m, *batch: original_loss(m, *batch) - retention_loss(m, *batch)
        acquisition_value, acquisition_gradient = mean_batch_gradient(model, next_batches,
            loss=acquisition_loss, weight_by_rows=False)
        anchors = [row for row in selected
                   if any(row['mastered_anchor_retention']['boundaries'].values())]
        anchor_indices = [selected_indices[i] for i,row in enumerate(selected)
                          if any(row['mastered_anchor_retention']['boundaries'].values())]
        anchor_value, anchor_gradient = mean_batch_gradient(model,
            tensor_batches(anchors, probe['anchor_batch_size'], config['max_seq_length'],
                           include_partial=True), loss=retention_loss, weight_by_rows=True)
        gradient = tree_map(lambda a,b: a+b, acquisition_gradient, anchor_gradient)
        optimizer.update(model, gradient)
        mx.eval(model.parameters(), optimizer.state)
        candidate = assess('train')
        acquisition_after = sum(float(acquisition_loss(model, *b).item())
                                for b in next_batches) / len(next_batches)
        report['complete_anchor_probe'] = {
            'update': probe['update'], 'reference_sha256': file_digest(probe['reference_report']),
            'prefix_score_delta': prefix_delta, 'control_score_delta': control_delta,
            'next_microbatch_indices': next_draws, 'anchor_indices': anchor_indices,
            'before': before_probe, 'control': control, 'candidate': candidate,
            'control_progress': update_progress(before_probe['boundaries'], control['boundaries']),
            'candidate_progress': update_progress(before_probe['boundaries'], candidate['boundaries']),
            'acquisition_loss_before': float(acquisition_value.item()),
            'acquisition_loss_after': acquisition_after,
            'complete_anchor_loss_before': float(anchor_value.item()),
            'normalization': 'equal acquisition microbatch mean plus complete anchor row mean',
            'optimizer': 'same native optimizer; restored weights, complete state and RNG before candidate'}
        atomic_json(destination / 'report.json', report)
        print('[complete-anchor] ' + json.dumps(report['complete_anchor_probe']['candidate_progress']), flush=True)
    report['after'] = {'train': assess('train'), 'valid': assess('valid')}
    for role in ('train', 'valid'):
        report[role + '_comparison'] = compare_learning(report['before'][role]['boundaries'],
                                                        report['after'][role]['boundaries'])
    (destination / 'adapter').mkdir()
    export_policy_weights(model, destination / 'adapter')
    atomic_json(destination / 'adapter' / 'adapter_config.json', config)
    # Load using the ordinary policy interface, not a diagnostic-only loader.
    saved_config = evaluation_recipe(config, adapter_path=str(destination / 'adapter'))
    atomic_json(destination / 'policy.json', saved_config)
    del optimizer, policy, model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    policy = MLXActionPolicy.from_config(destination / 'policy.json')
    model = policy.model
    repeated = assess('train')
    delta = max(abs(a-b) for left, right in zip(report['after']['train']['scores'], repeated['scores'])
                for a, b in zip(left, right))
    report['reload_max_score_delta'] = delta
    report['reload_parity'] = delta <= plan['reload_tolerance']
    report['status'] = 'COMPLETE_DIAGNOSTIC'
    report['acquisition_passed'] = report['train_comparison']['all_clear_correct']
    report['retention_passed'] = all(s['vs_previous']['forgotten'] == 0 for s in report['stages'])
    atomic_json(destination / 'report.json', report)
    print('[decisive] complete acquisition=' + str(report['acquisition_passed']) +
          ' retention=' + str(report['retention_passed']) + ' reload=' + str(report['reload_parity']), flush=True)


if __name__ == '__main__':
    main()

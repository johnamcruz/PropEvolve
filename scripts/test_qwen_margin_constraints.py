"""Test individual acquisition/retention constraints on a saved real native step."""
import argparse
import copy
import json
from functools import partial
from pathlib import Path
import numpy as np

from propevolve.reasoning_policy.decisive_learning import (
    component_snapshot, constraint_multipliers, decision_evidence,
    differentiable_boundary_margin, fractional_snapshot, mean_batch_gradient, precise_gram,
    retained_margin_progress, update_progress,
)
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import (
    _batch_outputs, batch_loss, build_optimizer, configure_trainable_components, evaluate_action_validation, tensor_batches,
)
from propevolve.reasoning_policy.training_checkpoint import load_training_state, save_training_state
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
    before_step = next(s for s in source['updates'] if s['offset'] == plan['before_offset'])
    after_step = next(s for s in source['updates'] if s['offset'] == plan['after_offset'])
    before_path = path.parent / f"update-{plan['before_offset']+1:02d}/state/weights.safetensors"
    native_path = path.parent / f"update-{plan['after_offset']+1:02d}/native.safetensors"
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'config_sha256': file_digest(args.config),
        'source_sha256': file_digest(path), 'weights': {str(p): file_digest(p) for p in (before_path,native_path)},
        'constraints': [], 'variants': [], 'promotion': 'NOT_AUTHORIZED', 'holdouts': 'NOT_ACCESSED'}
    atomic_json(output / 'report.json', report)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_memory_limit(int(plan['memory_gb']*1024**3))
    mx.set_cache_limit(int(plan['cache_mb']*1024**2))
    policy = MLXActionPolicy.from_config(original_path.parent / 'policy.json')
    model = policy.model
    configure_trainable_components(model, config['trainable_components'])
    before, native = mx.load(str(before_path)), mx.load(str(native_path))
    component_snapshot(before,native,component='both')
    data = PreparedDataset(prior['view'], 'train')
    rows = [data[i] for i in original['indices']['train']]

    def assess(weights):
        model.load_weights(list(weights.items()), strict=False)
        model.eval()
        scores = {}
        evaluate_action_validation(model,rows,config,on_scored=lambda i,s:scores.__setitem__(i,s))
        ordered = [scores[i] for i in range(len(rows))]
        return {'scores': ordered, 'boundaries': [decision_evidence(r,s,ambiguity_r=prior['ambiguity_r']) for r,s in zip(rows,ordered)]}

    if plan.get('continue_report'):
        previous_path = Path(plan['continue_report'])
        previous = json.loads(previous_path.read_text())
        if previous['source_sha256'] != report['source_sha256']:
            raise ValueError('continuation source changed')
        chosen = next(v for v in previous['variants'] if v['fraction'] == plan['continue_fraction'])
        if not chosen['passed']:
            target = previous['acquisition_target']
            old = previous['before']['boundaries'][target['row']][target['boundary']]['margin']
            new = chosen['evidence']['boundaries'][target['row']][target['boundary']]['margin']
            if (not plan.get('allow_diagnostic_partial', False)
                    or not retained_margin_progress(old,new,forgotten=chosen['progress']['forgotten'],
                        minimum_gain=plan['minimum_partial_gain'])):
                raise ValueError('cannot continue a failed candidate without retained diagnostic progress')
            report['partial_continuation'] = {'margin_gain':new-old,'promotion':'NOT_AUTHORIZED'}
        optimizer = build_optimizer(config)
        native_offset = previous.get('native_offset', plan['after_offset']) + 1
        state = previous_path.parent / 'native-state'
        if not state.exists():
            if plan['continue_report'] != plan.get('initial_candidate_report'):
                raise ValueError('continuation optimizer state missing')
            state = path.parent / f"update-{plan['after_offset']+1:02d}/state"
        load_training_state(state, model, optimizer)
        before_path = previous_path.parent / f"candidate-{plan['continue_fraction']}.safetensors"
        before = mx.load(str(before_path))
        observed = assess(before)
        delta = max(abs(a-b) for x,y in zip(observed['scores'],chosen['evidence']['scores']) for a,b in zip(x,y))
        if delta > prior['reload_tolerance']:
            raise ValueError(f'continuation parity failed: {delta}')
        report['continuation_parity'] = delta
        report['continue_sha256'] = file_digest(previous_path)
        report['native_offset'] = native_offset
        before_step = {'candidate': observed}
        prefix = json.loads((original_path.parent/'probe-prefix-state/receipt.json').read_text())['receipt']
        selected_rows = []
        for i, retention in zip(prefix['selected_indices'],prefix['frozen_retention']):
            row = copy.copy(data[i]); row['mastered_anchor_retention'] = retention
            selected_rows.append(row)
        config['mastered_anchor_retention'] = {**prior['retention'],
            'supervision_weight':prior['retention_switch']['supervision_weight']}
        from mlx_lm.tuner.trainer import train, TrainingArgs
        accumulation = config['grad_accumulation_steps']
        native_path = output/'native.safetensors'
        train(model,optimizer,selected_rows,None,
            args=TrainingArgs(batch_size=config['batch_size'],iters=accumulation,
                val_batches=0,steps_per_report=accumulation,steps_per_eval=accumulation+1,
                steps_per_save=accumulation,adapter_file=str(native_path),
                max_seq_length=config['max_seq_length'],grad_checkpoint=config['grad_checkpoint'],
                grad_accumulation_steps=accumulation,clear_cache_threshold=config['clear_cache_threshold']),
            loss=partial(batch_loss,config=config),
            iterate_batches=partial(tensor_batches,seed=prior['seed'],include_partial=True,
                skip_batches=prefix['iteration']+native_offset*accumulation,
                sampling_strategy=prior['sampling_strategy']))
        native = dict(tree_flatten(model.trainable_parameters()))
        after_step = {'native': assess(native)}
        save_training_state(output/'native-state',model,optimizer,{'native_offset':native_offset,
            'config_sha256':report['config_sha256']})
        report['weights'] = {str(p):file_digest(p) for p in (before_path,native_path)}
        atomic_json(output/'report.json',report)
    baseline = None
    for name, weights, reference in [('before',before,before_step['candidate']),('native',native,after_step['native'])]:
        if plan.get('continue_report'):
            # These exact weights were assessed above; a new native step has
            # no historical score reference. Do not repeat inference or claim
            # parity by comparing a measurement against itself.
            observed = reference
            delta = report['continuation_parity'] if name == 'before' else None
        else:
            observed = assess(weights)
            delta = max(abs(a-b) for x,y in zip(observed['scores'],reference['scores']) for a,b in zip(x,y))
            if delta > prior['reload_tolerance']:
                raise ValueError(f'{name} parity failed: {delta}')
        report[name] = observed
        report[name+'_parity'] = delta
        if name == 'before':
            baseline = observed
    atomic_json(output / 'report.json',report)
    model.load_weights(list(before.items()),strict=False)
    keys = sorted(before)
    group_norms = {}
    for group in ('lora','projector'):
        names = [n for n in keys if n.startswith('market_projector.') == (group=='projector')]
        group_norms[group] = float(mx.sqrt(sum(mx.sum((native[n].astype(mx.float32)-before[n].astype(mx.float32))**2) for n in names)).item())
        if group_norms[group] <= 0:
            raise ValueError('zero native component displacement')
    scales = {n: group_norms['projector' if n.startswith('market_projector.') else 'lora'] for n in keys}
    displacement = np.concatenate([np.asarray((native[n].astype(mx.float32)-before[n].astype(mx.float32))/scales[n]).reshape(-1) for n in keys])
    if 'gram_block_size' in plan:
        displacement = displacement.astype(np.float64)
    constraints = [(i,k,v['margin'],min(v['margin'],plan['target_margin']))
                   for i,row in enumerate(baseline['boundaries']) for k,v in row.items() if v['correct'] and not v['ambiguous']]
    if plan.get('acquisition_selection') == 'closest_clear_mistake':
        _, acquire_row, acquire_boundary = min((abs(v['margin']),i,k)
            for i,row in enumerate(baseline['boundaries']) for k,v in row.items()
            if not v['correct'] and not v['ambiguous'])
    else:
        acquire_row, acquire_boundary = plan['acquire_row'], plan['acquire_boundary']
    report['acquisition_target'] = {'row':acquire_row,'boundary':acquire_boundary}
    selected = baseline['boundaries'][acquire_row][acquire_boundary]
    if selected['correct'] or selected['ambiguous']:
        raise ValueError('acquisition target must be an authenticated mistake')
    constraints.append((acquire_row,acquire_boundary,selected['margin'],plan['target_margin']))
    gradients, targets = [], []
    eval_config = {**config,'mastered_anchor_retention':None,'error_selected_distillation':None,'market_distillation':None}
    for index,(i,boundary,margin,target) in enumerate(constraints):
        batch = next(tensor_batches([rows[i]],1,config['max_seq_length'],include_partial=True))
        names = rows[i]['action_targets']['names']

        def margin_loss(m,*tensors):
            scores = _batch_outputs(m,*tensors[:10],config=eval_config)[2][0]
            return differentiable_boundary_margin(scores,names,boundary,xp=mx)

        value, gradient = mean_batch_gradient(model,[batch],loss=margin_loss,weight_by_rows=True)
        flat = dict(tree_flatten(gradient))
        gradient_path = output / f'gradient-{index:02d}.safetensors'
        mx.save_safetensors(str(gradient_path),flat)
        vector = np.concatenate([np.asarray(flat[n].astype(mx.float32)*scales[n]).reshape(-1) for n in keys])
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('invalid boundary gradient')
        gradients.append(vector/norm)
        targets.append((target-margin)/norm)
        report['constraints'].append({'row':i,'boundary':boundary,'before_margin':margin,'target':target,
            'gradient_margin':float(value.item()),'gradient_path':str(gradient_path),'sha256':file_digest(gradient_path),'norm':norm})
        atomic_json(output/'report.json',report)
        print('[margin-constraint]',index+1,'/',len(constraints),'row',i,boundary,flush=True)
        del flat, gradient, batch
        mx.clear_cache()
    matrix = np.stack(gradients)
    del gradients
    gram = (precise_gram(matrix, block_size=plan['gram_block_size'])
            if 'gram_block_size' in plan else (matrix @ matrix.T).astype(float))
    multipliers = constraint_multipliers(gram,np.asarray(targets)-matrix@displacement,
        tolerance=plan['solver_tolerance'],maximum_cycles=plan['maximum_solver_cycles'])
    corrected = displacement + multipliers @ matrix
    ratio = float(np.linalg.norm(corrected)/np.linalg.norm(displacement))
    report['relative_displacement'] = ratio
    report['linear_residual'] = (np.asarray(targets)-matrix@corrected).tolist()
    if ratio > plan['maximum_relative_displacement']:
        report['status'] = 'FALSIFIED_STEP_BUDGET'
        atomic_json(output/'report.json',report)
        return
    candidate = {}
    offset = 0
    for n in keys:
        size = before[n].size
        delta = mx.array(corrected[offset:offset+size].reshape(before[n].shape),dtype=mx.float32)*scales[n]
        candidate[n] = (before[n].astype(mx.float32)+delta).astype(before[n].dtype)
        offset += size
    for fraction in plan['fractions']:
        weights = fractional_snapshot(before,candidate,fraction=fraction)
        evidence = assess(weights)
        progress = update_progress(baseline['boundaries'],evidence['boundaries'])
        passed = progress['forgotten']==0 and evidence['boundaries'][acquire_row][acquire_boundary]['correct']
        report['variants'].append({'fraction':fraction,'evidence':evidence,'progress':progress,'passed':passed})
        mx.save_safetensors(str(output/f'candidate-{fraction}.safetensors'),weights)
        print('[margin-candidate]',fraction,json.dumps(progress),'passed',passed,flush=True)
        atomic_json(output/'report.json',report)
    report['status'] = 'PASSED_SINGLE_STEP' if any(v['passed'] for v in report['variants']) else 'FALSIFIED_FINITE_STEP'
    atomic_json(output/'report.json',report)


if __name__=='__main__':
    main()

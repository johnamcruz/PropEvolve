"""Reload a bounded margin candidate and assess the fixed chronological panel."""
import argparse
import json
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import decision_evidence, update_progress
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config, verify_mlx_view
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import evaluate_action_validation
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    baseline = json.loads(Path(plan['baseline_report']).read_text())
    candidate_path = Path(plan['candidate_report'])
    candidate = json.loads(candidate_path.read_text())
    if candidate['status'] != 'PASSED_SINGLE_STEP' or candidate['source_sha256'] != baseline['source_sha256']:
        raise ValueError('candidate is not an accepted matched diagnostic')
    selected = next(v for v in candidate['variants'] if v['fraction'] == plan['fraction'])
    if not selected['passed']:
        raise ValueError('chosen candidate failed')
    original_path = Path(plan['dataset_reference_report'])
    original = json.loads(original_path.read_text())
    source = json.loads(Path(plan['source_report']).read_text())
    if (file_digest(plan['source_report']) != baseline['source_sha256']
            or source['source_sha256'] != file_digest(original_path)
            or baseline['weights'].get(plan['before_weights']) != file_digest(plan['before_weights'])):
        raise ValueError('assessment lineage changed')
    prior = original['plan']
    config = read_sft_config(prior['learner_config'], root=Path.cwd())
    verify_mlx_view(prior['view_config'], prior['view'], root=Path.cwd())
    if file_digest(Path(config['data'])/'manifest.json') != original['dataset_sha256']:
        raise ValueError('dataset changed')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    result = {'status':'RUNNING', 'promotion':'NOT_AUTHORIZED',
        'config_sha256':file_digest(args.config), 'candidate_sha256':file_digest(candidate_path),
        'baseline_sha256':file_digest(plan['baseline_report']), 'variants':{}}
    atomic_json(output/'report.json',result)
    import mlx.core as mx
    mx.set_memory_limit(int(plan['memory_gb']*1024**3))
    mx.set_cache_limit(int(plan['cache_mb']*1024**2))
    policy = MLXActionPolicy.from_config(original_path.parent/'policy.json')
    paths = {'before':Path(plan['before_weights']),
             'candidate':candidate_path.parent/f"candidate-{plan['fraction']}.safetensors"}
    result['weights'] = {key:file_digest(path) for key,path in paths.items()}
    for name,path in paths.items():
        policy.model.load_weights(str(path),strict=False)
        policy.model.eval()
        result['variants'][name] = {}
        for role in ('train','valid'):
            data = PreparedDataset(prior['view'],role)
            rows = [data[i] for i in original['indices'][role]]
            scores = {}
            metrics = evaluate_action_validation(policy.model,rows,config,
                on_scored=lambda i,s:scores.__setitem__(i,s))
            ordered = [scores[i] for i in range(len(rows))]
            evidence = {'metrics':metrics,'scores':ordered,'boundaries':[
                decision_evidence(r,s,ambiguity_r=prior['ambiguity_r']) for r,s in zip(rows,ordered)]}
            if role=='train':
                expected = baseline['before'] if name=='before' else selected['evidence']
                delta = max(abs(a-b) for x,y in zip(ordered,expected['scores']) for a,b in zip(x,y))
                if delta > prior['reload_tolerance']:
                    raise ValueError(f'reload parity failed: {delta}')
                evidence['reload_delta'] = delta
            result['variants'][name][role] = evidence
            atomic_json(output/'report.json',result)
            print('[margin-assessment]',name,role,'complete',flush=True)
    result['progress'] = {role:update_progress(result['variants']['before'][role]['boundaries'],
        result['variants']['candidate'][role]['boundaries']) for role in ('train','valid')}
    result['status'] = 'COMPLETE_DIAGNOSTIC'
    atomic_json(output/'report.json',result)
    print('[margin-assessment]',json.dumps(result['progress']),flush=True)


if __name__=='__main__':
    main()

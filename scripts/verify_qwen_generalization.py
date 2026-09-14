"""Basic chronological/cross-ticker real-Qwen tests; no learning or selection."""
import argparse
import gc
import json
from pathlib import Path

from propevolve.reasoning_policy.decisive_learning import (
    generalization_indices, decision_evidence, compare_learning,
)
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import (
    PreparedDataset, IndexedJsonRows, read_sft_config, verify_mlx_view,
)
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.supervised_trainer import evaluate_action_validation
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    root = Path.cwd()
    manifest = json.loads((Path(plan['dataset']) / 'manifest.json').read_text())
    source = IndexedJsonRows(Path(plan['dataset']) / 'valid.jsonl')
    metadata = [{'ticker': r['ticker'], 'target': r['messages'][-1]['content'],
        'source_id': r['source_id'], 'completed_at_ns': r['completed_at_ns'],
        'label_end_ns': r['label_end_ns']} for r in source]
    indices = generalization_indices(metadata, tickers=plan['tickers'], actions=plan['actions'],
        rows_per_group=plan['rows_per_group'], start_ns=manifest['splits']['valid'][0],
        end_ns=manifest['splits']['valid'][1], training_end_ns=manifest['splits']['train'][1],
        seed=plan['seed'])
    if manifest['splits']['valid'][1] > manifest['sealed_start_ns']:
        raise ValueError('generalization reserve crosses sealed boundary')
    configs = {}
    for name, path in plan['policies'].items():
        verify_mlx_view(path, plan['view'], root=root)
        configs[name] = read_sft_config(path, root=root)
    dataset = PreparedDataset(plan['view'], 'valid')
    rows = [dataset[i] for i in indices]
    destination = Path(plan['output'])
    destination.mkdir(parents=True, exist_ok=False)
    log = (destination / 'controller.log').open('x')
    def event(message):
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()
    import mlx.core as mx
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    report = {'plan': plan, 'indices': indices, 'rows': [metadata[i] for i in indices],
              'dataset_sha256': file_digest(Path(plan['dataset']) / 'manifest.json'),
              'plan_sha256': file_digest(args.config), 'results': {},
              'scope': '2024 development generalization; not final holdout', 'weights_updated': False}
    for name, path in plan['policies'].items():
        event(f'[generalization] loading={name} rows={len(rows)}')
        policy = MLXActionPolicy.from_config(path, root=root)
        policy.model.eval()
        scores = {}
        metrics = evaluate_action_validation(policy.model, rows, configs[name],
            on_scored=lambda i, s: scores.__setitem__(i, s))
        evidence = [decision_evidence(row, scores[i], ambiguity_r=plan['ambiguity_r'])
                    for i, row in enumerate(rows)]
        report['results'][name] = {'metrics': metrics, 'scores': [scores[i] for i in range(len(rows))],
                                  'evidence': evidence}
        event(f'[generalization] completed={name} ' + json.dumps(compare_learning(evidence, evidence)))
        atomic_json(destination / 'report.json', report)
        del policy
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    before = report['results']['parent']['evidence']
    after = report['results']['candidate']['evidence']
    report['comparison'] = compare_learning(before, after)
    report['per_ticker'] = {}
    for ticker in plan['tickers']:
        positions = [i for i, original in enumerate(indices) if metadata[original]['ticker'] == ticker]
        report['per_ticker'][ticker] = compare_learning([before[i] for i in positions],
                                                       [after[i] for i in positions])
    atomic_json(destination / 'report.json', report)
    event('[generalization] comparison=' + json.dumps(report['comparison']))
    log.close()


if __name__ == '__main__':
    main()

"""Matched teacher-free real-market prefix evaluation; no training or pass-rate claims."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.decisive_learning import simulate_prefix
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.job import load_role
from propevolve.reasoning_policy.policy import MLXActionPolicy
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    root = Path.cwd()
    source = json.loads(Path(plan['source']).read_text())
    manifest = json.loads(Path(plan['dataset_manifest']).read_text())
    begin, end = (int(np.datetime64(plan[key], 'ns').astype(np.int64))
                  for key in ('validation_start', 'validation_end'))
    allowed = manifest['splits']['valid']
    if not allowed[0] <= begin < end <= allowed[1] <= manifest['sealed_start_ns']:
        raise ValueError('prefix must remain inside the audited development role')
    source['temporal'].update(validation_start=plan['validation_start'], validation_end=plan['validation_end'])
    for path in plan['policies'].values():
        if not Path(path).is_file():
            raise ValueError('complete and save the learning diagnostic first')
    destination = Path(plan['output'])
    destination.mkdir(parents=True, exist_ok=False)
    report = {'plan': plan, 'plan_sha256': file_digest(args.config),
              'source_sha256': file_digest(plan['source']), 'results': {},
              'scope': 'bounded chronological development prefixes, not full challenge validation'}
    log = (destination / 'controller.log').open('x')
    def event(message):
        print(message, flush=True)
        log.write(message + '\n')
        log.flush()
    import mlx.core as mx
    mx.set_memory_limit(int(plan['memory_gb'] * 1024**3))
    mx.set_cache_limit(int(plan['cache_mb'] * 1024**2))
    tickers = sorted({row['ticker'] for row in plan['episodes']})
    environment, _ = load_role({'tickers': {'valid': tickers}, 'seed': plan['seed']},
        root, source, 'valid', include_specialists=False)
    context = ContextConfig.load(plan['context'])
    for name, path in plan['policies'].items():
        event('[simulator] loading=' + name)
        policy = MLXActionPolicy.from_config(path, root=root)
        policy.model.eval()
        receipts = []
        with (destination / (name + '-decisions.jsonl')).open('x') as stream:
            count = 0
            def record(value):
                nonlocal count
                stream.write(json.dumps(value, allow_nan=False) + '\n')
                stream.flush()
                count += 1
                if count % plan['progress_every'] == 0:
                    event(f'[simulator] policy={name} decisions={count}')
            for episode in plan['episodes']:
                market = environment.markets[episode['ticker']]
                start = int(np.searchsorted(market.timestamps, np.datetime64(episode['start_time'])))
                receipts.append(simulate_prefix(policy, environment,
                    options={'ticker': episode['ticker'], 'start': start},
                    context_config=context, max_steps=plan['max_steps'], on_decision=record))
        report['results'][name] = receipts
        atomic_json(destination / 'report.json', report)
        event('[simulator] ' + name + '=' + json.dumps(receipts))
        del policy
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    log.close()


if __name__ == '__main__':
    main()

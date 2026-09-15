"""Replay configured collected management rows through the unchanged simulator."""
import argparse
from collections import Counter
import json
from pathlib import Path

from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.job import read_job, load_source_contract, collection_source_for_role, load_role
from propevolve.reasoning_policy.label_reference import verify_management_execution
from propevolve.reasoning_policy.workflow import atomic_json


def audit(config_path):
    plan = json.loads(Path(config_path).read_text())
    config, root = read_job(plan['collection_config'])
    source, _, splits, _, _ = load_source_contract(config, root)
    dataset = root / config['dataset_output']
    manifest = json.loads((dataset/'manifest.json').read_text())
    reports = {}
    contract = config['opportunity_contract']
    if contract.get('management_label_mode') != 'simulator_continuation':
        raise ValueError('execution audit requires simulator continuation labels')
    for role in ('train', 'valid'):
        path = dataset / f'{role}.jsonl'
        if file_digest(path) != manifest['files'][role]:
            raise ValueError('audit input changed')
        rows = [json.loads(line) for line in path.open()]
        bounded, bounds = collection_source_for_role(config, source, role, splits)
        if bounds != manifest['splits'][role]:
            raise ValueError('audit temporal role changed')
        for ticker in config['tickers'][role]:
            selected = [(index,row) for index,row in enumerate(rows)
                        if row['ticker'] == ticker and row['messages'][-1]['content'] in ('HOLD','CLOSE')]
            if not selected:
                continue
            single = {**config, 'tickers': {**config['tickers'], role: [ticker]}}
            env, _ = load_role(single, root, bounded, role, include_specialists=False)
            issues = []
            for index, row in selected:
                reset = json.loads(row['source_id'].split(':', 1)[1])
                discrepancies = verify_management_execution(env, row, reset_options=reset,
                    horizon=contract['horizon'], minimum_improvement_r=contract['position_minimum_improvement_r'],
                    tolerance=plan['economic_tolerance_r'])
                if discrepancies:
                    issues.append(dict(index=index, discrepancies=discrepancies))
            reports[f'{role}/{ticker}'] = dict(rows=len(selected), issues=issues,
                actions=dict(Counter(row['messages'][-1]['content'] for _,row in selected)))
            print(f'[management-replay] role={role} ticker={ticker} rows={len(selected)} issues={len(issues)}', flush=True)
            del env
    result = dict(status='PASS' if all(not r['issues'] for r in reports.values()) else 'BLOCKED',
        dataset_manifest_sha256=file_digest(dataset/'manifest.json'),
        collection_config_sha256=file_digest(plan['collection_config']), reports=reports)
    atomic_json(Path(plan['output']), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    result = audit(parser.parse_args().config)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)

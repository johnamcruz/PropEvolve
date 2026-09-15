"""Read-only independent OHLC audit of the exact reasoning training corpus."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from propevolve.assets import AssetContract
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.label_reference import barrier_result, management_entry, label_audit_passed
from propevolve.reasoning_policy.workflow import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    dataset = Path(config['dataset'])
    source = json.loads(Path(config['source']).read_text())
    manifest = json.loads((dataset / 'manifest.json').read_text())
    economics = manifest['lineage']['economic_contract']
    risk = source['challenge']['per_trade_risk_dollars']
    assets = AssetContract.load(source['assets'])
    stop_at = max(bounds[1] for bounds in manifest['splits'].values())
    roles = {}
    entry_rows = defaultdict(list)
    for role, bounds in manifest['splits'].items():
        if file_digest(dataset / f'{role}.jsonl') != manifest['files'][role]:
            raise ValueError('dataset row checksum changed')
        roles[role] = [json.loads(line) for line in (dataset / f'{role}.jsonl').open()]
    # A selected cohort may omit its original flat-entry row. Resolve it
    # through the authenticated source corpus, not by guessing its direction.
    for selected_source in manifest['lineage'].get('selection', []):
        source_dataset = Path(selected_source['dataset'])
        source_manifest_path = source_dataset / 'manifest.json'
        if file_digest(source_manifest_path) != selected_source['manifest_sha256']:
            raise ValueError('selection source manifest changed')
        source_manifest = json.loads(source_manifest_path.read_text())
        role = selected_source['role']
        rows_path = source_dataset / f'{role}.jsonl'
        if file_digest(rows_path) != source_manifest['files'][role]:
            raise ValueError('selection source rows changed')
        with rows_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['messages'][-1]['content'] in ('ENTER_LONG_1', 'ENTER_SHORT_1'):
                    entry_rows[role].append(row)
    reports = {}
    for ticker in source['tickers']:
        # CSV timestamps are bar opens; model decisions are completed-bar closes.
        bars = {}
        with (Path(assets.market_data) / f'{ticker}_{source["timeframe_minutes"]}min.csv').open() as stream:
            for row in csv.DictReader(stream):
                ns = int(datetime.fromisoformat(row['datetime']).timestamp()) * 10**9
                ns += source['timeframe_minutes'] * 60 * 10**9
                if ns >= stop_at:
                    break
                bars[ns] = [float(np.float32(row[key])) for key in ('open', 'high', 'low', 'close')]
        cache_root = Path(manifest['embedding_storage']['cache_root']) / ticker
        times = np.load(cache_root / 'timestamps.npy', mmap_mode='r').astype('datetime64[ns]').astype(np.int64)
        times = times[times < stop_at]
        prices = np.asarray([bars[int(ns)] for ns in times])
        lookup = {int(ns): i for i, ns in enumerate(times)}
        pv, fee = source['point_values'][ticker], source['round_trip_fees'][ticker]
        teachers = []
        for spec in source['teachers']:
            base = Path(spec['cache_root']) / ticker
            meta = json.loads((base / 'manifest.json').read_text())
            for filename in ('probabilities', 'availability', 'timestamps'):
                if file_digest(base / f'{filename}.npy') != meta[f'{filename}_sha256']:
                    raise ValueError('teacher artifact checksum changed')
            if list(meta['channels']) != spec['channels']:
                raise ValueError('teacher channel order changed')
            teacher_times = np.load(base/'timestamps.npy', mmap_mode='r').astype('datetime64[ns]').astype(np.int64)
            teachers.append((spec, teacher_times,
                np.load(base/'probabilities.npy', mmap_mode='r'),
                np.load(base/'availability.npy', mmap_mode='r')))
        for role, rows in roles.items():
            selected = [(i, r) for i, r in enumerate(rows) if r['ticker'] == ticker]
            counts, issues, notes = Counter(), Counter(), Counter()
            examples = defaultdict(list)
            parents = {r['source_id']: r for _, r in selected
                       if r['messages'][-1]['content'] not in ('HOLD', 'CLOSE')}
            for row in entry_rows[role]:
                if row['ticker'] == ticker:
                    parents.setdefault(row['source_id'], row)
            def flag(name, index):
                issues[name] += 1
                if len(examples[name]) < config['max_examples_per_issue']:
                    examples[name].append(index)
            for index, row in selected:
                action = row['messages'][-1]['content']
                counts[action] += 1
                if not manifest['splits'][role][0] <= row['completed_at_ns'] <= row['label_end_ns'] < manifest['splits'][role][1]:
                    flag('temporal_row_bounds', index)
                prompt = row['messages'][-2]['content']
                if any(key in prompt for key in ('future_excursions', 'specialist_targets', 'reward_to_go', 'target_before_stop')):
                    flag('future_target_in_prompt', index)
                decision = lookup[row['completed_at_ns']]
                first, end = decision + 1, decision + 1 + economics['horizon']
                if end > len(times) or times[end-1] >= manifest['splits'][role][1]:
                    flag('insufficient_future_reserve', index)
                    continue
                window = prices[first:end]
                targets = row['targets']
                for spec, teacher_times, probabilities, availability in teachers:
                    ti = int(np.searchsorted(teacher_times, row['completed_at_ns']))
                    if ti >= len(teacher_times) or teacher_times[ti] != row['completed_at_ns'] or not availability[ti]:
                        flag('teacher_unavailable_' + spec['kind'], index)
                        continue
                    actual = [targets['specialist_targets'][spec['kind']+'.'+channel] for channel in spec['channels']]
                    if not np.array_equal(np.asarray(actual, np.float32), probabilities[ti]):
                        flag('teacher_alignment_' + spec['kind'], index)
                values = {name: outcome['reward_to_go'] for name, outcome in targets['outcomes'].items()}
                if values[action] != max(values.values()):
                    flag('target_not_economic_max', index)
                gap = values[action] - max(v for name, v in values.items() if name != action)
                if gap <= config['near_tie_r']:
                    notes['near_tie_' + action] += 1
                for level, expected in targets['target_before_stop_by_r'].items():
                    for side, sign in (('long', 1), ('short', -1)):
                        actual = barrier_result(window[0,0], window[:,1], window[:,2],
                            side=sign, risk=risk, point_value=pv, fee=fee,
                            target=float(level), stop=economics['stop_r'])
                        if actual != expected[side]:
                            flag('barrier_' + side + '_' + level, index)
                for side, sign in (('long', 1), ('short', -1)):
                    favorable = (window[:,1].max()-window[0,0] if sign == 1 else window[0,0]-window[:,2].min())
                    adverse = (window[0,0]-window[:,2].min() if sign == 1 else window[:,1].max()-window[0,0])
                    expected = {'mfe_r_gross': max(0., favorable)*pv/risk,
                                'mae_r_gross': max(0., adverse)*pv/risk,
                                'terminal_r_net': (sign*(window[-1,3]-window[0,0])*pv-fee)/risk}
                    for key, value in expected.items():
                        if abs(value-targets['future_excursions'][side][key]) > config['economic_tolerance_r']:
                            flag('excursion_' + side + '_' + key, index)
                if action in ('HOLD', 'CLOSE'):
                    parent = parents.get(row['source_id'])
                    try:
                        entry_i, sign = management_entry(row, parent, lookup)
                    except ValueError:
                        flag('invalid_entry_lineage', index)
                        continue
                    if parent is not None and parent['messages'][-1]['content'] in ('ENTER_LONG_1', 'ENTER_SHORT_1'):
                        notes['management_from_expost_winner'] += 1
                    entry = prices[entry_i,0]
                    close_r = (sign*(prices[first,0]-entry)*pv-fee)/risk
                    if abs(close_r-values['CLOSE']) > config['economic_tolerance_r']:
                        flag('close_execution', index)
                    # Any claimed HOLD exit must be reachable without touching
                    # the stop on an earlier bar. Opening gaps count before exit.
                    outcome = targets['outcomes']['HOLD']
                    exit_i = lookup[outcome['outcome_end_ns']]
                    if outcome['outcome'] == 'continued_to_better_exit':
                        for j in range(first, exit_i):
                            worst = prices[j,2] if sign == 1 else prices[j,1]
                            if (sign*(worst-entry)*pv-fee)/risk <= -economics['stop_r']:
                                flag('hold_exit_after_stop', index)
                                break
                        realized = (sign*(prices[exit_i,0]-entry)*pv-fee)/risk
                        if abs(realized - outcome['terminal_pnl']/risk) > config['economic_tolerance_r']:
                            flag('hold_execution', index)
                if row['market_embedding_reference']['row'] != decision:
                    flag('embedding_row_alignment', index)
            reports[f'{role}/{ticker}'] = {'counts': counts, 'issues': issues,
                                          'diagnostics': notes, 'examples': examples}
        print(f'[label-audit] ticker={ticker} completed', flush=True)
    passed = label_audit_passed(reports)
    report = {'status': 'PASS' if passed else 'BLOCKED',
              'scope': 'independent OHLC label audit; not proof of causal predictability',
              'dataset_manifest_sha256': file_digest(dataset/'manifest.json'),
              'sealed_2026_used': False, 'reports': reports}
    atomic_json(Path(config['output']), report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

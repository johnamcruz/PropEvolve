"""Add causal economic inputs to existing authenticated labels without relabeling."""
import copy
import json
from pathlib import Path

import numpy as np

from .inputs import trade_r_context


def enrich_record(record, market, *, context, risk_dollars, point_value, round_trip_fee):
    """Preserve row/label identity; append only completed-bar R context."""
    result = copy.deepcopy(record)
    prompt = json.loads(result['messages'][1]['content'])
    old_fields = prompt['fields']
    extra = list(context.fields[len(old_fields):])
    if (list(context.fields[:len(old_fields)]) != old_fields
            or set(extra) != {'trade.volatility_r', 'trade.cost_r', 'trade.volatility_available'}):
        raise ValueError('R enrichment must append only the configured economic fields')
    reference = result['market_embedding_reference']
    row = reference['row']
    if (reference['ticker'] != market.ticker or result['ticker'] != market.ticker
            or not 0 <= row < len(market.close)
            or int(market.timestamps[row].astype('datetime64[ns]').astype(np.int64))
            != result['completed_at_ns']):
        raise ValueError('R enrichment row is not aligned to its embedding')
    history = prompt['history_oldest_first']
    if not 0 < len(history) <= reference['available_count'] or row + 1 < len(history):
        raise ValueError('R enrichment history differs from embedding availability')
    for index, values in enumerate(history):
        fields = trade_r_context(market, row=row-len(history)+1+index,
            risk_dollars=risk_dollars, point_value=point_value, round_trip_fee=round_trip_fee,
            lookback=context.volatility_lookback)
        # Match RollingContext's float32 representation exactly.
        values.extend(float(np.float32(fields[name])) for name in extra)
    prompt['fields'] = list(context.fields)
    result['messages'][1]['content'] = json.dumps(prompt, separators=(',', ':'), allow_nan=False)
    return result


def enrich_dataset(config_path):
    """JSON-driven, append-only publication; never overwrite parent artifacts."""
    from ..cache import EmbeddingCache, load_market_series
    from .context import ContextConfig
    from .dataset import write_supervised_dataset, audit_supervised_dataset
    from .integrity import file_digest
    from .mlx_sft import verify_dataset
    plan = json.loads(Path(config_path).read_text())
    parent = Path(plan['dataset'])
    manifest = verify_dataset(parent, requirements=None)
    source = json.loads(Path(plan['source']).read_text())
    context = ContextConfig.load(plan['context'])
    storage = manifest['embedding_storage']
    if storage['kind'] != 'source_embedding_reference_v1':
        raise ValueError('R enrichment requires authenticated source embedding references')
    stop = max(bounds[1] for bounds in manifest['splits'].values())
    markets, caches = {}, {}
    for ticker in storage['sources']:
        cache = EmbeddingCache.load(Path(storage['cache_root']) / ticker)
        if file_digest(cache.root/'manifest.json') != storage['sources'][ticker]['manifest_sha256']:
            raise ValueError('R enrichment embedding identity drift')
        caches[ticker] = cache
        markets[ticker] = load_market_series(cache.manifest['source'], cache.root,
            ticker=ticker, end=str(np.datetime64(stop, 'ns')))
    def records():
        for role in ('train', 'valid'):
            count = 0
            with (parent/f'{role}.jsonl').open() as stream:
                for line in stream:
                    original = json.loads(line)
                    ticker = original['ticker']
                    record = enrich_record(original, markets[ticker], context=context,
                        risk_dollars=source['challenge']['per_trade_risk_dollars'],
                        point_value=source['point_values'][ticker],
                        round_trip_fee=source['round_trip_fees'][ticker])
                    reference = record['market_embedding_reference']
                    row, count_available = reference['row'], reference['available_count']
                    values = np.zeros((storage['context_steps'], storage['embedding_dim']), np.float32)
                    values[-count_available:] = caches[ticker].embeddings[row-count_available+1:row+1]
                    record['market_embeddings'] = values
                    record['market_available'] = np.arange(storage['context_steps']) >= storage['context_steps']-count_available
                    yield record
                    count += 1
            print(f'[trade-r-context] role={role} rows={count} labels=unchanged', flush=True)
    lineage = {**manifest['lineage'], 'context_config_sha256': file_digest(plan['context']),
        'r_context_augmentation': {'parent_manifest_sha256': file_digest(parent/'manifest.json'),
            'source_config_sha256': file_digest(plan['source']),
            'config_sha256': file_digest(config_path), 'lookback': context.volatility_lookback,
            'input_code_sha256': file_digest(Path(__file__).with_name('inputs.py')),
            'augmentation_code_sha256': file_digest(__file__)}}
    output = write_supervised_dataset(records(), plan['output'], splits=manifest['splits'],
        lineage=lineage, sealed_start_ns=manifest['sealed_start_ns'],
        embedding_storage=storage['kind'], embedding_source_cache_root=storage['cache_root'])
    audit = audit_supervised_dataset(plan['output'], specialist_score_mode=plan['specialist_score_mode'])
    return {'counts': output['counts'], 'audit': audit}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    print(json.dumps(enrich_dataset(parser.parse_args().config), indent=2))

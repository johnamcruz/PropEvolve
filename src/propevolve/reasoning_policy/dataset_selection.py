"""Publish explicit JSON-selected rows without changing labels or temporal roles."""
import json
from pathlib import Path
import numpy as np


def select_dataset(config_path):
    from ..cache import EmbeddingCache
    from .dataset import write_supervised_dataset, audit_supervised_dataset
    from .mlx_sft import verify_dataset
    from .integrity import file_digest
    plan = json.loads(Path(config_path).read_text())
    base = verify_dataset(plan['base_dataset'], requirements=None)
    storage = base['embedding_storage']
    if storage['kind'] != 'source_embedding_reference_v1':
        raise ValueError('selection requires source-referenced datasets')
    inputs, identities = [], []
    for spec in plan['sources']:
        path = Path(spec['dataset'])
        manifest = verify_dataset(path, requirements=None)
        if (manifest['splits'] != base['splits']
                or manifest['sealed_start_ns'] != base['sealed_start_ns']
                or manifest['embedding_storage'] != storage
                or manifest['lineage']['source_identity'] != base['lineage']['source_identity']):
            raise ValueError('selected datasets have incompatible lineage or temporal roles')
        role, indices = spec['role'], spec['indices']
        if (role not in ('train','valid') or not isinstance(indices,list)
                or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('invalid selected row indices')
        wanted = set(indices)
        found = {}
        with (path/f'{role}.jsonl').open() as stream:
            for index,line in enumerate(stream):
                if index in wanted:
                    found[index] = json.loads(line)
        if set(found) != wanted:
            raise ValueError('selected row index is outside dataset')
        inputs.extend(found[index] for index in indices)
        identities.append({**spec,'manifest_sha256':file_digest(path/'manifest.json')})
    caches = {}
    def records():
        for original in inputs:
            record = dict(original)
            ref = record['market_embedding_reference']
            ticker = ref['ticker']
            if ticker not in caches:
                cache = EmbeddingCache.load(Path(storage['cache_root'])/ticker)
                if file_digest(cache.root/'manifest.json') != storage['sources'][ticker]['manifest_sha256']:
                    raise ValueError('selection embedding identity drift')
                caches[ticker] = cache
            row,count = ref['row'],ref['available_count']
            values = np.zeros((storage['context_steps'],storage['embedding_dim']),np.float32)
            values[-count:] = caches[ticker].embeddings[row-count+1:row+1]
            record['market_embeddings'] = values
            record['market_available'] = np.arange(storage['context_steps']) >= storage['context_steps']-count
            yield record
    result = write_supervised_dataset(records(),plan['output'],splits=base['splits'],
        lineage={**base['lineage'],'selection':identities,'selection_config_sha256':file_digest(config_path)},
        sealed_start_ns=base['sealed_start_ns'],embedding_storage=storage['kind'],
        embedding_source_cache_root=storage['cache_root'])
    audit = audit_supervised_dataset(plan['output'],specialist_score_mode=plan['specialist_score_mode'])
    return {'counts':result['counts'],'audit':audit}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    print(json.dumps(select_dataset(parser.parse_args().config),indent=2))

"""Configured causal trade inputs through the shared collector/live boundary."""
import json
from dataclasses import replace

import numpy as np
import pytest

from propevolve.environment import HistoricalChallengeEnv
from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.inputs import observe_context
from test_reasoning_challenger_e2e import environment


def test_json_r_context_uses_completed_true_ranges_and_cost_not_account(tmp_path):
    path = tmp_path / "context.json"
    path.write_text(json.dumps({"context_steps": 2, "text_steps": 1,
        "input_mode": "embeddings", "volatility_lookback": 2,
        "fields": ["trade.volatility_r", "trade.cost_r", "trade.volatility_available"]}))
    config = ContextConfig.load(path)
    base = environment()
    market = base.markets["NQ"]
    market.open[:] = 100
    market.close[:] = 100
    market.high[:] = 103
    market.low[:] = 99
    # First measured bar gaps upward: TR=8; next bar TR=4; mean=6 points.
    market.high[1] = 108
    market.low[1] = 101
    market.open[1] = 102
    market.close[1] = 102
    env = HistoricalChallengeEnv(base.markets, tick_values={"NQ": 20.},
        round_trip_fees={"NQ": 6.}, spec=replace(base.spec, per_trade_risk_dollars=300.,
            ratchet_activation_r=10., ratchet_giveback_r=1.), seed=7)
    observation, _ = env.reset(options={"ticker": "NQ", "start": 2})
    history = RollingContext(config)
    observe_context(history, env, observation, ticker="NQ", row=2, sources=())
    np.testing.assert_allclose(history.snapshot().values[-1], [.4, .02, 1.])
    market.high[3:] = 100000
    market.low[3:] = 1
    repeated = RollingContext(config)
    observe_context(repeated, env, observation, ticker="NQ", row=2, sources=())
    np.testing.assert_array_equal(history.snapshot().values, repeated.snapshot().values)


def test_r_context_scales_economics_not_price_level_and_marks_warmup():
    from propevolve.reasoning_policy.inputs import trade_r_context
    market = environment().markets["NQ"]
    market.high[:] = 106
    market.low[:] = 100
    market.close[:] = 103
    arguments = dict(row=2, risk_dollars=300., point_value=20., round_trip_fee=6., lookback=2)
    original = trade_r_context(market, **arguments)
    assert original == {"trade.volatility_r": .4, "trade.cost_r": .02,
                        "trade.volatility_available": 1.}
    assert trade_r_context(market, **{**arguments, "row": 1}) == {
        "trade.volatility_r": 0., "trade.cost_r": .02, "trade.volatility_available": 0.}
    for name in ("high", "low", "close"):
        getattr(market, name)[:] *= 2
    assert trade_r_context(market, **arguments)["trade.volatility_r"] == .8
    assert trade_r_context(market, **{**arguments, "risk_dollars": 600.,
        "round_trip_fee": 12.}) == original
    for name in ("high", "low", "close"):
        getattr(market, name)[:] += 10000
    assert trade_r_context(market, **arguments)["trade.volatility_r"] == .8


@pytest.mark.parametrize("lookback", [None, 0, -1, True, 2.5])
def test_economic_fields_require_explicit_valid_json_lookback(lookback):
    with pytest.raises(ValueError, match="R context"):
        ContextConfig(20, ("trade.volatility_r", "trade.cost_r", "trade.volatility_available"),
                      "embeddings", 1, lookback)


@pytest.mark.parametrize("values", [dict(risk_dollars=None), dict(risk_dollars=0),
    dict(point_value=float("nan")), dict(round_trip_fee=-1), dict(round_trip_fee=300)])
def test_r_context_rejects_undefined_rather_than_fabricating_economics(values):
    from propevolve.reasoning_policy.inputs import trade_r_context
    arguments = dict(row=2, risk_dollars=300., point_value=20., round_trip_fee=6., lookback=2)
    with pytest.raises(ValueError, match="R context"):
        trade_r_context(environment().markets["NQ"], **{**arguments, **values})


@pytest.mark.parametrize('numeric_only', [False, True])
def test_existing_labeled_row_enrichment_matches_live_serialization(numeric_only):
    from propevolve.reasoning_policy.trade_r_dataset import enrich_record
    from propevolve.reasoning_policy.dataset import context_messages, embedding_payload
    from propevolve.decision import Action
    base = environment()
    env = HistoricalChallengeEnv(base.markets, tick_values={"NQ": 20.},
        round_trip_fees={"NQ": 6.}, spec=replace(base.spec, per_trade_risk_dollars=300.,
            ratchet_activation_r=10., ratchet_giveback_r=1.), seed=7)
    observation, _ = env.reset(options={"ticker": "NQ", "start": 2})
    old = ContextConfig(2, ("trade.open",), "embeddings", 1)
    new = ContextConfig(2, ("trade.open", "trade.volatility_r", "trade.cost_r",
        "trade.volatility_available"), "embeddings", 1, 2,
        text_fields=old.fields if numeric_only else None)
    actions = [Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1]
    histories = [RollingContext(c) for c in (old, new)]
    for history in histories:
        observe_context(history, env, observation, ticker="NQ", row=2, sources=())
    before = {"messages": context_messages(histories[0].snapshot(), actions)
        + [{"role": "assistant", "content": "WAIT"}], "targets": {"sentinel": "unchanged"},
        "completed_at_ns": int(env.markets['NQ'].timestamps[2].astype('datetime64[ns]').astype(np.int64)),
        "ticker": "NQ", "market_embedding_reference": {"ticker": "NQ", "row": 2, "available_count": 1}}
    result = enrich_record(before, env.markets['NQ'], context=new,
        risk_dollars=300., point_value=20., round_trip_fee=6.)
    assert result['messages'][:2] == context_messages(histories[1].snapshot(), actions)
    assert result['targets'] == before['targets']
    assert len(json.loads(before['messages'][1]['content'])['fields']) == 1
    payload = embedding_payload(histories[1].snapshot(), state_fields=new.fields)
    np.testing.assert_array_equal(payload['causal_state'],
        result['causal_state'] if numeric_only else
        json.loads(result['messages'][1]['content'])['history_oldest_first'][-1])


def test_trade_r_inputs_do_not_depend_on_challenge_target_or_mll():
    base = environment()
    config = ContextConfig(2, ('trade.volatility_r', 'trade.cost_r',
        'trade.volatility_available'), 'embeddings', 1, 2)
    snapshots = []
    for target, max_loss in ((6000., 3000.), (12000., 6000.)):
        env = HistoricalChallengeEnv(base.markets, tick_values={'NQ': 20.},
            round_trip_fees={'NQ': 6.}, spec=replace(base.spec,
                profit_target=target, max_loss=max_loss, per_trade_risk_dollars=300.,
                ratchet_activation_r=10., ratchet_giveback_r=1.), seed=7)
        observation, _ = env.reset(options={'ticker': 'NQ', 'start': 2})
        history = RollingContext(config)
        observe_context(history, env, observation, ticker='NQ', row=2, sources=())
        snapshots.append(history.snapshot().values)
    np.testing.assert_array_equal(*snapshots)


def test_continuous_r_inputs_do_not_change_the_frozen_parent_prompt():
    from propevolve.reasoning_policy.dataset import context_messages, embedding_payload
    from propevolve.decision import Action
    base = environment()
    env = HistoricalChallengeEnv(base.markets, tick_values={'NQ': 20.}, round_trip_fees={'NQ': 6.},
        spec=replace(base.spec, per_trade_risk_dollars=300., ratchet_activation_r=10., ratchet_giveback_r=1.), seed=7)
    old_fields = ('trade.open', 'trade.current_r')
    fields = (*old_fields, 'trade.volatility_r', 'trade.cost_r', 'trade.volatility_available')
    configs = [ContextConfig(2, old_fields, 'embeddings', 1),
        ContextConfig(2, fields, 'embeddings', 1, 2, text_fields=old_fields)]
    windows = []
    observation, _ = env.reset(options={'ticker': 'NQ', 'start': 2})
    for config in configs:
        history = RollingContext(config)
        observe_context(history, env, observation, ticker='NQ', row=2, sources=())
        windows.append(history.snapshot())
    actions = [Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1]
    assert context_messages(windows[0], actions) == context_messages(windows[1], actions)
    payload = embedding_payload(windows[1])
    assert payload['causal_state_fields'] == list(fields)
    assert payload['causal_state'][-3] > 0
    assert payload['causal_state'][-2] == pytest.approx(.02)


@pytest.mark.parametrize('mutation', ['future', 'nonfinite', 'visible_mismatch'])
def test_audit_rejects_corrupt_continuous_state_even_when_prompt_is_valid(tmp_path, mutation):
    from test_reasoning_dataset_audit_e2e import _dataset
    from propevolve.reasoning_policy.dataset import audit_supervised_dataset
    from propevolve.reasoning_policy.integrity import file_digest
    root = _dataset(tmp_path)
    rows = [json.loads(s) for s in (root/'train.jsonl').read_text().splitlines()]
    prompt = json.loads(rows[0]['messages'][1]['content'])
    fields = [*prompt['fields'], 'future_mfe' if mutation == 'future' else 'trade.volatility_r']
    values = [*prompt['history_oldest_first'][-1], float('nan') if mutation == 'nonfinite' else .5]
    if mutation == 'visible_mismatch':
        values[0] += 10.
    rows[0].update(causal_state_fields=fields, causal_state=values)
    (root/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['files']['train'] = file_digest(root/'train.jsonl')
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='continuous causal state'):
        audit_supervised_dataset(root, specialist_score_mode='out_of_fold')


@pytest.mark.parametrize('numeric_only', [False, True])
def test_json_enrichment_to_prepared_batches_preserves_labels_and_cached_embeddings(tmp_path, numeric_only):
    import pandas as pd
    from propevolve.cache import build_embedding_cache, EmbeddingCache
    from propevolve.decision import Action
    from propevolve.reasoning_policy.dataset import write_supervised_dataset, audit_supervised_dataset
    from propevolve.reasoning_policy.trade_r_dataset import enrich_dataset
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view, PreparedDataset
    from test_cache import MeanEncoder
    from test_reasoning_dataset_audit_e2e import _record
    from test_reasoning_token_parity_e2e import LiteralTokenizer, ACTION_VERBALIZERS
    times = pd.date_range('2024-01-01', periods=20, freq='3min', tz='UTC')
    source = tmp_path/'NQ.csv'
    pd.DataFrame({'datetime': times, 'open': 100., 'high': 103., 'low': 99.,
        'close': 100., 'volume': 100.}).to_csv(source, index=False)
    cache_path = build_embedding_cache(source=source, destination=tmp_path/'cache/NQ',
        ticker='NQ', encoder=MeanEncoder(), checkpoint_sha256='fixture', context_length=3,
        stride=1, chunk_windows=2, timeframe_minutes=3, research_end_exclusive='2026-01-01')
    cache = EmbeddingCache.load(cache_path)
    ns = cache.timestamps.astype('datetime64[ns]').astype(np.int64)
    records = []
    for i, action in zip((2, 3, 4, 12, 13, 14), [Action.ENTER_LONG_1, Action.ENTER_SHORT_1, Action.WAIT]*2):
        r = _record(int(ns[i]), best=action)
        prompt = json.loads(r['messages'][1]['content'])
        prompt['fields'] = ['trade.open', 'trade.current_r']
        prompt['history_oldest_first'] = [[0., 0.], [0., 0.]]
        r['messages'][1]['content'] = json.dumps(prompt, separators=(',', ':'))
        r['market_embeddings'] = cache.embeddings[i-1:i+1]
        records.append(r)
    parent = tmp_path/'parent'
    write_supervised_dataset(records, parent,
        splits={'train': [int(ns[0]), int(ns[10])], 'valid': [int(ns[10]), int(ns[-1])+100]},
        sealed_start_ns=int(ns[-1])+1000,
        lineage={'source_identity': 'fixture', 'specialist_identities': ['fixture'],
            'economic_contract': 'fixture', 'split_audit': {'status': 'PASS'}},
        embedding_storage='source_embedding_reference_v1', embedding_source_cache_root=cache_path.parent)
    audit_supervised_dataset(parent, specialist_score_mode='post_fit')
    from propevolve.reasoning_policy.dataset_selection import select_dataset
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'base_dataset':str(parent),
        'output':str(tmp_path/'selected'), 'specialist_score_mode':'post_fit',
        'sources':[{'dataset':str(parent),'role':role,'indices':[0,2]}
                   for role in ('train','valid')]}))
    assert select_dataset(selection)['counts'] == {'train':2,'valid':2}
    for role in ('train','valid'):
        original = [json.loads(line) for line in (parent/f'{role}.jsonl').read_text().splitlines()]
        selected = [json.loads(line) for line in (tmp_path/f'selected/{role}.jsonl').read_text().splitlines()]
        assert selected == [original[0],original[2]]
    economics = tmp_path/'economics.json'
    economics.write_text(json.dumps({'challenge': {'per_trade_risk_dollars': 300.},
        'point_values': {'NQ': 20.}, 'round_trip_fees': {'NQ': 6.}}))
    fields = ['trade.open', 'trade.current_r', 'trade.volatility_r', 'trade.cost_r', 'trade.volatility_available']
    context = tmp_path/'context.json'
    context.write_text(json.dumps({'context_steps': 2, 'fields': fields, 'input_mode': 'embeddings',
        'text_steps': 2, 'volatility_lookback': 2,
        **({'text_fields': fields[:2]} if numeric_only else {})}))
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'dataset': str(parent), 'source': str(economics), 'context': str(context),
        'output': str(tmp_path/'enriched'), 'specialist_score_mode': 'post_fit'}))
    assert enrich_dataset(plan)['audit']['status'] == 'PASS'
    for role in ('train', 'valid'):
        before = [json.loads(s) for s in (parent/f'{role}.jsonl').read_text().splitlines()]
        after = [json.loads(s) for s in (tmp_path/f'enriched/{role}.jsonl').read_text().splitlines()]
        assert [r['targets'] for r in before] == [r['targets'] for r in after]
        assert [r['market_embedding_reference'] for r in before] == [r['market_embedding_reference'] for r in after]
        if numeric_only:
            assert [r['messages'] for r in before] == [r['messages'] for r in after]
    recipe = tmp_path/'sft.json'
    recipe.write_text(json.dumps({'model': 'external-runtime', 'data': str(tmp_path/'enriched'),
        'adapter_path': str(tmp_path/'adapter'), 'num_layers': 1, 'batch_size': 1,
        'train': True, 'fine_tune_type': 'lora', 'mask_prompt': True, 'iters': 1, 'trust_remote_code': False,
        'learning_rate': 1e-5, 'grad_checkpoint': False, 'grad_accumulation_steps': 1,
        'lora_parameters': {'rank': 2, 'scale': 4., 'dropout': 0.},
        'max_seq_length': 4096, 'input_mode': 'embeddings', 'action_verbalizers': ACTION_VERBALIZERS,
        'projector': {'embedding_dim': 5, 'context_steps': 2, 'market_tokens': 2,
            'temporal_encoding': 'pooled_levels', 'state_fields': fields, 'state_scales': [1.]*5},
        'action_supervision': {'enabled': True, 'soft_target_weight': 1., 'ranking_weight': 1., 'margin': .25}}))
    prepare_mlx_view(recipe, tmp_path/'view', tokenizer=LiteralTokenizer())
    row = PreparedDataset(tmp_path/'view', 'train')[0]
    np.testing.assert_allclose(row['causal_state'], [0., 0., 4./15., .02, 1.])
    np.testing.assert_array_equal(row['market_embeddings'], cache.embeddings[1:3])

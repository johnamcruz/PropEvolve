"""The bounded real-data recipe cannot inherit the retired direct-action parent."""
from pathlib import Path

from propevolve.reasoning_policy.job import read_job
from propevolve.reasoning_policy.model_config import read_recipe, validate_model_settings


def test_real_staged_recipe_keeps_corrected_labels_and_frozen_temporal_roles():
    root = Path(__file__).resolve().parents[1]
    job, _ = read_job(root / 'config/diagnostics/staged_real_collection.json')
    assert job['dataset_temporal'] == dict(train_start='2021-01-01', train_end='2024-01-01',
        validation_start='2024-01-01', validation_end='2025-01-01')
    assert job['opportunity_contract']['management_label_mode'] == 'simulator_continuation'
    assert job['opportunity_contract']['target_rs'][0] == 2.
    assert job['opportunity_contract']['stop_r'] == 1.
    base = read_recipe(root / 'config/diagnostics/staged_real_base.json')
    validate_model_settings(base)
    assert base['architecture'] == 'staged_reasoning_v1'
    assert base['selection'] == 'hierarchical_greedy'
    assert base['adapter_path'] is None and base['resume_adapter_file'] is None
    assert not base['projector'].get('state_fields')
    assert len(base['staged_policy']['market']['channels']) == 15
    assert all(field.startswith('trade.') for field in base['staged_policy']['state_fields'])
    assert base['projector']['context_steps'] == 20
    import json
    source = json.loads((root / job['source_recipe']).read_text())
    expected = {teacher['kind'] + '.' + channel
                for teacher in source['teachers'] for channel in teacher['channels']}
    assert {channel['name'] for channel in base['staged_policy']['market']['channels']} == expected
    context = json.loads((root / job['context_config']).read_text())
    assert context['fields'] == base['staged_policy']['state_fields']
    assert context['context_steps'] == base['projector']['context_steps']

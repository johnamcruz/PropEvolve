"""Public job boundary: real files/simulator, external model runtime optional."""

import json

from propevolve.reasoning_policy.job import readiness


def test_readiness_reports_missing_sources_without_loading_a_model(tmp_path):
    config = tmp_path / "arbitrary-name.json"
    config.write_text(json.dumps({
        "workspace_root": str(tmp_path), "source_recipe": None,
        "temporal_split_audit": None, "context_config": "absent-context.json",
        "sft_config": "absent-model.json", "collection_policy": {
            "checkpoint": None, "sha256": None},
    }))
    result = readiness(config)
    assert result["ready"] is False
    assert {"source_recipe", "temporal_split_audit", "context_config", "sft_config",
            "collection_policy.checkpoint"}.issubset(result["blockers"])

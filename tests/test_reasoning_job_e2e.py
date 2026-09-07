"""Public job boundary: real files/simulator, external model runtime optional."""

import json

from propevolve.decision import Action
from propevolve.reasoning_policy.job import collection_factory, readiness


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


def test_reset_state_collection_needs_no_prior_policy_artifact(tmp_path):
    required = {}
    for name in ("source.json", "audit.json", "context.json", "sft.json", "rl.json",
                 "policy.json", "metrics.json"):
        (tmp_path / name).write_text("{}")
    config = tmp_path / "scratch.json"
    config.write_text(json.dumps({
        "workspace_root": str(tmp_path), "source_recipe": "source.json",
        "temporal_split_audit": "audit.json", "context_config": "context.json",
        "sft_config": "sft.json", "rl_config": "rl.json",
        "evaluation_policy_config": "policy.json",
        "evaluation_metrics_config": "metrics.json",
        "collection_policy": {"kind": "reset_states"},
        "episode_sampling": {"train": {"count": 1, "seed": 1},
                             "valid": {"count": 1, "seed": 2}},
        "evaluation_episode_sampling": {"count": 1, "seed": 3},
    }))
    result = readiness(config)
    assert "collection_policy.checkpoint" not in result["blockers"]
    decide = collection_factory(json.loads(config.read_text()), tmp_path)()
    assert decide(None, {"valid_actions": [Action.WAIT]}) is Action.WAIT

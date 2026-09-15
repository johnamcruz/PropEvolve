"""Config-to-runtime contract; no real model downloads or training."""

import json

import pytest

from propevolve.reasoning_policy.staged_inference import StagedReasoningPolicy
from propevolve.reasoning_policy.model_config import (
    validate_trade_mastery_parent,
    validate_trade_mastery_settings,
)


def test_arbitrary_recipe_selects_model_and_matching_adapter(tmp_path, monkeypatch):
    calls = []
    from test_staged_queries import settings
    from propevolve.reasoning_policy.integrity import file_digest
    def load(config):
        calls.append((config["model"], config["adapter_path"]))
        return object(), object()
    for name in ("example/backbone-a", "example/backbone-b"):
        adapter = tmp_path / name.rsplit("/", 1)[-1]
        adapter.mkdir()
        (adapter / "projector.safetensors").write_bytes(b"external backend fixture")
        config = {"model": name, "adapter_path": str(adapter), "max_seq_length": 1024,
            "architecture": "staged_reasoning_v1", "staged_policy": settings(),
            "projector": {"state_fields": []}, "selection": "hierarchical_greedy",
            "chat_template_kwargs": {}}
        (adapter / "adapter_config.json").write_text(json.dumps({**config,
            "weight_files": {"projector.safetensors": file_digest(adapter / "projector.safetensors")}}))
        recipe = tmp_path / "any-name.json"
        recipe.write_text(json.dumps(config))
        policy = StagedReasoningPolicy.from_config(recipe, backend_factory=load)
        assert policy.settings["max_seq_length"] == 1024
    assert calls == [("example/backbone-a", str(tmp_path / "backbone-a")),
                     ("example/backbone-b", str(tmp_path / "backbone-b"))]


@pytest.mark.parametrize("change", [{"model": ""}, {"max_seq_length": True},
                                   {"adapter_path": ""},
                                   {"chat_template_kwargs": {"tokenize": True}}])
def test_invalid_runtime_recipe_fails_before_loading_model(tmp_path, change):
    recipe = tmp_path / "arbitrary.json"
    recipe.write_text(json.dumps({"model": "example/base", "adapter_path": None,
                                  "max_seq_length": 1024, **change}))
    with pytest.raises(ValueError):
        StagedReasoningPolicy.from_config(recipe)


def test_wrong_base_adapter_rejected_before_optional_runtime_load(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"model": "example/old",
        "architecture": "staged_reasoning_v1"}))
    recipe = tmp_path / "replacement.json"
    recipe.write_text(json.dumps({"model": "example/new", "adapter_path": str(adapter),
                                  "max_seq_length": 1024, "staged_policy": {}}))
    with pytest.raises(ValueError, match="base model"):
        StagedReasoningPolicy.from_config(recipe)


def test_rl_parent_must_be_a_complete_teacher_free_trade_mastery_policy():
    actions = {name: 1 for name in (
        "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")}
    settings = {
        "adapter_path": "adapter",
        "input_mode": "embeddings",
        "action_supervision": {"enabled": True},
        "dataset_requirements": {
            "minimum_rows_per_action": {"train": actions, "valid": actions},
        },
    }
    assert validate_trade_mastery_settings(settings) is settings
    with pytest.raises(ValueError, match="trade-mastery"):
        validate_trade_mastery_settings({**settings, "action_supervision": {"enabled": False}})
    missing_close = {**settings, "dataset_requirements": {
        "minimum_rows_per_action": {
            "train": {key: value for key, value in actions.items() if key != "CLOSE"},
            "valid": actions,
        },
    }}
    with pytest.raises(ValueError, match="trade-mastery"):
        validate_trade_mastery_settings(missing_close)


def test_rl_parent_cannot_bypass_sft_with_a_base_model():
    actions = {name: 1 for name in (
        "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")}
    settings = {
        "adapter_path": None,
        "input_mode": "embeddings",
        "action_supervision": {"enabled": True},
        "dataset_requirements": {
            "minimum_rows_per_action": {"train": actions, "valid": actions},
        },
    }
    with pytest.raises(ValueError, match="trade-mastery"):
        validate_trade_mastery_settings(settings)


def test_rl_parent_is_bound_to_saved_five_action_sft_metadata(tmp_path):
    actions = {name: 1 for name in (
        "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")}
    settings = {
        "model": "example/model", "adapter_path": str(tmp_path),
        "input_mode": "embeddings", "projector": {"kind": "fixture"},
        "action_verbalizers": {name: name for name in actions},
        "action_supervision": {"enabled": True},
        "dataset_requirements": {
            "minimum_rows_per_action": {"train": actions, "valid": actions}},
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(settings))
    assert validate_trade_mastery_parent(settings) is settings

    metadata = {**settings, "action_supervision": {"enabled": False}}
    (tmp_path / "adapter_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="artifact"):
        validate_trade_mastery_parent(settings)

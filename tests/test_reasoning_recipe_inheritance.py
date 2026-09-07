"""Approved JSON -> model settings seam; no model imports or training."""
import json
import pytest
from propevolve.reasoning_policy.model_config import read_model_settings


def test_evaluation_inherits_training_model_and_template_without_filename_rules(tmp_path):
    parent = tmp_path / "parent-any-name.json"
    parent.write_text(json.dumps({"model": "local-base-A", "adapter_path": "sft-adapter",
        "max_seq_length": 800, "chat_template_kwargs": {"enable_thinking": False}}))
    child = tmp_path / "consumer.json"
    child.write_text(json.dumps({"inherits": parent.name, "adapter_path": "rl-adapter"}))
    selected = read_model_settings(child)
    assert selected["model"] == "local-base-A"
    assert selected["adapter_path"] == "rl-adapter"
    assert selected["max_seq_length"] == 800
    assert selected["chat_template_kwargs"] == {"enable_thinking": False}
    parent.write_text(json.dumps({"model": "local-base-B", "adapter_path": None, "max_seq_length": 400}))
    assert read_model_settings(child)["model"] == "local-base-B"


def test_cyclic_inheritance_is_rejected_before_model_loading(tmp_path):
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps({"inherits": path.name}))
    with pytest.raises(ValueError, match="cyclic"):
        read_model_settings(path)

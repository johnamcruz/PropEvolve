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


def test_explicit_workspace_resolves_nested_resources_independently_of_cwd(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    recipe = root / "arbitrary.json"
    recipe.write_text(json.dumps({"model": "publisher/base", "adapter_path": "adapters/one",
        "data": "dataset", "resume_adapter_file": "parent/adapters.safetensors",
        "max_seq_length": 800}))
    monkeypatch.chdir(tmp_path)
    settings = read_model_settings(recipe, root=root)
    assert settings["model"] == "publisher/base"
    assert settings["adapter_path"] == str(root / "adapters/one")
    assert settings["data"] == str(root / "dataset")
    assert settings["resume_adapter_file"] == str(root / "parent/adapters.safetensors")


def test_sft_rejects_unapplied_partial_accumulation_before_loading_model(tmp_path):
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    recipe = tmp_path / "training.json"
    values = {"model": "external/base", "adapter_path": "output", "data": "data",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "trust_remote_code": False, "num_layers": 1, "batch_size": 1,
        "iters": 3, "grad_accumulation_steps": 2, "learning_rate": 1e-5,
        "max_seq_length": 64, "grad_checkpoint": False, "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.}}
    recipe.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="complete gradient accumulation"):
        read_sft_config(recipe)
    values["iters"] = 4
    recipe.write_text(json.dumps(values))
    assert read_sft_config(recipe)["iters"] == 4


def test_sft_applies_json_declared_runtime_defaults_before_loading_model(tmp_path):
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    recipe = tmp_path / "training.json"
    recipe.write_text(json.dumps({"model": "external/base", "adapter_path": "output", "data": "data",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "trust_remote_code": False, "num_layers": 1, "batch_size": 1,
        "iters": 2, "grad_accumulation_steps": 1, "learning_rate": 1e-5,
        "max_seq_length": 64, "grad_checkpoint": False, "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.}}))
    selected = read_sft_config(recipe)
    assert selected["seed"] == 17
    assert selected["val_batches"] == 4
    assert selected["steps_per_report"] == 1
    assert selected["steps_per_eval"] == 5
    assert selected["save_every"] == 10

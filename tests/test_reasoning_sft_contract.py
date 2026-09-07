"""Challenger preparation regressions. Written but not run during volume training."""

import json
from dataclasses import replace

import pytest

from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import write_supervised_dataset
from propevolve.reasoning_policy.mlx_sft import read_sft_config, verify_dataset


def test_chronological_writer_rejects_label_crossing_train_boundary(tmp_path):
    record = {"source_id": "state", "completed_at_ns": 90, "label_end_ns": 110}
    with pytest.raises(ValueError, match="temporal role"):
        write_supervised_dataset([record], tmp_path / "dataset",
            splits={"train": [0, 100], "valid": [100, 200]}, sealed_start_ns=200, lineage={
                "source_identity": "fixture", "specialist_identities": "fixture",
                "economic_contract": "fixture", "split_audit": "fixture",
            })
    assert not (tmp_path / "dataset").exists()


def test_sealed_year_cannot_be_used_for_development_labels(tmp_path):
    with pytest.raises(ValueError, match="sealed"):
        write_supervised_dataset([], tmp_path / "dataset",
            splits={"train": [0, 100], "valid": [100, 201]},
            lineage={}, sealed_start_ns=200)


def test_native_training_config_rejects_full_finetuning(tmp_path):
    config = {
        "model": "fixture-model", "data": "fixture-data", "adapter_path": "new-adapter",
        "train": True, "fine_tune_type": "full", "mask_prompt": True,
        "num_layers": 1, "batch_size": 1, "iters": 1, "learning_rate": 1e-5,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1, "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False,
    }
    path = tmp_path / "any-name.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="LoRA"):
        read_sft_config(path)
    path.write_text(json.dumps({**config, "fine_tune_type": "lora"}))
    assert read_sft_config(path)["lora_parameters"]["rank"] == 2


def test_unreviewed_specialist_dataset_cannot_start_finetuning(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "propevolve_reasoning_dataset_v1"}))
    (tmp_path / "audit.json").write_text(json.dumps({"status": "BLOCKED"}))
    with pytest.raises(ValueError, match="audit"):
        verify_dataset(tmp_path)


def test_context_snapshot_cannot_be_mutated_by_dataset_consumer():
    context = RollingContext(ContextConfig(20, ("balance",)))
    context.append(1, {"balance": 0})
    with pytest.raises(ValueError):
        context.snapshot().values[-1, 0] = 1


def test_every_sft_routes_through_shared_guarded_trainer(tmp_path, monkeypatch):
    from propevolve.reasoning_policy import mlx_sft, supervised_trainer

    effective = tmp_path / "sft.json"
    effective.write_text(json.dumps({"action_supervision": {"enabled": False},
        "input_mode": "specialists"}))
    monkeypatch.setattr(mlx_sft, "verify_mlx_view",
                        lambda *args, **kwargs: effective)
    calls = []
    monkeypatch.setattr(supervised_trainer, "train_supervised",
                        lambda config, view: calls.append((config, view)) or "guarded")

    assert mlx_sft.train_prepared("recipe.json", "prepared-view") == "guarded"
    assert calls == [({"action_supervision": {"enabled": False},
                       "input_mode": "specialists"}, "prepared-view")]


def test_sft_learning_rate_schedule_is_config_driven(tmp_path):
    recipe = tmp_path / "schedule.json"
    payload = {
        "model": "fixture-model", "data": "fixture-data", "adapter_path": "new-adapter",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "num_layers": 1, "batch_size": 1, "iters": 3, "learning_rate": 1e-5,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False,
    }
    payload["lr_schedule"] = {"kind": "cosine_decay", "end": 1e-6,
                              "decay_updates": 32}
    recipe.write_text(json.dumps(payload))
    assert read_sft_config(recipe)["lr_schedule"]["decay_updates"] == 32
    payload["lr_schedule"]["end"] = payload["learning_rate"] * 2
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="learning-rate schedule"):
        read_sft_config(recipe)


def test_sft_trainable_components_are_config_driven_and_fail_closed(tmp_path):
    recipe = tmp_path / "components.json"
    payload = {
        "model": "fixture-model", "data": "fixture-data", "adapter_path": "new-adapter",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "num_layers": 1, "batch_size": 1, "iters": 1, "learning_rate": 1e-5,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False,
        "input_mode": "embeddings",
        "projector": {"embedding_dim": 2, "context_steps": 3, "market_tokens": 2,
                      "temporal_encoding": "pooled_levels"},
        "trainable_components": ["projector"],
    }
    recipe.write_text(json.dumps(payload))
    assert read_sft_config(recipe)["trainable_components"] == ["projector"]
    payload["trainable_components"] = ["projector", "unknown"]
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trainable components"):
        read_sft_config(recipe)

"""Challenger preparation regressions. Written but not run during volume training."""

import json
from dataclasses import replace
from pathlib import Path

import pytest


def test_epoch_budget_scales_to_actual_corpus_and_finishes_optimizer_updates():
    from propevolve.reasoning_policy.supervised_trainer import resolve_training_budget
    config = {"epochs": 20, "batch_size": 4, "grad_accumulation_steps": 2,
              "iters": 10, "val_batches": 2, "validation_batch_size": 4}
    result = resolve_training_budget(config, train_rows=101, valid_rows=9)
    assert result["iters"] == 520
    assert config["iters"] == 10
    assert resolve_training_budget({**config, "epochs": None},
                                   train_rows=101, valid_rows=9)["iters"] == 10

from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import context_messages, write_supervised_dataset
from propevolve.decision import Action
from propevolve.reasoning_policy.mlx_sft import (
    read_sft_config,
    verify_dataset,
    verify_mlx_view,
    view_contract,
)


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


def test_market_distillation_requires_every_declared_teacher_group(tmp_path):
    root = tmp_path / "market"
    root.mkdir()
    from propevolve.reasoning_policy.integrity import file_digest
    record = {
        "targets": {"specialist_targets": {
            "expansion.long_probability": 0.8,
            "regime.chop_probability": 0.1,
        }}
    }
    for role in ("train", "valid"):
        (root / f"{role}.jsonl").write_text(json.dumps(record) + "\n")
    manifest = {
        "schema": "propevolve_reasoning_dataset_v1",
        "splits": {"train": [0, 100], "valid": [100, 200]},
        "sealed_start_ns": 200,
        "files": {role: file_digest(root / f"{role}.jsonl")
                  for role in ("train", "valid")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "audit.json").write_text(json.dumps({
        "status": "PASS", "manifest_sha256": file_digest(root / "manifest.json"),
        "specialist_score_mode": "out_of_fold", "sealed_touched": False,
    }))

    with pytest.raises(ValueError, match="trend"):
        verify_dataset(root, required_target_groups=["expansion", "trend", "regime"])

    record["targets"]["specialist_targets"]["trend.long_probability"] = 0.7
    for role in ("train", "valid"):
        (root / f"{role}.jsonl").write_text(json.dumps(record) + "\n")
    manifest["files"] = {role: file_digest(root / f"{role}.jsonl")
                         for role in ("train", "valid")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    audit = json.loads((root / "audit.json").read_text())
    audit["manifest_sha256"] = file_digest(root / "manifest.json")
    (root / "audit.json").write_text(json.dumps(audit))
    assert verify_dataset(
        root, required_target_groups=["expansion", "trend", "regime"]
    )["schema"] == "propevolve_reasoning_dataset_v1"

    with pytest.raises(ValueError, match="volume"):
        verify_dataset(
            root,
            required_target_groups=["expansion", "trend", "regime", "volume"],
        )

    record["targets"]["specialist_targets"]["volume.long_probability"] = 0.6
    for role in ("train", "valid"):
        (root / f"{role}.jsonl").write_text(json.dumps(record) + "\n")
    manifest["files"] = {role: file_digest(root / f"{role}.jsonl")
                         for role in ("train", "valid")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    audit["manifest_sha256"] = file_digest(root / "manifest.json")
    (root / "audit.json").write_text(json.dumps(audit))
    assert verify_dataset(
        root,
        required_target_groups=["expansion", "trend", "regime", "volume"],
    )["schema"] == "propevolve_reasoning_dataset_v1"


def test_action_sft_rejects_parent_without_declared_market_distillation():
    from propevolve.reasoning_policy.model_config import validate_sft_parent_contract
    child = {
        "resume_adapter_requirements": {
            "stage_role": "market_distillation",
            "distillation_targets": ["expansion", "trend", "regime", "volume"],
        }
    }
    smoke = {"stage_role": "smoke", "distillation_targets": ["expansion"]}
    with pytest.raises(ValueError, match="stage_role"):
        validate_sft_parent_contract(child, smoke)
    incomplete = {
        "stage_role": "market_distillation",
        "distillation_targets": ["expansion", "regime"],
    }
    with pytest.raises(ValueError, match="distillation_targets"):
        validate_sft_parent_contract(child, incomplete)
    assert validate_sft_parent_contract(child, {
        "stage_role": "market_distillation",
        "distillation_targets": ["regime", "volume", "expansion", "trend"],
    }) is None


def test_mastery_dataset_contract_rejects_tiny_golden_corpus(tmp_path):
    root = tmp_path / "golden"
    root.mkdir()
    from propevolve.reasoning_policy.integrity import file_digest
    for role in ("train", "valid"):
        (root / f"{role}.jsonl").write_text("{}\n{}\n{}\n")
    manifest = {
        "schema": "propevolve_reasoning_dataset_v1",
        "splits": {"train": [0, 100], "valid": [100, 200]},
        "counts": {"train": 3, "valid": 3},
        "sealed_start_ns": 200,
        "files": {role: file_digest(root / f"{role}.jsonl")
                  for role in ("train", "valid")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "audit.json").write_text(json.dumps({
        "status": "PASS", "manifest_sha256": file_digest(root / "manifest.json"),
        "specialist_score_mode": "out_of_fold", "sealed_touched": False,
        "actions_by_role": {
            role: {"WAIT": 1, "ENTER_LONG_1": 1, "ENTER_SHORT_1": 1}
            for role in ("train", "valid")
        },
    }))

    with pytest.raises(ValueError, match="minimum rows per action"):
        verify_dataset(root, requirements={
            "minimum_rows_per_action": {"train": 2, "valid": 2},
            "expected_splits": {"train": [0, 100], "valid": [100, 200]},
            "sealed_start_ns": 200,
        })


def test_trade_mastery_dataset_contract_requires_every_declared_action(tmp_path):
    root = tmp_path / "trade-mastery"
    root.mkdir()
    from propevolve.reasoning_policy.integrity import file_digest
    for role in ("train", "valid"):
        (root / f"{role}.jsonl").write_text("{}\n")
    manifest = {
        "schema": "propevolve_reasoning_dataset_v1",
        "splits": {"train": [0, 100], "valid": [100, 200]},
        "counts": {"train": 1, "valid": 1},
        "sealed_start_ns": 200,
        "files": {role: file_digest(root / f"{role}.jsonl")
                  for role in ("train", "valid")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "audit.json").write_text(json.dumps({
        "status": "PASS", "manifest_sha256": file_digest(root / "manifest.json"),
        "specialist_score_mode": "out_of_fold", "sealed_touched": False,
        "actions_by_role": {
            role: {"WAIT": 10, "ENTER_LONG_1": 10, "ENTER_SHORT_1": 10}
            for role in ("train", "valid")
        },
    }))
    requirements = {
        "minimum_rows_per_action": {
            role: {
                "WAIT": 1, "ENTER_LONG_1": 1, "ENTER_SHORT_1": 1,
                "HOLD": 1, "CLOSE": 1,
            } for role in ("train", "valid")
        },
        "expected_splits": {"train": [0, 100], "valid": [100, 200]},
        "sealed_start_ns": 200,
    }

    with pytest.raises(ValueError, match="minimum rows per action"):
        verify_dataset(root, requirements=requirements)


def test_context_snapshot_cannot_be_mutated_by_dataset_consumer():
    context = RollingContext(ContextConfig(20, ("balance",)))
    context.append(1, {"balance": 0})
    with pytest.raises(ValueError):
        context.snapshot().values[-1, 0] = 1


def test_sft_prompt_teaches_trade_mastery_without_challenge_objectives():
    context = RollingContext(ContextConfig(2, ("balance",)))
    context.append(1, {"balance": 0})
    prompt = " ".join(message["content"] for message in context_messages(
        context.snapshot(), (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)))

    assert "trade quality" in prompt.lower()
    for forbidden in ("pass the challenge", "profit target", "mll", "blow", "timeout"):
        assert forbidden not in prompt.lower()


def test_every_sft_routes_through_shared_guarded_trainer(tmp_path, monkeypatch):
    from propevolve.reasoning_policy import mlx_sft, supervised_trainer

    effective = tmp_path / "sft.json"
    effective.write_text(json.dumps({"action_supervision": {"enabled": False},
        "input_mode": "specialists"}))
    monkeypatch.setattr(mlx_sft, "verify_mlx_view",
                        lambda *args, **kwargs: effective)
    monkeypatch.setattr(mlx_sft, "read_sft_config", lambda *args, **kwargs: {
        "action_supervision": {"enabled": False}, "input_mode": "specialists"})
    calls = []
    monkeypatch.setattr(supervised_trainer, "train_supervised",
                        lambda config, view: calls.append((config, view)) or "guarded")

    assert mlx_sft.train_prepared("recipe.json", "prepared-view") == "guarded"
    assert calls == [({"action_supervision": {"enabled": False},
                       "input_mode": "specialists",
                       "data": str(Path("prepared-view").resolve())},
                      "prepared-view")]


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


def test_sft_component_learning_rates_are_config_driven_and_fail_closed(tmp_path):
    recipe = tmp_path / "component-rates.json"
    payload = {
        "model": "fixture-model", "data": "fixture-data", "adapter_path": "new-adapter",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "num_layers": 1, "batch_size": 1, "iters": 1, "learning_rate": 3e-6,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False, "input_mode": "embeddings",
        "projector": {"embedding_dim": 2, "context_steps": 3,
                      "market_tokens": 2, "temporal_encoding": "pooled_levels"},
        "trainable_components": ["lora", "projector"],
        "component_learning_rates": {"lora": 1e-6, "projector": 3e-5},
    }
    recipe.write_text(json.dumps(payload))
    assert read_sft_config(recipe)["component_learning_rates"] == {
        "lora": 1e-6, "projector": 3e-5,
    }

    for invalid in (
            {"lora": 1e-6},
            {"lora": 1e-6, "projector": 3e-5, "other": 1e-5},
            {"lora": 0., "projector": 3e-5},
            {"lora": 1e-6, "projector": float("nan")}):
        recipe.write_text(json.dumps({**payload, "component_learning_rates": invalid}))
        with pytest.raises(ValueError, match="component learning rates"):
            read_sft_config(recipe)

    recipe.write_text(json.dumps({**payload, "trainable_components": ["projector"]}))
    with pytest.raises(ValueError, match="component learning rates"):
        read_sft_config(recipe)

def test_sft_causal_state_projector_is_config_driven_and_fail_closed(tmp_path):
    recipe = tmp_path / "causal-state.json"
    payload = {
        "model": "fixture-model", "data": "fixture-data", "adapter_path": "new-adapter",
        "train": True, "fine_tune_type": "lora", "mask_prompt": True,
        "num_layers": 1, "batch_size": 1, "iters": 1, "learning_rate": 3e-6,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False, "input_mode": "embeddings",
        "projector": {"embedding_dim": 2, "context_steps": 3,
                      "market_tokens": 2, "temporal_encoding": "pooled_levels",
                      "state_fields": ["trade.current_r", "trade.hold_bars"],
                      "state_scales": [4.0, 150.0]},
        "trainable_components": ["lora", "projector"],
        "component_learning_rates": {"lora": 1e-6, "projector": 3e-5},
    }
    recipe.write_text(json.dumps(payload))
    assert read_sft_config(recipe)["projector"]["state_fields"] == [
        "trade.current_r", "trade.hold_bars"]

    for invalid in ([1.0], [4.0, 0.0], [4.0, "bad"]):
        broken = json.loads(json.dumps(payload))
        broken["projector"]["state_scales"] = invalid
        recipe.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="causal state contract"):
            read_sft_config(recipe)

    recipe.write_text(json.dumps({**payload,
        "lr_schedule": {"kind": "cosine_decay", "end": 1e-6,
                        "decay_updates": 32}}))
    with pytest.raises(ValueError, match="component learning rates"):
        read_sft_config(recipe)


def test_production_trade_mastery_projector_uses_only_causal_trade_state():
    recipe = (Path(__file__).resolve().parents[1]
              / "config/reasoning/action_mastery_hierarchical_sft.json")
    state_fields = read_sft_config(recipe)["projector"]["state_fields"]
    assert state_fields == [
        "trade.open", "trade.position_side", "trade.risk_available",
        "trade.mfe_r_so_far", "trade.mae_r_so_far", "trade.current_r",
        "trade.giveback_r", "trade.hold_bars",
    ]
    assert not any(field.startswith(("account.", "challenge."))
                   for field in state_fields)


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


def test_prepared_view_can_be_reused_across_learning_hyperparameters(tmp_path):
    view = tmp_path / "view"
    view.mkdir()
    data = tmp_path / "dataset"
    data.mkdir()
    base = {
        "model": "fixture-model", "data": str(data),
        "adapter_path": str(tmp_path / "adapter"), "train": True,
        "fine_tune_type": "lora", "mask_prompt": True, "num_layers": 1,
        "batch_size": 1, "iters": 3, "learning_rate": 1e-5,
        "max_seq_length": 1024, "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False, "input_mode": "embeddings",
        "projector": {"embedding_dim": 2, "context_steps": 3,
                      "market_tokens": 2, "temporal_encoding": "pooled_levels"},
        "trainable_components": ["lora", "projector"],
        "action_supervision": {"enabled": True, "soft_target_weight": 1.,
                               "ranking_weight": 1., "margin": .25},
    }
    first = tmp_path / "first.json"
    first.write_text(json.dumps(base))
    changed = {**base, "learning_rate": 3e-5,
               "action_supervision": {**base["action_supervision"],
                                      "ranking_weight": 4.}}
    second = tmp_path / "second.json"
    second.write_text(json.dumps(changed))
    manifest = {"schema": "propevolve_reasoning_dataset_v1"}
    for role in ("train", "valid"):
        (view / f"{role}.jsonl").write_text(role)
    from propevolve.reasoning_policy.integrity import file_digest
    (view / "view_manifest.json").write_text(json.dumps({
        "view_contract": view_contract(read_sft_config(first)),
        "source_manifest": manifest,
        "files": {role: file_digest(view / f"{role}.jsonl")
                  for role in ("train", "valid")},
    }))
    from propevolve.reasoning_policy import mlx_sft
    original = mlx_sft.verify_dataset
    mlx_sft.verify_dataset = lambda path, **kwargs: manifest
    try:
        assert verify_mlx_view(second, view) == view / "sft.json"
        incompatible = {**changed, "max_seq_length": 2048}
        third = tmp_path / "third.json"
        third.write_text(json.dumps(incompatible))
        with pytest.raises(ValueError, match="prepared view contract"):
            verify_mlx_view(third, view)
    finally:
        mlx_sft.verify_dataset = original


def test_read_only_prepare_may_use_an_existing_frozen_adapter(monkeypatch, tmp_path):
    from propevolve.reasoning_policy import mlx_sft

    adapter = tmp_path / "frozen-adapter"
    adapter.mkdir()
    view = tmp_path / "view"
    view.mkdir()
    config = {"data": str(tmp_path / "data"), "adapter_path": str(adapter),
              "dataset_requirements": None, "distillation_targets": None}
    monkeypatch.setattr(mlx_sft, "read_sft_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(mlx_sft, "verify_dataset", lambda *args, **kwargs: {})
    monkeypatch.setattr(mlx_sft, "verify_mlx_view", lambda *args, **kwargs: view / "sft.json")

    assert mlx_sft.main(["--config", "recipe.json", "--view", str(view)]) == 0
    with pytest.raises(FileExistsError, match="adapter output exists"):
        mlx_sft.main(["--config", "recipe.json", "--view", str(view), "--train"])

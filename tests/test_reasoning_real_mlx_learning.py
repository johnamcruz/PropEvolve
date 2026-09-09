"""Explicit opt-in REAL model learning/reload test; never downloads by default.

Requires an approved local quantized model and audited real action dataset. No
synthetic backbone is substituted. This test is intentionally expensive and was
not run during implementation. Set PROPEVOLVE_REASONING_SFT_TEST_CONFIG only
when model loading/training are authorized. The fixture must contain all actions.
"""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("PROPEVOLVE_REASONING_SFT_TEST_CONFIG"),
                    reason="real MLX model training requires explicit opt-in")
def test_real_qlora_learns_all_action_classes_and_reload_preserves_scores(tmp_path):
    import copy
    import numpy as np
    from propevolve.cache import EmbeddingCache
    from propevolve.reasoning_policy.dataset import (
        audit_supervised_dataset, write_supervised_dataset,
    )
    from propevolve.reasoning_policy.mlx_sft import EncodedDataset, read_sft_config, verify_dataset
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from propevolve.reasoning_policy.learning_audit import score_labeled_examples
    from propevolve.decision import Action

    path = Path(os.environ["PROPEVOLVE_REASONING_SFT_TEST_CONFIG"])
    config = read_sft_config(path)
    assert (Path(config["model"]).is_dir()
            or os.environ.get("HF_HUB_OFFLINE") == "1"), (
                "use an approved local model or force offline cached-model loading")
    verify_dataset(config["data"])
    source_root = Path(config["data"])
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    storage = source_manifest["embedding_storage"]
    caches = {ticker: EmbeddingCache.load(Path(storage["cache_root"]) / ticker)
              for ticker in storage["sources"]}
    cohorts = {}
    for role in ("train", "valid"):
        cohort = {}
        with (source_root / f"{role}.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                cohort.setdefault(record["messages"][-1]["content"], record)
                if set(cohort) == {action.name for action in Action}:
                    break
        assert set(cohort) == {action.name for action in Action}, (
            "real fixture must cover all five actions")
        cohorts[role] = cohort

    compact_records = []
    for role in ("train", "valid"):
        for record in cohorts[role].values():
            record = copy.deepcopy(record)
            reference = record.pop("market_embedding_reference")
            cache = caches[reference["ticker"]]
            row, count = reference["row"], reference["available_count"]
            values = np.asarray(cache.embeddings[row - count + 1:row + 1], np.float32)
            pad = storage["context_steps"] - count
            record["market_embeddings"] = np.pad(values, ((pad, 0), (0, 0))).tolist()
            record["market_available"] = ([False] * pad) + ([True] * count)
            compact_records.append(record)
    data = tmp_path / "data"
    write_supervised_dataset(
        compact_records, data, splits=source_manifest["splits"],
        lineage={
            "source_identity": "real-five-action-e2e",
            "specialist_identities": source_manifest["lineage"]["specialist_identities"],
            "economic_contract": source_manifest["lineage"]["economic_contract"],
            "supervision_scope": "trade_mastery",
            "split_audit": source_manifest["lineage"]["split_audit"],
        }, sealed_start_ns=source_manifest["sealed_start_ns"],
    )
    audit_supervised_dataset(data, specialist_score_mode="post_fit")
    records = compact_records[:len(Action)]
    unseen_records = compact_records[len(Action):]
    parent_adapter = Path(config["resume_adapter_file"]).parent
    policy = MLXActionPolicy.load(config["model"], adapter_path=parent_adapter,
                                 max_seq_length=config["max_seq_length"],
                                 chat_template_kwargs=config["chat_template_kwargs"],
                                 input_mode=config["input_mode"],
                                 projector=config["projector"],
                                 action_verbalizers=config["action_verbalizers"])
    before = score_labeled_examples(policy, records)
    unseen_before = score_labeled_examples(policy, unseen_records)
    del policy
    gc.collect()
    import mlx.core as mx
    mx.synchronize()
    mx.clear_cache()
    minimums = {action.name: 1 for action in Action}
    effective = {
        **config,
        "data": str(data),
        "adapter_path": str(tmp_path / "adapter"),
        "iters": 50,
        "val_batches": 5,
        "steps_per_eval": 50,
        "save_every": 50,
        "grad_accumulation_steps": 5,
        # The smoke has one row per action and verifies wiring, not selected
        # generalization hyperparameters. A stronger bounded rate keeps this
        # explicit opt-in test finite; production rates remain config-owned.
        "component_learning_rates": {"lora": 1e-5, "projector": 1e-5},
        "validation_metrics_path": None,
        "dataset_requirements": {
            "minimum_rows_per_action": {"train": minimums, "valid": minimums},
            "expected_splits": source_manifest["splits"],
            "sealed_start_ns": source_manifest["sealed_start_ns"],
        },
    }
    recipe = tmp_path / "sft.json"
    recipe.write_text(json.dumps(effective))
    subprocess.run([sys.executable, "-m", "propevolve.reasoning_policy.mlx_sft",
                    "--config", str(recipe), "--view", str(tmp_path / "view"), "--train"], check=True)
    policy = MLXActionPolicy.from_config(recipe)
    encoded = EncodedDataset(tmp_path / "view" / "train.jsonl")
    with (data / "train.jsonl").open() as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            messages = record["messages"]
            tokens, offset = encoded.process(encoded[index])
            # Real model tokenizer: training and inference must score exactly
            # the same answer tokens, with exactly the same causal prefix.
            market_context = {key: record[key]
                              for key in ("market_embeddings", "market_available")}
            inference = policy.tokenize_completions(
                messages[:-1], [messages[-1]["content"]],
                market_context=market_context)
            assert inference[0][:2] == (tuple(tokens), offset)
    after = score_labeled_examples(policy, records)
    unseen_after = score_labeled_examples(policy, unseen_records)
    del policy
    gc.collect()
    mx.clear_cache()
    reloaded = MLXActionPolicy.from_config(recipe)
    again = score_labeled_examples(reloaded, records)
    for old, new, repeat in zip(before, after, again):
        assert new["target_log_likelihood"] > old["target_log_likelihood"], new["target"]
        assert new["correct"], new
        assert repeat["target_log_likelihood"] == pytest.approx(new["target_log_likelihood"], abs=1e-4)
    # This tiny smoke proves that each action's update reaches unseen temporal
    # rows. Positive unseen action margins remain the full selection-SFT gate;
    # five examples are intentionally insufficient to claim mastery.
    for old, new in zip(unseen_before, unseen_after):
        assert new["target"] == old["target"]
        assert new["target_log_likelihood"] > old["target_log_likelihood"], new["target"]

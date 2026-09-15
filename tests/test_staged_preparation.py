"""Audited row preparation uses the same target-free queries as live inference."""
import copy
import json
import numpy as np
import pytest

from test_staged_queries import settings, Tokenizer


def test_preparation_separates_teacher_and_economic_answers_from_model_inputs():
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    from propevolve.reasoning_policy.staged_preparation import encode_staged_record
    config = {"staged_policy": settings(), "max_seq_length": 1024,
        "chat_template_kwargs": {}, "projector": {"market_tokens": 2,
            "embedding_dim": 2, "context_steps": 4, "temporal_encoding": "pooled_levels"}}
    prompt = {"fields": ["trade.unrealized_r"], "history_oldest_first": [[0.]],
              "legal_actions": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]}
    record = {"messages": [{"role": "system", "content": "Trade"},
        {"role": "user", "content": json.dumps(prompt)},
        {"role": "assistant", "content": "ENTER_LONG_1"}],
        "targets": {"action_order": prompt["legal_actions"],
            "action_probabilities": [0., 1., 0.],
            "outcomes": {"WAIT": {"reward_to_go": 0.},
                         "ENTER_LONG_1": {"reward_to_go": 2.},
                         "ENTER_SHORT_1": {"reward_to_go": -1.}},
            "specialist_targets": {"expansion.long": .9, "expansion.short": .1}}}
    first = encode_staged_record(record, config, Tokenizer())
    changed = copy.deepcopy(record)
    changed["targets"]["specialist_targets"]["expansion.long"] = .2
    changed["targets"]["outcomes"]["ENTER_LONG_1"]["reward_to_go"] = 4.
    second = encode_staged_record(changed, config, Tokenizer())
    expected = prepare_staged_queries({"trade.unrealized_r": 0.}, settings(), Tokenizer(),
        max_seq_length=1022, chat_template_kwargs={})
    assert first["staged_queries"] == second["staged_queries"] == expected
    assert first["teacher_probabilities"] == [.9, .1]
    assert second["teacher_probabilities"] == [.2, .1]
    assert first["action_targets"]["values"] == [0., 2., -1.]
    assert "alternatives" not in first


def test_audited_dataset_reaches_staged_native_batches_without_action_completions(tmp_path):
    pytest.importorskip("mlx.core")
    from test_reasoning_prepared_full_action_e2e import prepared_action_view
    from propevolve.reasoning_policy.dataset import write_supervised_dataset
    from propevolve.reasoning_policy.integrity import file_digest
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view, PreparedDataset
    from propevolve.reasoning_policy.supervised_trainer import tensor_batches
    from test_reasoning_local_qlora_e2e import tiny_quantized_qwen
    from mlx_lm import load
    model_path = tiny_quantized_qwen(tmp_path / "model", tied=True)
    _, tokenizer = load(model_path)
    _, original, config = prepared_action_view(tmp_path, embeddings=True,
        model=model_path, tokenizer=tokenizer)
    prompt = json.loads(original["messages"][1]["content"])
    prompt["fields"] = ["trade.unrealized_r"]
    original["messages"][1]["content"] = json.dumps(prompt)
    original["targets"]["specialist_targets"] = {"expansion.long": .9, "expansion.short": .1}
    start, day = original["completed_at_ns"], 86400 * 10**9
    valid = copy.deepcopy(original)
    valid.update(source_id="valid-staged", completed_at_ns=start + day,
                 label_end_ns=original["label_end_ns"] + day)
    source = tmp_path / "staged-data"
    write_supervised_dataset([original, valid], source,
        splits={"train": [start, start + day], "valid": [start + day, start + 2 * day]},
        sealed_start_ns=start + 2 * day,
        lineage={"source_identity": "test", "specialist_identities": "test",
                 "economic_contract": "test", "split_audit": "test"})
    (source / "audit.json").write_text(json.dumps({"status": "PASS",
        "manifest_sha256": file_digest(source / "manifest.json"),
        "specialist_score_mode": "post_fit", "sealed_touched": False}))
    config.update(data=str(source), architecture="staged_reasoning_v1",
                  staged_policy=settings(), interpretation_loss_weight=.5,
                  selection="hierarchical_greedy", chat_template_kwargs={"enable_thinking": False})
    recipe = tmp_path / "staged.json"
    from propevolve.reasoning_policy.staged_inference import StagedReasoningPolicy
    frozen_config = tmp_path / "frozen-base.json"
    frozen_config.write_text(json.dumps({**config, "adapter_path": None}))
    frozen = StagedReasoningPolicy.from_config(frozen_config)
    frozen.save(config["adapter_path"])
    recipe.write_text(json.dumps(config))
    view = tmp_path / "staged-view"
    prepare_mlx_view(recipe, view, tokenizer=tokenizer)
    dataset = PreparedDataset(view, "train")
    row = dataset[0]
    assert "alternatives" not in row and "tokens" not in row
    batch = next(tensor_batches(dataset, 1, config["max_seq_length"]))[0]
    np.testing.assert_array_equal(batch["inputs"]["embeddings"][0], original["market_embeddings"])
    assert batch["targets"]["teacher_probabilities"].tolist()[0] == pytest.approx([.9, .1])
    assert len(batch["inputs"]["legal_actions"][0]) == 3
    from propevolve.reasoning_policy.learning_audit import assess_prepared
    output = tmp_path / "assessment"
    report = assess_prepared(recipe, view, role="train", output=output)
    scores = json.loads((output / "scores.jsonl").read_text())
    assert report["weights_updated"] is False
    assert report["metrics"]["decision_boundary_semantics"] == "staged_independent_binary_v1"
    assert set(scores["assessment"]) == {"entry", "direction", "management"}
    assert set(scores["interpretation"]) == {"expansion.long", "expansion.short"}
    assert scores["score_type"] == "log_probability"
    from propevolve.reasoning_policy.learning_audit import score_labeled_examples
    direct = score_labeled_examples(frozen, [original])[0]
    assert direct["assessment"] == scores["assessment"]
    assert direct["predicted"] == scores["predicted"]
    altered = copy.deepcopy(original)
    altered["targets"]["specialist_targets"] = {"expansion.long": .1, "expansion.short": .9}
    assert score_labeled_examples(frozen, [altered])[0]["assessment"] == direct["assessment"]
    from propevolve.reasoning_policy.mlx_sft import train_prepared
    trained = tmp_path / "trained"
    config.update(adapter_path=str(trained), iters=2, steps_per_eval=1,
        steps_per_report=1, save_every=1, val_batches=1, seed=11,
        trainable_components=["lora", "projector"], learning_rate=1e-3,
        optimizer="adam", optimizer_config={"adam": {}}, lr_schedule=None)
    recipe.write_text(json.dumps(config))
    train_prepared(recipe, view)
    reloaded = StagedReasoningPolicy.from_config(recipe)
    assert reloaded.requires_specialists is False
    metadata = json.loads((trained / "adapter_config.json").read_text())
    assert metadata["architecture"] == "staged_reasoning_v1"
    assert {"adapters.safetensors", "projector.safetensors"} <= set(metadata["weight_files"])
    final_report = assess_prepared(recipe, view, role="valid", output=tmp_path / "trained-assessment")
    assert final_report["rows"] == 1
    assert final_report["metrics"]["decision_boundary_semantics"] == "staged_independent_binary_v1"
    from propevolve.reasoning_policy.rl import rollout, MLXAdapterLearner
    from propevolve.reasoning_policy.context import ContextConfig
    from test_reasoning_challenger_e2e import environment
    # Trade-context rollout here tests the shared computation; full challenge
    # context eligibility is tested separately by the RL job contract.
    config["staged_policy"]["state_fields"] = ["trade.current_r"]
    reloaded.settings["staged_policy"]["state_fields"] = ["trade.current_r"]
    decisions, terminal = rollout(reloaded, environment(), options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(3, ("trade.current_r",), input_mode="embeddings"),
        sources=(), rng=np.random.default_rng(11), max_steps=8)
    assert decisions and terminal["outcome"] in {"pass", "blow", "timeout"}
    assert decisions[0].staged_context.embeddings is not None
    rl_config = {"seed": 11, "learning_rate": 1e-4, "weight_decay": 0.,
        "max_update_rows": 1, "epochs": 1, "minibatch_size": 1,
        "clip_epsilon": .2, "kl_weight": .01, "entropy_weight": 0., "max_grad_norm": 1.}
    learner = MLXAdapterLearner(reloaded, rl_config)
    update = learner.update([(decisions[0], 1.)], np.random.default_rng(11))
    assert update["mean_gradient_norm"] > 0

    # Shape-compatible old action adapters must not become staged parents just
    # because a caller omitted optional warm-start requirements.
    metadata["architecture"] = "direct_action"
    (trained / "adapter_config.json").write_text(json.dumps(metadata))
    config.update(adapter_path=str(tmp_path / "invalid-warm-start"),
        resume_adapter_file=str(trained / "adapters.safetensors"),
        resume_adapter_requirements=None)
    # Restore the original SFT state schema used by this prepared view.
    config["staged_policy"]["state_fields"] = ["trade.unrealized_r"]
    recipe.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="staged.*parent"):
        train_prepared(recipe, view)

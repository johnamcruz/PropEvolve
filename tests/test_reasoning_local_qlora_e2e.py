"""Offline real MLX-LM QLoRA tracer: simulator labels -> train -> reload."""
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")


@pytest.fixture(autouse=True)
def deterministic_mlx():
    mx.random.seed(11)
    yield
    mx.synchronize()
    mx.clear_cache()


def tiny_quantized_qwen(path, *, tied=False):
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import save_config, save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    args = ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=4096,
        rope_theta=10000., head_dim=16, tie_word_embeddings=tied)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    save_model(path, model)
    config = vars(args) | {"quantization": {"group_size": 32, "bits": 4},
                           "eos_token_id": 127}
    save_config(config, path / "config.json")
    alphabet = [chr(i) for i in range(32, 127)]
    vocabulary = {"[UNK]": 0, "[PAD]": 1, **{character: i + 2 for i, character in enumerate(alphabet)}}
    vocabulary["[EOS]"] = 127
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
        pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    tokenizer.save_pretrained(path)
    return path


def test_inference_assessment_exports_indexed_scores_without_changing_metrics(tmp_path):
    from mlx_lm import load
    from propevolve.reasoning_policy.mlx_sft import PreparedDataset, read_sft_config
    from propevolve.reasoning_policy.projector import attach_projector
    from propevolve.reasoning_policy.supervised_trainer import evaluate_action_validation
    from test_reasoning_prepared_full_action_e2e import prepared_action_view

    model_path = tiny_quantized_qwen(tmp_path / "model")
    model, tokenizer = load(model_path)
    prepared_action_view(tmp_path, embeddings=True, tokenizer=tokenizer, model=model_path)
    config = read_sft_config(tmp_path / "recipe.json")
    config["validation_batch_size"] = 1
    attach_projector(model, config["projector"])
    model.eval()
    data = PreparedDataset(tmp_path / "view", "train")
    observed = []
    result = evaluate_action_validation(model, data, config,
        on_scored=lambda index, scores: observed.append((index, scores)))
    assert len(observed) == 1
    assert observed[0][0] == 0
    assert len(observed[0][1]) == 3
    assert np.isfinite(observed[0][1]).all()
    assert result == evaluate_action_validation(model, data, config)




def test_staged_adapter_feeds_corrective_sft_in_existing_workflow(tmp_path):
    from mlx_lm import load
    from propevolve.reasoning_policy.workflow import run_workflow
    from test_reasoning_prepared_full_action_e2e import prepared_action_view
    model_path = tiny_quantized_qwen(tmp_path / "model")
    _, tokenizer = load(model_path)
    _, _, action = prepared_action_view(tmp_path, embeddings=True, tokenizer=tokenizer,
        model=model_path, iters=2, staged=True)
    action.update(seed=11, validation_batch_size=1, val_batches=1, steps_per_report=1,
        steps_per_eval=1, save_every=2, trainable_components=["lora", "projector"],
        component_learning_rates={"lora": 1e-3, "projector": 1e-3})
    parent = {**action, "adapter_path": str(tmp_path / "parent-adapter")}
    action.update(resume_adapter_file=str(tmp_path / "parent-adapter/adapters.safetensors"),
                  resume_adapter_requirements={"architecture": "staged_reasoning_v1",
                                               "staged_policy": parent["staged_policy"]})
    (tmp_path / "parent.json").write_text(json.dumps(parent))
    (tmp_path / "action.json").write_text(json.dumps(action))
    steps = []
    for name, adapter in (("parent", "parent-adapter"), ("action", "adapter")):
        job = {"workspace_root": str(tmp_path), "sft_config": f"{name}.json",
               "mlx_view": f"{name}-workflow-view"}
        (tmp_path / f"{name}-job.json").write_text(json.dumps(job))
        steps.append({"id": name, "stage": "train", "job_config": f"{name}-job.json",
            "inputs": [f"{name}.json"], "outputs": [f"{adapter}/training_selection.json",
                f"{adapter}/adapters.safetensors", f"{adapter}/projector.safetensors"],
            "log": f"{name}.log", "timeout_seconds": 120})
    plan = {"workspace_root": str(tmp_path), "state_file": "state.json", "steps": steps}
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(plan))
    result = run_workflow(path)
    assert result["status"] == "COMPLETE"
    assert set(result["completed"]) == {"parent", "action"}
    saved = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert saved["resume_adapter_file"] == action["resume_adapter_file"]
    assert "status=complete" in (tmp_path / "adapter/training.log").read_text()






def test_production_sft_resume_matches_uninterrupted_optimizer_path(tmp_path):
    from mlx_lm import load
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view, train_prepared
    from test_reasoning_prepared_full_action_e2e import prepared_action_view
    import shutil
    model_path = tiny_quantized_qwen(tmp_path / "model")
    _, tokenizer = load(model_path)
    _, _, config = prepared_action_view(tmp_path, embeddings=True, tokenizer=tokenizer,
                                        model=model_path, iters=2, staged=True)
    config.update(seed=11, val_batches=1, steps_per_report=1, steps_per_eval=1,
                  save_every=1, trainable_components=["lora", "projector"],
                  component_learning_rates={"lora": 1e-4, "projector": 1e-3},
                  save_training_state=True,
                  early_stopping={"enabled": True, "patience_evaluations": 8,
                                  "min_delta": 0., "restore_best": True})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(config))
    shutil.rmtree(tmp_path / "view")
    prepare_mlx_view(recipe, tmp_path / "view", tokenizer=tokenizer)
    train_prepared(recipe, tmp_path / "view")
    expected = mx.load(str(Path(config["adapter_path"]) / "training-state" / "weights.safetensors"))
    # Same prepared data, model and learner; stop after one completed update.
    first = {**config, "iters": 1, "adapter_path": str(tmp_path / "first")}
    recipe.write_text(json.dumps(first))
    train_prepared(recipe, tmp_path / "view")
    resumed = {**config, "adapter_path": str(tmp_path / "resumed"),
               "resume_training_state": str(tmp_path / "first" / "training-state")}
    recipe.write_text(json.dumps(resumed))
    train_prepared(recipe, tmp_path / "view")
    actual = mx.load(str(tmp_path / "resumed" / "training-state" / "weights.safetensors"))
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("tied", [False, True])
def test_chunked_market_head_preserves_unequal_completion_gradients(tmp_path, tied):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    from propevolve.reasoning_policy.projector import attach_projector
    from propevolve.reasoning_policy.supervised_trainer import batch_loss, pack_examples
    model, _ = load(tiny_quantized_qwen(tmp_path / "model", tied=tied))
    model.freeze()
    linear_to_lora_layers(model, 1, {"rank": 2, "scale": 4., "dropout": 0.})
    attach_projector(model, {"embedding_dim": 2, "context_steps": 3,
                             "market_tokens": 2, "temporal_encoding": "pooled_levels"})
    rows = [{"tokens": [1, 2, 3, 4, 5], "offset": 2,
             "market_embeddings": [[1., 2.], [2., 3.], [3., 4.]],
             "market_available": [True, True, True]},
            {"tokens": [2, 3, 4], "offset": 1,
             "market_embeddings": [[2., 1.], [3., 2.], [4., 3.]],
             "market_available": [True, True, True]}]
    packed = tuple(mx.array(x) for x in pack_examples(rows, max_seq_length=8))
    config = {"input_mode": "embeddings", "action_supervision": {"enabled": False}}
    with pytest.raises(ValueError, match="market SFT"):
        batch_loss(model, *packed, config={**config, "market_loss_chunk_size": 3,
                   "action_supervision": {"enabled": True}})
    expected, gradients = nn.value_and_grad(model, lambda m: batch_loss(m, *packed, config=config)[0])(model)
    actual, chunk_gradients = nn.value_and_grad(model, lambda m:
        batch_loss(m, *packed, config={**config, "market_loss_chunk_size": 3})[0])(model)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    reference = dict(tree_flatten(gradients))
    for name, gradient in tree_flatten(chunk_gradients):
        np.testing.assert_allclose(gradient, reference[name], atol=1e-5, rtol=1e-5)

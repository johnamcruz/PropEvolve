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


def tiny_quantized_qwen(path):
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import save_config, save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    args = ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=4096,
        rope_theta=10000., head_dim=16, tie_word_embeddings=False)
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


def test_local_quantized_model_learns_full_action_labels_and_reloads(tmp_path):
    from mlx_lm import load
    from propevolve.reasoning_policy.mlx_sft import train_prepared
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from propevolve.reasoning_policy.learning_audit import score_labeled_examples
    from test_reasoning_prepared_full_action_e2e import prepared_action_view

    model_path = tiny_quantized_qwen(tmp_path / "model")
    _, tokenizer = load(model_path)
    prepared, record, config = prepared_action_view(tmp_path, tokenizer=tokenizer,
        model=model_path, iters=2)
    config.update(seed=11, learning_rate=1e-3, max_seq_length=4096,
        val_batches=1, steps_per_report=1, steps_per_eval=1, save_every=2,
        optimizer="adam", optimizer_config={"adam": {}}, lr_schedule=None,
        clear_cache_threshold=0, chat_template_kwargs={"enable_thinking": False})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(config))
    # Rebuild the view after the recipe changes so its identity remains exact.
    import shutil
    shutil.rmtree(tmp_path / "view")
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view
    prepare_mlx_view(recipe, tmp_path / "view", tokenizer=tokenizer)
    base = MLXActionPolicy.load(model_path, adapter_path=None, max_seq_length=4096,
                                action_verbalizers=config["action_verbalizers"])
    before = score_labeled_examples(base, [record])[0]
    train_prepared(recipe, tmp_path / "view")
    trained = MLXActionPolicy.load(model_path, adapter_path=config["adapter_path"], max_seq_length=4096,
                                   action_verbalizers=config["action_verbalizers"])
    after = score_labeled_examples(trained, [record])[0]
    reloaded = MLXActionPolicy.load(model_path, adapter_path=config["adapter_path"], max_seq_length=4096,
                                    action_verbalizers=config["action_verbalizers"])
    repeat = score_labeled_examples(reloaded, [record])[0]
    assert after["target_log_likelihood"] > before["target_log_likelihood"]
    np.testing.assert_allclose(list(after["scores"].values()), list(repeat["scores"].values()), atol=1e-5)

    from propevolve.reasoning_policy.rl import MLXAdapterLearner, RLDecision
    names = tuple(record["targets"]["action_order"])
    logits = np.asarray([after["scores"][name] for name in names])
    old_log_probs = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
    selected = names.index(record["messages"][-1]["content"])
    decision = RLDecision(record["messages"][:-1], names, selected,
        tuple(old_log_probs), reward=1.)
    rl_config = {"seed": 11, "learning_rate": 1e-3, "weight_decay": 0.,
        "max_update_rows": 1, "epochs": 1, "minibatch_size": 1,
        "clip_epsilon": .2, "kl_weight": .01, "entropy_weight": 0.,
        "max_grad_norm": 1.}
    learner = MLXAdapterLearner(trained, rl_config)
    update = learner.update([(decision, 1.)], np.random.default_rng(11))
    assert update["sampled_action_mass"] == {names[selected]: 1}
    assert update["mean_gradient_norm"] > 0
    rl_path = tmp_path / "rl-adapter"
    learner.save(rl_path, config["adapter_path"], {"contract": {"fixture": "real-mlx"}},
                 runtime={"next_group": 1})
    from propevolve.reasoning_policy.checkpoints import verify_checkpoint
    assert verify_checkpoint(rl_path)["contract"] == {"fixture": "real-mlx"}
    from propevolve.reasoning_policy.checkpoints import restore_training_state
    rl_reloaded = MLXActionPolicy.load(model_path, adapter_path=rl_path, max_seq_length=4096,
                                       action_verbalizers=config["action_verbalizers"])
    rl_scores = score_labeled_examples(rl_reloaded, [record])[0]["scores"]
    np.testing.assert_allclose(list(rl_scores.values()),
                               list(score_labeled_examples(trained, [record])[0]["scores"].values()), atol=1e-5)
    resumed = MLXAdapterLearner(rl_reloaded, rl_config)
    assert restore_training_state(rl_path, optimizer=resumed.optimizer) == {"next_group": 1}
    learner.update([(decision, 1.)], np.random.default_rng(22))
    resumed.update([(decision, 1.)], np.random.default_rng(22))
    continued = score_labeled_examples(trained, [record])[0]["scores"]
    resumed_scores = score_labeled_examples(rl_reloaded, [record])[0]["scores"]
    np.testing.assert_allclose(list(continued.values()), list(resumed_scores.values()), atol=1e-5)


def test_local_quantized_embedding_policy_evaluates_without_teacher_lookups(tmp_path):
    from mlx_lm import load
    from propevolve.reasoning_policy.context import ContextConfig
    from propevolve.reasoning_policy.evaluation import evaluate_policy
    from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view, train_prepared
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from test_reasoning_challenger_e2e import environment
    from test_reasoning_prepared_full_action_e2e import prepared_action_view

    model_path = tiny_quantized_qwen(tmp_path / "model")
    _, tokenizer = load(model_path)
    _, _, config = prepared_action_view(tmp_path, embeddings=True, tokenizer=tokenizer,
        model=model_path, iters=1)
    config.update(seed=11, learning_rate=1e-3, max_seq_length=4096,
        val_batches=1, steps_per_report=1, steps_per_eval=1, save_every=1,
        optimizer="adam", optimizer_config={"adam": {}}, lr_schedule=None,
        trainable_components=["lora", "projector"],
        component_learning_rates={"lora": 1e-4, "projector": 1e-3},
        clear_cache_threshold=0, chat_template_kwargs={"enable_thinking": False})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(config))
    import shutil
    shutil.rmtree(tmp_path / "view")
    prepare_mlx_view(recipe, tmp_path / "view", tokenizer=tokenizer)
    train_prepared(recipe, tmp_path / "view")
    policy = MLXActionPolicy.load(model_path, adapter_path=config["adapter_path"],
        max_seq_length=4096, input_mode="embeddings", projector=config["projector"],
        action_verbalizers=config["action_verbalizers"])
    class ForbiddenSources:
        def __iter__(self):
            raise AssertionError("teacher lookup during teacher-free evaluation")
    result = evaluate_policy(policy, environment(), episodes=[{"ticker": "NQ", "start": 0}],
        context_config=ContextConfig(3, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=ForbiddenSources(), max_steps=8)
    assert result["teacher_free"] is True
    assert result["episodes"][0]["outcome"] in {"pass", "blow", "timeout"}

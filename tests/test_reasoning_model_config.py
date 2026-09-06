"""Config-to-runtime contract; no real model downloads or training."""

import json
import sys
from types import SimpleNamespace

import pytest

from propevolve.reasoning_policy.policy import MLXActionPolicy


def test_arbitrary_recipe_selects_model_and_matching_adapter(tmp_path, monkeypatch):
    # MLX-LM is the external runtime seam, not a mocked application collaborator.
    calls = []
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(chat_template="template", eos_token="END",
                                encode=lambda text: [1], apply_chat_template=lambda *a, **k: "prompt")
    def load(name, **kwargs):
        calls.append((name, kwargs["adapter_path"]))
        return model, tokenizer
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    for name in ("example/backbone-a", "example/backbone-b"):
        adapter = tmp_path / name.rsplit("/", 1)[-1]
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"model": name}))
        recipe = tmp_path / "any-name.json"
        recipe.write_text(json.dumps({"model": name, "adapter_path": str(adapter),
                                      "max_seq_length": 1024}))
        policy = MLXActionPolicy.from_config(recipe)
        assert policy.max_seq_length == 1024
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
        MLXActionPolicy.from_config(recipe)


def test_wrong_base_adapter_rejected_before_optional_runtime_load(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"model": "example/old"}))
    recipe = tmp_path / "replacement.json"
    recipe.write_text(json.dumps({"model": "example/new", "adapter_path": str(adapter),
                                  "max_seq_length": 1024}))
    with pytest.raises(ValueError, match="base model"):
        MLXActionPolicy.from_config(recipe)

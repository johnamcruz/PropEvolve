"""Opt-in real SFT-adapter -> simulator RL -> reload -> evaluation acceptance.

Requires reviewed local resources and explicit permission to run model compute.
This is not enabled by the normal test suite.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("PROPEVOLVE_REASONING_RL_TEST_CONFIG"),
                    reason="real simulator/MLX RL needs explicit local-resource opt-in")
def test_real_rl_job_updates_adapter_and_evaluates_saved_policy(tmp_path):
    from propevolve.reasoning_policy.job import main
    from propevolve.reasoning_policy.model_config import read_model_settings
    import mlx.core as mx
    config = json.loads(Path(os.environ["PROPEVOLVE_REASONING_RL_TEST_CONFIG"]).read_text())
    root = Path(config["workspace_root"])
    rl = json.loads((root / config["rl_config"]).read_text())
    model = read_model_settings(root / rl["input_policy_config"])
    assert Path(model["model"]).is_dir(), "use an approved local base, never an implicit download"
    output = tmp_path / "rl-adapter"
    rl["output_adapter"] = str(output)
    rl["checkpoint_root"] = str(tmp_path / "rl-checkpoints")
    rl["resume_checkpoint"] = None
    rl_path = tmp_path / "rl.json"
    rl_path.write_text(json.dumps(rl))
    evaluation_model = tmp_path / "policy.json"
    evaluation_model.write_text(json.dumps({**model, "adapter_path": str(output)}))
    config.update(rl_config=str(rl_path), evaluation_policy_config=str(evaluation_model),
                  evaluation_output=str(tmp_path / "evaluation.json"),
                  evaluation_decisions=str(tmp_path / "decisions.jsonl"))
    recipe = tmp_path / "job.json"
    recipe.write_text(json.dumps(config))
    assert main(["--config", str(recipe), "rl"]) == 0
    before = mx.load(str(Path(model["adapter_path"]) / "adapters.safetensors"))
    after = mx.load(str(output / "adapters.safetensors"))
    assert set(after).issubset(before)
    assert any(bool(mx.any(after[key] != before[key]).item()) for key in after), "no adapter learning occurred"
    del before, after
    mx.clear_cache()
    assert main(["--config", str(recipe), "evaluate"]) == 0
    receipt = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(receipt["episodes"]) == len(config["evaluation_episodes"])
    assert all(item["outcome"] in {"pass", "blow", "timeout"} for item in receipt["episodes"])
    assert receipt["teacher_free"] is False


@pytest.mark.skipif(not os.environ.get("PROPEVOLVE_REASONING_RL_TEST_CONFIG"),
                    reason="real optimizer resume parity requires explicit local-resource opt-in")
def test_real_rl_group_resume_matches_uninterrupted_training(tmp_path):
    from propevolve.reasoning_policy.job import main
    import mlx.core as mx
    original = json.loads(Path(os.environ["PROPEVOLVE_REASONING_RL_TEST_CONFIG"]).read_text())
    root = Path(original["workspace_root"])
    learning = json.loads((root / original["rl_config"]).read_text())
    def run(name, groups, resume=None):
        base = tmp_path / name
        base.mkdir()
        rl = {**learning, "groups": groups, "resume_checkpoint": resume,
              "checkpoint_every_groups": 1, "checkpoint_root": str(base / "checkpoints"),
              "output_adapter": str(base / "adapter")}
        recipe = base / "rl.json"
        recipe.write_text(json.dumps(rl))
        job = base / "job.json"
        job.write_text(json.dumps({**original, "rl_config": str(recipe)}))
        assert main(["--config", str(job), "rl"]) == 0
        return base
    uninterrupted = run("uninterrupted", 2)
    first = run("first", 1)
    resumed = run("resumed", 2, str(first / "checkpoints" / "group-000001"))
    left = mx.load(str(uninterrupted / "adapter" / "adapters.safetensors"))
    right = mx.load(str(resumed / "adapter" / "adapters.safetensors"))
    assert set(left) == set(right)
    for key in left:
        assert bool(mx.allclose(left[key], right[key], atol=1e-6, rtol=1e-5).item()), key

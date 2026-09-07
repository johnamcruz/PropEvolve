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
    rl_path = tmp_path / "rl.json"
    rl_path.write_text(json.dumps(rl))
    evaluation_model = tmp_path / "policy.json"
    evaluation_model.write_text(json.dumps({**model, "adapter_path": str(output)}))
    config.update(rl_config=str(rl_path), evaluation_policy_config=str(evaluation_model),
                  evaluation_output=str(tmp_path / "evaluation.json"))
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

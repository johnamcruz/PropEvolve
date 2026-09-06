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
    from propevolve.reasoning_policy.mlx_sft import read_sft_config, verify_dataset
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from propevolve.reasoning_policy.learning_audit import score_labeled_examples
    from propevolve.decision import Action

    path = Path(os.environ["PROPEVOLVE_REASONING_SFT_TEST_CONFIG"])
    config = read_sft_config(path)
    assert Path(config["model"]).is_dir(), "use an approved local model, not an implicit download"
    verify_dataset(config["data"])
    cohort = {}
    with (Path(config["data"]) / "train.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            cohort.setdefault(record["messages"][-1]["content"], record)
            if set(cohort) == {action.name for action in Action}:
                break
    assert set(cohort) == {action.name for action in Action}, "real fixture must cover all five actions"
    records = list(cohort.values())
    policy = MLXActionPolicy.load(config["model"], adapter_path=None,
                                 max_seq_length=config["max_seq_length"])
    before = score_labeled_examples(policy, records)
    del policy
    gc.collect()
    import mlx.core as mx
    mx.synchronize()
    mx.clear_cache()
    effective = {**config, "adapter_path": str(tmp_path / "adapter")}
    recipe = tmp_path / "sft.json"
    recipe.write_text(json.dumps(effective))
    subprocess.run([sys.executable, "-m", "propevolve.reasoning_policy.mlx_sft",
                    "--config", str(recipe), "--view", str(tmp_path / "view"), "--train"], check=True)
    policy = MLXActionPolicy.load(config["model"], adapter_path=effective["adapter_path"],
                                 max_seq_length=config["max_seq_length"])
    after = score_labeled_examples(policy, records)
    del policy
    gc.collect()
    mx.clear_cache()
    reloaded = MLXActionPolicy.load(config["model"], adapter_path=effective["adapter_path"],
                                   max_seq_length=config["max_seq_length"])
    again = score_labeled_examples(reloaded, records)
    for old, new, repeat in zip(before, after, again):
        assert new["target_log_likelihood"] > old["target_log_likelihood"], new["target"]
        assert new["correct"], new
        assert repeat["target_log_likelihood"] == pytest.approx(new["target_log_likelihood"], abs=1e-4)

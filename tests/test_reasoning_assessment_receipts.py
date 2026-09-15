"""Receipt reuse at the assessment process boundary, without loading a model."""

import json
from pathlib import Path

import pytest


def fixture(tmp_path):
    from test_reasoning_corrective_campaign import sft_config
    config = tmp_path / "policy.json"
    sft_config(config, tmp_path / "adapter")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_text("base")
    (model / "tokenizer.json").write_text("tokenizer")
    data = tmp_path / "data"
    data.mkdir()
    view = tmp_path / "view"
    view.mkdir()
    for directory in (data, view):
        for role in ("train", "valid"):
            (directory / f"{role}.jsonl").write_text('{"row":1}\n')
    (view / "view_manifest.json").write_text('{}')
    payload = json.loads(config.read_text())
    payload.update(model=str(model), data=str(data))
    config.write_text(json.dumps(payload))
    (tmp_path / "adapter/projector.safetensors").write_text("projector")
    return config, view


def produce(config, view, role, output, log):
    from propevolve.reasoning_policy.integrity import file_digest
    output.mkdir()
    (output / "scores.jsonl").write_text('{"target":"WAIT","scores":{"WAIT":1,"ENTER_LONG_1":0}}\n')
    (output / "summary.json").write_text(json.dumps({
        "role": role, "rows": 1, "weights_updated": False,
        "config_sha256": file_digest(config),
        "view_manifest_sha256": file_digest(view / "view_manifest.json"),
        "metrics": {"decision_boundary_semantics": "independent_enter_direction_v1"},
    }))


def forbidden(*args):
    pytest.fail("identical frozen inputs must not execute assessment inference")


def test_new_campaign_reuses_receipt_after_training_only_changes(tmp_path):
    from propevolve.reasoning_policy.assessment_receipts import cached_assessment
    config, view = fixture(tmp_path)
    cache = tmp_path / "receipts"
    first = tmp_path / "first"
    cached_assessment(config, view, "train", first, tmp_path / "first.log",
                      cache_root=cache, run=produce)
    payload = json.loads(config.read_text())
    payload.update(learning_rate=1e-6, iters=20,
                   component_learning_rates={"lora": 1e-6, "projector": 1e-6})
    config.write_text(json.dumps(payload))
    second = tmp_path / "second"
    cached_assessment(config, view, "train", second, tmp_path / "second.log",
                      cache_root=cache, run=forbidden)
    assert (second / "scores.jsonl").read_bytes() == (first / "scores.jsonl").read_bytes()
    from propevolve.reasoning_policy.integrity import file_digest
    assert json.loads((second / "summary.json").read_text())["config_sha256"] == file_digest(config)
    assert "status=reused" in (tmp_path / "second.log").read_text()


@pytest.mark.parametrize("changed", ["adapter/adapters.safetensors", "adapter/projector.safetensors",
                                    "model/model.safetensors", "model/tokenizer.json",
                                    "data/train.jsonl", "view/train.jsonl", "view/view_manifest.json"])
def test_changed_frozen_input_runs_new_assessment(tmp_path, changed):
    from propevolve.reasoning_policy.assessment_receipts import cached_assessment
    config, view = fixture(tmp_path)
    cache = tmp_path / "receipts"
    cached_assessment(config, view, "train", tmp_path / "first", tmp_path / "a.log",
                      cache_root=cache, run=produce)
    (tmp_path / changed).write_text("changed")
    result = tmp_path / "second"
    cached_assessment(config, view, "train", result, tmp_path / "b.log",
                      cache_root=cache, run=produce)
    assert len(list(cache.glob("*.json"))) == 2
    assert (result / "summary.json").is_file()


@pytest.mark.parametrize("setting,value", [("decision_objective", "full_action"),
                                          ("validation_batch_size", 2), ("seed", 99),
                                          ("architecture", "staged_reasoning_v1"),
                                          ("staged_policy", {"assessment_instruction": "changed"}),
                                          ("selection", "hierarchical_greedy")])
def test_changed_scoring_contract_does_not_reuse_metrics(tmp_path, setting, value):
    from propevolve.reasoning_policy.assessment_receipts import cached_assessment
    config, view = fixture(tmp_path)
    cache = tmp_path / "receipts"
    cached_assessment(config, view, "train", tmp_path / "first", tmp_path / "a.log",
                      cache_root=cache, run=produce)
    payload = json.loads(config.read_text())
    payload[setting] = value
    config.write_text(json.dumps(payload))
    cached_assessment(config, view, "train", tmp_path / "second", tmp_path / "b.log",
                      cache_root=cache, run=produce)
    assert len(list(cache.glob("*.json"))) == 2


def test_corrupt_receipt_fails_without_repeating_inference(tmp_path):
    from propevolve.reasoning_policy.assessment_receipts import cached_assessment
    config, view = fixture(tmp_path)
    cache = tmp_path / "receipts"
    cached_assessment(config, view, "train", tmp_path / "first", tmp_path / "a.log",
                      cache_root=cache, run=produce)
    (tmp_path / "first/scores.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="artifact changed"):
        cached_assessment(config, view, "train", tmp_path / "second", tmp_path / "b.log",
                          cache_root=cache, run=forbidden)


def test_failed_assessment_is_not_published_as_reusable(tmp_path):
    from propevolve.reasoning_policy.assessment_receipts import cached_assessment
    config, view = fixture(tmp_path)
    def fail(*args):
        raise RuntimeError("process failed")
    with pytest.raises(RuntimeError, match="process failed"):
        cached_assessment(config, view, "train", tmp_path / "first", tmp_path / "a.log",
                          cache_root=tmp_path / "receipts", run=fail)
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_production_phase_reuses_across_runs_with_real_prepared_data_and_subprocess(tmp_path):
    from test_reasoning_prepared_full_action_e2e import prepared_action_view
    from propevolve.reasoning_policy.corrective_campaign import SubprocessPhases
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_text("external-model-fixture")
    _, _, config = prepared_action_view(tmp_path, model=model)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_text("external-adapter-fixture")
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    script = tmp_path / "scripts/assess_reasoning_trade.py"
    script.parent.mkdir()
    # Replace only the expensive external model process, not campaign/cache/data
    # internals. The child fails if a cache hit accidentally launches it again.
    script.write_text(
        "import argparse, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_reasoning_assessment_receipts import produce\n"
        "p=argparse.ArgumentParser()\n"
        "for name in ('config','view','role','output','root'): p.add_argument('--'+name)\n"
        "a=p.parse_args()\n"
        "with (Path(a.root)/'inference-was-executed').open('x'): pass\n"
        "produce(Path(a.config),Path(a.view),a.role,Path(a.output),None)\n")
    phase = SubprocessPhases(tmp_path, {"assessment_seconds": 30, "training_seconds": 30})
    recipe, view = tmp_path / "recipe.json", tmp_path / "view"
    phase.assess(recipe, view, "train", tmp_path / "first", tmp_path / "first.log")
    config["learning_rate"] = 1e-6
    recipe.write_text(json.dumps(config))
    phase.assess(recipe, view, "train", tmp_path / "second", tmp_path / "second.log")
    assert "status=reused" in (tmp_path / "second.log").read_text()
    assert (tmp_path / "first/scores.jsonl").read_bytes() == (tmp_path / "second/scores.jsonl").read_bytes()

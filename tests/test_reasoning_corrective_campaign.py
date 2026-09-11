import json

import pytest


ACTIONS = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")


def assessment(path, advantages, *, primary, task_advantages=None):
    path.mkdir()
    rows = []
    for index, (target, advantage) in enumerate(advantages):
        rows.append({"index": index, "source_id": f"row-{index}",
            "completed_at_ns": 100 + index, "ticker": "NQ", "target": target,
            "predicted": target if advantage >= 0 else "other", "correct": advantage >= 0,
            "scores": {}, "target_advantage": advantage, "specialist_targets": {}})
    (path / "scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    tasks = task_advantages or {"trade": primary}
    (path / "summary.json").write_text(json.dumps({
        "role": "valid", "rows": len(rows), "weights_updated": False,
        "metrics": {"worst_task_advantage": primary,
                    "per_task": {name: {"mean_target_advantage": value}
                                 for name, value in tasks.items()}},
    }))


def gate(**updates):
    value = {"primary_metric": "worst_task_advantage",
        "minimum_primary_improvement": 0.1,
        "minimum_mean_mistake_advantage_delta": 0.1,
        "minimum_retained_mastery_rate": 1.0,
        "maximum_per_action_mistake_regression": 0.0,
        "maximum_per_task_advantage_regression": 0.0}
    value.update(updates)
    return value


def test_frozen_candidate_is_accepted_only_after_correcting_mistakes_and_retaining_mastery(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import compare_frozen_assessments
    before_rows, after_rows = [], []
    for action in ACTIONS:
        before_rows.extend(((action, -1.), (action, 1.)))
        after_rows.extend(((action, .5), (action, .8)))
    before, after = tmp_path / "before", tmp_path / "after"
    assessment(before, before_rows, primary=-1., task_advantages={"entry": -.5, "management": -.2})
    assessment(after, after_rows, primary=.2, task_advantages={"entry": .2, "management": .3})
    report = compare_frozen_assessments(before, after, gate())
    assert report["decision"] == "ACCEPTED"
    assert report["corrected_mistake_rate"] == 1.0
    assert report["minimum_retained_mastery_rate"] == 1.0
    assert set(report["per_action"]) == set(ACTIONS)


def test_frozen_candidate_is_rejected_when_one_action_forgets_mastered_rows(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import compare_frozen_assessments
    before_rows, after_rows = [], []
    for action in ACTIONS:
        before_rows.extend(((action, -1.), (action, 1.)))
        after_rows.extend(((action, .5), (action, .8)))
    after_rows[ACTIONS.index("ENTER_SHORT_1") * 2 + 1] = ("ENTER_SHORT_1", -.1)
    before, after = tmp_path / "before", tmp_path / "after"
    assessment(before, before_rows, primary=-1.)
    assessment(after, after_rows, primary=.2)
    report = compare_frozen_assessments(before, after, gate())
    assert report["decision"] == "REJECTED"
    assert "retention" in report["failed_gates"]


def test_frozen_assessment_comparison_rejects_row_drift(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import compare_frozen_assessments
    rows = [(action, -1.) for action in ACTIONS]
    before, after = tmp_path / "before", tmp_path / "after"
    assessment(before, rows, primary=-1.)
    assessment(after, rows, primary=-.5)
    score_path = after / "scores.jsonl"
    changed = [json.loads(line) for line in score_path.read_text().splitlines()]
    changed[0]["source_id"] = "different"
    score_path.write_text("".join(json.dumps(row) + "\n" for row in changed))
    with pytest.raises(ValueError, match="rows differ"):
        compare_frozen_assessments(before, after, gate())


def sft_config(path, adapter):
    payload = {
        "model": "fixture-model", "data": "fixture-data",
        "adapter_path": str(adapter), "train": True, "fine_tune_type": "lora",
        "mask_prompt": True, "num_layers": 1, "batch_size": 1, "iters": 5,
        "learning_rate": 1e-5, "max_seq_length": 64, "grad_checkpoint": True,
        "grad_accumulation_steps": 5, "trust_remote_code": False,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "input_mode": "embeddings", "trainable_components": ["lora", "projector"],
        "component_learning_rates": {"lora": 1e-6, "projector": 3e-6},
        "projector": {"embedding_dim": 2, "context_steps": 2,
                      "market_tokens": 1, "temporal_encoding": "pooled_levels"},
        "action_supervision": {"enabled": True, "soft_target_weight": 1.,
                               "ranking_weight": 1., "margin": .25},
        "decision_objective": "hierarchical_binary",
        "batch_sampling": "balanced_actions",
        "early_stopping": {"enabled": True, "patience_evaluations": 2,
                           "min_delta": 0., "restore_best": True,
                           "monitor": "worst_task_advantage", "mode": "max"},
    }
    path.write_text(json.dumps(payload))
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_text("weights")
    from propevolve.reasoning_policy.mlx_sft import read_sft_config
    (adapter / "adapter_config.json").write_text(json.dumps(read_sft_config(path)))


class FakePhases:
    def __init__(self):
        self.calls = []

    def assess(self, policy_config, view, role, output, log):
        self.calls.append(("assess", role, str(policy_config)))
        name = __import__("pathlib").Path(policy_config).parent.name
        generation = int(name.split("-")[-1]) if name.startswith("round-") else 0
        rows = []
        for action in ACTIONS:
            rows.extend(((action, -.8 + generation * .5),
                         (action, 1. - generation * .05)))
        assessment(output, rows, primary=-.8 + generation * .5,
                   task_advantages={"entry": -.6 + generation * .5,
                                    "management": -.4 + generation * .4})
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["role"] = role
        from propevolve.reasoning_policy.integrity import file_digest
        summary["view_manifest_sha256"] = file_digest(view / "view_manifest.json")
        summary_path.write_text(json.dumps(summary))

    def train(self, config_path, view, log):
        self.calls.append(("train", str(config_path)))
        from propevolve.reasoning_policy.mlx_sft import read_sft_config
        config = read_sft_config(config_path)
        adapter = __import__("pathlib").Path(config["adapter_path"])
        adapter.mkdir(parents=True)
        (adapter / "adapters.safetensors").write_text("weights")
        (adapter / "projector.safetensors").write_text("projector")
        (adapter / "training_selection.json").write_text("{}")
        (adapter / "targeted_sampling_receipt.json").write_text("{}")
        (adapter / "adapter_config.json").write_text(json.dumps(config))


class InterruptOnce(FakePhases):
    def __init__(self):
        super().__init__()
        self.interrupted = False

    def train(self, config_path, view, log):
        if not self.interrupted:
            self.interrupted = True
            self.calls.append(("interrupted", str(config_path)))
            raise RuntimeError("simulated interruption before training output")
        super().train(config_path, view, log)


class ForgetShort(FakePhases):
    def assess(self, policy_config, view, role, output, log):
        super().assess(policy_config, view, role, output, log)
        if "candidate-policy" not in str(policy_config) or role != "valid":
            return
        scores = output / "scores.jsonl"
        rows = [json.loads(line) for line in scores.read_text().splitlines()]
        for row in rows:
            if row["target"] == "ENTER_SHORT_1" and row["target_advantage"] >= 0:
                row["target_advantage"] = -.1
                row["correct"] = False
                row["predicted"] = "other"
        scores.write_text("".join(json.dumps(row) + "\n" for row in rows))


def campaign_config(tmp_path, *, rounds=2):
    initial = tmp_path / "initial.json"
    sft_config(initial, tmp_path / "initial-adapter")
    template = tmp_path / "template.json"
    template.write_text(initial.read_text())
    view = tmp_path / "view"
    view.mkdir()
    (view / "view_manifest.json").write_text("{}")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "schema": "propevolve_reasoning_corrective_campaign_v1",
        "workspace_root": str(tmp_path), "state_file": "run/state.json",
        "output_root": "run", "initial_policy_config": "initial.json",
        "sft_template_config": "template.json", "prepared_view": "view",
        "rounds": rounds, "initial_assessments": {"train": None, "valid": None},
        "subset": {"rows_per_group": 2, "mistake_fraction": .5, "seed": 17},
        "acceptance": gate(minimum_retained_mastery_rate=.9),
        "timeouts": {"assessment_seconds": 60, "training_seconds": 60},
    }))
    return campaign


def test_reasoning_campaign_repeats_assess_correct_reassess_and_resumes(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import run_campaign
    campaign = campaign_config(tmp_path)
    phases = FakePhases()
    result = run_campaign(campaign, phases=phases)
    assert result["status"] == "COMPLETE"
    assert [item["decision"] for item in result["rounds"]] == ["ACCEPTED", "ACCEPTED"]
    assert len([call for call in phases.calls if call[0] == "train"]) == 2
    assert len([call for call in phases.calls if call[0] == "assess"]) == 6
    calls = list(phases.calls)
    assert run_campaign(campaign, phases=phases) == result
    assert phases.calls == calls
    first_child = json.loads((tmp_path / "run/round-01/candidate-policy.json").read_text())
    assert first_child["targeted_sampling"]["assessment_path"].endswith(
        "parent-train-assessment")


def test_reasoning_campaign_resumes_the_interrupted_round_without_reassessment(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import run_campaign
    campaign = campaign_config(tmp_path, rounds=1)
    phases = InterruptOnce()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_campaign(campaign, phases=phases)
    assessment_calls = [call for call in phases.calls if call[0] == "assess"]
    state = json.loads((tmp_path / "run/state.json").read_text())
    assert state["status"] == "BLOCKED"
    assert len(state["rounds"]) == 1
    result = run_campaign(campaign, phases=phases)
    assert result["status"] == "COMPLETE"
    assert [call for call in phases.calls if call[0] == "assess"][:2] == assessment_calls
    assert len([call for call in phases.calls if call[0] == "assess"]) == 4


def test_reasoning_campaign_rejects_forgetting_and_keeps_the_parent(tmp_path):
    from propevolve.reasoning_policy.corrective_campaign import run_campaign
    campaign = campaign_config(tmp_path, rounds=2)
    result = run_campaign(campaign, phases=ForgetShort())
    assert result["status"] == "FAILED_GATE"
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["decision"] == "REJECTED"
    assert "retention" in result["rounds"][0]["failed_gates"]
    assert result["selected_policy_config"] == str((tmp_path / "initial.json").resolve())

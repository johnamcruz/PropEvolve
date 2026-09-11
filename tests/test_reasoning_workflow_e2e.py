"""Real subprocess stage execution; no mocked campaign or training internals."""
import json
from pathlib import Path

import pytest

from propevolve.reasoning_policy.dataset import write_supervised_dataset
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.workflow import run_workflow, stage_inputs


def test_workflow_runs_stages_resumes_and_rejects_changed_evidence(tmp_path):
    dataset = tmp_path / "dataset"
    write_supervised_dataset([
        {"source_id": "train", "completed_at_ns": 10, "label_end_ns": 11},
        {"source_id": "valid", "completed_at_ns": 110, "label_end_ns": 111}], dataset,
        splits={"train": [0, 100], "valid": [100, 200]}, sealed_start_ns=200,
        lineage={"source_identity": "fixture", "specialist_identities": "fixture",
                 "economic_contract": "fixture", "split_audit": "fixture"})
    audit = dataset / "audit.json"
    audit.write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(dataset / "manifest.json"),
        "specialist_score_mode": "out_of_fold", "sealed_touched": False}))
    root = Path(__file__).resolve().parents[1]
    job = tmp_path / "arbitrary-job.json"
    job.write_text(json.dumps({"workspace_root": str(root), "dataset_output": str(dataset)}))
    plan = tmp_path / "arbitrary-plan.json"
    state = tmp_path / "state.json"
    plan.write_text(json.dumps({"workspace_root": str(root), "state_file": str(state),
        "steps": [{"id": name, "stage": "audit", "job_config": str(job),
            "inputs": [str(dataset / "manifest.json"), str(audit)], "outputs": [str(audit)],
            "log": str(tmp_path / f"{name}.log"), "timeout_seconds": 60}
            for name in ("first", "second")]}))
    first = run_workflow(plan)
    assert first["status"] == "COMPLETE"
    assert set(first["completed"]) == {"first", "second"}
    logs = {name: (tmp_path / f"{name}.log").read_bytes() for name in ("first", "second")}
    assert run_workflow(plan)["status"] == "COMPLETE"
    assert all((tmp_path / f"{name}.log").read_bytes() == content for name, content in logs.items())
    audit.write_text(json.dumps({"status": "BLOCKED"}))
    with pytest.raises(ValueError, match="artifacts changed"):
        run_workflow(plan)
    assert json.loads(state.read_text())["status"] == "BLOCKED"


def test_reasoning_workflow_rejects_removed_policy_kinds(tmp_path):
    job = tmp_path / "job.json"
    policy = tmp_path / "policy.json"
    checkpoint = tmp_path / "legacy-checkpoint"
    checkpoint.write_text("removed")
    policy.write_text(json.dumps({"kind": "r2d2", "checkpoint": checkpoint.name}))
    job.write_text(json.dumps({
        "policy_config": policy.name,
        "source_recipe": "source.json",
        "temporal_split_audit": "audit.json",
        "context_config": "context.json",
        "evaluation_policy_config": "evaluation-policy.json",
        "evaluation_metrics_config": "metrics.json",
    }))
    for name in ("source.json", "audit.json", "context.json",
                 "evaluation-policy.json", "metrics.json"):
        (tmp_path / name).write_text("{}")
    step = {
        "stage": "evaluate", "job_config": job.name,
        "inputs": [job.name],
    }

    with pytest.raises(ValueError, match="reasoning policy"):
        stage_inputs(tmp_path, step)


def test_workflow_dispatches_config_driven_corrective_loop_and_receipts_it(
        tmp_path, monkeypatch):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    initial = tmp_path / "initial.json"
    template = tmp_path / "template.json"
    view = tmp_path / "view"
    view.mkdir()
    initial.write_text("{}")
    template.write_text("{}")
    (view / "view_manifest.json").write_text("{}")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"workspace_root": str(root),
        "initial_policy_config": str(initial),
        "sft_template_config": str(template), "prepared_view": str(view)}))
    declared = tmp_path / "declared-input.json"
    declared.write_text("{}")
    output = tmp_path / "campaign-state.json"
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        output.write_text(json.dumps({"status": "COMPLETE"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    plan = tmp_path / "workflow.json"
    plan.write_text(json.dumps({"workspace_root": str(root),
        "state_file": str(tmp_path / "workflow-state.json"), "steps": [{
            "id": "correct", "stage": "corrective", "job_config": str(campaign),
            "inputs": [str(declared)], "outputs": [str(output)],
            "log": str(tmp_path / "controller.log"), "timeout_seconds": 60,
        }]}))
    assert run_workflow(plan)["status"] == "COMPLETE"
    assert seen == [[__import__("sys").executable, "-m",
        "propevolve.reasoning_policy.corrective_campaign", "--config", str(campaign)]]


@pytest.mark.parametrize("steps", [
    [],
    [{"id": "same", "stage": "audit", "inputs": ["in"], "outputs": ["out"],
      "job_config": "job.json", "log": "log", "timeout_seconds": 1}] * 2,
])
def test_workflow_requires_nonempty_unique_stage_ids(tmp_path, steps):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "workspace_root": str(tmp_path), "state_file": "state.json", "steps": steps,
    }))
    with pytest.raises(ValueError, match="unique stage IDs"):
        run_workflow(plan)

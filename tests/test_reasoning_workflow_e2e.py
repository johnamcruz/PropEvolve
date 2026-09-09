"""Real subprocess stage execution; no mocked campaign or training internals."""
import json
from pathlib import Path

import pytest

from propevolve.reasoning_policy.dataset import write_supervised_dataset
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.workflow import run_workflow


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

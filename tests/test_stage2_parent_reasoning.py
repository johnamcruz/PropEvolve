from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_training_loop import Decision, Phase, ReasoningOutcome, Revision
from ml_training_loop.domain import SkillBootstrapReceipt, SkillStatus

from propevolve.evolution import CandidateArchive
from propevolve.orchestration import run_evolution_campaign


class _ReadySkills:
    def ensure(self, required):
        return SkillBootstrapReceipt(tuple(
            SkillStatus(name, "already_present") for name in required
        ))


class _FailThenPassRunner:
    def __init__(self, archive_root: Path) -> None:
        self.archive = CandidateArchive(archive_root)
        self.candidate_ids: list[str] = []

    def run(self, config, *, parent_candidate_ids, hypothesis):
        attempt = len(self.candidate_ids) + 1
        output = Path(config["_root"]) / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "training-diagnostic-summary.json").write_text(json.dumps({
            "schema": "propevolve_training_diagnostic_summary_v1",
            "overall": {"attempt": attempt},
        }))
        model = output / "model.pt"
        model.write_bytes(f"stage2a-attempt-{attempt}".encode())
        candidate = self.archive.register_candidate(
            model,
            contract={"temporal": dict(config["temporal"])},
            recipe=config,
            parent_candidate_ids=parent_candidate_ids,
            hypothesis=hypothesis,
        )
        passed = attempt == 2
        evaluation = self.archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": "stage2a-selection-test-v1"},
            metrics={
                "selection.pass_minus_blow": 0.1 if passed else -0.1,
                "selection.pass_rate": 0.25 if passed else 0.05,
                "selection.blow_rate": 0.0,
                "selection.mean_reward": 0.1 if passed else -0.1,
            },
            stages=({
                "name": "selection",
                "status": "PASS" if passed else "FAIL",
            },),
            status="PASS" if passed else "FAIL",
        )
        self.candidate_ids.append(candidate.candidate_id)
        return candidate, evaluation


class _CaptureFirstReasoningPacket:
    def __init__(self) -> None:
        self.reference: dict | None = None
        self.payload: dict | None = None

    def revise(self, request):
        self.reference = dict(request.receipt.outputs["reasoning_packet"])
        self.payload = json.loads(Path(self.reference["path"]).read_text())
        return ReasoningOutcome(
            Decision.REVISE,
            "increase capacity after the failed Stage2A challenger",
            Revision(
                stage=request.stage.name,
                rationale="increase capacity",
                config_override={"agent.hidden_dim": 256},
            ),
        )


def _stage2_config_with_external_stage1_parent(tmp_path: Path):
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["output"] = "runs/stage2-parent-evidence"
    payload["campaign"]["state_root"] = (
        "runs/stage2-parent-evidence/ml-loop-state"
    )
    payload["challenge"]["ratchet_lock_floor_r"] = 0.0
    payload["campaign"]["budget_stages"] = [{
        "name": "regime_selectivity_1m",
        "minimum_environment_steps": 1_000_000,
        "selection_requirements": payload["campaign"][
            "selection_requirements"
        ],
        "warm_start_parent": True,
    }]

    source = CandidateArchive(tmp_path / "immutable-stage1/archive")
    model = tmp_path / "immutable-stage1.pt"
    model.write_bytes(b"immutable-stage1-policy")
    parent = source.register_candidate(
        model,
        contract={
            "checkpoint_sha256": "test-checkpoint",
            "training_tickers": list(payload["tickers"]),
            "deployment_tickers": list(payload["deployment_tickers"]),
            "training_only_tickers": list(payload["training_only_tickers"]),
            "temporal": dict(payload["temporal"]),
            "sealed_holdout_touched": False,
        },
        recipe=payload,
        hypothesis="selected autonomous Stage1 policy",
    )
    parent_evaluation = source.record_evaluation(
        parent.candidate_id,
        evaluator_contract={"name": "stage1-teacher-free-selection-v1"},
        metrics={
            "selection.pass_rate": 0.23,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 1.952,
        },
        stages=(
            {"name": "training", "status": "PASS"},
            {"name": "selection", "status": "PASS"},
        ),
        status="PASS",
    )
    payload["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(source.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": parent_evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path = tmp_path / "stage2.json"
    config_path.write_text(json.dumps(payload))
    return config_path, source, parent, parent_evaluation


def test_first_stage2a_reasoning_packet_contains_authenticated_stage1_parent(
    tmp_path: Path,
) -> None:
    config_path, source, parent, parent_evaluation = (
        _stage2_config_with_external_stage1_parent(tmp_path)
    )
    output = tmp_path / "runs/stage2-parent-evidence"
    runner = _FailThenPassRunner(output / "archive")
    reasoning = _CaptureFirstReasoningPacket()

    state = run_evolution_campaign(
        config_path,
        run_id="stage2-parent-evidence-test",
        candidate_runner=runner,
        reasoning=reasoning,
        skills=_ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert reasoning.reference is not None
    assert reasoning.payload is not None
    packet_path = Path(reasoning.reference["path"])
    assert hashlib.sha256(packet_path.read_bytes()).hexdigest() == (
        reasoning.reference["file_sha256"]
    )
    assert reasoning.payload["packet_sha256"] == reasoning.reference[
        "identity_sha256"
    ]
    assert reasoning.payload["champion"]["candidate"]["candidate_id"] == (
        runner.candidate_ids[0]
    )

    parent_evidence = next(
        evidence
        for evidence in reasoning.payload["inspirations"]
        if evidence["candidate"]["candidate_id"] == parent.candidate_id
    )
    assert parent_evidence["latest_evaluation"]["evaluation_id"] == (
        parent_evaluation.evaluation_id
    )
    assert parent_evidence["latest_evaluation"]["metrics"] == {
        "selection.average_win_r": 1.952,
        "selection.blow_rate": 0.0,
        "selection.pass_rate": 0.23,
    }
    assert parent_evidence["latest_evaluation"]["stages"] == [
        {"name": "training", "status": "PASS"},
        {"name": "selection", "status": "PASS"},
    ]
    assert [
        item["receipt"]["evaluation_id"]
        for item in parent_evidence["evaluation_history"]
    ] == [parent_evaluation.evaluation_id]

    source_reference = parent_evidence["source"]
    assert source_reference["kind"] == "external"
    assert source_reference["reference"]["source_archive_root"] == str(
        source.root.resolve()
    )
    assert source_reference["reference"]["candidate_manifest_sha256"] == (
        hashlib.sha256((parent.path / "manifest.json").read_bytes()).hexdigest()
    )
    assert source_reference["reference"]["evaluation_sha256"] == (
        hashlib.sha256(parent_evaluation.path.read_bytes()).hexdigest()
    )
    reference_path = (
        runner.archive.external_parents_root / f"{parent.candidate_id}.json"
    )
    assert source_reference["reference_file_sha256"] == hashlib.sha256(
        reference_path.read_bytes()
    ).hexdigest()

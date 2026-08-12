from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_training_loop import (
    Decision,
    Phase,
    ReasoningOutcome,
    Revision,
    SurrogateAdvice,
)
from ml_training_loop.domain import SkillBootstrapReceipt, SkillStatus

from propevolve.evolution import CandidateArchive
from propevolve.orchestration import (
    _reasoning_prompt,
    _resolve_codex_executable,
    run_evolution_campaign,
)


class ReadySkills:
    def ensure(self, required):
        return SkillBootstrapReceipt(tuple(
            SkillStatus(name, "already_present") for name in required
        ))


class FakeCandidateRunner:
    def __init__(self, output_root: Path) -> None:
        self.archive = CandidateArchive(output_root / "archive")
        self.configs: list[dict] = []

    def run(self, config, *, parent_candidate_ids, hypothesis):
        self.configs.append(config)
        output = Path(config["_root"]) / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "training-diagnostic-summary.json").write_text(json.dumps({
            "schema": "propevolve_training_diagnostic_summary_v1",
            "overall": {
                "ratchet_activation_rate": 0.03,
                "voluntary_close_count": 100,
            },
        }))
        hidden_dim = int(config["agent"]["hidden_dim"])
        model = self.archive.root / f"candidate-{hidden_dim}.pt"
        model.write_bytes(str(hidden_dim).encode())
        candidate = self.archive.register_candidate(
            model,
            contract={"split": config["temporal"], "max_loss": 3000},
            recipe=config,
            parent_candidate_ids=parent_candidate_ids,
            hypothesis=hypothesis,
        )
        safety_shaping = float(
            config["challenge"].get("mll_proximity_penalty_coefficient", 0.0)
        )
        safe_revision = safety_shaping > 0.0001
        delta = 0.20 if hidden_dim == 256 or safe_revision else -0.10
        evaluation = self.archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": "fake-economic-v1"},
            metrics={
                "selection.pass_minus_blow": delta,
                "selection.pass_rate": 0.60 if delta > 0 else 0.20,
                "selection.blow_rate": (
                    0.0 if safe_revision else 0.10 if delta > 0 else 0.30
                ),
                "selection.two_r_mfe_capture_ratio": (
                    0.55 if delta > 0 else 0.20
                ),
            },
            stages=({"name": "selection", "status": "PASS" if delta > 0 else "FAIL"},),
            status="PASS" if delta > 0 else "FAIL",
        )
        return candidate, evaluation


class ShortCircuitCandidateRunner(FakeCandidateRunner):
    def run(self, config, *, parent_candidate_ids, hypothesis):
        self.configs.append(config)
        output = Path(config["_root"]) / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "training-diagnostic-summary.json").write_text(json.dumps({
            "schema": "propevolve_training_diagnostic_summary_v1",
            "overall": {
                "short_circuited": True,
                "short_circuit_reason": "passes 0 < 1",
            },
        }))
        model = self.archive.root / "short-circuit.pt"
        model.write_bytes(b"short-circuit")
        candidate = self.archive.register_candidate(
            model,
            contract={"split": config["temporal"], "max_loss": 3000},
            recipe=config,
            parent_candidate_ids=parent_candidate_ids,
            hypothesis=hypothesis,
        )
        evaluation = self.archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": "short-circuit-v1"},
            metrics={"training.short_circuited": 1.0},
            stages=({"name": "training", "status": "FAIL"},),
            status="FAIL",
        )
        return candidate, evaluation


class ImproveHiddenDimension:
    def __init__(self) -> None:
        self.packet_paths: list[Path] = []

    def revise(self, request):
        reference = request.receipt.outputs["reasoning_packet"]
        packet = Path(reference["path"])
        assert packet.is_file()
        payload = json.loads(packet.read_text())
        assert reference["identity_sha256"] == payload["packet_sha256"]
        assert reference["file_sha256"] == hashlib.sha256(packet.read_bytes()).hexdigest()
        assert "frozen_recipe_sha256" in request.stage.config
        assert "frozen_contract_sha256" not in request.stage.config
        assert payload["training_diagnostics"]["overall"] == {
            "ratchet_activation_rate": 0.03,
            "voluntary_close_count": 100,
        }
        self.packet_paths.append(packet)
        return ReasoningOutcome(
            Decision.REVISE,
            "increase recurrent capacity after weak economic selection",
            Revision(
                stage=request.stage.name,
                rationale="increase recurrent capacity",
                config_override={"agent.hidden_dim": 256},
            ),
        )


class InspectGepaReflection(ImproveHiddenDimension):
    def __init__(self) -> None:
        super().__init__()
        self.reflections: list[dict] = []

    def revise(self, request):
        reference = request.receipt.outputs["gepa_reflection"]
        path = Path(reference["path"])
        payload = json.loads(path.read_text())
        assert payload["schema"] == "propevolve_gepa_reflection_v1"
        assert payload["identity_sha256"] == reference["identity_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["file_sha256"]
        assert payload["actionable_side_information"]["gate"]["decision"] == "REVISE"
        assert payload["actionable_side_information"]["gate"]["evidence"]["failures"]
        assert payload["parent_reasoning_packet"]["identity_sha256"]
        assert payload["experiment_ledger"]["identity"]
        assert payload["reflection_contract"]["hard_feasibility_gate"] == (
            "selection.blow_rate == 0"
        )
        self.reflections.append(payload)
        return super().revise(request)


class IllegalTemporalRevision:
    def revise(self, request):
        return ReasoningOutcome(
            Decision.REVISE,
            "move the holdout",
            Revision(
                stage=request.stage.name,
                rationale="move the holdout",
                config_override={"temporal.validation_start": "2024-01-01"},
            ),
        )


class IncreaseSafetyShaping:
    def revise(self, request):
        return ReasoningOutcome(
            Decision.REVISE,
            "increase local MLL-cliff learning signal after nonzero blow rate",
            Revision(
                stage=request.stage.name,
                rationale="increase bounded MLL-proximity shaping",
                config_override={
                    "challenge.mll_proximity_penalty_coefficient": 0.0002,
                    "training.terminal_sequence_fraction": 0.5,
                },
            ),
        )


class MissingReasoningExecutable:
    def revise(self, request):
        raise FileNotFoundError("Codex executable not found: codex")


class RecordingSurrogate:
    def __init__(self) -> None:
        self.requests = []

    def advise(self, request):
        self.requests.append(request)
        return SurrogateAdvice(
            backend="fake-optuna-advisor",
            diagnostics={"uncertainty": "high"},
            proposals=({"agent.hidden_dim": 192},),
            evidence={"completed_trials": 1},
        )


class ReasoningRejectsSurrogateProposal(ImproveHiddenDimension):
    def revise(self, request):
        assert request.surrogate_advice is not None
        assert request.surrogate_advice.backend == "fake-optuna-advisor"
        assert request.experiment_ledger is not None
        assert len(request.experiment_ledger.entries) == 1
        metrics = request.experiment_ledger.entries[0].outputs["metrics"]
        assert metrics["selection.pass_minus_blow"] == -0.10
        return super().revise(request)


def _config(tmp_path: Path) -> Path:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["output"] = "runs/evolution-test"
    payload["campaign"]["state_root"] = "runs/evolution-test/ml-loop-state"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def _register_external_stage1_parent(
    archive: CandidateArchive,
    payload: dict,
    *,
    status: str = "PASS",
):
    model = archive.root / "immutable-stage-1.pt"
    model.write_bytes(b"immutable-stage-1-policy")
    candidate = archive.register_candidate(
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
        hypothesis="Stage 1 learned autonomous expansion entries",
    )
    evaluation = archive.record_evaluation(
        candidate.candidate_id,
        evaluator_contract={"name": "stage-1-teacher-free-selection-v1"},
        metrics={
            "selection.pass_rate": 0.23,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 1.952,
        },
        stages=({"name": "selection", "status": status},),
        status=status,
    )
    return candidate, evaluation


def test_fresh_campaign_warm_starts_first_stage_from_external_stage1_candidate(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["agent"]["hidden_dim"] = 256
    payload["challenge"]["ratchet_lock_floor_r"] = 0.0
    payload["campaign"]["budget_stages"] = [{
        "name": "regime_recovery_stage_2",
        "minimum_environment_steps": 1_000_000,
        "selection_requirements": payload["campaign"]["selection_requirements"],
        "parent_improvement_requirements": [{
            "metric": "selection.pass_rate",
            "direction": "maximize",
            "minimum_delta": 0.0,
        }],
        "warm_start_parent": True,
    }]
    output = tmp_path / "runs/evolution-test"
    runner = FakeCandidateRunner(output)
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, parent_evaluation = _register_external_stage1_parent(
        stage1_archive, payload
    )
    parent_bytes = parent.model_path.read_bytes()
    payload["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": parent_evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(payload))

    state = run_evolution_campaign(
        config_path,
        run_id="external-stage-1-warm-start-test",
        candidate_runner=runner,
        reasoning=None,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert runner.configs[0]["_warm_start_model"] == {
        "candidate_id": parent.candidate_id,
        "model_path": str(parent.model_path),
        "model_sha256": parent.manifest["model_sha256"],
    }
    assert parent.model_path.read_bytes() == parent_bytes
    child = next(
        candidate
        for candidate in runner.archive.list_candidates()
        if candidate.candidate_id != parent.candidate_id
    )
    assert child.manifest["parent_candidate_ids"] == [parent.candidate_id]


def test_external_stage1_warm_start_rejects_multiple_declared_parents(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [{
        "name": "regime_recovery_stage_2",
        "minimum_environment_steps": 1_000_000,
        "selection_requirements": payload["campaign"]["selection_requirements"],
        "warm_start_parent": True,
    }]
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(stage1_archive, payload)
    payload["evolution"]["parent_candidate_ids"] = [
        parent.candidate_id,
        "another-parent",
    ]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exactly one parent"):
        run_evolution_campaign(
            config_path,
            run_id="ambiguous-stage-1-parent-test",
            candidate_runner=FakeCandidateRunner(
                tmp_path / "runs/evolution-test"
            ),
            reasoning=None,
            skills=ReadySkills(),
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("missing", "unknown or incomplete candidate"),
        ("non_passing", "requires a PASS evaluation"),
        ("hash_drift", "model identity drifted"),
        ("causal_drift", "causal contract drifted"),
    ),
)
def test_external_stage1_parent_fails_closed_before_child_training(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [{
        "name": "regime_recovery_stage_2",
        "minimum_environment_steps": 1_000_000,
        "selection_requirements": payload["campaign"]["selection_requirements"],
        "warm_start_parent": True,
    }]
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(
        stage1_archive,
        payload,
        status="FAIL" if failure == "non_passing" else "PASS",
    )
    candidate_id = "missing-candidate" if failure == "missing" else parent.candidate_id
    evaluation_id = (
        "missing-evaluation" if failure == "missing" else evaluation.evaluation_id
    )
    if failure == "causal_drift":
        contract_path = parent.path / "contract.json"
        contract = json.loads(contract_path.read_text())
        contract["temporal"]["train_start"] = "2020-01-01"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        # Re-seal the deliberately incompatible fixture as its own valid bundle.
        parent = stage1_archive.register_candidate(
            parent.model_path,
            contract=contract,
            recipe=payload,
            hypothesis="incompatible external parent fixture",
        )
        evaluation = stage1_archive.record_evaluation(
            parent.candidate_id,
            evaluator_contract={"name": "stage-1-selection-v1"},
            metrics={"selection.pass_rate": 0.23},
            stages=({"name": "selection", "status": "PASS"},),
            status="PASS",
        )
        candidate_id = parent.candidate_id
        evaluation_id = evaluation.evaluation_id
    payload["evolution"]["parent_candidate_ids"] = [candidate_id]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": candidate_id,
        "evaluation_id": evaluation_id,
        "model_sha256": (
            "0" * 64
            if failure == "hash_drift"
            else parent.manifest["model_sha256"]
        ),
    }
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    with pytest.raises(ValueError, match=message):
        run_evolution_campaign(
            config_path,
            run_id=f"external-parent-{failure}-test",
            candidate_runner=runner,
            reasoning=None,
            skills=ReadySkills(),
        )
    assert runner.configs == []


def test_external_stage1_parent_freezes_ordinary_mll_entry_guard(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    parent_recipe = json.loads(config_path.read_text())
    parent_recipe["challenge"]["minimum_mll_headroom"] = 500.0
    parent_recipe["challenge"]["ratchet_lock_floor_r"] = 0.0
    parent_recipe["campaign"]["budget_stages"] = [{
        "name": "regime_recovery_stage_2",
        "minimum_environment_steps": 1_000_000,
        "selection_requirements": parent_recipe["campaign"][
            "selection_requirements"
        ],
        "warm_start_parent": True,
    }]
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(
        stage1_archive, parent_recipe
    )
    child_recipe = json.loads(json.dumps(parent_recipe))
    child_recipe["challenge"]["minimum_mll_headroom"] = 250.0
    child_recipe["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    child_recipe["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(child_recipe))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    with pytest.raises(
        ValueError,
        match=r"economic contract drifted at challenge\.minimum_mll_headroom",
    ):
        run_evolution_campaign(
            config_path,
            run_id="stage-2-mll-entry-guard-drift-test",
            candidate_runner=runner,
            reasoning=None,
            skills=ReadySkills(),
        )
    assert runner.configs == []


def test_same_stage_revision_does_not_promote_failed_attempt_to_parent(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [{
        "name": "historical_candidate",
        "minimum_environment_steps": payload["training"][
            "minimum_environment_steps"
        ],
        "selection_requirements": payload["campaign"]["selection_requirements"],
        "warm_start_parent": True,
    }]
    config_path.write_text(json.dumps(payload))
    output = tmp_path / "runs/evolution-test"
    runner = FakeCandidateRunner(output)

    paused = run_evolution_campaign(
        config_path,
        run_id="resume-test",
        candidate_runner=runner,
        reasoning=None,
        skills=ReadySkills(),
    )
    assert paused.phase is Phase.NEEDS_REASONING

    reasoning = ImproveHiddenDimension()
    completed = run_evolution_campaign(
        config_path,
        run_id="resume-test",
        candidate_runner=runner,
        reasoning=reasoning,
        skills=ReadySkills(),
    )

    assert completed.phase is Phase.COMPLETE
    assert completed.attempts["historical_candidate"] == 2
    assert [config["agent"]["hidden_dim"] for config in runner.configs] == [128, 256]
    candidates = runner.archive.list_candidates()
    child = next(item for item in candidates if item.model_path.read_bytes() == b"256")
    failed = next(item for item in candidates if item.model_path.read_bytes() == b"128")
    assert "_warm_start_model" not in runner.configs[0]
    assert "_warm_start_model" not in runner.configs[1]
    assert child.manifest["parent_candidate_ids"] == []
    assert failed.candidate_id != child.candidate_id
    assert reasoning.packet_paths


def test_training_short_circuit_hands_off_to_reasoning_without_selection_metrics(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    runner = ShortCircuitCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="training-short-circuit-test",
        candidate_runner=runner,
        reasoning=None,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.NEEDS_REASONING
    assert "short circuit" in state.message
    assert len(runner.configs) == 1


def test_campaign_recovers_blocked_reasoning_without_retraining_candidate(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    blocked = run_evolution_campaign(
        config_path,
        run_id="recover-provider-test",
        candidate_runner=runner,
        reasoning=MissingReasoningExecutable(),
        skills=ReadySkills(),
    )
    assert blocked.phase is Phase.BLOCKED
    assert len(runner.configs) == 1

    completed = run_evolution_campaign(
        config_path,
        run_id="recover-provider-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
        recover_reasoning=True,
    )

    assert completed.phase is Phase.COMPLETE
    assert len(runner.configs) == 2
    assert [config["agent"]["hidden_dim"] for config in runner.configs] == [128, 256]


def test_codex_executable_can_be_supplied_outside_launchd_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PROPEVOLVE_CODEX_EXECUTABLE", str(executable))

    assert _resolve_codex_executable({}) == executable.resolve()


def test_campaign_advances_through_screen_confirm_and_final_budgets(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [
        {
            "name": "screen_1m",
            "minimum_environment_steps": 1_000_000,
            "selection_requirements": payload["campaign"]["selection_requirements"],
        },
        {
            "name": "confirm_2m",
            "minimum_environment_steps": 2_000_000,
            "selection_requirements": payload["campaign"]["selection_requirements"],
        },
        {
            "name": "final_5m_multiseed",
            "minimum_environment_steps": 5_000_000,
            "seeds": [11111, 22222, 33333],
            "max_parallel": 3,
            "allow_revisions": False,
            "selection_requirements": payload["campaign"]["selection_requirements"],
        },
    ]
    payload["campaign"]["finalization"] = {
        "registry_root": "runs/evolution-test/registry",
        "export_root": "runs/evolution-test/export",
        "minimum_seed_count": 3,
        "ranking": [
            {"metric": "selection.blow_rate", "direction": "minimize"},
            {"metric": "selection.pass_rate", "direction": "maximize"},
        ],
    }
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="staged-budget-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert [item["training"]["minimum_environment_steps"] for item in runner.configs[:3]] == [
        1_000_000,
        1_000_000,
        2_000_000,
    ]
    assert [item["agent"]["hidden_dim"] for item in runner.configs[:3]] == [
        128,
        256,
        256,
    ]
    final_configs = runner.configs[3:]
    assert sorted(item["training"]["seed"] for item in final_configs) == [
        11111,
        22222,
        33333,
    ]
    assert all(
        item["training"]["minimum_environment_steps"] == 5_000_000
        for item in final_configs
    )
    assert all(item["agent"]["hidden_dim"] == 256 for item in final_configs)
    assert len({item["output"] for item in runner.configs}) == 6
    report = json.loads(
        (tmp_path / "runs/evolution-test/export/gauntlet-report.json").read_text()
    )
    assert report["status"] == "PASS"
    assert report["evaluated_seed_count"] == 3
    assert report["selected_seed"] == 11111
    assert (tmp_path / "runs/evolution-test/export/model.pt").is_file()
    champion = json.loads(
        (tmp_path / "runs/evolution-test/registry/champion.json").read_text()
    )
    assert champion["candidate_id"] == report["selected_candidate_id"]


def test_exact_stage2_campaign_revises_from_selected_parents_and_hands_off_recovery(
    tmp_path: Path,
) -> None:
    source = Path(
        "config/historical_mask_expansion_regime_stage2_selectivity_recovery_v1.json"
    )
    payload = json.loads(source.read_text())
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    payload["output"] = "runs/exact-stage2-flow"
    payload["campaign"]["state_root"] = (
        "runs/exact-stage2-flow/ml-loop-state"
    )
    payload["campaign"]["reasoning"]["proposer"] = "standard"

    stage1_archive = CandidateArchive(tmp_path / "stage1/archive")
    parent_model = tmp_path / "stage1-policy.pt"
    parent_model.write_bytes(b"selected-stage1")
    parent = stage1_archive.register_candidate(
        parent_model,
        contract={
            "checkpoint_sha256": "test-checkpoint",
            "training_tickers": list(payload["tickers"]),
            "deployment_tickers": list(payload["deployment_tickers"]),
            "training_only_tickers": list(payload["training_only_tickers"]),
            "temporal": dict(payload["temporal"]),
            "sealed_holdout_touched": False,
        },
        recipe=payload,
        hypothesis="selected autonomous Stage 1 fixture",
    )
    parent_evaluation = stage1_archive.record_evaluation(
        parent.candidate_id,
        evaluator_contract={"name": "stage1-teacher-free"},
        metrics={
            "selection.pass_rate": 0.23,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 1.952,
            "selection.near_blow_timeout_rate": 0.6363636363636364,
            "recovery_stress.recovery_success_rate": 0.0,
            "recovery_stress.mean_terminal_pnl": -2_700.0,
        },
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )
    payload["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": parent_evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path = tmp_path / "config" / "stage2.json"
    config_path.write_text(json.dumps(payload))

    class ExactStage2Runner:
        def __init__(self) -> None:
            self.archive = CandidateArchive(
                tmp_path / "runs/exact-stage2-flow/archive"
            )
            self.calls: list[dict] = []

        def run(self, config, *, parent_candidate_ids, hypothesis):
            stage = Path(config["output"]).parts[-2]
            attempt = int(Path(config["output"]).name.removeprefix("attempt-"))
            self.calls.append({
                "stage": stage,
                "attempt": attempt,
                "parents": tuple(parent_candidate_ids),
                "warm_start": dict(config["_warm_start_model"]),
                "regime_loss": config["regime_selectivity"]["loss_weight"],
                "recovery_fraction": config["recovery_curriculum"][
                    "episode_fraction"
                ],
            })
            output = Path(config["_root"]) / config["output"]
            output.mkdir(parents=True, exist_ok=True)
            (output / "training-diagnostic-summary.json").write_text(json.dumps({
                "schema": "propevolve_training_diagnostic_summary_v1",
                "overall": {"stage": stage, "attempt": attempt},
            }))
            model = output / "model.pt"
            model.write_bytes(f"{stage}-{attempt}".encode())
            candidate = self.archive.register_candidate(
                model,
                contract={"stage": stage, "attempt": attempt},
                recipe=config,
                parent_candidate_ids=parent_candidate_ids,
                hypothesis=hypothesis,
            )
            failed_first_stage2a = stage == "regime_selectivity_1m" and attempt == 1
            if failed_first_stage2a:
                training_metrics = {
                    "short_circuited": 0.0,
                    "regime_selectivity_positive_long_rows": 100.0,
                    "regime_selectivity_positive_short_rows": 100.0,
                    "regime_selectivity_chop_minus_nonchop_target_wait": 0.0,
                }
                metrics = {
                    f"training.{metric}": value
                    for metric, value in training_metrics.items()
                }
                stages = ({
                    "name": "training",
                    "status": "FAIL",
                    "metrics": training_metrics,
                },)
            else:
                metrics = {
                    "training.short_circuited": 0.0,
                    "selection.blow_rate": 0.0,
                    "selection.pass_rate": 0.24,
                    "selection.average_win_r": 1.90,
                    "selection.expectancy_r": 0.05,
                    "selection.two_r_mfe_capture_ratio": 0.72,
                    "selection.greedy_entry_rate": 0.09,
                    "selection.near_blow_timeout_rate": (
                        0.50 if stage == "regime_selectivity_1m" else 0.45
                    ),
                    "recovery_stress.blow_rate": (
                        0.10 if stage == "regime_selectivity_1m" else 0.0
                    ),
                    "recovery_stress.recovery_success_rate": (
                        0.05 if stage == "regime_selectivity_1m" else 0.20
                    ),
                    "recovery_stress.mean_terminal_pnl": (
                        -2_650.0
                        if stage == "regime_selectivity_1m"
                        else -2_500.0
                    ),
                    "recovery_stress.entries_used": 10.0,
                    "recovery_stress.one_entry_violations": 0.0,
                }
                stages = ({"name": "selection", "status": "PASS"},)
            passed = not failed_first_stage2a
            evaluation = self.archive.record_evaluation(
                candidate.candidate_id,
                evaluator_contract={"name": "exact-stage2-fake"},
                metrics=metrics,
                stages=stages,
                status="PASS" if passed else "FAIL",
            )
            return candidate, evaluation

    class ReviseSelectivity:
        def revise(self, request):
            assert request.stage.name == "regime_selectivity_1m"
            assert request.gate.decision is Decision.REVISE
            assert request.gate.evidence["evaluation_status"] == "FAIL"
            assert request.gate.evidence["values"] == {}
            assert request.gate.evidence["selection_economics_available"] is False
            assert all(
                not metric.startswith("selection.")
                for metric in request.gate.evidence["available_metrics"]
            )
            assert request.gate.evidence["failures"] == [{
                "metric": "evaluator_stage.training",
                "expected": "PASS",
                "actual": "FAIL",
            }]
            return ReasoningOutcome(
                Decision.REVISE,
                "increase the bounded Regime-selectivity auxiliary weight",
                Revision(
                    stage=request.stage.name,
                    rationale="strengthen the failed selectivity boundary",
                    config_override={"regime_selectivity.loss_weight": 0.35},
                ),
            )

    runner = ExactStage2Runner()
    state = run_evolution_campaign(
        config_path,
        run_id="exact-stage2-flow",
        candidate_runner=runner,
        reasoning=ReviseSelectivity(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert [(call["stage"], call["attempt"]) for call in runner.calls] == [
        ("regime_selectivity_1m", 1),
        ("regime_selectivity_1m", 2),
        ("deficit_recovery_1m", 1),
    ]
    assert runner.calls[0]["parents"] == (parent.candidate_id,)
    assert runner.calls[1]["parents"] == (parent.candidate_id,)
    assert runner.calls[0]["warm_start"]["candidate_id"] == parent.candidate_id
    assert runner.calls[1]["warm_start"]["candidate_id"] == parent.candidate_id
    selected_stage2a = runner.calls[2]["parents"][0]
    assert selected_stage2a not in {
        parent.candidate_id,
        runner.calls[0]["warm_start"]["candidate_id"],
    }
    assert runner.calls[2]["warm_start"]["candidate_id"] == selected_stage2a
    assert runner.calls[2]["regime_loss"] == 0.35
    assert runner.calls[2]["recovery_fraction"] == 0.25


@pytest.mark.parametrize(
    ("stage_name", "revision_paths", "required_text"),
    (
        (
            "regime_selectivity_1m",
            ("regime_selectivity.loss_weight",),
            "dominant-chop versus non-chop",
        ),
        (
            "deficit_recovery_1m",
            ("recovery_curriculum.episode_fraction",),
            "one-entry, $300-risk, -$2,700 start",
        ),
    ),
)
def test_stage2_reasoning_prompt_names_exact_revision_surface_and_evidence(
    stage_name: str,
    revision_paths: tuple[str, ...],
    required_text: str,
) -> None:
    request = SimpleNamespace(
        stage=SimpleNamespace(
            name=stage_name,
            config={"revision_paths": list(revision_paths)},
        ),
        receipt=SimpleNamespace(outputs={"reasoning_packet": {}}),
    )

    prompt = _reasoning_prompt(request)

    assert json.dumps(revision_paths) in prompt
    assert required_text in prompt
    assert "only one or more paths from that exact list" in prompt


def test_parent_retention_gate_rejects_frequency_only_near_blow_lift(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    parent_model = tmp_path / "parent.pt"
    parent_model.write_bytes(b"parent")
    parent = archive.register_candidate(
        parent_model,
        contract={"kind": "parent"},
        recipe={"kind": "parent"},
        hypothesis="parent",
    )
    archive.record_evaluation(
        parent.candidate_id,
        evaluator_contract={"name": "parent"},
        metrics={
            "selection.near_blow_timeout_rate": 0.50,
            "selection.greedy_entry_rate": 0.09,
        },
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )
    child_model = tmp_path / "child.pt"
    child_model.write_bytes(b"child")
    child = archive.register_candidate(
        child_model,
        contract={"kind": "child"},
        recipe={"kind": "child"},
        parent_candidate_ids=(parent.candidate_id,),
        hypothesis="frequency-only child",
    )
    evaluation = archive.record_evaluation(
        child.candidate_id,
        evaluator_contract={"name": "child"},
        metrics={
            "selection.near_blow_timeout_rate": 0.40,
            "selection.greedy_entry_rate": 0.12,
        },
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )
    from propevolve.orchestration import _EconomicEvidenceGate

    evidence = _EconomicEvidenceGate(archive)._evaluate_outputs(
        {
            "candidate_id": child.candidate_id,
            "evaluation_id": evaluation.evaluation_id,
            "evaluation_path": str(evaluation.path),
            "evaluation_sha256": hashlib.sha256(
                evaluation.path.read_bytes()
            ).hexdigest(),
            "metrics": evaluation.metrics,
        },
        (),
        parent_ids=(parent.candidate_id,),
        parent_requirements=({
            "metric": "selection.near_blow_timeout_rate",
            "direction": "minimize",
            "minimum_delta": 0.0,
        },),
        parent_retention_requirements=({
            "metric": "selection.greedy_entry_rate",
            "maximum_regression": 0.02,
        },),
    )

    assert any(
        failure.get("direction") == "retain_upper_bound"
        for failure in evidence["failures"]
    )


def test_optional_gepa_proposer_adds_authenticated_actionable_side_information(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["reasoning"]["proposer"] = "gepa_reflective"
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")
    reasoning = InspectGepaReflection()

    state = run_evolution_campaign(
        config_path,
        run_id="gepa-reflection-test",
        candidate_runner=runner,
        reasoning=reasoning,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert len(reasoning.reflections) == 1


def test_standard_proposer_does_not_add_gepa_reflection(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    class InspectStandard(ImproveHiddenDimension):
        def revise(self, request):
            assert "gepa_reflection" not in request.receipt.outputs
            return super().revise(request)

    state = run_evolution_campaign(
        config_path,
        run_id="standard-reflection-test",
        candidate_runner=runner,
        reasoning=InspectStandard(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE


def test_campaign_blocks_reasoning_that_changes_frozen_temporal_contract(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="frozen-test",
        candidate_runner=runner,
        reasoning=IllegalTemporalRevision(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.BLOCKED
    assert "not allowlisted" in state.message
    assert len(runner.archive.list_candidates()) == 1


def test_campaign_reasoning_can_revise_bounded_reward_and_replay_fields(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path("config/historical_mask_safety_replay_v1.json").read_text()
    )
    payload["output"] = "runs/reward-revision-test"
    payload["campaign"]["state_root"] = (
        "runs/reward-revision-test/ml-loop-state"
    )
    config_path = tmp_path / "reward-config.json"
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/reward-revision-test")

    state = run_evolution_campaign(
        config_path,
        run_id="reward-revision-test",
        candidate_runner=runner,
        reasoning=IncreaseSafetyShaping(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert len(runner.configs) == 2
    assert all(config["_validation_stop_on_blow"] for config in runner.configs)
    assert (
        runner.configs[-1]["challenge"]["mll_proximity_penalty_coefficient"]
        == 0.0002
    )
    assert runner.configs[-1]["training"]["terminal_sequence_fraction"] == 0.5


def test_campaign_promotes_only_parent_improvements_with_zero_blow(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [{
        "name": "historical_candidate",
        "minimum_environment_steps": payload["training"][
            "minimum_environment_steps"
        ],
        "selection_requirements": payload["campaign"]["selection_requirements"],
        "parent_improvement_requirements": [
        {
            "metric": "selection.pass_rate",
            "direction": "maximize",
            "minimum_delta": 0.0,
        },
        {
            "metric": "selection.two_r_mfe_capture_ratio",
            "direction": "maximize",
            "minimum_delta": 0.0,
        },
        ],
    }]
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="retention-promotion-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert len(runner.configs) == 2


def test_campaign_blocks_reasoning_outside_declared_numeric_bounds(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["evolution"]["revision_bounds"] = {
        "agent.hidden_dim": {"minimum": 64, "maximum": 192}
    }
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="bounded-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.BLOCKED
    assert "outside declared bounds" in state.message
    assert len(runner.archive.list_candidates()) == 1


def test_surrogate_advice_is_optional_and_reasoning_remains_controller(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")
    surrogate = RecordingSurrogate()

    state = run_evolution_campaign(
        config_path,
        run_id="surrogate-test",
        candidate_runner=runner,
        reasoning=ReasoningRejectsSurrogateProposal(),
        surrogate=surrogate,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert len(surrogate.requests) == 1
    assert [config["agent"]["hidden_dim"] for config in runner.configs] == [128, 256]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def test_campaign_resumes_reasoning_and_links_revised_child_to_parent(
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
    parent = next(item for item in candidates if item.model_path.read_bytes() == b"128")
    assert "_warm_start_model" not in runner.configs[0]
    assert runner.configs[1]["_warm_start_model"] == {
        "candidate_id": parent.candidate_id,
        "model_path": str(parent.model_path),
        "model_sha256": parent.manifest["model_sha256"],
    }
    assert child.manifest["parent_candidate_ids"] == [parent.candidate_id]
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

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
from propevolve.orchestration import run_evolution_campaign


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
        delta = 0.20 if hidden_dim == 256 else -0.10
        evaluation = self.archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": "fake-economic-v1"},
            metrics={
                "selection.pass_minus_blow": delta,
                "selection.pass_rate": 0.60 if delta > 0 else 0.20,
                "selection.blow_rate": 0.10 if delta > 0 else 0.30,
            },
            stages=({"name": "selection", "status": "PASS" if delta > 0 else "FAIL"},),
            status="PASS" if delta > 0 else "FAIL",
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
    assert child.manifest["parent_candidate_ids"] == [parent.candidate_id]
    assert reasoning.packet_paths


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

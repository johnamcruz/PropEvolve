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
    _assert_parent_causal_contract,
    _plan,
    _reasoning_prompt,
    _resolve_codex_executable,
    run_evolution_campaign,
)
from tests.recipe_fixtures import paired_aplus_recipe, paired_recurrent_aplus_recipe


_CURRENT_CONFIG = paired_aplus_recipe(100)
_PAIRED_RECURRENT_V21_CONFIG = paired_recurrent_aplus_recipe(200)
_ENTRY_CENTER_RECEIPT = Path(
    "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
)
_GENERIC_REVISION_PATHS = (
    "agent.hidden_dim",
    "challenge.mll_proximity_penalty_coefficient",
    "training.terminal_sequence_fraction",
)


def _generic_v4_payload() -> dict:
    """Derive a revisable orchestration fixture from the active recipe."""
    payload = json.loads(_CURRENT_CONFIG.read_text())
    payload["workspace_root"] = "."
    challenge_frozen_paths = [
        f"challenge.{field}"
        for field in payload["challenge"]
        if f"challenge.{field}" not in _GENERIC_REVISION_PATHS
    ]
    payload["evolution"] = {
        "hypothesis": "A bounded challenger can improve matched OOS economics.",
        "parent_candidate_ids": [],
        "allowed_revision_paths": list(_GENERIC_REVISION_PATHS),
        "frozen_paths": [
            path
            for path in payload["evolution"]["frozen_paths"]
            if path != "challenge" and path not in _GENERIC_REVISION_PATHS
        ] + challenge_frozen_paths,
        "revision_bounds": {
            "agent.hidden_dim": {"minimum": 64, "maximum": 512},
            "challenge.mll_proximity_penalty_coefficient": {
                "minimum": 0,
                "maximum": 0.01,
            },
            "training.terminal_sequence_fraction": {
                "minimum": 0,
                "maximum": 1,
            },
        },
    }
    requirements = [{
        "metric": "selection.pass_minus_blow",
        "operator": ">",
        "value": 0,
    }]
    payload["campaign"]["max_revisions_per_stage"] = 24
    payload["campaign"]["budget_stages"] = [{
        "name": "historical_candidate",
        "budget_mode": "episodes",
        "training_episodes": 100,
        "validation_episodes": 200,
        "short_circuit_minimum_episodes": 18,
        "allow_revisions": True,
        "warm_start_parent": True,
        "curriculum_override": {},
        "revision_paths": list(_GENERIC_REVISION_PATHS),
        "selection_requirements": requirements,
    }]
    payload["campaign"]["selection_requirements"] = requirements
    payload["campaign"]["reasoning"]["proposer"] = "standard"
    return payload


def _write_entry_center_receipt(root: Path) -> None:
    destination = root / "config" / "receipts" / _ENTRY_CENTER_RECEIPT.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_ENTRY_CENTER_RECEIPT.read_bytes())


def _episode_stage(payload: dict, name: str, episodes: int = 100) -> dict:
    return {
        "name": name,
        "budget_mode": "episodes",
        "training_episodes": episodes,
        "validation_episodes": 200,
        "short_circuit_minimum_episodes": 18,
        "selection_requirements": payload["campaign"][
            "selection_requirements"
        ],
        "warm_start_parent": True,
    }


def test_stage2a_plan_enforces_the_near_blow_improvement_gate() -> None:
    """The stage adapter, not only top-level JSON, must enforce the target."""
    from propevolve.config import load_experiment_config

    config = load_experiment_config(_CURRENT_CONFIG)
    plan = _plan(config)
    requirements = plan.stages[0].config["selection_requirements"]

    assert plan.stages[0].config["budget_mode"] == "episodes"
    assert {
        "metric": "selection.near_blow_timeout_rate",
        "operator": "<=",
        "value": 0.6263636363636363,
    } in requirements


def test_stage2a_projects_frozen_exact_tiers_and_decisive_validation_stops() -> None:
    from propevolve.config import load_experiment_config

    config = load_experiment_config(_CURRENT_CONFIG)
    plan = _plan(config)

    assert [stage.name for stage in plan.stages] == [
        "paired_aplus_contrastive_100ep",
    ]
    assert [stage.config["training_episodes"] for stage in plan.stages] == [100]
    assert all(
        stage.config["budget_mode"] == "episodes"
        and stage.config["validation_episodes"] == 200
        and stage.config["short_circuit_minimum_episodes"] == 18
        and stage.config["allow_revisions"] is False
        and stage.config["revision_paths"] == []
        and stage.config["parent_improvement_requirements"]
        == [
            {
                "metric": "selection.pass_rate",
                "direction": "maximize",
                "minimum_delta": 0.0,
            },
            {
                "metric": "selection.near_blow_timeout_rate",
                "direction": "minimize",
                "minimum_delta": 0.0,
            },
        ]
        and {
            "metric": "selection.blow_rate",
            "operator": "==",
            "value": 0,
        } in stage.config["selection_requirements"]
        for stage in plan.stages
    )
    assert config["training"]["validation_no_trade_patience_episodes"] == 5
    assert config["training"]["short_circuit"]["policy_health"][
        "require_positive_persistent_regime_association"
    ] is True
    assert "episode_coverage" not in plan.stages[-1].config


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("loss_weight", 0.49),
        ("q_temperature", 1.7),
        ("formula", "changed-formula"),
        ("semantics", "static_state_v1"),
        ("persistent_chop_negative_emphasis", 0.5),
    ),
)
def test_training_plan_identity_binds_active_stage2a_recipe_values(
    path: str,
    value: object,
) -> None:
    from propevolve.config import load_experiment_config

    baseline = load_experiment_config(_CURRENT_CONFIG)
    changed = json.loads(json.dumps(baseline))
    changed["_root"] = baseline["_root"]
    changed["_path"] = baseline["_path"]
    changed["regime_selectivity"][path] = value

    assert _plan(changed).identity != _plan(baseline).identity


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
        model = output / "model.pt"
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
                "training.minimum_market_episode_coverage": 1.0,
                "training.episode_coverage_complete": 1.0,
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
        assert "public_base_recipe_sha256" in request.stage.config
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
    payload = _generic_v4_payload()
    payload["workspace_root"] = "."
    payload["output"] = "runs/evolution-test"
    payload["campaign"]["state_root"] = "runs/evolution-test/ml-loop-state"
    _write_entry_center_receipt(tmp_path)
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


def test_v21_matches_frozen_parent_causal_recipe_identity(
    tmp_path: Path,
) -> None:
    from propevolve.config import load_experiment_config

    child_recipe = load_experiment_config(_PAIRED_RECURRENT_V21_CONFIG)
    parent_recipe = json.loads(json.dumps(child_recipe))
    parent_recipe["sealed_confirmation"]["maximum_blow_rate"] = 0.0
    parent_recipe["sealed_confirmation"]["minimum_expectancy_r"] = 0.0
    parent_recipe["entry_supervision"]["target_r"] = 2.0
    parent_recipe["entry_supervision"]["stop_r"] = 1.0
    child_recipe["entry_supervision"]["loss_weight"] = 0.6
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    (candidate_path / "recipe.json").write_text(json.dumps(parent_recipe))
    (candidate_path / "contract.json").write_text(json.dumps({
        "training_tickers": list(child_recipe["tickers"]),
        "deployment_tickers": list(child_recipe["deployment_tickers"]),
        "training_only_tickers": list(child_recipe["training_only_tickers"]),
        "temporal": dict(child_recipe["temporal"]),
        "sealed_holdout_touched": False,
    }))

    _assert_parent_causal_contract(
        SimpleNamespace(path=candidate_path),
        child_recipe,
    )

    child_recipe["entry_supervision"]["target_r"] = 3.0
    with pytest.raises(
        ValueError,
        match="external parent causal recipe drifted at entry_supervision",
    ):
        _assert_parent_causal_contract(
            SimpleNamespace(path=candidate_path),
            child_recipe,
        )


def test_fresh_campaign_warm_starts_first_stage_from_external_stage1_candidate(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["agent"]["hidden_dim"] = 256
    payload["challenge"]["ratchet_lock_floor_r"] = 0.0
    payload["campaign"]["budget_stages"] = [{
        **_episode_stage(payload, "regime_recovery_stage_2"),
        "parent_improvement_requirements": [{
            "metric": "selection.pass_rate",
            "direction": "maximize",
            "minimum_delta": 0.0,
        }],
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


def test_external_stage1_parent_allows_training_only_teacher_replacement(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    child_recipe = json.loads(config_path.read_text())
    child_recipe["agent"]["hidden_dim"] = 256
    child_recipe["campaign"]["budget_stages"] = [
        _episode_stage(child_recipe, "regime_recovery_stage_2")
    ]
    parent_recipe = json.loads(json.dumps(child_recipe))
    parent_recipe["teachers"][1] = {
        "kind": "regime",
        "cache_root": "cache/regime_teacher_9market_3min_pre2025_v1",
        "channels": ["retired_regime_probability"],
        "loss_weight": 0.1,
        "entry_search_loss_weight": 0.0,
    }
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(
        stage1_archive, parent_recipe
    )
    child_recipe["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    child_recipe["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(child_recipe))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="stage2a-replaces-training-only-teacher-test",
        candidate_runner=runner,
        reasoning=None,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert runner.configs[0]["teachers"][1]["cache_root"] == (
        child_recipe["teachers"][1]["cache_root"]
    )
    assert tuple(runner.configs[0]["teachers"][1]["channels"]) == tuple(
        child_recipe["teachers"][1]["channels"]
    )


def test_external_stage1_warm_start_rejects_multiple_declared_parents(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [
        _episode_stage(payload, "regime_recovery_stage_2")
    ]
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
    payload["campaign"]["budget_stages"] = [
        _episode_stage(payload, "regime_recovery_stage_2")
    ]
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
    parent_recipe["campaign"]["budget_stages"] = [
        _episode_stage(parent_recipe, "regime_recovery_stage_2")
    ]
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


def test_external_parent_accepts_explicit_frozen_mll_guard_override(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    parent_recipe = json.loads(config_path.read_text())
    parent_recipe["challenge"]["minimum_mll_headroom"] = 500.0
    parent_recipe["challenge"]["ratchet_lock_floor_r"] = 0.0
    parent_recipe["campaign"]["budget_stages"] = [
        _episode_stage(parent_recipe, "balance_curriculum")
    ]
    stage1_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(
        stage1_archive, parent_recipe
    )
    child_recipe = json.loads(json.dumps(parent_recipe))
    child_recipe["challenge"]["minimum_mll_headroom"] = 200.0
    child_recipe["evolution"]["external_parent_economic_overrides"] = [
        "minimum_mll_headroom"
    ]
    child_recipe["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    child_recipe["evolution"]["base_parent"] = {
        "archive_root": str(stage1_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(child_recipe))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="explicit-mll-entry-guard-override-test",
        candidate_runner=runner,
        reasoning=None,
        skills=ReadySkills(),
    )

    assert state.phase is Phase.NEEDS_REASONING
    assert runner.configs[0]["challenge"]["minimum_mll_headroom"] == 200.0


def test_same_stage_revision_does_not_promote_failed_attempt_to_parent(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["campaign"]["budget_stages"] = [
        _episode_stage(payload, "historical_candidate")
    ]
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


def test_multistage_retry_hands_selected_child_to_next_warm_start(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    requirements = payload["campaign"]["selection_requirements"]
    payload["campaign"]["budget_stages"] = [
        {
            **_episode_stage(payload, "matched_screen", 100),
        },
        {
            **_episode_stage(payload, "matched_confirmation", 250),
            "allow_revisions": False,
            "revision_paths": [],
        },
    ]

    external_archive = CandidateArchive(tmp_path / "stage-1-output/archive")
    parent, evaluation = _register_external_stage1_parent(
        external_archive,
        payload,
    )
    payload["evolution"]["parent_candidate_ids"] = [parent.candidate_id]
    payload["evolution"]["base_parent"] = {
        "archive_root": str(external_archive.root),
        "candidate_id": parent.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "model_sha256": parent.manifest["model_sha256"],
    }
    config_path.write_text(json.dumps(payload))

    class LineageRunner(FakeCandidateRunner):
        def __init__(self, output_root: Path) -> None:
            super().__init__(output_root)
            self.lineage: list[dict] = []

        def run(self, config, *, parent_candidate_ids, hypothesis):
            candidate, candidate_evaluation = super().run(
                config,
                parent_candidate_ids=parent_candidate_ids,
                hypothesis=hypothesis,
            )
            output = Path(config["output"])
            self.lineage.append({
                "stage": output.parts[-2],
                "attempt": int(output.name.removeprefix("attempt-")),
                "parent_candidate_ids": tuple(parent_candidate_ids),
                "warm_start": dict(config["_warm_start_model"]),
                "candidate_id": candidate.candidate_id,
            })
            return candidate, candidate_evaluation

    runner = LineageRunner(tmp_path / "runs/evolution-test")
    state = run_evolution_campaign(
        config_path,
        run_id="multistage-parent-handoff-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert [
        (call["stage"], call["attempt"])
        for call in runner.lineage
    ] == [
        ("matched_screen", 1),
        ("matched_screen", 2),
        ("matched_confirmation", 1),
    ]
    first_attempt, passing_retry, confirmation = runner.lineage
    for call in (first_attempt, passing_retry):
        assert call["parent_candidate_ids"] == (parent.candidate_id,)
        assert call["warm_start"]["candidate_id"] == parent.candidate_id

    selected_child = runner.archive.load_candidate(
        passing_retry["candidate_id"]
    )
    assert confirmation["parent_candidate_ids"] == (
        selected_child.candidate_id,
    )
    assert confirmation["warm_start"] == {
        "candidate_id": selected_child.candidate_id,
        "model_path": str(selected_child.model_path),
        "model_sha256": selected_child.manifest["model_sha256"],
    }
    assert confirmation["parent_candidate_ids"] != (
        first_attempt["candidate_id"],
    )


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
            **_episode_stage(payload, "screen_100ep", 100),
        },
        {
            **_episode_stage(payload, "confirm_250ep", 250),
        },
        {
            **_episode_stage(payload, "final_500ep_multiseed", 500),
            "seeds": [11111, 22222, 33333],
            "max_parallel": 3,
            "allow_revisions": False,
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
    assert [item["training"]["episodes"] for item in runner.configs[:3]] == [
        100,
        100,
        250,
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
        item["training"]["episodes"] == 500
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


def test_campaign_projects_exact_episode_budgets_and_short_circuit_boundaries(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["training"]["budget_mode"] = "episodes"
    payload["training"]["episodes"] = 500
    payload["challenge"]["mll_proximity_penalty_coefficient"] = 0.0002
    payload["training"].pop("minimum_environment_steps", None)
    payload["evolution"]["frozen_paths"] = [
        path for path in payload["evolution"]["frozen_paths"]
        if path != "training.minimum_environment_steps"
    ]
    short_circuit = payload["training"]["short_circuit"]
    short_circuit["minimum_completed_episodes"] = 18
    short_circuit.pop("minimum_environment_steps", None)
    requirements = [
        *payload["campaign"]["selection_requirements"],
        {
            "metric": "selection.blow_rate",
            "operator": "==",
            "value": 0.0,
        },
    ]
    payload["campaign"]["selection_requirements"] = requirements
    payload["campaign"]["budget_stages"] = [
        {
            "name": f"episode_{episodes}",
            "budget_mode": "episodes",
            "training_episodes": episodes,
            "validation_episodes": 200,
            "short_circuit_minimum_episodes": 18,
            "selection_requirements": requirements,
            "warm_start_parent": index > 0,
        }
        for index, episodes in enumerate((200, 300, 500))
    ]
    coverage = {
        "schema": "full_data_episode_coverage_v1",
        "episode_budget": 500,
    }
    final_stage = payload["campaign"]["budget_stages"][-1]
    final_stage["episode_coverage"] = coverage
    final_stage["selection_requirements"] = [
        *requirements,
        {
            "metric": "training.minimum_market_episode_coverage",
            "operator": "==",
            "value": 1.0,
        },
        {
            "metric": "training.episode_coverage_complete",
            "operator": "==",
            "value": 1.0,
        },
    ]
    config_path.write_text(json.dumps(payload))
    runner = FakeCandidateRunner(tmp_path / "runs/evolution-test")

    state = run_evolution_campaign(
        config_path,
        run_id="episode-budget-test",
        candidate_runner=runner,
        reasoning=ImproveHiddenDimension(),
        skills=ReadySkills(),
    )

    assert state.phase is Phase.COMPLETE
    assert [config["training"]["episodes"] for config in runner.configs] == [
        200,
        300,
        500,
    ]
    assert all(
        config["training"]["budget_mode"] == "episodes"
        and config["training"]["validation_episodes"] == 200
        and config["training"]["short_circuit"][
            "minimum_completed_episodes"
        ] == 18
        for config in runner.configs
    )
    assert all(
        config["_validation_stop_on_blow"] is True
        and config["training"]["validation_no_trade_patience_episodes"] == 5
        for config in runner.configs
    )
    assert all(
        "episode_coverage" not in config["training"]
        for config in runner.configs[:-1]
    )
    assert runner.configs[-1]["training"]["episode_coverage"] == coverage


def test_episode_stage_plan_preserves_budget_units_without_step_aliases(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["training"]["budget_mode"] = "episodes"
    payload["training"]["episodes"] = 200
    payload["training"].pop("minimum_environment_steps", None)
    payload["evolution"]["frozen_paths"] = [
        path for path in payload["evolution"]["frozen_paths"]
        if path != "training.minimum_environment_steps"
    ]
    short_circuit = payload["training"]["short_circuit"]
    short_circuit["minimum_completed_episodes"] = 18
    short_circuit.pop("minimum_environment_steps", None)
    payload["campaign"]["budget_stages"] = [{
        "name": "episode_200",
        "budget_mode": "episodes",
        "training_episodes": 200,
        "validation_episodes": 200,
        "short_circuit_minimum_episodes": 18,
        "selection_requirements": payload["campaign"][
            "selection_requirements"
        ],
    }]
    config_path.write_text(json.dumps(payload))

    from propevolve.config import load_experiment_config

    stage = _plan(load_experiment_config(config_path)).stages[0]

    assert stage.config["budget_mode"] == "episodes"
    assert stage.config["training_episodes"] == 200
    assert stage.config["validation_episodes"] == 200
    assert stage.config["short_circuit_minimum_episodes"] == 18
    assert "minimum_environment_steps" not in stage.config


def test_episode_stage_plan_accepts_configured_full_budget_without_short_circuit(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["training"]["budget_mode"] = "episodes"
    payload["training"]["episodes"] = 200
    payload["training"]["short_circuit"] = None
    payload["training"].pop("minimum_environment_steps", None)
    payload["evolution"]["frozen_paths"] = [
        path
        for path in payload["evolution"]["frozen_paths"]
        if path not in {
            "training.minimum_environment_steps",
            "training.short_circuit.policy_health",
        }
    ]
    payload["campaign"]["budget_stages"] = [{
        "name": "complete_200_episode_budget",
        "budget_mode": "episodes",
        "training_episodes": 200,
        "validation_episodes": 200,
        "selection_requirements": payload["campaign"][
            "selection_requirements"
        ],
    }]
    config_path.write_text(json.dumps(payload))

    from propevolve.config import load_experiment_config

    stage = _plan(load_experiment_config(config_path)).stages[0]

    assert stage.config["budget_mode"] == "episodes"
    assert stage.config["training_episodes"] == 200
    assert stage.config["validation_episodes"] == 200
    assert "short_circuit_minimum_episodes" not in stage.config


def test_v4_reasoning_prompt_names_current_stage_and_empty_revision_surface() -> None:
    request = SimpleNamespace(
        stage=SimpleNamespace(
            name="persistent_chop_negative_500k",
            config={"revision_paths": []},
        ),
        receipt=SimpleNamespace(outputs={"reasoning_packet": {}}),
    )

    prompt = _reasoning_prompt(request)

    assert "The current stage is persistent_chop_negative_500k" in prompt
    assert "complete revision allowlist is []" in prompt
    assert "only one or more paths from that exact list" in prompt
    assert "Never change data, FFM lineage, temporal roles" in prompt


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


def test_parent_retention_gate_rejects_regression_in_a_maximized_economic_metric(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    parent_model = tmp_path / "parent.pt"
    parent_model.write_bytes(b"parent")
    parent = archive.register_candidate(
        parent_model,
        contract={"kind": "parent"},
        recipe={"kind": "parent"},
        hypothesis="selected Stage 1 parent",
    )
    archive.record_evaluation(
        parent.candidate_id,
        evaluator_contract={"name": "parent"},
        metrics={"selection.pass_rate": 0.23},
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
        hypothesis="Regime child that forgot Stage 1 economics",
    )
    evaluation = archive.record_evaluation(
        child.candidate_id,
        evaluator_contract={"name": "child"},
        metrics={"selection.pass_rate": 0.22},
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
        parent_requirements=(),
        parent_retention_requirements=({
            "metric": "selection.pass_rate",
            "direction": "maximize",
            "maximum_regression": 0.0,
        },),
    )

    assert evidence["parent_improvements"]["retain:selection.pass_rate"] == {
        "current": 0.22,
        "parent": 0.23,
        "direction": "maximize",
        "regression": pytest.approx(0.01),
        "maximum_regression": 0.0,
    }
    assert evidence["failures"] == [{
        "metric": "selection.pass_rate",
        "direction": "retain_maximize",
        "parent": 0.23,
        "maximum_regression": 0.0,
        "actual": 0.22,
    }]


def test_legacy_parent_retention_keeps_its_max_parent_baseline(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")

    def candidate_with_metric(name: str, metric: float):
        model = tmp_path / f"{name}.pt"
        model.write_bytes(name.encode())
        candidate = archive.register_candidate(
            model,
            contract={"kind": name},
            recipe={"kind": name},
            hypothesis=name,
        )
        archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": name},
            metrics={"selection.greedy_entry_rate": metric},
            stages=({"name": "selection", "status": "PASS"},),
            status="PASS",
        )
        return candidate

    weak = candidate_with_metric("weak", 0.10)
    strong = candidate_with_metric("strong", 0.20)
    child_model = tmp_path / "child.pt"
    child_model.write_bytes(b"child")
    child = archive.register_candidate(
        child_model,
        contract={"kind": "child"},
        recipe={"kind": "child"},
        parent_candidate_ids=(weak.candidate_id, strong.candidate_id),
        hypothesis="legacy child",
    )
    evaluation = archive.record_evaluation(
        child.candidate_id,
        evaluator_contract={"name": "child"},
        metrics={"selection.greedy_entry_rate": 0.15},
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
        parent_ids=(weak.candidate_id, strong.candidate_id),
        parent_requirements=(),
        parent_retention_requirements=({
            "metric": "selection.greedy_entry_rate",
            "maximum_regression": 0.0,
        },),
    )

    assert evidence["failures"] == []
    assert evidence["parent_improvements"][
        "retain:selection.greedy_entry_rate"
    ]["parent"] == 0.20


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
    payload = _generic_v4_payload()
    payload["output"] = "runs/reward-revision-test"
    payload["campaign"]["state_root"] = (
        "runs/reward-revision-test/ml-loop-state"
    )
    _write_entry_center_receipt(tmp_path)
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
    assert all(
        config["_validation_stop_on_blow"] is False
        for config in runner.configs
    )
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
        **_episode_stage(payload, "historical_candidate"),
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

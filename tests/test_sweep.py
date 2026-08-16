from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pytest

from ml_training_loop import Phase, RunState, StageReceipt

from propevolve.orchestration import _plan
from propevolve.sweep import load_grid_sweep, run_grid_sweep


STUDY = Path("config/sweeps/stage2a_regime_selectivity_grid_v1.json")


def test_stage2a_grid_compiles_four_unique_frozen_campaign_cells() -> None:
    sweep = load_grid_sweep(STUDY)

    cells = sweep.cells()

    assert [cell.parameters for cell in cells] == [
        {
            "regime_selectivity.persistent_chop_negative_emphasis": 1.0,
            "training.teacher_guidance_dropout_end": 0.5,
        },
        {
            "regime_selectivity.persistent_chop_negative_emphasis": 1.0,
            "training.teacher_guidance_dropout_end": 1.0,
        },
        {
            "regime_selectivity.persistent_chop_negative_emphasis": 2.0,
            "training.teacher_guidance_dropout_end": 0.5,
        },
        {
            "regime_selectivity.persistent_chop_negative_emphasis": 2.0,
            "training.teacher_guidance_dropout_end": 1.0,
        },
    ]
    assert len({cell.identity_sha256 for cell in cells}) == 4
    assert all(
        [stage["name"] for stage in cell.config["campaign"]["budget_stages"]]
        == ["persistent_chop_association_100ep"]
        for cell in cells
    )
    assert all(cell.config["campaign"]["max_revisions_per_stage"] == 0 for cell in cells)
    assert all(cell.config["evolution"]["allowed_revision_paths"] == [] for cell in cells)
    assert sweep.mps_workers == 1


def _complete_state(run_id: str, near_blow_rate: float) -> RunState:
    metrics = {
        "selection.blow_rate": 0.0,
        "selection.near_blow_timeout_rate": near_blow_rate,
        "selection.pass_rate": 0.25,
        "selection.expectancy_r": 0.1,
        "selection.average_win_r": 2.0,
        "selection.two_r_mfe_capture_ratio": 0.75,
        "selection.long_entry_count": 1.0,
        "selection.short_entry_count": 1.0,
        "training.short_circuited": 0.0,
    }
    return RunState(
        run_id,
        "plan",
        Phase.COMPLETE,
        receipts=(StageReceipt(
            "persistent_chop_association_100ep",
            1,
            "complete",
            {"metrics": metrics},
        ),),
    )


def _with_current_plan(state: RunState, config_path: Path) -> RunState:
    return replace(
        state,
        plan_identity=_plan(json.loads(config_path.read_text())).identity,
    )


def _failed_state(
    run_id: str,
    *,
    training_short_circuit: bool = False,
    validation_short_circuit: bool = False,
) -> RunState:
    metrics = {
        "training.short_circuited": float(training_short_circuit),
        "selection.short_circuited": float(validation_short_circuit),
    }
    if validation_short_circuit:
        metrics["selection.blow_rate"] = 1.0
    return RunState(
        run_id,
        "plan",
        Phase.FAILED_GATE,
        receipts=(StageReceipt(
            "persistent_chop_association_100ep",
            1,
            "complete",
            {"metrics": metrics},
        ),),
    )


def test_grid_runs_existing_campaign_cells_sequentially_and_resumes_receipts(
    tmp_path: Path,
) -> None:
    calls = []
    events = []
    states = {}
    near_blow_rates = iter((0.40, 0.30, 0.20, 0.10))

    def state_loader(config_path: Path, run_id: str):
        return states.get(run_id)

    def runner(config_path: Path, *, run_id: str):
        events.append(("run", config_path.name))
        calls.append((config_path, run_id))
        state = _with_current_plan(
            _complete_state(run_id, next(near_blow_rates)),
            config_path,
        )
        states[run_id] = state
        return state

    result = run_grid_sweep(
        STUDY,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        runner=runner,
        state_loader=state_loader,
        config_validator=lambda path: events.append(("validate", path.name)),
    )

    assert len(calls) == 4
    assert [kind for kind, _ in events[:4]] == ["validate"] * 4
    assert [kind for kind, _ in events[4:]] == ["run"] * 4
    assert result.status == "COMPLETE"
    assert result.winner_cell == "cell-04"
    assert [cell.status for cell in result.cells] == ["COMPLETE_SAFE"] * 4
    ledger = json.loads((tmp_path / "study" / "leaderboard.json").read_text())
    assert ledger["winner_cell"] == "cell-04"

    def forbidden_runner(config_path: Path, *, run_id: str):
        raise AssertionError("authenticated terminal cells must be reused")

    resumed = run_grid_sweep(
        STUDY,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        runner=forbidden_runner,
        state_loader=state_loader,
        config_validator=lambda path: None,
    )

    assert resumed.winner_cell == "cell-04"
    assert all(cell.reused for cell in resumed.cells)


def test_grid_releases_failed_cells_and_stops_only_on_shared_blocker(
    tmp_path: Path,
) -> None:
    states = {}
    calls = []

    def state_loader(config_path: Path, run_id: str):
        return states.get(run_id)

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        index = len(calls)
        if index == 1:
            state = _failed_state(run_id, training_short_circuit=True)
        elif index == 2:
            state = _failed_state(run_id, validation_short_circuit=True)
        else:
            state = _complete_state(run_id, 0.2 + index / 100)
        state = _with_current_plan(state, config_path)
        states[run_id] = state
        return state

    result = run_grid_sweep(
        STUDY,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        runner=runner,
        state_loader=state_loader,
        config_validator=lambda path: None,
    )

    assert len(calls) == 4
    assert [cell.status for cell in result.cells] == [
        "TRAINING_SHORT_CIRCUIT",
        "VALIDATION_SHORT_CIRCUIT",
        "COMPLETE_SAFE",
        "COMPLETE_SAFE",
    ]

    blocked_calls = []

    def blocked_runner(config_path: Path, *, run_id: str):
        blocked_calls.append(run_id)
        if len(blocked_calls) == 2:
            return _with_current_plan(
                RunState(run_id, "plan", Phase.BLOCKED),
                config_path,
            )
        return _with_current_plan(_complete_state(run_id, 0.2), config_path)

    with pytest.raises(RuntimeError, match="blocked; queue stopped: cell-02"):
        run_grid_sweep(
            STUDY,
            artifact_root=tmp_path / "blocked-study",
            config_root=tmp_path / "blocked-config",
            runner=blocked_runner,
            state_loader=lambda config_path, run_id: None,
            config_validator=lambda path: None,
        )
    assert len(blocked_calls) == 2


def test_grid_rejects_stale_terminal_state_before_reuse(tmp_path: Path) -> None:
    calls = []

    with pytest.raises(ValueError, match="plan identity drifted: cell-01"):
        run_grid_sweep(
            STUDY,
            artifact_root=tmp_path / "study",
            config_root=tmp_path / "config",
            runner=lambda config_path, run_id: calls.append(run_id),
            state_loader=lambda config_path, run_id: _complete_state(run_id, 0.2),
            config_validator=lambda path: None,
        )

    assert calls == []


def test_grid_compiles_from_the_exact_base_bytes_loaded_once(tmp_path: Path) -> None:
    source = load_grid_sweep(STUDY)
    base_path = tmp_path / "config" / source.base_config_path.name
    base_path.parent.mkdir(parents=True)
    original = json.loads(source.base_config_path.read_text())
    base_path.write_text(json.dumps(original))
    contract = json.loads(STUDY.read_text())
    contract["base_config"] = base_path.name
    contract_path = base_path.parent / "sweep.json"
    contract_path.write_text(json.dumps(contract))

    sweep = load_grid_sweep(contract_path)
    changed = json.loads(json.dumps(original))
    changed["training"]["teacher_guidance_dropout_end"] = 0.25
    base_path.write_text(json.dumps(changed))

    assert {
        cell.config["training"]["teacher_guidance_dropout_end"]
        for cell in sweep.cells()
    } == {0.5, 1.0}

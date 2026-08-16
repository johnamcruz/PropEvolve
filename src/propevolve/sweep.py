"""Small, deterministic grid studies over the existing campaign runner."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping


SWEEP_SCHEMA = "propevolve_grid_sweep_v1"


def _canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_path(payload: dict, dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"sweep path does not resolve to an object: {dotted_path}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"sweep path does not exist: {dotted_path}")
    target[parts[-1]] = value


@dataclass(frozen=True)
class GridCell:
    name: str
    parameters: dict[str, object]
    config: dict
    identity_sha256: str


@dataclass(frozen=True)
class GridCellResult:
    name: str
    parameters: dict[str, object]
    identity_sha256: str
    config_path: Path
    run_id: str
    phase: str
    status: str
    metrics: dict[str, float]
    reused: bool


@dataclass(frozen=True)
class GridSweepResult:
    name: str
    status: str
    winner_cell: str | None
    cells: tuple[GridCellResult, ...]
    leaderboard_path: Path


@dataclass(frozen=True)
class GridSweep:
    path: Path
    name: str
    base_config_path: Path
    base_config: dict
    base_config_sha256: str
    screening_stage: str
    study_root: str
    searched: tuple[tuple[str, tuple[object, ...]], ...]
    mps_workers: int
    preparation_workers: int
    hypothesis_template: str
    ranking: tuple[tuple[str, str], ...]

    def cells(self) -> tuple[GridCell, ...]:
        base = deepcopy(self.base_config)
        stages = [
            stage
            for stage in base["campaign"]["budget_stages"]
            if stage["name"] == self.screening_stage
        ]
        if len(stages) != 1:
            raise ValueError("sweep screening stage must resolve exactly once")
        keys = tuple(path for path, _ in self.searched)
        choices = tuple(values for _, values in self.searched)
        cells = []
        for index, values in enumerate(product(*choices), start=1):
            parameters = dict(zip(keys, values, strict=True))
            payload = deepcopy(base)
            for path, value in parameters.items():
                _set_path(payload, path, value)
            cell_name = f"cell-{index:02d}"
            cell_root = f"{self.study_root}/cells/{cell_name}"
            payload["output"] = cell_root
            payload["campaign"]["state_root"] = f"{cell_root}/ml-loop-state"
            payload["campaign"]["budget_stages"] = deepcopy(stages)
            payload["campaign"]["max_revisions_per_stage"] = 0
            payload["evolution"]["allowed_revision_paths"] = []
            payload["evolution"]["revision_bounds"] = {}
            payload["evolution"]["hypothesis"] = self.hypothesis_template.format(
                **{path.replace(".", "_"): value for path, value in parameters.items()}
            )
            identity = _canonical_sha256({
                "schema": SWEEP_SCHEMA,
                "study": self.name,
                "base_config_sha256": self.base_config_sha256,
                "screening_stage": self.screening_stage,
                "parameters": parameters,
                "compiled_config": payload,
            })
            cells.append(GridCell(cell_name, parameters, payload, identity))
        return tuple(cells)


def load_grid_sweep(path: str | Path) -> GridSweep:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    required = {
        "schema",
        "name",
        "base_config",
        "screening_stage",
        "study_root",
        "search_space",
        "execution",
        "hypothesis_template",
        "ranking",
    }
    if set(payload) != required or payload["schema"] != SWEEP_SCHEMA:
        raise ValueError("grid sweep contract is invalid")
    searched_payload = payload["search_space"].get("searched")
    if not isinstance(searched_payload, dict) or not searched_payload:
        raise ValueError("grid sweep searched space must be nonempty")
    searched = []
    for dotted_path, specification in searched_payload.items():
        choices = specification.get("choices") if isinstance(specification, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"grid sweep choices are invalid: {dotted_path}")
        searched.append((str(dotted_path), tuple(choices)))
    execution = payload["execution"]
    mps_workers = int(execution.get("mps_workers", 0))
    preparation_workers = int(execution.get("preparation_workers", 0))
    if mps_workers != 1 or preparation_workers < 1:
        raise ValueError("Stage 2A grid requires one MPS worker and positive preparation workers")
    raw_ranking = payload["ranking"]
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError("grid sweep ranking must be nonempty")
    ranking = []
    for rule in raw_ranking:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"metric", "direction"}
            or rule["direction"] not in {"minimize", "maximize"}
            or not str(rule["metric"]).strip()
        ):
            raise ValueError("grid sweep ranking rule is invalid")
        ranking.append((str(rule["metric"]), str(rule["direction"])))
    base_path = (path.parent / str(payload["base_config"])).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"grid sweep base config is unavailable: {base_path}")
    base_bytes = base_path.read_bytes()
    base_config = json.loads(base_bytes)
    return GridSweep(
        path=path,
        name=str(payload["name"]),
        base_config_path=base_path,
        base_config=base_config,
        base_config_sha256=hashlib.sha256(base_bytes).hexdigest(),
        screening_stage=str(payload["screening_stage"]),
        study_root=str(payload["study_root"]),
        searched=tuple(searched),
        mps_workers=mps_workers,
        preparation_workers=preparation_workers,
        hypothesis_template=str(payload["hypothesis_template"]),
        ranking=tuple(ranking),
    )


def _write_exact(path: Path, payload: Mapping) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"existing sweep artifact identity drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


def _default_state_loader(config_path: Path, run_id: str):
    from ml_training_loop.stores import JsonRunStore

    from .config import load_experiment_config

    config = load_experiment_config(config_path)
    state_root = Path(config["_root"]) / str(config["campaign"]["state_root"])
    return JsonRunStore(state_root).load(run_id)


def _default_runner(
    config_path: Path,
    *,
    run_id: str,
    stdout_path: Path,
    stderr_path: Path,
):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "propevolve.cli",
                "evolve",
                "--config",
                str(config_path),
                "--run-id",
                run_id,
            ],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    state = _default_state_loader(config_path, run_id)
    if state is None:
        raise RuntimeError(
            f"grid cell exited {process.returncode} without durable campaign state"
        )
    if process.returncode not in {0, 2}:
        raise RuntimeError(
            f"grid cell {run_id} exited unexpectedly with {process.returncode}"
        )
    return state


def _cell_metrics(state, screening_stage: str) -> dict[str, float]:
    receipts = tuple(
        receipt for receipt in state.receipts if receipt.stage == screening_stage
    )
    if len(receipts) != 1:
        raise ValueError(
            f"terminal grid cell must have one {screening_stage!r} receipt"
        )
    raw = receipts[0].outputs.get("metrics")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terminal grid cell metrics are missing")
    metrics = {str(key): float(value) for key, value in raw.items()}
    if any(value != value or abs(value) == float("inf") for value in metrics.values()):
        raise ValueError("terminal grid cell metrics must be finite")
    return metrics


def _cell_status(phase: str, metrics: Mapping[str, float]) -> str:
    if phase == "COMPLETE":
        return "COMPLETE_SAFE"
    if phase == "BLOCKED":
        return "BLOCKED"
    if float(metrics.get("training.short_circuited", 0.0)) == 1.0:
        return "TRAINING_SHORT_CIRCUIT"
    if float(metrics.get("selection.short_circuited", 0.0)) == 1.0:
        return "VALIDATION_SHORT_CIRCUIT"
    return "FAILED_GATE"


def _plan_identity(config: Mapping) -> str:
    from .orchestration import _plan

    return _plan(config).identity


def _rank_key(
    cell: GridCellResult,
    ranking: tuple[tuple[str, str], ...],
) -> tuple[object, ...]:
    values = []
    for metric, direction in ranking:
        if metric not in cell.metrics:
            raise ValueError(f"eligible grid cell is missing ranking metric: {metric}")
        value = float(cell.metrics[metric])
        values.append(value if direction == "minimize" else -value)
    return (*values, cell.name)


def run_grid_sweep(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    config_root: str | Path | None = None,
    runner=None,
    state_loader=None,
    config_validator=None,
) -> GridSweepResult:
    """Run or resume four isolated campaigns through one physical MPS slot."""
    sweep = load_grid_sweep(path)
    repository_root = sweep.base_config_path.parent.parent
    artifacts = (
        repository_root / sweep.study_root
        if artifact_root is None
        else Path(artifact_root)
    )
    configs = (
        sweep.base_config_path.parent
        if config_root is None
        else Path(config_root)
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    load_state = state_loader or _default_state_loader
    execute = runner
    cells = sweep.cells()
    prepared = []
    for cell in cells:
        config_path = configs / f".sweep-{sweep.name}-{cell.name}.json"
        _write_exact(config_path, cell.config)
        prepared.append((cell, config_path))
    if config_validator is None:
        from .config import load_experiment_config

        validate = load_experiment_config
    else:
        validate = config_validator
    with ThreadPoolExecutor(max_workers=sweep.preparation_workers) as executor:
        futures = tuple(executor.submit(validate, path) for _, path in prepared)
        for future in futures:
            future.result()
    results = []
    for cell, config_path in prepared:
        run_id = f"{sweep.name}-{cell.name}"
        expected_plan_identity = _plan_identity(cell.config)
        state = load_state(config_path, run_id)
        reused = state is not None and state.phase.value in {
            "COMPLETE",
            "FAILED_GATE",
            "STOPPED",
            "BLOCKED",
        }
        if not reused:
            if execute is None:
                state = _default_runner(
                    config_path,
                    run_id=run_id,
                    stdout_path=artifacts / "logs" / f"{cell.name}.stdout.log",
                    stderr_path=artifacts / "logs" / f"{cell.name}.stderr.log",
                )
            else:
                state = execute(config_path, run_id=run_id)
        if state is None:
            raise RuntimeError(f"grid cell produced no state: {cell.name}")
        if state.plan_identity != expected_plan_identity:
            raise ValueError(f"grid cell plan identity drifted: {cell.name}")
        phase = state.phase.value
        if phase not in {"COMPLETE", "FAILED_GATE", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"grid cell is not terminal: {cell.name} phase={phase}")
        if phase == "BLOCKED":
            raise RuntimeError(f"grid cell blocked; queue stopped: {cell.name}")
        metrics = _cell_metrics(state, sweep.screening_stage)
        results.append(GridCellResult(
            name=cell.name,
            parameters=cell.parameters,
            identity_sha256=cell.identity_sha256,
            config_path=config_path,
            run_id=run_id,
            phase=phase,
            status=_cell_status(phase, metrics),
            metrics=metrics,
            reused=reused,
        ))
    eligible = tuple(cell for cell in results if cell.status == "COMPLETE_SAFE")
    winner = (
        min(eligible, key=lambda cell: _rank_key(cell, sweep.ranking))
        if eligible else None
    )
    status = "COMPLETE" if winner is not None else "FAILED_GATE"
    body = {
        "schema": "propevolve_grid_sweep_result_v1",
        "study": sweep.name,
        "sweep_config_sha256": _file_sha256(sweep.path),
        "base_config_sha256": sweep.base_config_sha256,
        "status": status,
        "winner_cell": None if winner is None else winner.name,
        "cells": [
            {
                "name": cell.name,
                "parameters": cell.parameters,
                "identity_sha256": cell.identity_sha256,
                "config_path": str(cell.config_path),
                "run_id": cell.run_id,
                "phase": cell.phase,
                "status": cell.status,
                "metrics": cell.metrics,
                "reused": cell.reused,
            }
            for cell in results
        ],
    }
    leaderboard = {
        **body,
        "identity_sha256": _canonical_sha256(body),
    }
    leaderboard_path = artifacts / "leaderboard.json"
    rendered = json.dumps(leaderboard, indent=2, sort_keys=True) + "\n"
    temporary = leaderboard_path.with_name(f".{leaderboard_path.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(leaderboard_path)
    return GridSweepResult(
        name=sweep.name,
        status=status,
        winner_cell=None if winner is None else winner.name,
        cells=tuple(results),
        leaderboard_path=leaderboard_path,
    )


__all__ = [
    "GridCell",
    "GridCellResult",
    "GridSweep",
    "GridSweepResult",
    "SWEEP_SCHEMA",
    "load_grid_sweep",
    "run_grid_sweep",
]

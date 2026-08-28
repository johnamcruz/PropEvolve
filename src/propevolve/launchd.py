"""Generate and bootstrap one launchd job from a runtime experiment config."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
from typing import Callable, Sequence

from .config import load_experiment_config
from .orchestration import reserve_evolution_run_id


_VERSIONED_RUN_ID = re.compile(r"^(?P<stem>.+)-r(?P<revision>[1-9][0-9]*)$")


@dataclass(frozen=True)
class EvolutionLaunch:
    run_id: str
    label: str
    plist_path: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class OptunaSweepLaunch:
    run_id: str
    label: str
    plist_path: Path
    stdout_path: Path
    stderr_path: Path


def _run_launchctl(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def launch_evolution_config(
    config_path: str | Path,
    *,
    launch_agents_root: Path | None = None,
    log_root: Path | None = None,
    python_executable: Path | None = None,
    user_id: int | None = None,
    launchctl: Callable[[Sequence[str]], None] = _run_launchctl,
) -> EvolutionLaunch:
    """Validate a runtime config, reserve its next run, and launch it once."""
    config = load_experiment_config(config_path)
    root = Path(str(config["_root"]))
    resolved_config_path = Path(str(config["_path"])).resolve()
    run_id = reserve_evolution_run_id(config)
    agents = (
        Path.home() / "Library/LaunchAgents"
        if launch_agents_root is None
        else launch_agents_root
    )
    logs = (
        Path.home() / "Library/Logs/PropEvolve"
        if log_root is None
        else log_root
    )
    agents.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    match = _VERSIONED_RUN_ID.fullmatch(run_id)
    if match is None:
        raise ValueError("reserved campaign run identity is not versioned")
    stem = match.group("stem")
    revision = int(match.group("revision"))
    for revision in range(revision, revision + 10_000):
        run_id = f"{stem}-r{revision}"
        label = f"com.johnmcruz.propevolve.{run_id}"
        plist_path = agents / f"{label}.plist"
        stdout_path = logs / f"{run_id}.stdout.log"
        stderr_path = logs / f"{run_id}.stderr.log"
        if not any(
            path.exists() for path in (plist_path, stdout_path, stderr_path)
        ):
            break
    else:
        raise RuntimeError("no free versioned launch identity is available")
    interpreter = Path(sys.executable) if python_executable is None else python_executable
    payload = {
        "Label": label,
        "ProgramArguments": [
            str(interpreter),
            "-m",
            "propevolve.cli",
            "evolve",
            "--config",
            str(resolved_config_path),
            "--run-id",
            run_id,
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }
    with plist_path.open("xb") as destination:
        plistlib.dump(payload, destination, sort_keys=True)
    domain = f"gui/{os.getuid() if user_id is None else user_id}"
    launchctl(["launchctl", "bootstrap", domain, str(plist_path)])
    return EvolutionLaunch(
        run_id=run_id,
        label=label,
        plist_path=plist_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def launch_optuna_sweep_config(
    config_path: str | Path,
    *,
    launch_agents_root: Path | None = None,
    log_root: Path | None = None,
    python_executable: Path | None = None,
    user_id: int | None = None,
    launchctl: Callable[[Sequence[str]], None] = _run_launchctl,
) -> OptunaSweepLaunch:
    """Validate and launch one JSON-driven Optuna sweep through launchd."""
    from .optuna_engine import load_optuna_sweep

    sweep = load_optuna_sweep(config_path)
    resolved_config_path = sweep.path.resolve(strict=True)
    workspace = Path(str(sweep.base_config["workspace_root"])).expanduser()
    if not workspace.is_absolute():
        workspace = sweep.base_config_path.parent / workspace
    workspace = workspace.resolve(strict=True)
    agents = (
        Path.home() / "Library/LaunchAgents"
        if launch_agents_root is None
        else launch_agents_root
    )
    logs = (
        Path.home() / "Library/Logs/PropEvolve"
        if log_root is None
        else log_root
    )
    agents.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9.-]+", "-", sweep.name).strip("-.")
    if not stem:
        raise ValueError("Optuna sweep name cannot form a launch identity")
    for revision in range(1, 10_000):
        run_id = f"{stem}-r{revision}"
        label = f"com.johnmcruz.propevolve.optuna.{run_id}"
        plist_path = agents / f"{label}.plist"
        stdout_path = logs / f"{run_id}.stdout.log"
        stderr_path = logs / f"{run_id}.stderr.log"
        if not any(
            path.exists() for path in (plist_path, stdout_path, stderr_path)
        ):
            break
    else:
        raise RuntimeError("no free Optuna launch identity is available")
    interpreter = Path(sys.executable) if python_executable is None else python_executable
    payload = {
        "Label": label,
        "ProgramArguments": [
            str(interpreter),
            "-u",
            "-m",
            "propevolve.cli",
            "optuna-sweep",
            "--config",
            str(resolved_config_path),
        ],
        "WorkingDirectory": str(workspace),
        "EnvironmentVariables": {
            "PATH": (
                f"{interpreter.parent}:/opt/homebrew/bin:/usr/local/bin:"
                "/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "PYTHONPATH": str(workspace / "src"),
            "PYTHONUNBUFFERED": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }
    with plist_path.open("xb") as destination:
        plistlib.dump(payload, destination, sort_keys=True)
    domain = f"gui/{os.getuid() if user_id is None else user_id}"
    launchctl(["launchctl", "bootstrap", domain, str(plist_path)])
    return OptunaSweepLaunch(
        run_id=run_id,
        label=label,
        plist_path=plist_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


__all__ = [
    "EvolutionLaunch",
    "OptunaSweepLaunch",
    "launch_evolution_config",
    "launch_optuna_sweep_config",
]

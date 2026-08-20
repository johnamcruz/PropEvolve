from __future__ import annotations

import plistlib
from pathlib import Path
from types import SimpleNamespace

import propevolve.launchd
from propevolve.cli import main


def test_launch_evolve_cli_passes_the_runtime_config_path(monkeypatch) -> None:
    calls = []

    def fake_launch(config_path):
        calls.append(config_path)
        return SimpleNamespace(
            run_id="dynamic-campaign-r4",
            label="com.johnmcruz.propevolve.dynamic-campaign-r4",
            plist_path=Path("/tmp/dynamic-campaign-r4.plist"),
            stdout_path=Path("/tmp/dynamic-campaign-r4.stdout.log"),
            stderr_path=Path("/tmp/dynamic-campaign-r4.stderr.log"),
        )

    monkeypatch.setattr(propevolve.launchd, "launch_evolution_config", fake_launch)

    assert main(["launch-evolve", "--config", "runtime-selected.json"]) == 0
    assert calls == ["runtime-selected.json"]


def test_launch_evolution_config_generates_plist_from_passed_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "recipes" / "user-selected.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")
    state_root = tmp_path / "runs/user-selected/ml-loop-state"
    (state_root / "earlier-r3").mkdir(parents=True)
    config = {
        "_root": str(tmp_path),
        "_path": str(config_path),
        "output": "runs/user-selected",
        "campaign": {"state_root": "runs/user-selected/ml-loop-state"},
    }
    monkeypatch.setattr(
        propevolve.launchd,
        "load_experiment_config",
        lambda path: config,
    )
    calls = []

    result = propevolve.launchd.launch_evolution_config(
        config_path,
        launch_agents_root=tmp_path / "LaunchAgents",
        log_root=tmp_path / "Logs",
        python_executable=tmp_path / ".venv/bin/python",
        user_id=501,
        launchctl=lambda command: calls.append(command),
    )

    assert result.run_id == "user-selected-r4"
    assert calls == [[
        "launchctl",
        "bootstrap",
        "gui/501",
        str(result.plist_path),
    ]]
    payload = plistlib.loads(result.plist_path.read_bytes())
    assert payload["ProgramArguments"] == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "propevolve.cli",
        "evolve",
        "--config",
        str(config_path),
        "--run-id",
        "user-selected-r4",
    ]
    assert payload["WorkingDirectory"] == str(tmp_path)
    assert payload["StandardOutPath"] == str(result.stdout_path)
    assert payload["StandardErrorPath"] == str(result.stderr_path)
    assert payload["RunAtLoad"] is True


def test_launch_evolution_config_skips_stale_plist_when_state_tree_is_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "recipes" / "recovery.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    stale = agents / "com.johnmcruz.propevolve.recovery-r1.plist"
    stale.write_bytes(b"stale")
    config = {
        "_root": str(tmp_path),
        "_path": str(config_path),
        "output": "runs/recovery",
        "campaign": {"state_root": "runs/recovery/ml-loop-state"},
    }
    monkeypatch.setattr(
        propevolve.launchd,
        "load_experiment_config",
        lambda path: config,
    )

    result = propevolve.launchd.launch_evolution_config(
        config_path,
        launch_agents_root=agents,
        log_root=tmp_path / "Logs",
        python_executable=tmp_path / ".venv/bin/python",
        user_id=501,
        launchctl=lambda command: None,
    )

    assert result.run_id == "recovery-r2"
    assert stale.read_bytes() == b"stale"

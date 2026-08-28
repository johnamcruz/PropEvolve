from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import propevolve.optuna_trial as optuna_trial
from propevolve.training import HistoricalCandidateRunner


def test_direct_optuna_trial_runs_shared_training_and_validation_without_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "compiled-trial.json"
    config_path.write_text("{}\n")
    config = {
        "_root": str(tmp_path),
        "_path": str(config_path),
        "output": "trial-output",
        "training": {
            "budget_mode": "episodes",
            "episodes": 200,
            "validation_episodes": 200,
            "short_circuit": {
                "minimum_completed_episodes": 50,
                "minimum_environment_steps": 1,
            },
            "episode_coverage": {"mode": "all"},
        },
        "campaign": {
            "budget_stages": [{
                "name": "screening",
                "budget_mode": "episodes",
                "training_episodes": 12,
                "validation_episodes": 8,
                "short_circuit_minimum_episodes": 10,
                "selection_requirements": [{
                    "metric": "selection.blow_rate",
                    "operator": "==",
                    "value": 0.0,
                }],
                "warm_start_parent": False,
            }],
        },
        "evolution": {
            "base_parent": None,
            "parent_candidate_ids": [],
            "hypothesis": "select the best training configuration",
        },
    }
    monkeypatch.setattr(
        optuna_trial,
        "load_experiment_config",
        lambda _path: config,
    )
    captured: dict[str, object] = {}

    def fake_run(
        self,
        effective_config,
        *,
        parent_candidate_ids,
        hypothesis,
        collect_all_evidence=False,
    ):
        captured.update({
            "config": effective_config,
            "parents": parent_candidate_ids,
            "hypothesis": hypothesis,
            "collect_all_evidence": collect_all_evidence,
        })
        return (
            SimpleNamespace(candidate_id="candidate"),
            SimpleNamespace(
                evaluation_id="evaluation",
                status="FAIL",
                metrics={
                    "training.short_circuited": 0.0,
                    "selection.pass_rate": 0.60,
                    "selection.blow_rate": 0.0,
                },
            ),
        )

    monkeypatch.setattr(HistoricalCandidateRunner, "run", fake_run)
    result_path = tmp_path / "trial-result.json"

    result = optuna_trial.run_optuna_trial(
        config_path,
        result_path=result_path,
    )

    effective = captured["config"]
    assert effective["training"]["episodes"] == 12
    assert effective["training"]["validation_episodes"] == 8
    assert effective["training"]["short_circuit"] == {
        "minimum_completed_episodes": 10,
    }
    assert "episode_coverage" not in effective["training"]
    assert effective["_validation_stop_on_blow"] is True
    assert captured["collect_all_evidence"] is True
    assert captured["parents"] == ()
    assert not (tmp_path / "trial-output" / "ml-loop-state").exists()
    assert result["evaluation_status"] == "FAIL"
    assert result["metrics"]["selection.pass_rate"] == pytest.approx(0.60)
    assert json.loads(result_path.read_text()) == result

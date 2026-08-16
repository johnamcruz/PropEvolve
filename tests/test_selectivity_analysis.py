from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np

from propevolve.decision import Action
from propevolve.replay import (
    FinalRegimeProbeSequence,
    Transition,
    final_regime_probe_row_identity,
)
from propevolve.selectivity_analysis import analyze_selectivity_attempt


def _probe_sample(
    index: int,
    *,
    target: Action,
    prediction: Action,
    expansion: tuple[float, float, float, float],
    regime: tuple[float, float, float],
) -> tuple[FinalRegimeProbeSequence, dict[str, object]]:
    observation = np.asarray([index + 0.25], dtype=np.float32)
    teachers = np.asarray((*expansion, *regime), dtype=np.float32)
    identity = final_regime_probe_row_identity(
        ticker="NQ",
        source_decision_index=100 + index,
        target_action=target,
        observation=observation,
        teacher_target=teachers,
    )
    transition = Transition(
        observation=observation,
        action=Action.WAIT,
        reward=0.0,
        next_observation=observation,
        terminated=False,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        next_valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        teacher_target=teachers,
        entry_action_target=target,
        source_decision_index=100 + index,
    )
    sample = FinalRegimeProbeSequence(
        episode_id=f"episode-{index}",
        ticker="NQ",
        anchor_index=index,
        sequence_anchor_index=0,
        source_decision_index=100 + index,
        target_action=target,
        row_identity_sha256=identity,
        sequence=(transition,),
    )
    row = {
        "row_identity_sha256": identity,
        "ticker": "NQ",
        "source_decision_index": 100 + index,
        "target_action": target.name,
        "greedy_action": prediction.name,
        "correct": target == prediction,
    }
    return sample, row


def test_selectivity_analysis_reports_regime_expansion_side_and_economics(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    cases = (
        (Action.ENTER_LONG_1, Action.ENTER_LONG_1, (.8, .8, .1, .2), (.1, .2, .7)),
        (Action.WAIT, Action.ENTER_LONG_1, (.7, .7, .1, .2), (.8, .1, .1)),
        (Action.ENTER_SHORT_1, Action.WAIT, (.1, .2, .6, .7), (.1, .7, .2)),
        (Action.WAIT, Action.WAIT, (.1, .2, .2, .2), (.7, .2, .1)),
    )
    samples = []
    rows = []
    for index, case in enumerate(cases):
        sample, row = _probe_sample(
            index,
            target=case[0],
            prediction=case[1],
            expansion=case[2],
            regime=case[3],
        )
        samples.append(sample)
        rows.append(row)
    with (attempt / "training-policy-health-probe.pkl").open("wb") as stream:
        pickle.dump({
            "schema": "propevolve_training_policy_health_probe_corpus_v1",
            "resume_identity": "resume-1",
            "completed_episodes": 100,
            "samples": tuple(samples),
        }, stream)
    (attempt / "final-regime-probe.json").write_text(json.dumps({
        "schema": "propevolve_final_regime_probe_v1",
        "source_period": ["2021-01-01", "2025-01-01"],
        "sample_identity_sha256": "sample-1",
        "rows": rows,
        "metrics": {},
    }))
    (attempt / "training-diagnostic-summary.json").write_text(json.dumps({
        "schema": "propevolve_training_diagnostic_summary_v1",
        "overall": {
            "episodes": 10,
            "passes": 2,
            "blows": 1,
            "timeouts": 7,
            "near_blow_timeout_count": 2,
            "near_blow_timeout_rate": 2 / 7,
            "sampled_entry_action_target_counts": {
                "WAIT": 20,
                "ENTER_LONG_1": 10,
                "ENTER_SHORT_1": 10,
            },
            "sampled_entry_action_prediction_counts": {
                "WAIT": 16,
                "ENTER_LONG_1": 12,
                "ENTER_SHORT_1": 12,
            },
            "sampled_entry_action_correct_counts": {
                "WAIT": 14,
                "ENTER_LONG_1": 8,
                "ENTER_SHORT_1": 6,
            },
            "sampled_entry_action_precision": {
                "WAIT": .875,
                "ENTER_LONG_1": 2 / 3,
                "ENTER_SHORT_1": .5,
            },
            "sampled_entry_action_recall": {
                "WAIT": .7,
                "ENTER_LONG_1": .8,
                "ENTER_SHORT_1": .6,
            },
            "shadow_h50": {
                "complete_trades": 20,
                "hit_2r_before_1r_rate": .4,
                "hit_3r_before_1r_rate": .25,
                "average_mfe_r": 2.5,
                "average_mae_r": .8,
            },
        },
    }))
    config = tmp_path / "trial.json"
    config.write_text(json.dumps({
        "entry_supervision": {"target_r": 2.0, "stop_r": 1.0},
    }))
    output = tmp_path / "selectivity.json"

    report = analyze_selectivity_attempt(
        attempt,
        config_path=config,
        output_path=output,
    )

    assert report["status"] == "COMPLETE"
    assert report["probe_population"] == "balanced_final_regime_probe"
    assert report["entry_target"] == {"target_r": 2.0, "stop_r": 1.0}
    assert report["overall_probe_selectivity"] == {
        "rows": 4,
        "opportunities": 2,
        "predicted_entries": 2,
        "correct_entries": 1,
        "entry_precision": .5,
        "opportunity_recall": .5,
        "failed_setups": 2,
        "failed_setup_rejections": 1,
        "failed_setup_rejection_rate": .5,
    }
    assert set(report["by_regime_state"]) == {
        "chop_end_transition",
        "chop_no_trend",
        "expansion_trend",
    }
    assert set(report["by_expansion_quantile"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert set(report["by_side"]) == {"long", "short"}
    assert len(report["by_regime_expansion_side"]) == 4
    assert report["training_economics"]["blow_rate"] == .1
    assert report["training_economics"]["near_blow_timeout_rate"] == 2 / 7
    assert report["training_economics"]["shadow_outcomes"]["h50"][
        "hit_3r_before_1r_rate"
    ] == .25
    assert report["coverage"]["three_r_is_setup_joined"] is False
    assert json.loads(output.read_text()) == report


def test_selectivity_analysis_module_exposes_cli() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "propevolve.selectivity_analysis", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert "--attempt-dir" in result.stdout
    assert "--config" in result.stdout
    assert "--output" in result.stdout

"""Resumable corrective trade-mastery campaign for the reasoning policy."""

from collections import defaultdict
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..decision import Action
from .integrity import file_digest
from .model_config import read_recipe
from .workflow import atomic_json


_GATE_KEYS = {
    "primary_metric", "minimum_primary_improvement",
    "minimum_mean_mistake_advantage_delta",
    "minimum_retained_mastery_rate",
    "maximum_per_action_mistake_regression",
    "maximum_per_task_advantage_regression",
}


def validate_acceptance(settings):
    if (not isinstance(settings, dict) or set(settings) != _GATE_KEYS
            or not isinstance(settings["primary_metric"], str)
            or not settings["primary_metric"].strip()):
        raise ValueError("invalid corrective campaign acceptance settings")
    for name in _GATE_KEYS - {"primary_metric"}:
        value = settings[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError("invalid corrective campaign acceptance settings")
    if settings["minimum_retained_mastery_rate"] > 1:
        raise ValueError("invalid corrective campaign acceptance settings")


def _read_assessment(path):
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text())
    with (path / "scores.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if (summary.get("role") != "valid" or summary.get("weights_updated") is not False
            or summary.get("rows") != len(rows) or not rows):
        raise ValueError("candidate gate requires complete frozen validation assessments")
    indexed = {}
    for row in rows:
        key = row.get("index")
        advantage = row.get("target_advantage")
        if (type(key) is not int or key in indexed
                or isinstance(advantage, bool)
                or not isinstance(advantage, (int, float))
                or not math.isfinite(float(advantage))):
            raise ValueError("invalid frozen assessment row")
        indexed[key] = row
    return summary, indexed


def _metric(summary, name):
    value = summary.get("metrics", {}).get(name)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"frozen assessment lacks primary metric: {name}")
    return float(value)


def _task_advantages(summary):
    tasks = summary.get("metrics", {}).get("per_task")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("frozen assessment lacks per-task evidence")
    result = {}
    for name, row in tasks.items():
        value = row.get("mean_target_advantage") if isinstance(row, dict) else None
        if (not isinstance(name, str) or not name or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            raise ValueError("invalid frozen per-task evidence")
        result[name] = float(value)
    return result


def compare_frozen_assessments(before, after, settings):
    """Gate one candidate using identical chronological validation decisions."""
    validate_acceptance(settings)
    before_summary, before_rows = _read_assessment(before)
    after_summary, after_rows = _read_assessment(after)
    if set(before_rows) != set(after_rows):
        raise ValueError("frozen assessment rows differ")
    identity = ("source_id", "completed_at_ns", "ticker", "target")
    expected_actions = {action.name for action in Action}
    evidence = defaultdict(lambda: {"mistake_deltas": [], "retained": []})
    for index in sorted(before_rows):
        parent, candidate = before_rows[index], after_rows[index]
        if any(parent.get(key) != candidate.get(key) for key in identity):
            raise ValueError("frozen assessment rows differ")
        target = parent.get("target")
        if target not in expected_actions:
            raise ValueError("frozen assessment contains an unknown action")
        old, new = float(parent["target_advantage"]), float(candidate["target_advantage"])
        if old < 0:
            evidence[target]["mistake_deltas"].append((new - old, new >= 0))
        else:
            evidence[target]["retained"].append(new >= 0)
    if set(evidence) != expected_actions:
        raise ValueError("frozen assessment does not cover every legal action")
    per_action = {}
    all_mistake_deltas, all_corrected, all_retained = [], [], []
    retention_rates = []
    for action in sorted(evidence):
        mistakes = evidence[action]["mistake_deltas"]
        retained = evidence[action]["retained"]
        all_mistake_deltas.extend(delta for delta, _ in mistakes)
        all_corrected.extend(corrected for _, corrected in mistakes)
        all_retained.extend(retained)
        retention = None if not retained else float(np.mean(retained))
        if retention is not None:
            retention_rates.append(retention)
        per_action[action] = {
            "mistakes": len(mistakes),
            "mean_mistake_advantage_delta": (
                None if not mistakes else float(np.mean([delta for delta, _ in mistakes]))),
            "corrected_mistake_rate": (
                None if not mistakes else float(np.mean([flag for _, flag in mistakes]))),
            "mastered": len(retained),
            "retained_mastery_rate": retention,
        }
    if not all_mistake_deltas or not all_retained:
        raise ValueError("candidate gate requires both mistakes and mastered examples")
    primary = settings["primary_metric"]
    primary_delta = _metric(after_summary, primary) - _metric(before_summary, primary)
    before_tasks, after_tasks = _task_advantages(before_summary), _task_advantages(after_summary)
    if set(before_tasks) != set(after_tasks):
        raise ValueError("frozen assessment task evidence differs")
    task_deltas = {name: after_tasks[name] - before_tasks[name] for name in before_tasks}
    mean_mistake_delta = float(np.mean(all_mistake_deltas))
    minimum_retention = min(retention_rates)
    action_mistake_deltas = [row["mean_mistake_advantage_delta"]
        for row in per_action.values()
        if row["mean_mistake_advantage_delta"] is not None]
    failed = []
    if primary_delta < settings["minimum_primary_improvement"]:
        failed.append("primary_improvement")
    if mean_mistake_delta < settings["minimum_mean_mistake_advantage_delta"]:
        failed.append("mistake_improvement")
    if minimum_retention < settings["minimum_retained_mastery_rate"]:
        failed.append("retention")
    if min(action_mistake_deltas) < -settings["maximum_per_action_mistake_regression"]:
        failed.append("action_regression")
    if min(task_deltas.values()) < -settings["maximum_per_task_advantage_regression"]:
        failed.append("task_regression")
    return {
        "decision": "ACCEPTED" if not failed else "REJECTED",
        "failed_gates": failed,
        "primary_metric": primary,
        "primary_improvement": primary_delta,
        "mean_mistake_advantage_delta": mean_mistake_delta,
        "corrected_mistake_rate": float(np.mean(all_corrected)),
        "minimum_retained_mastery_rate": minimum_retention,
        "task_advantage_deltas": task_deltas,
        "per_action": per_action,
    }


_CAMPAIGN_KEYS = {
    "schema", "workspace_root", "state_file", "output_root",
    "initial_policy_config", "sft_template_config", "prepared_view",
    "rounds", "initial_assessments", "subset", "acceptance", "timeouts",
}


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_campaign(path):
    plan = read_recipe(path)
    if (set(plan) != _CAMPAIGN_KEYS
            or plan["schema"] != "propevolve_reasoning_corrective_campaign_v1"
            or not isinstance(plan["workspace_root"], str)
            or type(plan["rounds"]) is not int or plan["rounds"] < 1
            or not isinstance(plan["initial_assessments"], dict)
            or set(plan["initial_assessments"]) != {"train", "valid"}
            or not isinstance(plan["subset"], dict)
            or set(plan["subset"]) != {"rows_per_group", "mistake_fraction", "seed"}
            or not isinstance(plan["timeouts"], dict)
            or set(plan["timeouts"]) != {"assessment_seconds", "training_seconds"}
            or any(isinstance(plan["timeouts"][name], bool)
                   or not isinstance(plan["timeouts"][name], (int, float))
                   or plan["timeouts"][name] <= 0 for name in plan["timeouts"])):
        raise ValueError("invalid reasoning corrective campaign configuration")
    validate_acceptance(plan["acceptance"])
    from .targeted_subset import validate_targeted_sampling
    validate_targeted_sampling({**plan["subset"], "assessment_path": "pending",
        "scores_sha256": "0" * 64, "summary_sha256": "0" * 64})
    for name in ("state_file", "output_root", "initial_policy_config",
                 "sft_template_config", "prepared_view"):
        if not isinstance(plan[name], str) or not plan[name].strip():
            raise ValueError("invalid reasoning corrective campaign path")
    for descriptor in plan["initial_assessments"].values():
        if descriptor is not None and (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "scores_sha256", "summary_sha256"}):
            raise ValueError("invalid initial assessment descriptor")
    return plan


def _assessment_descriptor(path, *, role, view_manifest):
    path = Path(path)
    summary_path, scores_path = path / "summary.json", path / "scores.jsonl"
    summary = json.loads(summary_path.read_text())
    with scores_path.open() as stream:
        rows = sum(1 for line in stream if line.strip())
    if (summary.get("role") != role or summary.get("weights_updated") is not False
            or summary.get("rows") != rows or rows < 1
            or summary.get("view_manifest_sha256") != file_digest(view_manifest)):
        raise ValueError("assessment does not match its frozen role and view")
    return {"path": str(path.resolve()), "scores_sha256": file_digest(scores_path),
            "summary_sha256": file_digest(summary_path)}


def _verify_assessment(descriptor, *, role, view_manifest):
    actual = _assessment_descriptor(
        descriptor["path"], role=role, view_manifest=view_manifest)
    if actual != descriptor:
        raise ValueError("completed campaign assessment changed")
    return actual


def _campaign_identity(plan, root):
    from .mlx_sft import read_sft_config
    payload = {
        "plan": plan,
        "initial_policy": read_sft_config(
            _resolve(root, plan["initial_policy_config"]), root=root),
        "sft_template": read_sft_config(
            _resolve(root, plan["sft_template_config"]), root=root),
        "view_manifest_sha256": file_digest(
            _resolve(root, plan["prepared_view"]) / "view_manifest.json"),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_child_config(plan, root, round_root, parent_config, train_assessment):
    from .mlx_sft import read_sft_config
    template = read_sft_config(
        _resolve(root, plan["sft_template_config"]), root=root)
    parent = read_sft_config(parent_config, root=root)
    preserved = ("model", "data", "input_mode", "projector", "action_verbalizers",
                 "decision_objective", "lora_parameters", "num_layers",
                 "max_seq_length", "chat_template_kwargs")
    if any(template.get(name) != parent.get(name) for name in preserved):
        raise ValueError("corrective SFT template differs from its frozen parent")
    parent_adapter = Path(parent["adapter_path"])
    parent_metadata_path = parent_adapter / "adapter_config.json"
    parent_weights = parent_adapter / "adapters.safetensors"
    if not parent_metadata_path.is_file() or not parent_weights.is_file():
        raise ValueError("corrective campaign parent adapter is incomplete")
    parent_metadata = json.loads(parent_metadata_path.read_text())
    requirements = {name: parent_metadata.get(name) for name in (
        "input_mode", "decision_objective", "action_supervision", "projector")}
    child = dict(template)
    child.update({
        "workspace_root": str(root),
        "adapter_path": str((round_root / "candidate-adapter").resolve()),
        "resume_adapter_file": str(parent_weights.resolve()),
        "resume_adapter_requirements": requirements,
        "resume_training_state": None,
        "stage_role": "trade_mastery",
        "validation_metrics_path": str(
            (round_root / "candidate-adapter" / "validation-metrics.jsonl").resolve()),
        "targeted_sampling": {
            **plan["subset"], "assessment_path": train_assessment["path"],
            "scores_sha256": train_assessment["scores_sha256"],
            "summary_sha256": train_assessment["summary_sha256"],
        },
    })
    path = round_root / "candidate-policy.json"
    atomic_json(path, child)
    return path


class SubprocessPhases:
    """Isolate model phases so MLX memory is released between campaign steps."""

    def __init__(self, root, timeouts):
        self.root = Path(root)
        self.timeouts = timeouts

    def _run(self, command, log, timeout):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("ab") as stream:
            subprocess.run(command, cwd=self.root, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def assess(self, policy_config, view, role, output, log):
        self._run([sys.executable, "-u", str(self.root / "scripts/assess_reasoning_trade.py"),
            "--config", str(policy_config), "--view", str(view), "--role", role,
            "--output", str(output), "--root", str(self.root)], log,
            self.timeouts["assessment_seconds"])

    def train(self, config_path, view, log):
        self._run([sys.executable, "-u", "-m", "propevolve.reasoning_policy.mlx_sft",
            "--config", str(config_path), "--view", str(view),
            "--root", str(self.root), "--train"], log,
            self.timeouts["training_seconds"])


def _record_assessment(phases, campaign_state, round_state, state_path, policy,
                       view, role, output, log, key):
    if round_state.get(key) is not None:
        return _verify_assessment(round_state[key], role=role,
                                  view_manifest=view / "view_manifest.json")
    if output.exists():
        raise ValueError("unreceipted campaign assessment output already exists")
    phases.assess(policy, view, role, output, log)
    descriptor = _assessment_descriptor(
        output, role=role, view_manifest=view / "view_manifest.json")
    round_state[key] = descriptor
    atomic_json(state_path, campaign_state)
    return descriptor


def _initial_descriptor(plan, root, role, view):
    descriptor = plan["initial_assessments"][role]
    if descriptor is None:
        return None
    resolved = dict(descriptor)
    resolved["path"] = str(_resolve(root, descriptor["path"]).resolve())
    return _verify_assessment(
        resolved, role=role, view_manifest=view / "view_manifest.json")


def _verify_completed_state(state, view):
    for round_state in state["rounds"]:
        for name, role in (("parent_train", "train"), ("parent_valid", "valid"),
                           ("candidate_train", "train"), ("candidate_valid", "valid")):
            if round_state.get(name) is not None:
                _verify_assessment(round_state[name], role=role,
                    view_manifest=view / "view_manifest.json")
        config_path = round_state.get("candidate_policy_config")
        artifacts = round_state.get("candidate_artifacts")
        if artifacts is not None:
            if not isinstance(config_path, str) or not isinstance(artifacts, dict):
                raise ValueError("invalid completed candidate receipt")
            adapter = Path(json.loads(Path(config_path).read_text())["adapter_path"])
            for name, digest in artifacts.items():
                if file_digest(adapter / name) != digest:
                    raise ValueError("completed campaign candidate changed")


def run_campaign(path, *, phases=None):
    """Run or resume the reasoning trade-mastery corrective campaign."""
    plan = _read_campaign(path)
    root = Path(plan["workspace_root"]).resolve()
    output_root = _resolve(root, plan["output_root"])
    state_path = _resolve(root, plan["state_file"])
    view = _resolve(root, plan["prepared_view"])
    identity = _campaign_identity(plan, root)
    output_root.mkdir(parents=True, exist_ok=True)
    state = (json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "propevolve_reasoning_corrective_campaign_state_v1",
        "identity_sha256": identity, "status": "PENDING",
        "current_policy_config": str(
            _resolve(root, plan["initial_policy_config"]).resolve()),
        "rounds": [],
    })
    if state.get("identity_sha256") != identity:
        raise ValueError("reasoning campaign configuration or input identity changed")
    _verify_completed_state(state, view)
    if state.get("status") in {"COMPLETE", "FAILED_GATE"}:
        return state
    if phases is None:
        phases = SubprocessPhases(root, plan["timeouts"])
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("reasoning corrective campaign is already running") from error
        try:
            state["status"] = "RUNNING"
            state.pop("error", None)
            atomic_json(state_path, state)
            for index in range(plan["rounds"]):
                round_root = output_root / f"round-{index + 1:02d}"
                round_root.mkdir(parents=True, exist_ok=True)
                if index < len(state["rounds"]):
                    round_state = state["rounds"][index]
                    if round_state.get("decision") == "ACCEPTED":
                        continue
                    parent_policy = Path(round_state["parent_policy_config"])
                else:
                    parent_policy = Path(state["current_policy_config"])
                    round_state = {"round": index + 1,
                        "parent_policy_config": str(parent_policy),
                        "parent_train": None, "parent_valid": None,
                        "candidate_policy_config": None,
                        "candidate_artifacts": None,
                        "candidate_train": None, "candidate_valid": None,
                        "decision": None}
                    if index == 0:
                        round_state["parent_train"] = _initial_descriptor(
                            plan, root, "train", view)
                        round_state["parent_valid"] = _initial_descriptor(
                            plan, root, "valid", view)
                    else:
                        prior = state["rounds"][index - 1]
                        round_state["parent_train"] = prior["candidate_train"]
                        round_state["parent_valid"] = prior["candidate_valid"]
                    state["rounds"].append(round_state)
                    atomic_json(state_path, state)
                train_assessment = _record_assessment(
                    phases, state, round_state, state_path, parent_policy, view, "train",
                    round_root / "parent-train-assessment",
                    round_root / "parent-train-assessment.log", "parent_train")
                valid_assessment = _record_assessment(
                    phases, state, round_state, state_path, parent_policy, view, "valid",
                    round_root / "parent-valid-assessment",
                    round_root / "parent-valid-assessment.log", "parent_valid")
                if round_state["candidate_policy_config"] is None:
                    child_config = _write_child_config(
                        plan, root, round_root, parent_policy, train_assessment)
                    round_state["candidate_policy_config"] = str(child_config.resolve())
                    atomic_json(state_path, state)
                else:
                    child_config = Path(round_state["candidate_policy_config"])
                candidate_adapter = round_root / "candidate-adapter"
                required = ("adapters.safetensors", "projector.safetensors",
                            "training_selection.json", "adapter_config.json",
                            "targeted_sampling_receipt.json")
                if round_state["candidate_artifacts"] is None:
                    if candidate_adapter.exists():
                        raise ValueError("unreceipted candidate adapter already exists")
                    phases.train(child_config, view, round_root / "candidate-training.log")
                    if any(not (candidate_adapter / name).is_file() for name in required):
                        raise ValueError("candidate SFT did not produce a complete adapter")
                    round_state["candidate_artifacts"] = {
                        name: file_digest(candidate_adapter / name) for name in required}
                    atomic_json(state_path, state)
                _record_assessment(
                    phases, state, round_state, state_path, child_config, view, "train",
                    round_root / "candidate-train-assessment",
                    round_root / "candidate-train-assessment.log", "candidate_train")
                candidate_valid = _record_assessment(
                    phases, state, round_state, state_path, child_config, view, "valid",
                    round_root / "candidate-valid-assessment",
                    round_root / "candidate-valid-assessment.log", "candidate_valid")
                comparison = compare_frozen_assessments(
                    valid_assessment["path"], candidate_valid["path"],
                    plan["acceptance"])
                atomic_json(round_root / "comparison.json", comparison)
                round_state.update(comparison)
                atomic_json(state_path, state)
                if comparison["decision"] != "ACCEPTED":
                    state["status"] = "FAILED_GATE"
                    state["selected_policy_config"] = str(parent_policy)
                    atomic_json(state_path, state)
                    return state
                state["current_policy_config"] = str(child_config.resolve())
                state["selected_policy_config"] = str(child_config.resolve())
                atomic_json(state_path, state)
            state["status"] = "COMPLETE"
            state.pop("error", None)
            atomic_json(state_path, state)
            return state
        except (Exception, KeyboardInterrupt) as error:
            state["status"] = "BLOCKED"
            state["error"] = str(error)
            atomic_json(state_path, state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_campaign(args.config), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

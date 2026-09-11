"""Sequential, explicit challenger stages in isolated processes.

It runs the reasoning-policy commands
and stops on the first failing stage. Resume skips only receipts whose configured
inputs and outputs still match. It never generates a passing scientific audit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .integrity import file_digest
from .model_config import read_recipe


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def file_identities(root, paths):
    return {name: file_digest(root / name) for name in paths}


def stage_inputs(root, step):
    """Include inherited recipe files and source audits, not merely top-level IDs."""
    result = file_identities(root, step["inputs"])
    visited = set()
    def recipe(path):
        path = Path(path).resolve()
        if path in visited:
            return
        visited.add(path)
        result[str(path)] = file_digest(path)
        payload = json.loads(path.read_text())
        if payload.get("inherits") is not None:
            recipe(path.parent / payload["inherits"])
    job_path = root / step["job_config"]
    recipe(job_path)
    job = read_recipe(job_path)
    for key in {
        "collect": ("source_recipe", "temporal_split_audit", "context_config"),
        "audit": (), "prepare": ("sft_config",), "train": ("sft_config",),
        "rl": ("source_recipe", "temporal_split_audit", "context_config", "rl_config", "sft_config"),
        "evaluate": ("source_recipe", "temporal_split_audit", "context_config", "evaluation_policy_config", "evaluation_metrics_config"),
    }[step["stage"]]:
        if not job.get(key):
            raise ValueError(f"missing stage configuration: {key}")
        recipe(root / job[key])
    if step["stage"] == "evaluate" and job.get("policy_config") is not None:
        policy_path = root / job["policy_config"]
        recipe(policy_path)
        policy = read_recipe(policy_path)
        if policy.get("kind") == "reasoning":
            recipe(root / policy["model_config"])
        else:
            raise ValueError("evaluation workflow requires a reasoning policy")
    if step["stage"] in {"collect", "rl", "evaluate"} and job.get("volume_source") is not None:
        for value in job["volume_source"].values():
            recipe(root / value)
    return result


def run_workflow(path):
    plan = read_recipe(path)
    root = Path(plan["workspace_root"]).resolve()
    state_path = root / plan["state_file"]
    steps = plan["steps"]
    names = [step["id"] for step in steps]
    if not steps or len(names) != len(set(names)):
        raise ValueError("workflow needs unique stage IDs")
    for step in steps:
        if step["stage"] not in {"collect", "audit", "prepare", "train", "rl", "evaluate"}:
            raise ValueError("unsupported challenger stage")
        if not step["outputs"] or not step["inputs"]:
            raise ValueError("stage input and output receipts must be declared")
        if type(step["timeout_seconds"]) not in (int, float) or step["timeout_seconds"] <= 0:
            raise ValueError("stage timeout must be positive")
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "plan": plan, "completed": {}, "status": "PENDING"}
    if state["plan"] != plan:
        raise ValueError("workflow plan changed; use a new state file")
    lock = state_path.with_suffix(state_path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    # macOS flock releases on process exit; stale PID files do not block resumes.
    import fcntl
    with lock.open("a") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("workflow is already running") from exc
        for step in steps:
            name = step["id"]
            try:
                inputs = stage_inputs(root, step)
                previous = state["completed"].get(name)
                if previous is not None:
                    if previous != {"inputs": inputs, "outputs": file_identities(root, step["outputs"])}:
                        raise ValueError("completed stage artifacts changed; do not silently reuse")
                    continue
                state.update(status="RUNNING", active_stage=name)
                atomic_json(state_path, state)
                log = root / step["log"]
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("ab") as stream:
                    subprocess.run([sys.executable, "-m", "propevolve.reasoning_policy.job",
                        "--config", str(root / step["job_config"]), step["stage"]],
                        cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True,
                        timeout=step["timeout_seconds"])
                if stage_inputs(root, step) != inputs:
                    raise ValueError("stage inputs changed during execution")
                state["completed"][name] = {"inputs": inputs, "outputs": file_identities(root, step["outputs"])}
                atomic_json(state_path, state)
            except (Exception, KeyboardInterrupt) as exc:
                state.update(status="BLOCKED", active_stage=name, error=str(exc))
                atomic_json(state_path, state)
                raise
        state.update(status="COMPLETE", active_stage=None)
        state.pop("error", None)
        # COMPLETE means these commands finished, not a promoted economic model.
        atomic_json(state_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_workflow(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

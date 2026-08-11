"""Matched microbenchmark for PropEvolve's recurrent training update."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


@dataclass(frozen=True)
class RuntimeArm:
    name: str
    mixed_precision: str
    mps_prefer_metal: bool
    compile_model: bool


def runtime_benchmark_arms() -> tuple[RuntimeArm, ...]:
    """Return the frozen four-arm runtime comparison in execution order."""
    return (
        RuntimeArm("eager_fp32", "off", False, False),
        RuntimeArm("fp16_autocast", "fp16", False, False),
        RuntimeArm("fp16_metal_matmul", "fp16", True, False),
        RuntimeArm("fp16_compile", "fp16", False, True),
    )


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _synthetic_sequences(
    *,
    batch_sequences: int,
    sequence_length: int,
    observation_dim: int,
    teacher_channels: int,
    seed: int,
):
    from .decision import Action
    from .replay import Transition

    rng = np.random.default_rng(seed)
    valid = tuple(Action)
    sequences = []
    for batch_index in range(batch_sequences):
        observations = rng.normal(
            size=(sequence_length + 1, observation_dim)
        ).astype(np.float32)
        sequence = []
        for time_index in range(sequence_length):
            teacher = (
                rng.uniform(size=teacher_channels).astype(np.float32)
                if teacher_channels
                else None
            )
            sequence.append(Transition(
                observation=observations[time_index],
                action=valid[(batch_index + time_index) % len(valid)],
                reward=float(rng.normal(scale=0.1)),
                next_observation=observations[time_index + 1],
                terminated=time_index == sequence_length - 1,
                valid_actions=valid,
                next_valid_actions=valid,
                teacher_target=teacher,
            ))
        sequences.append(tuple(sequence))
    return tuple(sequences)


def run_arm(
    config_path: str | Path,
    arm: RuntimeArm,
    *,
    observation_dim: int,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, object]:
    """Measure one arm in the current isolated process."""
    from .agent import RecurrentC51Agent
    from .config import agent_runtime_settings, load_experiment_config

    config = load_experiment_config(config_path)
    agent_settings = dict(config["agent"])
    runtime = dict(config["runtime"])
    runtime.update(
        mixed_precision=arm.mixed_precision,
        mps_prefer_metal=arm.mps_prefer_metal,
        compile_model=arm.compile_model,
    )
    agent_settings.update(agent_runtime_settings(runtime))
    teachers = tuple(config.get("teachers") or ())
    teacher_channels = sum(len(item["channels"]) for item in teachers)
    if teacher_channels:
        agent_settings.update(
            teacher_channels=teacher_channels,
            teacher_loss_weight=sum(float(item["loss_weight"]) for item in teachers),
            teacher_channel_loss_weights=tuple(
                float(item["loss_weight"]) / len(item["channels"])
                for item in teachers
                for _ in item["channels"]
            ),
            teacher_entry_search_loss_weight=float(
                teachers[0].get("entry_search_loss_weight", 0.0)
            ),
        )
    agent = RecurrentC51Agent(
        observation_dim,
        seed=int(config["training"]["seed"]),
        **agent_settings,
    )
    batch = _synthetic_sequences(
        batch_sequences=int(config["training"]["batch_sequences"]),
        sequence_length=int(config["training"]["sequence_length"]),
        observation_dim=observation_dim,
        teacher_channels=teacher_channels,
        seed=int(config["training"]["seed"]),
    )
    for _ in range(warmup_updates):
        agent.train_batch(batch)
    _synchronize(agent.device.type)
    started = time.perf_counter()
    losses = [agent.train_batch(batch) for _ in range(measured_updates)]
    _synchronize(agent.device.type)
    elapsed = time.perf_counter() - started
    if not np.isfinite(losses).all():
        raise RuntimeError(f"runtime arm {arm.name} produced non-finite loss")
    return {
        "schema": "propevolve_runtime_benchmark_arm_v1",
        "arm": asdict(arm),
        "device": agent.device.type,
        "compile_status": agent.compile_status,
        "compile_error": agent.compile_error,
        "updates": measured_updates,
        "seconds": elapsed,
        "milliseconds_per_update": 1000.0 * elapsed / measured_updates,
        "updates_per_second": measured_updates / elapsed,
        "final_loss": losses[-1],
    }


def run_benchmark(
    config_path: str | Path,
    *,
    observation_dim: int | None,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, object]:
    """Run each arm in a fresh process so Metal environment flags are real."""
    from .config import load_experiment_config

    if observation_dim is None:
        config = load_experiment_config(config_path)
        root = Path(config["_root"])
        cache_root = Path(config["cache_root"])
        if not cache_root.is_absolute():
            cache_root = root / cache_root
        manifest = json.loads(
            (cache_root / config["tickers"][0] / "manifest.json").read_text()
        )
        observation_dim = int(manifest["embedding_dim"]) + 12
    results = []
    for arm in runtime_benchmark_arms():
        environment = dict(os.environ)
        environment["PYTORCH_MPS_PREFER_METAL"] = (
            "1" if arm.mps_prefer_metal else "0"
        )
        environment["PYTORCH_MPS_FAST_MATH"] = "0"
        command = [
            sys.executable,
            "-m",
            "propevolve.runtime_benchmark",
            "--child",
            "--config",
            str(Path(config_path).resolve()),
            "--arm",
            arm.name,
            "--observation-dim",
            str(observation_dim),
            "--warmup-updates",
            str(warmup_updates),
            "--measured-updates",
            str(measured_updates),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))
    baseline = float(results[0]["milliseconds_per_update"])
    baseline_loss = float(results[0]["final_loss"])
    tolerance = float(
        load_experiment_config(config_path)["runtime"][
            "benchmark_max_relative_loss_drift"
        ]
    )
    for result in results:
        relative_loss_drift = abs(float(result["final_loss"]) - baseline_loss) / max(
            abs(baseline_loss), 1e-8
        )
        compile_supported = not (
            bool(result["arm"]["compile_model"])
            and result["compile_status"] != "compiled"
        )
        result["speedup_vs_eager_fp32"] = (
            baseline / float(result["milliseconds_per_update"])
        )
        result["relative_loss_drift_vs_eager_fp32"] = relative_loss_drift
        result["numerically_valid"] = relative_loss_drift <= tolerance
        result["runtime_supported"] = compile_supported
        result["eligible"] = result["numerically_valid"] and compile_supported
    return {
        "schema": "propevolve_runtime_benchmark_v1",
        "config": str(Path(config_path).resolve()),
        "observation_dim": observation_dim,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "maximum_relative_loss_drift": tolerance,
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--observation-dim", type=int)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measured-updates", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arm")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        (args.observation_dim is not None and args.observation_dim < 1)
        or args.warmup_updates < 1
        or args.measured_updates < 1
    ):
        raise ValueError("benchmark dimensions and update counts must be positive")
    if args.child:
        if args.observation_dim is None:
            raise ValueError("child runtime benchmark requires observation dimension")
        arms = {arm.name: arm for arm in runtime_benchmark_arms()}
        if args.arm not in arms:
            raise ValueError("unknown runtime benchmark arm")
        payload = run_arm(
            args.config,
            arms[args.arm],
            observation_dim=args.observation_dim,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
        )
    else:
        payload = run_benchmark(
            args.config,
            observation_dim=args.observation_dim,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
        )
    rendered = json.dumps(
        payload,
        indent=None if args.child else 2,
        sort_keys=True,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    learner_backend: str = "pytorch"


def runtime_benchmark_arms() -> tuple[RuntimeArm, ...]:
    """Return the frozen four-arm runtime comparison in execution order."""
    return (
        RuntimeArm("eager_fp32", "off", False, False),
        RuntimeArm("fp16_autocast", "fp16", False, False),
        RuntimeArm("fp16_metal_matmul", "fp16", True, False),
        RuntimeArm("fp16_compile", "fp16", False, True),
    )


def backend_benchmark_arms() -> tuple[RuntimeArm, ...]:
    """Compare the one shared learner with only its tensor backend changed."""
    return (
        RuntimeArm("pytorch_mps", "off", False, False, "pytorch"),
        RuntimeArm("mlx_mps", "off", False, False, "mlx"),
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


def _authenticated_replay_sequences(
    path: str | Path,
    *,
    batch_sequences: int,
):
    """Sample a production batch from a bounded causal replay artifact."""
    import torch

    from .replay import BalancedSequenceReplay

    artifact = Path(path).resolve()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if payload.get("schema") != "propevolve_balance_pass_replay_v1":
        raise ValueError("benchmark replay schema is invalid")
    sources = payload.get("source_checkpoints")
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(source, dict)
            or not isinstance(source.get("causal_identity_sha256"), str)
            or len(source["causal_identity_sha256"]) != 64
            or not isinstance(source.get("resume_identity"), str)
            or not source["resume_identity"]
            for source in sources
        )
    ):
        raise ValueError("benchmark replay source identity is invalid")
    state = payload.get("replay_state")
    if not isinstance(state, dict) or not isinstance(state.get("contract"), dict):
        raise ValueError("benchmark replay state is invalid")
    contract = state["contract"]
    replay = BalancedSequenceReplay(
        capacity_episodes=int(contract["capacity_episodes"]),
        capacity_transitions=contract["capacity_transitions"],
        sequence_length=int(contract["sequence_length"]),
        recurrent_burn_in=int(contract["recurrent_burn_in"]),
        n_step_return=int(contract["n_step_return"]),
        terminal_sequence_fraction=float(contract["terminal_sequence_fraction"]),
        safety_sequence_fraction=float(contract["safety_sequence_fraction"]),
        entry_opportunity_sequence_fraction=float(
            contract["entry_opportunity_sequence_fraction"]
        ),
        regime_wait_sequence_fraction=float(
            contract["regime_wait_sequence_fraction"]
        ),
        regime_wait_sequence_update_period=int(
            contract["regime_wait_sequence_update_period"]
        ),
        entry_opportunity_side_balance=str(
            contract["entry_opportunity_side_balance"]
        ),
        paired_a_plus_population_weighting=str(
            contract["paired_a_plus_population_weighting"]
        ),
        paired_a_plus_context_matching=str(
            contract.get(
                "paired_a_plus_context_matching",
                "static_expansion_regime_v1",
            )
        ),
        seed=0,
    )
    replay.load_state_dict(state)
    sequences = replay.sample(batch_sequences)
    if len(sequences) != batch_sequences:
        raise ValueError("benchmark replay cannot provide the requested batch")
    return sequences


def _memory_metrics(agent) -> dict[str, int]:
    import torch

    metrics: dict[str, int] = {}
    if agent.device.type == "mps":
        metrics.update(
            torch_active_memory_bytes=int(torch.mps.current_allocated_memory()),
            torch_driver_memory_bytes=int(torch.mps.driver_allocated_memory()),
        )
    if agent.learner_backend == "mlx":
        from .mlx_backend import mlx_memory_metrics

        metrics.update(
            {f"mlx_{key}": value for key, value in mlx_memory_metrics().items()}
        )
    return metrics


def run_arm(
    config_path: str | Path,
    arm: RuntimeArm,
    *,
    observation_dim: int,
    warmup_updates: int,
    measured_updates: int,
    replay_artifact: str | Path | None = None,
) -> dict[str, object]:
    """Measure one arm in the current isolated process."""
    from .agent import RecurrentC51Agent
    from .config import agent_runtime_settings, load_experiment_config
    from .teachers import agent_teacher_settings

    config = load_experiment_config(config_path)
    agent_settings = dict(config["agent"])
    runtime = dict(config["runtime"])
    runtime.update(
        learner_backend=arm.learner_backend,
        mixed_precision=arm.mixed_precision,
        mps_prefer_metal=arm.mps_prefer_metal,
        compile_model=arm.compile_model,
    )
    agent_settings.update(agent_runtime_settings(runtime))
    teachers = tuple(config.get("teachers") or ())
    teacher_channels = sum(len(item["channels"]) for item in teachers)
    if teacher_channels:
        agent_settings.update(agent_teacher_settings(teachers))
    agent = RecurrentC51Agent(
        observation_dim,
        seed=int(config["training"]["seed"]),
        **agent_settings,
    )
    if replay_artifact is None:
        batch = _synthetic_sequences(
            batch_sequences=int(config["training"]["batch_sequences"]),
            sequence_length=int(config["training"]["sequence_length"]),
            observation_dim=observation_dim,
            teacher_channels=teacher_channels,
            seed=int(config["training"]["seed"]),
        )
    else:
        batch = _authenticated_replay_sequences(
            replay_artifact,
            batch_sequences=int(config["training"]["batch_sequences"]),
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
        "memory": _memory_metrics(agent),
    }


def _run_isolated_arms(
    config_path: str | Path,
    *,
    arms: tuple[RuntimeArm, ...],
    observation_dim: int,
    warmup_updates: int,
    measured_updates: int,
    backend_comparison: bool,
    replay_artifact: str | Path | None = None,
) -> list[dict[str, object]]:
    results = []
    for arm in arms:
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
        if backend_comparison:
            command.append("--backend-comparison")
        if replay_artifact is not None:
            command.extend(
                ["--replay-artifact", str(Path(replay_artifact).resolve())]
            )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))
    return results


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
        from .observation import TradeManagementObservationSpec

        root = Path(config["_root"])
        cache_root = Path(config["cache_root"])
        if not cache_root.is_absolute():
            cache_root = root / cache_root
        manifest = json.loads(
            (cache_root / config["tickers"][0] / "manifest.json").read_text()
        )
        management = TradeManagementObservationSpec.from_config(
            config.get("observation")
        )
        observation_dim = int(manifest["embedding_dim"]) + 12 + management.output_dim
    results = _run_isolated_arms(
        config_path,
        arms=runtime_benchmark_arms(),
        observation_dim=observation_dim,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        backend_comparison=False,
    )
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


def run_backend_benchmark(
    config_path: str | Path,
    *,
    replay_artifact: str | Path,
    observation_dim: int,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, object]:
    """Benchmark Torch and MLX on one authenticated production replay batch."""
    from .config import load_experiment_config

    replay_path = Path(replay_artifact).resolve()
    results = _run_isolated_arms(
        config_path,
        arms=backend_benchmark_arms(),
        observation_dim=observation_dim,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        backend_comparison=True,
        replay_artifact=replay_path,
    )
    baseline_time = float(results[0]["milliseconds_per_update"])
    baseline_loss = float(results[0]["final_loss"])
    tolerance = float(
        load_experiment_config(config_path)["runtime"][
            "benchmark_max_relative_loss_drift"
        ]
    )
    for result in results:
        loss_drift = abs(float(result["final_loss"]) - baseline_loss) / max(
            abs(baseline_loss), 1e-8
        )
        result["speedup_vs_pytorch"] = (
            baseline_time / float(result["milliseconds_per_update"])
        )
        result["relative_loss_drift_vs_pytorch"] = loss_drift
        result["numerically_valid"] = loss_drift <= tolerance
    return {
        "schema": "propevolve_backend_benchmark_v1",
        "config": str(Path(config_path).resolve()),
        "replay_artifact": str(replay_path),
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
    parser.add_argument("--backend-comparison", action="store_true")
    parser.add_argument("--replay-artifact")
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
        arm_source = (
            backend_benchmark_arms()
            if args.backend_comparison
            else runtime_benchmark_arms()
        )
        arms = {arm.name: arm for arm in arm_source}
        if args.arm not in arms:
            raise ValueError("unknown runtime benchmark arm")
        payload = run_arm(
            args.config,
            arms[args.arm],
            observation_dim=args.observation_dim,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
            replay_artifact=args.replay_artifact,
        )
    elif args.backend_comparison:
        if args.observation_dim is None or args.replay_artifact is None:
            raise ValueError(
                "backend benchmark requires observation dimension and replay artifact"
            )
        payload = run_backend_benchmark(
            args.config,
            replay_artifact=args.replay_artifact,
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

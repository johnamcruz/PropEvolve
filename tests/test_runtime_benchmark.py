from __future__ import annotations

import json
from types import SimpleNamespace

from propevolve.runtime_benchmark import run_benchmark, runtime_benchmark_arms
from tests.recipe_fixtures import paired_aplus_recipe


def test_runtime_benchmark_uses_exact_frozen_four_arm_comparison() -> None:
    arms = runtime_benchmark_arms()

    assert [arm.name for arm in arms] == [
        "eager_fp32",
        "fp16_autocast",
        "fp16_metal_matmul",
        "fp16_compile",
    ]
    assert [arm.mixed_precision for arm in arms] == [
        "off", "fp16", "fp16", "fp16"
    ]
    assert [arm.mps_prefer_metal for arm in arms] == [
        False, False, True, False
    ]
    assert [arm.compile_model for arm in arms] == [
        False, False, False, True
    ]


def test_runtime_benchmark_orchestrates_four_isolated_arms_and_gates_results(
    monkeypatch,
) -> None:
    calls = []

    def completed(command, **kwargs):
        arm_name = command[command.index("--arm") + 1]
        calls.append((arm_name, kwargs["env"]["PYTORCH_MPS_PREFER_METAL"]))
        compile_model = arm_name == "fp16_compile"
        payload = {
            "schema": "propevolve_runtime_benchmark_arm_v1",
            "arm": {"name": arm_name, "compile_model": compile_model},
            "compile_status": "fallback_eager" if compile_model else "disabled",
            "final_loss": 1.01 if arm_name == "fp16_metal_matmul" else 1.0,
            "milliseconds_per_update": 10.0,
        }
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr("subprocess.run", completed)

    report = run_benchmark(
        paired_aplus_recipe(100),
        observation_dim=8,
        warmup_updates=1,
        measured_updates=2,
    )

    assert calls == [
        ("eager_fp32", "0"),
        ("fp16_autocast", "0"),
        ("fp16_metal_matmul", "1"),
        ("fp16_compile", "0"),
    ]
    assert [result["numerically_valid"] for result in report["results"]] == [
        True, True, True, True
    ]
    assert [result["eligible"] for result in report["results"]] == [
        True, True, True, False
    ]

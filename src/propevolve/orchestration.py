"""AlphaEvolve-inspired PropEvolve campaign on the shared ML Training Loop."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Protocol

from ml_training_loop import (
    Decision,
    GateResult,
    Phase,
    ReasoningOutcome,
    Revision,
    StageReceipt,
    StageSpec,
    TrainingLoop,
    TrainingPlan,
)
from ml_training_loop.adapters import DictAdapterRegistry
from ml_training_loop.cli import DEFAULT_BUNDLE
from ml_training_loop.interfaces import (
    GateRequest,
    ReasoningAdapter,
    ReasoningRequest,
    SkillBootstrapper,
    StageRequest,
    SurrogateAdvisor,
)
from ml_training_loop.integrations import (
    CodexCliReasoningAdapter,
    SubprocessCodexExecutor,
)
from ml_training_loop.skills import BundledSkillBootstrapper
from ml_training_loop.stores import JsonRunStore

from .config import load_experiment_config
from .evolution import CandidateArchive, ModelRegistry, Niche, RevisionPolicy
from .reasoning import GepaReflectiveReasoningAdapter


_STAGE_ADAPTER = "propevolve-candidate"


def reserve_evolution_run_id(config: Mapping[str, object]) -> str:
    """Atomically reserve the next revision for one frozen campaign config."""
    root = Path(str(config["_root"]))
    state_root = root / str(config["campaign"]["state_root"])
    state_root.mkdir(parents=True, exist_ok=True)
    stem = re.sub(
        r"[^A-Za-z0-9]+",
        "-",
        Path(str(config["output"])).name,
    ).strip("-")
    if not stem:
        raise ValueError("campaign output must provide a run identity stem")
    revision_pattern = re.compile(r"(?:^|-)r([1-9][0-9]*)(?:-|$)")
    revision = 1 + max(
        (
            int(match.group(1))
            for entry in state_root.iterdir()
            if (match := revision_pattern.search(entry.name)) is not None
        ),
        default=0,
    )
    while True:
        run_id = f"{stem}-r{revision}"
        try:
            (state_root / run_id).mkdir(exist_ok=False)
        except FileExistsError:
            revision += 1
            continue
        return run_id

_EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS = (
    "assets",
    "cache",
    "cache_root",
    "tickers",
    "deployment_tickers",
    "training_only_tickers",
    "timeframe_minutes",
    "temporal",
    "point_values",
    "round_trip_fees",
    "sealed_confirmation",
    "entry_supervision",
)
_EXTERNAL_PARENT_TRAINING_ONLY_RECIPE_PATHS = (
    "entry_supervision.loss_weight",
    "entry_supervision.opportunity_loss_multiplier",
)
_EXTERNAL_PARENT_ECONOMIC_FIELDS = (
    "profit_target",
    "max_loss",
    "episode_days",
    "bars_per_day",
    "max_position_size",
    "minimum_mll_headroom",
    "trailing_mll_lock",
    "per_trade_risk_dollars",
    "ratchet_activation_r",
    "ratchet_giveback_r",
    "ratchet_lock_floor_r",
)
_GATE_ADAPTER = "propevolve-economic-evidence"
_DEFAULT_REASONING = object()


def _resolve_codex_executable(reasoning: Mapping) -> Path:
    """Resolve Codex when unattended launchd jobs lack the editor's PATH."""

    explicit = reasoning.get("executable") or os.environ.get(
        "PROPEVOLVE_CODEX_EXECUTABLE"
    )
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(
                f"configured Codex executable is unavailable: {path}"
            )
        return path

    discovered = shutil.which("codex")
    if discovered:
        return Path(discovered).resolve()

    home = Path.home()
    candidates = tuple(
        path
        for pattern in (
            ".vscode/extensions/openai.chatgpt-*/bin/*/codex",
            ".vscode-insiders/extensions/openai.chatgpt-*/bin/*/codex",
            ".cursor/extensions/openai.chatgpt-*/bin/*/codex",
        )
        for path in home.glob(pattern)
        if path.is_file() and os.access(path, os.X_OK)
    )
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    raise FileNotFoundError(
        "Codex executable is unavailable; install Codex, add it to PATH, or set "
        "PROPEVOLVE_CODEX_EXECUTABLE"
    )


class CandidateRunner(Protocol):
    def run(
        self,
        config: dict,
        *,
        parent_candidate_ids: tuple[str, ...],
        hypothesis: str,
    ): ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_value(config: Mapping, path: str) -> object:
    value: object = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"frozen recipe path does not exist: {path}")
        value = value[part]
    return value


def _public_config(config: Mapping) -> dict:
    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def _budget_stage_fields(stage: Mapping) -> dict[str, object]:
    """Project one validated budget without mixing episode and step units."""
    mode = str(stage.get("budget_mode", "environment_steps"))
    if mode == "environment_steps":
        return {
            "minimum_environment_steps": int(
                stage["minimum_environment_steps"]
            ),
        }
    if mode == "episodes":
        fields: dict[str, object] = {
            "budget_mode": mode,
            "training_episodes": int(stage["training_episodes"]),
            "validation_episodes": int(stage["validation_episodes"]),
        }
        if stage.get("short_circuit_minimum_episodes") is not None:
            fields["short_circuit_minimum_episodes"] = int(
                stage["short_circuit_minimum_episodes"]
            )
        if stage.get("episode_coverage") is not None:
            fields["episode_coverage"] = dict(stage["episode_coverage"])
        return fields
    raise ValueError(f"unsupported campaign budget mode: {mode}")


def _assert_parent_causal_contract(candidate, config: Mapping) -> None:
    parent_contract = json.loads((candidate.path / "contract.json").read_text())
    parent_recipe = json.loads((candidate.path / "recipe.json").read_text())
    expected_contract = {
        "training_tickers": list(config["tickers"]),
        "deployment_tickers": list(config["deployment_tickers"]),
        "training_only_tickers": list(config["training_only_tickers"]),
        "temporal": dict(config["temporal"]),
        "sealed_holdout_touched": False,
    }
    if any(
        parent_contract.get(field) != expected
        for field, expected in expected_contract.items()
    ):
        raise ValueError("external parent causal contract drifted")
    # Teachers supervise training only; Stage 2 may replace them while the
    # deployed parent's causal observation and economic contracts stay fixed.
    def same_json(left: object, right: object) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left == right
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                same_json(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                same_json(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        return type(left) is type(right) and left == right

    def causal_recipe_value(recipe: Mapping, path: str) -> object:
        value = recipe[path]
        if path == "entry_supervision":
            return {
                key: item
                for key, item in value.items()
                if f"{path}.{key}"
                not in _EXTERNAL_PARENT_TRAINING_ONLY_RECIPE_PATHS
            }
        return value

    for path in _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS:
        parent_present = path in parent_recipe
        child_present = path in config
        if parent_present != child_present or (
            parent_present
            and not same_json(
                causal_recipe_value(parent_recipe, path),
                causal_recipe_value(config, path),
            )
        ):
            raise ValueError(
                f"external parent causal recipe drifted at {path}"
            )
    economic_overrides = set(
        config["evolution"].get("external_parent_economic_overrides", ())
    )
    for field in _EXTERNAL_PARENT_ECONOMIC_FIELDS:
        parent_present = field in parent_recipe["challenge"]
        child_present = field in config["challenge"]
        if field in economic_overrides:
            if (
                not parent_present
                or not child_present
                or parent_recipe["challenge"][field]
                == config["challenge"][field]
            ):
                raise ValueError(
                    "external parent economic override is inactive at "
                    f"challenge.{field}"
                )
            continue
        if parent_present != child_present or (
            parent_present
            and parent_recipe["challenge"][field] != config["challenge"][field]
        ):
            raise ValueError(
                f"external parent economic contract drifted at challenge.{field}"
            )


def _revision_policy(config: Mapping) -> RevisionPolicy:
    bounds = config["evolution"].get("revision_bounds", {})
    return RevisionPolicy(
        tuple(config["evolution"]["allowed_revision_paths"]),
        tuple(config["evolution"]["frozen_paths"]),
        tuple(
            (str(path), float(values["minimum"]), float(values["maximum"]))
            for path, values in sorted(bounds.items())
        ),
    )


class _CandidateStageAdapter:
    def __init__(
        self,
        base_config: dict,
        runner: CandidateRunner,
        policy: RevisionPolicy,
        revision_root: Path,
        archive: CandidateArchive,
    ) -> None:
        self._base_config = base_config
        self._runner = runner
        self._policy = policy
        self._revision_root = revision_root
        self._archive = archive

    def execute(self, request: StageRequest) -> StageReceipt:
        public = _public_config(self._base_config)
        illegal = set(request.config_override) - set(
            request.stage.config.get("revision_paths", ())
        )
        if illegal:
            raise ValueError(
                "reasoning revised fields outside the current curriculum stage: "
                + ", ".join(sorted(illegal))
            )
        previous_stage_receipt = next(
            (
                receipt
                for receipt in reversed(request.prior_receipts)
                if receipt.stage != request.stage.name
                and receipt.outputs.get("candidate_id") is not None
            ),
            None,
        )
        previous_attempt_receipt = next(
            (
                receipt
                for receipt in reversed(request.prior_receipts)
                if receipt.stage == request.stage.name
                and receipt.outputs.get("candidate_id") is not None
            ),
            None,
        )
        selected_parent_receipt = previous_stage_receipt
        inherited_override = (
            {}
            if selected_parent_receipt is None
            else dict(
                selected_parent_receipt.outputs.get("effective_config_override", {})
            )
        )
        combined_override = {
            **inherited_override,
            **dict(request.stage.config.get("curriculum_override", {})),
            **request.config_override,
        }
        if combined_override:
            revision = self._policy.apply(public, combined_override)
            revision.write(
                self._revision_root
                / request.run_id
                / "revisions"
                / f"{request.stage.name}.attempt-{request.attempt}.json"
            )
            effective = revision.config
        else:
            effective = public
        base_output = str(self._base_config["output"])
        prior = (
            None
            if selected_parent_receipt is None
            else selected_parent_receipt.outputs.get("candidate_id")
        )
        parents = (
            (str(prior),)
            if prior is not None
            else tuple(self._base_config["evolution"]["parent_candidate_ids"])
        )

        def run_one(seed: int | None) -> dict:
            seed_config = deepcopy(effective)
            seed_config["_validation_stop_on_blow"] = any(
                requirement.get("metric") == "selection.blow_rate"
                and requirement.get("operator") == "=="
                and float(requirement.get("value")) == 0.0
                for requirement in request.stage.config[
                    "selection_requirements"
                ]
            )
            seed_config["training"] = dict(seed_config["training"])
            budget_mode = str(
                request.stage.config.get("budget_mode", "environment_steps")
            )
            if budget_mode == "environment_steps":
                seed_config["training"]["minimum_environment_steps"] = int(
                    request.stage.config["minimum_environment_steps"]
                )
            elif budget_mode == "episodes":
                seed_config["training"]["budget_mode"] = budget_mode
                seed_config["training"]["episodes"] = int(
                    request.stage.config["training_episodes"]
                )
                seed_config["training"]["validation_episodes"] = int(
                    request.stage.config["validation_episodes"]
                )
                stage_short_circuit = request.stage.config.get(
                    "short_circuit_minimum_episodes"
                )
                if stage_short_circuit is None:
                    seed_config["training"]["short_circuit"] = None
                else:
                    short_circuit = dict(
                        seed_config["training"]["short_circuit"]
                    )
                    short_circuit.pop("minimum_environment_steps", None)
                    short_circuit["minimum_completed_episodes"] = int(
                        stage_short_circuit
                    )
                    seed_config["training"]["short_circuit"] = short_circuit
                episode_coverage = request.stage.config.get("episode_coverage")
                if episode_coverage is None:
                    seed_config["training"].pop("episode_coverage", None)
                else:
                    seed_config["training"]["episode_coverage"] = dict(
                        episode_coverage
                    )
            else:
                raise ValueError(
                    f"unsupported campaign budget mode: {budget_mode}"
                )
            if seed is not None:
                seed_config["training"]["seed"] = int(seed)
            seed_path = "" if seed is None else f"seed-{seed}"
            seed_config["output"] = str(
                Path(base_output)
                / "campaign-runs"
                / request.run_id
                / request.stage.name
                / seed_path
                / f"attempt-{request.attempt}"
            )
            seed_config["_archive_output"] = base_output
            seed_config["_path"] = self._base_config["_path"]
            seed_config["_root"] = self._base_config["_root"]
            has_warm_start_source = prior is not None or (
                self._base_config["evolution"].get("base_parent") is not None
            )
            if (
                bool(request.stage.config.get("warm_start_parent", False))
                and has_warm_start_source
            ):
                if len(parents) != 1:
                    raise ValueError(
                        "warm-start stage requires exactly one parent candidate"
                    )
                parent = self._archive.load_candidate(parents[0])
                seed_config["_warm_start_model"] = {
                    "candidate_id": parent.candidate_id,
                    "model_path": str(parent.model_path),
                    "model_sha256": parent.manifest["model_sha256"],
                }
            candidate, evaluation = self._runner.run(
                seed_config,
                parent_candidate_ids=parents,
                hypothesis=str(self._base_config["evolution"]["hypothesis"]),
            )
            output_root = Path(str(seed_config["output"]))
            if not output_root.is_absolute():
                output_root = Path(str(seed_config["_root"])) / output_root
            diagnostic_summary = output_root / "training-diagnostic-summary.json"
            diagnostic_reference = None
            if diagnostic_summary.is_file():
                diagnostic_reference = {
                    "path": str(diagnostic_summary),
                    "file_sha256": _sha256(diagnostic_summary),
                }
            validation_diagnostics = output_root / "validation-diagnostics.jsonl"
            validation_reference = None
            if validation_diagnostics.is_file():
                validation_reference = {
                    "path": str(validation_diagnostics),
                    "file_sha256": _sha256(validation_diagnostics),
                }
            return {
                "seed": seed,
                "candidate_id": candidate.candidate_id,
                "candidate_manifest": str(candidate.path / "manifest.json"),
                "candidate_model_sha256": candidate.manifest["model_sha256"],
                "evaluation_id": evaluation.evaluation_id,
                "evaluation_path": str(evaluation.path),
                "evaluation_sha256": _sha256(evaluation.path),
                "evaluation_status": evaluation.status,
                "metrics": dict(evaluation.metrics),
                "training_diagnostic_summary": diagnostic_reference,
                "validation_diagnostics": validation_reference,
            }

        seeds = request.stage.config.get("seeds")
        if seeds is not None:
            results = []
            with ThreadPoolExecutor(
                max_workers=int(request.stage.config["max_parallel"])
            ) as executor:
                futures = {
                    executor.submit(run_one, int(seed)): int(seed)
                    for seed in seeds
                }
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda item: int(item["seed"]))
            return StageReceipt(
                stage=request.stage.name,
                attempt=request.attempt,
                status="complete",
                outputs={
                    "seed_results": results,
                    "parent_candidate_ids": list(parents),
                    "effective_config_override": combined_override,
                },
            )

        result = run_one(request.stage.config.get("seed"))
        return StageReceipt(
            stage=request.stage.name,
            attempt=request.attempt,
            status="complete",
            outputs={
                **result,
                "parent_candidate_ids": list(parents),
                "prior_attempt_candidate_id": (
                    None
                    if previous_attempt_receipt is None
                    else previous_attempt_receipt.outputs.get("candidate_id")
                ),
                "effective_config_override": combined_override,
            },
        )


class _EconomicEvidenceGate:
    def __init__(self, archive: CandidateArchive) -> None:
        self._archive = archive

    def evaluate(self, request: GateRequest) -> GateResult:
        outputs = request.receipt.outputs
        parent_ids = tuple(str(value) for value in outputs.get(
            "parent_candidate_ids", ()
        ))
        parent_requirements = request.stage.config.get(
            "parent_improvement_requirements", ()
        )
        parent_any_requirements = request.stage.config.get(
            "parent_improvement_any_requirements", ()
        )
        parent_retention_requirements = request.stage.config.get(
            "parent_retention_requirements", ()
        )
        if outputs.get("seed_results") is not None:
            failures = []
            seed_evidence = []
            for result in outputs["seed_results"]:
                evidence = self._evaluate_outputs(
                    result,
                    request.stage.config["selection_requirements"],
                    parent_ids=parent_ids,
                    parent_requirements=parent_requirements,
                    parent_any_requirements=parent_any_requirements,
                    parent_retention_requirements=parent_retention_requirements,
                )
                seed_evidence.append({"seed": result["seed"], **evidence})
                failures.extend(
                    {"seed": result["seed"], **failure}
                    for failure in evidence["failures"]
                )
                if evidence["evaluation_status"] != "PASS":
                    failures.append({
                        "seed": result["seed"],
                        "metric": "evaluation_status",
                        "expected": "PASS",
                        "actual": evidence["evaluation_status"],
                    })
            evidence = {"seeds": seed_evidence, "failures": failures}
            if failures:
                return GateResult(
                    Decision.STOP,
                    "frozen multi-seed confirmation failed its economic gate",
                    evidence,
                )
            return GateResult(
                Decision.PROCEED,
                "all frozen final seeds passed the economic gate",
                evidence,
            )
        evidence = self._evaluate_outputs(
            outputs,
            request.stage.config["selection_requirements"],
            parent_ids=parent_ids,
            parent_requirements=parent_requirements,
            parent_any_requirements=parent_any_requirements,
            parent_retention_requirements=parent_retention_requirements,
        )
        failures = evidence["failures"]
        evaluation_status = evidence["evaluation_status"]
        if failures or evaluation_status != "PASS":
            if not bool(request.stage.config.get("allow_revisions", True)):
                return GateResult(
                    Decision.STOP,
                    "frozen final recipe failed multi-seed economic confirmation",
                    evidence,
                )
            if any(
                failure.get("metric") == "training.short_circuited"
                for failure in failures
            ):
                reason = "training short circuit reached its declared evidence boundary"
            elif evidence.get("selection_economics_available") is False:
                reason = "candidate evaluator failed before economic selection"
            else:
                reason = "challenger missed the declared economic selection gate"
            return GateResult(
                Decision.REVISE,
                reason,
                evidence,
            )
        return GateResult(
            Decision.PROCEED,
            "challenger passed the declared economic selection gate",
            evidence,
        )

    def _evaluate_outputs(
        self,
        outputs: Mapping,
        requirements,
        *,
        parent_ids: tuple[str, ...],
        parent_requirements,
        parent_any_requirements=(),
        parent_retention_requirements=(),
    ) -> dict:
        candidate = self._archive.load_candidate(str(outputs["candidate_id"]))
        evaluation = self._archive.load_evaluation(str(outputs["evaluation_id"]))
        path = Path(str(outputs["evaluation_path"]))
        if (
            evaluation.candidate_id != candidate.candidate_id
            or path != evaluation.path
            or _sha256(path) != outputs.get("evaluation_sha256")
            or dict(evaluation.metrics) != dict(outputs.get("metrics", {}))
        ):
            raise ValueError("candidate evaluation handoff identity drifted")
        if float(evaluation.metrics.get("training.short_circuited", 0.0)) == 1.0:
            return {
                "candidate_id": candidate.candidate_id,
                "evaluation_id": evaluation.evaluation_id,
                "values": {"training.short_circuited": 1.0},
                "parent_improvements": {},
                "failures": [{
                    "metric": "training.short_circuited",
                    "operator": "==",
                    "threshold": 0.0,
                    "actual": 1.0,
                }],
                "evaluation_status": evaluation.status,
            }
        selection_economics_available = any(
            stage.get("name") == "selection" for stage in evaluation.stages
        )
        if not selection_economics_available:
            failed_stages = tuple(
                stage
                for stage in evaluation.stages
                if stage.get("status") == "FAIL"
            )
            if evaluation.status != "FAIL" or not failed_stages:
                raise ValueError(
                    "evaluation ended before selection without a failing "
                    "evaluator-stage receipt"
                )
            return {
                "candidate_id": candidate.candidate_id,
                "evaluation_id": evaluation.evaluation_id,
                "values": {},
                "available_metrics": dict(evaluation.metrics),
                "selection_economics_available": False,
                "parent_improvements": {},
                "failures": [
                    {
                        "metric": f"evaluator_stage.{stage['name']}",
                        "expected": "PASS",
                        "actual": "FAIL",
                    }
                    for stage in failed_stages
                ],
                "evaluation_status": evaluation.status,
            }
        failures = []
        values = {}
        for requirement in requirements:
            metric = requirement["metric"]
            if metric not in evaluation.metrics:
                raise ValueError(f"required economic metric is missing: {metric}")
            value = float(evaluation.metrics[metric])
            threshold = float(requirement["value"])
            operator = requirement["operator"]
            comparisons = {
                ">": value > threshold,
                ">=": value >= threshold,
                "<": value < threshold,
                "<=": value <= threshold,
                "==": value == threshold,
            }
            values[metric] = value
            if not comparisons[operator]:
                failures.append({
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "actual": value,
                })
        parent_values = {}
        if parent_ids and (
            parent_requirements
            or parent_any_requirements
            or parent_retention_requirements
        ):
            parent_evaluations = []
            for parent_id in parent_ids:
                parent = self._archive.latest_evaluation(parent_id)
                if parent is None:
                    raise ValueError(
                        f"parent candidate has no evaluation: {parent_id}"
                    )
                parent_evaluations.append(parent)
            def evaluate_parent_requirement(requirement):
                metric = requirement["metric"]
                if metric not in evaluation.metrics or any(
                    metric not in parent.metrics for parent in parent_evaluations
                ):
                    raise ValueError(
                        f"parent-improvement metric is missing: {metric}"
                    )
                current = float(evaluation.metrics[metric])
                parents = [float(parent.metrics[metric]) for parent in parent_evaluations]
                direction = requirement["direction"]
                baseline = max(parents) if direction == "maximize" else min(parents)
                minimum_delta = float(requirement["minimum_delta"])
                improvement = (
                    current - baseline
                    if direction == "maximize"
                    else baseline - current
                )
                parent_values[metric] = {
                    "actual": current,
                    "parent": baseline,
                    "improvement": improvement,
                    "minimum_delta": minimum_delta,
                }
                return improvement > minimum_delta, {
                    "metric": metric,
                    "direction": direction,
                    "parent": baseline,
                    "minimum_delta": minimum_delta,
                    "actual": current,
                }

            for requirement in parent_requirements:
                passed, failure = evaluate_parent_requirement(requirement)
                if not passed:
                    failures.append(failure)
            if parent_any_requirements:
                alternatives = [
                    evaluate_parent_requirement(requirement)
                    for requirement in parent_any_requirements
                ]
                if not any(passed for passed, _ in alternatives):
                    failures.append({
                        "metric": "parent_improvement_any",
                        "alternatives": [
                            failure for _, failure in alternatives
                        ],
                    })
            for requirement in parent_retention_requirements:
                metric = requirement["metric"]
                if metric not in evaluation.metrics or any(
                    metric not in parent.metrics for parent in parent_evaluations
                ):
                    raise ValueError(
                        f"parent-retention metric is missing: {metric}"
                    )
                current = float(evaluation.metrics[metric])
                explicit_direction = "direction" in requirement
                direction = requirement.get("direction", "minimize")
                parent_metrics = [
                    float(parent.metrics[metric])
                    for parent in parent_evaluations
                ]
                baseline = (
                    max(parent_metrics)
                    if not explicit_direction or direction == "maximize"
                    else min(parent_metrics)
                )
                maximum_regression = float(requirement["maximum_regression"])
                regression = (
                    baseline - current
                    if direction == "maximize"
                    else current - baseline
                )
                parent_values[f"retain:{metric}"] = {
                    "current": current,
                    "parent": baseline,
                    "direction": direction,
                    "regression": regression,
                    "maximum_regression": maximum_regression,
                }
                if regression > maximum_regression:
                    failures.append({
                        "metric": metric,
                        "direction": (
                            f"retain_{direction}"
                            if explicit_direction
                            else "retain_upper_bound"
                        ),
                        "parent": baseline,
                        "maximum_regression": maximum_regression,
                        "actual": current,
                    })
        evidence = {
            "candidate_id": candidate.candidate_id,
            "evaluation_id": evaluation.evaluation_id,
            "values": values,
            "parent_improvements": parent_values,
            "failures": failures,
            "evaluation_status": evaluation.status,
        }
        return evidence


class _ArchiveReasoningAdapter:
    def __init__(
        self,
        *,
        provider: ReasoningAdapter,
        archive: CandidateArchive,
        policy: RevisionPolicy,
        base_config: Mapping,
        niches: tuple[Niche, ...],
        receipt_root: Path,
    ) -> None:
        self._provider = provider
        self._archive = archive
        self._policy = policy
        self._base_config = _public_config(base_config)
        self._niches = niches
        self._receipt_root = receipt_root

    def revise(self, request: ReasoningRequest):
        current = str(request.receipt.outputs["candidate_id"])
        candidate = self._archive.load_candidate(current)
        elites = self._archive.elites(self._niches)
        inspirations = tuple(dict.fromkeys(
            candidate_id
            for candidate_id in (
                *candidate.manifest.get("parent_candidate_ids", ()),
                *elites.values(),
            )
            if candidate_id != current
        ))
        frozen_contract = json.loads((candidate.path / "contract.json").read_text())
        failures = tuple(
            str(item.get("metric", item))
            for item in request.gate.evidence.get("failures", ())
        )
        training_diagnostics = None
        diagnostic_reference = request.receipt.outputs.get(
            "training_diagnostic_summary"
        )
        if diagnostic_reference is not None:
            diagnostic_path = Path(str(diagnostic_reference["path"]))
            if (
                not diagnostic_path.is_file()
                or _sha256(diagnostic_path)
                != diagnostic_reference.get("file_sha256")
            ):
                raise ValueError("training diagnostic summary identity drifted")
            training_diagnostics = json.loads(diagnostic_path.read_text())
            if (
                training_diagnostics.get("schema")
                != "propevolve_training_diagnostic_summary_v1"
            ):
                raise ValueError("training diagnostic summary schema drifted")
        packet = self._archive.create_reasoning_packet(
            champion_candidate_id=current,
            inspiration_candidate_ids=inspirations,
            frozen_contract=frozen_contract,
            failure_taxonomy=failures,
            training_diagnostics=training_diagnostics,
        )
        enriched = replace(
            request,
            receipt=replace(
                request.receipt,
                outputs={
                    **request.receipt.outputs,
                    "reasoning_packet": {
                        "path": str(packet.path),
                        "identity_sha256": packet.packet_sha256,
                        "file_sha256": _sha256(packet.path),
                        "elite_candidate_ids": elites,
                    },
                },
            ),
        )
        outcome = self._provider.revise(enriched)
        revision = (
            outcome.revision
            if isinstance(outcome, ReasoningOutcome)
            else outcome
        )
        if revision is not None:
            if revision.stage != request.stage.name:
                raise ValueError("reasoning attempted to revise a different stage")
            validated = self._policy.apply(
                self._base_config,
                revision.config_override,
            )
            validated.write(
                self._receipt_root
                / request.run_id
                / "reasoning"
                / f"revision-{request.revision_number}.json"
            )
        return outcome


def _reasoning_prompt(request: ReasoningRequest) -> str:
    packet = request.receipt.outputs.get("reasoning_packet", {})
    revision_paths = tuple(request.stage.config.get("revision_paths", ()))
    prompt = (
        "Use $ml-diagnose-experiment to identify the first failed learning or "
        "economic boundary, then $ml-design-experiment to propose one smallest "
        "causal revision and $ml-train-select-model to check training validity. "
        "Read the authenticated PropEvolve reasoning packet at "
        f"{packet.get('path')}. Its serialized file SHA-256 is "
        f"{packet.get('file_sha256')}; its embedded content identity is "
        f"{packet.get('identity_sha256')}. These authenticate different "
        "representations and must not be compared to each other. The packet "
        "contains the current parent, diverse metric elites, prior evaluation "
        "evidence, failure taxonomy, and materialized candidate contract. Its "
        "frozen_contract_sha256 covers concrete checkpoint, cache, split, cost, "
        "and market identities. The stage frozen_recipe_sha256 separately covers "
        "the immutable JSON recipe fields; these hashes are also not expected to "
        "be equal. Verify each hash only against the representation it names. "
        "The authenticated "
        "experiment ledger supplied with this request contains every completed "
        "attempt in the current campaign, including its effective revision, "
        "training and selection metrics, artifact identities, and parent lineage. "
        "Propose only a novel recipe not already present in that ledger. If no "
        "scientifically defensible allowlisted revision remains, fail closed with "
        "a genuine exhausted-hypothesis blocker instead of repeating a recipe or "
        "performing an arbitrary numerical permutation. "
        "Any numeric override must remain within the revision_bounds declared "
        "in the stage contract. Diagnostic targets guide diagnosis but are not "
        "promotion gates. "
        "Treat safety lexicographically: any nonzero selection blow rate makes "
        "a candidate infeasible regardless of pass rate. First diagnose whether "
        "the next bounded revision should change risk headroom, terminal reward "
        "strength, MLL-proximity shaping, near-target lead protection, large-win "
        "credit, terminal-aware replay, management exploration, or ratchet "
        "activation/giveback. Use closed-trade MFE, MAE, MFE-to-realized gap, "
        "capture ratio, winner round-trip rate, exit reasons, and PASS-versus-"
        "TIMEOUT slices to distinguish entry quality from winner-retention "
        "failure. Treat MFE/MAE as diagnostics rather than direct optimization "
        "targets so a short-horizon scalping policy cannot win by construction. "
        "Only optimize pass rate among zero-blow candidates. A promotable "
        "revision must improve pass rate and retained-winner economics relative "
        "to its matched parent while preserving zero blow; reject reward settings "
        "that collapse the profitable right tail, inactivity, or timeouts. Once "
        "a causal mechanism family is identified, prefer uncertainty-aware "
        "numeric search inside only that bounded family over either global brute "
        "force or unrelated one-field guesses. "
        "If optional surrogate advice is present, treat it only as uncertainty-aware "
        "diagnostics and proposals; reasoning remains responsible for accepting, "
        "refining, or rejecting it. Return REVISE with one complete "
        "JSON object of allowlisted dot-path overrides for the next challenger. "
        "Preserve any still-needed prior overrides because each revision replaces "
        "the previous effective override. Never change data, FFM lineage, temporal "
        "roles, sealed periods, costs, prop rules, evaluator gates, or deployment "
        "markets. Choose STOP when the scientific path is falsified and BLOCKED "
        "only for integrity, causality, lineage, or executable-contract faults."
    )
    prompt += (
        f" The current stage is {request.stage.name}. Its complete revision "
        f"allowlist is {json.dumps(revision_paths)}. A REVISE response must use "
        "only one or more paths from that exact list; do not propose a frozen "
        "mechanism merely because it appears in the general diagnostic guidance."
    )
    if request.stage.name == "regime_side_balance_500k":
        prompt += (
            " These Stage 2A.1 instructions supersede the general reward, risk, "
            "and exit-mechanism menu. Diagnose the matched side-balance mechanism "
            "using sampled Long and Short Regime rows and global recall together "
            "with teacher-free selection Long and Short entry counts. Preserve "
            "the exact immutable Stage 1 parent, static-state target semantics, "
            "zero selection blows, pass rate, winner R, and winner retention. "
            "For REVISE, revise only loss weight or Q temperature through the "
            "stage's exact allowlist; do not propose reward, risk, replay, target "
            "formula, semantics, teacher, data, or evaluator changes. Never "
            "disable or weaken Short chop learning, and never introduce an "
            "inference-time teacher gate."
        )
    elif request.stage.name == "persistent_chop_regime_500k":
        prompt += (
            " These Stage 2A.2 instructions supersede the general reward, risk, "
            "and exit-mechanism menu. Diagnose learned persistent-chop behavior "
            "with same-label exact-WAIT evidence: the authenticated "
            "regime_selectivity_dead_wait_minus_transition_ready_wait_model_wait "
            "delta must compare model WAIT probability on persistent-dead-chop "
            "WAIT rows against transition-ready WAIT rows. Separately inspect "
            "transition-positive Long and Short rows and declared-side response "
            "to retain both directions, teacher-free selection Long and Short "
            "entry counts, and selection.near_blow_timeout_rate relative to the "
            "selected Stage 2A.1 parent. Preserve zero selection blows, pass "
            "rate, winner R, and winner retention. For REVISE, revise only loss "
            "weight or persistent-chop emphasis through the stage's exact "
            "allowlist; do not propose reward, risk, replay, formula-family, "
            "teacher, data, or evaluator changes. Never disable or weaken Short "
            "chop learning, and never introduce an inference-time teacher gate."
        )
    elif request.stage.name == "regime_selectivity_1m":
        prompt += (
            " For this Stage 2A boundary, use the authenticated Regime-selectivity "
            "strata to compare dominant-chop versus non-chop and low-headroom "
            "versus safe-headroom WAIT pressure, while preserving nonzero Long "
            "and Short declared-side response, the Stage 1 entry-timing skill, "
            "zero selection blows, winner R, and teacher-free greedy activity. "
            "Teachers are training-only evidence; never propose an inference gate."
        )
    elif request.stage.name == "deficit_recovery_1m":
        prompt += (
            " For this Stage 2B boundary, compare teacher-free recovery stress "
            "against the selected Stage 2A parent. Preserve the ordinary $500 "
            "headroom guard and the frozen one-entry, $300-risk, -$2,700 start "
            "contract. Diagnose recovery success, mean terminal PnL, WAIT timeout, "
            "and blow evidence; zero recovery-stress blows remains a hard gate."
        )
    reflection = request.receipt.outputs.get("gepa_reflection")
    if reflection:
        prompt += (
            " This checkpoint uses the optional GEPA-style reflective proposer. "
            "Read its authenticated actionable-side-information packet at "
            f"{reflection.get('path')}. Verify its file SHA-256 "
            f"{reflection.get('file_sha256')} and embedded content identity "
            f"{reflection.get('identity_sha256')} against the representation each "
            "name identifies. Compare the current parent evidence with exactly one "
            "causally motivated child revision. Explain internally why the proposed "
            "mutation addresses the first failed boundary and is not a repeated "
            "recipe. The existing evaluator—not the reflection model—determines "
            "acceptance. Never trade a nonzero blow rate for aggregate improvement."
        )
    return prompt


def _build_codex_provider(config: Mapping, state_root: Path) -> ReasoningAdapter:
    policy = _revision_policy(config)
    base = _public_config(config)

    def validate(revision: Revision) -> None:
        policy.apply(base, revision.config_override)

    reasoning = config["campaign"]["reasoning"]
    return CodexCliReasoningAdapter(
        repository_root=Path(config["_root"]),
        receipt_root=state_root / "provider-receipts",
        prompt_builder=_reasoning_prompt,
        revision_validator=validate,
        executor=SubprocessCodexExecutor(_resolve_codex_executable(reasoning)),
        model=str(reasoning.get("model", "gpt-5.6-sol")),
        reasoning_effort=str(reasoning.get("reasoning_effort", "medium")),
        sandbox="read-only",
        timeout_seconds=int(reasoning.get("timeout_seconds", 1800)),
    )


def _plan(config: Mapping) -> TrainingPlan:
    campaign = config["campaign"]
    public_base_recipe_sha256 = hashlib.sha256(
        json.dumps(
            _public_config(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    frozen_recipe_sha256 = hashlib.sha256(
        json.dumps(
            {
                path: _config_value(config, path)
                for path in config["evolution"]["frozen_paths"]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return TrainingPlan(
        name="propevolve-offline-evolution",
        stages=tuple(StageSpec(
            name=str(budget_stage["name"]),
            stage_adapter=_STAGE_ADAPTER,
            gate_adapter=_GATE_ADAPTER,
            config={
                "schema": "propevolve_evolution_stage_v2",
                **_budget_stage_fields(budget_stage),
                "seed": budget_stage.get("seed"),
                "seeds": budget_stage.get("seeds"),
                "max_parallel": budget_stage.get("max_parallel"),
                "allow_revisions": bool(
                    budget_stage.get("allow_revisions", True)
                ),
                "selection_requirements": list(
                    budget_stage["selection_requirements"]
                ),
                "parent_improvement_requirements": list(
                    budget_stage.get("parent_improvement_requirements", ())
                ),
                "parent_improvement_any_requirements": list(
                    budget_stage.get(
                        "parent_improvement_any_requirements", ()
                    )
                ),
                "parent_retention_requirements": list(
                    budget_stage.get("parent_retention_requirements", ())
                ),
                "warm_start_parent": bool(
                    budget_stage.get("warm_start_parent", False)
                ),
                "curriculum_override": dict(
                    budget_stage.get("curriculum_override", {})
                ),
                "diagnostic_targets": list(campaign.get("diagnostic_targets", ())),
                "revision_bounds": dict(
                    (
                        path,
                        config["evolution"].get("revision_bounds", {})[path],
                    )
                    for path in budget_stage.get(
                        "revision_paths",
                        config["evolution"]["allowed_revision_paths"],
                    )
                    if path in config["evolution"].get("revision_bounds", {})
                ),
                "revision_paths": list(
                    budget_stage.get(
                        "revision_paths",
                        config["evolution"]["allowed_revision_paths"],
                    )
                ),
                "public_base_recipe_sha256": public_base_recipe_sha256,
                "frozen_recipe_sha256": frozen_recipe_sha256,
            },
            required_skills=(
                "ml-train-select-model",
                "ml-validate-temporal",
            ),
        ) for budget_stage in campaign["budget_stages"]),
        required_skills=(
            "ml-rigor-workflow",
            "ml-audit-data-labels",
            "ml-diagnose-experiment",
            "ml-design-experiment",
        ),
        max_revisions_per_stage=int(campaign["max_revisions_per_stage"]),
    )


def _niches(config: Mapping) -> tuple[Niche, ...]:
    return tuple(Niche(
        str(item["name"]),
        str(item["metric"]),
        bool(item.get("maximize", True)),
    ) for item in config["campaign"]["niches"])


def _campaign_path(config: Mapping, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(str(config["_root"])) / path


def _finalize_campaign(config: Mapping, state, archive: CandidateArchive) -> None:
    """Select and package one champion after every declared seed passed."""
    finalization = config["campaign"].get("finalization")
    if finalization is None or state.phase.value != "COMPLETE":
        return
    seeded_stages = {
        str(stage["name"]): tuple(int(seed) for seed in stage.get("seeds", ()))
        for stage in config["campaign"]["budget_stages"]
        if stage.get("seeds") is not None
    }
    receipts = {
        receipt.stage: receipt for receipt in state.receipts
        if receipt.stage in seeded_stages and receipt.outputs.get("seed_results")
    }
    minimum = int(finalization["minimum_seed_count"])
    if len(receipts) != len(seeded_stages):
        raise ValueError(
            "completed campaign lacks the declared final-seed evidence"
        )

    rows = []
    for stage_name, declared_seeds in seeded_stages.items():
        results = receipts[stage_name].outputs["seed_results"]
        if tuple(sorted(int(item["seed"]) for item in results)) != tuple(
            sorted(declared_seeds)
        ):
            raise ValueError("final-seed receipt disagrees with the declared seeds")
        for outputs in results:
            candidate = archive.load_candidate(str(outputs["candidate_id"]))
            evaluation = archive.load_evaluation(str(outputs["evaluation_id"]))
            if evaluation.candidate_id != candidate.candidate_id:
                raise ValueError("final-seed candidate and evaluation disagree")
            if evaluation.status != "PASS":
                raise ValueError("only passing final-seed candidates may be ranked")
            rows.append({
                "stage": stage_name,
                "seed": int(outputs["seed"]),
                "candidate_id": candidate.candidate_id,
                "evaluation_id": evaluation.evaluation_id,
                "metrics": dict(evaluation.metrics),
            })
    if len(rows) < minimum:
        raise ValueError("completed campaign has too few final-seed candidates")

    def rank(row: Mapping) -> tuple:
        values = []
        for rule in finalization["ranking"]:
            metric = str(rule["metric"])
            if metric not in row["metrics"]:
                raise ValueError(f"final ranking metric is missing: {metric}")
            value = float(row["metrics"][metric])
            values.append(value if rule["direction"] == "minimize" else -value)
        values.extend((int(row["seed"]), str(row["candidate_id"])))
        return tuple(values)

    selected = min(rows, key=rank)
    candidate = archive.load_candidate(str(selected["candidate_id"]))
    evaluation = archive.load_evaluation(str(selected["evaluation_id"]))
    report_body = {
        "schema": "propevolve_multiseed_gauntlet_v1",
        "run_id": state.run_id,
        "status": "PASS",
        "evaluated_seed_count": len(rows),
        "minimum_seed_count": minimum,
        "ranking": list(finalization["ranking"]),
        "selected_seed": int(selected["seed"]),
        "selected_candidate_id": candidate.candidate_id,
        "selected_evaluation_id": evaluation.evaluation_id,
        "seed_evidence": rows,
    }
    report_identity = hashlib.sha256(json.dumps(
        report_body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    report = {**report_body, "identity_sha256": report_identity}

    registry = ModelRegistry(
        _campaign_path(config, str(finalization["registry_root"])), archive
    )
    try:
        current = registry.champion()
    except ValueError as error:
        if "no champion" not in str(error):
            raise
        current = None
    if (
        current is None
        or current["candidate_id"] != candidate.candidate_id
        or current["evaluation_id"] != evaluation.evaluation_id
    ):
        registry.activate(
            candidate.candidate_id,
            evaluation.evaluation_id,
            reason=f"multi-seed gauntlet passed for {state.run_id}",
        )

    export_root = _campaign_path(config, str(finalization["export_root"]))
    if export_root.exists():
        existing = export_root / "gauntlet-report.json"
        if not existing.is_file() or json.loads(existing.read_text()) != report:
            raise ValueError("existing campaign export has a different identity")
        return
    export_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=export_root.parent))
    try:
        shutil.copyfile(candidate.model_path, temporary / "model.pt")
        for name in ("manifest.json", "contract.json", "recipe.json"):
            shutil.copyfile(candidate.path / name, temporary / name)
        (temporary / "gauntlet-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        bundle = {
            "schema": "propevolve_export_bundle_v1",
            "candidate_id": candidate.candidate_id,
            "evaluation_id": evaluation.evaluation_id,
            "model_sha256": _sha256(temporary / "model.pt"),
            "gauntlet_report_sha256": _sha256(
                temporary / "gauntlet-report.json"
            ),
        }
        (temporary / "bundle-manifest.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(export_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_evolution_campaign(
    config_path: str | Path,
    *,
    run_id: str,
    candidate_runner: CandidateRunner | None = None,
    reasoning: ReasoningAdapter | None | object = _DEFAULT_REASONING,
    surrogate: SurrogateAdvisor | None = None,
    skills: SkillBootstrapper | None = None,
    recover_reasoning: bool = False,
):
    """Run or interruption-safely resume offline challenger evolution."""
    config = load_experiment_config(config_path)
    root = Path(config["_root"])
    output = root / str(config["output"])
    state_root = root / str(config["campaign"]["state_root"])
    archive = CandidateArchive(output / "archive")
    base_parent = config["evolution"].get("base_parent")
    if base_parent is not None:
        parent = archive.register_external_parent(
            _campaign_path(config, str(base_parent["archive_root"])),
            candidate_id=str(base_parent["candidate_id"]),
            evaluation_id=str(base_parent["evaluation_id"]),
            model_sha256=str(base_parent["model_sha256"]),
        )
        _assert_parent_causal_contract(parent, config)
    if candidate_runner is None:
        from .training import HistoricalCandidateRunner

        candidate_runner = HistoricalCandidateRunner()
    policy = _revision_policy(config)
    if reasoning is _DEFAULT_REASONING:
        provider = (
            _build_codex_provider(config, state_root)
            if config["campaign"]["reasoning"]["provider"] == "codex"
            else None
        )
    else:
        provider = reasoning
    if (
        provider is not None
        and config["campaign"]["reasoning"]["proposer"] == "gepa_reflective"
    ):
        provider = GepaReflectiveReasoningAdapter(
            provider=provider,
            packet_root=state_root / "gepa-reflections",
        )
    archive_reasoning = (
        None
        if provider is None
        else _ArchiveReasoningAdapter(
            provider=provider,
            archive=archive,
            policy=policy,
            base_config=config,
            niches=_niches(config),
            receipt_root=state_root,
        )
    )
    loop = TrainingLoop(
        adapters=DictAdapterRegistry(
            stages={
                _STAGE_ADAPTER: _CandidateStageAdapter(
                    config,
                    candidate_runner,
                    policy,
                    state_root,
                    archive,
                )
            },
            gates={_GATE_ADAPTER: _EconomicEvidenceGate(archive)},
        ),
        store=JsonRunStore(state_root),
        skills=skills or BundledSkillBootstrapper(DEFAULT_BUNDLE),
        reasoning=archive_reasoning,
        surrogate=surrogate,
    )
    run_directory = state_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"PropEvolve campaign is already active: {run_id}") from error
        try:
            plan = _plan(config)
            if recover_reasoning:
                checkpoints = tuple(
                    checkpoint
                    for checkpoint in loop.history(run_id)
                    if checkpoint.state.phase is Phase.NEEDS_REASONING
                )
                if not checkpoints:
                    raise ValueError(
                        "campaign has no durable NEEDS_REASONING checkpoint to recover"
                    )
                state = loop.recover(
                    plan,
                    run_id,
                    checkpoints[-1].checkpoint_id,
                )
            else:
                state = loop.run(plan, run_id=run_id)
            _finalize_campaign(config, state, archive)
            return state
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

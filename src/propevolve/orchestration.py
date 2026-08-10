"""AlphaEvolve-inspired PropEvolve campaign on the shared ML Training Loop."""

from __future__ import annotations

from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol

from ml_training_loop import (
    Decision,
    GateResult,
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
from ml_training_loop.integrations import CodexCliReasoningAdapter
from ml_training_loop.skills import BundledSkillBootstrapper
from ml_training_loop.stores import JsonRunStore

from .config import load_experiment_config
from .evolution import CandidateArchive, Niche, RevisionPolicy
from .reasoning import GepaReflectiveReasoningAdapter


_STAGE_ADAPTER = "propevolve-candidate"
_GATE_ADAPTER = "propevolve-economic-evidence"
_DEFAULT_REASONING = object()


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
    ) -> None:
        self._base_config = base_config
        self._runner = runner
        self._policy = policy
        self._revision_root = revision_root

    def execute(self, request: StageRequest) -> StageReceipt:
        public = _public_config(self._base_config)
        if request.config_override:
            revision = self._policy.apply(public, request.config_override)
            revision.write(
                self._revision_root
                / request.run_id
                / "revisions"
                / f"{request.stage.name}.attempt-{request.attempt}.json"
            )
            effective = revision.config
        else:
            effective = public
        effective["_path"] = self._base_config["_path"]
        effective["_root"] = self._base_config["_root"]
        prior = next(
            (
                receipt.outputs.get("candidate_id")
                for receipt in reversed(request.prior_receipts)
                if receipt.stage == request.stage.name
            ),
            None,
        )
        parents = (
            (str(prior),)
            if prior is not None
            else tuple(self._base_config["evolution"]["parent_candidate_ids"])
        )
        candidate, evaluation = self._runner.run(
            effective,
            parent_candidate_ids=parents,
            hypothesis=str(self._base_config["evolution"]["hypothesis"]),
        )
        output_root = Path(str(effective["output"]))
        if not output_root.is_absolute():
            output_root = Path(str(effective["_root"])) / output_root
        diagnostic_summary = output_root / "training-diagnostic-summary.json"
        diagnostic_reference = None
        if diagnostic_summary.is_file():
            diagnostic_reference = {
                "path": str(diagnostic_summary),
                "file_sha256": _sha256(diagnostic_summary),
            }
        return StageReceipt(
            stage=request.stage.name,
            attempt=request.attempt,
            status="complete",
            outputs={
                "candidate_id": candidate.candidate_id,
                "candidate_manifest": str(candidate.path / "manifest.json"),
                "candidate_model_sha256": candidate.manifest["model_sha256"],
                "evaluation_id": evaluation.evaluation_id,
                "evaluation_path": str(evaluation.path),
                "evaluation_sha256": _sha256(evaluation.path),
                "evaluation_status": evaluation.status,
                "metrics": dict(evaluation.metrics),
                "parent_candidate_ids": list(parents),
                "effective_config_override": dict(request.config_override),
                "training_diagnostic_summary": diagnostic_reference,
            },
        )


class _EconomicEvidenceGate:
    def __init__(self, archive: CandidateArchive) -> None:
        self._archive = archive

    def evaluate(self, request: GateRequest) -> GateResult:
        outputs = request.receipt.outputs
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
        failures = []
        values = {}
        for requirement in request.stage.config["selection_requirements"]:
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
        evidence = {
            "candidate_id": candidate.candidate_id,
            "evaluation_id": evaluation.evaluation_id,
            "values": values,
            "failures": failures,
        }
        if failures or evaluation.status != "PASS":
            return GateResult(
                Decision.REVISE,
                "challenger missed the declared economic selection gate",
                evidence,
            )
        return GateResult(
            Decision.PROCEED,
            "challenger passed the declared economic selection gate",
            evidence,
        )


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
        elites = self._archive.elites(self._niches)
        inspirations = tuple(dict.fromkeys(
            candidate_id
            for candidate_id in elites.values()
            if candidate_id != current
        ))
        candidate = self._archive.load_candidate(current)
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
        "credit, or terminal-aware replay. Only optimize pass rate among "
        "zero-blow candidates, and reject reward settings that collapse into "
        "inactivity or excessive timeouts. "
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
        model=str(reasoning.get("model", "gpt-5.6-sol")),
        reasoning_effort=str(reasoning.get("reasoning_effort", "medium")),
        sandbox="read-only",
        timeout_seconds=int(reasoning.get("timeout_seconds", 1800)),
    )


def _plan(config: Mapping) -> TrainingPlan:
    campaign = config["campaign"]
    return TrainingPlan(
        name="propevolve-offline-evolution",
        stages=(StageSpec(
            name="historical_candidate",
            stage_adapter=_STAGE_ADAPTER,
            gate_adapter=_GATE_ADAPTER,
            config={
                "schema": "propevolve_evolution_stage_v1",
                "selection_requirements": list(campaign["selection_requirements"]),
                "diagnostic_targets": list(campaign.get("diagnostic_targets", ())),
                "revision_bounds": dict(
                    config["evolution"].get("revision_bounds", {})
                ),
                "frozen_recipe_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            path: _config_value(config, path)
                            for path in config["evolution"]["frozen_paths"]
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
            required_skills=(
                "ml-train-select-model",
                "ml-validate-temporal",
            ),
        ),),
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


def run_evolution_campaign(
    config_path: str | Path,
    *,
    run_id: str,
    candidate_runner: CandidateRunner | None = None,
    reasoning: ReasoningAdapter | None | object = _DEFAULT_REASONING,
    surrogate: SurrogateAdvisor | None = None,
    skills: SkillBootstrapper | None = None,
):
    """Run or interruption-safely resume offline challenger evolution."""
    config = load_experiment_config(config_path)
    root = Path(config["_root"])
    output = root / str(config["output"])
    state_root = root / str(config["campaign"]["state_root"])
    archive = CandidateArchive(output / "archive")
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
            return loop.run(_plan(config), run_id=run_id)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

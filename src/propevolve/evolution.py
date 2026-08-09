"""Evidence-grounded offline evolution for immutable PropEvolve challengers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_once(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable artifact identity conflict: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"immutable artifact identity conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in metrics.items()}
    if not normalized or not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError("evaluation metrics must be a nonempty finite mapping")
    return normalized


@dataclass(frozen=True)
class CandidateBundle:
    candidate_id: str
    path: Path
    manifest: dict

    @property
    def model_path(self) -> Path:
        return self.path / "model.pt"


@dataclass(frozen=True)
class EvaluationReceipt:
    evaluation_id: str
    path: Path
    candidate_id: str
    status: str
    metrics: dict[str, float]
    stages: tuple[dict, ...]


@dataclass(frozen=True)
class ReasoningPacket:
    packet_sha256: str
    path: Path


@dataclass(frozen=True)
class Niche:
    name: str
    metric: str
    maximize: bool = True


class CandidateArchive:
    """Content-addressed candidates plus append-only evaluation evidence."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.candidates_root = self.root / "candidates"
        self.evaluations_root = self.root / "evaluations"
        self.reasoning_root = self.root / "reasoning"
        self.candidates_root.mkdir(parents=True, exist_ok=True)
        self.evaluations_root.mkdir(parents=True, exist_ok=True)
        self.reasoning_root.mkdir(parents=True, exist_ok=True)

    def register_candidate(
        self,
        model_path: str | Path,
        *,
        contract: Mapping,
        recipe: Mapping,
        parent_candidate_ids: Sequence[str] = (),
        hypothesis: str,
    ) -> CandidateBundle:
        model_path = Path(model_path).resolve(strict=True)
        parents = tuple(str(value) for value in parent_candidate_ids)
        for parent in parents:
            self.load_candidate(parent)
        identity = {
            "model_sha256": _file_sha256(model_path),
            "contract_sha256": _object_sha256(contract),
            "recipe_sha256": _object_sha256(recipe),
            "parent_candidate_ids": list(parents),
            "hypothesis": str(hypothesis),
        }
        candidate_id = _object_sha256(identity)
        destination = self.candidates_root / candidate_id
        if destination.exists():
            candidate = self.load_candidate(candidate_id)
            if candidate.manifest["identity"] != identity:
                raise ValueError("candidate identity collision")
            return candidate

        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.candidates_root))
        try:
            shutil.copyfile(model_path, temporary / "model.pt")
            (temporary / "contract.json").write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n"
            )
            (temporary / "recipe.json").write_text(
                json.dumps(recipe, indent=2, sort_keys=True) + "\n"
            )
            manifest = {
                "schema": "propevolve_candidate_bundle_v1",
                "candidate_id": candidate_id,
                "created_at": _now(),
                "identity": identity,
                "model_sha256": identity["model_sha256"],
                "contract_sha256": identity["contract_sha256"],
                "recipe_sha256": identity["recipe_sha256"],
                "parent_candidate_ids": list(parents),
                "hypothesis": str(hypothesis),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.load_candidate(candidate_id)

    def load_candidate(self, candidate_id: str) -> CandidateBundle:
        path = self.candidates_root / str(candidate_id)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or not (path / "model.pt").is_file():
            raise ValueError(f"unknown or incomplete candidate: {candidate_id}")
        manifest = json.loads(manifest_path.read_text())
        contract_path = path / "contract.json"
        recipe_path = path / "recipe.json"
        if (
            manifest.get("schema") != "propevolve_candidate_bundle_v1"
            or manifest.get("candidate_id") != candidate_id
            or _file_sha256(path / "model.pt") != manifest.get("model_sha256")
            or not contract_path.is_file()
            or not recipe_path.is_file()
            or _object_sha256(json.loads(contract_path.read_text()))
            != manifest.get("contract_sha256")
            or _object_sha256(json.loads(recipe_path.read_text()))
            != manifest.get("recipe_sha256")
            or _object_sha256(manifest.get("identity")) != candidate_id
        ):
            raise ValueError(f"candidate bundle integrity failure: {candidate_id}")
        return CandidateBundle(str(candidate_id), path, manifest)

    def list_candidates(self) -> tuple[CandidateBundle, ...]:
        return tuple(
            self.load_candidate(path.name)
            for path in sorted(self.candidates_root.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        )

    def record_evaluation(
        self,
        candidate_id: str,
        *,
        evaluator_contract: Mapping,
        metrics: Mapping[str, float],
        stages: Sequence[Mapping],
        status: str,
    ) -> EvaluationReceipt:
        self.load_candidate(candidate_id)
        status = str(status).upper()
        if status not in {"PASS", "FAIL", "REVISE"}:
            raise ValueError("evaluation status must be PASS, FAIL, or REVISE")
        normalized_metrics = _finite_metrics(metrics)
        normalized_stages = tuple(dict(stage) for stage in stages)
        identity = {
            "candidate_id": candidate_id,
            "evaluator_contract_sha256": _object_sha256(evaluator_contract),
            "metrics": normalized_metrics,
            "stages": list(normalized_stages),
            "status": status,
        }
        evaluation_id = _object_sha256(identity)
        payload = {
            "schema": "propevolve_evaluation_receipt_v1",
            "evaluation_id": evaluation_id,
            "created_at": _now(),
            "evaluator_contract": dict(evaluator_contract),
            **identity,
        }
        path = self.evaluations_root / f"{evaluation_id}.json"
        if not path.exists():
            _write_once(path, payload)
        return self.load_evaluation(evaluation_id)

    def load_evaluation(self, evaluation_id: str) -> EvaluationReceipt:
        path = self.evaluations_root / f"{evaluation_id}.json"
        if not path.is_file():
            raise ValueError(f"unknown evaluation: {evaluation_id}")
        payload = json.loads(path.read_text())
        identity = {
            "candidate_id": payload.get("candidate_id"),
            "evaluator_contract_sha256": payload.get("evaluator_contract_sha256"),
            "metrics": payload.get("metrics"),
            "stages": payload.get("stages"),
            "status": payload.get("status"),
        }
        if (
            payload.get("schema") != "propevolve_evaluation_receipt_v1"
            or payload.get("evaluation_id") != evaluation_id
            or _object_sha256(payload.get("evaluator_contract"))
            != payload.get("evaluator_contract_sha256")
            or _object_sha256(identity) != evaluation_id
        ):
            raise ValueError(f"evaluation receipt integrity failure: {evaluation_id}")
        return EvaluationReceipt(
            evaluation_id=evaluation_id,
            path=path,
            candidate_id=str(payload["candidate_id"]),
            status=str(payload["status"]),
            metrics={key: float(value) for key, value in payload["metrics"].items()},
            stages=tuple(dict(stage) for stage in payload["stages"]),
        )

    def list_evaluations(self, candidate_id: str | None = None) -> tuple[EvaluationReceipt, ...]:
        receipts = tuple(
            self.load_evaluation(path.stem)
            for path in sorted(self.evaluations_root.glob("*.json"))
        )
        if candidate_id is None:
            return receipts
        return tuple(item for item in receipts if item.candidate_id == candidate_id)

    def latest_evaluation(self, candidate_id: str) -> EvaluationReceipt | None:
        receipts = self.list_evaluations(candidate_id)
        if not receipts:
            return None
        return max(receipts, key=lambda receipt: receipt.path.stat().st_mtime_ns)

    def elites(self, niches: Sequence[Niche]) -> dict[str, str]:
        evaluated = []
        for candidate in self.list_candidates():
            receipt = self.latest_evaluation(candidate.candidate_id)
            if receipt is not None:
                evaluated.append((candidate, receipt))
        elites: dict[str, str] = {}
        for niche in niches:
            eligible = [
                pair for pair in evaluated if niche.metric in pair[1].metrics
            ]
            if not eligible:
                continue
            selected = (max if niche.maximize else min)(
                eligible, key=lambda pair: pair[1].metrics[niche.metric]
            )
            elites[niche.name] = selected[0].candidate_id
        return elites

    def create_reasoning_packet(
        self,
        *,
        champion_candidate_id: str,
        inspiration_candidate_ids: Sequence[str],
        frozen_contract: Mapping,
        failure_taxonomy: Sequence[str] = (),
    ) -> ReasoningPacket:
        champion = self.load_candidate(champion_candidate_id)

        def evidence(candidate: CandidateBundle) -> dict:
            evaluation = self.latest_evaluation(candidate.candidate_id)
            return {
                "candidate": candidate.manifest,
                "latest_evaluation": None
                if evaluation is None
                else json.loads(evaluation.path.read_text()),
            }

        payload = {
            "schema": "propevolve_reasoning_packet_v1",
            "created_at": _now(),
            "frozen_contract": dict(frozen_contract),
            "frozen_contract_sha256": _object_sha256(frozen_contract),
            "champion": evidence(champion),
            "inspirations": [
                evidence(self.load_candidate(candidate_id))
                for candidate_id in inspiration_candidate_ids
            ],
            "failure_taxonomy": [str(value) for value in failure_taxonomy],
        }
        packet_sha256 = _object_sha256(payload)
        payload["packet_sha256"] = packet_sha256
        path = self.reasoning_root / f"{packet_sha256}.json"
        _write_once(path, payload)
        return ReasoningPacket(packet_sha256, path)


@dataclass(frozen=True)
class EvaluationGate:
    metric: str
    operator: str
    threshold: float

    def passes(self, metrics: Mapping[str, float]) -> bool:
        if self.metric not in metrics:
            raise ValueError(f"evaluation metric is missing: {self.metric}")
        value = float(metrics[self.metric])
        operators = {
            ">": lambda: value > self.threshold,
            ">=": lambda: value >= self.threshold,
            "<": lambda: value < self.threshold,
            "<=": lambda: value <= self.threshold,
            "==": lambda: value == self.threshold,
        }
        if self.operator not in operators:
            raise ValueError(f"unsupported gate operator: {self.operator}")
        return operators[self.operator]()


@dataclass(frozen=True)
class EvaluationStage:
    name: str
    evaluator: Callable[[CandidateBundle], Mapping[str, float]]
    gates: tuple[EvaluationGate, ...] = ()


class EvaluatorCascade:
    """Run cheap evaluators first and persist one authenticated result vector."""

    def __init__(
        self,
        archive: CandidateArchive,
        evaluator_contract: Mapping,
        stages: Sequence[EvaluationStage],
    ) -> None:
        if not stages:
            raise ValueError("an evaluator cascade requires at least one stage")
        self.archive = archive
        self.evaluator_contract = dict(evaluator_contract)
        self.stages = tuple(stages)

    def evaluate(self, candidate_id: str) -> EvaluationReceipt:
        candidate = self.archive.load_candidate(candidate_id)
        all_metrics: dict[str, float] = {}
        receipts = []
        status = "PASS"
        for stage in self.stages:
            metrics = _finite_metrics(stage.evaluator(candidate))
            passed = all(gate.passes(metrics) for gate in stage.gates)
            receipts.append({
                "name": stage.name,
                "status": "PASS" if passed else "FAIL",
                "metrics": metrics,
            })
            for key, value in metrics.items():
                all_metrics[f"{stage.name}.{key}"] = value
            if not passed:
                status = "FAIL"
                break
        return self.archive.record_evaluation(
            candidate_id,
            evaluator_contract=self.evaluator_contract,
            metrics=all_metrics,
            stages=receipts,
            status=status,
        )


@dataclass(frozen=True)
class RevisionReceipt:
    config: dict
    diff: dict
    base_sha256: str
    revised_sha256: str

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        _write_once(path, {
            "schema": "propevolve_recipe_revision_v1",
            "base_sha256": self.base_sha256,
            "revised_sha256": self.revised_sha256,
            "diff": self.diff,
            "config": self.config,
        })
        return path


@dataclass(frozen=True)
class RevisionPolicy:
    allowed_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    numeric_bounds: tuple[tuple[str, float, float], ...] = ()

    def apply(self, base: Mapping, changes: Mapping[str, object]) -> RevisionReceipt:
        if not changes:
            raise ValueError("a recipe revision requires at least one change")
        revised = deepcopy(dict(base))
        diff = {}
        bounds = {
            path: (minimum, maximum)
            for path, minimum, maximum in self.numeric_bounds
        }
        for path, value in changes.items():
            if path not in self.allowed_paths:
                raise ValueError(f"revision path is not allowlisted: {path}")
            if any(path == frozen or path.startswith(frozen + ".") for frozen in self.frozen_paths):
                raise ValueError(f"revision path is frozen: {path}")
            if path in bounds:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"bounded revision must be numeric: {path}")
                minimum, maximum = bounds[path]
                if not minimum <= float(value) <= maximum:
                    raise ValueError(
                        f"revision path {path} is outside declared bounds "
                        f"[{minimum}, {maximum}]"
                    )
            parts = path.split(".")
            cursor = revised
            for part in parts[:-1]:
                if not isinstance(cursor, dict) or part not in cursor:
                    raise ValueError(f"revision path does not exist: {path}")
                cursor = cursor[part]
            leaf = parts[-1]
            if not isinstance(cursor, dict) or leaf not in cursor:
                raise ValueError(f"revision path does not exist: {path}")
            before = cursor[leaf]
            cursor[leaf] = deepcopy(value)
            diff[path] = {"before": before, "after": value}
        return RevisionReceipt(
            config=revised,
            diff=diff,
            base_sha256=_object_sha256(base),
            revised_sha256=_object_sha256(revised),
        )


@dataclass(frozen=True)
class RegistryEvent:
    event_id: str
    candidate_id: str
    evaluation_id: str
    previous_candidate_id: str | None
    reason: str


class ModelRegistry:
    """Reversible champion pointer backed by append-only activation receipts."""

    def __init__(self, root: str | Path, archive: CandidateArchive) -> None:
        self.root = Path(root)
        self.archive = archive
        self.events_root = self.root / "events"
        self.events_root.mkdir(parents=True, exist_ok=True)

    def activate(
        self,
        candidate_id: str,
        evaluation_id: str,
        *,
        reason: str,
    ) -> RegistryEvent:
        self.archive.load_candidate(candidate_id)
        evaluation = self.archive.load_evaluation(evaluation_id)
        if evaluation.candidate_id != candidate_id or evaluation.status != "PASS":
            raise ValueError("only a passing evaluation for the same candidate can activate")
        previous = self.champion() if (self.root / "champion.json").exists() else None
        event_payload = {
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "previous_candidate_id": None if previous is None else previous["candidate_id"],
            "reason": str(reason),
            "created_at": _now(),
        }
        event_id = _object_sha256(event_payload)
        payload = {
            "schema": "propevolve_registry_event_v1",
            "event_id": event_id,
            **event_payload,
        }
        _write_once(self.events_root / f"{event_id}.json", payload)
        pointer = {
            "schema": "propevolve_champion_pointer_v1",
            "event_id": event_id,
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
        }
        temporary = self.root / ".champion.tmp.json"
        temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.root / "champion.json")
        return RegistryEvent(
            event_id,
            candidate_id,
            evaluation_id,
            event_payload["previous_candidate_id"],
            str(reason),
        )

    def champion(self) -> dict:
        path = self.root / "champion.json"
        if not path.is_file():
            raise ValueError("no champion has been activated")
        payload = json.loads(path.read_text())
        if payload.get("schema") != "propevolve_champion_pointer_v1":
            raise ValueError("champion pointer integrity failure")
        event_path = self.events_root / f"{payload.get('event_id')}.json"
        if not event_path.is_file():
            raise ValueError("champion activation receipt is missing")
        event = json.loads(event_path.read_text())
        if (
            event.get("candidate_id") != payload.get("candidate_id")
            or event.get("evaluation_id") != payload.get("evaluation_id")
        ):
            raise ValueError("champion pointer disagrees with activation receipt")
        self.archive.load_candidate(payload["candidate_id"])
        self.archive.load_evaluation(payload["evaluation_id"])
        return payload

    def history(self) -> tuple[RegistryEvent, ...]:
        events = []
        for path in sorted(self.events_root.glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
            payload = json.loads(path.read_text())
            events.append(RegistryEvent(
                payload["event_id"],
                payload["candidate_id"],
                payload["evaluation_id"],
                payload["previous_candidate_id"],
                payload["reason"],
            ))
        return tuple(events)

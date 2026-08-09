"""Optional evidence-grounded reasoning proposers."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from ml_training_loop.interfaces import ReasoningAdapter, ReasoningRequest


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_once(path: Path, payload: Mapping) -> None:
    encoded = _canonical_json(payload)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("GEPA reflection packet identity collision")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _authenticated_parent_packet(request: ReasoningRequest) -> tuple[dict, dict]:
    reference = request.receipt.outputs.get("reasoning_packet")
    if not isinstance(reference, Mapping):
        raise ValueError("GEPA proposer requires an authenticated reasoning packet")
    path = Path(str(reference.get("path", "")))
    if not path.is_file():
        raise ValueError("GEPA parent reasoning packet is missing")
    if _sha256_file(path) != reference.get("file_sha256"):
        raise ValueError("GEPA parent reasoning packet file identity drifted")
    payload = json.loads(path.read_text())
    if payload.get("packet_sha256") != reference.get("identity_sha256"):
        raise ValueError("GEPA parent reasoning packet content identity drifted")
    return dict(reference), payload


class GepaReflectiveReasoningAdapter:
    """Add GEPA-style actionable side information before proposing a revision.

    This adapter does not optimize the trading policy. It supplies a
    content-addressed reflection packet to the configured reasoning provider;
    the existing revision policy, frozen contract, and economic gates remain
    authoritative.
    """

    def __init__(
        self,
        *,
        provider: ReasoningAdapter,
        packet_root: str | Path,
    ) -> None:
        self._provider = provider
        self._packet_root = Path(packet_root)

    def revise(self, request: ReasoningRequest):
        parent_reference, parent = _authenticated_parent_packet(request)
        ledger = request.experiment_ledger
        ledger_payload = {
            "identity": None if ledger is None else ledger.identity,
            "entries": [] if ledger is None else [
                asdict(entry) for entry in ledger.entries
            ],
        }
        payload = {
            "schema": "propevolve_gepa_reflection_v1",
            "run_id": request.run_id,
            "plan_identity": request.plan.identity,
            "stage": request.stage.name,
            "revision_number": request.revision_number,
            "parent_reasoning_packet": parent_reference,
            "candidate_evidence": {
                "champion": parent.get("champion"),
                "inspirations": parent.get("inspirations", []),
                "failure_taxonomy": parent.get("failure_taxonomy", []),
            },
            "actionable_side_information": {
                "stage_receipt": asdict(request.receipt),
                "gate": {
                    "decision": request.gate.decision.value,
                    "reason": request.gate.reason,
                    "evidence": dict(request.gate.evidence),
                },
                "prior_revisions": [
                    asdict(revision) for revision in request.prior_revisions
                ],
                "effective_config_override": dict(
                    request.effective_config_override
                ),
                "surrogate_advice": (
                    None
                    if request.surrogate_advice is None
                    else asdict(request.surrogate_advice)
                ),
            },
            "experiment_ledger": ledger_payload,
            "reflection_contract": {
                "candidate": "one schema-valid allowlisted JSON revision",
                "comparison": "parent versus one causally motivated child",
                "acceptance": "existing authenticated economic evaluator",
                "hard_feasibility_gate": "selection.blow_rate == 0",
                "forbidden": [
                    "data lineage",
                    "temporal roles or sealed evidence",
                    "FFM checkpoint or cache identity",
                    "costs, prop rules, or evaluator gates",
                    "deployment markets",
                ],
            },
        }
        identity = _sha256_bytes(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        payload["identity_sha256"] = identity
        path = (
            self._packet_root
            / request.run_id
            / request.stage.name
            / f"revision-{request.revision_number}-{identity}.json"
        )
        _write_once(path, payload)
        reference = {
            "path": str(path.resolve()),
            "identity_sha256": identity,
            "file_sha256": _sha256_file(path),
        }
        enriched = replace(
            request,
            receipt=replace(
                request.receipt,
                outputs={
                    **request.receipt.outputs,
                    "gepa_reflection": reference,
                },
            ),
        )
        return self._provider.revise(enriched)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.evolution import (
    CandidateArchive,
    EvaluationGate,
    EvaluationStage,
    EvaluatorCascade,
    ModelRegistry,
    Niche,
    RevisionPolicy,
)


def _candidate(
    archive: CandidateArchive,
    tmp_path: Path,
    name: str,
    *,
    parents: tuple[str, ...] = (),
):
    model = tmp_path / f"{name}.pt"
    model.write_bytes(name.encode())
    return archive.register_candidate(
        model,
        contract={"checkpoint": "mask", "split": "2021-2025"},
        recipe={"agent": {"hidden_dim": 16}, "name": name},
        parent_candidate_ids=parents,
        hypothesis=f"candidate {name}",
    )


def test_archive_preserves_immutable_parent_and_child_bundles(tmp_path: Path) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    parent = _candidate(archive, tmp_path, "parent")
    child = _candidate(archive, tmp_path, "child", parents=(parent.candidate_id,))

    assert parent.model_path.read_bytes() == b"parent"
    assert child.model_path.read_bytes() == b"child"
    assert child.manifest["parent_candidate_ids"] == [parent.candidate_id]
    assert archive.load_candidate(parent.candidate_id) == parent
    assert len(archive.list_candidates()) == 2

    duplicate = archive.register_candidate(
        tmp_path / "parent.pt",
        contract={"checkpoint": "mask", "split": "2021-2025"},
        recipe={"agent": {"hidden_dim": 16}, "name": "parent"},
        hypothesis="candidate parent",
    )
    assert duplicate.candidate_id == parent.candidate_id

    parent.model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        archive.load_candidate(parent.candidate_id)


def test_archive_keeps_distinct_elites_for_multiple_objectives(tmp_path: Path) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    safe = _candidate(archive, tmp_path, "safe")
    profitable = _candidate(archive, tmp_path, "profitable")
    archive.record_evaluation(
        safe.candidate_id,
        evaluator_contract={"name": "selection-v1"},
        metrics={"pass_rate": 0.45, "blow_rate": 0.05, "expectancy": 0.10},
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )
    archive.record_evaluation(
        profitable.candidate_id,
        evaluator_contract={"name": "selection-v1"},
        metrics={"pass_rate": 0.55, "blow_rate": 0.20, "expectancy": 0.30},
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )

    elites = archive.elites((
        Niche("highest_pass", "pass_rate", maximize=True),
        Niche("lowest_blow", "blow_rate", maximize=False),
        Niche("best_expectancy", "expectancy", maximize=True),
    ))

    assert elites == {
        "highest_pass": profitable.candidate_id,
        "lowest_blow": safe.candidate_id,
        "best_expectancy": profitable.candidate_id,
    }


def test_evaluator_cascade_stops_before_expensive_stage_on_gate_failure(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    candidate = _candidate(archive, tmp_path, "weak")
    expensive_calls = 0

    def smoke(_candidate):
        return {"finite": 1.0, "learning_delta": -0.1}

    def expensive(_candidate):
        nonlocal expensive_calls
        expensive_calls += 1
        return {"pass_rate": 0.9}

    cascade = EvaluatorCascade(archive, {"name": "cascade-v1"}, (
        EvaluationStage(
            "smoke",
            smoke,
            gates=(EvaluationGate("learning_delta", ">", 0.0),),
        ),
        EvaluationStage("walk_forward", expensive),
    ))

    receipt = cascade.evaluate(candidate.candidate_id)

    assert receipt.status == "FAIL"
    assert expensive_calls == 0
    assert [stage["name"] for stage in receipt.stages] == ["smoke"]


def test_reasoning_packet_is_authenticated_and_contains_prior_evidence(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    champion = _candidate(archive, tmp_path, "champion")
    challenger = _candidate(
        archive, tmp_path, "challenger", parents=(champion.candidate_id,)
    )
    evaluation = archive.record_evaluation(
        challenger.candidate_id,
        evaluator_contract={"name": "selection-v1"},
        metrics={"pass_rate": 0.40, "blow_rate": 0.10},
        stages=({"name": "selection", "status": "PASS"},),
        status="PASS",
    )

    packet = archive.create_reasoning_packet(
        champion_candidate_id=champion.candidate_id,
        inspiration_candidate_ids=(challenger.candidate_id,),
        frozen_contract={"split": "2021-2025", "max_loss": 3000},
        failure_taxonomy=("excessive_abstention",),
    )

    payload = json.loads(packet.path.read_text())
    assert payload["packet_sha256"] == packet.packet_sha256
    assert payload["inspirations"][0]["latest_evaluation"]["evaluation_id"] == (
        evaluation.evaluation_id
    )
    assert payload["failure_taxonomy"] == ["excessive_abstention"]


def test_revision_policy_changes_only_allowlisted_json_fields() -> None:
    base = {
        "agent": {"hidden_dim": 128, "learning_rate": 1e-4},
        "temporal": {"train_start": "2021-01-01"},
        "challenge": {"max_loss": 3000},
    }
    policy = RevisionPolicy(
        allowed_paths=("agent.hidden_dim", "agent.learning_rate"),
        frozen_paths=("temporal", "challenge"),
    )

    revision = policy.apply(base, {"agent.hidden_dim": 256})

    assert revision.config["agent"]["hidden_dim"] == 256
    assert revision.config["temporal"] == base["temporal"]
    assert revision.diff == {"agent.hidden_dim": {"before": 128, "after": 256}}

    with pytest.raises(ValueError, match="not allowlisted"):
        policy.apply(base, {"challenge.max_loss": 4000})


def test_registry_can_promote_then_roll_back_without_deleting_models(
    tmp_path: Path,
) -> None:
    archive = CandidateArchive(tmp_path / "archive")
    first = _candidate(archive, tmp_path, "first")
    second = _candidate(archive, tmp_path, "second", parents=(first.candidate_id,))
    evaluations = []
    for candidate in (first, second):
        evaluations.append(archive.record_evaluation(
            candidate.candidate_id,
            evaluator_contract={"name": "promotion-v1"},
            metrics={"pass_rate": 0.60},
            stages=({"name": "sealed", "status": "PASS"},),
            status="PASS",
        ))
    registry = ModelRegistry(tmp_path / "registry", archive)

    registry.activate(first.candidate_id, evaluations[0].evaluation_id, reason="initial")
    registry.activate(second.candidate_id, evaluations[1].evaluation_id, reason="promote")
    rollback = registry.activate(
        first.candidate_id, evaluations[0].evaluation_id, reason="rollback"
    )

    assert registry.champion()["candidate_id"] == first.candidate_id
    assert rollback.previous_candidate_id == second.candidate_id
    assert first.model_path.exists() and second.model_path.exists()
    assert len(registry.history()) == 3

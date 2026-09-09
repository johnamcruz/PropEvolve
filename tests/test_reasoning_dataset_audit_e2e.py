"""Production-shaped dataset audit boundaries for the reasoning challenger."""

import json
import hashlib

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.reasoning_policy.dataset import (
    audit_supervised_dataset,
    supervised_record,
    write_supervised_dataset,
)
from propevolve.reasoning_policy.labels import ActionLabels, ActionOutcome
from propevolve.reasoning_policy.context import ContextWindow
from propevolve.reasoning_policy.integrity import file_digest


def _window(timestamp):
    return ContextWindow(
        timestamps=(timestamp - 1, timestamp),
        fields=("account.realized_pnl_norm", "trade.current_r"),
        values=np.asarray([[0.0, 0.0], [0.1, 0.2]], dtype=np.float32),
        available=np.asarray([True, True]),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def _record(timestamp, *, best, ticker="NQ", source_id=None):
    rewards = {
        Action.WAIT: 0.0,
        Action.ENTER_LONG_1: 2.0 if best is Action.ENTER_LONG_1 else -1.0,
        Action.ENTER_SHORT_1: 2.0 if best is Action.ENTER_SHORT_1 else -1.0,
    }
    record = supervised_record(
        _window(timestamp),
        ActionLabels(np.zeros(1, dtype=np.float32), {
            action: ActionOutcome(
                reward_to_go=reward,
                outcome="pass" if reward > 0 else "timeout",
                terminal_pnl=6000.0 if reward > 0 else 0.0,
                minimum_mll_headroom=1000.0,
                steps=1,
                outcome_end_ns=timestamp + 10,
            )
            for action, reward in rewards.items()
        }),
        source_id=source_id or f"source-{timestamp}", continuation_id="frozen-policy",
        target_temperature=1.0,
    )
    record["ticker"] = ticker
    return record


def _position_record(timestamp, *, best=Action.HOLD, ticker="NQ"):
    rewards = {Action.HOLD: 1.0 if best is Action.HOLD else 0.0,
               Action.CLOSE: 1.0 if best is Action.CLOSE else 0.0}
    record = supervised_record(
        _window(timestamp),
        ActionLabels(np.zeros(1, dtype=np.float32), {
            action: ActionOutcome(
                reward_to_go=reward, outcome="managed", terminal_pnl=reward * 300,
                minimum_mll_headroom=1000.0, steps=1, outcome_end_ns=timestamp + 10,
            ) for action, reward in rewards.items()
        }), source_id=f"position-{timestamp}", continuation_id="trade-mastery",
        target_temperature=1.0,
    )
    record["ticker"] = ticker
    return record


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    write_supervised_dataset(
        [_record(100, best=Action.ENTER_LONG_1),
         _record(110, best=Action.ENTER_SHORT_1),
         _record(120, best=Action.WAIT),
         _record(300, best=Action.ENTER_LONG_1),
         _record(310, best=Action.ENTER_SHORT_1),
         _record(320, best=Action.WAIT)],
        root,
        splits={"train": [0, 200], "valid": [200, 400]},
        lineage={
            "source_identity": "source", "specialist_identities": ["expansion", "trend", "regime"],
            "economic_contract": {"profit_target": 6000, "maximum_loss": -3000},
            "split_audit": {"status": "PASS"},
        },
        sealed_start_ns=500,
    )
    return root


def _mutate_first(path, mutate):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[0])
    path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows))


def test_audit_publishes_hash_bound_pass_for_causal_teacher_free_records(tmp_path):
    root = _dataset(tmp_path)
    audit = audit_supervised_dataset(root, specialist_score_mode="out_of_fold")

    assert audit["status"] == "PASS"
    assert audit["manifest_sha256"] == file_digest(root / "manifest.json")
    assert audit["sealed_touched"] is False
    assert audit["counts"] == {"train": 3, "valid": 3}
    assert audit["actions"] == {"ENTER_LONG_1": 2, "ENTER_SHORT_1": 2, "WAIT": 2}
    assert audit["actions_by_role"] == {
        "train": {"ENTER_LONG_1": 1, "ENTER_SHORT_1": 1, "WAIT": 1},
        "valid": {"ENTER_LONG_1": 1, "ENTER_SHORT_1": 1, "WAIT": 1},
    }
    assert audit["tickers_by_role"] == {"train": {"NQ": 3}, "valid": {"NQ": 3}}
    assert audit["teacher_free_prompt_records"] == 6
    assert (root / "audit.json").is_file()


def test_trade_mastery_audit_rejects_challenge_outcomes(tmp_path):
    root = _dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["lineage"]["supervision_scope"] = "trade_mastery"
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="challenge objective"):
        audit_supervised_dataset(root, specialist_score_mode="out_of_fold")
    assert not (root / "audit.json").exists()


def test_trade_mastery_audit_rejects_account_or_challenge_prompt_state(tmp_path):
    root = _dataset(tmp_path)
    for role in ("train", "valid"):
        path = root / f"{role}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for record in rows:
            for outcome in record["targets"]["outcomes"].values():
                outcome["outcome"] = "failed_target"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["lineage"]["supervision_scope"] = "trade_mastery"
    manifest["files"] = {
        role: file_digest(root / f"{role}.jsonl") for role in ("train", "valid")}
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="account state"):
        audit_supervised_dataset(root, specialist_score_mode="out_of_fold")


@pytest.mark.parametrize("mutation, message", [
    ("future_prompt", "future target leaked"),
    ("teacher_prompt", "specialist field leaked"),
    ("nonfinite_embedding", "nonfinite embedding"),
    ("action_mismatch", "legal action targets differ"),
])
def test_audit_fails_closed_without_publishing_receipt(tmp_path, mutation, message):
    root = _dataset(tmp_path)
    path = root / "train.jsonl"
    def mutate(record):
        prompt = json.loads(record["messages"][1]["content"])
        if mutation == "future_prompt":
            prompt["future_mfe_r"] = 2.0
            record["messages"][1]["content"] = json.dumps(prompt)
        elif mutation == "teacher_prompt":
            prompt["fields"].append("expansion.long_attempt_probability")
            prompt["history_oldest_first"] = [[0.0, 0.0, 0.5], [0.1, 0.2, 0.6]]
            record["messages"][1]["content"] = json.dumps(prompt)
        elif mutation == "nonfinite_embedding":
            record["market_embeddings"][0][0] = float("nan")
        else:
            record["targets"]["action_order"] = ["WAIT", "ENTER_LONG_1"]
            record["targets"]["action_probabilities"] = [0.5, 0.5]
            record["targets"]["outcomes"].pop("ENTER_SHORT_1")
    _mutate_first(path, mutate)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["train"] = file_digest(path)
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        audit_supervised_dataset(root, specialist_score_mode="out_of_fold")
    assert not (root / "audit.json").exists()


def test_audit_refuses_undeclared_specialist_score_semantics(tmp_path):
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match="specialist score mode"):
        audit_supervised_dataset(root, specialist_score_mode="unknown")
    assert not (root / "audit.json").exists()


def test_audit_rejects_nonfinite_specialist_training_target(tmp_path):
    root = _dataset(tmp_path)
    path = root / "train.jsonl"
    _mutate_first(path, lambda record: record["targets"].update(
        specialist_targets={"expansion.long_probability": float("nan")}))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["train"] = file_digest(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="specialist training target"):
        audit_supervised_dataset(root, specialist_score_mode="post_fit")


def test_writer_rejects_overlapping_episode_duplicates_for_the_same_market(tmp_path):
    with pytest.raises(ValueError, match="duplicate supervised state"):
        write_supervised_dataset(
            [_record(100, best=Action.ENTER_LONG_1, source_id="episode-a"),
             _record(100, best=Action.ENTER_LONG_1, source_id="episode-b"),
             _record(300, best=Action.WAIT)],
            tmp_path / "duplicate",
            splits={"train": [0, 200], "valid": [200, 400]},
            lineage={
                "source_identity": "source", "specialist_identities": ["expansion"],
                "economic_contract": {"profit_target": 6000},
                "split_audit": {"status": "PASS"},
            },
            sealed_start_ns=500,
        )


def test_writer_and_audit_distinguish_flat_and_positioned_states_at_same_market_bar(tmp_path):
    root = tmp_path / "same-bar-different-state"
    write_supervised_dataset(
        [_record(100, best=Action.ENTER_LONG_1), _position_record(100),
         _record(300, best=Action.WAIT), _position_record(300, best=Action.CLOSE)],
        root,
        splits={"train": [0, 200], "valid": [200, 400]},
        lineage={
            "source_identity": "source", "specialist_identities": ["expansion"],
            "economic_contract": {"profit_target": 6000},
            "split_audit": {"status": "PASS"},
        },
        sealed_start_ns=500,
    )
    audit = audit_supervised_dataset(root, specialist_score_mode="out_of_fold")
    assert audit["status"] == "PASS"
    assert audit["actions"] == {
        "CLOSE": 1, "ENTER_LONG_1": 1, "HOLD": 1, "WAIT": 1,
    }


def test_writer_uses_compact_embedding_sidecars(tmp_path):
    root = tmp_path / "compact"
    records = [
        _record(100, best=Action.ENTER_LONG_1),
        _record(110, best=Action.ENTER_SHORT_1),
        _record(120, best=Action.WAIT),
        _record(300, best=Action.ENTER_LONG_1),
        _record(310, best=Action.ENTER_SHORT_1),
        _record(320, best=Action.WAIT),
    ]
    manifest = write_supervised_dataset(
        records, root,
        splits={"train": [0, 200], "valid": [200, 400]},
        lineage={
            "source_identity": "source", "specialist_identities": ["expansion"],
            "economic_contract": {"profit_target": 6000},
            "split_audit": {"status": "PASS"},
        },
        sealed_start_ns=500,
        embedding_storage="float32_sidecar_v1",
    )

    row = json.loads((root / "train.jsonl").read_text().splitlines()[0])
    assert "market_embeddings" not in row
    assert "market_available" not in row
    assert row["market_embedding_index"] == 0
    assert manifest["embedding_storage"]["kind"] == "float32_sidecar_v1"
    assert manifest["embedding_storage"]["roles"]["train"]["shape"] == [3, 2, 2]
    assert (root / "train.embeddings.f32").stat().st_size == 3 * 2 * 2 * 4
    assert audit_supervised_dataset(root, specialist_score_mode="out_of_fold")["status"] == "PASS"


def test_writer_indexes_authenticated_frozen_embeddings_without_copying_windows(tmp_path):
    cache = tmp_path / "cache" / "NQ"
    cache.mkdir(parents=True)
    decision_times = (100, 110, 120, 300, 310, 320)
    timestamps = np.asarray(
        [value for decision in decision_times for value in (decision - 1, decision)],
        dtype="datetime64[ns]",
    )
    embeddings = np.tile(np.asarray([[1., 2.], [3., 4.]], np.float32), (6, 1))
    np.save(cache / "timestamps.npy", timestamps)
    np.save(cache / "embeddings.npy", embeddings)
    cache_manifest = {
        "schema": "propevolve_chronos2_embedding_cache_v2", "ticker": "NQ",
        "rows": len(embeddings), "research_end_exclusive": "2026-01-01T00:00:00Z",
        "sealed_holdout_touched": False,
    }
    (cache / "manifest.json").write_text(json.dumps(cache_manifest))
    root = tmp_path / "indexed"
    manifest = write_supervised_dataset(
        [_record(timestamp, best=action) for timestamp, action in zip(decision_times, (
            Action.ENTER_LONG_1, Action.ENTER_SHORT_1, Action.WAIT,
            Action.ENTER_LONG_1, Action.ENTER_SHORT_1, Action.WAIT,
        ))],
        root,
        splits={"train": [0, 200], "valid": [200, 400]},
        lineage={"source_identity": "source", "specialist_identities": ["expansion"],
                 "economic_contract": {"profit_target": 6000},
                 "split_audit": {"status": "PASS"}},
        sealed_start_ns=500,
        embedding_storage="source_embedding_reference_v1",
        embedding_source_cache_root=cache.parent,
    )

    row = json.loads((root / "train.jsonl").read_text().splitlines()[0])
    assert "market_embeddings" not in row
    assert "market_available" not in row
    assert row["market_embedding_reference"] == {
        "ticker": "NQ", "row": 1, "available_count": 2,
    }
    assert manifest["embedding_storage"] == {
        "kind": "source_embedding_reference_v1", "cache_root": str(cache.parent),
        "context_steps": 2, "embedding_dim": 2,
        "sources": {"NQ": {"manifest_sha256": hashlib.sha256(
            (cache / "manifest.json").read_bytes()).hexdigest(), "rows": 12}},
    }
    assert audit_supervised_dataset(root, specialist_score_mode="out_of_fold")["status"] == "PASS"

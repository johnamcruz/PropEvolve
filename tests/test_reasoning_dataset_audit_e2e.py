"""Production-shaped dataset audit boundaries for the reasoning challenger."""

import json

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


def _record(timestamp, *, best):
    rewards = {
        Action.WAIT: 0.0,
        Action.ENTER_LONG_1: 2.0 if best is Action.ENTER_LONG_1 else -1.0,
        Action.ENTER_SHORT_1: 2.0 if best is Action.ENTER_SHORT_1 else -1.0,
    }
    return supervised_record(
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
        source_id=f"source-{timestamp}", continuation_id="frozen-policy",
        target_temperature=1.0,
    )


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    write_supervised_dataset(
        [_record(100, best=Action.ENTER_LONG_1),
         _record(300, best=Action.ENTER_SHORT_1)],
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


def test_audit_publishes_hash_bound_pass_for_causal_teacher_free_records(tmp_path):
    root = _dataset(tmp_path)
    audit = audit_supervised_dataset(root, specialist_score_mode="out_of_fold")

    assert audit["status"] == "PASS"
    assert audit["manifest_sha256"] == file_digest(root / "manifest.json")
    assert audit["sealed_touched"] is False
    assert audit["counts"] == {"train": 1, "valid": 1}
    assert audit["actions"] == {"ENTER_LONG_1": 1, "ENTER_SHORT_1": 1}
    assert audit["teacher_free_prompt_records"] == 2
    assert (root / "audit.json").is_file()


@pytest.mark.parametrize("mutation, message", [
    ("future_prompt", "future target leaked"),
    ("teacher_prompt", "specialist field leaked"),
    ("nonfinite_embedding", "nonfinite embedding"),
    ("action_mismatch", "legal action targets differ"),
])
def test_audit_fails_closed_without_publishing_receipt(tmp_path, mutation, message):
    root = _dataset(tmp_path)
    path = root / "train.jsonl"
    record = json.loads(path.read_text())
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
    path.write_text(json.dumps(record, allow_nan=True) + "\n")
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
    record = json.loads(path.read_text())
    record["targets"]["specialist_targets"] = {"expansion.long_probability": float("nan")}
    path.write_text(json.dumps(record, allow_nan=True) + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["train"] = file_digest(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="specialist training target"):
        audit_supervised_dataset(root, specialist_score_mode="post_fit")

"""Actual simulator -> audited dataset -> encoded tensor batch, all alternatives."""
import copy
import json
import numpy as np
import pytest

from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import supervised_record, write_supervised_dataset
from propevolve.reasoning_policy.labels import label_actions
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import prepare_mlx_view
from propevolve.reasoning_policy.supervised_trainer import pack_examples
from test_reasoning_challenger_e2e import environment, passive_factory
from test_reasoning_token_parity_e2e import ACTION_VERBALIZERS, LiteralTokenizer


def prepared_action_view(tmp_path, *, embeddings=False, tokenizer=None, model="external-runtime", iters=1):
    env = environment()
    labels = label_actions(env, reset_options={"ticker": "NQ", "start": 0}, prefix=(),
        continuation_factory=passive_factory, max_steps=8)
    start = int(env.markets["NQ"].timestamps[0].astype("datetime64[ns]").astype(np.int64))
    history = RollingContext(ContextConfig(3, ("account.realized_pnl_norm",),
        input_mode="embeddings" if embeddings else "specialists"))
    history.append(start, {"account.realized_pnl_norm": 0.}, embedding=np.ones(2) if embeddings else None)
    train = supervised_record(history.snapshot(), labels, source_id="train", continuation_id="fixed",
        target_temperature=1)
    day = 86400 * 10**9
    valid = copy.deepcopy(train)
    valid.update(source_id="valid", completed_at_ns=start+day, label_end_ns=train["label_end_ns"]+day)
    data = tmp_path / "data"
    write_supervised_dataset([train, valid], data,
        splits={"train": [start, start+day], "valid": [start+day, start+2*day]}, sealed_start_ns=start+2*day,
        lineage={"source_identity": "test", "specialist_identities": "test",
                 "economic_contract": "test", "split_audit": "test"})
    (data / "audit.json").write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(data / "manifest.json"),
        "specialist_score_mode": "post_fit", "sealed_touched": False}))
    config = {"model": str(model), "data": str(data), "adapter_path": str(tmp_path / "adapter"),
        "train": True, "fine_tune_type": "lora", "mask_prompt": True, "num_layers": 1,
        "batch_size": 1, "iters": iters, "learning_rate": 1e-5, "max_seq_length": 2048,
        "grad_checkpoint": False, "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False, "input_mode": "embeddings" if embeddings else "specialists",
        "projector": {"embedding_dim": 2, "context_steps": 3, "market_tokens": 2} if embeddings else None,
        "action_verbalizers": ACTION_VERBALIZERS,
        "action_supervision": {"enabled": True, "soft_target_weight": 1., "ranking_weight": 1., "margin": .25}}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config))
    view = tmp_path / "view"
    prepare_mlx_view(path, view, tokenizer=tokenizer or LiteralTokenizer())
    return json.loads((view / "train.jsonl").read_text()), train, config


def test_all_same_state_outcomes_survive_real_preparation(tmp_path):
    prepared, original, config = prepared_action_view(tmp_path, embeddings=True)
    batch = pack_examples([prepared], max_seq_length=2048)
    assert prepared["action_targets"]["names"] == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    assert prepared["action_targets"]["probabilities"] == original["targets"]["action_probabilities"]
    assert batch[0].shape[:2] == (1, 3)
    assert batch[5][0, 1] > batch[5][0, 0] > batch[5][0, 2]
    np.testing.assert_array_equal(batch[-1], [[False, False, True]])
    assert "market_embeddings" not in original["messages"][1]["content"]


def test_full_action_targets_reject_a_legal_mask_mismatch(tmp_path):
    from propevolve.reasoning_policy.supervision import action_targets
    _, record, _ = prepared_action_view(tmp_path)
    prompt = json.loads(record["messages"][1]["content"])
    prompt["legal_actions"] = ["WAIT"]
    record["messages"][1]["content"] = json.dumps(prompt)
    with pytest.raises(ValueError, match="legal actions"):
        action_targets(record)

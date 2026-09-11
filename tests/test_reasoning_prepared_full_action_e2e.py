"""Actual simulator -> audited dataset -> encoded tensor batch, all alternatives."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import supervised_record, write_supervised_dataset
from propevolve.reasoning_policy.labels import label_actions
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import PreparedDataset, prepare_mlx_view
from propevolve.reasoning_policy.supervised_trainer import pack_examples
from test_reasoning_challenger_e2e import environment, passive_factory
from test_reasoning_token_parity_e2e import ACTION_VERBALIZERS, LiteralTokenizer


def _embedding_cache(tmp_path, *, ticker="NQ", rows=8, width=2):
    root = tmp_path / "embedding-cache" / ticker
    root.mkdir(parents=True)
    embeddings = np.arange(rows * width, dtype=np.float32).reshape(rows, width)
    timestamps = np.arange(100, 100 + rows, dtype="datetime64[ns]")
    np.save(root / "embeddings.npy", embeddings)
    np.save(root / "timestamps.npy", timestamps)
    manifest = {
        "schema": "propevolve_chronos2_embedding_cache_v2",
        "ticker": ticker,
        "rows": rows,
        "research_end_exclusive": "2026-01-01T00:00:00Z",
        "sealed_holdout_touched": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, embeddings, timestamps, hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()


def prepared_action_view(tmp_path, *, embeddings=False, tokenizer=None, model="external-runtime",
                         iters=1, embedding_storage="json", causal_state=False,
                         state_defaults=None):
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
                 "economic_contract": "test", "split_audit": "test"},
        embedding_storage=embedding_storage)
    (data / "audit.json").write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(data / "manifest.json"),
        "specialist_score_mode": "post_fit", "sealed_touched": False}))
    config = {"model": str(model), "data": str(data), "adapter_path": str(tmp_path / "adapter"),
        "train": True, "fine_tune_type": "lora", "mask_prompt": True, "num_layers": 1,
        "batch_size": 1, "iters": iters, "learning_rate": 1e-5, "max_seq_length": 2048,
        "grad_checkpoint": False, "grad_accumulation_steps": 1,
        "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False, "input_mode": "embeddings" if embeddings else "specialists",
        "projector": ({"embedding_dim": 2, "context_steps": 3, "market_tokens": 2,
                       "temporal_encoding": "pooled_levels",
                       **({"state_fields": (["account.realized_pnl_norm"] if causal_state
                                             else list(state_defaults)),
                           "state_scales": [1.0]} if causal_state or state_defaults else {})}
                      if embeddings else None),
        "prepared_state_defaults": state_defaults,
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


def test_prepared_embedding_view_carries_configured_causal_state_as_numeric_input(tmp_path):
    prepared, _, _ = prepared_action_view(tmp_path, embeddings=True, causal_state=True)
    batch = pack_examples([prepared], max_seq_length=2048)

    assert prepared["causal_state"] == [0.0]
    np.testing.assert_array_equal(batch[-3], [[0.0]])
    np.testing.assert_array_equal(batch[-1], [[False, False, True]])


def test_market_only_preparation_uses_only_explicit_configured_state_defaults(tmp_path):
    prepared, _, _ = prepared_action_view(
        tmp_path, embeddings=True, state_defaults={"trade.current_r": 0.0})
    assert prepared["causal_state"] == [0.0]


def test_prepared_dataset_lazily_resolves_compact_embedding_sidecars(tmp_path):
    prepared, original, _ = prepared_action_view(
        tmp_path, embeddings=True, embedding_storage="float32_sidecar_v1")
    dataset = PreparedDataset(tmp_path / "view", "train")
    loaded = dataset[0]

    assert loaded["market_embeddings"].tolist() == original["market_embeddings"]
    assert loaded["market_available"].tolist() == original["market_available"]
    assert "market_embeddings" not in prepared


def test_prepared_rows_stream_without_reading_whole_file_and_preserve_batch_values(tmp_path, monkeypatch):
    prepared, _, _ = prepared_action_view(tmp_path, embeddings=True)
    view = tmp_path / "view"
    # Filesystem boundary: prohibit whole-file text reads of the row corpus.
    original = Path.read_text
    def bounded_read(path, *args, **kwargs):
        if path.suffix == ".jsonl":
            raise AssertionError("row corpus must not be read into RAM")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", bounded_read)
    dataset = PreparedDataset(view, "train")
    assert len(dataset) == 1
    expected = pack_examples([prepared], max_seq_length=2048)
    actual = pack_examples([dataset[0]], max_seq_length=2048)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)
    assert list(dataset.sampling_rows()) == [prepared]
    with pytest.raises(IndexError):
        dataset[1]


def test_prepared_dataset_reconstructs_exact_window_from_frozen_source_reference(tmp_path):
    cache, embeddings, timestamps, digest = _embedding_cache(tmp_path)
    prepared, original, config = prepared_action_view(tmp_path, embeddings=True)
    data = Path(config["data"])
    manifest = json.loads((data / "manifest.json").read_text())
    for role in ("train", "valid"):
        rows = [json.loads(line) for line in (data / f"{role}.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("market_embeddings")
            row.pop("market_available")
            row["market_embedding_reference"] = {
                "ticker": "NQ", "row": 2, "available_count": 3,
            }
        (data / f"{role}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["files"][role] = file_digest(data / f"{role}.jsonl")
    manifest["embedding_storage"] = {
        "kind": "source_embedding_reference_v1",
        "context_steps": 3,
        "embedding_dim": 2,
        "cache_root": str(cache.parent),
        "sources": {"NQ": {"manifest_sha256": digest, "rows": len(embeddings)}},
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    (data / "audit.json").write_text(json.dumps({
        "status": "PASS", "manifest_sha256": file_digest(data / "manifest.json"),
        "specialist_score_mode": "post_fit", "sealed_touched": False,
    }))
    view = tmp_path / "indexed-view"
    prepare_mlx_view(tmp_path / "recipe.json", view, tokenizer=LiteralTokenizer())
    row = PreparedDataset(view, "train")[0]

    np.testing.assert_array_equal(row["market_embeddings"], embeddings[:3])
    np.testing.assert_array_equal(row["market_available"], [True, True, True])
    assert "market_embeddings" not in json.loads((view / "train.jsonl").read_text())


def test_full_action_targets_reject_a_legal_mask_mismatch(tmp_path):
    from propevolve.reasoning_policy.supervision import action_targets
    _, record, _ = prepared_action_view(tmp_path)
    prompt = json.loads(record["messages"][1]["content"])
    prompt["legal_actions"] = ["WAIT"]
    record["messages"][1]["content"] = json.dumps(prompt)
    with pytest.raises(ValueError, match="legal actions"):
        action_targets(record)

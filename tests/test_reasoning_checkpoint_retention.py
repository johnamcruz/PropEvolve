"""Bounded, owned checkpoint cleanup; only explicit artifacts may be removed."""
import json
import pytest
from propevolve.reasoning_policy.checkpoints import seal_checkpoint, prune_checkpoints


def test_cleanup_keeps_latest_and_leaves_foreign_files_and_parent_untouched(tmp_path):
    contract = {"source": "fixture"}
    def snapshot(name, group, selected_contract):
        path = tmp_path / name
        path.mkdir()
        for file in ("adapters.safetensors", "training.safetensors"):
            (path / file).write_bytes(b"external-storage-fixture")
        (path / "adapter_config.json").write_text("{}")
        (path / "training.json").write_text(json.dumps({"runtime": {"next_group": group}}))
        (path / "rl_receipt.json").write_text(json.dumps({"contract": selected_contract}))
        seal_checkpoint(path)
        return path
    parent = snapshot("parent", 0, contract)
    old = snapshot("old", 1, contract)
    newest = snapshot("newest", 2, contract)
    foreign = snapshot("other-run", 3, {"source": "other"})
    extra = snapshot("contains-user-file", 4, contract)
    (extra / "keep.txt").write_text("not owned by checkpoint manager")
    removed = prune_checkpoints(tmp_path, keep=1, contract=contract, protected=[parent])
    assert removed == ["old"]
    assert not old.exists()
    assert all(path.exists() for path in (parent, newest, foreign, extra))


def test_embedding_checkpoint_requires_projector_but_legacy_policy_does_not(tmp_path):
    from propevolve.reasoning_policy.checkpoints import verify_checkpoint
    for name in ("adapters.safetensors", "training.json", "training.safetensors"):
        (tmp_path / name).write_bytes(b"storage-fixture")
    (tmp_path / "rl_receipt.json").write_text(json.dumps({"contract": {}}))
    metadata = tmp_path / "adapter_config.json"
    metadata.write_text(json.dumps({"input_mode": "embeddings"}))
    seal_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="projector"):
        verify_checkpoint(tmp_path)
    (tmp_path / "checkpoint_manifest.json").unlink()
    (tmp_path / "projector.safetensors").write_bytes(b"projector-fixture")
    seal_checkpoint(tmp_path)
    assert verify_checkpoint(tmp_path) == {"contract": {}}
    (tmp_path / "checkpoint_manifest.json").unlink()
    (tmp_path / "projector.safetensors").unlink()
    metadata.write_text(json.dumps({"input_mode": "specialists"}))
    seal_checkpoint(tmp_path)
    assert verify_checkpoint(tmp_path) == {"contract": {}}

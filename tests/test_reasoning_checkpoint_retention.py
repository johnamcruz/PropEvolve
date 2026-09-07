"""Bounded, owned checkpoint cleanup; only explicit artifacts may be removed."""
import json
from propevolve.reasoning_policy.checkpoints import seal_checkpoint, prune_checkpoints


def test_cleanup_keeps_latest_and_leaves_foreign_files_and_parent_untouched(tmp_path):
    contract = {"source": "fixture"}
    def snapshot(name, group, selected_contract):
        path = tmp_path / name
        path.mkdir()
        for file in ("adapters.safetensors", "adapter_config.json", "training.safetensors"):
            (path / file).write_bytes(b"external-storage-fixture")
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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_only_verified_matching_expansion_teacher_is_packaged() -> None:
    root = Path("models/teachers")
    manifest = json.loads((root / "manifest.json").read_text())
    expansion = manifest["expansion"]
    checkpoint = root / expansion["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert manifest["status"] == "verified_checkpoint_only"
    assert manifest["teacher_cache_status"] == "must_be_recreated_from_checkpoint"
    assert manifest["pivot"] is None
    assert _sha256(checkpoint) == expansion["checkpoint_sha256"]
    assert payload["schema"] == expansion["artifact_schema"]
    assert payload["training_schema"] == expansion["training_schema"]
    assert payload["lineage"]["encoder_sha256"] == (
        expansion["encoder_checkpoint_sha256"]
    )
    assert payload["lineage"]["encoder_identity_sha256"] == (
        expansion["encoder_identity_sha256"]
    )
    assert payload["lineage"]["dataset_lineage_sha256"] == (
        expansion["dataset_lineage_sha256"]
    )
    assert set(payload["lineage"]["cache_sha256s"]) == set(
        expansion["cache_sha256s"].values()
    )
    assert set(expansion["universe"]) == {
        "CL", "ES", "GC", "NQ", "RTY", "SI", "YM", "ZB", "ZN"
    }

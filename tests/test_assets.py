from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from propevolve.assets import AssetContract, link_local_assets


def test_linked_assets_resolve_without_copying_and_are_authenticated(tmp_path: Path) -> None:
    source_data = tmp_path / "source-data"
    source_data.mkdir()
    (source_data / "NQ_3min.csv").write_text("datetime,open,high,low,close,volume\n")
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"mask-adapter")
    (checkpoint / "adapter_config.json").write_text("{}\n")
    embedding_cache = tmp_path / "source-embedding-cache"
    embedding_cache.mkdir()

    contract = link_local_assets(
        workspace=tmp_path / "workspace",
        market_data=source_data,
        checkpoint=checkpoint,
        embedding_cache=embedding_cache,
    )

    assert (tmp_path / "workspace/data/ohlcv").is_symlink()
    assert (tmp_path / "workspace/checkpoints/chronos2_mask_full").is_symlink()
    assert contract.embedding_cache == str(embedding_cache.resolve())
    assert contract == AssetContract.load(tmp_path / "workspace/config/local-assets.json")
    assert contract.checkpoint_sha256 == hashlib.sha256(b"mask-adapter").hexdigest()
    contract.verify()


def test_asset_contract_fails_closed_after_checkpoint_drift(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"stable")
    (checkpoint / "adapter_config.json").write_text("{}")
    contract = link_local_assets(tmp_path / "workspace", data, checkpoint)

    weights.write_bytes(b"drifted")

    with pytest.raises(ValueError, match="checkpoint identity drift"):
        contract.verify()

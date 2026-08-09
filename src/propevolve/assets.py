"""Authenticated local asset links for data and frozen FFM checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetContract:
    schema: str
    market_data: str
    checkpoint: str
    checkpoint_sha256: str
    adapter_config_sha256: str
    embedding_cache: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "AssetContract":
        payload = json.loads(Path(path).read_text())
        return cls(**payload)

    def verify(self) -> None:
        market_data = Path(self.market_data)
        checkpoint = Path(self.checkpoint)
        weights = checkpoint / "adapter_model.safetensors"
        config = checkpoint / "adapter_config.json"
        if not market_data.is_dir():
            raise ValueError(f"market data link target is unavailable: {market_data}")
        if not weights.is_file() or not config.is_file():
            raise ValueError(f"frozen checkpoint is incomplete: {checkpoint}")
        if self.embedding_cache is not None and not Path(self.embedding_cache).is_dir():
            raise ValueError(
                f"embedding cache target is unavailable: {self.embedding_cache}"
            )
        if _sha256(weights) != self.checkpoint_sha256:
            raise ValueError("checkpoint identity drift")
        if _sha256(config) != self.adapter_config_sha256:
            raise ValueError("adapter config identity drift")


def _ensure_link(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=False) != target:
            raise ValueError(f"existing link points elsewhere: {link}")
        return
    if link.exists():
        raise ValueError(f"refusing to replace existing asset path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link, target_is_directory=True)


def link_local_assets(
    workspace: str | Path,
    market_data: str | Path,
    checkpoint: str | Path,
    embedding_cache: str | Path | None = None,
) -> AssetContract:
    """Create local symlinks and seal the exact resolved asset identities."""
    workspace = Path(workspace)
    source_data = Path(market_data).resolve(strict=True)
    source_checkpoint = Path(checkpoint).resolve(strict=True)
    source_embedding_cache = (
        Path(embedding_cache).resolve(strict=True)
        if embedding_cache is not None
        else None
    )
    weights = source_checkpoint / "adapter_model.safetensors"
    adapter_config = source_checkpoint / "adapter_config.json"
    if not source_data.is_dir():
        raise ValueError(f"market data is not a directory: {source_data}")
    if not weights.is_file() or not adapter_config.is_file():
        raise ValueError(f"frozen checkpoint is incomplete: {source_checkpoint}")

    data_link = workspace / "data/ohlcv"
    checkpoint_link = workspace / "checkpoints/chronos2_mask_full"
    _ensure_link(data_link, source_data)
    _ensure_link(checkpoint_link, source_checkpoint)
    contract = AssetContract(
        schema="propevolve_local_assets_v1",
        market_data=str(source_data),
        checkpoint=str(source_checkpoint),
        checkpoint_sha256=_sha256(weights),
        adapter_config_sha256=_sha256(adapter_config),
        embedding_cache=(
            str(source_embedding_cache) if source_embedding_cache is not None else None
        ),
    )
    destination = workspace / "config/local-assets.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(contract), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return contract

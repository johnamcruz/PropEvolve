from __future__ import annotations

from pathlib import Path

from propevolve.cli import main


def test_setup_assets_command_creates_links_without_copying(tmp_path: Path) -> None:
    data = tmp_path / "source-data"
    data.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text("{}")
    workspace = tmp_path / "workspace"

    code = main([
        "setup-assets",
        "--workspace", str(workspace),
        "--market-data", str(data),
        "--checkpoint", str(checkpoint),
    ])

    assert code == 0
    assert (workspace / "data/ohlcv").is_symlink()
    assert (workspace / "checkpoints/chronos2_mask_full").is_symlink()


def test_validate_config_command_accepts_promoted_recipe() -> None:
    assert main(["validate-config", "--config", "config/historical_mask_v1.json"]) == 0


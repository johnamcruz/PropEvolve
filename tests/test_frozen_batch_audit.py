from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_frozen_batch_audit_cli_is_path_driven() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "audit_frozen_checkpoint_batch.py"),
            "--help",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--attempt-dir" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--replay-root" in result.stdout
    assert "--output" in result.stdout
    assert "--near-blow-pnl" in result.stdout
    assert "--pair-count" in result.stdout

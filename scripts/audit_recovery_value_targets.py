#!/usr/bin/env python3
"""Audit a recovery checkpoint's training-only action-value targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from export_recovery_success_replay import (
    _load_checkpoint_without_array_buffers,
)
from propevolve.recovery import audit_recovery_action_values


def audit_checkpoint(checkpoint: Path) -> dict[str, object]:
    payload = _load_checkpoint_without_array_buffers(checkpoint.resolve(strict=True))
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("recovery checkpoint manifest is missing")
    store = manifest.get("recovery_value_store_state")
    if not isinstance(store, Mapping):
        raise ValueError("recovery checkpoint has no target store")
    targets = store.get("targets")
    if not isinstance(targets, list):
        raise ValueError("recovery checkpoint target store is invalid")
    action_values = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("recovery checkpoint target row is invalid")
        action_values.append(tuple(target.get("action_values", ())))
    audit = audit_recovery_action_values(tuple(action_values))
    return {
        "schema": "propevolve_recovery_target_audit_v1",
        "checkpoint": checkpoint.name,
        "total": audit.total,
        "discriminative": audit.discriminative,
        "ambiguous": audit.ambiguous,
        "all_blow": audit.all_blow,
        "all_recover": audit.all_recover,
        "wait_best": audit.wait_best,
        "long_best": audit.long_best,
        "short_best": audit.short_best,
        "valid_for_training": audit.valid_for_training,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    report = audit_checkpoint(arguments.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid_for_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

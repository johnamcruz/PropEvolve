"""Durable paper/shadow episodes for offline challenger training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .decision import Action
from .replay import Episode, Transition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShadowEpisodeStore:
    """Append authenticated completed episodes; never mutate live weights."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def append(
        self,
        episode: Episode,
        *,
        policy_sha256: str,
        checkpoint_sha256: str,
    ) -> dict:
        if not episode.transitions:
            raise ValueError("cannot store an empty shadow episode")
        self.root.mkdir(parents=True, exist_ok=True)
        data_path = self.root / f"{episode.episode_id}.npz"
        receipt_path = self.root / f"{episode.episode_id}.json"
        if data_path.exists() or receipt_path.exists():
            raise ValueError(f"shadow episode already exists: {episode.episode_id}")
        temporary = self.root / f".{episode.episode_id}.tmp.npz"
        np.savez(
            temporary,
            observations=np.stack([item.observation for item in episode.transitions]),
            actions=np.asarray([int(item.action) for item in episode.transitions], np.int16),
            rewards=np.asarray([item.reward for item in episode.transitions], np.float32),
            next_observations=np.stack([
                item.next_observation for item in episode.transitions
            ]),
            terminated=np.asarray([
                item.terminated for item in episode.transitions
            ], np.bool_),
            valid_masks=np.asarray([[
                action in item.valid_actions for action in Action
            ] for item in episode.transitions], np.bool_),
            next_valid_masks=np.asarray([[
                action in item.next_valid_actions for action in Action
            ] for item in episode.transitions], np.bool_),
        )
        temporary.replace(data_path)
        receipt = {
            "schema": "propevolve_shadow_episode_v1",
            "episode_id": episode.episode_id,
            "ticker": episode.ticker,
            "outcome": episode.outcome,
            "primary_side": episode.primary_side,
            "ended_at_ns": episode.ended_at_ns,
            "rows": len(episode.transitions),
            "policy_sha256": str(policy_sha256),
            "checkpoint_sha256": str(checkpoint_sha256),
            "data": data_path.name,
            "data_sha256": _sha256(data_path),
        }
        temporary_receipt = receipt_path.with_suffix(".json.tmp")
        temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary_receipt.replace(receipt_path)
        return receipt

    def load(self, receipt: dict | str | Path) -> Episode:
        if not isinstance(receipt, dict):
            receipt = json.loads(Path(receipt).read_text())
        if receipt.get("schema") != "propevolve_shadow_episode_v1":
            raise ValueError("unsupported shadow episode receipt")
        data_path = self.root / str(receipt["data"])
        if _sha256(data_path) != receipt.get("data_sha256"):
            raise ValueError("shadow episode data identity drift")
        with np.load(data_path, allow_pickle=False) as values:
            rows = len(values["actions"])
            if rows != int(receipt["rows"]):
                raise ValueError("shadow episode row count drift")
            transitions = []
            for index in range(rows):
                valid = tuple(
                    action for action in Action if values["valid_masks"][index, int(action)]
                )
                next_valid = tuple(
                    action for action in Action
                    if values["next_valid_masks"][index, int(action)]
                )
                transitions.append(Transition(
                    observation=values["observations"][index].copy(),
                    action=Action(int(values["actions"][index])),
                    reward=float(values["rewards"][index]),
                    next_observation=values["next_observations"][index].copy(),
                    terminated=bool(values["terminated"][index]),
                    valid_actions=valid,
                    next_valid_actions=next_valid,
                ))
        return Episode(
            episode_id=str(receipt["episode_id"]),
            ticker=str(receipt["ticker"]),
            outcome=str(receipt["outcome"]),
            primary_side=str(receipt["primary_side"]),
            ended_at_ns=int(receipt["ended_at_ns"]),
            transitions=tuple(transitions),
        )


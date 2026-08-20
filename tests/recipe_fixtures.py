from __future__ import annotations

import json
from pathlib import Path

from propevolve.balance_aware_regime_selectivity import (
    PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
)


def retained_stage2_recipes() -> tuple[Path, ...]:
    """Discover retained Stage 2 recipes by serialized contract values."""

    recipes = []
    for path in Path("config").glob("*.json"):
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") == "propevolve_historical_training_v1"
            and isinstance(payload.get("regime_selectivity"), dict)
        ):
            recipes.append(path)
    if not recipes:
        raise AssertionError("at least one Stage 2 recipe must be retained")
    return tuple(sorted(recipes))


def stage2_recipe(*, semantics: str, training_episodes: int) -> Path:
    candidates = []
    for path in retained_stage2_recipes():
        payload = json.loads(path.read_text())
        stages = payload.get("campaign", {}).get("budget_stages", ())
        if (
            payload["regime_selectivity"].get("semantics") == semantics
            and len(stages) == 1
            and stages[0].get("training_episodes") == training_episodes
            and payload.get("recovery_curriculum") is None
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise AssertionError(
            "expected one retained Stage 2 recipe for "
            f"semantics={semantics!r}, training_episodes={training_episodes}; "
            f"got {tuple(candidates)}"
        )
    return candidates[0]


def paired_aplus_recipe(training_episodes: int) -> Path:
    return stage2_recipe(
        semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        training_episodes=training_episodes,
    )


def paired_recurrent_aplus_recipe(training_episodes: int) -> Path:
    return stage2_recipe(
        semantics=PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        training_episodes=training_episodes,
    )


def retained_sweep_recipe() -> Path:
    candidates = tuple(sorted(Path("config/sweeps").glob("*.json")))
    if len(candidates) != 1:
        raise AssertionError(f"expected one retained sweep recipe, got {candidates}")
    return candidates[0]

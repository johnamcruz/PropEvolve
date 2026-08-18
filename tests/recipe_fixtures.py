from __future__ import annotations

from pathlib import Path
import re


_STAGE2_VERSION = re.compile(r"_stage2_v(?P<version>\d+)")


def retained_stage2_recipes() -> dict[int, tuple[Path, ...]]:
    """Discover the intentionally retained Stage 2 recipe families."""

    recipes: dict[int, list[Path]] = {}
    for path in Path("config").glob("historical_mask_expansion*_stage2_v*.json"):
        match = _STAGE2_VERSION.search(path.name)
        if match is None:
            continue
        recipes.setdefault(int(match.group("version")), []).append(path)
    if not recipes or min(recipes) < 19:
        raise AssertionError("only Stage 2 v19+ recipes may be retained")
    return {
        version: tuple(sorted(paths))
        for version, paths in sorted(recipes.items())
    }


def stage2_recipe(version: int, *, contains: str | None = None) -> Path:
    candidates = retained_stage2_recipes().get(version, ())
    if contains is not None:
        candidates = tuple(path for path in candidates if contains in path.name)
    if len(candidates) != 1:
        raise AssertionError(
            f"expected one retained Stage 2 v{version} recipe, got {candidates}"
        )
    return candidates[0]


def retained_sweep_recipe() -> Path:
    candidates = tuple(sorted(Path("config/sweeps").glob("*.json")))
    if len(candidates) != 1:
        raise AssertionError(f"expected one retained sweep recipe, got {candidates}")
    return candidates[0]

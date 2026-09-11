"""Authenticated rotating mistake/retention sampling for trade-mastery SFT."""

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .integrity import file_digest


_SETTING_KEYS = {
    "assessment_path", "scores_sha256", "summary_sha256",
    "rows_per_group", "mistake_fraction", "seed",
}
_PRIORITY_KEYS = {"priority_actions", "priority_multiplier"}


def validate_targeted_sampling(settings):
    if settings is None:
        return
    keys = set(settings) if isinstance(settings, dict) else set()
    if (not isinstance(settings, dict)
            or keys != _SETTING_KEYS | _PRIORITY_KEYS
            or not isinstance(settings["assessment_path"], str)
            or not settings["assessment_path"].strip()
            or any(not isinstance(settings[name], str) or len(settings[name]) != 64
                   or any(character not in "0123456789abcdef"
                          for character in settings[name])
                   for name in ("scores_sha256", "summary_sha256"))
            or type(settings["rows_per_group"]) is not int
            or settings["rows_per_group"] < 2
            or type(settings["seed"]) is not int or settings["seed"] < 0
            or isinstance(settings["mistake_fraction"], bool)
            or not isinstance(settings["mistake_fraction"], (int, float))
            or not math.isfinite(float(settings["mistake_fraction"]))
            or not 0 < settings["mistake_fraction"] < 1
            or (not isinstance(settings["priority_actions"], list)
                or any(not isinstance(name, str) or not name
                       for name in settings["priority_actions"])
                or len(set(settings["priority_actions"]))
                    != len(settings["priority_actions"])
                or type(settings["priority_multiplier"]) is not int
                or settings["priority_multiplier"] < 1)):
        raise ValueError("invalid targeted sampling configuration")


def _validated_groups(scored, settings, *, train_bounds, expected_rows=None):
    validate_targeted_sampling(settings)
    lower, upper = train_bounds
    if type(lower) is not int or type(upper) is not int or lower >= upper:
        raise ValueError("invalid training bounds")
    groups = defaultdict(lambda: ([], []))
    seen = set()
    for row in scored:
        try:
            index = row["index"]
            timestamp = row["completed_at_ns"]
            advantage = row["target_advantage"]
            ticker = row["ticker"]
            target = row["target"]
        except (KeyError, TypeError) as error:
            raise ValueError("invalid training assessment row") from error
        if (type(index) is not int or index < 0 or index in seen
                or type(timestamp) is not int or not lower <= timestamp < upper
                or isinstance(advantage, bool)
                or not isinstance(advantage, (int, float))
                or not math.isfinite(float(advantage))
                or not isinstance(ticker, str) or not ticker
                or not isinstance(target, str) or not target):
            raise ValueError("invalid or non-training assessment row")
        seen.add(index)
        year = str(np.datetime64(timestamp, "ns"))[:4]
        groups[(ticker, target, year)][int(advantage >= 0)].append(index)
    if not seen:
        raise ValueError("empty training assessment")
    if expected_rows is not None and seen != set(range(expected_rows)):
        raise ValueError("training assessment does not cover the prepared role exactly")
    return groups


class TargetedSampler:
    """Rotate mistakes and retained examples, then balance every action."""

    def __init__(self, scored, settings, *, train_bounds, expected_rows=None):
        groups = _validated_groups(
            scored, settings, train_bounds=train_bounds, expected_rows=expected_rows)
        self.evidence = {int(row["index"]): {
            "source_id": row.get("source_id"),
            "completed_at_ns": int(row["completed_at_ns"]),
            "ticker": row["ticker"],
            "target": row["target"],
            "predicted": row.get("predicted"),
            "target_advantage": float(row["target_advantage"]),
            "scores": row.get("scores"),
        } for row in scored}
        self.seed = settings["seed"]
        self.quota = settings["rows_per_group"]
        self.mistake_fraction = float(settings["mistake_fraction"])
        self.priority_actions = frozenset(settings["priority_actions"])
        self.priority_multiplier = settings["priority_multiplier"]
        rng = np.random.default_rng(self.seed)
        self.groups = {
            key: tuple(rng.permutation(np.asarray(part, dtype=np.int64)) for part in parts)
            for key, parts in sorted(groups.items())
        }
        self.pool_rows = sum(len(part) for parts in self.groups.values() for part in parts)
        selected = self._selected(0)
        counts = defaultdict(int)
        for _, target in selected:
            counts[target] += 1
        if not counts:
            raise ValueError("targeted sampling produced no action classes")
        self.action_names = tuple(sorted(counts))
        if not self.priority_actions <= set(self.action_names):
            raise ValueError("targeted sampling priority action is absent")
        self.rows_per_action = max(counts.values())
        self.action_draws = {
            name: self.rows_per_action * (
                self.priority_multiplier if name in self.priority_actions else 1)
            for name in self.action_names
        }
        self.round_rows = sum(self.action_draws.values())

    @classmethod
    def from_assessment(cls, settings, *, view_manifest_path, train_bounds,
                        expected_rows):
        validate_targeted_sampling(settings)
        root = Path(settings["assessment_path"])
        summary_path = root / "summary.json"
        scores_path = root / "scores.jsonl"
        if (file_digest(summary_path) != settings["summary_sha256"]
                or file_digest(scores_path) != settings["scores_sha256"]):
            raise ValueError("targeted assessment artifact changed")
        summary = json.loads(summary_path.read_text())
        if (summary.get("role") != "train" or summary.get("weights_updated") is not False
                or summary.get("rows") != expected_rows
                or summary.get("view_manifest_sha256") != file_digest(view_manifest_path)):
            raise ValueError("targeted assessment does not match the training view")
        with scores_path.open() as stream:
            scored = [json.loads(line) for line in stream if line.strip()]
        return cls(scored, settings, train_bounds=train_bounds,
                   expected_rows=expected_rows)

    @staticmethod
    def _rotate(pool, count, offset):
        if not len(pool) or count < 1:
            return []
        return [int(pool[(offset + index) % len(pool)]) for index in range(count)]

    def _selected(self, round_index):
        selected = []
        desired_mistakes = max(
            1, min(self.quota - 1, round(self.quota * self.mistake_fraction)))
        for (_, target, _), (mistakes, retained) in self.groups.items():
            count = min(self.quota, len(mistakes) + len(retained))
            mistake_count = min(desired_mistakes, len(mistakes), count)
            retained_count = min(count - mistake_count, len(retained))
            remaining = count - mistake_count - retained_count
            if remaining:
                extra_mistakes = min(remaining, len(mistakes) - mistake_count)
                mistake_count += extra_mistakes
                retained_count += remaining - extra_mistakes
            mistake_offset = round_index * max(1, mistake_count)
            retained_offset = round_index * max(1, retained_count)
            indices = self._rotate(mistakes, mistake_count, mistake_offset)
            indices += self._rotate(retained, retained_count, retained_offset)
            selected.extend((index, target) for index in indices)
        return selected

    def order(self, round_index):
        if type(round_index) is not int or round_index < 0:
            raise ValueError("targeted sampling round must be nonnegative")
        by_action = defaultdict(list)
        for index, target in self._selected(round_index):
            by_action[target].append(index)
        if set(by_action) != set(self.action_names):
            raise ValueError("targeted sampling lost an action class")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, round_index]))
        queues = {name: list(rng.permutation(by_action[name]))
                  for name in self.action_names}
        order = []
        for position in range(max(self.action_draws.values())):
            for name in self.action_names:
                if position >= self.action_draws[name]:
                    continue
                queue = queues[name]
                order.append(int(queue[position % len(queue)]))
        return np.asarray(order, dtype=np.int64)

    def selection_receipt(self, round_index):
        """Describe exact mistake corrections and retained anchors for one round."""
        order = tuple(map(int, self.order(round_index)))
        kinds = {}
        years = {}
        for (_, _, year), (mistakes, retained) in self.groups.items():
            for index in map(int, mistakes):
                kinds[index], years[index] = "mistake", year
            for index in map(int, retained):
                kinds[index], years[index] = "anchor", year
        per_action = defaultdict(lambda: {"mistake_draws": 0, "anchor_draws": 0})
        per_ticker = defaultdict(lambda: {"mistake_draws": 0, "anchor_draws": 0})
        draws = []
        for index in order:
            evidence = self.evidence[index]
            kind = kinds[index]
            field = f"{kind}_draws"
            per_action[evidence["target"]][field] += 1
            per_ticker[evidence["ticker"]][field] += 1
            predicted = evidence["predicted"]
            target = evidence["target"]
            feedback = (
                f"incorrect: predicted {predicted}; target is {target}"
                if kind == "mistake" else
                f"correct: retain {target} above alternatives"
            )
            draws.append({**evidence, "index": index, "year": years[index],
                          "kind": kind, "feedback": feedback})
        mistake_draws = sum(row["mistake_draws"] for row in per_action.values())
        anchor_draws = sum(row["anchor_draws"] for row in per_action.values())
        return {
            "round": round_index,
            "draw_count": len(order),
            "unique_rows": len(set(order)),
            "mistake_draws": mistake_draws,
            "anchor_draws": anchor_draws,
            "per_action": dict(sorted(per_action.items())),
            "per_ticker": dict(sorted(per_ticker.items())),
            "draws": draws,
        }


def select_training_indices(scored, settings, *, train_bounds):
    """Compatibility helper exposing the first deterministic targeted round."""
    sampler = TargetedSampler(scored, settings, train_bounds=train_bounds)
    return sorted(set(map(int, sampler.order(0))))

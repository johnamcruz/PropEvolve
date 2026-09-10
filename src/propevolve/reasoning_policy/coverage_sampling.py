"""Deterministic rotating training coverage; no validation-based selection."""
from array import array
from collections import defaultdict
import numpy as np
import json


def validate_prepared_sampling(settings):
    if settings is None:
        return
    if (not isinstance(settings, dict) or set(settings) != {"fields", "rows_per_stratum", "seed"}
            or type(settings["seed"]) is not int or settings["seed"] < 0
            or not isinstance(settings["rows_per_stratum"], dict)
            or set(settings["rows_per_stratum"]) != {"train", "valid"}):
        raise ValueError("invalid prepared sampling configuration")
    for count in settings["rows_per_stratum"].values():
        validate_coverage({"fields": settings["fields"], "rows_per_stratum": count,
                           "probability_bins": {}})


def select_prepared_rows(path, settings, *, role):
    validate_prepared_sampling(settings)
    coverage = {"fields": settings["fields"], "rows_per_stratum": settings["rows_per_stratum"][role],
                "probability_bins": {}}
    with path.open() as stream:
        rows = ({"coverage": coverage_metadata(json.loads(line), coverage)} for line in stream)
        sampler = CoverageSampler(rows, coverage, seed=settings["seed"])
    return set(map(int, sampler.order(0)))


def validate_coverage(settings):
    if settings is None:
        return
    if (not isinstance(settings, dict) or set(settings) != {
            "fields", "probability_bins", "rows_per_stratum"}
            or type(settings["rows_per_stratum"]) is not int
            or settings["rows_per_stratum"] < 1
            or not isinstance(settings["fields"], list) or not settings["fields"]
            or any(field not in {"ticker", "year", "month"} for field in settings["fields"])
            or len(set(settings["fields"])) != len(settings["fields"])
            or not isinstance(settings["probability_bins"], dict)):
        raise ValueError("invalid coverage sampling configuration")
    for name, edges in settings["probability_bins"].items():
        if (not isinstance(name, str) or not name or not isinstance(edges, list)
                or not edges or any(isinstance(x, bool) or not isinstance(x, (int, float))
                    or not np.isfinite(x) or not 0 < x < 1 for x in edges)
                or edges != sorted(set(edges))):
            raise ValueError("coverage probability bins must be increasing inside (0, 1)")


def coverage_metadata(record, settings):
    validate_coverage(settings)
    date = str(np.datetime64(record["completed_at_ns"], "ns"))
    metadata = {"ticker": record.get("ticker"), "year": date[:4], "month": date[:7]}
    if not isinstance(metadata["ticker"], str) or not metadata["ticker"]:
        raise ValueError("coverage sampling requires an authenticated ticker")
    for name in settings["probability_bins"]:
        value = record.get("targets", {}).get("specialist_targets", {}).get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("coverage sampling requires valid teacher probabilities")
        metadata[name] = value
    return metadata


class CoverageSampler:
    def __init__(self, rows, settings, *, seed):
        validate_coverage(settings)
        self.seed, self.quota = seed, settings["rows_per_stratum"]
        grouped = defaultdict(lambda: array("Q"))
        for index, row in enumerate(rows):
            metadata = row["coverage"]
            key = tuple(metadata[field] for field in settings["fields"])
            key += tuple(int(np.searchsorted(edges, metadata[name], side="right"))
                         for name, edges in settings["probability_bins"].items())
            grouped[key].append(index)
        rng = np.random.default_rng(seed)
        self.groups = [rng.permutation(np.asarray(indices, dtype=np.int64))
                       for _, indices in sorted(grouped.items())]
        self.round_rows = sum(min(len(group), self.quota) for group in self.groups)
        self.pool_rows = sum(map(len, self.groups))

    def order(self, round_index):
        if type(round_index) is not int or round_index < 0:
            raise ValueError("coverage round must be nonnegative")
        selected = [group[(round_index * self.quota + np.arange(min(self.quota, len(group)))) % len(group)]
                    for group in self.groups]
        if not selected:
            raise ValueError("empty training coverage pool")
        return np.random.default_rng(np.random.SeedSequence([self.seed, round_index])).permutation(
            np.concatenate(selected))

"""Read-only, aligned specialist interchange; no imports from sibling projects.

Each shard is scored by one frozen source fit strictly before that shard. Multiple
shards support expanding-window out-of-fold scores. Audits are supplied by the
source review, never manufactured here. Arrays remain memory mapped.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .integrity import file_digest


@dataclass(frozen=True)
class SpecialistSource:
    kind: str
    channels: tuple[str, ...]
    targets: object
    bounds: tuple


class AlignedTargets:
    def __init__(self, streams):
        self.streams = streams

    def target(self, ticker, row):
        shards, shard_ids, indices = self.streams[ticker]
        if type(row) is not int or row < 0 or row >= len(indices):
            raise ValueError("specialist row outside aligned market")
        return shards[int(shard_ids[row])][int(indices[row])]


def load_specialist_cache(path, *, audit_path, markets):
    path = Path(path)
    manifest = json.loads(path.read_text())
    audit = json.loads(Path(audit_path).read_text())
    if (manifest.get("schema") != "reasoning_specialist_cache_v1"
            or manifest.get("timestamp_semantics") != "completed_bar_utc_ns"
            or audit.get("status") != "PASS"
            or audit.get("manifest_sha256") != file_digest(path)
            or audit.get("specialist_score_mode") not in {"out_of_fold", "post_fit"}
            or audit.get("sealed_touched") is not False):
        raise ValueError("specialist cache requires a matching causal audit")
    channels = tuple(manifest["channels"])
    bounds = np.asarray(manifest["bounds"], dtype=float)
    if (not channels or len(set(channels)) != len(channels)
            or any(not isinstance(x, str) or not x for x in channels)
            or bounds.shape != (len(channels), 2) or not np.isfinite(bounds).all()
            or (bounds[:, 0] > bounds[:, 1]).any()):
        raise ValueError("invalid specialist channel contract")
    streams = {}
    for ticker, market in markets.items():
        expected = market.timestamps.astype("datetime64[ns]").astype(np.int64)
        if not len(expected) or (np.diff(expected) <= 0).any():
            raise ValueError("market timestamps must be increasing")
        shard_ids, indices = np.full(len(expected), -1), np.full(len(expected), -1)
        shards = []
        for shard in manifest["streams"][ticker]:
            for name in ("timestamps", "values"):
                if file_digest(path.parent / shard[name]) != shard[f"{name}_sha256"]:
                    raise ValueError("specialist content changed after audit")
            times = np.load(path.parent / shard["timestamps"], mmap_mode="r", allow_pickle=False)
            values = np.load(path.parent / shard["values"], mmap_mode="r", allow_pickle=False)
            if (times.dtype != np.dtype("int64") or times.ndim != 1 or not len(times)
                    or (np.diff(times) <= 0).any()
                    or values.shape != (len(times), len(channels))):
                raise ValueError("invalid specialist shape or timestamps")
            if (type(shard["fit_end_ns"]) is not int or shard["fit_end_ns"] >= int(times[0])
                    or not shard.get("model_identity")):
                raise ValueError("specialist fit boundary is not before scored rows")
            # Check channel by channel to avoid copying an entire multi-channel cache.
            for column, (lower, upper) in enumerate(bounds):
                vector = values[:, column]
                if not np.isfinite(vector).all() or (vector < lower).any() or (vector > upper).any():
                    raise ValueError("specialist values violate declared bounds")
            positions = np.searchsorted(times, expected)
            valid = positions < len(times)
            valid[valid] &= times[positions[valid]] == expected[valid]
            if (shard_ids[valid] >= 0).any():
                raise ValueError("overlapping specialist shards")
            shard_ids[valid], indices[valid] = len(shards), positions[valid]
            shards.append(values)
        if (indices < 0).any():
            raise ValueError("specialist coverage missing completed market rows")
        streams[ticker] = (tuple(shards), shard_ids, indices)
    return SpecialistSource(manifest["kind"], channels, AlignedTargets(streams), tuple(map(tuple, bounds)))

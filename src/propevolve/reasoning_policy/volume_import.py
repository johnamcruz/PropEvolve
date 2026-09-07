"""Explicit import of reviewed Volume score arrays; never imports sibling code.

Source export must declare timestamps, channels, fit/calibration cutoffs and
file identities. This does not execute a Volume checkpoint or certify its edge.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .model_config import read_recipe
from .integrity import file_digest


def import_volume(config_path):
    config = read_recipe(config_path)
    root = Path(config["workspace_root"]).resolve()
    path = root / config["source_manifest"]
    source = json.loads(path.read_text())
    audit = json.loads((root / config["source_audit"]).read_text())
    if (source.get("kind") != "volume" or source.get("timestamp_semantics") != "completed_bar_utc_ns"
            or audit.get("status") != "PASS" or audit.get("manifest_sha256") != file_digest(path)
            or audit.get("sealed_touched") is not False
            or audit.get("specialist_score_mode") not in {"post_fit", "out_of_fold"}):
        raise ValueError("Volume source requires a matching fold-safe review")
    names = config["channels"]
    original = source["channels"]
    if not names or len(set(names)) != len(names) or set(names) != set(original) or len(set(original)) != len(original):
        raise ValueError("Volume channel mapping must preserve every declared channel exactly once")
    columns = [original.index(name) for name in names]
    bounds = np.asarray(source["bounds"], float)
    if bounds.shape != (len(original), 2) or not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError("invalid Volume bounds")
    output = root / config["output"]
    if output.exists():
        raise FileExistsError("Volume import output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".volume-import-", dir=output.parent))
    try:
        streams = {}
        number = 0
        for ticker, shards in source["streams"].items():
            streams[ticker] = []
            for shard in shards:
                for key in ("timestamps", "values"):
                    if file_digest(path.parent / shard[key]) != shard[key + "_sha256"]:
                        raise ValueError("Volume source bytes changed after review")
                times = np.load(path.parent / shard["timestamps"], mmap_mode="r", allow_pickle=False)
                values = np.load(path.parent / shard["values"], mmap_mode="r", allow_pickle=False)
                if (times.dtype != np.dtype("int64") or times.ndim != 1 or not len(times)
                        or (np.diff(times) <= 0).any() or values.shape != (len(times), len(original))):
                    raise ValueError("invalid Volume source shape/timeline")
                cutoffs = [shard["fit_end_ns"], shard["calibration_end_ns"]]
                if any(type(value) is not int for value in cutoffs) or max(cutoffs) >= int(times[0]) or not shard.get("model_identity"):
                    raise ValueError("Volume fit/calibration must precede its scored rows")
                time_name, value_name = f"{number}-times.npy", f"{number}-values.npy"
                shutil.copyfile(path.parent / shard["timestamps"], temporary / time_name)
                mapped = np.lib.format.open_memmap(temporary / value_name, mode="w+", dtype=values.dtype,
                    shape=values.shape)
                for destination, column in enumerate(columns):
                    vector = values[:, column]
                    if not np.isfinite(vector).all() or (vector < bounds[column, 0]).any() or (vector > bounds[column, 1]).any():
                        raise ValueError("Volume scores violate reviewed bounds")
                    mapped[:, destination] = vector
                mapped.flush()
                del mapped
                streams[ticker].append({"timestamps": time_name, "values": value_name,
                    "timestamps_sha256": file_digest(temporary / time_name),
                    "values_sha256": file_digest(temporary / value_name), "fit_end_ns": max(cutoffs),
                    "model_identity": shard["model_identity"]})
                number += 1
        if not number:
            raise ValueError("Volume export has no score shards")
        manifest = {"schema": "reasoning_specialist_cache_v1", "kind": "volume", "channels": names,
            "bounds": bounds[columns].tolist(), "timestamp_semantics": "completed_bar_utc_ns",
            "streams": streams, "source_manifest_sha256": file_digest(path)}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "REQUIRES_REVIEW", "manifest": str(output / "manifest.json"), "shards": number}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(import_volume(args.config), indent=2))


if __name__ == "__main__":
    main()

"""Read only a declared, hash-bound training diagnostic cohort."""
import json
from pathlib import Path
from .integrity import file_digest


def load_frozen_records(config, *, root):
    path = Path(root) / config["records"]
    if file_digest(path) != config["sha256"]:
        raise ValueError("frozen diagnostic records changed")
    limit = config["maximum_records"]
    if type(limit) is not int or limit < 1:
        raise ValueError("frozen diagnostic limit must be positive")
    lower = config.get("role_start_ns", config.get("train_start_ns"))
    upper = config.get("role_end_ns", config.get("train_end_ns"))
    if type(lower) is not int or type(upper) is not int or lower >= upper:
        raise ValueError("frozen diagnostic temporal role is invalid")
    records = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if not lower <= record["completed_at_ns"] < record["label_end_ns"] < upper:
                raise ValueError("frozen diagnostic must remain inside its declared role")
            records.append(record)
            if len(records) == limit:
                break
    if not records:
        raise ValueError("frozen diagnostic cohort is empty")
    return records

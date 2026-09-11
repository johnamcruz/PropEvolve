"""Frozen diagnostic cohorts fail closed before live reasoning uses them."""

import json

import pytest

from propevolve.reasoning_policy.frozen_audit import load_frozen_records
from propevolve.reasoning_policy.integrity import file_digest


def _write_records(tmp_path, records):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _config(path, **overrides):
    config = {
        "records": path.name,
        "sha256": file_digest(path),
        "maximum_records": 2,
        "role_start_ns": 100,
        "role_end_ns": 500,
    }
    config.update(overrides)
    return config


def test_frozen_audit_loads_only_the_declared_bounded_causal_cohort(tmp_path):
    path = _write_records(tmp_path, [
        {"completed_at_ns": 110, "label_end_ns": 120, "action": "WAIT"},
        {"completed_at_ns": 210, "label_end_ns": 230, "action": "ENTER_LONG_1"},
        {"completed_at_ns": 310, "label_end_ns": 340, "action": "ENTER_SHORT_1"},
    ])

    assert load_frozen_records(_config(path), root=tmp_path) == [
        {"completed_at_ns": 110, "label_end_ns": 120, "action": "WAIT"},
        {"completed_at_ns": 210, "label_end_ns": 230, "action": "ENTER_LONG_1"},
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sha256": "0" * 64}, "records changed"),
        ({"maximum_records": 0}, "limit must be positive"),
        ({"maximum_records": True}, "limit must be positive"),
        ({"role_start_ns": 500}, "temporal role is invalid"),
    ],
)
def test_frozen_audit_rejects_untrusted_or_invalid_contracts(tmp_path, overrides, message):
    path = _write_records(
        tmp_path, [{"completed_at_ns": 110, "label_end_ns": 120, "action": "WAIT"}]
    )

    with pytest.raises(ValueError, match=message):
        load_frozen_records(_config(path, **overrides), root=tmp_path)


def test_frozen_audit_rejects_rows_crossing_the_declared_role_boundary(tmp_path):
    path = _write_records(
        tmp_path, [{"completed_at_ns": 490, "label_end_ns": 510, "action": "WAIT"}]
    )

    with pytest.raises(ValueError, match="remain inside its declared role"):
        load_frozen_records(_config(path), root=tmp_path)


def test_frozen_audit_rejects_an_empty_authenticated_cohort(tmp_path):
    path = _write_records(tmp_path, [])

    with pytest.raises(ValueError, match="cohort is empty"):
        load_frozen_records(_config(path), root=tmp_path)


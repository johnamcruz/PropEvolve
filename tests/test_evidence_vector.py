"""One ordered evidence contract, shared by the compact RL policy and the reasoning model.

The plan is to let a small fast policy search for a profitable trend-following policy over
thousands of episodes -- which the reasoning model could never afford at ~4.5 minutes an
episode -- and then teach it to the reasoning model. That transfer is only exact if both
read the SAME evidence, which is why the field list lives in one place with a fixed order
rather than being rebuilt per consumer.

The order must be STABLE: a trained policy's weights are bound to it, so a silent
reordering would keep running while feeding every input to the wrong neuron.
"""
from __future__ import annotations

import numpy as np
import pytest

from propevolve.reasoning_policy.evidence import (
    EVIDENCE_FIELDS, evidence_vector, TEACHER_PREFIXES)


def test_every_teacher_is_represented():
    """Expansion, trend, regime and volume are the evidence RL has to learn to use."""
    for prefix in ("expansion", "trend", "regime", "volume"):
        assert any(f.startswith(prefix + ".") for f in EVIDENCE_FIELDS), prefix


def test_the_setup_and_risk_context_is_represented():
    for prefix in ("setup", "account", "challenge", "trade"):
        assert any(f.startswith(prefix + ".") for f in EVIDENCE_FIELDS), prefix


def test_the_field_list_has_no_duplicates():
    assert len(EVIDENCE_FIELDS) == len(set(EVIDENCE_FIELDS))


def test_the_order_is_fixed_and_documented():
    """A trained policy's weights bind to this order; reordering silently corrupts it."""
    assert isinstance(EVIDENCE_FIELDS, tuple)
    # grouped by prefix, alphabetical within group, groups in a declared order
    prefixes = [f.split(".")[0] for f in EVIDENCE_FIELDS]
    first_seen = []
    for p in prefixes:
        if p not in first_seen:
            first_seen.append(p)
    # each prefix occupies one contiguous block
    assert prefixes == sorted(prefixes, key=lambda p: first_seen.index(p))


def test_teacher_prefixes_name_the_distillable_half():
    """The student runs teacher-free, so the split has to be explicit, not implied."""
    assert set(TEACHER_PREFIXES) == {"expansion", "trend", "regime", "volume"}


def test_the_vector_matches_the_field_list_length():
    fields = {f: 0.25 for f in EVIDENCE_FIELDS}
    out = evidence_vector(fields)
    assert out.shape == (len(EVIDENCE_FIELDS),)
    assert out.dtype == np.float32


def test_the_vector_preserves_field_order():
    fields = {f: float(i) for i, f in enumerate(EVIDENCE_FIELDS)}
    out = evidence_vector(fields)
    assert np.allclose(out, np.arange(len(EVIDENCE_FIELDS), dtype=np.float32))


def test_a_missing_field_is_refused_not_zero_filled():
    """A fabricated zero would look like a real reading and silently degrade the policy."""
    fields = {f: 0.0 for f in EVIDENCE_FIELDS}
    fields.pop(EVIDENCE_FIELDS[3])
    with pytest.raises(ValueError, match="missing"):
        evidence_vector(fields)


def test_a_non_finite_field_is_refused():
    fields = {f: 0.0 for f in EVIDENCE_FIELDS}
    fields[EVIDENCE_FIELDS[0]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evidence_vector(fields)


def test_extra_fields_are_ignored_not_appended():
    """observe_context supplies more than the contract; width must not drift."""
    fields = {f: 1.0 for f in EVIDENCE_FIELDS}
    fields["something.else"] = 9.0
    assert evidence_vector(fields).shape == (len(EVIDENCE_FIELDS),)

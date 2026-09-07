"""Configurable external Volume scores through the unchanged evaluation path.

Written for the approved causal source -> policy -> simulator seam. Not yet run.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.evaluation import evaluate_policy
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.specialist_cache import load_specialist_cache
from test_reasoning_challenger_e2e import environment
from test_reasoning_collection_evaluation_e2e import sources


def volume_cache(tmp_path, market):
    times = market.timestamps.astype("datetime64[ns]").astype(np.int64)
    np.save(tmp_path / "times.npy", times)
    np.save(tmp_path / "scores.npy", np.full((len(times), 1), 0.8, dtype=np.float32))
    manifest = {"schema": "reasoning_specialist_cache_v1", "kind": "volume",
        "channels": ["clean_participation"], "bounds": [[0.0, 1.0]],
        "timestamp_semantics": "completed_bar_utc_ns",
        "streams": {"NQ": [{"timestamps": "times.npy", "values": "scores.npy",
            "timestamps_sha256": file_digest(tmp_path / "times.npy"),
            "values_sha256": file_digest(tmp_path / "scores.npy"),
            "fit_end_ns": int(times[0]) - 1, "model_identity": "test-only-frozen-source"}]}}
    path = tmp_path / "any-name.json"
    path.write_text(json.dumps(manifest))
    audit = tmp_path / "review.json"
    audit.write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(path),
        "specialist_score_mode": "post_fit", "sealed_touched": False}))
    return path, audit


def test_volume_reaches_policy_without_changing_economic_outcomes(tmp_path):
    env = environment()
    manifest, audit = volume_cache(tmp_path, env.markets["NQ"])
    volume = load_specialist_cache(manifest, audit_path=audit, markets=env.markets)
    class VolumeConsumer:
        def decide(self, context, legal_actions):
            assert context.values[-1, 0] == pytest.approx(0.8)
            return (Action.HOLD if Action.HOLD in legal_actions else Action.ENTER_LONG_1), {}
    result = evaluate_policy(VolumeConsumer(), env, episodes=[{"ticker": "NQ", "start": 0}],
        context_config=ContextConfig(2, ("volume.clean_participation",)),
        sources=(*sources(), volume), max_steps=8)
    assert result["pass_rate"] == 1.0
    assert result["teacher_free"] is False


def test_volume_rejects_post_decision_fitting_even_with_audit(tmp_path):
    env = environment()
    path, audit = volume_cache(tmp_path, env.markets["NQ"])
    payload = json.loads(path.read_text())
    payload["streams"]["NQ"][0]["fit_end_ns"] = int(env.markets["NQ"].timestamps[-1].astype("datetime64[ns]").astype(np.int64))
    path.write_text(json.dumps(payload))
    receipt = json.loads(audit.read_text())
    receipt["manifest_sha256"] = file_digest(path)
    audit.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="fit boundary"):
        load_specialist_cache(path, audit_path=audit, markets=env.markets)


def test_volume_missing_rows_are_not_fabricated(tmp_path):
    env = environment()
    path, audit = volume_cache(tmp_path, env.markets["NQ"])
    missing = SimpleNamespace(timestamps=env.markets["NQ"].timestamps + np.timedelta64(1, "s"))
    with pytest.raises(ValueError, match="coverage"):
        load_specialist_cache(path, audit_path=audit, markets={"NQ": missing})

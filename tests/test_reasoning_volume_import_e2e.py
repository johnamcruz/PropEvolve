"""External calibrated arrays -> mapped interchange -> aligned consumer."""
import json
import numpy as np
import pytest
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.volume_import import import_volume
from propevolve.reasoning_policy.specialist_cache import load_specialist_cache
from test_reasoning_challenger_e2e import environment


def test_volume_import_preserves_declared_long_short_order_and_requires_review(tmp_path):
    env = environment()
    times = env.markets["NQ"].timestamps.astype("datetime64[ns]").astype(np.int64)
    np.save(tmp_path / "times.npy", times)
    np.save(tmp_path / "scores.npy", np.tile([.2, .8], (len(times), 1)))
    source = {"kind": "volume", "channels": ["short", "long"],
        "bounds": [[0, 1], [0, 1]], "timestamp_semantics": "completed_bar_utc_ns",
        "streams": {"NQ": [{"timestamps": "times.npy", "values": "scores.npy",
            "fit_end_ns": int(times[0])-2, "calibration_end_ns": int(times[0])-1,
            "model_identity": "external-reviewed-fixture",
            "timestamps_sha256": file_digest(tmp_path / "times.npy"),
            "values_sha256": file_digest(tmp_path / "scores.npy")}]}}
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(source))
    audit = tmp_path / "source-audit.json"
    audit.write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(manifest),
        "sealed_touched": False, "specialist_score_mode": "post_fit"}))
    destination = tmp_path / "mapped"
    recipe = tmp_path / "import.json"
    recipe.write_text(json.dumps({"workspace_root": str(tmp_path), "source_manifest": manifest.name,
        "source_audit": audit.name, "channels": ["long", "short"], "output": str(destination)}))
    result = import_volume(recipe)
    assert result["status"] == "REQUIRES_REVIEW"
    assert not (destination / "audit.json").exists()
    reviewed = destination / "review.json"
    reviewed.write_text(json.dumps({"status": "PASS", "manifest_sha256": file_digest(destination / "manifest.json"),
        "sealed_touched": False, "specialist_score_mode": "post_fit"}))
    loaded = load_specialist_cache(destination / "manifest.json", audit_path=reviewed, markets=env.markets)
    np.testing.assert_allclose(loaded.targets.target("NQ", 0), [.8, .2])
    with pytest.raises(FileExistsError):
        import_volume(recipe)

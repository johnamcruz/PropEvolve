"""Production training cannot silently fall back to the retired scorer."""
import pytest


@pytest.mark.parametrize("architecture", [None, "direct_action", "unknown"])
def test_training_rejects_nonstaged_recipe_before_loading_weights(architecture, tmp_path):
    from propevolve.reasoning_policy.supervised_trainer import train_supervised
    config = {"architecture": architecture, "adapter_path": str(tmp_path / "output")}
    with pytest.raises(ValueError, match="staged_reasoning_v1"):
        train_supervised(config, tmp_path / "absent-view")
    assert not (tmp_path / "output").exists()

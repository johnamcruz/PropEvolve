"""All legal actions from ONE simulator state, never matched across states."""
import json
import numpy as np
from ..decision import Action


def action_targets(record):
    target = record["targets"]
    names = target["action_order"]
    probabilities = np.asarray(target["action_probabilities"], dtype=float)
    if (not names or len(set(names)) != len(names) or set(names) - {a.name for a in Action}
            or probabilities.shape != (len(names),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.)):
        raise ValueError("invalid full-action probability targets")
    if set(target["outcomes"]) != set(names):
        raise ValueError("full-action outcomes differ from legal alternatives")
    prompt = json.loads(record["messages"][-2]["content"])
    legal = prompt["legal_actions"]
    if len(legal) != len(set(legal)) or set(legal) != set(names):
        raise ValueError("full-action targets differ from prompt legal actions")
    values = [target["outcomes"][name]["reward_to_go"] for name in names]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite full-action economic values")
    return {"names": names, "probabilities": probabilities.tolist(), "values": values}


def action_objective(scores, probabilities, values, config, *, xp, valid=None):
    """Listwise economic preferences plus gap-weighted pairwise ordering.

No fixed LONG > WAIT rule: simulator values decide the ordering. Exact economic
ties receive no ranking margin. Soft labels preserve their uncertainty.
    """
    if valid is not None:
        scores = xp.where(valid, scores, -1e9)
    shifted = scores - xp.max(scores)
    log_probs = shifted - xp.log(xp.exp(shifted).sum())
    ce = -(probabilities * log_probs).sum()
    gaps = xp.maximum(values[:, None] - values[None, :], 0.)
    if valid is not None:
        gaps = gaps * (valid[:, None] & valid[None, :])
    mass = gaps.sum()
    penalties = xp.logaddexp(0., config["margin"] - (scores[:, None] - scores[None, :]))
    ranking = (gaps * penalties).sum() / xp.maximum(mass, 1e-12)
    return config["soft_target_weight"] * ce + config["ranking_weight"] * ranking

"""All legal actions from ONE simulator state, never matched across states."""
import json
import numpy as np
from ..decision import Action


def mean_completion_scores(token_log_probs, mask, *, xp):
    """Comparable categorical scores despite unequal action token lengths."""
    counts = mask.sum(axis=-1)
    if xp is np and (counts <= 0).any():
        raise ValueError("completion score requires at least one target token")
    return xp.where(mask, token_log_probs, 0.).sum(axis=-1) / xp.maximum(counts, 1)


def action_completion_scores(token_log_probs, mask, *, xp):
    """Score only the one legal-action token; EOS is formatting, not policy credit."""
    counts = mask.sum(axis=-1)
    if xp is np and ((counts != 2).any() or not mask.any(axis=-1).all()):
        raise ValueError("action completion requires one action token followed by EOS")
    first = xp.argmax(mask, axis=-1)
    return xp.take_along_axis(token_log_probs, first[:, None], axis=-1).squeeze(-1)


def completion_objective(scores, valid, *, xp):
    """Average already-normalized completion scores exactly once."""
    return -(xp.where(valid, scores, 0.).sum() / xp.maximum(valid.sum(), 1))


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


def hierarchical_action_objective(scores, probabilities, values, config, *,
                                  task_code, xp, correction_boundaries=None):
    """Optimize state-appropriate binary decisions with shared action scores.

    ``task_code`` is zero for a flat WAIT/LONG/SHORT state and one for a
    positioned HOLD/CLOSE state. Flat winners learn ENTER versus WAIT and,
    conditionally, LONG versus SHORT. Failed or conflicted flat states never
    receive an arbitrary direction gradient.
    """
    scores = xp.array(scores)
    probabilities = xp.array(probabilities)
    values = xp.array(values)
    boundaries = (xp.ones(3) if correction_boundaries is None
                  else xp.array(correction_boundaries).astype(scores.dtype))
    if scores.shape[0] == 2:
        # Positioned batches have only HOLD/CLOSE. Pad the unused flat branch
        # because MLX traces both sides of the final ``where``.
        scores = xp.concatenate([scores, xp.array([-1e9])])
        probabilities = xp.concatenate([probabilities, xp.array([0.0])])
        values = xp.concatenate([values, xp.array([0.0])])

    direction_eligible = ((xp.maximum(values[1], values[2]) > values[0])
                          & (values[1] != values[2]))
    authenticated_side_score = xp.where(values[1] > values[2], scores[1], scores[2])
    # A winner learns its authenticated side over WAIT. A WAIT row compares
    # against the model's strongest side so neither losing side can escape.
    enter_score = xp.where(
        direction_eligible, authenticated_side_score,
        xp.maximum(scores[1], scores[2]))
    entry_scores = xp.stack([scores[0], enter_score])
    entry_probabilities = xp.stack([
        probabilities[0], probabilities[1] + probabilities[2]])
    entry_values = xp.stack([values[0], xp.maximum(values[1], values[2])])
    entry_loss = action_objective(
        entry_scores, entry_probabilities, entry_values, config, xp=xp)

    direction_probabilities = probabilities[1:3]
    direction_probabilities = direction_probabilities / xp.maximum(
        direction_probabilities.sum(), 1e-12)
    direction_loss = action_objective(
        scores[1:3], direction_probabilities, values[1:3], config, xp=xp)
    entry_weight = boundaries[0]
    direction_weight = boundaries[1] * direction_eligible
    flat_weight = entry_weight + direction_weight
    flat_loss = (entry_weight * entry_loss + direction_weight * direction_loss) / xp.maximum(
        flat_weight, 1.0)

    management_loss = action_objective(
        scores[:2], probabilities[:2] / xp.maximum(probabilities[:2].sum(), 1e-12),
        values[:2], config, xp=xp)
    management_loss = boundaries[2] * management_loss
    return xp.where(task_code == 0, flat_loss, management_loss)

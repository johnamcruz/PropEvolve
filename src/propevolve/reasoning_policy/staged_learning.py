"""Economic supervision for the shared staged reasoning forward path."""
from .supervision import action_objective

LOSS_SEMANTICS = "staged_row_mean_v1"


def supervised_outputs(backend, batch, config):
    """Supervise the same staged forward used by inference, without teacher forcing.

    Targets are separate from causal inputs. Both objectives update the same
    model through the predicted interpretation; there is no action-only bypass.
    """
    import mlx.core as mx
    from .staged_policy import staged_forward
    from .market_distillation import probability_loss
    if set(batch) != {"inputs", "targets"}:
        raise ValueError("staged training requires separate inputs and targets")
    result = staged_forward(backend, **batch["inputs"])
    targets = batch["targets"]
    required = {"probabilities", "values", "boundary_weights",
                "teacher_probabilities", "teacher_weights"}
    if config.get("mastered_anchor_retention") is not None:
        required |= {"parent_assessment", "retention_weights"}
    if set(targets) != required:
        raise ValueError("invalid staged supervision fields")
    if (targets["teacher_probabilities"].shape != result["interpretation_scores"].shape
            or targets["teacher_weights"].shape != result["interpretation_scores"].shape):
        raise ValueError("teacher targets must match configured interpretation channels")
    trade = corrective_trade_objective(result["assessment_scores"], targets, config)
    interpretation = probability_loss(result["interpretation_scores"],
        targets["teacher_probabilities"], targets["teacher_weights"])
    loss = trade + config["interpretation_loss_weight"] * interpretation
    return loss, mx.array(result["assessment_scores"].shape[0]), result


def corrective_trade_objective(scores, targets, config):
    """Correct mistakes and retain only authenticated mastered binary outputs."""
    import mlx.core as mx
    settings = config.get("mastered_anchor_retention")
    weights = targets["boundary_weights"]
    mastered = (mx.zeros_like(weights) if settings is None else
                targets["retention_weights"].astype(weights.dtype))
    loss = trade_objective(scores, targets["probabilities"], targets["values"],
        weights * (1. - mastered), config["action_supervision"], xp=mx)
    if settings is None:
        return loss
    temperature = settings["temperature"]
    parent = mx.stop_gradient(targets["parent_assessment"] / temperature)
    candidate = scores / temperature
    p = mx.sigmoid(parent)
    divergence = ((mx.logaddexp(candidate, 0.) - p * candidate)
                  - (mx.logaddexp(parent, 0.) - p * parent))
    retention = ((divergence * mastered).sum() / scores.shape[0]
                 * temperature ** 2)
    loss = loss + settings["loss_weight"] * retention
    if settings.get("supervision_weight", 0.):
        loss = loss + settings["supervision_weight"] * trade_objective(scores,
            targets["probabilities"], targets["values"], weights * mastered,
            config["action_supervision"], xp=mx)
    return loss


def trade_objective(scores, probabilities, values, boundary_weights, settings, *, xp):
    """Supervise independent binary outputs, only at applicable boundaries.

    Sum applicable task contributions per example, then average examples.
    Dividing by the number of active boundaries in each microbatch changes
    task weights when equal-sized microbatches are accumulated: ENTER rows
    have two boundaries, WAIT and management rows only one. The row mean
    preserves the sampler's evidence mass independently of partitioning.
    Masks suppress direct losses, not indirect shared-weight effects; retention
    must still be measured after the actual parameter update.
    """
    batch = scores.shape[0]
    if (scores.shape != (batch, 3) or probabilities.shape != (batch, 3, 2)
            or values.shape != probabilities.shape or boundary_weights.shape != scores.shape):
        raise ValueError("invalid independent trade supervision shapes")
    terms = []
    for i in range(batch):
        for task in range(3):
            logits = xp.stack([xp.array(0.), scores[i, task]])
            terms.append(boundary_weights[i, task] * action_objective(
                logits, probabilities[i, task], values[i, task], settings, xp=xp))
    return xp.stack(terms).sum() / batch

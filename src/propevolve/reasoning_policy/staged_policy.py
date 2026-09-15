"""Shared differentiable mapping from binary assessments to legal actions.

Binary log odds are ENTER over WAIT, LONG over SHORT, and HOLD over CLOSE.
This is a mapping, not a learned policy or a substitute for Qwen assessment.
"""
from ..decision import Action
from .backend import ReasoningBackend


def select_legal_action(assessment, legal_actions):
    """One deterministic hierarchical decision rule for inference and audits."""
    import numpy as np
    values = np.asarray(assessment, float)
    actions = tuple(Action(a) for a in legal_actions)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("selection requires three finite binary assessments")
    # Also validates the legal position state and uniqueness.
    legal_action_log_probs(values, actions, xp=np)
    entry, direction, management = values
    if Action.WAIT in actions:
        action = (Action.WAIT if entry <= 0 or direction == 0 else
                  Action.ENTER_LONG_1 if direction > 0 else Action.ENTER_SHORT_1)
    else:
        action = Action.HOLD if management > 0 else Action.CLOSE
    if action not in actions:
        if Action.WAIT in actions:
            return Action.WAIT
        raise ValueError("assessment requested an unavailable positioned action")
    return action


def staged_forward(backend: ReasoningBackend, embeddings, available, market_query,
                   assessment_query, legal_actions):
    """One target-free differentiable computation for training and inference.

Trade/position state belongs in the assessment query, never the market stage.
Query builders authenticate token positions and causal serialization upstream.
    """
    import mlx.core as mx
    from .market_distillation import interpretation_scores
    if (set(market_query) != {"tokens", "positions", "label_ids"}
            or set(assessment_query) != {"tokens", "interpretation_positions",
                "interpretation_label_ids", "task_positions", "task_label_ids"}):
        raise ValueError("invalid staged query fields; targets are not inference inputs")
    if len(legal_actions) != embeddings.shape[0]:
        raise ValueError("staged legal states must match the context batch")
    interpretation = interpretation_scores(backend, embeddings=embeddings,
        available=available, causal_states=mx.zeros((embeddings.shape[0], 0)), **market_query)
    assessment = assess_interpretation(backend, interpretation_scores=interpretation,
                                       **assessment_query)
    log_probs = [legal_action_log_probs(row, actions, xp=mx)
                 for row, actions in zip(assessment, legal_actions)]
    return {"interpretation_scores": interpretation,
            "assessment_scores": assessment, "log_probs": log_probs}


def assess_interpretation(backend: ReasoningBackend, tokens, interpretation_scores, interpretation_positions,
                          interpretation_label_ids, task_positions, task_label_ids):
    """The reasoning model assesses predicted concepts, not raw market inputs.

Each named concept's answer slot receives the soft embedding of the model's
predicted Bernoulli distribution. Surrounding tokens identify the concept and
contain causal trade state. This keeps the connection differentiable without
serializing probabilities as rounded decimal strings. Targets are not accepted.
    """
    import mlx.core as mx
    if (tokens.ndim != 2 or interpretation_scores.ndim != 2
            or interpretation_scores.shape != interpretation_positions.shape
            or tokens.shape[0] != interpretation_scores.shape[0]
            or interpretation_label_ids.shape != (tokens.shape[0], 2)
            or task_positions.shape != (tokens.shape[0], 3)
            or task_label_ids.shape != (tokens.shape[0], 3, 2)):
        raise ValueError("invalid staged assessment tensor shapes")
    text = backend.embed_tokens(tokens)
    labels = backend.embed_tokens(interpretation_label_ids)
    probabilities = mx.sigmoid(interpretation_scores)[..., None]
    beliefs = ((1. - probabilities) * labels[:, None, 0, :]
               + probabilities * labels[:, None, 1, :])
    slots = (mx.arange(tokens.shape[1])[None, None, :]
             == interpretation_positions[..., None])
    replacements = mx.einsum("bcl,bcd->bld", slots.astype(text.dtype), beliefs)
    joined = mx.where(mx.any(slots, axis=1)[..., None], replacements, text)
    hidden = backend.hidden_states(tokens, joined)
    queried = mx.take_along_axis(hidden, task_positions[..., None], axis=1)
    logits = backend.output_logits(queried).astype(mx.float32)
    selected = mx.take_along_axis(logits, task_label_ids, axis=-1)
    return selected[..., 1] - selected[..., 0]


def legal_action_log_probs(binary_logits, legal_actions, *, xp):
    """Return normalized log probabilities in the supplied legal-action order.

    The array namespace is injected so MLX training and CPU reference evaluation
    use exactly the same mathematics without detaching gradients.
    """
    if binary_logits.shape != (3,):
        raise ValueError("assessment requires three scalar binary log odds")
    actions = tuple(Action(action) for action in legal_actions)
    flat = {Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
    positioned = {Action.HOLD, Action.CLOSE}
    if (not actions or len(set(actions)) != len(actions)
            or not (set(actions) <= flat or set(actions) <= positioned)):
        raise ValueError("legal actions must form one nonempty unique position state")
    entry, direction, management = binary_logits
    log_enter = -xp.logaddexp(0., -entry)
    scores = xp.stack([
        -xp.logaddexp(0., entry),
        log_enter - xp.logaddexp(0., -direction),
        log_enter - xp.logaddexp(0., direction),
        -xp.logaddexp(0., -management),
        -xp.logaddexp(0., management),
    ])
    selected = xp.stack([scores[int(action)] for action in actions])
    largest = xp.max(selected)
    return selected - largest - xp.log(xp.sum(xp.exp(selected - largest)))

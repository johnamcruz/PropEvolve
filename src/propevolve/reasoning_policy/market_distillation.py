"""Training-only probability queries; targets never become input tokens."""
import math
from collections.abc import Mapping


def validate_market_distillation(settings):
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) != {
            "negative_token", "positive_token", "instruction", "channels"}:
        raise ValueError("invalid market distillation configuration")
    for key in ("negative_token", "positive_token", "instruction"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError("market label tokens must be nonempty")
    channels = settings["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError("market distillation requires channels")
    names, queries = set(), set()
    for channel in channels:
        if not isinstance(channel, dict) or set(channel) != {"name", "query", "weight"}:
            raise ValueError("invalid market distillation channel")
        name, query, weight = channel["name"], channel["query"], channel["weight"]
        if (not isinstance(name, str) or not name.strip() or name in names
                or not isinstance(query, str) or not query.strip() or query in queries
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError("market channels need unique names/queries and positive weights")
        names.add(name)
        queries.add(query)


def encode_market_targets(record, settings, tokenizer, *, max_seq_length,
                          chat_template_kwargs):
    validate_market_distillation(settings)
    targets = record.get("targets", {}).get("specialist_targets", {})
    if set(targets) != {channel["name"] for channel in settings["channels"]}:
        raise ValueError("teacher channels differ from the declared distillation contract")
    probabilities = [targets.get(channel["name"]) for channel in settings["channels"]]
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("missing or invalid teacher probability")
    # No simplex normalization: Long/Short probabilities can both be high or low.
    labels = [tokenizer.encode(settings[key], add_special_tokens=False)
              for key in ("negative_token", "positive_token")]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError("market labels must be distinct single tokenizer tokens")
    messages = [{**message} for message in record["messages"][:-1]]
    messages[0] = {"role": "system", "content": settings["instruction"]}
    rendered = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, **chat_template_kwargs)
    tokens = list(rendered["input_ids"] if isinstance(rendered, Mapping) else rendered)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("market prompt must contain one sequence of integer token IDs")
    offset = len(tokens)
    positions = []
    for channel in settings["channels"]:
        query = tokenizer.encode(channel["query"], add_special_tokens=False)
        if not query:
            raise ValueError("empty encoded market query")
        tokens.extend(query)
        positions.append(len(tokens) - 1)
    if tokenizer.eos_token_id is None:
        raise ValueError("market distillation needs an EOS token")
    tokens.append(tokenizer.eos_token_id)
    if len(tokens) > max_seq_length:
        raise ValueError("market queries exceed token budget; truncation forbidden")
    return {"tokens": tokens, "offset": offset, "market_targets": {
        "positions": positions, "probabilities": probabilities,
        "weights": [channel["weight"] for channel in settings["channels"]],
        "label_ids": [ids[0] for ids in labels]}}


def probability_loss(scores, probabilities, weights):
    """Independent soft Bernoulli targets; preserve each example's total weight."""
    import mlx.core as mx
    scores = scores.astype(mx.float32)
    losses = mx.logaddexp(scores, 0.) - probabilities * scores
    return ((losses * weights).sum(axis=-1) / weights.sum(axis=-1)).mean()


def market_outputs(model, tokens, embeddings, available, causal_states,
                   positions, probabilities, weights, label_ids):
    import mlx.core as mx
    inputs = tokens[:, 0, :-1]
    prefix = model.market_projector(embeddings, available, causal_states)
    joined = mx.concatenate([prefix, model.model.embed_tokens(inputs)], axis=1)
    hidden = model.model(inputs, input_embeddings=joined)
    indices = mx.stop_gradient(positions + prefix.shape[1])
    queried = mx.take_along_axis(hidden, indices[..., None], axis=1)
    # Project only query positions, not hundreds of numerical answer positions.
    logits = (model.model.embed_tokens.as_linear(queried)
              if model.args.tie_word_embeddings else model.lm_head(queried)).astype(mx.float32)
    selected = mx.take_along_axis(logits,
        mx.broadcast_to(mx.stop_gradient(label_ids[:, None, :]),
                        (*logits.shape[:2], 2)), axis=-1)
    scores = selected[..., 1] - selected[..., 0]
    return probability_loss(scores, probabilities, weights), mx.array(tokens.shape[0]), scores


def evaluate_market_validation(model, dataset, config):
    """Fixed full validation, in batches; teacher values are used only for metrics."""
    import mlx.core as mx
    import numpy as np
    from .supervised_trainer import tensor_batches, _batch_outputs
    model.eval()
    count, loss_sum = 0, 0.
    squared = np.zeros(len(config["market_distillation"]["channels"]), dtype=np.float64)
    for batch in tensor_batches(dataset, config["validation_batch_size"],
            config["max_seq_length"], include_partial=True):
        loss, _, scores = _batch_outputs(model, *batch, config=config)
        errors = ((mx.sigmoid(scores) - batch[11]) ** 2).sum(axis=0)
        mx.eval(loss, errors)
        size = batch[0].shape[0]
        count += size
        loss_sum += float(loss) * size
        squared += np.asarray(errors)
    if not count:
        raise ValueError("empty market validation")
    by_channel = {channel["name"]: float(value / count)
        for channel, value in zip(config["market_distillation"]["channels"], squared)}
    return {"val_loss": loss_sum / count, "market_brier": float(squared.mean() / count),
            "worst_market_brier": max(by_channel.values()),
            "market_brier_by_channel": by_channel, "validation_rows": count}

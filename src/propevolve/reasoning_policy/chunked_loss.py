"""Optional rematerialized vocabulary loss for frozen-head Qwen3 market SFT."""


def chunked_market_outputs(model, tokens, offsets, lengths, valid, probabilities, values,
                           task_codes, causal_states, embeddings, available, *, config, chunk_size):
    import mlx.core as mx
    if (config["action_supervision"]["enabled"] or config["input_mode"] != "embeddings"
            or tokens.shape[1] != 1 or getattr(model, "model_type", None) != "qwen3"):
        raise ValueError("chunked loss requires single-completion Qwen3 market SFT")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk size must be positive")
    prefix = model.market_projector(embeddings, available, causal_states)
    inputs = tokens[:, 0, :-1]
    joined = mx.concatenate([prefix, model.model.embed_tokens(inputs)], axis=1)
    hidden = model.model(inputs, input_embeddings=joined)[:, prefix.shape[1]:, :]
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    head_module = model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    if head_module.trainable_parameters():
        raise ValueError("chunked loss requires a frozen vocabulary head")
    targets = tokens[:, 0, 1:]
    steps = mx.arange(1, tokens.shape[-1])
    mask = ((steps[None, :] >= offsets[:, 0, None])
            & (steps[None, :] < lengths[:, 0, None]) & valid[:, 0, None])
    def chunk_sum(h, y, selected):
        y, selected = mx.stop_gradient(y), mx.stop_gradient(selected)
        logits = head(h).astype(mx.float32)
        scores = mx.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
        scores = scores - mx.logsumexp(logits, axis=-1)
        return (scores * selected).sum(axis=-1)
    rematerialized = mx.checkpoint(chunk_sum)
    sums = mx.zeros((tokens.shape[0],), dtype=mx.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        end = start + chunk_size
        sums = sums + rematerialized(hidden[:, start:end], targets[:, start:end], mask[:, start:end])
    scores = sums / mx.maximum(mask.sum(axis=-1), 1)
    return -scores.mean(), mx.array(tokens.shape[0]), scores[:, None]

"""Compare identical real prepared batches through the production MLX learner.

No training artifacts or source rows are modified. Run each batch size in a
fresh process so peak memory measurements are independent.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--candidate", choices=("native_ce", "native_prefix", "target_gather", "compiled_validation"), default="target_gather")
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch size and repeats must be positive")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    from propevolve.reasoning_policy.mlx_sft import (
        read_sft_config, verify_mlx_view, PreparedDataset,
    )
    from propevolve.reasoning_policy.projector import attach_projector
    from propevolve.reasoning_policy import supervised_trainer as trainer
    from propevolve.reasoning_policy import projector
    config = read_sft_config(args.config, root=Path.cwd())
    verify_mlx_view(args.config, args.view, root=Path.cwd())
    if config["resume_adapter_file"] is not None:
        raise ValueError("this base-model benchmark requires a scratch SFT recipe")
    mx.random.seed(config["seed"])
    model, _ = load(config["model"])
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    attach_projector(model, config["projector"])
    trainer.configure_trainable_components(model, config["trainable_components"])
    model.eval()  # Disable dropout for exact same-state comparisons.
    if args.gradients and config["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])
    data = PreparedDataset(args.view, "train")
    if args.batch_size > len(data):
        raise ValueError("batch exceeds dataset")
    start = time.perf_counter()
    packed = tuple(mx.array(x) for x in trainer.pack_examples(
        [data[i] for i in range(args.batch_size)], max_seq_length=config["max_seq_length"]))
    mx.eval(packed, model.parameters())
    print(json.dumps({"stage": "ready", "batch": args.batch_size,
        "tokens_shape": list(packed[0].shape),
        "token_embedding_dtype": str(model.model.embed_tokens(packed[0][:, 0, :1]).dtype),
        "market_prefix_dtype": str(model.market_projector(packed[-2], packed[-1]).dtype),
        "pack_seconds": time.perf_counter() - start}), flush=True)
    original = trainer.selected_token_scores
    def legacy(logits, targets):
        logits = logits.astype(mx.float32)
        probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.take_along_axis(probabilities, targets[..., None], -1).squeeze(-1)
    original_logits = projector.market_logits
    def native_prefix(m, tokens, embeddings, available, causal_state=None):
        prefix = m.market_projector(embeddings, available, causal_state)
        text = m.model.embed_tokens(tokens)
        joined = mx.concatenate([prefix.astype(text.dtype), text], axis=1)
        return m(tokens, input_embeddings=joined)[:, prefix.shape[1]:, :]
    def native_ce(logits, targets):
        return -nn.losses.cross_entropy(logits.astype(mx.float32), targets)
    candidates = {"native_ce": native_ce, "target_gather": original, "compiled_validation": original,
                  "native_prefix": legacy}
    reference_loss = reference_grads = None
    try:
        for name, scoring, compiled in (
                ("baseline", original if args.candidate == "compiled_validation" else legacy,
                 args.candidate != "compiled_validation"),
                (args.candidate + "_compiled",
                 candidates[args.candidate], True)):
            if name == "native_prefix_compiled":
                projector.market_logits = native_prefix
            trainer.selected_token_scores = scoring
            loss = lambda m, *batch: trainer.batch_loss(m, *batch, config=config)[0]
            evaluator = nn.value_and_grad(model, loss) if args.gradients else loss
            fn = lambda *batch: evaluator(model, *batch)
            if compiled:
                state = [model.state, mx.random.state]
                fn = mx.compile(fn, inputs=state, outputs=state)
            mx.clear_cache()
            mx.reset_peak_memory()
            times = []
            for repeat in range(args.repeats + 1):
                start = time.perf_counter()
                result = fn(*packed)
                mx.eval(result)
                seconds = time.perf_counter() - start
                if repeat:
                    times.append(seconds)
            value = float((result[0] if args.gradients else result).item())
            grads = dict(tree_flatten(result[1])) if args.gradients else {}
            if reference_loss is None:
                reference_loss, reference_grads = value, grads
            error = max((float(mx.max(mx.abs(g - reference_grads[k])).item())
                         for k, g in grads.items()), default=0.)
            print(json.dumps({"mode": name, "gradients": args.gradients,
                "batch": args.batch_size, "seconds": times, "loss": value,
                "loss_abs_difference": abs(value - reference_loss),
                "gradient_max_abs_difference": error,
                "peak_gb": mx.get_peak_memory() / 1e9}), flush=True)
            if abs(value - reference_loss) > 1e-5 or error > 1e-5:
                raise ValueError("benchmark numerical parity failed")
            del result, fn
    finally:
        trainer.selected_token_scores = original
        projector.market_logits = original_logits


if __name__ == "__main__":
    main()

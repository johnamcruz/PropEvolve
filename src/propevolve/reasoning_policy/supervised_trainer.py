"""Native MLX-LM trainer with explicit tensor batches and a trading loss.

No second optimizer loop. Reuses native LoRA conversion, accumulation,
checkpointing, evaluation and loss callback interfaces.
"""
import json
from pathlib import Path
import numpy as np

from .supervision import action_objective


def pack_examples(rows, *, max_seq_length):
    alternatives = [row.get("alternatives", [(row["tokens"], row["offset"])]) for row in rows]
    actions = max(map(len, alternatives))
    length = max(len(tokens) for group in alternatives for tokens, _ in group)
    if length > max_seq_length:
        raise ValueError("supervised batch exceeds token budget; truncation forbidden")
    tokens = np.zeros((len(rows), actions, length), np.int32)
    offsets = np.zeros((len(rows), actions), np.int32)
    lengths = np.zeros_like(offsets)
    valid = np.zeros_like(offsets, dtype=bool)
    probabilities = np.zeros_like(offsets, dtype=np.float32)
    values = np.zeros_like(probabilities)
    for i, (row, group) in enumerate(zip(rows, alternatives)):
        for j, (sequence, offset) in enumerate(group):
            if not 0 < offset < len(sequence):
                raise ValueError("invalid supervised answer offset")
            tokens[i, j, :len(sequence)] = sequence
            offsets[i, j], lengths[i, j], valid[i, j] = offset, len(sequence), True
        target = row.get("action_targets")
        probabilities[i, :len(group)] = [1.] if target is None else target["probabilities"]
        values[i, :len(group)] = [0.] if target is None else target["values"]
    embeddings = [row.get("market_embeddings", [[0.]]) for row in rows]
    if len({np.asarray(x).shape for x in embeddings}) != 1:
        raise ValueError("embedding windows must share one configured shape")
    available = [row.get("market_available", [True]) for row in rows]
    return (tokens, offsets, lengths, valid, probabilities, values,
            np.asarray(embeddings, np.float32), np.asarray(available, bool))


def tensor_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    import mlx.core as mx
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("reasoning trainer currently supports one local worker")
    if len(dataset) < batch_size:
        raise ValueError("not enough supervised rows for a batch")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(dataset)) if loop else np.arange(len(dataset))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            rows = [dataset[int(i)] for i in order[start:start + batch_size]]
            yield tuple(mx.array(x) for x in pack_examples(rows, max_seq_length=max_seq_length))
        if not loop:
            return


def batch_loss(model, tokens, offsets, lengths, valid, probabilities, values, embeddings, available, *, config):
    import mlx.core as mx
    from .projector import market_logits
    losses = []
    for index in range(tokens.shape[0]):
        inputs = tokens[index, :, :-1]
        logits = (market_logits(model, inputs, embeddings[index:index+1], available[index:index+1])
                  if config["input_mode"] == "embeddings" else model(inputs)).astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = tokens[index, :, 1:, None]
        token_scores = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
        steps = mx.arange(1, tokens.shape[-1])
        mask = (steps >= offsets[index, :, None]) & (steps < lengths[index, :, None]) & valid[index, :, None]
        scores = mx.where(mask, token_scores, 0.).sum(axis=-1)
        if config["action_supervision"]["enabled"]:
            losses.append(action_objective(scores, probabilities[index], values[index],
                config["action_supervision"], xp=mx, valid=valid[index]))
        else:
            losses.append(-scores.sum() / mx.maximum(mask.sum(), 1))
    return mx.stack(losses).mean(), mx.array(tokens.shape[0])


def train_supervised(config, view):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import train, TrainingArgs
    from functools import partial
    from .model_config import verify_adapter_base
    from .projector import attach_projector, export_policy_weights, restore_projector
    destination = Path(config["adapter_path"])
    if destination.exists():
        raise FileExistsError("supervised adapter output exists")
    mx.random.seed(config["seed"])
    model, _ = load(config["model"], tokenizer_config={"trust_remote_code": False})
    if not any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
        raise ValueError("QLoRA requires a quantized base")
    if config["num_layers"] > len(model.layers):
        raise ValueError("requested LoRA layers exceed model depth")
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    parent = config["resume_adapter_file"]
    if parent is not None:
        directory = Path(parent).parent
        verify_adapter_base(config["model"], directory)
        metadata = json.loads((directory / "adapter_config.json").read_text())
        for key in ("lora_parameters", "num_layers", "chat_template_kwargs", "input_mode", "projector"):
            if metadata.get(key) != config.get(key):
                raise ValueError(f"SFT warm-start contract differs at {key}")
        model.load_weights(parent, strict=False)
    if config["input_mode"] == "embeddings":
        attach_projector(model, config["projector"])
        if parent is not None:
            restore_projector(model, Path(parent).parent)
    optimizer_kind = config["optimizer"]
    constructors = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_kind not in constructors or config["lr_schedule"] is not None:
        raise ValueError("configured optimizer/schedule unsupported by tensor SFT adapter")
    optimizer = constructors[optimizer_kind](learning_rate=config["learning_rate"],
        **config["optimizer_config"].get(optimizer_kind, {}))
    datasets = {role: [json.loads(line) for line in (Path(view) / f"{role}.jsonl").read_text().splitlines()]
                for role in ("train", "valid")}
    destination.mkdir(parents=True)
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2))
    args = TrainingArgs(batch_size=config["batch_size"], iters=config["iters"],
        val_batches=config["val_batches"], steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"], steps_per_save=config["save_every"],
        adapter_file=str(destination / "adapters.safetensors"), max_seq_length=config["max_seq_length"],
        grad_checkpoint=config["grad_checkpoint"], grad_accumulation_steps=config["grad_accumulation_steps"],
        clear_cache_threshold=config["clear_cache_threshold"])
    train(model, optimizer, datasets["train"], datasets["valid"], args=args,
        loss=partial(batch_loss, config=config),
        iterate_batches=partial(tensor_batches, seed=config["seed"]))
    export_policy_weights(model, destination)

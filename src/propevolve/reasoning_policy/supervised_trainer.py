"""Native MLX-LM trainer with explicit tensor batches and a trading loss.

No second optimizer loop. Reuses native LoRA conversion, accumulation,
checkpointing, evaluation and loss callback interfaces.
"""
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

from .supervision import (
    action_completion_scores, action_objective, completion_objective,
    mean_completion_scores,
)


class EarlyStopTraining(RuntimeError):
    """Private control signal raised only at a completed validation boundary."""


class ValidationLossGuard:
    """Track validation loss, snapshot improvements, and stop stale training."""

    def __init__(self, settings, *, on_improvement):
        required = {"enabled", "patience_evaluations", "min_delta", "restore_best"}
        optional = {"monitor", "mode"}
        if (not isinstance(settings, dict) or not required.issubset(settings)
                or set(settings) - required - optional
                or type(settings["enabled"]) is not bool
                or type(settings["restore_best"]) is not bool
                or type(settings["patience_evaluations"]) is not int
                or settings["patience_evaluations"] < 1
                or isinstance(settings["min_delta"], bool)
                or not math.isfinite(float(settings["min_delta"]))
                or settings["min_delta"] < 0
                or settings["restore_best"] and not settings["enabled"]):
            raise ValueError("invalid early stopping settings")
        self.settings = dict(settings)
        self.monitor = settings.get("monitor", "val_loss")
        self.mode = settings.get("mode", "min")
        if (not isinstance(self.monitor, str) or not self.monitor
                or self.mode not in {"min", "max"}):
            raise ValueError("invalid early stopping monitor")
        self.on_improvement = on_improvement
        self.best_metric = math.inf if self.mode == "min" else -math.inf
        self.best_loss = math.inf
        self.best_report = None
        self.best_iteration = None
        self.evaluations = 0
        self.stale_evaluations = 0
        self.stopped_early = False
        self.stop_iteration = None
        self.history = []

    def on_train_loss_report(self, train_info):
        return None

    def on_val_loss_report(self, val_info):
        loss = val_info.get("val_loss")
        iteration = val_info.get("iteration")
        if (isinstance(loss, bool) or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss)) or type(iteration) is not int
                or iteration < 0):
            raise ValueError("invalid validation loss report")
        metric = val_info.get(self.monitor)
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))):
            raise ValueError("invalid early stopping monitor report")
        self.evaluations += 1
        if not self.settings["enabled"]:
            return
        improved = (float(metric) < self.best_metric - self.settings["min_delta"]
                    if self.mode == "min" else
                    float(metric) > self.best_metric + self.settings["min_delta"])
        history = {"iteration": iteration, "validation_loss": float(loss),
                   "checkpoint_selected": improved}
        if self.monitor != "val_loss":
            history[self.monitor] = float(metric)
        self.history.append(history)
        if improved:
            self.best_metric = float(metric)
            self.best_loss = float(loss)
            self.best_report = dict(val_info)
            self.best_iteration = iteration
            self.stale_evaluations = 0
            if self.settings["restore_best"]:
                self.on_improvement(dict(val_info))
            return
        self.stale_evaluations += 1
        if self.stale_evaluations >= self.settings["patience_evaluations"]:
            self.stopped_early = True
            self.stop_iteration = iteration
            raise EarlyStopTraining("validation loss stopped improving")

    def summary(self):
        return {
            "best_iteration": self.best_iteration,
            "best_validation_loss": None if self.best_iteration is None else self.best_loss,
            "best_metric": None if self.best_iteration is None else self.best_metric,
            "monitor": self.monitor,
            "mode": self.mode,
            "best_report": self.best_report,
            "evaluations": self.evaluations,
            "stopped_early": self.stopped_early,
            "stop_iteration": self.stop_iteration,
            "history": list(self.history),
        }


class PostUpdateValidation:
    """Run fixed validation only after completed optimizer updates."""

    def __init__(self, guard, *, every, total_iterations, evaluate_loss, progress=print,
                 record_validation=lambda report: None):
        if (not isinstance(guard, ValidationLossGuard) or type(every) is not int
                or every < 1 or type(total_iterations) is not int
                or total_iterations < 1 or not callable(evaluate_loss)
                or not callable(progress) or not callable(record_validation)):
            raise ValueError("invalid post-update validation settings")
        self.guard = guard
        self.every = every
        self.total_iterations = total_iterations
        self.evaluate_loss = evaluate_loss
        self.progress = progress
        self.record_validation = record_validation

    def evaluate(self, iteration):
        started = time.perf_counter()
        result = self.evaluate_loss()
        report = dict(result) if isinstance(result, dict) else {"val_loss": float(result)}
        loss = float(report["val_loss"])
        elapsed = time.perf_counter() - started
        boundary = ("" if "worst_action_advantage" not in report else
                    f", Worst action advantage {report['worst_action_advantage']:+.3f}, "
                    f"Macro accuracy {report['macro_accuracy']:.1%}")
        self.progress(f"Iter {iteration}: Val loss {loss:.3f}{boundary}, Val took {elapsed:.3f}s")
        completed = {"iteration": iteration, "val_time": elapsed, **report}
        self.record_validation(dict(completed))
        self.guard.on_val_loss_report(completed)

    def on_train_loss_report(self, train_info):
        iteration = train_info.get("iteration")
        if type(iteration) is not int or iteration < 1:
            raise ValueError("invalid training iteration report")
        if iteration % self.every == 0 or iteration == self.total_iterations:
            self.evaluate(iteration)

    def on_val_loss_report(self, val_info):
        raise AssertionError("native pre-update validation must remain disabled")


def balanced_action_order(rows, *, count, rng):
    """Round-robin target classes so no optimizer window erases a side."""
    if type(count) is not int or count < 1:
        raise ValueError("balanced action sample count must be positive")
    sampling_rows = rows.sampling_rows() if hasattr(rows, "sampling_rows") else rows
    groups = {}
    for index, row in enumerate(sampling_rows):
        target = row.get("target_name")
        if not isinstance(target, str) or not target:
            raise ValueError("balanced action row lacks target_name")
        groups.setdefault(target, []).append(index)
    if len(groups) < 3:
        raise ValueError("balanced action sampling requires at least three target classes")
    names = sorted(groups)
    # A new shuffled epoch must never bisect an accumulated equal-action
    # optimizer window. The omitted tail is reshuffled into a later epoch.
    count -= count % len(names)
    references = [row.get("market_embedding_reference") for row in sampling_rows]
    if all(isinstance(reference, dict) and isinstance(reference.get("ticker"), str)
           for reference in references):
        by_ticker = {}
        for index, row in enumerate(sampling_rows):
            by_ticker.setdefault(references[index]["ticker"], {}).setdefault(
                row["target_name"], []).append(index)
        if any(set(group) != set(names) for group in by_ticker.values()):
            raise ValueError("indexed balanced sampling requires every action per ticker")
        order = []
        ticker_order = list(rng.permutation(sorted(by_ticker)))
        remaining = count
        for ticker in ticker_order:
            local = by_ticker[ticker]
            capacity = min(len(local[name]) for name in names) * len(names)
            take = min(capacity, remaining)
            take -= take % len(names)
            queues = {name: list(rng.permutation(local[name])) for name in names}
            cursors = {name: 0 for name in names}
            for position in range(take):
                name = names[position % len(names)]
                order.append(int(queues[name][cursors[name]]))
                cursors[name] += 1
            remaining -= take
        if remaining:
            raise ValueError("indexed action corpus cannot satisfy balanced sample count")
        return np.asarray(order, dtype=np.int64)
    queues = {name: list(rng.permutation(groups[name])) for name in names}
    cursors = {name: 0 for name in names}
    order = []
    for position in range(count):
        name = names[position % len(names)]
        if cursors[name] == len(queues[name]):
            queues[name] = list(rng.permutation(groups[name]))
            cursors[name] = 0
        order.append(int(queues[name][cursors[name]]))
        cursors[name] += 1
    return np.asarray(order, dtype=np.int64)


def balanced_validation_order(rows, *, rng):
    """Interleave action classes while visiting every fixed validation row once."""
    sampling_rows = rows.sampling_rows() if hasattr(rows, "sampling_rows") else rows
    groups = {}
    for index, row in enumerate(sampling_rows):
        target = row.get("target_name")
        if not isinstance(target, str) or not target:
            raise ValueError("balanced validation row lacks target_name")
        groups.setdefault(target, []).append(index)
    queues = {name: list(rng.permutation(indices)) for name, indices in sorted(groups.items())}
    order = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                order.append(int(queues[name].pop()))
    return np.asarray(order, dtype=np.int64)


def validate_balanced_optimizer_windows(config, rows):
    """Each accumulated update must contain equal evidence from every action."""
    if (not config["action_supervision"]["enabled"]
            or config.get("batch_sampling") != "balanced_actions"):
        return
    classes = {row.get("target_name") for row in rows}
    if None in classes or len(classes) < 2:
        raise ValueError("balanced action optimizer requires multiple target classes")
    examples_per_update = config["grad_accumulation_steps"] * config["batch_size"]
    if examples_per_update % len(classes):
        raise ValueError("balanced action optimizer window must divide evenly across target classes")


def action_boundary_metrics(rows, score_rows, *, margin=0.0):
    """Balanced exact-action evidence for checkpoint selection and promotion."""
    if (len(rows) != len(score_rows) or not rows or isinstance(margin, bool)
            or not math.isfinite(float(margin)) or margin < 0):
        raise ValueError("action boundary metrics require aligned nonempty rows and scores")
    advantages = {}
    boundary_losses = {}
    correct = {}
    for row, raw_scores in zip(rows, score_rows):
        target = row.get("target_name")
        names = row.get("action_targets", {}).get("names")
        scores = np.asarray(raw_scores, dtype=float)
        if (not isinstance(target, str) or not isinstance(names, list)
                or target not in names or scores.shape != (len(names),)
                or not np.isfinite(scores).all() or len(names) < 2):
            raise ValueError("invalid action boundary evidence")
        index = names.index(target)
        alternative = np.max(np.delete(scores, index))
        advantages.setdefault(target, []).append(float(scores[index] - alternative))
        boundary_losses.setdefault(target, []).append(
            float(np.logaddexp(0.0, float(margin) - (scores[index] - alternative))))
        correct.setdefault(target, []).append(int(index == int(np.argmax(scores))))
    per_action = {name: {
        "count": len(advantages[name]),
        "mean_target_advantage": float(np.mean(advantages[name])),
        "mean_boundary_loss": float(np.mean(boundary_losses[name])),
        "accuracy": float(np.mean(correct[name])),
    } for name in sorted(advantages)}
    return {
        "worst_action_advantage": min(row["mean_target_advantage"] for row in per_action.values()),
        "worst_action_boundary_loss": max(row["mean_boundary_loss"] for row in per_action.values()),
        "macro_accuracy": float(np.mean([row["accuracy"] for row in per_action.values()])),
        "per_action": per_action,
    }


def validate_early_stopping_coverage(config, datasets):
    """Fail closed when checkpoint selection sees partial or missing action evidence."""
    if not config["early_stopping"]["enabled"]:
        return
    valid = datasets["valid"]
    if config["val_batches"] * config["batch_size"] < len(valid):
        raise ValueError("early stopping requires complete validation coverage")
    if config["action_supervision"]["enabled"]:
        train_classes = {row.get("target_name") for row in datasets["train"]}
        valid_classes = {row.get("target_name") for row in valid}
        if train_classes != valid_classes or None in train_classes:
            raise ValueError("early stopping validation must cover all training action classes")


def configure_trainable_components(model, components):
    """Select LoRA/projector learning without changing the serialized policy contract."""
    from mlx.utils import tree_flatten
    if (not isinstance(components, list) or not components
            or len(components) != len(set(components))
            or set(components) - {"lora", "projector"}):
        raise ValueError("invalid trainable components")
    model.freeze()
    found_lora = 0
    if "lora" in components:
        for _, module in model.named_modules():
            keys = [key for key in ("lora_a", "lora_b") if hasattr(module, key)]
            if keys:
                module.unfreeze(keys=keys, recurse=False)
                found_lora += 1
        if not found_lora:
            raise ValueError("LoRA component requested but no adapters are attached")
    if "projector" in components:
        if not hasattr(model, "market_projector"):
            raise ValueError("projector component requested but no projector is attached")
        model.market_projector.unfreeze()
    leaves = dict(tree_flatten(model.trainable_parameters()))
    invalid = [name for name in leaves if not (
        name.startswith("market_projector.") or name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"})]
    if invalid or not leaves:
        raise ValueError("trainable component selection exposed invalid parameters")
    return tuple(sorted(leaves))


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


def tensor_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None,
                   sampling_strategy="random"):
    import mlx.core as mx
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("reasoning trainer currently supports one local worker")
    if len(dataset) < batch_size:
        raise ValueError("not enough supervised rows for a batch")
    rng = np.random.default_rng(seed)
    while True:
        if loop and sampling_strategy == "balanced_actions":
            order = balanced_action_order(dataset, count=len(dataset), rng=rng)
        elif not loop and sampling_strategy == "balanced_actions":
            order = balanced_validation_order(dataset, rng=rng)
        else:
            order = rng.permutation(len(dataset)) if loop else np.arange(len(dataset))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            rows = [dataset[int(i)] for i in order[start:start + batch_size]]
            yield tuple(mx.array(x) for x in pack_examples(rows, max_seq_length=max_seq_length))
        if not loop:
            return


def _batch_outputs(model, tokens, offsets, lengths, valid, probabilities, values,
                   embeddings, available, *, config):
    import mlx.core as mx
    from .projector import market_logits
    losses = []
    action_scores = []
    for index in range(tokens.shape[0]):
        inputs = tokens[index, :, :-1]
        logits = (market_logits(model, inputs, embeddings[index:index+1], available[index:index+1])
                  if config["input_mode"] == "embeddings" else model(inputs)).astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = tokens[index, :, 1:, None]
        token_scores = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
        steps = mx.arange(1, tokens.shape[-1])
        mask = (steps >= offsets[index, :, None]) & (steps < lengths[index, :, None]) & valid[index, :, None]
        scores = (action_completion_scores(token_scores, mask, xp=mx)
                  if config["action_supervision"]["enabled"] else
                  mean_completion_scores(token_scores, mask, xp=mx))
        action_scores.append(scores)
        if config["action_supervision"]["enabled"]:
            losses.append(action_objective(scores, probabilities[index], values[index],
                config["action_supervision"], xp=mx, valid=valid[index]))
        else:
            losses.append(completion_objective(scores, valid[index], xp=mx))
    return (mx.stack(losses).mean(), mx.array(tokens.shape[0]),
            mx.stack(action_scores))


def batch_loss(model, tokens, offsets, lengths, valid, probabilities, values,
               embeddings, available, *, config):
    loss, tokens_count, _ = _batch_outputs(
        model, tokens, offsets, lengths, valid, probabilities, values,
        embeddings, available, config=config)
    return loss, tokens_count


def evaluate_action_validation(model, dataset, config):
    """Evaluate every fixed validation row once and expose balanced boundaries."""
    import mlx.core as mx
    order = balanced_validation_order(dataset, rng=np.random.default_rng(config["seed"]))
    batch_size = config["batch_size"]
    rows_seen, score_rows, weighted_loss = [], [], 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        if len(indices) < batch_size:
            raise ValueError("validation rows must form complete batches")
        rows = [dataset[int(index)] for index in indices]
        tensors = tuple(mx.array(value) for value in pack_examples(
            rows, max_seq_length=config["max_seq_length"]))
        loss, _, scores = _batch_outputs(model, *tensors, config=config)
        mx.eval(loss, scores)
        weighted_loss += float(loss.item()) * len(rows)
        rows_seen.extend(rows)
        score_rows.extend(scores.tolist())
    metrics = action_boundary_metrics(
        rows_seen, score_rows, margin=config["action_supervision"]["margin"])
    return {"val_loss": weighted_loss / len(rows_seen), **metrics}


def train_supervised(config, view):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import evaluate, train, TrainingArgs
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
    configure_trainable_components(model, config["trainable_components"])
    optimizer_kind = config["optimizer"]
    constructors = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_kind not in constructors:
        raise ValueError("configured optimizer/schedule unsupported by tensor SFT adapter")
    learning_rate = config["learning_rate"]
    if config["lr_schedule"] is not None:
        schedule = config["lr_schedule"]
        learning_rate = optim.cosine_decay(
            learning_rate, schedule["decay_updates"], end=schedule["end"])
    optimizer = constructors[optimizer_kind](learning_rate=learning_rate,
        **config["optimizer_config"].get(optimizer_kind, {}))
    from .mlx_sft import PreparedDataset
    datasets = {role: PreparedDataset(view, role) for role in ("train", "valid")}
    validate_early_stopping_coverage(config, datasets)
    validate_balanced_optimizer_windows(config, datasets["train"])
    destination.mkdir(parents=True)
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2))
    best = destination / ".best-validation"
    guard = ValidationLossGuard(config["early_stopping"],
        on_improvement=lambda report: (
            best.mkdir(exist_ok=True), export_policy_weights(model, best)))
    iterator = partial(tensor_batches, seed=config["seed"],
                       sampling_strategy=config.get("batch_sampling", "random"))
    def evaluate_loss():
        if config["action_supervision"]["enabled"]:
            value = evaluate_action_validation(model, datasets["valid"], config)
        else:
            value = evaluate(model, datasets["valid"], batch_size=config["batch_size"],
                num_batches=config["val_batches"], max_seq_length=config["max_seq_length"],
                loss=partial(batch_loss, config=config), iterate_batches=iterator,
                clear_cache_threshold=config["clear_cache_threshold"])
        model.train()
        return value
    metrics_path = config.get("validation_metrics_path")
    def record_validation(report):
        if metrics_path is None:
            return
        with Path(metrics_path).open("a") as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    validation = PostUpdateValidation(guard, every=config["steps_per_eval"],
        total_iterations=config["iters"], evaluate_loss=evaluate_loss,
        record_validation=record_validation)
    args = TrainingArgs(batch_size=config["batch_size"], iters=config["iters"],
        val_batches=config["val_batches"], steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"], steps_per_save=config["save_every"],
        adapter_file=str(destination / "adapters.safetensors"), max_seq_length=config["max_seq_length"],
        grad_checkpoint=config["grad_checkpoint"], grad_accumulation_steps=config["grad_accumulation_steps"],
        clear_cache_threshold=config["clear_cache_threshold"])
    try:
        validation.evaluate(0)
        train(model, optimizer, datasets["train"], None, args=args,
            loss=partial(batch_loss, config=config),
            iterate_batches=iterator, training_callback=validation)
    except EarlyStopTraining:
        print(f"Early stopping at validation iteration {guard.stop_iteration}; "
              f"best iteration was {guard.best_iteration}.", flush=True)
    if config["early_stopping"]["restore_best"]:
        if guard.best_iteration is None or not (best / "adapters.safetensors").is_file():
            raise ValueError("early stopping did not produce a best validation checkpoint")
        model.load_weights(str(best / "adapters.safetensors"), strict=False)
        if config["input_mode"] == "embeddings":
            restore_projector(model, best)
    export_policy_weights(model, destination)
    (destination / "training_selection.json").write_text(json.dumps(guard.summary(), indent=2))
    if best.exists():
        shutil.rmtree(best)

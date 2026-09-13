"""Native MLX-LM trainer with explicit tensor batches and a trading loss.

No second optimizer loop. Reuses native LoRA conversion, accumulation,
checkpointing, evaluation and loss callback interfaces.
"""
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np

from .supervision import (
    action_completion_scores, action_objective, completion_objective,
    hierarchical_action_objective, mean_completion_scores,
)


class EarlyStopTraining(RuntimeError):
    """Private control signal raised only at a completed validation boundary."""


def resolve_training_budget(config, *, train_rows, valid_rows):
    """Resolve an optional epoch ceiling against the actual prepared corpus."""
    result = dict(config)
    epochs = config.get("epochs")
    if epochs is not None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer or null")
        batches = math.ceil(train_rows / config["batch_size"])
        accumulation = config["grad_accumulation_steps"]
        result["iters"] = math.ceil(epochs * batches / accumulation) * accumulation
    return result


class TrainingEventLog:
    """Human training progress plus durable JSONL machine evidence."""

    def __init__(self, path, *, events_path, prefix, train_batches, total_iterations):
        if (type(train_batches) is not int or train_batches < 1
                or type(total_iterations) is not int or total_iterations < 1
                or not isinstance(prefix, str) or not prefix.strip()):
            raise ValueError("training log requires positive epoch dimensions")
        self.path = Path(path)
        self.events_path = Path(events_path)
        self.prefix = prefix.strip()
        self.train_batches = train_batches
        self.total_iterations = total_iterations
        self.maximum_epochs = total_iterations / train_batches
        self.restore_best = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [path for path in (self.path, self.events_path) if path.exists()]
        if existing:
            raise FileExistsError(f"training log already exists: {existing[0]}")

    @staticmethod
    def _epoch(value):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")

    def _write_event(self, payload):
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_line(self, value):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(value + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record_start(self, payload):
        event = {"event": "start", "total_iterations": self.total_iterations,
                 "train_batches_per_epoch": self.train_batches,
                 "maximum_epochs": self.maximum_epochs, **payload}
        self.restore_best = bool(payload.get("early_stopping", {}).get("restore_best"))
        self._write_event(event)
        targeted = payload.get("targeted_sampling_summary")
        targeted_text = ("" if targeted is None else
            f" mistake_draws={targeted['mistake_draws']}"
            f" anchor_draws={targeted['anchor_draws']}"
            f" selection_rounds={targeted['rounds']}")
        self._write_line(
            f"[{self.prefix}] status=started epochs={self._epoch(self.maximum_epochs)} "
            f"train_rows={payload['train_rows']} valid_rows={payload['valid_rows']} "
            f"eval_every={self._epoch(payload['evaluation_every_epochs'])} "
            f"patience={payload['early_stopping']['patience_evaluations']} "
            f"min_delta={payload['early_stopping']['min_delta']}"
            f"{targeted_text}")

    def record_training(self, report):
        self._write_event({"event": "training", **report})
        fields = [
            f"[{self.prefix}] epoch={self._epoch(report['epoch'])}/"
            f"{self._epoch(self.maximum_epochs)}",
            f"train_loss={float(report['train_loss']):.4f}",
        ]
        if "learning_rate" in report:
            fields.append(f"lr={float(report['learning_rate']):.3e}")
        if "iterations_per_second" in report:
            fields.append(f"iter_sec={float(report['iterations_per_second']):.3f}")
        if "tokens_per_second" in report:
            fields.append(f"tokens_sec={float(report['tokens_per_second']):.1f}")
        if "peak_memory" in report:
            fields.append(f"peak_mem_gb={float(report['peak_memory']):.3f}")
        self._write_line(" ".join(fields))

    def record_validation_started(self, report):
        self._write_event({"event": "validation_started", **report})
        self._write_line(
            f"[{self.prefix}] epoch={self._epoch(report['epoch'])}/"
            f"{self._epoch(self.maximum_epochs)} validation=started")

    def record_validation(self, report):
        self._write_event({"event": "validation", **report})
        train_loss = ("NA" if report.get("train_loss") is None else
                      f"{float(report['train_loss']):.4f}")
        marker = "  *" if report["checkpoint_selected"] else ""
        self._write_line(
            f"[{self.prefix}] epoch={self._epoch(report['epoch'])}/"
            f"{self._epoch(self.maximum_epochs)} train_loss={train_loss} "
            f"val_loss={float(report['val_loss']):.4f} "
            f"{report['monitor']}={float(report['monitor_value']):.4f} "
            f"best={float(report['best_metric']):.4f} "
            f"patience={report['stale_evaluations']}/"
            f"{report['patience_evaluations']}{marker}")

    def record_complete(self, summary):
        best = summary.get("best_iteration")
        best_epoch = None if best is None else best / self.train_batches
        self._write_event({"event": "complete", **summary, "best_epoch": best_epoch})
        self._write_line(
            f"[{self.prefix}] status=complete "
            f"best_epoch={'NA' if best_epoch is None else self._epoch(best_epoch)} "
            f"restored_best={str(self.restore_best).lower()} "
            f"stopped_early={str(bool(summary.get('stopped_early'))).lower()}")

    def record_failure(self, error):
        self._write_event({"event": "failed", "error_type": type(error).__name__,
                           "message": str(error)})
        self._write_line(
            f"[{self.prefix}] status=failed error={type(error).__name__} "
            f"message={str(error)}")


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
        self.last_decision = None

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
            self.last_decision = self._decision(False, float(metric))
            return self.last_decision
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
            self.last_decision = self._decision(True, float(metric))
            return self.last_decision
        self.stale_evaluations += 1
        if self.stale_evaluations >= self.settings["patience_evaluations"]:
            self.stopped_early = True
            self.stop_iteration = iteration
            self.last_decision = self._decision(False, float(metric))
            raise EarlyStopTraining("validation loss stopped improving")
        self.last_decision = self._decision(False, float(metric))
        return self.last_decision

    def _decision(self, checkpoint_selected, monitor_value):
        return {
            "checkpoint_selected": checkpoint_selected,
            "monitor": self.monitor,
            "monitor_value": monitor_value,
            "best_metric": None if self.best_iteration is None else self.best_metric,
            "best_iteration": self.best_iteration,
            "stale_evaluations": self.stale_evaluations,
            "patience_evaluations": self.settings["patience_evaluations"],
            "min_delta": float(self.settings["min_delta"]),
            "stopped_early": self.stopped_early,
        }

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
                 train_batches=None, record_training=lambda report: None,
                 record_validation_started=lambda report: None,
                 record_validation=lambda report: None):
        if train_batches is None:
            train_batches = total_iterations
        if (not isinstance(guard, ValidationLossGuard) or type(every) is not int
                or every < 1 or type(total_iterations) is not int
                or total_iterations < 1 or not callable(evaluate_loss)
                or type(train_batches) is not int or train_batches < 1
                or not callable(progress) or not callable(record_training)
                or not callable(record_validation_started)
                or not callable(record_validation)):
            raise ValueError("invalid post-update validation settings")
        self.guard = guard
        self.every = every
        self.total_iterations = total_iterations
        self.evaluate_loss = evaluate_loss
        self.progress = progress
        self.train_batches = train_batches
        self.record_training = record_training
        self.record_validation_started = record_validation_started
        self.record_validation = record_validation
        self.latest_training = None

    def evaluate(self, iteration):
        epoch = iteration / self.train_batches
        self.record_validation_started({"iteration": iteration, "epoch": epoch})
        started = time.perf_counter()
        result = self.evaluate_loss()
        report = dict(result) if isinstance(result, dict) else {"val_loss": float(result)}
        loss = float(report["val_loss"])
        elapsed = time.perf_counter() - started
        if "worst_task_advantage" in report:
            boundary = (
                f", Worst task advantage {report['worst_task_advantage']:+.3f}, "
                f"Task macro accuracy {report['task_macro_accuracy']:.1%}")
        elif "worst_action_advantage" in report:
            boundary = (
                f", Worst action advantage {report['worst_action_advantage']:+.3f}, "
                f"Macro accuracy {report['macro_accuracy']:.1%}")
        else:
            boundary = ""
        completed = {"iteration": iteration, "epoch": epoch,
                     "val_time": elapsed, **report}
        if self.latest_training is not None:
            completed["train_loss"] = self.latest_training.get("train_loss")
        stopping = None
        try:
            decision = self.guard.on_val_loss_report(completed)
        except EarlyStopTraining as error:
            decision, stopping = self.guard.last_decision, error
        completed.update(decision)
        patience = (f"{completed['stale_evaluations']}/"
                    f"{completed['patience_evaluations']}")
        self.progress(
            f"Epoch {epoch:.3f} (iteration {iteration}/{self.total_iterations}): "
            f"Val loss {loss:.3f}{boundary}, Best {completed['best_metric']}, "
            f"Patience {patience}, Val took {elapsed:.3f}s")
        self.record_validation(dict(completed))
        if stopping is not None:
            raise stopping

    def reuse(self, iteration, result):
        """Seed validation from an authenticated identical-policy receipt."""
        epoch = iteration / self.train_batches
        report = {"iteration": iteration, "epoch": epoch, "val_time": 0.0,
                  "receipt_reused": True, **result}
        decision = self.guard.on_val_loss_report(report)
        report.update(decision)
        self.progress(
            f"Epoch {epoch:.3f} (iteration {iteration}/{self.total_iterations}): "
            f"reused authenticated validation receipt, Best {report['best_metric']}")
        self.record_validation(dict(report))

    def on_train_loss_report(self, train_info):
        iteration = train_info.get("iteration")
        if type(iteration) is not int or iteration < 1:
            raise ValueError("invalid training iteration report")
        completed = dict(train_info)
        completed["epoch"] = iteration / self.train_batches
        self.latest_training = completed
        self.record_training(completed)
        self.guard.on_train_loss_report(completed)
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
        if all(set(group) == set(names) for group in by_ticker.values()):
            order = []
            ticker_order = list(rng.permutation(sorted(by_ticker)))
            windows, remainder = divmod(count // len(names), len(ticker_order))
            for ticker_index, ticker in enumerate(ticker_order):
                local = by_ticker[ticker]
                queues = {name: list(rng.permutation(local[name])) for name in names}
                cursors = {name: 0 for name in names}
                local_windows = windows + (1 if ticker_index < remainder else 0)
                for _ in range(local_windows):
                    for name in names:
                        if cursors[name] == len(queues[name]):
                            queues[name] = list(rng.permutation(local[name]))
                            cursors[name] = 0
                        order.append(int(queues[name][cursors[name]]))
                        cursors[name] += 1
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


def hierarchical_boundary_metrics(rows, score_rows, *, margin=0.0):
    """Measure each binary trade decision and reconstructed legal actions."""
    action_metrics = action_boundary_metrics(rows, score_rows, margin=margin)
    evidence = {}
    for row, raw_scores in zip(rows, score_rows):
        names = row.get("action_targets", {}).get("names")
        values = np.asarray(row.get("action_targets", {}).get("values"), dtype=float)
        scores = np.asarray(raw_scores, dtype=float)
        if set(names or ()) == {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}:
            by_name = dict(zip(names, zip(values, scores)))
            wait_value, wait_score = by_name["WAIT"]
            long_value, long_score = by_name["ENTER_LONG_1"]
            short_value, short_score = by_name["ENTER_SHORT_1"]
            best_value = max(long_value, short_value)
            directional = long_value != short_value
            enter = best_value > wait_value and directional
            best_side_score = (long_score if long_value > short_value else short_score)
            if not enter:
                best_side_score = max(long_score, short_score)
            target = "entry.ENTER" if enter else "entry.WAIT"
            advantage = (best_side_score - wait_score) if enter else (wait_score - best_side_score)
            evidence.setdefault(target, []).append(advantage)
            if enter:
                target = "direction.LONG" if long_value > short_value else "direction.SHORT"
                advantage = (long_score - short_score) if long_value > short_value else (short_score - long_score)
                evidence.setdefault(target, []).append(advantage)
        elif set(names or ()) == {"HOLD", "CLOSE"}:
            by_name = dict(zip(names, zip(values, scores)))
            hold_value, hold_score = by_name["HOLD"]
            close_value, close_score = by_name["CLOSE"]
            hold = hold_value >= close_value
            target = "management.HOLD" if hold else "management.CLOSE"
            advantage = (hold_score - close_score) if hold else (close_score - hold_score)
            evidence.setdefault(target, []).append(advantage)
        else:
            raise ValueError("invalid hierarchical boundary evidence")
    per_task = {name: {
        "count": len(advantages),
        "mean_target_advantage": float(np.mean(advantages)),
        "mean_boundary_loss": float(np.mean([
            np.logaddexp(0.0, float(margin) - advantage) for advantage in advantages])),
        "accuracy": float(np.mean([advantage >= 0 for advantage in advantages])),
    } for name, advantages in sorted(evidence.items())}
    if not per_task:
        raise ValueError("hierarchical metrics require decision evidence")
    return {
        **action_metrics,
        "worst_task_advantage": min(row["mean_target_advantage"] for row in per_task.values()),
        "worst_task_boundary_loss": max(row["mean_boundary_loss"] for row in per_task.values()),
        "task_macro_accuracy": float(np.mean([row["accuracy"] for row in per_task.values()])),
        "per_task": per_task,
    }


def validate_early_stopping_coverage(config, datasets):
    """Fail closed when checkpoint selection sees partial or missing action evidence."""
    if not config["early_stopping"]["enabled"]:
        return
    valid = datasets["valid"]
    if config["val_batches"] * config["validation_batch_size"] < len(valid):
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


def build_optimizer(config):
    """Build the configured MLX optimizer without coupling LoRA and projector rates."""
    import mlx.optimizers as optim

    optimizer_kind = config["optimizer"]
    constructors = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_kind not in constructors:
        raise ValueError("configured optimizer/schedule unsupported by tensor SFT adapter")
    constructor = constructors[optimizer_kind]
    options = config["optimizer_config"].get(optimizer_kind, {})
    component_rates = config.get("component_learning_rates")
    if component_rates is not None:
        projector = constructor(
            learning_rate=component_rates["projector"], **options)
        lora = constructor(learning_rate=component_rates["lora"], **options)
        return optim.MultiOptimizer(
            [projector, lora],
            filters=[lambda name, _: name.startswith("market_projector.")],
        )
    learning_rate = config["learning_rate"]
    if config["lr_schedule"] is not None:
        schedule = config["lr_schedule"]
        learning_rate = optim.cosine_decay(
            learning_rate, schedule["decay_updates"], end=schedule["end"])
    return constructor(learning_rate=learning_rate, **options)


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
    task_codes = np.full(len(rows), -1, dtype=np.int32)
    for i, (row, group) in enumerate(zip(rows, alternatives)):
        for j, (sequence, offset) in enumerate(group):
            if not 0 < offset < len(sequence):
                raise ValueError("invalid supervised answer offset")
            tokens[i, j, :len(sequence)] = sequence
            offsets[i, j], lengths[i, j], valid[i, j] = offset, len(sequence), True
        target = row.get("action_targets")
        probabilities[i, :len(group)] = [1.] if target is None else target["probabilities"]
        values[i, :len(group)] = [0.] if target is None else target["values"]
        if target is not None:
            names = target["names"]
            if names == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]:
                task_codes[i] = 0
            elif names == ["HOLD", "CLOSE"]:
                task_codes[i] = 1
            else:
                raise ValueError("unsupported legal-action state")
    causal_states = [row.get("causal_state", []) for row in rows]
    if len({np.asarray(x).shape for x in causal_states}) != 1:
        raise ValueError("causal states must share one configured shape")
    embeddings = [row.get("market_embeddings", [[0.]]) for row in rows]
    if len({np.asarray(x).shape for x in embeddings}) != 1:
        raise ValueError("embedding windows must share one configured shape")
    available = [row.get("market_available", [True]) for row in rows]
    packed = (tokens, offsets, lengths, valid, probabilities, values, task_codes,
            np.asarray(causal_states, np.float32),
            np.asarray(embeddings, np.float32), np.asarray(available, bool))
    market = [row.get("market_targets") for row in rows]
    if any(item is not None for item in market):
        if not all(item is not None for item in market):
            raise ValueError("cannot mix market distillation and token training rows")
        return packed + tuple(np.asarray([item[key] for item in market], dtype=dtype)
            for key, dtype in (("positions", np.int32), ("probabilities", np.float32),
                               ("weights", np.float32), ("label_ids", np.int32)))
    corrective = [row.get("error_selected_distillation") for row in rows]
    result = packed
    if any(item is not None for item in corrective):
        if not all(item is not None for item in corrective):
            raise ValueError("cannot mix corrected action rows with missing teacher targets")
        width = max(len(item["tokens"]) for item in corrective)
        query_tokens = np.zeros((len(rows), 1, width), np.int32)
        for index, item in enumerate(corrective):
            query_tokens[index, 0, :len(item["tokens"])] = item["tokens"]
        result += (query_tokens,) + tuple(
            np.asarray([item["market_targets"][key] for item in corrective], dtype=dtype)
            for key, dtype in (("positions", np.int32), ("probabilities", np.float32),
                               ("weights", np.float32), ("label_ids", np.int32)))
    retention = [row.get("mastered_anchor_retention") for row in rows]
    if any(item is not None for item in retention):
        if not all(item is not None for item in retention):
            raise ValueError("cannot mix retained anchors with unmarked correction rows")
        parent_scores = np.zeros_like(probabilities)
        boundary_masks = np.zeros((len(rows), 3), dtype=bool)
        boundary_names = ("entry", "direction", "management")
        for index, (item, group) in enumerate(zip(retention, alternatives)):
            scores = item.get("scores")
            boundaries = item.get("boundaries")
            if (not isinstance(boundaries, dict)
                    or set(boundaries) != set(boundary_names)
                    or any(not isinstance(boundaries[name], (bool, np.bool_))
                           for name in boundary_names)
                    or not isinstance(scores, list) or len(scores) != len(group)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) for value in scores)):
                raise ValueError("invalid mastered anchor retention row")
            parent_scores[index, :len(group)] = scores
            boundary_masks[index] = [boundaries[name] for name in boundary_names]
        result += (parent_scores, boundary_masks)
    return result


def tensor_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None,
                   sampling_strategy="random", include_partial=False, skip_batches=0,
                   coverage_sampler=None, targeted_sampler=None,
                   mastered_anchor_retention=None):
    import mlx.core as mx
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("reasoning trainer currently supports one local worker")
    if len(dataset) < batch_size:
        raise ValueError("not enough supervised rows for a batch")
    if coverage_sampler is not None and targeted_sampler is not None:
        raise ValueError("training batches require one rotating sampler")
    rng = np.random.default_rng(seed)
    round_index = 0
    while True:
        if loop and targeted_sampler is not None:
            order = targeted_sampler.order(round_index)
        elif loop and coverage_sampler is not None:
            order = coverage_sampler.order(round_index)
        elif loop and sampling_strategy == "balanced_actions":
            order = balanced_action_order(dataset, count=len(dataset), rng=rng)
        elif not loop and sampling_strategy == "balanced_actions":
            order = balanced_validation_order(dataset, rng=rng)
        else:
            order = rng.permutation(len(dataset)) if loop else np.arange(len(dataset))
        stop = len(order) if include_partial else len(order) - batch_size + 1
        for start in range(0, stop, batch_size):
            if loop and skip_batches:
                skip_batches -= 1
                continue
            indices = order[start:start + batch_size]
            rows = [dataset[int(i)] for i in indices]
            if mastered_anchor_retention is not None:
                if targeted_sampler is None:
                    raise ValueError("mastered anchor retention requires targeted sampling")
                rows = [targeted_sampler.training_row(
                    int(index), row, retain_mastery=True)
                    for index, row in zip(indices, rows)]
            yield tuple(mx.array(x) for x in pack_examples(rows, max_seq_length=max_seq_length))
        if not loop:
            return
        round_index += 1


def selected_token_scores(logits, targets):
    """Preserve target scores without allocating full-vocabulary log probabilities."""
    import mlx.core as mx
    logits = logits.astype(mx.float32)
    selected = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return selected - mx.logsumexp(logits, axis=-1)


def anchor_retention_loss(scores, parent_scores, anchor_mask, *, temperature, valid=None):
    """KL from a correct frozen parent, with mistake rows contributing exactly zero."""
    import mlx.core as mx
    if valid is None:
        valid = mx.ones_like(scores, dtype=mx.bool_)
    floor = mx.array(-1e9, dtype=mx.float32)
    student = mx.where(valid, scores.astype(mx.float32) / temperature, floor)
    teacher = mx.where(valid, parent_scores.astype(mx.float32) / temperature, floor)
    student_log = student - mx.logsumexp(student, axis=-1, keepdims=True)
    teacher_log = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    per_row = mx.sum(mx.exp(teacher_log) * (teacher_log - student_log), axis=-1)
    mask = anchor_mask.astype(mx.float32)
    return temperature ** 2 * mx.sum(per_row * mask) / mx.maximum(mx.sum(mask), 1.)


def boundary_retention_loss(scores, parent_scores, boundary_masks, *, temperature):
    """Preserve only the hierarchical decisions mastered by the frozen parent."""
    import mlx.core as mx
    scores = scores.astype(mx.float32)
    parent_scores = parent_scores.astype(mx.float32)
    if scores.shape[1] == 2:
        padding = mx.zeros((scores.shape[0], 1), dtype=mx.float32)
        scores = mx.concatenate([scores, padding], axis=1)
        parent_scores = mx.concatenate([parent_scores, padding], axis=1)

    def divergence(student, teacher):
        student = student / temperature
        teacher = teacher / temperature
        student_log = student - mx.logsumexp(student, axis=-1, keepdims=True)
        teacher_log = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
        return mx.sum(mx.exp(teacher_log) * (teacher_log - student_log), axis=-1)

    entry = divergence(
        mx.stack([scores[:, 0], mx.maximum(scores[:, 1], scores[:, 2])], axis=-1),
        mx.stack([parent_scores[:, 0],
                  mx.maximum(parent_scores[:, 1], parent_scores[:, 2])], axis=-1))
    direction = divergence(scores[:, 1:3], parent_scores[:, 1:3])
    management = divergence(scores[:, :2], parent_scores[:, :2])
    losses = mx.stack([entry, direction, management], axis=-1)
    mask = boundary_masks.astype(mx.float32)
    return temperature ** 2 * mx.sum(losses * mask) / mx.maximum(mx.sum(mask), 1.)


def _batch_outputs(model, tokens, offsets, lengths, valid, probabilities, values, task_codes,
                   causal_states, embeddings, available, *extras, config):
    if config.get("market_distillation") is not None:
        if len(extras) != 4:
            raise ValueError("market distillation requires an authenticated short-query view")
        query_positions, teacher_probabilities, teacher_weights, label_ids = extras
        from .market_distillation import market_outputs
        return market_outputs(model, tokens, embeddings, available, causal_states,
            query_positions, teacher_probabilities, teacher_weights, label_ids)
    corrective = config.get("error_selected_distillation")
    retention = config.get("mastered_anchor_retention")
    if corrective is None and retention is None and extras:
        raise ValueError("short-query view cannot be trained with token loss")
    expected_extras = (5 if corrective is not None else 0) + (2 if retention is not None else 0)
    if (corrective is not None or retention is not None) and len(extras) != expected_extras:
        raise ValueError("error-selected action correction requires four-teacher targets")
    if config.get("market_loss_chunk_size") is not None:
        from .chunked_loss import chunked_market_outputs
        return chunked_market_outputs(model, tokens, offsets, lengths, valid, probabilities,
            values, task_codes, causal_states, embeddings, available, config=config,
            chunk_size=config["market_loss_chunk_size"])
    import mlx.core as mx
    from .projector import market_logits
    batch_size, actions, sequence_length = tokens.shape
    flat_inputs = tokens[:, :, :-1].reshape(
        batch_size * actions, sequence_length - 1)
    if config["input_mode"] == "embeddings":
        context_steps, embedding_dim = embeddings.shape[1:]
        flat_embeddings = mx.broadcast_to(
            embeddings[:, None, :, :],
            (batch_size, actions, context_steps, embedding_dim),
        ).reshape(batch_size * actions, context_steps, embedding_dim)
        flat_available = mx.broadcast_to(
            available[:, None, :],
            (batch_size, actions, context_steps),
        ).reshape(batch_size * actions, context_steps)
        state_dim = causal_states.shape[-1]
        flat_causal_states = mx.broadcast_to(
            causal_states[:, None, :],
            (batch_size, actions, state_dim),
        ).reshape(batch_size * actions, state_dim)
        logits = market_logits(
            model, flat_inputs, flat_embeddings, flat_available, flat_causal_states)
    else:
        logits = model(flat_inputs)
    targets = tokens[:, :, 1:].reshape(
        batch_size * actions, sequence_length - 1)
    token_scores = selected_token_scores(logits, targets).reshape(
            batch_size, actions, sequence_length - 1)
    steps = mx.arange(1, sequence_length)
    mask = ((steps[None, None, :] >= offsets[:, :, None])
            & (steps[None, None, :] < lengths[:, :, None])
            & valid[:, :, None])
    if config["action_supervision"]["enabled"]:
        scores = action_completion_scores(
            token_scores.reshape(batch_size * actions, sequence_length - 1),
            mask.reshape(batch_size * actions, sequence_length - 1), xp=mx,
        ).reshape(batch_size, actions)
    else:
        scores = mean_completion_scores(token_scores, mask, xp=mx)
    losses = []
    correction_masks = None if retention is None else ~extras[-1]
    for index in range(batch_size):
        if config["action_supervision"]["enabled"]:
            if config.get("decision_objective", "full_action") == "hierarchical_binary":
                losses.append(hierarchical_action_objective(
                    scores[index], probabilities[index], values[index],
                    config["action_supervision"], task_code=task_codes[index], xp=mx,
                    correction_boundaries=(None if correction_masks is None
                                           else correction_masks[index])))
            else:
                losses.append(action_objective(scores[index], probabilities[index], values[index],
                    config["action_supervision"], xp=mx, valid=valid[index]))
        else:
            losses.append(completion_objective(scores[index], valid[index], xp=mx))
    loss = mx.stack(losses).mean()
    if corrective is not None:
        query_tokens, positions, teacher_probabilities, teacher_weights, label_ids = extras[:5]
        from .market_distillation import market_outputs
        teacher_loss, _, _ = market_outputs(
            model, query_tokens, embeddings, available, causal_states,
            positions, teacher_probabilities, teacher_weights, label_ids)
        loss = loss + corrective["loss_weight"] * teacher_loss
    if retention is not None:
        parent_scores, boundary_masks = extras[-2:]
        loss = loss + retention["loss_weight"] * boundary_retention_loss(
            scores, parent_scores, boundary_masks,
            temperature=retention["temperature"])
    return loss, mx.array(tokens.shape[0]), scores


def batch_loss(model, tokens, offsets, lengths, valid, probabilities, values, task_codes,
               causal_states, embeddings, available, *extras, config):
    loss, tokens_count, _ = _batch_outputs(
        model, tokens, offsets, lengths, valid, probabilities, values, task_codes,
        causal_states, embeddings, available, *extras, config=config)
    return loss, tokens_count


def evaluate_action_validation(model, dataset, config, *, on_scored=None):
    """Evaluate every fixed validation row once and expose balanced boundaries."""
    import mlx.core as mx
    # Frozen validation rows are never training anchors. Retention is evaluated
    # by the campaign's same-row parent/candidate gate, not added to val loss.
    evaluation_config = {**config, "mastered_anchor_retention": None,
                         "error_selected_distillation": None,
                         "market_distillation": None}
    order = balanced_validation_order(dataset, rng=np.random.default_rng(config["seed"]))
    batch_size = config["validation_batch_size"]
    rows_seen, score_rows, weighted_loss = [], [], 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        if len(indices) < batch_size:
            raise ValueError("validation rows must form complete batches")
        rows = [dataset[int(index)] for index in indices]
        tensors = tuple(mx.array(value) for value in pack_examples(
            rows, max_seq_length=config["max_seq_length"]))
        # Frozen action validation is teacher-free. The prepared rows may carry
        # training-only distillation/retention tensors, but they must neither be
        # computed nor influence checkpoint selection.
        loss, _, scores = _batch_outputs(model, *tensors[:10], config=evaluation_config)
        mx.eval(loss, scores)
        weighted_loss += float(loss.item()) * len(rows)
        rows_seen.extend(rows)
        # Mixed flat/position batches pad HOLD/CLOSE rows to the wider
        # WAIT/LONG/SHORT tensor. Metrics and exported assessment evidence must
        # contain only actions legal for that exact row.
        legal_action_mask = np.asarray(tensors[3], dtype=bool)
        batch_scores = [np.asarray(values)[mask].tolist()
                        for values, mask in zip(scores.tolist(), legal_action_mask)]
        score_rows.extend(batch_scores)
        if on_scored is not None:
            for index, values in zip(indices, batch_scores):
                on_scored(int(index), list(values))
    metric = (hierarchical_boundary_metrics
              if config.get("decision_objective", "full_action") == "hierarchical_binary"
              else action_boundary_metrics)
    metrics = metric(rows_seen, score_rows, margin=config["action_supervision"]["margin"])
    return {"val_loss": weighted_loss / len(rows_seen), **metrics}


def authenticated_initial_validation(config, view, *, valid_rows):
    """Load a frozen parent receipt only when every relevant identity matches."""
    descriptor = config.get("initial_validation_receipt")
    if descriptor is None:
        return None
    from .integrity import file_digest
    from .mlx_sft import read_sft_config
    root = Path(descriptor["path"])
    summary_path, scores_path = root / "summary.json", root / "scores.jsonl"
    policy_path = Path(descriptor["policy_config_path"])
    view_manifest = Path(view) / "view_manifest.json"
    if (file_digest(summary_path) != descriptor["summary_sha256"]
            or file_digest(scores_path) != descriptor["scores_sha256"]
            or file_digest(policy_path) != descriptor["policy_config_sha256"]
            or file_digest(view_manifest) != descriptor["view_manifest_sha256"]):
        raise ValueError("initial validation receipt identity changed")
    summary = json.loads(summary_path.read_text())
    score_rows = sum(1 for line in scores_path.open() if line.strip())
    if (summary.get("role") != "valid"
            or summary.get("weights_updated") is not False
            or summary.get("rows") != valid_rows
            or score_rows != valid_rows
            or summary.get("config_sha256") != descriptor["policy_config_sha256"]
            or summary.get("view_manifest_sha256") != descriptor["view_manifest_sha256"]):
        raise ValueError("initial validation receipt does not match the frozen parent")
    parent = read_sft_config(policy_path, root=config.get("workspace_root"))
    expected_weights = Path(parent["adapter_path"]) / "adapters.safetensors"
    if expected_weights.resolve() != Path(config["resume_adapter_file"]).resolve():
        raise ValueError("initial validation receipt parent differs from warm start")
    metrics = summary.get("metrics")
    monitor = config["early_stopping"]["monitor"]
    if (not isinstance(metrics, dict)
            or isinstance(metrics.get("val_loss"), bool)
            or not isinstance(metrics.get("val_loss"), (int, float))
            or not math.isfinite(float(metrics["val_loss"]))
            or isinstance(metrics.get(monitor), bool)
            or not isinstance(metrics.get(monitor), (int, float))
            or not math.isfinite(float(metrics[monitor]))):
        raise ValueError("initial validation receipt lacks checkpoint metrics")
    return dict(metrics)


def train_supervised(config, view):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import evaluate, train, TrainingArgs
    from functools import partial
    from .model_config import verify_adapter_base, validate_sft_parent_contract
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
        validate_sft_parent_contract(config, metadata)
        for key in ("lora_parameters", "num_layers", "chat_template_kwargs", "input_mode"):
            if metadata.get(key) != config.get(key):
                raise ValueError(f"SFT warm-start contract differs at {key}")
        from .projector import state_extension_of
        extends_state = state_extension_of(metadata.get("projector"), config.get("projector"))
        if metadata.get("projector") != config.get("projector") and not extends_state:
            raise ValueError("SFT warm-start contract differs at projector")
        model.load_weights(parent, strict=False)
    if config["input_mode"] == "embeddings":
        attach_projector(model, config["projector"])
        if parent is not None:
            restore_projector(model, Path(parent).parent,
                              allow_state_extension=extends_state)
    configure_trainable_components(model, config["trainable_components"])
    optimizer = build_optimizer(config)
    from .mlx_sft import PreparedDataset
    datasets = {role: PreparedDataset(view, role) for role in ("train", "valid")}
    initial_validation = authenticated_initial_validation(
        config, view, valid_rows=len(datasets["valid"]))
    coverage_sampler = None
    targeted_sampler = None
    round_rows = len(datasets["train"])
    if config.get("coverage_sampling") is not None:
        from .coverage_sampling import CoverageSampler
        coverage_sampler = CoverageSampler(datasets["train"].sampling_rows(),
            config["coverage_sampling"], seed=config["seed"])
        round_rows = coverage_sampler.round_rows
    if config.get("targeted_sampling") is not None:
        from .targeted_subset import TargetedSampler
        view_manifest_path = Path(view) / "view_manifest.json"
        view_receipt = json.loads(view_manifest_path.read_text())
        train_bounds = view_receipt["source_manifest"]["splits"]["train"]
        targeted_sampler = TargetedSampler.from_assessment(
            config["targeted_sampling"], view_manifest_path=view_manifest_path,
            train_bounds=train_bounds, expected_rows=len(datasets["train"]))
        expected_actions = {row.get("target_name") for row in datasets["train"].sampling_rows()}
        if set(targeted_sampler.action_names) != expected_actions or None in expected_actions:
            raise ValueError("targeted assessment does not preserve every training action")
        round_rows = targeted_sampler.round_rows
    config = resolve_training_budget(config, train_rows=round_rows,
                                     valid_rows=len(datasets["valid"]))
    validate_early_stopping_coverage(config, datasets)
    validate_balanced_optimizer_windows(config, datasets["train"])
    destination.mkdir(parents=True)
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2))
    train_batches = (math.ceil(round_rows / config["batch_size"])
                     if config.get("include_partial_batch") else
                     round_rows // config["batch_size"])
    targeted_receipt = None
    if targeted_sampler is not None:
        selection_rounds = max(1, math.ceil(config["iters"] / train_batches))
        targeted_receipt = {
            "schema": "propevolve_targeted_sampling_receipt_v1",
            "assessment": dict(config["targeted_sampling"]),
            "mastered_anchor_retention": config.get("mastered_anchor_retention"),
            "pool_rows": targeted_sampler.pool_rows,
            "rounds": [targeted_sampler.selection_receipt(index)
                       for index in range(selection_rounds)],
        }
        (destination / "targeted_sampling_receipt.json").write_text(
            json.dumps(targeted_receipt, indent=2, allow_nan=False))
    event_log = TrainingEventLog(
        destination / config["training_log_filename"],
        events_path=destination / config["training_events_filename"],
        prefix=config["training_log_prefix"],
        train_batches=train_batches,
        total_iterations=config["iters"],
    )
    event_log.record_start({
        "train_rows": len(datasets["train"]),
        "training_rows_per_round": round_rows,
        "training_pool_rows": (len(datasets["train"]) if targeted_sampler is None
                               else targeted_sampler.pool_rows),
        "training_strata": (len(coverage_sampler.groups) if coverage_sampler is not None
                            else len(targeted_sampler.groups)
                            if targeted_sampler is not None else None),
        "targeted_sampling": targeted_sampler is not None,
        "mastered_anchor_retention": config.get("mastered_anchor_retention"),
        "targeted_sampling_summary": (None if targeted_receipt is None else {
            "rounds": len(targeted_receipt["rounds"]),
            "mistake_draws": sum(row["mistake_draws"]
                for row in targeted_receipt["rounds"]),
            "anchor_draws": sum(row["anchor_draws"]
                for row in targeted_receipt["rounds"]),
        }),
        "valid_rows": len(datasets["valid"]),
        "evaluation_every_iterations": config["steps_per_eval"],
        "evaluation_every_epochs": config["steps_per_eval"] / train_batches,
        "report_every_iterations": config["steps_per_report"],
        "report_every_epochs": config["steps_per_report"] / train_batches,
        "early_stopping": dict(config["early_stopping"]),
    })
    best = destination / ".best-validation"
    guard = ValidationLossGuard(config["early_stopping"],
        on_improvement=lambda report: (
            best.mkdir(exist_ok=True), export_policy_weights(model, best)))
    from .training_checkpoint import load_training_state, save_training_state
    from .integrity import file_digest
    import tempfile
    resume = config.get("resume_training_state")
    resume_iteration = 0
    # Paths/budget may change for continuation; all learning/data settings must agree.
    mutable = {"adapter_path", "resume_training_state", "epochs", "iters",
               "validation_metrics_path", "save_training_state"}
    identity = {"view_sha256": file_digest(Path(view) / "view_manifest.json"),
                "config": {key: value for key, value in config.items() if key not in mutable}}
    if resume is not None:
        receipt = json.loads((Path(resume) / "receipt.json").read_text())["receipt"]
        if receipt["identity"] != identity:
            raise ValueError("training resume data or learner configuration differs")
        resume_iteration = receipt["iteration"]
        if (resume_iteration % config["grad_accumulation_steps"]
                or resume_iteration % config["steps_per_report"]
                or not 0 <= resume_iteration < config["iters"]):
            raise ValueError("training resume must be an earlier completed optimizer boundary")
        load_training_state(resume, model, optimizer)
        for key, value in receipt["guard"].items():
            setattr(guard, key, value)
        if guard.stopped_early:
            raise ValueError("cannot resume a run already stopped for overfitting")
        if (Path(resume) / "best").exists():
            shutil.copytree(Path(resume) / "best", best)

    def snapshot(iteration):
        if not config.get("save_training_state"):
            return
        if not config["early_stopping"]["enabled"]:
            raise ValueError("resumable SFT requires the validation guard")
        stage = Path(tempfile.mkdtemp(prefix=".resume-", dir=destination))
        guard_state = {key: value for key, value in vars(guard).items()
                       if key not in {"on_improvement"}}
        save_training_state(stage / "state", model, optimizer,
                            {"iteration": iteration, "identity": identity, "guard": guard_state})
        if best.exists():
            shutil.copytree(best, stage / "state" / "best")
        latest = destination / "training-state"
        old = stage / "old"
        if latest.exists():
            latest.rename(old)
        (stage / "state").rename(latest)
        shutil.rmtree(stage)
    iterator = partial(tensor_batches, seed=config["seed"],
                       sampling_strategy=config.get("batch_sampling", "random"),
                       include_partial=config.get("include_partial_batch", False),
                       coverage_sampler=coverage_sampler,
                       targeted_sampler=targeted_sampler,
                       mastered_anchor_retention=config.get("mastered_anchor_retention"),
                       skip_batches=resume_iteration)
    def evaluate_loss():
        if config.get("market_distillation") is not None:
            from .market_distillation import evaluate_market_validation
            value = evaluate_market_validation(model, datasets["valid"], config)
        elif config["action_supervision"]["enabled"]:
            value = evaluate_action_validation(model, datasets["valid"], config)
        else:
            value = evaluate(model, datasets["valid"],
                batch_size=config["validation_batch_size"],
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
        total_iterations=config["iters"], train_batches=train_batches,
        evaluate_loss=evaluate_loss, record_training=event_log.record_training,
        record_validation_started=event_log.record_validation_started,
        record_validation=lambda report: (
            event_log.record_validation(report), record_validation(report)))
    args = TrainingArgs(batch_size=config["batch_size"], iters=config["iters"],
        val_batches=config["val_batches"], steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"], steps_per_save=config["save_every"],
        adapter_file=str(destination / "adapters.safetensors"), max_seq_length=config["max_seq_length"],
        grad_checkpoint=config["grad_checkpoint"], grad_accumulation_steps=config["grad_accumulation_steps"],
        clear_cache_threshold=config["clear_cache_threshold"])
    args.iters = config["iters"] - resume_iteration
    class ResumeCallback:
        def on_train_loss_report(self, report):
            report = {**report, "iteration": report["iteration"] + resume_iteration}
            try:
                validation.on_train_loss_report(report)
            finally:
                if (report["iteration"] % config["save_every"] == 0
                        or report["iteration"] == config["iters"]):
                    snapshot(report["iteration"])
        def on_val_loss_report(self, report):
            validation.on_val_loss_report(report)
    try:
        try:
            if not resume_iteration:
                if initial_validation is None:
                    validation.evaluate(0)
                else:
                    validation.reuse(0, initial_validation)
            train(model, optimizer, datasets["train"], None, args=args,
                loss=partial(batch_loss, config=config),
                iterate_batches=iterator, training_callback=ResumeCallback())
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
        summary = guard.summary()
        (destination / "training_selection.json").write_text(json.dumps(summary, indent=2))
        event_log.record_complete(summary)
        if best.exists():
            shutil.rmtree(best)
    except BaseException as error:
        event_log.record_failure(error)
        raise

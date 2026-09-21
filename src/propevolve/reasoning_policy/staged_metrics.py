"""Frozen evidence for independent entry, direction and management assessments."""
import numpy as np
from ..decision import Action
from .staged_policy import select_legal_action


def assessment_advantages(row):
    """Target-relative frozen evidence for correction and promotion consumers."""
    assessment = row.get("assessment")
    if (row.get("score_type") != "log_probability" or not isinstance(assessment, dict)
            or set(assessment) != {"entry", "direction", "management"}):
        raise ValueError("frozen staged evidence requires independent assessments")
    e, d, m = (assessment[name] for name in ("entry", "direction", "management"))
    if any(isinstance(x, bool) or not isinstance(x, (float, int))
           or not np.isfinite(x) for x in (e, d, m)):
        raise ValueError("invalid frozen staged assessment")
    target = row.get("target")
    if target == "WAIT":
        return {"entry.WAIT": -e}
    if target in {"ENTER_LONG_1", "ENTER_SHORT_1"}:
        return {"entry.ENTER": e,
            "direction.LONG" if target == "ENTER_LONG_1" else "direction.SHORT":
                d if target == "ENTER_LONG_1" else -d}
    if target in {"HOLD", "CLOSE"}:
        return {"management." + target: m if target == "HOLD" else -m}
    raise ValueError("unknown staged target")


def evaluate_staged_validation(backend, dataset, config, *, on_scored=None):
    """Read every fixed row once; never consume teacher targets or training anchors."""
    import mlx.core as mx
    from mlx.utils import tree_map
    from .staged_batches import pack_staged_examples
    from .staged_learning import LOSS_SEMANTICS, trade_objective
    from .staged_policy import staged_forward
    size = config["validation_batch_size"]
    if not len(dataset) or type(size) is not int or size < 1:
        raise ValueError("staged validation requires rows and a positive batch size")
    rows_seen, scores_seen, total_loss = [], [], 0.
    for start in range(0, len(dataset), size):
        rows = [dataset[i] for i in range(start, min(start + size, len(dataset)))]
        packed = pack_staged_examples(rows, max_seq_length=config["max_seq_length"],
                                      include_teachers=False)
        batch = tree_map(lambda x: mx.array(x) if isinstance(x, np.ndarray) else x, packed)
        result = staged_forward(backend, **batch["inputs"])
        targets = batch["targets"]
        loss = trade_objective(result["assessment_scores"], targets["probabilities"],
            targets["values"], targets["boundary_weights"], config["action_supervision"], xp=mx)
        mx.eval(result, loss)
        total_loss += float(loss) * len(rows)
        scores = result["assessment_scores"].tolist()
        rows_seen.extend(rows)
        scores_seen.extend(scores)
        if on_scored is not None:
            for offset, assessment in enumerate(scores):
                on_scored(start + offset, {"assessment": assessment,
                    "interpretation": mx.sigmoid(result["interpretation_scores"][offset]).tolist(),
                    "log_probs": result["log_probs"][offset].tolist()})
    return {"val_loss": total_loss / len(rows_seen), "loss_semantics": LOSS_SEMANTICS,
        **boundary_metrics(rows_seen, scores_seen, margin=config["action_supervision"]["margin"])}


def boundary_metrics(rows, score_rows, *, margin=0.):
    if (not rows or len(rows) != len(score_rows) or not np.isfinite(margin)
            or margin < 0):
        raise ValueError("staged metrics require aligned nonempty rows")
    tasks, actions = {}, {}
    for row, raw in zip(rows, score_rows):
        scores = np.asarray(raw, float)
        if scores.shape != (3,) or not np.isfinite(scores).all():
            raise ValueError("staged metrics require three finite binary assessments")
        e, d, m = scores
        target = row["target_name"]
        names = row["action_targets"]["names"]
        if target not in names:
            raise ValueError("staged target must be legal")
        predicted = select_legal_action(scores, [Action[name] for name in names]).name
        if names == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]:
            if target == "WAIT":
                evidence = [("entry.WAIT", -e, e <= 0)]
            else:
                long = target == "ENTER_LONG_1"
                evidence = [("entry.ENTER", e, e > 0),
                    ("direction.LONG" if long else "direction.SHORT",
                     d if long else -d, d > 0 if long else d < 0)]
        elif names == ["HOLD", "CLOSE"]:
            hold = target == "HOLD"
            evidence = [("management." + target, m if hold else -m,
                         m > 0 if hold else m <= 0)]
        else:
            raise ValueError("invalid staged legal state")
        for task, advantage, correct in evidence:
            tasks.setdefault(task, []).append((float(advantage), bool(correct)))
        actions.setdefault(target, []).append((min(x[1] for x in evidence), predicted == target))

    def summarize(groups):
        return {name: {"count": len(items),
            "mean_target_advantage": float(np.mean([a for a, _ in items])),
            "mean_boundary_loss": float(np.mean([np.logaddexp(0., margin - a) for a, _ in items])),
            "accuracy": float(np.mean([c for _, c in items]))}
            for name, items in sorted(groups.items())}
    per_task, per_action = summarize(tasks), summarize(actions)
    return {"decision_boundary_semantics": "staged_independent_binary_v1",
        "per_task": per_task, "per_action": per_action,
        "worst_task_advantage": min(r["mean_target_advantage"] for r in per_task.values()),
        "worst_action_advantage": min(r["mean_target_advantage"] for r in per_action.values()),
        "worst_task_boundary_loss": max(r["mean_boundary_loss"] for r in per_task.values()),
        "worst_action_boundary_loss": max(r["mean_boundary_loss"] for r in per_action.values()),
        "task_macro_accuracy": float(np.mean([r["accuracy"] for r in per_task.values()])),
        "worst_task_accuracy": min(r["accuracy"] for r in per_task.values()),
        "worst_action_accuracy": min(r["accuracy"] for r in per_action.values()),
        "macro_accuracy": float(np.mean([r["accuracy"] for r in per_action.values()]))}

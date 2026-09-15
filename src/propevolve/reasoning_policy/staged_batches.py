"""Pack staged causal inputs separately from authenticated teaching targets."""
import numpy as np

from ..decision import Action


def binary_targets(target):
    """Preserve economic soft labels, with direction applicable only to entries.

    Output order is WAIT/ENTER, SHORT/LONG, CLOSE/HOLD. Executable action
    probabilities are supervision, not predictions or inference inputs.
    """
    names = target["names"]
    p, v = np.asarray(target["probabilities"], float), np.asarray(target["values"], float)
    if (p.shape != (len(names),) or v.shape != p.shape or not np.isfinite([p, v]).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1.)):
        raise ValueError("invalid staged economic targets")
    probabilities = np.full((3, 2), .5, np.float32)
    values = np.zeros((3, 2), np.float32)
    weights = np.zeros(3, np.float32)
    if names == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]:
        enter = p[1] + p[2]
        # ENTER competes on the best available economic action, not on the
        # number of executable sides. Summing two losing-side masses can teach
        # ENTER even when WAIT has the highest utility. Pair normalization
        # preserves the source softmax temperature without that multiplicity.
        best_side = 1 + int(np.argmax(v[1:]))
        entry_mass = p[0] + p[best_side]
        if entry_mass <= 0:
            raise ValueError("entry boundary has no probability mass")
        probabilities[0] = [p[0] / entry_mass, p[best_side] / entry_mass]
        values[0] = [v[0], max(v[1], v[2])]
        weights[0] = 1.
        if max(v[1], v[2]) > v[0] and v[1] != v[2]:
            if enter <= 0:
                raise ValueError("entry winner has no direction probability mass")
            probabilities[1] = [p[2] / enter, p[1] / enter]
            values[1] = [v[2], v[1]]
            weights[1] = 1.
    elif names == ["HOLD", "CLOSE"]:
        probabilities[2], values[2], weights[2] = p[::-1], v[::-1], 1.
    else:
        raise ValueError("unsupported staged legal-action targets")
    return probabilities, values, weights


def pack_staged_examples(rows, *, max_seq_length, include_teachers=True):
    if not rows or not all("staged_queries" in row for row in rows):
        raise ValueError("cannot mix staged and legacy training rows")
    queries = [row["staged_queries"] for row in rows]
    if any(q["channel_names"] != queries[0]["channel_names"] for q in queries):
        raise ValueError("staged interpretation channel order differs across rows")
    inputs = {}
    for kind in ("market_query", "assessment_query"):
        groups = [q[kind] for q in queries]
        width = max(np.asarray(q["tokens"]).shape[-1] for q in groups)
        if width > max_seq_length:
            raise ValueError("staged queries exceed token budget; truncation forbidden")
        tokens = []
        for q in groups:
            array = np.asarray(q["tokens"], np.int32)
            padding = [(0, 0)] * array.ndim
            padding[-1] = (0, width - array.shape[-1])
            tokens.append(np.pad(array, padding))
        inputs[kind] = {key: np.concatenate(
            tokens if key == "tokens" else [np.asarray(q[key], np.int32) for q in groups], axis=0)
            for key in groups[0]}
    inputs["embeddings"] = np.asarray([r["market_embeddings"] for r in rows], np.float32)
    inputs["available"] = np.asarray([r["market_available"] for r in rows], bool)
    if (inputs["embeddings"].ndim != 3
            or inputs["available"].shape != inputs["embeddings"].shape[:2]
            or not inputs["available"].any(axis=1).all()
            or not np.isfinite(inputs["embeddings"]).all()):
        raise ValueError("invalid staged causal embedding batch")
    inputs["legal_actions"] = [tuple(Action[name] for name in r["action_targets"]["names"])
                               for r in rows]
    targets = {key: np.asarray(values, np.float32) for key, values in zip(
        ("probabilities", "values", "boundary_weights"),
        zip(*(binary_targets(r["action_targets"]) for r in rows)))}
    if not include_teachers:
        return {"inputs": inputs, "targets": targets}
    for key in ("teacher_probabilities", "teacher_weights"):
        targets[key] = np.asarray([r[key] for r in rows], np.float32)
    p, w = targets["teacher_probabilities"], targets["teacher_weights"]
    if (p.shape != (len(rows), len(queries[0]["channel_names"])) or w.shape != p.shape
            or not np.isfinite([p, w]).all() or (p < 0).any() or (p > 1).any()
            or (w < 0).any() or not (w.sum(axis=1) > 0).all()):
        raise ValueError("invalid staged teacher targets")
    retention = [r.get("mastered_anchor_retention") for r in rows]
    if any(item is not None for item in retention):
        boundary_names = ("entry", "direction", "management")
        if not all(isinstance(item, dict) and set(item) == {"assessment", "boundaries"}
                   and set(item["boundaries"]) == set(boundary_names)
                   and all(type(item["boundaries"][name]) is bool for name in boundary_names)
                   for item in retention):
            raise ValueError("staged retention requires independent frozen evidence on every row")
        parent = np.asarray([item["assessment"] for item in retention], np.float32)
        masks = np.asarray([[item["boundaries"][name] for name in boundary_names]
                            for item in retention], bool)
        if parent.shape != (len(rows), 3) or not np.isfinite(parent).all():
            raise ValueError("invalid frozen staged assessment")
        if np.any(masks & (targets["boundary_weights"] == 0)):
            raise ValueError("cannot retain an inapplicable trade boundary")
        targets.update(parent_assessment=parent, retention_weights=masks)
    return {"inputs": inputs, "targets": targets}

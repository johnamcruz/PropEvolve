"""Translate authenticated action rows without serializing their answers as inputs."""
import json
import math

from .projector import projector_prefix_tokens
from .staged_queries import prepare_staged_queries
from .supervision import action_targets


def causal_record_state(record, fields_required):
    """Read only completed trade state, never targets or teacher probabilities."""
    prompt = json.loads(record["messages"][-2]["content"])
    fields = record.get("causal_state_fields", prompt["fields"])
    values = (record["causal_state"] if "causal_state_fields" in record
              else prompt["history_oldest_first"][-1])
    if (len(fields) != len(values) or len(set(fields)) != len(fields)
            or any(name not in fields for name in fields_required)):
        raise ValueError("staged row causal state differs from configured fields")
    return {name: values[fields.index(name)] for name in fields_required}


def record_context(record, fields):
    """Reconstruct the causal inference boundary from an authenticated raw row."""
    import numpy as np
    from .context import ContextWindow
    state = causal_record_state(record, fields)
    embeddings = np.asarray(record["market_embeddings"], dtype=np.float32)
    available = np.asarray(record["market_available"], dtype=bool)
    if (embeddings.ndim != 2 or available.shape != (len(embeddings),)
            or not available.any() or not np.isfinite(embeddings).all()):
        raise ValueError("invalid frozen causal embedding context")
    # The assessment consumes the latest completed trade state. The market
    # interpretation consumes the full embedding history, without trade state.
    values = np.zeros((len(embeddings), len(fields)), dtype=np.float32)
    values[np.flatnonzero(available)[-1]] = list(state.values())
    return ContextWindow(values, available, (), tuple(fields), embeddings)


def encode_staged_record(record, config, tokenizer):
    settings = config["staged_policy"]
    state = causal_record_state(record, settings["state_fields"])
    queries = prepare_staged_queries(state, settings, tokenizer,
        max_seq_length=config["max_seq_length"] - projector_prefix_tokens(config["projector"]),
        chat_template_kwargs=config["chat_template_kwargs"])
    teachers = record["targets"]["specialist_targets"]
    if set(teachers) != set(queries["channel_names"]):
        raise ValueError("teacher channels differ from staged interpretation contract")
    probabilities = [teachers[name] for name in queries["channel_names"]]
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid staged teacher probability")
    return {"staged_queries": queries, "action_targets": action_targets(record),
        "target_name": record["messages"][-1]["content"],
        "teacher_probabilities": probabilities,
        "teacher_weights": [c["weight"] for c in settings["market"]["channels"]]}

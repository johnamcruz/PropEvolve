"""Config-owned, target-free queries shared by reasoning training and inference."""
from collections.abc import Mapping
import json
import math

from .market_distillation import validate_market_distillation


def prepare_staged_queries(state, settings, tokenizer, *, max_seq_length, chat_template_kwargs):
    """Serialize only declared causal state; model targets are not arguments."""
    validate_market_distillation(settings["market"])
    if (not isinstance(state, Mapping) or set(state) != set(settings["state_fields"])
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in state.values())):
        raise ValueError("causal state must contain exactly the declared finite numeric fields")
    if type(max_seq_length) is not int or max_seq_length < 1:
        raise ValueError("query token budget must be positive")
    if [task["name"] for task in settings["tasks"]] != ["entry", "direction", "management"]:
        raise ValueError("staged tasks must be entry, direction, management in that order")
    market = settings["market"]

    def prompt(instruction):
        result = tokenizer.apply_chat_template(
            [{"role": "system", "content": instruction}], tokenize=True,
            add_generation_prompt=True, **chat_template_kwargs)
        return list(result["input_ids"] if isinstance(result, Mapping) else result)

    def label_pair(source):
        pair = [tokenizer.encode(source[key], add_special_tokens=False)
                for key in ("negative_token", "positive_token")]
        if any(len(ids) != 1 for ids in pair) or pair[0] == pair[1]:
            raise ValueError("binary labels require distinct single tokenizer tokens")
        return [ids[0] for ids in pair]

    market_tokens = prompt(market["instruction"])
    market_positions = []
    for channel in market["channels"]:
        market_tokens.extend(tokenizer.encode(channel["query"], add_special_tokens=False))
        market_positions.append(len(market_tokens) - 1)
    market_tokens.append(tokenizer.eos_token_id)
    interpretation_ids = label_pair(market)
    assessment = prompt(settings["assessment_instruction"])
    assessment.extend(tokenizer.encode(json.dumps(state, sort_keys=True, allow_nan=False),
                                       add_special_tokens=False))
    interpretation_positions = []
    for channel in market["channels"]:
        assessment.extend(tokenizer.encode(channel["query"], add_special_tokens=False))
        interpretation_positions.append(len(assessment))
        assessment.append(interpretation_ids[0])
    task_positions, task_ids = [], []
    for task in settings["tasks"]:
        assessment.extend(tokenizer.encode(task["query"], add_special_tokens=False))
        task_positions.append(len(assessment) - 1)
        task_ids.append(label_pair(task))
    if max(len(market_tokens), len(assessment)) > max_seq_length:
        raise ValueError("staged queries exceed token budget; truncation forbidden")
    return {"channel_names": [c["name"] for c in market["channels"]],
        "market_query": {"tokens": [[market_tokens]], "positions": [market_positions],
                         "label_ids": [interpretation_ids]},
        "assessment_query": {"tokens": [assessment],
            "interpretation_positions": [interpretation_positions],
            "interpretation_label_ids": [interpretation_ids],
            "task_positions": [task_positions], "task_label_ids": [task_ids]}}

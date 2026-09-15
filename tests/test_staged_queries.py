"""JSON settings and causal state -> shared staged inference queries."""
from collections import UserDict

import pytest


class Tokenizer:
    eos_token_id = 1

    def encode(self, value, **kwargs):
        return [ord(c) + 2 for c in value]

    def apply_chat_template(self, messages, **kwargs):
        return UserDict(input_ids=self.encode("\n".join(m["content"] for m in messages)))


def settings():
    return {"market": {"instruction": "Interpret completed market context.",
        "negative_token": "N", "positive_token": "Y", "channels": [
            {"name": "expansion.long", "query": "Long expansion?", "weight": 1.},
            {"name": "expansion.short", "query": "Short expansion?", "weight": 1.}]},
        "assessment_instruction": "Assess trade using predicted interpretation.",
        "state_fields": ["trade.unrealized_r"],
        "tasks": [
            {"name": "entry", "query": "Enter?", "negative_token": "W", "positive_token": "E"},
            {"name": "direction", "query": "Long?", "negative_token": "S", "positive_token": "L"},
            {"name": "management", "query": "Hold?", "negative_token": "C", "positive_token": "H"}]}


def test_state_only_changes_assessment_not_market_interpretation_prompt():
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    first = prepare_staged_queries({"trade.unrealized_r": 1.}, settings(), Tokenizer(),
        max_seq_length=1024, chat_template_kwargs={})
    second = prepare_staged_queries({"trade.unrealized_r": -1.}, settings(), Tokenizer(),
        max_seq_length=1024, chat_template_kwargs={})
    assert first["market_query"] == second["market_query"]
    assert first["assessment_query"]["tokens"] != second["assessment_query"]["tokens"]
    assert first["assessment_query"]["task_label_ids"] == [[[89, 71], [85, 78], [69, 74]]]
    assert first["channel_names"] == ["expansion.long", "expansion.short"]


@pytest.mark.parametrize("state", [
    {"trade.unrealized_r": 1., "teacher_targets": .9},
    {"trade.unrealized_r": float("nan")}, {},
])
def test_query_builder_rejects_undeclared_or_invalid_state(state):
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    with pytest.raises(ValueError, match="state"):
        prepare_staged_queries(state, settings(), Tokenizer(), max_seq_length=1024,
                              chat_template_kwargs={})


def test_query_budget_is_enforced_without_silent_truncation():
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    with pytest.raises(ValueError, match="budget"):
        prepare_staged_queries({"trade.unrealized_r": 1.}, settings(), Tokenizer(),
                              max_seq_length=4, chat_template_kwargs={})


@pytest.mark.parametrize("change", ["task_order", "multi_token", "duplicate_labels"])
def test_query_contract_rejects_ambiguous_decision_encoding(change):
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    config = settings()
    if change == "task_order":
        config["tasks"].reverse()
    elif change == "multi_token":
        config["tasks"][0]["positive_token"] = "ENTER"
    else:
        config["market"]["positive_token"] = "N"
    with pytest.raises(ValueError):
        prepare_staged_queries({"trade.unrealized_r": 1.}, config, Tokenizer(),
                              max_seq_length=1024, chat_template_kwargs={})

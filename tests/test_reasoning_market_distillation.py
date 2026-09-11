"""Public short-query dataset and learner contract, independent of JSON spelling."""
import copy
from collections import UserDict

import pytest


class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]
    def encode(self, text, **kwargs):
        return {"Y": [7], "N": [8], "Long?": [4], "Short?": [5]}[text]


def settings():
    return {"instruction": "Estimate market probabilities from completed context.",
        "negative_token": "N", "positive_token": "Y", "channels": [
        {"name": "expansion.long", "query": "Long?", "weight": 1.0},
        {"name": "expansion.short", "query": "Short?", "weight": 1.0}]}


def test_teacher_values_change_targets_never_causal_query_tokens():
    from propevolve.reasoning_policy.market_distillation import encode_market_targets
    record = {"messages": [{"role": "system", "content": "market"},
        {"role": "user", "content": "causal"},
        {"role": "assistant", "content": "future answer"}],
        "targets": {"specialist_targets": {"expansion.long": .73, "expansion.short": .82}}}
    before = encode_market_targets(record, settings(), Tokenizer(), max_seq_length=20,
                                   chat_template_kwargs={})
    changed = copy.deepcopy(record)
    changed["targets"]["specialist_targets"]["expansion.long"] = .11
    changed["messages"][-1]["content"] = "different future answer"
    after = encode_market_targets(changed, settings(), Tokenizer(), max_seq_length=20,
                                  chat_template_kwargs={})
    assert before["tokens"] == after["tokens"] == [1, 2, 3, 4, 5, 9]
    assert before["market_targets"]["probabilities"] == [.73, .82]
    assert after["market_targets"]["probabilities"] == [.11, .82]
    assert before["market_targets"]["positions"] == [3, 4]
    assert before["market_targets"]["label_ids"] == [8, 7]


@pytest.mark.parametrize("bad", [float("nan"), -.1, 1.1, True, None])
def test_invalid_teacher_probability_is_rejected(bad):
    from propevolve.reasoning_policy.market_distillation import encode_market_targets
    record = {"messages": [{}, {}, {}], "targets": {"specialist_targets": {
        "expansion.long": bad, "expansion.short": .2}}}
    with pytest.raises(ValueError, match="probability"):
        encode_market_targets(record, settings(), Tokenizer(), max_seq_length=20,
                              chat_template_kwargs={})


def test_packed_soft_targets_train_both_independent_directions():
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.market_distillation import encode_market_targets, probability_loss
    from propevolve.reasoning_policy.supervised_trainer import pack_examples
    record = {"messages": [{}, {}, {}], "targets": {"specialist_targets": {
        "expansion.long": .73, "expansion.short": .82}}}
    row = encode_market_targets(record, settings(), Tokenizer(), max_seq_length=20,
                                chat_template_kwargs={})
    packed = pack_examples([row], max_seq_length=20)
    assert len(packed) == 14
    assert packed[10].tolist() == [[3, 4]]
    assert packed[13].tolist() == [[8, 7]]
    loss, grad = mx.value_and_grad(lambda scores: probability_loss(
        scores, mx.array([[.73, .82]]), mx.array([[1., 1.]])))(mx.zeros((1, 2)))
    assert float(loss) == pytest.approx(.69314718)
    assert grad.tolist()[0] == pytest.approx([-.115, -.16])


def test_coverage_rounds_keep_rare_strata_and_rotate_full_training_pool():
    from propevolve.reasoning_policy.coverage_sampling import CoverageSampler
    rows = [{"coverage": {"ticker": "NQ", "year": "2023", "strength": .8}} for _ in range(4)]
    rows += [{"coverage": {"ticker": "ES", "year": "2022", "strength": .1}}]
    config = {"fields": ["ticker", "year"], "probability_bins": {"strength": [.25, .75]},
              "rows_per_stratum": 1}
    sampler = CoverageSampler(rows, config, seed=17)
    rounds = [sampler.order(i).tolist() for i in range(4)]
    assert all(len(part) == 2 and 4 in part for part in rounds)
    assert set(sum(rounds, [])) == set(range(5))
    assert CoverageSampler(rows, config, seed=17).order(2).tolist() == rounds[2]
    assert sampler.round_rows == 2


def test_probability_stage_rejects_omitted_teacher_channel():
    from propevolve.reasoning_policy.market_distillation import encode_market_targets
    record = {"messages": [{}, {}, {}], "targets": {"specialist_targets": {
        "expansion.long": .73, "expansion.short": .82, "trend.long": .9}}}
    with pytest.raises(ValueError, match="channel"):
        encode_market_targets(record, settings(), Tokenizer(), max_seq_length=20,
                              chat_template_kwargs={})


def test_fixed_diagnostic_selection_covers_ticker_year_without_source_changes(tmp_path):
    import json
    from propevolve.reasoning_policy.coverage_sampling import select_prepared_rows
    path = tmp_path / "rows.jsonl"
    rows = [{"ticker": ticker, "completed_at_ns": 1640995200000000000,
             "targets": {"specialist_targets": {}}} for ticker in ["NQ"] * 5 + ["ES"] * 3]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = path.read_bytes()
    config = {"fields": ["ticker", "year"], "rows_per_stratum": {"train": 2, "valid": 1}, "seed": 11}
    selected = select_prepared_rows(path, config, role="train")
    assert len(selected) == 4
    assert sum(rows[i]["ticker"] == "NQ" for i in selected) == 2
    assert sum(rows[i]["ticker"] == "ES" for i in selected) == 2
    assert path.read_bytes() == before


def test_chat_template_mapping_preserves_actual_prompt_token_ids():
    from propevolve.reasoning_policy.market_distillation import encode_market_targets

    class TokenizerMapping(UserDict):
        """Match the Mapping contract returned by production tokenizers."""

    class MappingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return TokenizerMapping({
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            })
    record = {"messages": [{}, {}, {}], "targets": {"specialist_targets": {
        "expansion.long": .73, "expansion.short": .82}}}
    result = encode_market_targets(record, settings(), MappingTokenizer(), max_seq_length=20,
                                   chat_template_kwargs={})
    assert result["tokens"] == [1, 2, 3, 4, 5, 9]

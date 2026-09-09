"""Approved JSON -> dataset -> policy seam; execution deferred during volume work.

The tokenizer is an external boundary fixture. Actual tokenizer/model parity is
also required by the opt-in real MLX tests before training is accepted.
"""

import json

import pytest

from propevolve.reasoning_policy.dataset import write_supervised_dataset
from propevolve.reasoning_policy.integrity import file_digest
from propevolve.reasoning_policy.mlx_sft import (
    EncodedDataset, prepare_mlx_view, verify_mlx_view,
)
from propevolve.reasoning_policy.policy import MLXActionPolicy


class LiteralTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text):
        return [ord(character) for character in text]


class NoLoadModel:
    def eval(self):
        pass


ACTION_VERBALIZERS = {
    "WAIT": "A",
    "ENTER_LONG_1": "B",
    "ENTER_SHORT_1": "C",
    "HOLD": "D",
    "CLOSE": "E",
}


@pytest.mark.parametrize("action", ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE"])
def test_prepared_supervision_matches_policy_tokens_and_completion_boundary(tmp_path, action):
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "state"}]
    records = [{"source_id": str(start), "completed_at_ns": start, "label_end_ns": start + 1,
                "messages": messages + [{"role": "assistant", "content": action}]}
               for start in (10, 110)]
    dataset = tmp_path / "source"
    write_supervised_dataset(records, dataset,
        splits={"train": [0, 100], "valid": [100, 200]}, sealed_start_ns=200,
        lineage={"source_identity": "test", "specialist_identities": "test",
                 "economic_contract": "test", "split_audit": "test"})
    (dataset / "audit.json").write_text(json.dumps({
        "status": "PASS", "manifest_sha256": file_digest(dataset / "manifest.json"),
        "specialist_score_mode": "out_of_fold", "sealed_touched": False,
    }))
    config = tmp_path / "arbitrary-name.json"
    config.write_text(json.dumps({
        "model": "external-tokenizer-fixture", "data": str(dataset),
        "adapter_path": str(tmp_path / "adapter"), "train": True,
        "fine_tune_type": "lora", "mask_prompt": True, "num_layers": 1,
        "batch_size": 1, "iters": 1, "learning_rate": 1e-5,
        "max_seq_length": 1024, "grad_checkpoint": False,
        "grad_accumulation_steps": 1, "lora_parameters": {"rank": 2, "scale": 4., "dropout": 0.},
        "trust_remote_code": False,
        "action_verbalizers": ACTION_VERBALIZERS,
    }))
    tokenizer = LiteralTokenizer()
    view = tmp_path / "view"
    prepared = prepare_mlx_view(config, view, tokenizer=tokenizer)
    native = EncodedDataset(view / "train.jsonl")
    tokens, offset = native.process(native[0])
    # Independent literal expectation detects double chat wrapping and target leakage.
    assert "".join(map(chr, tokens[:offset])) == "<system>rules<user>state<assistant>"
    assert "".join(map(chr, tokens[offset:])) == ACTION_VERBALIZERS[action] + "!"
    policy = MLXActionPolicy(NoLoadModel(), tokenizer, max_seq_length=1024,
                             action_verbalizers=ACTION_VERBALIZERS)
    assert policy.tokenize_completions(messages, [action]) == ((tuple(tokens), offset),)
    assert verify_mlx_view(config, view) == prepared
    with (view / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="prepared view changed"):
        verify_mlx_view(config, view)


def test_policy_rejects_truncation_instead_of_dropping_action_tokens():
    policy = MLXActionPolicy(NoLoadModel(), LiteralTokenizer(), max_seq_length=2)
    with pytest.raises(ValueError, match="token budget"):
        policy.tokenize_completions([{"role": "user", "content": "state"}], ["WAIT"])


def test_policy_rejects_missing_or_ambiguous_action_verbalizers():
    with pytest.raises(ValueError, match="every action"):
        MLXActionPolicy(NoLoadModel(), LiteralTokenizer(), max_seq_length=1024,
                        action_verbalizers={"WAIT": "WAIT"})
    ambiguous = dict(ACTION_VERBALIZERS, ENTER_SHORT_1="B")
    with pytest.raises(ValueError, match="unique"):
        MLXActionPolicy(NoLoadModel(), LiteralTokenizer(), max_seq_length=1024,
                        action_verbalizers=ambiguous)

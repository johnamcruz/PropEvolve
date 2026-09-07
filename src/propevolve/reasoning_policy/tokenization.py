"""One exact chat/completion boundary for SFT, RL and frozen inference."""

from .model_config import template_options


def encode_completion(tokenizer, messages, completion, *, max_seq_length, chat_template_kwargs):
    if not isinstance(completion, str) or not completion:
        raise ValueError("completion must be nonempty text")
    if not isinstance(tokenizer.eos_token, str) or not tokenizer.eos_token:
        raise ValueError("tokenizer requires an explicit EOS token")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **template_options(chat_template_kwargs))
    prefix = tokenizer.encode(prompt)
    tokens = tokenizer.encode(prompt + completion + tokenizer.eos_token)
    if not prefix or tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
        raise ValueError("tokenizer changes the prompt/action boundary")
    if len(tokens) > max_seq_length:
        raise ValueError("token budget exceeded; refusing truncation")
    return tuple(tokens), len(prefix)

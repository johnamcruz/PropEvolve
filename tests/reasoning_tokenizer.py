"""External tokenizer fixture shared by data-integrity and tokenization tests."""


class LiteralTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text):
        return [ord(character) for character in text]


ACTION_VERBALIZERS = {
    "WAIT": "A",
    "ENTER_LONG_1": "B",
    "ENTER_SHORT_1": "C",
    "HOLD": "D",
    "CLOSE": "E",
}

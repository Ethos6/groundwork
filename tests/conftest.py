import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeModel:
    """Chat model driven by a function: fn(messages, call_index, **kwargs) -> str or list[str]."""

    def __init__(self, fn, name="fake"):
        self.fn = fn
        self.name = name
        self.calls = []

    def chat(self, messages, *, n=1, temperature=0.0, max_tokens=None, stop=None, prefill=None):
        self.calls.append({"messages": messages, "n": n, "temperature": temperature, "prefill": prefill})
        out = self.fn(messages, len(self.calls) - 1, n=n, temperature=temperature, prefill=prefill)
        outs = out if isinstance(out, list) else [out] * n
        return outs[:n] if len(outs) >= n else outs + [outs[-1]] * (n - len(outs))


@pytest.fixture
def fake_model():
    return FakeModel


@pytest.fixture
def root():
    return ROOT


def make_tiny_tokenizer(tmp_path):
    """A small word-level fast tokenizer with a ChatML-style template (no downloads)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "system", "user", "assistant", "\n"]
    words += "the a is what of to and in answer hello hi world 42 paris capital france yes no ok".split()
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split("\n", "isolated"), pre_tokenizers.WhitespaceSplit()])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    fast.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} \n {{ m['content'] }} <|im_end|> \n {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant \n {% endif %}"
    )
    path = tmp_path / "tok"
    fast.save_pretrained(path)
    return fast, path

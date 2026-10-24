import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import make_tiny_tokenizer  # noqa: E402
from groundwork.config import load_config  # noqa: E402
from groundwork.train.encode import IGNORE, ChatEncoder, collate  # noqa: E402

CONVO = [
    {"role": "system", "content": "hi"},
    {"role": "user", "content": "what is the capital of france"},
    {"role": "assistant", "content": "paris"},
    {"role": "tool", "content": "ok"},
    {"role": "assistant", "content": "the answer is paris"},
]


def _decoded_labels(tok, enc):
    return tok.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != IGNORE])


def test_only_assistant_tokens_are_trained(tmp_path):
    tok, _ = make_tiny_tokenizer(tmp_path)
    enc = ChatEncoder(tok, max_length=512).encode(CONVO)
    trained = _decoded_labels(tok, enc)
    assert trained.split() == ["paris", "<|im_end|>", "the", "answer", "is", "paris", "<|im_end|>"]
    # the tool observation and the user turn are context only
    assert "what" not in trained and "ok" not in trained


def test_fallback_when_template_is_not_prefix_consistent(tmp_path):
    tok, _ = make_tiny_tokenizer(tmp_path)
    # a template that changes earlier turns depending on what follows (like thinking-stripping templates)
    tok.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} \n {{ m['content'] }} <|im_end|> \n {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant \n {% else %} ok {% endif %}"
    )
    encoder = ChatEncoder(tok, max_length=512)
    enc = encoder.encode(CONVO)
    assert encoder.stats["span_fallbacks"] >= 1
    assert _decoded_labels(tok, enc).split()[:2] == ["paris", "<|im_end|>"]


def test_truncation_and_collate(tmp_path):
    tok, _ = make_tiny_tokenizer(tmp_path)
    enc = ChatEncoder(tok, max_length=8).encode(CONVO)
    assert enc is None or len(enc["input_ids"]) <= 8
    full = ChatEncoder(tok, max_length=512)
    batch = collate([full.encode(CONVO), full.encode(CONVO[:3])], tok.pad_token_id)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert int(batch["attention_mask"][1].sum()) < batch["input_ids"].shape[1]


def test_sft_smoke_run_reduces_loss(tmp_path):
    from groundwork.train.sft import train_sft

    tok, tok_path = make_tiny_tokenizer(tmp_path)
    cfg_model = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model_dir = tmp_path / "base"
    transformers.LlamaForCausalLM(cfg_model).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    data = tmp_path / "sft.jsonl"
    rows = [{"messages": CONVO, "source": "trace"}, {"messages": CONVO[:3], "source": "closed_book_known"}] * 8
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out"
    cfg = load_config(None, [
        f"train.base_model={model_dir}", f"train.data={data}", f"train.output_dir={out}", "train.dtype=float32",
        "train.lr=5e-3", "train.epochs=6", "train.grad_accum=2", "train.batch_size=2", "train.log_every_steps=1",
        "train.gradient_checkpointing=false", "train.warmup_ratio=0",
    ])
    train_sft(cfg)
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < log[0]["loss"] * 0.7
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert reloaded.config.hidden_size == 32


class _FakeVLProcessor:
    """Mimics a vision-language processor: renders an image placeholder, then expands it
    into several image tokens (as real processors do), which breaks character offsets."""

    chat_template = "fake"

    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
        out = ""
        for m in msgs:
            body = " ".join("IMG" if c["type"] == "image" else c["text"] for c in m["content"])
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, images=None, videos=None, return_tensors="pt"):
        expanded = text[0].replace("IMG", " ".join(["yes"] * 5))
        ids = self.tokenizer(expanded, add_special_tokens=False)["input_ids"]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids)),
                "pixel_values": torch.zeros(len(images or []), 3, 4, 4)}


def test_multimodal_encoding_masks_image_and_user_tokens(tmp_path, root):
    tok, _ = make_tiny_tokenizer(tmp_path)
    encoder = ChatEncoder(tok, _FakeVLProcessor(tok), multimodal=True, media_root=str(root))
    msgs = [
        {"role": "user", "content": [{"type": "image", "path": "examples/files/bar_chart.png"}, {"type": "text", "text": "what is the capital"}]},
        {"role": "assistant", "content": "paris"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "the answer is paris"},
    ]
    enc = encoder.encode(msgs)
    trained = [tok.convert_ids_to_tokens(t) for t, l in zip(enc["input_ids"], enc["labels"]) if l != IGNORE]
    assert trained == ["paris", "<|im_end|>", "the", "answer", "is", "paris", "<|im_end|>"]
    assert enc["extras"]["pixel_values"].shape[0] == 1
    batch = collate([enc, enc], tok.pad_token_id)
    assert batch["pixel_values"].shape[0] == 2

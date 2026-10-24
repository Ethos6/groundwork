"""Local Hugging Face Transformers models (text-only or vision-language)."""

from __future__ import annotations

import importlib
from pathlib import Path

from .. import media
from ..messages import parts, tool_to_user
from .base import trim_at_stop

# Used when a tokenizer ships without a chat template.
SIMPLE_ROLE_TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
SIMPLE_END = "<|end|>"


def render_simple(messages: list[dict], add_generation_prompt: bool) -> str:
    from ..messages import text_of

    out = []
    for m in messages:
        tag = SIMPLE_ROLE_TAGS.get(m["role"], "<|user|>")
        body = m["content"] if isinstance(m.get("content"), str) else text_of(m)
        out.append(f"{tag}\n{body}{SIMPLE_END}\n")
    if add_generation_prompt:
        out.append(SIMPLE_ROLE_TAGS["assistant"] + "\n")
    return "".join(out)


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}[name]


def load_hf(path: str, dtype: str = "bfloat16", device: str = "auto", trust_remote_code: bool = False,
            multimodal: bool = False, loader: str = ""):
    """Return ``(model, tokenizer, processor)``. ``processor`` is None for text-only models.

    ``loader`` may point to a custom function ``"pkg.module:fn"`` with the same signature,
    for models that don't load through ``transformers`` Auto classes.
    """
    if loader:
        mod_name, fn_name = loader.split(":", 1)
        fn = getattr(importlib.import_module(mod_name), fn_name)
        return fn(path=path, dtype=dtype, device=device, multimodal=multimodal)

    import transformers

    kwargs = {"trust_remote_code": trust_remote_code, "dtype": _torch_dtype(dtype)}
    if device == "auto":
        try:
            import accelerate  # noqa: F401

            kwargs["device_map"] = "auto"
        except ImportError:
            pass
    def _from_pretrained(auto_cls):
        try:
            return auto_cls.from_pretrained(path, **kwargs)
        except TypeError:  # transformers < 4.56 calls it torch_dtype
            kw = dict(kwargs)
            kw["torch_dtype"] = kw.pop("dtype")
            return auto_cls.from_pretrained(path, **kw)

    processor = None
    if multimodal:
        processor = transformers.AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", processor)
        auto = getattr(transformers, "AutoModelForImageTextToText", None) or transformers.AutoModelForVision2Seq
        model = _from_pretrained(auto)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        model = _from_pretrained(transformers.AutoModelForCausalLM)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "device_map" in kwargs or device == "none":
        pass  # placed by accelerate, or left for the caller (training)
    elif device not in ("auto", ""):
        model.to(device)
    else:
        import torch

        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, processor


def has_chat_template(tok) -> bool:
    return bool(getattr(tok, "chat_template", None))


def to_template_messages(messages: list[dict], media_root: str, video_frames: int, multimodal: bool,
                         tool_role_as_user: bool = True):
    """Convert Groundwork messages into what HF chat templates expect.

    Returns ``(template_messages, images, videos)``. For text-only models media parts are
    replaced by a short text note (images/videos) or the document text (files).
    """
    if tool_role_as_user:
        messages = tool_to_user(messages)
    images, videos, out = [], [], []
    for m in messages:
        content = []
        for p in parts(m.get("content")):
            t = p["type"]
            if t == "text":
                content.append({"type": "text", "text": p["text"]})
            elif t == "image":
                if multimodal:
                    images.append(media.load_image(p, media_root))
                    content.append({"type": "image"})
                else:
                    content.append({"type": "text", "text": "[image omitted: text-only model]"})
            elif t == "video":
                if multimodal:
                    videos.append(media.video_frames(p["path"], video_frames, media_root))
                    content.append({"type": "video"})
                else:
                    content.append({"type": "text", "text": "[video omitted: text-only model]"})
            elif t == "file":
                body = media.read_document(p["path"], media_root)
                content.append({"type": "text", "text": f"[file {Path(p['path']).name}]\n{body}\n[end of file]"})
        if multimodal:
            out.append({"role": m["role"], "content": content})
        else:
            out.append({"role": m["role"], "content": "\n\n".join(c["text"] for c in content)})
    return out, images, videos


class HFChatModel:
    parallel_safe = False  # one generate() at a time on a local model

    def __init__(self, path: str, dtype="bfloat16", device="auto", trust_remote_code=False, multimodal=False,
                 media_root=".", video_frames=8, max_tokens=1024, loader="", loaded=None):
        self.name = Path(path).name or path
        self.multimodal = multimodal
        self.media_root = media_root
        self.video_frames = video_frames
        self.max_tokens = max_tokens
        if loaded is not None:
            self.model, self.tokenizer, self.processor = loaded
        else:
            self.model, self.tokenizer, self.processor = load_hf(path, dtype, device, trust_remote_code, multimodal, loader)

    def render(self, messages: list[dict], add_generation_prompt: bool = True):
        tmpl_msgs, images, videos = to_template_messages(messages, self.media_root, self.video_frames, self.multimodal)
        renderer = self.processor if (self.multimodal and self.processor is not None) else self.tokenizer
        if has_chat_template(renderer) or has_chat_template(self.tokenizer):
            src = renderer if has_chat_template(renderer) else self.tokenizer
            text = src.apply_chat_template(tmpl_msgs, add_generation_prompt=add_generation_prompt, tokenize=False)
        else:
            text = render_simple(tmpl_msgs, add_generation_prompt)
        return text, images, videos

    def chat(self, messages, *, n=1, temperature=0.0, max_tokens=None, stop=None, prefill=None):
        import torch

        text, images, videos = self.render(messages, add_generation_prompt=True)
        if prefill:
            text += prefill
        if self.multimodal and self.processor is not None:
            kw = {"text": [text], "return_tensors": "pt"}
            if images:
                kw["images"] = images
            if videos:
                kw["videos"] = videos
            enc = self.processor(**kw)
        else:
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
        gen_kwargs = dict(
            max_new_tokens=max_tokens or self.max_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
            if n > 1:  # greedy decoding can't produce distinct samples
                gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95)
        if stop:
            gen_kwargs.update(stop_strings=stop, tokenizer=self.tokenizer)
        with torch.no_grad():
            try:
                out = self.model.generate(**enc, **gen_kwargs)
            except (TypeError, ValueError):
                # older transformers, or a tokenizer stop_strings can't handle: stop by trimming instead
                if "stop_strings" not in gen_kwargs:
                    raise
                gen_kwargs.pop("stop_strings", None)
                gen_kwargs.pop("tokenizer", None)
                out = self.model.generate(**enc, **gen_kwargs)
        prompt_len = enc["input_ids"].shape[1]
        texts = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        texts = [trim_at_stop(t, stop) for t in texts]
        if prefill:
            texts = [prefill + t for t in texts]
        return texts

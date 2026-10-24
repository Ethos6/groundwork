"""Render conversations with the model's chat template and mark which tokens to train on.

Only assistant turns are trained. System prompts, user turns, images and tool
observations are context: they are masked out (label -100) so the model learns to act,
not to predict tool outputs.
"""

from __future__ import annotations

import re
import warnings

from ..messages import text_of
from ..models.hf import has_chat_template, render_simple, to_template_messages

IGNORE = -100


class ChatEncoder:
    def __init__(self, tokenizer, processor=None, multimodal: bool = False, max_length: int = 4096,
                 media_root: str = ".", video_frames: int = 8, tool_role_as_user: bool = True):
        self.tok = tokenizer
        self.processor = processor
        self.multimodal = multimodal and processor is not None
        self.max_length = max_length
        self.media_root = media_root
        self.video_frames = video_frames
        self.tool_role_as_user = tool_role_as_user
        self._renderer = self.processor if (self.multimodal and has_chat_template(self.processor)) else self.tok
        self._end_marker: str | None = None
        self.stats = {"encoded": 0, "skipped_too_long": 0, "skipped_no_labels": 0, "span_fallbacks": 0}

    # ------------------------------------------------------------------ rendering
    def _render_tmpl(self, tmpl_msgs: list[dict], add_generation_prompt: bool) -> str:
        if has_chat_template(self._renderer):
            return self._renderer.apply_chat_template(tmpl_msgs, add_generation_prompt=add_generation_prompt, tokenize=False)
        return render_simple(tmpl_msgs, add_generation_prompt)

    def _convert(self, messages: list[dict]):
        return to_template_messages(messages, self.media_root, self.video_frames, self.multimodal, self.tool_role_as_user)

    @property
    def end_marker(self) -> str:
        if self._end_marker is None:
            demo = [{"role": "user", "content": "PING"}, {"role": "assistant", "content": "PONG"}]
            if self.multimodal:
                demo = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in demo]
            text = self._render_tmpl(demo, False)
            i = text.rfind("PONG")
            m = re.match(r"\s*(\S+)", text[i + 4 :]) if i != -1 else None
            self._end_marker = m.group(1) if m else ""
        return self._end_marker

    def assistant_spans(self, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
        """Return the rendered text and the character spans of each assistant turn
        (content plus the end-of-turn marker)."""
        tmpl, _, _ = self._convert(messages)
        full = self._render_tmpl(tmpl, False)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for a, m in enumerate(tmpl):
            if m["role"] != "assistant":
                continue
            span = None
            try:
                prefix = self._render_tmpl(tmpl[:a], True) if a > 0 else ""
                upto = self._render_tmpl(tmpl[: a + 1], False)
                if a > 0 and full.startswith(upto) and upto.startswith(prefix) and len(prefix) >= cursor:
                    end = len(prefix) + len(upto[len(prefix):].rstrip())
                    span = (len(prefix), end)
            except Exception:
                span = None
            if span is None:  # template isn't prefix-consistent: locate the content directly
                self.stats["span_fallbacks"] += 1
                content = text_of(messages[a]) if messages[a].get("role") == "assistant" else (m["content"] if isinstance(m["content"], str) else "")
                idx = full.find(content, cursor) if content else -1
                if idx == -1:
                    warnings.warn("could not locate an assistant turn in the rendered text; it will not be trained")
                    continue
                end = idx + len(content)
                j = end
                while j < len(full) and full[j].isspace():
                    j += 1
                if self.end_marker and full.startswith(self.end_marker, j):
                    end = j + len(self.end_marker)
                span = (idx, end)
            spans.append(span)
            cursor = span[1]
        return full, spans

    # ------------------------------------------------------------------ encoding
    def encode(self, messages: list[dict]) -> dict | None:
        full, spans = self.assistant_spans(messages)
        if self.multimodal:
            return self._encode_multimodal(messages, full, spans)
        return self._encode_text(full, spans)

    def _finish(self, ids: list[int], labels: list[int], extras: dict, can_truncate: bool) -> dict | None:
        if len(ids) > self.max_length:
            if not can_truncate:
                self.stats["skipped_too_long"] += 1
                return None
            ids, labels = ids[: self.max_length], labels[: self.max_length]
        if not any(l != IGNORE for l in labels):
            self.stats["skipped_no_labels"] += 1
            return None
        self.stats["encoded"] += 1
        return {"input_ids": ids, "labels": labels, "extras": extras}

    def _encode_text(self, full: str, spans: list[tuple[int, int]]) -> dict | None:
        if getattr(self.tok, "is_fast", False):
            enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            labels = [IGNORE] * len(ids)
            j = 0
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                while j < len(spans) and s >= spans[j][1]:
                    j += 1
                if j < len(spans) and spans[j][0] <= s < spans[j][1] and e > s:
                    labels[i] = ids[i]
            return self._finish(ids, labels, {}, can_truncate=True)
        # slow tokenizer: tokenize segment by segment
        ids, labels, pos = [], [], 0
        for s, e in spans:
            if s > pos:
                seg = self.tok(full[pos:s], add_special_tokens=False)["input_ids"]
                ids += seg
                labels += [IGNORE] * len(seg)
            seg = self.tok(full[s:e], add_special_tokens=False)["input_ids"]
            ids += seg
            labels += seg
            pos = e
        if pos < len(full):
            seg = self.tok(full[pos:], add_special_tokens=False)["input_ids"]
            ids += seg
            labels += [IGNORE] * len(seg)
        return self._finish(ids, labels, {}, can_truncate=True)

    def _encode_multimodal(self, messages: list[dict], full: str, spans: list[tuple[int, int]]) -> dict | None:
        _, images, videos = self._convert(messages)
        kw = {"text": [full], "return_tensors": "pt"}
        if images:
            kw["images"] = images
        if videos:
            kw["videos"] = videos
        enc = self.processor(**kw)
        ids = enc["input_ids"][0].tolist()
        labels = [IGNORE] * len(ids)
        cursor = 0
        for s, e in spans:  # assistant text has no media, so its tokens appear verbatim
            seg = self.tok(full[s:e], add_special_tokens=False)["input_ids"]
            hit = _find_subsequence(ids, seg, cursor)
            offset = 0
            if hit == -1 and len(seg) > 1:  # first token may have merged with the header
                hit, offset = _find_subsequence(ids, seg[1:], cursor), 1
            if hit == -1:
                self.stats["span_fallbacks"] += 1
                continue
            start = hit - offset if offset else hit
            end = hit + len(seg) - offset
            for i in range(start, end):
                labels[i] = ids[i]
            cursor = end
        extras = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}
        return self._finish(ids, labels, extras, can_truncate=False)


def _find_subsequence(seq: list[int], sub: list[int], start: int = 0) -> int:
    if not sub:
        return -1
    first = sub[0]
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i] == first and seq[i : i + n] == sub:
            return i
    return -1


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
    keys = set().union(*(b["extras"].keys() for b in batch))
    for k in keys:
        vals = [b["extras"][k] for b in batch if k in b["extras"]]
        out[k] = torch.cat(vals, dim=0) if len(vals) > 1 else vals[0]
    return out

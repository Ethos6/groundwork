"""Connector for any OpenAI-compatible chat completions endpoint.

Works with self-hosted servers (vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio) and with
hosted APIs that speak the same protocol.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import requests

from .. import media
from ..config import EndpointConfig
from ..messages import parts
from .base import GenerationError, trim_at_stop


class OpenAICompatModel:
    def __init__(self, cfg: EndpointConfig, media_root: str = ".", video_frames: int = 8):
        if not cfg.model:
            raise ValueError("model.model (the served model name) is required for kind=openai")
        self.cfg = cfg
        self.name = cfg.model
        self.media_root = media_root
        self.video_frames = video_frames
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"

    # ------------------------------------------------------------------ conversion
    def _convert_content(self, content):
        ps = parts(content)
        if all(p["type"] == "text" for p in ps):
            return "\n\n".join(p["text"] for p in ps)
        out = []
        for p in ps:
            t = p["type"]
            if t == "text":
                out.append({"type": "text", "text": p["text"]})
            elif t == "image":
                if p.get("url"):
                    url = p["url"]
                else:
                    url = media.image_to_data_url(media.load_image(p, self.media_root))
                out.append({"type": "image_url", "image_url": {"url": url}})
            elif t == "video":
                frames = media.video_frames(p["path"], self.video_frames, self.media_root)
                out.append({"type": "text", "text": f"[video {Path(p['path']).name}: {len(frames)} frames in order]"})
                for fr in frames:
                    out.append({"type": "image_url", "image_url": {"url": media.image_to_data_url(fr, 768)}})
            elif t == "file":
                body = media.read_document(p["path"], self.media_root)
                out.append({"type": "text", "text": f"[file {Path(p['path']).name}]\n{body}\n[end of file]"})
        return out

    def _convert(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = m["role"]
            if role == "tool":  # plain observation turn; avoids tool_call_id bookkeeping
                role = "user"
            out.append({"role": role, "content": self._convert_content(m.get("content"))})
        return out

    # ------------------------------------------------------------------ request
    def _post(self, payload: dict) -> dict:
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.cfg.timeout_s)
                if r.status_code == 200:
                    return r.json()
                last_err = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    break
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(min(30.0, (2**attempt) + random.random()))
        raise GenerationError(f"{self.name}: request failed ({last_err})")

    def chat(self, messages, *, n=1, temperature=0.0, max_tokens=None, stop=None, prefill=None):
        msgs = self._convert(messages)
        body = dict(self.cfg.extra_body)
        use_prefill = bool(prefill) and self.cfg.supports_prefill
        if prefill and not use_prefill:
            msgs.append({"role": "user", "content": f"Begin your reply with this first thought:\n{prefill}"})
        if use_prefill:
            msgs.append({"role": "assistant", "content": prefill})
            body.update({"continue_final_message": True, "add_generation_prompt": False})
        payload = {
            "model": self.cfg.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            **body,
        }
        if stop:
            payload["stop"] = stop[:4]
        outs: list[str] = []
        if n > 1 and self.cfg.supports_n:
            payload["n"] = n
            outs = [c["message"].get("content") or "" for c in self._post(payload).get("choices", [])]
        while len(outs) < n:
            payload.pop("n", None)
            choices = self._post(payload).get("choices", [])
            if not choices:
                raise GenerationError(f"{self.name}: response contained no choices")
            outs.append(choices[0]["message"].get("content") or "")
        outs = [trim_at_stop(o, stop) for o in outs[:n]]
        if use_prefill:
            outs = [prefill + o for o in outs]
        return outs

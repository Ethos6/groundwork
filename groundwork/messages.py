"""Chat message helpers.

Messages are plain dicts in an OpenAI-like shape so they serialize to JSONL directly::

    {"role": "user", "content": "plain text"}
    {"role": "user", "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image", "path": "photos/cat.jpg"},        # or "url": "data:image/png;base64,..."
        {"type": "video", "path": "clips/demo.mp4"},
        {"type": "file",  "path": "docs/report.pdf"},
    ]}

Roles: system, user, assistant, tool (tool = an observation returned to the model).
"""

from __future__ import annotations

import copy
from typing import Iterable

MEDIA_TYPES = ("image", "video", "file")


def parts(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for p in content:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
        elif p.get("type") == "image_url":  # OpenAI wire format
            url = p.get("image_url", {})
            out.append({"type": "image", "url": url.get("url") if isinstance(url, dict) else url})
        elif p.get("type") == "input_text":
            out.append({"type": "text", "text": p.get("text", "")})
        else:
            out.append(dict(p))
    return out


def text_of(message_or_content) -> str:
    content = message_or_content.get("content") if isinstance(message_or_content, dict) and "role" in message_or_content else message_or_content
    return "\n".join(p.get("text", "") for p in parts(content) if p.get("type") == "text")


def media_of(message: dict) -> list[dict]:
    return [p for p in parts(message.get("content")) if p.get("type") in MEDIA_TYPES]


def has_media(messages: Iterable[dict]) -> bool:
    return any(media_of(m) for m in messages)


def user_message(text: str, images: Iterable[str] = (), videos: Iterable[str] = (), files: Iterable[str] = ()) -> dict:
    content: list[dict] = []
    content += [{"type": "image", "path": p} for p in images]
    content += [{"type": "video", "path": p} for p in videos]
    content += [{"type": "file", "path": p} for p in files]
    if not content:
        return {"role": "user", "content": text}
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def with_system(messages: list[dict], system_text: str) -> list[dict]:
    """Return a copy with ``system_text`` prepended to (or used as) the system message."""
    if not system_text:
        return list(messages)
    msgs = copy.deepcopy(messages)
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = system_text + "\n\n" + text_of(msgs[0])
    else:
        msgs.insert(0, {"role": "system", "content": system_text})
    return msgs


def append_to_last_user(messages: list[dict], text: str) -> list[dict]:
    msgs = copy.deepcopy(messages)
    for m in reversed(msgs):
        if m.get("role") == "user":
            ps = parts(m.get("content"))
            ps.append({"type": "text", "text": text})
            m["content"] = ps if any(p["type"] != "text" for p in ps) else "\n\n".join(p["text"] for p in ps)
            return msgs
    msgs.append({"role": "user", "content": text})
    return msgs


def last_user(messages: list[dict]) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    return None


def tool_to_user(messages: list[dict]) -> list[dict]:
    """Many chat templates reject a bare 'tool' role; render observations as user turns."""
    out = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "user", "content": m.get("content")})
        else:
            out.append(m)
    return out

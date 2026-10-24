"""The minimal interface every model connector implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class GenerationError(RuntimeError):
    pass


@runtime_checkable
class ChatModel(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        prefill: str | None = None,
    ) -> list[str]:
        """Return ``n`` completions for ``messages``.

        ``prefill`` starts the assistant reply with the given text (when supported); the
        returned strings then include the prefill.
        """
        ...


def trim_at_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def max_workers(model, requested: int) -> int:
    """Local models run one request at a time; API endpoints can take several."""
    return max(1, requested) if getattr(model, "parallel_safe", True) else 1

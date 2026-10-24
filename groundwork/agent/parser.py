"""Parse agent steps of the form ``Thought: ... Code: ```py ... ``` ``."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:py|python|python3)?[ \t]*\n(.*?)(?:\n?```|<end_code>|$)", re.S | re.I)


class ParseError(ValueError):
    pass


@dataclass
class ParsedStep:
    thought: str
    code: str

    def render(self) -> str:
        """Canonical form stored in training data."""
        return f"Thought: {self.thought}\nCode:\n```py\n{self.code}\n```<end_code>"


def parse_step(text: str) -> ParsedStep:
    text = text.replace("<end_code>", "\n```" if text.count("```") % 2 == 1 else "")
    m = _FENCE_RE.search(text)
    if not m or not m.group(1).strip():
        raise ParseError(
            "Error in code parsing: no code block found. Write 'Thought:' then 'Code:' followed by "
            "a ```py ... ``` block."
        )
    code = m.group(1).strip("\n")
    head = text[: m.start()]
    head = re.sub(r"\n?\s*Code:\s*$", "", head.strip(), flags=re.I)
    thought = re.sub(r"^\s*Thought:\s*", "", head, flags=re.I).strip()
    try:
        compile(code, "<agent>", "exec")
    except SyntaxError as e:
        raise ParseError(f"Error in code parsing: SyntaxError: {e.msg} (line {e.lineno})") from e
    return ParsedStep(thought=thought, code=code)

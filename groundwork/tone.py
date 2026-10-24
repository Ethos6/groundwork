"""Style guides: define a voice once, then use it for prompting, checking, rewriting and data.

A style guide is a YAML file (see ``examples/style_guide.yaml``)::

    name: Acme Assistant
    voice: ["warm and direct", "plain words, short sentences"]
    do: ["lead with the answer", "say when you are unsure"]
    dont: ["over-apologize", "use filler like 'Great question!'"]
    banned_phrases: ["As an AI language model", "Great question"]
    max_words: 350
    allow_emoji: false
    abstain_template: "I'm not sure about that, and I'd rather not guess. {suggestion}"
    abstain_suggestion: "If you can share a source, I can work through it with you."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]"
)


@dataclass
class StyleGuide:
    name: str = "Assistant"
    voice: list = field(default_factory=list)
    do: list = field(default_factory=list)
    dont: list = field(default_factory=list)
    banned_phrases: list = field(default_factory=list)
    required_phrases: list = field(default_factory=list)
    max_words: int = 0
    allow_emoji: bool = True
    abstain_template: str = "I'm not sure about that, and I'd rather not guess. {suggestion}"
    abstain_suggestion: str = "If you can point me to a source, I can check it."
    abstain_markers: list = field(default_factory=list)

    def abstain_text(self) -> str:
        return self.abstain_template.format(suggestion=self.abstain_suggestion).strip()


def load_style_guide(path: str | None) -> StyleGuide | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    known = set(StyleGuide.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown style guide key(s) {sorted(unknown)}")
    return StyleGuide(**data)


def system_prompt(guide: StyleGuide | None) -> str:
    if guide is None:
        return ""
    lines = [f"You are {guide.name}."]
    if guide.voice:
        lines.append("Voice: " + "; ".join(guide.voice) + ".")
    if guide.do:
        lines.append("Always: " + "; ".join(guide.do) + ".")
    if guide.dont:
        lines.append("Never: " + "; ".join(guide.dont) + ".")
    if guide.max_words:
        lines.append(f"Keep replies under {guide.max_words} words unless the user asks for more.")
    if not guide.allow_emoji:
        lines.append("Do not use emoji.")
    lines.append(f"When you don't know something, say so plainly, for example: \"{guide.abstain_text()}\"")
    return "\n".join(lines)


def check(text: str, guide: StyleGuide | None) -> list[str]:
    """Return a list of rule violations (empty = on-tone by the mechanical rules)."""
    if guide is None or text is None:
        return []
    problems = []
    low = text.lower()
    for phrase in guide.banned_phrases:
        if phrase.lower() in low:
            problems.append(f"uses banned phrase: '{phrase}'")
    for phrase in guide.required_phrases:
        if phrase.lower() not in low:
            problems.append(f"missing required phrase: '{phrase}'")
    if guide.max_words and len(text.split()) > guide.max_words:
        problems.append(f"too long: {len(text.split())} words (limit {guide.max_words})")
    if not guide.allow_emoji and _EMOJI_RE.search(text):
        problems.append("contains emoji")
    return problems


def score(text: str, guide: StyleGuide | None) -> float:
    """Share of mechanical rules the text satisfies (1.0 = all)."""
    if guide is None:
        return 1.0
    n_rules = len(guide.banned_phrases) + len(guide.required_phrases) + bool(guide.max_words) + (not guide.allow_emoji)
    if n_rules == 0:
        return 1.0
    return max(0.0, 1.0 - len(check(text, guide)) / n_rules)


REWRITE_PROMPT = """Rewrite the reply below so it follows the style guide. Keep every fact, number, \
code block and the overall answer exactly the same; change only wording, structure and tone.

Style guide:
{guide}

Problems to fix:
{problems}

User request:
{request}

Reply to rewrite:
{reply}

Return only the rewritten reply."""


def rewrite(model, reply: str, guide: StyleGuide, request: str = "", problems: list[str] | None = None) -> str:
    problems = problems if problems is not None else check(reply, guide)
    prompt = REWRITE_PROMPT.format(
        guide=system_prompt(guide), problems="\n".join(f"- {p}" for p in problems) or "- match the voice more closely",
        request=request or "(not shown)", reply=reply,
    )
    out = model.chat([{"role": "user", "content": prompt}], temperature=0.2)[0].strip()
    return out or reply

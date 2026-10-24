"""Hallucination guard based on answer consistency.

If a model really knows a fact, independent samples tend to agree. If they scatter, the
model is probably guessing. The guard samples several short answers and measures how
many agree. Low agreement means: verify with tools, or say you're not sure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..agent.prompts import SHORT_ANSWER_SUFFIX
from ..config import GuardConfig
from ..data.scorers import extract_answer, is_abstention, normalize
from ..messages import append_to_last_user, last_user, text_of

FACT_GATE_PROMPT = """Does a correct reply to the request below depend on specific facts \
(names, dates, numbers, events, quotes, citations, technical specifics) that could be wrong \
if misremembered? Reply with one word: YES or NO.

Request:
{request}"""


@dataclass
class GuardResult:
    checked: bool
    agreement: float = 1.0
    answer: str | None = None
    samples: list = field(default_factory=list)
    reason: str = ""


class HallucinationGuard:
    def __init__(self, model, cfg: GuardConfig):
        self.model = model
        self.cfg = cfg

    def needs_check(self, messages: list[dict]) -> bool:
        if not self.cfg.fact_check_gate:
            return True
        request = text_of(last_user(messages) or {"role": "user", "content": ""})
        out = self.model.chat([{"role": "user", "content": FACT_GATE_PROMPT.format(request=request)}],
                              temperature=0.0, max_tokens=3)[0]
        return out.strip().upper().startswith("Y")

    def assess(self, messages: list[dict]) -> GuardResult:
        msgs = append_to_last_user(messages, SHORT_ANSWER_SUFFIX)
        outs = self.model.chat(msgs, n=self.cfg.samples, temperature=self.cfg.temperature, max_tokens=256)
        answers = []
        for o in outs:
            short = extract_answer(o)
            answers.append(None if is_abstention(short) else normalize(short))
        counts = Counter(a for a in answers if a)
        if not counts:
            return GuardResult(True, 0.0, None, outs, "every sample declined to answer")
        majority, c = counts.most_common(1)[0]
        agreement = c / max(1, len(outs))
        reason = f"{c}/{len(outs)} samples agree"
        return GuardResult(True, agreement, majority, outs, reason)

    def check(self, messages: list[dict]) -> GuardResult:
        if not self.cfg.enabled:
            return GuardResult(False, reason="guard disabled")
        if not self.needs_check(messages):
            return GuardResult(False, reason="request does not depend on specific facts")
        return self.assess(messages)

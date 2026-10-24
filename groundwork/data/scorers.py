"""Answer checking: exact / contains / numeric / choice / code / LLM judge, plus abstention detection."""

from __future__ import annotations

import re
import string
import subprocess
import sys
import tempfile
from fractions import Fraction

from .tasks import Task

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*[:\-]\s*(.+)", re.I)

ABSTAIN_PATTERNS = [
    r"\bi(?:'m| am) not (?:sure|certain)\b",
    r"\bi do(?:n't| not) know\b",
    r"\bi(?: cannot|can't|can not) (?:verify|confirm|determine|be certain)\b",
    r"\bnot enough information\b",
    r"\bi(?:'m| am) unable to (?:verify|confirm|determine)\b",
    r"\bi(?:'d| would) rather not guess\b",
    r"^\s*unknown\s*\.?\s*$",
    r"\banswer:\s*unknown\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.I | re.M)


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {"."})
    text = _ARTICLES.sub(" ", text)
    text = re.sub(r"\.(?!\d)", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    """Take the last 'Answer: ...' line if present, otherwise the whole text."""
    matches = _ANSWER_LINE.findall(text or "")
    if matches:
        return matches[-1].strip().strip("*").strip()
    return (text or "").strip()


def is_abstention(text: str, extra_markers: list[str] | None = None) -> bool:
    if not text:
        return False
    if _ABSTAIN_RE.search(text):
        return True
    low = text.lower()
    return any(m.lower() in low for m in (extra_markers or []))


def _to_number(text: str) -> float | None:
    t = text.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?(?:e-?\d+)?", t)
    if not m:
        return None
    s = m.group(0)
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def score_exact(pred: str, gold: list[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in gold))


def score_contains(pred: str, gold: list[str]) -> float:
    p = normalize(pred)
    return float(any(normalize(g) and normalize(g) in p for g in gold))


def score_numeric(pred: str, gold: list[str], rel_tol: float = 1e-4) -> float:
    pv = _to_number(pred)
    if pv is None:
        return 0.0
    for g in gold:
        gv = _to_number(g)
        if gv is not None and abs(pv - gv) <= rel_tol * max(1.0, abs(gv)):
            return 1.0
    return 0.0


def score_choice(pred: str, gold: list[str]) -> float:
    m = re.search(r"\b([A-J])\b", pred.strip().upper())
    return float(bool(m) and any(m.group(1) == g.strip().upper()[:1] for g in gold))


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:py|python)?\s*\n(.*?)```", text or "", re.S)
    return m[-1] if m else (text or "")


def score_code(pred: str, tests: str, timeout_s: float = 10.0) -> float:
    if not tests:
        return 0.0
    program = extract_code(pred) + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout_s)
        return float(r.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0.0


JUDGE_PROMPT = """You are grading an answer against a reference.
Question: {question}
Reference answer: {gold}
Candidate answer: {pred}

Is the candidate answer correct and consistent with the reference? Reply with exactly one word: CORRECT or INCORRECT."""


def score_judge(task: Task, pred: str, judge) -> float:
    if judge is None:
        raise ValueError(f"task {task.id} uses answer_type=judge but no judge endpoint is configured")
    out = judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(
        question=task.question, gold=" | ".join(task.gold), pred=pred)}], temperature=0.0, max_tokens=5)[0]
    return float(out.strip().upper().startswith("CORRECT"))


def score(task: Task, prediction: str | None, judge=None) -> float:
    if prediction is None:
        return 0.0
    kind = task.answer_type
    if kind == "none":
        return 0.0
    if kind == "code":
        return score_code(prediction, task.tests)
    if kind == "judge":
        return score_judge(task, prediction, judge)
    short = extract_answer(prediction)
    gold = task.gold
    if not gold:
        return 0.0
    if kind == "exact":
        return max(score_exact(short, gold), score_numeric(short, gold) if _to_number(gold[0]) is not None else 0.0)
    if kind == "contains":
        return score_contains(prediction, gold)
    if kind == "numeric":
        return score_numeric(short, gold)
    if kind == "choice":
        return score_choice(short, gold)
    raise ValueError(f"unknown answer_type {kind}")

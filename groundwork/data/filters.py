"""Keep only trustworthy traces, pick the best ones per task, and remove eval overlap."""

from __future__ import annotations

from collections import defaultdict

from ..messages import last_user, text_of
from .scorers import normalize


def filter_traces(records: list[dict], keep_recovered_errors: bool = True, min_score: float = 1.0) -> tuple[list[dict], dict]:
    """Keep traces that reached a correct final answer cleanly.

    * the final answer must score >= ``min_score``;
    * any unparseable step drops the trace (the model would learn broken formatting);
    * hitting the step limit drops the trace;
    * code that crashed and was then fixed is kept by default: it teaches recovery.
    """
    stats = defaultdict(int)
    kept = []
    for r in records:
        stats["total"] += 1
        correct = r.get("score", 0.0) >= min_score
        stats["correct"] += correct
        if r.get("final_answer") is None:
            stats["no_final_answer"] += 1
            continue
        if not correct:
            stats["incorrect"] += 1
            continue
        if r.get("n_parse_errors", 0) > 0:
            stats["dropped_parse_error"] += 1
            continue
        if r.get("max_steps_reached"):
            stats["dropped_max_steps"] += 1
            continue
        if r.get("n_exec_errors", 0) > 0:
            if not keep_recovered_errors:
                stats["dropped_exec_error"] += 1
                continue
            stats["kept_with_recovery"] += 1
        kept.append(r)
    stats["kept"] = len(kept)
    return kept, dict(stats)


def select_best(records: list[dict], per_task: int = 1) -> list[dict]:
    """Prefer short, clean traces; drop exact duplicates."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    out = []
    for task_id, rs in by_task.items():
        rs.sort(key=lambda r: (r.get("n_steps", 0), r.get("n_exec_errors", 0), r.get("attempt", 0)))
        seen = set()
        for r in rs:
            key = tuple(text_of(m) for m in r["messages"] if m["role"] == "assistant")
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(seen) >= per_task:
                break
    return out


def _ngrams(text: str, n: int) -> set[tuple]:
    toks = normalize(text).split()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class Decontaminator:
    """Flags training examples whose user prompt overlaps an evaluation question."""

    def __init__(self, eval_questions: list[str], n: int = 13):
        self.n = n
        self.exact = {normalize(q) for q in eval_questions}
        self.grams: set[tuple] = set()
        for q in eval_questions:
            self.grams |= _ngrams(q, n)

    def is_contaminated(self, text: str) -> bool:
        if normalize(text) in self.exact:
            return True
        return bool(self.grams and _ngrams(text, self.n) & self.grams)

    def filter_examples(self, examples: list[dict]) -> tuple[list[dict], int]:
        kept, dropped = [], 0
        for ex in examples:
            user = last_user(ex["messages"]) if ex.get("messages") else None
            first_user = next((m for m in ex.get("messages", []) if m["role"] == "user"), user)
            if first_user is not None and self.is_contaminated(text_of(first_user)):
                dropped += 1
                continue
            kept.append(ex)
        return kept, dropped

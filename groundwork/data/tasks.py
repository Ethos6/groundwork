"""Task files: one JSON object per line.

    {"id": "math-001",
     "question": "What is 17% of 2,340?",
     "answer": "397.8",                     # gold answer (string or list of acceptable strings)
     "answer_type": "numeric",              # exact | contains | numeric | choice | code | judge | none
     "domain": "math",
     "closed_book": false,                  # true = answerable from memory (used by the knowledge probe)
     "images": [], "videos": [], "files": [],
     "tests": "assert solve(2) == 4"}       # for answer_type=code
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..io import read_jsonl
from ..messages import user_message

ANSWER_TYPES = {"exact", "contains", "numeric", "choice", "code", "judge", "none"}


@dataclass
class Task:
    id: str
    question: str
    answer: str | list | None = None
    answer_type: str = "exact"
    domain: str = "general"
    closed_book: bool = False
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    files: list = field(default_factory=list)
    tests: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def gold(self) -> list[str]:
        if self.answer is None:
            return []
        return [str(a) for a in self.answer] if isinstance(self.answer, list) else [str(self.answer)]

    @property
    def has_media(self) -> bool:
        return bool(self.images or self.videos or self.files)

    def user_message(self) -> dict:
        return user_message(self.question, self.images, self.videos, self.files)


def load_tasks(path: str, limit: int | None = None) -> list[Task]:
    tasks, seen = [], set()
    known = set(Task.__dataclass_fields__)
    for i, row in enumerate(read_jsonl(path)):
        row = dict(row)
        row.setdefault("id", f"task-{i:06d}")
        if "question" not in row:
            raise ValueError(f"{path}: task {row['id']} has no 'question'")
        extra = {k: row.pop(k) for k in list(row) if k not in known}
        row.setdefault("meta", {}).update(extra)
        task = Task(**row)
        if task.answer_type not in ANSWER_TYPES:
            raise ValueError(f"{path}: task {task.id} has unknown answer_type {task.answer_type!r}")
        if task.id in seen:
            raise ValueError(f"{path}: duplicate task id {task.id}")
        seen.add(task.id)
        tasks.append(task)
        if limit and len(tasks) >= limit:
            break
    return tasks

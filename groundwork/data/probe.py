"""Knowledge probe: find out what the model actually knows before training on facts.

Each closed-book question is sampled several times. The share of correct samples decides
the category:

* known   (>= known_threshold): safe to reinforce the model's own correct answer;
* partial (between): reinforce the correct answer it sometimes gives;
* unknown (<= unknown_threshold): teach it to look the fact up, or to say it isn't sure.

Training a model on facts it doesn't know encourages confident guessing, so unknown
facts are never trained as closed-book answers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..agent.prompts import SHORT_ANSWER_SUFFIX
from ..config import Config
from ..io import write_jsonl
from ..messages import append_to_last_user
from ..models.base import max_workers
from ..models import build_data_source
from .scorers import extract_answer, is_abstention, score
from .tasks import Task

PROBE_TYPES = {"exact", "contains", "numeric", "choice", "judge"}


def categorize(p_correct: float, known: float, unknown: float) -> str:
    if p_correct >= known:
        return "known"
    if p_correct <= unknown:
        return "unknown"
    return "partial"


def run_probe(cfg: Config, tasks: list[Task], out_path: str | None = None) -> list[dict]:
    model = build_data_source(cfg, "model", "knowledge probe")
    eligible = [t for t in tasks if t.closed_book and t.gold and t.answer_type in PROBE_TYPES]
    out_path = out_path or str(Path(cfg.data.work_dir) / "probe.jsonl")

    def work(task: Task) -> dict:
        msgs = append_to_last_user([task.user_message()], SHORT_ANSWER_SUFFIX)
        outs = model.chat(msgs, n=cfg.data.probe_samples, temperature=cfg.data.probe_temperature)
        samples = []
        for o in outs:
            samples.append({"text": o, "score": score(task, o), "abstained": is_abstention(extract_answer(o))})
        p = sum(s["score"] for s in samples) / max(1, len(samples))
        return {
            "task_id": task.id,
            "p_correct": p,
            "p_abstain": sum(s["abstained"] for s in samples) / max(1, len(samples)),
            "category": categorize(p, cfg.data.known_threshold, cfg.data.unknown_threshold),
            "samples": samples,
        }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers(model, cfg.data.workers)) as pool:
        futures = [pool.submit(work, t) for t in eligible]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[probe] failed: {type(e).__name__}: {e}", flush=True)
    results.sort(key=lambda r: r["task_id"])
    write_jsonl(out_path, results)
    return results

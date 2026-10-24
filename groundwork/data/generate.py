"""Generate agent traces (rejection sampling) and first-thought prefixes."""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import tone
from ..agent.prompts import COT_PROMPT_SUFFIX
from ..agent.runner import AgentRunner
from ..config import Config
from ..io import append_jsonl, ensure_dir, iter_jsonl, read_json, write_json
from ..messages import append_to_last_user
from ..models.base import max_workers
from ..models import build_data_source, build_judge
from .scorers import score
from .tasks import Task


def _progress(done: int, total: int, label: str) -> None:
    if total and (done == total or done % max(1, total // 20) == 0):
        print(f"[{label}] {done}/{total}", flush=True)


def build_prefixes(cfg: Config, tasks: list[Task]) -> dict[str, str]:
    """First-thought prefixes: the opening paragraph of a step-by-step answer, reused as the
    agent's first 'Thought:' so the run starts from a plan."""
    model = build_data_source(cfg, cfg.data.actor, "first-thought prefixes")
    path = Path(cfg.data.work_dir) / "prefixes.json"
    prefixes: dict[str, str] = read_json(path) if path.exists() else {}
    todo = [t for t in tasks if t.id not in prefixes]

    def work(task: Task) -> tuple[str, str]:
        msgs = append_to_last_user([task.user_message()], COT_PROMPT_SUFFIX)
        text = model.chat(msgs, temperature=0.0, max_tokens=512)[0].strip()
        first = text.split("\n\n")[0].strip()
        first = first[len("Thought:"):].strip() if first.lower().startswith("thought:") else first
        return task.id, f"Thought: {first}\n"

    with ThreadPoolExecutor(max_workers=max_workers(model, cfg.data.workers)) as pool:
        futures = [pool.submit(work, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            tid, prefix = fut.result()
            prefixes[tid] = prefix
            _progress(i, len(todo), "prefix")
    write_json(path, prefixes)
    return prefixes


def generate_traces(cfg: Config, tasks: list[Task], out_path: str | None = None) -> str:
    """Run the agent ``attempts_per_task`` times per task and record scored trajectories.

    Re-running resumes: attempts already in the output file are skipped.
    """
    ensure_dir(cfg.data.work_dir)
    out_path = out_path or str(Path(cfg.data.work_dir) / "traces.jsonl")
    model = build_data_source(cfg, cfg.data.actor, "agent traces")
    judge = build_judge(cfg)
    guide = tone.load_style_guide(cfg.runtime.tone.style_guide)
    extra_system = tone.system_prompt(guide)

    prefixes: dict[str, str] = {}
    prefix_path = Path(cfg.data.work_dir) / "prefixes.json"
    if cfg.data.first_thought_prefix:
        prefixes = read_json(prefix_path) if prefix_path.exists() else build_prefixes(cfg, tasks)

    done = Counter()
    if Path(out_path).exists():
        for r in iter_jsonl(out_path):
            done[(r["task_id"], r.get("attempt", 0))] += 1
    jobs = [(t, a) for t in tasks for a in range(cfg.data.attempts_per_task) if (t.id, a) not in done]
    lock = threading.Lock()

    def work(task: Task, attempt: int) -> dict:
        runner = AgentRunner(model, cfg.agent, cfg.data.media_root, extra_system)
        temperature = 0.0 if attempt == 0 else cfg.data.attempt_temperature
        traj = runner.run(task.user_message(), task.id, cfg.data.actor, prefixes.get(task.id), temperature)
        rec = traj.to_dict()
        rec.update(attempt=attempt, domain=task.domain, score=score(task, traj.final_answer, judge))
        return rec

    with ThreadPoolExecutor(max_workers=max_workers(model, cfg.data.workers)) as pool:
        futures = [pool.submit(work, t, a) for t, a in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # one failed request shouldn't stop a long run
                print(f"[traces] attempt failed: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                append_jsonl(out_path, rec)
            _progress(i, len(jobs), "traces")
    return out_path

"""Measure the three issues on your task files, per mode, and compare with a baseline.

Modes:
* direct    -- the bare model, one reply
* agent     -- the tool-using agent's final answer
* assistant -- the full runtime layer (routing, guard, tone)

Metrics (tasks with a gold answer):
* accuracy            correct / all
* abstain_rate        declined to answer / all
* hallucination_rate  answered but wrong / all       (lower is better)
* selective_accuracy  correct / answered
* tone_score          share of style-guide rules satisfied (direct + assistant modes)
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import tone
from ..agent.runner import AgentRunner
from ..config import Config
from ..data.scorers import is_abstention, score
from ..data.tasks import Task, load_tasks
from ..io import ensure_dir, read_json, write_json, write_jsonl
from ..messages import append_to_last_user, with_system
from ..models.base import max_workers
from ..models import build_judge, build_model
from ..runtime.assistant import Assistant

EVAL_SUFFIX = (
    "\n\nIf the request has a short answer, end with a final line 'Answer: <answer>'. "
    "If you are not sure, say so instead of guessing."
)
HIGHER_IS_BETTER = ("accuracy", "selective_accuracy", "tone_score")
LOWER_IS_BETTER = ("hallucination_rate",)


def _run_one(mode: str, task: Task, model, cfg: Config, guide, assistant: Assistant | None) -> dict:
    msgs = [task.user_message()]
    steps = None
    if mode == "direct":
        out = model.chat(with_system(append_to_last_user(msgs, EVAL_SUFFIX), tone.system_prompt(guide)), temperature=0.0)[0]
    elif mode == "agent":
        traj = AgentRunner(model, cfg.agent, cfg.data.media_root, tone.system_prompt(guide)).run(msgs[0], task.id)
        out, steps = traj.final_answer or "", len(traj.steps)
    elif mode == "assistant":
        reply = assistant.respond(append_to_last_user(msgs, EVAL_SUFFIX))
        out = reply.text
        steps = reply.trajectory.get("n_steps") if reply.trajectory else 0
    else:
        raise ValueError(f"unknown eval mode {mode!r}")
    return {"task_id": task.id, "domain": task.domain, "mode": mode, "output": out, "steps": steps}


def summarize(rows: list[dict], tasks: dict[str, Task], judge, guide, mode: str) -> dict:
    n = correct = abstained = wrong = 0
    tone_scores, steps = [], []
    by_domain = defaultdict(lambda: [0, 0])
    markers = guide.abstain_markers if guide else []
    for r in rows:
        t = tasks[r["task_id"]]
        if mode != "agent":
            tone_scores.append(tone.score(r["output"], guide))
        if r.get("steps") is not None:
            steps.append(r["steps"])
        if not t.gold and t.answer_type != "code":
            continue
        n += 1
        s = score(t, r["output"], judge)
        r["score"] = s
        ok = s >= 1.0
        abst = not ok and is_abstention(r["output"], markers)
        r["abstained"] = abst
        correct += ok
        abstained += abst
        wrong += (not ok) and (not abst)
        by_domain[t.domain][0] += ok
        by_domain[t.domain][1] += 1
    answered = n - abstained
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "abstain_rate": round(abstained / n, 4) if n else None,
        "hallucination_rate": round(wrong / n, 4) if n else None,
        "selective_accuracy": round(correct / answered, 4) if answered else None,
        "tone_score": round(sum(tone_scores) / len(tone_scores), 4) if tone_scores and guide else None,
        "avg_steps": round(sum(steps) / len(steps), 2) if steps else None,
        "by_domain": {d: round(c / m, 4) for d, (c, m) in sorted(by_domain.items())},
    }


def compare(report: dict, baseline: dict, tol: float) -> list[str]:
    issues = []
    for name, modes in report["results"].items():
        for mode, metrics in modes.items():
            base = baseline.get("results", {}).get(name, {}).get(mode)
            if not base:
                continue
            for k in HIGHER_IS_BETTER:
                if metrics.get(k) is not None and base.get(k) is not None and metrics[k] < base[k] - tol:
                    issues.append(f"{name}/{mode}: {k} dropped {base[k]:.3f} -> {metrics[k]:.3f}")
            for k in LOWER_IS_BETTER:
                if metrics.get(k) is not None and base.get(k) is not None and metrics[k] > base[k] + tol:
                    issues.append(f"{name}/{mode}: {k} rose {base[k]:.3f} -> {metrics[k]:.3f}")
    return issues


def to_markdown(report: dict) -> str:
    lines = [f"# Evaluation: {report['model']}", ""]
    cols = ["n", "accuracy", "hallucination_rate", "abstain_rate", "selective_accuracy", "tone_score", "avg_steps"]
    lines.append("| task file | mode | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 2))
    for name, modes in report["results"].items():
        for mode, m in modes.items():
            vals = ["-" if m.get(c) is None else (f"{m[c]:.3f}" if isinstance(m[c], float) else str(m[c])) for c in cols]
            lines.append(f"| {name} | {mode} | " + " | ".join(vals) + " |")
    if report.get("regressions") is not None:
        lines += ["", "## Regressions vs baseline", ""]
        lines += [f"- {r}" for r in report["regressions"]] or ["- none"]
    return "\n".join(lines) + "\n"


def evaluate(cfg: Config, modes: list[str] | None = None, limit: int | None = None, label: str = "") -> dict:
    modes = modes or list(cfg.eval.modes)
    model = build_model(cfg.model, cfg)
    judge = build_judge(cfg)
    guide = tone.load_style_guide(cfg.runtime.tone.style_guide)
    assistant = Assistant(cfg, model) if "assistant" in modes else None
    out_dir = ensure_dir(Path(cfg.eval.output_dir) / (label or "latest"))
    report = {"model": cfg.model.model or cfg.model.path, "label": label, "results": {}}
    if not cfg.eval.tasks:
        raise ValueError("eval.tasks is empty: list one or more task JSONL files")
    for path in cfg.eval.tasks:
        name = Path(path).stem
        tasks = load_tasks(path, limit)
        by_id = {t.id: t for t in tasks}
        report["results"][name] = {}
        for mode in modes:
            with ThreadPoolExecutor(max_workers=max_workers(model, cfg.eval.workers)) as pool:
                rows = list(pool.map(lambda t: _run_one(mode, t, model, cfg, guide, assistant), tasks))
            report["results"][name][mode] = summarize(rows, by_id, judge, guide, mode)
            write_jsonl(out_dir / f"{name}.{mode}.outputs.jsonl", rows)
            print(f"[eval] {name}/{mode}: {report['results'][name][mode]}", flush=True)
    if cfg.eval.baseline_report and Path(cfg.eval.baseline_report).exists():
        report["regressions"] = compare(report, read_json(cfg.eval.baseline_report), cfg.eval.regression_tolerance)
    write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"[eval] report written to {out_dir}/report.md", flush=True)
    return report

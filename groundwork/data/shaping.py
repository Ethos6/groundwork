"""Turn traces, probe results and tone material into one fine-tuning mix."""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from .. import tone
from ..config import Config
from ..io import iter_jsonl, read_jsonl, write_json, write_jsonl
from ..messages import text_of
from ..models import build_data_source
from .filters import Decontaminator, filter_traces, select_best
from .scorers import _to_number, extract_answer, normalize
from .tasks import Task, load_tasks

_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*[:\-]\s*.+\s*$", re.I)


def _strip_answer_line(text: str) -> str:
    """Probe samples end with 'Answer: X' for scoring; make them read like normal replies."""
    body = _ANSWER_LINE_RE.sub("", text.strip()).strip()
    return body if body else extract_answer(text)


def still_correct(task: Task, text: str) -> bool:
    """Loose check that a rewritten reply still contains the gold answer."""
    if task.answer_type in ("exact", "contains"):
        t = normalize(text)
        return any(normalize(g) and re.search(rf"(?<!\w){re.escape(normalize(g))}(?!\w)", t) for g in task.gold)
    if task.answer_type == "numeric":
        nums = [_to_number(m) for m in re.findall(r"-?[\d,]*\.?\d+(?:/\d+)?", text)]
        golds = [_to_number(g) for g in task.gold]
        return any(n is not None and g is not None and abs(n - g) <= 1e-4 * max(1, abs(g)) for n in nums for g in golds)
    return True  # other types: keep the original check result


def _example(messages: list[dict], source: str, task_id: str = "") -> dict:
    return {"messages": messages, "source": source, "task_id": task_id}


def _maybe_system(sys_text: str, rng: random.Random, dropout: float) -> list[dict]:
    if sys_text and rng.random() >= dropout:
        return [{"role": "system", "content": sys_text}]
    return []


def build_closed_book(cfg: Config, tasks: dict[str, Task], probe: list[dict], guide, restyler, rng) -> list[dict]:
    sys_text = tone.system_prompt(guide)
    out = []
    for rec in probe:
        task = tasks.get(rec["task_id"])
        if task is None:
            continue
        cat = rec["category"]
        if cat in ("known", "partial"):
            correct = [s["text"] for s in rec["samples"] if s["score"] >= 1.0]
            if not correct:
                continue
            reply = _strip_answer_line(min(correct, key=len))
            if restyler is not None and guide is not None:
                styled = tone.rewrite(restyler, reply, guide, request=task.question)
                if still_correct(task, styled) and len(tone.check(styled, guide)) <= len(tone.check(reply, guide)):
                    reply = styled
            msgs = _maybe_system(sys_text, rng, cfg.data.system_prompt_dropout)
            msgs += [task.user_message(), {"role": "assistant", "content": reply}]
            out.append(_example(msgs, f"closed_book_{cat}", task.id))
        elif cat == "unknown" and cfg.data.abstain_when_unknown:
            text = guide.abstain_text() if guide else tone.StyleGuide().abstain_text()
            msgs = _maybe_system(sys_text, rng, cfg.data.system_prompt_dropout)
            msgs += [task.user_message(), {"role": "assistant", "content": text}]
            out.append(_example(msgs, "abstain", task.id))
    return out


def cap_abstentions(examples: list[dict], max_fraction: float, rng: random.Random) -> list[dict]:
    """Too many 'I'm not sure' examples teach over-refusal; keep them a small minority."""
    abstain = [e for e in examples if e["source"] == "abstain"]
    other = [e for e in examples if e["source"] != "abstain"]
    if not abstain or max_fraction >= 1:
        return examples
    allowed = int(max_fraction / max(1e-9, 1 - max_fraction) * len(other))
    rng.shuffle(abstain)
    return other + abstain[:allowed]


def build_tone_examples(cfg: Config, guide, rng) -> list[dict]:
    if not cfg.data.tone_examples:
        return []
    sys_text = tone.system_prompt(guide)
    out = []
    for row in read_jsonl(cfg.data.tone_examples):
        problems = tone.check(row["response"], guide)
        if problems:
            print(f"[tone] skipping example that breaks the style guide ({problems[0]}): {row['prompt'][:60]!r}")
            continue
        msgs = _maybe_system(sys_text, rng, cfg.data.system_prompt_dropout)
        msgs += [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}]
        out.append(_example(msgs, "tone"))
    return out


def load_replay(cfg: Config, n_new: int, rng) -> list[dict]:
    if not cfg.data.replay or cfg.data.replay_fraction <= 0:
        return []
    rows = [r for r in iter_jsonl(cfg.data.replay) if r.get("messages")]
    want = int(cfg.data.replay_fraction / max(1e-9, 1 - cfg.data.replay_fraction) * n_new)
    rng.shuffle(rows)
    return [_example(r["messages"], "replay") for r in rows[:want]]


def build_sft(cfg: Config, out_path: str | None = None) -> dict:
    """Assemble ``<work_dir>/sft.jsonl`` from whatever earlier stages produced."""
    work = Path(cfg.data.work_dir)
    rng = random.Random(cfg.data.seed)
    tasks = {t.id: t for t in load_tasks(cfg.data.tasks)}
    guide = tone.load_style_guide(cfg.runtime.tone.style_guide)
    stats: dict = {}
    examples: list[dict] = []

    traces_path = work / "traces.jsonl"
    if traces_path.exists():
        kept, fstats = filter_traces(read_jsonl(traces_path), cfg.data.keep_recovered_errors)
        best = select_best(kept, cfg.data.keep_per_task)
        stats["traces"] = {**fstats, "selected": len(best)}
        examples += [_example(r["messages"], "trace", r["task_id"]) for r in best]

    probe_path = work / "probe.jsonl"
    if probe_path.exists():
        probe = read_jsonl(probe_path)
        stats["probe"] = dict(Counter(r["category"] for r in probe))
        restyler = None
        if guide is not None and cfg.data.restyle_answers:
            restyler = build_data_source(cfg, "model", "tone restyling")
        examples += build_closed_book(cfg, tasks, probe, guide, restyler, rng)

    examples += build_tone_examples(cfg, guide, rng)
    examples = cap_abstentions(examples, cfg.data.max_abstain_fraction, rng)

    if cfg.data.decontaminate_against:
        questions = []
        for p in cfg.data.decontaminate_against:
            questions += [t.question for t in load_tasks(p)]
        examples, dropped = Decontaminator(questions).filter_examples(examples)
        stats["decontaminated"] = dropped

    examples += load_replay(cfg, len(examples), rng)
    rng.shuffle(examples)
    out_path = out_path or str(work / "sft.jsonl")
    write_jsonl(out_path, examples)
    stats["by_source"] = dict(Counter(e["source"] for e in examples))
    stats["total"] = len(examples)
    stats["avg_assistant_chars"] = round(
        sum(len(text_of(m)) for e in examples for m in e["messages"] if m["role"] == "assistant") / max(1, len(examples)), 1
    )
    write_json(work / "sft_stats.json", stats)
    return stats

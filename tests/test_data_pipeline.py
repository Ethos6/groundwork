import json
import random

from groundwork.config import load_config
from groundwork.data.filters import Decontaminator, filter_traces, select_best
from groundwork.data.probe import categorize
from groundwork.data.shaping import _strip_answer_line, build_sft, cap_abstentions, still_correct
from groundwork.data.tasks import Task


def _trace(task_id, score=1.0, parse=0, exec_=0, max_steps=False, steps=2, final="x", attempt=0, text="a"):
    return {"task_id": task_id, "score": score, "n_parse_errors": parse, "n_exec_errors": exec_,
            "max_steps_reached": max_steps, "n_steps": steps, "final_answer": final, "attempt": attempt,
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": text}]}


def test_filter_and_select():
    recs = [
        _trace("a"), _trace("a", steps=1, text="b"), _trace("b", score=0.0), _trace("c", parse=1),
        _trace("d", max_steps=True), _trace("e", exec_=1), _trace("f", final=None),
    ]
    kept, stats = filter_traces(recs)
    assert {r["task_id"] for r in kept} == {"a", "e"}
    assert stats["dropped_parse_error"] == 1 and stats["dropped_max_steps"] == 1 and stats["kept_with_recovery"] == 1
    kept_strict, _ = filter_traces(recs, keep_recovered_errors=False)
    assert {r["task_id"] for r in kept_strict} == {"a"}
    best = select_best(kept, per_task=1)
    assert [r["n_steps"] for r in best if r["task_id"] == "a"] == [1]


def test_categorize_and_cap():
    assert categorize(0.8, 0.6, 0.2) == "known"
    assert categorize(0.4, 0.6, 0.2) == "partial"
    assert categorize(0.0, 0.6, 0.2) == "unknown"
    ex = [{"source": "abstain"}] * 50 + [{"source": "trace"}] * 85
    capped = cap_abstentions(ex, 0.15, random.Random(0))
    n_abs = sum(e["source"] == "abstain" for e in capped)
    assert n_abs == 15 and n_abs / len(capped) <= 0.15


def test_decontamination():
    d = Decontaminator(["What is the capital city of France and when was its famous tower built exactly?"])
    assert d.is_contaminated("what is the capital city of France and when was its famous tower built exactly")
    assert not d.is_contaminated("What is the capital of Peru?")


def test_answer_line_and_still_correct():
    assert _strip_answer_line("Kilimanjaro is the highest.\nAnswer: Kilimanjaro") == "Kilimanjaro is the highest."
    assert _strip_answer_line("Answer: 1991") == "1991"
    t = Task(id="t", question="q", answer="4", answer_type="exact")
    assert still_correct(t, "It is 4.")
    assert not still_correct(t, "It is 14.")
    n = Task(id="n", question="q", answer="1642", answer_type="numeric")
    assert still_correct(n, "about 1,642 metres")


def test_build_sft_end_to_end(tmp_path, root):
    work = tmp_path / "work"
    work.mkdir()
    tasks = tmp_path / "tasks.jsonl"
    rows = [
        {"id": "k", "question": "Known fact?", "answer": "yes", "closed_book": True},
        {"id": "u", "question": "Unknown fact?", "answer": "zzz", "closed_book": True},
        {"id": "m", "question": "Math task?", "answer": "42", "answer_type": "numeric"},
    ]
    tasks.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (work / "traces.jsonl").write_text(json.dumps(_trace("m", final="42")) + "\n")
    probe = [
        {"task_id": "k", "category": "known", "p_correct": 1.0, "samples": [{"text": "Yes it is.\nAnswer: yes", "score": 1.0}]},
        {"task_id": "u", "category": "unknown", "p_correct": 0.0, "samples": [{"text": "Answer: abc", "score": 0.0}]},
    ]
    (work / "probe.jsonl").write_text("\n".join(json.dumps(p) for p in probe) + "\n")
    cfg = load_config(None, [
        f"data.tasks={tasks}", f"data.work_dir={work}", "data.restyle_answers=false",
        f"data.tone_examples={root / 'examples/tone_examples.jsonl'}",
        f"runtime.tone.style_guide={root / 'examples/style_guide.yaml'}", "data.max_abstain_fraction=0.5",
    ])
    stats = build_sft(cfg)
    assert stats["by_source"]["trace"] == 1
    assert stats["by_source"]["closed_book_known"] == 1
    assert stats["by_source"]["abstain"] == 1
    assert stats["by_source"]["tone"] == 3
    lines = [json.loads(l) for l in (work / "sft.jsonl").read_text().splitlines()]
    known = next(l for l in lines if l["source"] == "closed_book_known")
    assert known["messages"][-1]["content"] == "Yes it is."
    abstain = next(l for l in lines if l["source"] == "abstain")
    assert "rather not guess" in abstain["messages"][-1]["content"]

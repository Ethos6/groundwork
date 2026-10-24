"""End-to-end runs of generate -> probe -> build -> eval with a scripted model (no GPU, no network)."""

import json

import pytest

from groundwork.config import load_config


def _model_fn(messages, i, n=1, **k):
    text = json.dumps(messages)
    if "Answer: <short answer>" in text:  # knowledge probe / guard
        if "highest mountain" in text:
            return ["Kilimanjaro, in Tanzania.\nAnswer: Kilimanjaro"] * n
        return [f"Answer: guess{j}" for j in range(n)]
    if "Reply with one word: TOOLS or DIRECT" in text:
        return "TOOLS"
    if "Reply with one word: YES or NO" in text:
        return "YES"
    if "Notes from your work" in text:
        return "It costs $42.50.\nAnswer: 42.5"
    if messages[0]["role"] == "system" and "writing Python code" in messages[0]["content"]:
        n_steps = sum(m["role"] == "assistant" for m in messages)
        if "notebooks" in text:
            return ["Thought: price per notebook times 17\nCode:\n```py\nprint(7.5 / 3 * 17)\n```",
                    "Thought: done\nCode:\n```py\nfinal_answer(42.5)\n```"][min(n_steps, 1)]
        return "Thought: not sure\nCode:\n```py\nfinal_answer('unknown')\n```"
    return "Draft."


@pytest.fixture
def cfg(tmp_path, root):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps(r) for r in [
        {"id": "m1", "question": "A shop sells 3 notebooks for $7.50. How much do 17 notebooks cost?", "answer": "42.5", "answer_type": "numeric"},
        {"id": "f1", "question": "What is the highest mountain in Africa?", "answer": "Kilimanjaro", "answer_type": "contains", "closed_book": True},
        {"id": "f2", "question": "Who won the 1931 village fair pie contest?", "answer": "Ada", "answer_type": "exact", "closed_book": True},
    ]) + "\n")
    return load_config(None, [
        f"data.tasks={tasks}", f"data.work_dir={tmp_path / 'work'}", "data.attempts_per_task=2", "data.workers=1",
        "data.restyle_answers=false", "data.max_abstain_fraction=0.5", "agent.tools=[python]",
        f"runtime.tone.style_guide={root / 'examples/style_guide.yaml'}",
        f"eval.tasks=[{tasks}]", f"eval.output_dir={tmp_path / 'reports'}", "eval.workers=1",
        "runtime.guard.verify_with_agent=false",
    ])


def test_full_pipeline(cfg, fake_model, monkeypatch):
    import groundwork.data.generate as gen
    import groundwork.data.probe as probe
    import groundwork.eval.run as ev
    from groundwork.data.shaping import build_sft
    from groundwork.data.tasks import load_tasks

    model = fake_model(_model_fn, name="scripted")
    for mod in (gen, probe):
        monkeypatch.setattr(mod, "build_data_source", lambda *a, **k: model)
    monkeypatch.setattr(ev, "build_model", lambda *a, **k: model)

    tasks = load_tasks(cfg.data.tasks)
    traces_path = gen.generate_traces(cfg, tasks)
    traces = [json.loads(l) for l in open(traces_path)]
    assert len(traces) == 6
    assert {t["task_id"] for t in traces if t["score"] == 1.0} == {"m1"}
    # resume: nothing is regenerated
    gen.generate_traces(cfg, tasks)
    assert len(open(traces_path).read().splitlines()) == 6

    results = {r["task_id"]: r["category"] for r in probe.run_probe(cfg, tasks)}
    assert results == {"f1": "known", "f2": "unknown"}

    stats = build_sft(cfg)
    assert stats["by_source"] == {"trace": 1, "closed_book_known": 1, "abstain": 1}

    report = ev.evaluate(cfg, ["direct", "agent", "assistant"], label="test")
    agent = report["results"]["tasks"]["agent"]
    assert agent["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert agent["hallucination_rate"] == 0.0  # "unknown" counts as abstaining, not hallucinating
    assert (cfg.eval.output_dir and (__import__("pathlib").Path(cfg.eval.output_dir) / "test" / "report.md").exists())


def test_compliance_blocks_pipeline_before_any_call(cfg):
    from groundwork.compliance import ComplianceError
    from groundwork.data.generate import generate_traces
    from groundwork.data.tasks import load_tasks

    cfg.model.base_url = "https://api.anthropic.com/v1"
    cfg.model.model = "some-model"
    with pytest.raises(ComplianceError):
        generate_traces(cfg, load_tasks(cfg.data.tasks))


def test_chatml_template_masks_headers(tmp_path):
    pytest.importorskip("transformers")
    from conftest import make_tiny_tokenizer
    from groundwork.train.encode import IGNORE, ChatEncoder

    tok, _ = make_tiny_tokenizer(tmp_path)
    tok.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    enc = ChatEncoder(tok).encode([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi world"}])
    trained = [tok.convert_ids_to_tokens(t) for t, l in zip(enc["input_ids"], enc["labels"]) if l != IGNORE]
    assert trained == ["hi", "world", "<|im_end|>"]

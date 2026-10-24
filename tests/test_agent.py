from groundwork.agent.runner import AgentRunner
from groundwork.agent.tools import BM25Index, PythonExecutor, Toolbox
from groundwork.config import AgentConfig


def _cfg(tmp_path, **kw):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"1","title":"Kilimanjaro","text":"Kilimanjaro is the highest mountain in Africa."}\n'
                      '{"id":"2","title":"Baikal","text":"Lake Baikal is the deepest lake."}\n')
    (tmp_path / "note.txt").write_text("secret number: 7")
    base = dict(corpus=str(corpus), files_root=str(tmp_path), python_timeout_s=5, max_steps=4)
    base.update(kw)
    return AgentConfig(**base)


def test_bm25_ranks_relevant_doc_first():
    idx = BM25Index([{"title": "A", "text": "cats and dogs"}, {"title": "B", "text": "highest mountain in Africa"}])
    assert idx.search("Africa mountain", 1)[0]["title"] == "B"


def test_executor_state_tools_errors_timeout(tmp_path):
    ex = PythonExecutor(Toolbox(_cfg(tmp_path)), timeout_s=3)
    try:
        assert ex.run("x = 21").error is None
        assert ex.run("print(x * 2)").stdout.strip() == "42"
        assert "Kilimanjaro" in ex.run("print(search('highest mountain Africa'))").stdout
        assert ex.run("print(read_file('note.txt'))").stdout.strip() == "secret number: 7"
        bad = ex.run("read_file('../../etc/passwd')")
        assert bad.error and "outside" in bad.error
        err = ex.run("1/0")
        assert "ZeroDivisionError" in err.error
        assert ex.run("final_answer(x + 1)").final == "22"
        slow = ex.run("while True: pass")
        assert slow.timed_out
        assert ex.run("print('alive')").stdout.strip() == "alive"  # restarted after timeout
    finally:
        ex.close()


def test_agent_recovers_from_error_and_finishes(tmp_path, fake_model):
    script = [
        "Thought: compute\nCode:\n```py\ntotal = 17 * 7.5 / 3\nprint(totl)\n```",
        "Thought: fix the typo\nCode:\n```py\nprint(total)\n```",
        "Thought: done\nCode:\n```py\nfinal_answer(total)\n```",
    ]
    model = fake_model(lambda msgs, i, **k: script[i])
    traj = AgentRunner(model, _cfg(tmp_path)).run({"role": "user", "content": "17 notebooks?"}, "t1")
    assert traj.final_answer == "42.5"
    assert traj.n_exec_errors == 1 and traj.n_parse_errors == 0
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "NameError" in traj.messages[3]["content"]


def test_agent_parse_error_and_max_steps(tmp_path, fake_model):
    model = fake_model(lambda msgs, i, **k: "I think the answer is 5")
    traj = AgentRunner(model, _cfg(tmp_path, max_steps=2)).run({"role": "user", "content": "q"})
    assert traj.final_answer is None and traj.max_steps_reached
    assert traj.n_parse_errors == 3  # 2 steps + the wrap-up step


def test_self_consistency_picks_majority(tmp_path, fake_model):
    cands = [
        "Thought: a\nCode:\n```py\nfinal_answer(41)\n```",
        "Thought: b\nCode:\n```py\nfinal_answer(42)\n```",
        "Thought: c\nCode:\n```py\nfinal_answer(42)\n```",
        "Thought: d\nCode:\n```py\nboom(\n```",
    ]
    model = fake_model(lambda msgs, i, n=1, **k: cands)
    traj = AgentRunner(model, _cfg(tmp_path, self_consistency_n=4)).run({"role": "user", "content": "q"})
    assert traj.final_answer == "42"
    assert traj.steps[0].candidates == 4


def test_first_thought_prefix_is_passed_on_first_step(tmp_path, fake_model):
    model = fake_model(lambda msgs, i, prefill=None, **k: (prefill or "") + "Code:\n```py\nfinal_answer('ok')\n```")
    traj = AgentRunner(model, _cfg(tmp_path)).run({"role": "user", "content": "q"}, prefix="Thought: plan first\n")
    assert model.calls[0]["prefill"] == "Thought: plan first\n"
    assert traj.steps[0].thought == "plan first"

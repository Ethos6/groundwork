import json
import threading
import urllib.request

from groundwork.config import load_config
from groundwork.runtime.assistant import Assistant
from groundwork.runtime.guard import HallucinationGuard
from groundwork.runtime.server import ThreadingHTTPServer, make_handler
from groundwork.tone import StyleGuide, check, load_style_guide, score


def _router(answers, route="DIRECT", draft="Draft reply.", agent_steps=None, rewrite=None):
    """Fake model that recognizes the different internal prompts by their text."""
    agent_steps = list(agent_steps or [])

    def fn(messages, i, n=1, **k):
        text = json.dumps(messages)
        if "Reply with one word: TOOLS or DIRECT" in text:
            return route
        if "Reply with one word: YES or NO" in text:
            return "YES"
        if "Answer: <short answer>" in text:
            return answers
        if "Rewrite the reply below" in text:
            return rewrite or "Rewritten."
        if "Notes from your work" in text:
            return "Composed reply: 42."
        if messages and messages[0]["role"] == "system" and "writing Python code" in messages[0]["content"]:
            return agent_steps.pop(0) if agent_steps else "Thought: x\nCode:\n```py\nfinal_answer('42')\n```"
        return draft

    return fn


def _cfg(root, **sets):
    base = [f"runtime.tone.style_guide={root / 'examples/style_guide.yaml'}", "agent.tools=[python]"]
    return load_config(None, base + [f"{k}={v}" for k, v in sets.items()])


def test_guard_agreement(fake_model):
    from groundwork.config import GuardConfig

    g = HallucinationGuard(fake_model(lambda m, i, **k: ["Answer: Paris", "Answer: paris", "Answer: Lyon", "Answer: Paris", "Answer: unknown"]), GuardConfig(fact_check_gate=False))
    r = g.check([{"role": "user", "content": "capital of France?"}])
    assert r.answer == "paris" and abs(r.agreement - 0.6) < 1e-9


def test_direct_answer_kept_when_consistent(root, fake_model):
    model = fake_model(_router(["Answer: 1991"] * 5, draft="Python was first released in 1991."))
    reply = Assistant(_cfg(root), model).respond([{"role": "user", "content": "When was Python released?"}])
    assert reply.route == "direct" and "1991" in reply.text


def test_inconsistent_answer_is_verified_with_agent(root, fake_model):
    model = fake_model(_router(["Answer: 1", "Answer: 2", "Answer: 3", "Answer: 4", "Answer: 5"]))
    reply = Assistant(_cfg(root), model).respond([{"role": "user", "content": "Obscure fact?"}])
    assert reply.route == "verified" and reply.text == "Composed reply: 42."


def test_inconsistent_answer_abstains_without_tools(root, fake_model):
    model = fake_model(_router(["Answer: 1", "Answer: 2", "Answer: 3", "Answer: 4", "Answer: 5"]))
    cfg = _cfg(root, **{"runtime.guard.verify_with_agent": "false"})
    reply = Assistant(cfg, model).respond([{"role": "user", "content": "Obscure fact?"}])
    assert reply.route == "abstain" and "rather not guess" in reply.text


def test_agent_route_and_tone_rewrite(root, fake_model):
    model = fake_model(_router([], route="TOOLS"))
    reply = Assistant(_cfg(root), model).respond([{"role": "user", "content": "Compute something"}])
    assert reply.route == "agent" and reply.text == "Composed reply: 42."
    model2 = fake_model(_router(["Answer: x"] * 5, draft="Great question! Here you go.", rewrite="Here you go."))
    reply2 = Assistant(_cfg(root), model2).respond([{"role": "user", "content": "Tell me"}])
    assert reply2.text == "Here you go." and reply2.rewrites == 1 and reply2.tone_problems == []


def test_tone_rules(root):
    guide = load_style_guide(root / "examples/style_guide.yaml")
    assert check("Great question! 🙂 " + "word " * 400, guide) == [
        "uses banned phrase: 'Great question'", "too long: 403 words (limit 350)", "contains emoji"]
    assert score("Plain answer.", guide) == 1.0
    assert StyleGuide().abstain_text().startswith("I'm not sure")


def test_server_roundtrip(root, fake_model):
    model = fake_model(_router(["Answer: ok"] * 5, draft="Hello there."))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Assistant(_cfg(root), model), "m"))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        port = server.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert body["choices"][0]["message"]["content"] == "Hello there."
        assert body["groundwork"]["route"] == "direct"
    finally:
        server.shutdown()

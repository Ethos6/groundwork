"""Inference-time layer that wraps any chat model. No training required.

    request ──> route ──┬─> direct answer ──> hallucination guard ──┬─> keep
                        │                                           ├─> verify with tools
                        │                                           └─> say "not sure"
                        └─> tool-using agent ──> compose reply
                                   all replies ──> tone check ──> rewrite if needed
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from .. import tone
from ..agent.runner import AgentRunner
from ..config import Config
from ..messages import last_user, parts, text_of, with_system
from ..models import build_model
from .guard import GuardResult, HallucinationGuard

ROUTE_PROMPT = """Decide how to handle the request below.
Reply TOOLS if doing it well needs calculation, code, several steps of work, looking up facts, \
or reading an attached file. Reply DIRECT if it can be answered well in one go.
Reply with one word: TOOLS or DIRECT.

Request:
{request}"""

COMPOSE_PROMPT = """Notes from your work on the request above (the user can't see these):
{notes}

Result: {final}

Write your reply to the user. Use only what the work established, and say plainly if \
anything is still uncertain. Don't mention the notes or the tools."""


@dataclass
class Reply:
    text: str
    route: str
    guard: GuardResult | None = None
    trajectory: dict | None = None
    tone_problems: list = field(default_factory=list)
    rewrites: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.trajectory is not None:
            d["trajectory"] = {k: self.trajectory[k] for k in ("final_answer", "n_steps", "n_exec_errors", "max_steps_reached")}
        return d


class Assistant:
    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.model = model or build_model(cfg.model, cfg)
        self.guide = tone.load_style_guide(cfg.runtime.tone.style_guide)
        self.tone_system = tone.system_prompt(self.guide)
        self.runner = AgentRunner(self.model, cfg.agent, cfg.data.media_root, extra_system=self.tone_system)
        self.guard = HallucinationGuard(self.model, cfg.runtime.guard)

    # ------------------------------------------------------------------ pieces
    def abstain_text(self) -> str:
        return (self.guide or tone.StyleGuide()).abstain_text()

    def route(self, messages: list[dict]) -> str:
        mode = self.cfg.runtime.agent_mode
        if mode in ("always", "never"):
            return "agent" if mode == "always" else "direct"
        request = text_of(last_user(messages) or {"content": ""})
        out = self.model.chat([{"role": "user", "content": ROUTE_PROMPT.format(request=request)}],
                              temperature=0.0, max_tokens=3)[0]
        return "agent" if out.strip().upper().startswith("T") else "direct"

    def _agent_input(self, messages: list[dict]) -> dict:
        """The agent works on the latest request; earlier turns are given as context."""
        last = copy.deepcopy(last_user(messages) or {"role": "user", "content": ""})
        history = [m for m in messages if m is not None and m.get("role") in ("user", "assistant")]
        if len(history) > 1:
            lines = [f"{m['role']}: {text_of(m)}" for m in history[:-1][-6:]]
            ps = parts(last.get("content"))
            ps.insert(0, {"type": "text", "text": "Conversation so far:\n" + "\n".join(lines) + "\n\nCurrent request:"})
            last["content"] = ps
        return last

    def _run_agent(self, messages: list[dict]) -> tuple[str | None, dict]:
        traj = self.runner.run(self._agent_input(messages))
        record = traj.to_dict()
        if traj.final_answer is None:
            return None, record
        if not self.cfg.runtime.compose_reply:
            return traj.final_answer, record
        notes = []
        for s in traj.steps[-4:]:
            if s.thought:
                notes.append(f"- thought: {s.thought[:400]}")
            if s.observation:
                notes.append(f"  observed: {s.observation[:600]}")
        compose = with_system(messages, self.tone_system) + [
            {"role": "user", "content": COMPOSE_PROMPT.format(notes="\n".join(notes) or "-", final=traj.final_answer)}
        ]
        return self.model.chat(compose, temperature=0.3)[0].strip() or traj.final_answer, record

    def _tone_pass(self, reply: Reply, request: str) -> Reply:
        if self.guide is None:
            return reply
        problems = tone.check(reply.text, self.guide)
        tries = 0
        while problems and self.cfg.runtime.tone.rewrite and tries < self.cfg.runtime.tone.max_rewrites:
            candidate = tone.rewrite(self.model, reply.text, self.guide, request, problems)
            tries += 1
            new_problems = tone.check(candidate, self.guide)
            if len(new_problems) < len(problems):
                reply.text, problems = candidate, new_problems
        reply.tone_problems, reply.rewrites = problems, tries
        return reply

    # ------------------------------------------------------------------ main entry
    def respond(self, messages: list[dict]) -> Reply:
        request = text_of(last_user(messages) or {"content": ""})
        has_tools = bool(self.runner.toolbox.tools) or "python" in self.cfg.agent.tools
        route = self.route(messages) if has_tools else "direct"

        if route == "agent":
            text, traj = self._run_agent(messages)
            reply = Reply(text or self.abstain_text(), "agent" if text else "abstain", trajectory=traj)
            return self._tone_pass(reply, request)

        draft = self.model.chat(with_system(messages, self.tone_system), temperature=0.3)[0].strip()
        g = self.guard.check(messages)
        if not g.checked or g.agreement >= self.cfg.runtime.guard.accept_agreement:
            return self._tone_pass(Reply(draft, "direct", guard=g), request)

        if self.cfg.runtime.guard.verify_with_agent and has_tools:
            text, traj = self._run_agent(messages)
            if text:
                return self._tone_pass(Reply(text, "verified", guard=g, trajectory=traj), request)
        return self._tone_pass(Reply(self.abstain_text(), "abstain", guard=g), request)

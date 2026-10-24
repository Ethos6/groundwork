"""The agent loop: Thought -> Code -> Observation, repeated until final_answer().

Two techniques from agent-distillation research are built in:

* first-thought prefix: start the first step from a short plan (``prefix``), which makes
  runs on hard tasks noticeably more reliable;
* self-consistent action selection: sample several candidate steps, drop the ones that
  fail to parse or crash, and keep the one whose execution result most candidates agree on.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..config import AgentConfig
from ..messages import with_system
from ..models.base import ChatModel
from . import prompts
from .parser import ParseError, ParsedStep, parse_step
from .tools import ExecResult, PythonExecutor, Toolbox

STOP = ["<end_code>", "\nObservation:", "Observation:\n"]


@dataclass
class Step:
    raw: str
    thought: str = ""
    code: str = ""
    observation: str = ""
    error: str | None = None  # None | "parse" | "exec" | "timeout"
    candidates: int = 1


@dataclass
class Trajectory:
    task_id: str
    actor: str
    messages: list[dict]
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    max_steps_reached: bool = False
    prefix: str | None = None

    @property
    def n_parse_errors(self) -> int:
        return sum(s.error == "parse" for s in self.steps)

    @property
    def n_exec_errors(self) -> int:
        return sum(s.error in ("exec", "timeout") for s in self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(n_parse_errors=self.n_parse_errors, n_exec_errors=self.n_exec_errors, n_steps=len(self.steps))
        return d


def _normalize_obs(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


class AgentRunner:
    def __init__(self, model: ChatModel, cfg: AgentConfig, media_root: str = ".", extra_system: str = ""):
        self.model = model
        self.cfg = cfg
        self.toolbox = Toolbox(cfg, media_root)
        self.extra_system = extra_system

    def system_prompt(self) -> str:
        rule = prompts.FACT_RULE_WITH_SEARCH if "search" in self.toolbox.tools else prompts.FACT_RULE_NO_SEARCH
        text = prompts.AGENT_SYSTEM.format(tools=self.toolbox.describe(), fact_rule=rule)
        return text + ("\n\n" + self.extra_system if self.extra_system else "")

    # ------------------------------------------------------------------ helpers
    def _observation(self, res: ExecResult) -> tuple[str, str | None]:
        out = res.stdout.strip()
        if res.error:
            kind = "timeout" if res.timed_out else "exec"
            text = (out + "\n" if out else "") + f"Code execution failed:\n{res.error}"
            return self._clip(text), kind
        if res.final is not None:
            return self._clip((out + "\n" if out else "") + f"Final answer: {res.final}"), None
        return self._clip(out or "(no output printed)"), None

    def _clip(self, text: str) -> str:
        limit = self.cfg.max_observation_chars
        return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} characters truncated]"

    def _choose(self, raws: list[str], history: list[str]) -> tuple[str, ParsedStep | None, str | None]:
        """Pick one candidate step. Returns (raw, parsed or None, parse error)."""
        parsed: list[tuple[str, ParsedStep]] = []
        first_err = None
        for raw in raws:
            try:
                parsed.append((raw, parse_step(raw)))
            except ParseError as e:
                first_err = first_err or str(e)
        if not parsed:
            return raws[0], None, first_err
        if len(parsed) == 1:
            return parsed[0][0], parsed[0][1], None
        # Dry-run each candidate on a copy of the interpreter state and vote on results.
        groups: dict[str, list[int]] = defaultdict(list)
        probe = PythonExecutor(self.toolbox, self.cfg.python_timeout_s, self.cfg.python_memory_mb)
        try:
            for i, (_, step) in enumerate(parsed):
                probe.replay(history)
                res = probe.run(step.code)
                if res.error:
                    continue
                groups[_normalize_obs(res.stdout + "|" + str(res.final))].append(i)
        finally:
            probe.close()
        if not groups:
            return parsed[0][0], parsed[0][1], None
        best = max(groups.values(), key=len)
        idx = best[0] if len(best) > 1 else min(i for g in groups.values() for i in g)
        return parsed[idx][0], parsed[idx][1], None

    # ------------------------------------------------------------------ main loop
    def run(self, user_message: dict, task_id: str = "", actor: str = "", prefix: str | None = None,
            temperature: float | None = None) -> Trajectory:
        temp = self.cfg.temperature if temperature is None else temperature
        messages = with_system([copy.deepcopy(user_message)], self.system_prompt())
        traj = Trajectory(task_id=task_id, actor=actor or self.model.name, messages=messages, prefix=prefix)
        executor = PythonExecutor(self.toolbox, self.cfg.python_timeout_s, self.cfg.python_memory_mb)
        accepted_code: list[str] = []
        try:
            for i in range(self.cfg.max_steps + (1 if self.cfg.force_final_on_max_steps else 0)):
                if i == self.cfg.max_steps:  # one extra step to wrap up
                    traj.max_steps_reached = True
                    messages.append({"role": "user", "content": prompts.MAX_STEPS_NUDGE})
                n = max(1, self.cfg.self_consistency_n)
                step_prefix = prefix if (i == 0 and prefix) else None
                raws = self.model.chat(
                    messages, n=n, temperature=self.cfg.self_consistency_temperature if n > 1 else temp,
                    stop=STOP, prefill=step_prefix,
                )
                raw, parsed, perr = self._choose(raws, accepted_code)
                step = Step(raw=raw, candidates=len(raws))
                if parsed is None:
                    step.error = "parse"
                    step.observation = perr or "Error in code parsing"
                    messages.append({"role": "assistant", "content": raw.strip()})
                    messages.append({"role": "tool", "content": "Observation:\n" + step.observation})
                    traj.steps.append(step)
                    continue
                step.thought, step.code = parsed.thought, parsed.code
                res = executor.run(parsed.code)
                step.observation, step.error = self._observation(res)
                messages.append({"role": "assistant", "content": parsed.render()})
                traj.steps.append(step)
                if step.error is None:
                    accepted_code.append(parsed.code)
                if res.final is not None and not res.error:
                    traj.final_answer = res.final
                    break
                messages.append({"role": "tool", "content": "Observation:\n" + step.observation})
            else:
                traj.max_steps_reached = True
        finally:
            executor.close()
        return traj

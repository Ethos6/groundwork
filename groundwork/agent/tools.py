"""Tools available to the agent and the process that executes its code.

The agent writes Python; tool functions (search, read_file, ...) are called from that code.
Code runs in a separate worker process with a time and memory limit. Tool calls are
forwarded back to this process, so indexes and file access stay under our control.

The worker is *not* a security sandbox. Run untrusted workloads inside a container.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .. import media
from ..config import AgentConfig
from ..io import read_jsonl

WORKER_PATH = Path(__file__).with_name("_worker.py")


# --------------------------------------------------------------------------- search
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Small in-memory BM25 index over a JSONL corpus of {"id", "title", "text"}."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs: list[Counter] = []
        df: Counter = Counter()
        lengths = []
        for d in docs:
            toks = _tokens(f"{d.get('title', '')} {d.get('text', '')}")
            tf = Counter(toks)
            self.tfs.append(tf)
            df.update(tf.keys())
            lengths.append(len(toks))
        self.lengths = lengths
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_jsonl(cls, path: str) -> "BM25Index":
        return cls(read_jsonl(path))

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tokens(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:k]]


def format_results(results: list[dict], max_chars: int = 800) -> str:
    if not results:
        return "No results found."
    blocks = []
    for i, d in enumerate(results, 1):
        text = str(d.get("text", ""))
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        title = d.get("title") or d.get("id") or f"result {i}"
        blocks.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- toolbox
@dataclass
class Tool:
    name: str
    signature: str
    doc: str
    fn: Callable


class Toolbox:
    def __init__(self, cfg: AgentConfig, media_root: str = "."):
        self.cfg = cfg
        self.tools: dict[str, Tool] = {}
        self._index: BM25Index | None = None
        for name in cfg.tools:
            if name == "python":
                continue  # code execution itself, not a callable tool
            if name == "search":
                if not (cfg.corpus or cfg.search_url):
                    continue  # nothing to search; don't advertise the tool
                self.tools["search"] = Tool(
                    "search", "search(query: str) -> str",
                    "Look up facts. Returns the most relevant passages. Use it whenever you are not sure of a fact.",
                    self._search,
                )
            elif name == "read_file":
                self.tools["read_file"] = Tool(
                    "read_file", "read_file(path: str) -> str",
                    "Return the text content of an attached or local file (txt, md, csv, json, pdf, docx).",
                    self._read_file,
                )
            else:
                raise ValueError(f"Unknown tool {name!r}; available: python, search, read_file")
        self.files_root = Path(cfg.files_root or media_root).resolve()

    # tools -----------------------------------------------------------------
    def _search(self, query: str, k: int | None = None) -> str:
        k = int(k or self.cfg.search_top_k)
        if self.cfg.search_url:
            r = requests.post(self.cfg.search_url, json={"query": query, "top_k": k}, timeout=30)
            r.raise_for_status()
            return format_results(r.json().get("results", []))
        if self._index is None:
            self._index = BM25Index.from_jsonl(self.cfg.corpus)
        return format_results(self._index.search(query, k))

    def _read_file(self, path: str) -> str:
        full = (self.files_root / path).resolve()
        if os.path.commonpath([str(full), str(self.files_root)]) != str(self.files_root):
            raise PermissionError(f"{path} is outside the allowed files directory")
        if not full.exists():
            raise FileNotFoundError(path)
        return media.read_document(str(full), ".")

    def call(self, name: str, args: list, kwargs: dict) -> str:
        if name not in self.tools:
            raise KeyError(f"unknown tool {name}")
        result = self.tools[name].fn(*args, **kwargs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def describe(self) -> str:
        lines = ["- final_answer(answer) -> None: Finish the task and return the answer to the user."]
        for t in self.tools.values():
            lines.append(f"- {t.signature}: {t.doc}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- executor
@dataclass
class ExecResult:
    stdout: str
    error: str | None
    final: str | None
    timed_out: bool = False


class PythonExecutor:
    """Persistent Python worker: variables survive between steps of one task."""

    def __init__(self, toolbox: Toolbox, timeout_s: float = 10.0, memory_mb: int = 1024):
        self.toolbox = toolbox
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.proc: subprocess.Popen | None = None

    def _start(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(WORKER_PATH)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, cwd=str(self.toolbox.files_root),
        )
        self._write({"op": "init", "tools": list(self.toolbox.tools), "memory_mb": self.memory_mb})
        msg = self._read(deadline=time.monotonic() + 30)
        if not msg or msg.get("type") != "ready":
            raise RuntimeError("python worker failed to start")

    def _write(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read(self, deadline: float) -> dict | None:
        assert self.proc and self.proc.stdout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if ready:
                line = self.proc.stdout.readline()
                if not line:
                    return {"type": "crashed"}
                return json.loads(line)

    def run(self, code: str) -> ExecResult:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self._write({"op": "exec", "code": code})
        deadline = time.monotonic() + self.timeout_s
        while True:
            msg = self._read(deadline)
            if msg is None:
                self.close()
                return ExecResult("", f"Timed out after {self.timeout_s:.0f}s; interpreter state was reset.", None, True)
            kind = msg.get("type")
            if kind == "crashed":
                self.close()
                return ExecResult("", "The interpreter crashed (possibly out of memory); state was reset.", None)
            if kind == "tool_call":
                try:
                    value = self.toolbox.call(msg["name"], msg.get("args", []), msg.get("kwargs", {}))
                    self._write({"type": "tool_result", "ok": True, "value": value})
                except Exception as e:  # report tool errors to the agent's code
                    self._write({"type": "tool_result", "ok": False, "value": f"{type(e).__name__}: {e}"})
                deadline = max(deadline, time.monotonic() + self.timeout_s / 2)  # tool time doesn't count fully
            elif kind == "done":
                final = msg.get("final")
                return ExecResult(msg.get("stdout", ""), msg.get("error"), final["value"] if final else None)

    def replay(self, codes: list[str]) -> None:
        """Restart and re-run earlier code so state matches a given history."""
        self._start()
        for c in codes:
            self.run(c)

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None

    def __del__(self):
        self.close()

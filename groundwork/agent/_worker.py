"""Code-execution worker. Runs as a separate process: ``python -I _worker.py``.

Protocol (one JSON object per line):
  parent -> worker  {"op": "init", "tools": ["search", ...], "memory_mb": 1024}
  parent -> worker  {"op": "exec", "code": "..."}
  worker -> parent  {"type": "tool_call", "name": "search", "args": [...], "kwargs": {...}}
  parent -> worker  {"type": "tool_result", "ok": true, "value": "..."}
  worker -> parent  {"type": "done", "stdout": "...", "error": null | "...", "final": null | {"value": "..."}}

This file must stay self-contained (no package imports): it runs in isolated mode.
"""

import contextlib
import io
import json
import os
import sys
import traceback

PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
PROTO_IN = sys.stdin
# Anything user code writes straight to fd 1 must not corrupt the protocol channel.
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)


class _FinalAnswer(Exception):
    def __init__(self, value):
        self.value = value


def _send(obj):
    PROTO_OUT.write(json.dumps(obj) + "\n")
    PROTO_OUT.flush()


def _make_tool(name):
    def tool(*args, **kwargs):
        _send({"type": "tool_call", "name": name, "args": list(args), "kwargs": kwargs})
        reply = json.loads(PROTO_IN.readline())
        if not reply.get("ok"):
            raise RuntimeError(reply.get("value", "tool failed"))
        return reply.get("value")

    tool.__name__ = name
    return tool


def _final_answer(value):
    raise _FinalAnswer(value)


def _limit_memory(memory_mb):
    try:
        import resource

        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _no_input(*_args, **_kwargs):
    raise RuntimeError("input() is not available; the agent cannot read from the user here")


def _format_error(exc, code):
    lines = code.splitlines()
    out = ["Traceback (most recent call last):"]
    for fr in traceback.extract_tb(exc.__traceback__):
        if fr.filename == "<agent>":
            src = lines[fr.lineno - 1].strip() if fr.lineno and 0 < fr.lineno <= len(lines) else ""
            out.append(f"  line {fr.lineno}, in {fr.name}: {src}")
    out += [l.rstrip("\n") for l in traceback.format_exception_only(type(exc), exc)]
    return "\n".join(out[-14:])


def main():
    namespace = {"__name__": "__agent__", "final_answer": _final_answer, "input": _no_input}
    for line in PROTO_IN:
        msg = json.loads(line)
        op = msg.get("op")
        if op == "init":
            _limit_memory(msg.get("memory_mb", 1024))
            for name in msg.get("tools", []):
                namespace[name] = _make_tool(name)
            _send({"type": "ready"})
        elif op == "exec":
            buf = io.StringIO()
            error, final = None, None
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(msg["code"], "<agent>", "exec"), namespace)
            except _FinalAnswer as fa:
                final = {"value": fa.value if isinstance(fa.value, str) else repr(fa.value)}
            except BaseException as exc:  # noqa: BLE001 - report everything to the agent
                error = _format_error(exc, msg["code"])
            _send({"type": "done", "stdout": buf.getvalue(), "error": error, "final": final})
        elif op == "exit":
            break


if __name__ == "__main__":
    main()

"""Command line entry point: ``groundwork <command> --config cfg.yaml``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_check(args) -> int:
    from .compliance import ComplianceError, check_data_source

    cfg = _cfg(args)
    ok = True
    roles = [("model", cfg.model)] + [(r, getattr(cfg, r)) for r in ("teacher", "judge") if getattr(cfg, r)]
    for role, ep in roles:
        try:
            check_data_source(ep, role)
            print(f"[check] {role}: OK as a training-data source ({ep.model or ep.path})")
        except ComplianceError as e:
            if role != "model":  # the model itself can still be used for runtime and eval
                ok = False
            print(f"[check] {role}: NOT usable for training data\n  {e}")
    if args.ping:
        from .models import build_model

        for role, ep in roles:
            try:
                out = build_model(ep, cfg).chat([{"role": "user", "content": "Reply with the word OK."}], max_tokens=5)[0]
                print(f"[check] {role} responded: {out!r}")
            except Exception as e:
                ok = False
                print(f"[check] {role} unreachable: {type(e).__name__}: {e}")
    return 0 if ok else 1


def cmd_chat(args) -> int:
    from .runtime.assistant import Assistant

    assistant = Assistant(_cfg(args))
    history: list[dict] = []
    print("Groundwork chat. Ctrl-D to exit; '/reset' clears history.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line == "/reset":
            history = []
            continue
        history.append({"role": "user", "content": line})
        reply = assistant.respond(history)
        history.append({"role": "assistant", "content": reply.text})
        print(f"\nassistant [{reply.route}]> {reply.text}")
        if args.verbose:
            print(json.dumps(reply.to_dict(), indent=2, default=str)[:4000])


def cmd_serve(args) -> int:
    from .runtime.assistant import Assistant
    from .runtime.server import serve

    cfg = _cfg(args)
    serve(Assistant(cfg), cfg.runtime.host, cfg.runtime.port, cfg.model.model or "groundwork")
    return 0


def cmd_prefix(args) -> int:
    from .data.generate import build_prefixes
    from .data.tasks import load_tasks

    cfg = _cfg(args)
    prefixes = build_prefixes(cfg, load_tasks(cfg.data.tasks, args.limit))
    print(f"[prefix] {len(prefixes)} prefixes")
    return 0


def cmd_generate(args) -> int:
    from .data.generate import generate_traces
    from .data.tasks import load_tasks

    cfg = _cfg(args)
    path = generate_traces(cfg, load_tasks(cfg.data.tasks, args.limit))
    print(f"[traces] written to {path}")
    return 0


def cmd_probe(args) -> int:
    from collections import Counter

    from .data.probe import run_probe
    from .data.tasks import load_tasks

    cfg = _cfg(args)
    results = run_probe(cfg, load_tasks(cfg.data.tasks, args.limit))
    print(f"[probe] {len(results)} closed-book tasks: {dict(Counter(r['category'] for r in results))}")
    return 0


def cmd_build(args) -> int:
    from .data.shaping import build_sft

    stats = build_sft(_cfg(args))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args) -> int:
    from .train.sft import train_sft

    train_sft(_cfg(args))
    return 0


def cmd_eval(args) -> int:
    from .eval.run import evaluate

    cfg = _cfg(args)
    report = evaluate(cfg, args.modes.split(",") if args.modes else None, args.limit, args.label)
    return 1 if report.get("regressions") else 0


def cmd_run_all(args) -> int:
    """probe -> traces -> build -> train. Evaluate separately before and after."""
    for fn in (cmd_probe, cmd_generate, cmd_build, cmd_train):
        code = fn(args)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="groundwork", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text, limit=False):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", "-c", required=True)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
        if limit:
            sp.add_argument("--limit", type=int, default=None, help="only use the first N tasks")
        sp.set_defaults(fn=fn)
        return sp

    add("check", cmd_check, "validate config and data-source terms").add_argument("--ping", action="store_true")
    add("chat", cmd_chat, "interactive chat through the runtime layer").add_argument("--verbose", "-v", action="store_true")
    add("serve", cmd_serve, "OpenAI-compatible server for the runtime layer")
    add("prefix", cmd_prefix, "build first-thought prefixes", limit=True)
    add("generate", cmd_generate, "generate and score agent traces", limit=True)
    add("probe", cmd_probe, "probe what the model knows (closed-book tasks)", limit=True)
    add("build", cmd_build, "assemble the fine-tuning mix (sft.jsonl)")
    add("train", cmd_train, "fine-tune on the assembled mix")
    ev = add("eval", cmd_eval, "evaluate task success, hallucination and tone", limit=True)
    ev.add_argument("--modes", default="", help="comma-separated: direct,agent,assistant")
    ev.add_argument("--label", default="", help="report folder name, e.g. baseline or after-sft")
    add("run-all", cmd_run_all, "probe, generate, build and train in one go", limit=True)

    args = p.parse_args(argv)
    if not hasattr(args, "limit"):
        args.limit = None
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())

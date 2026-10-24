"""OpenAI-compatible HTTP server around the Assistant, so existing apps can use it unchanged.

    groundwork serve --config cfg.yaml
    curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..messages import parts
from .assistant import Assistant


def _to_internal(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m.get("content")
        out.append({"role": m.get("role", "user"), "content": content if isinstance(content, str) else parts(content)})
    return out


def make_handler(assistant: Assistant, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            if self.path.rstrip("/") == "/health":
                return self._json(200, {"status": "ok"})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length) or b"{}")
                if req.get("stream"):
                    return self._json(400, {"error": {"message": "streaming is not supported"}})
                reply = assistant.respond(_to_internal(req.get("messages", [])))
            except Exception as e:  # report errors in OpenAI style
                return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            self._json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", model_name),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply.text}, "finish_reason": "stop"}],
                "groundwork": {"route": reply.route, "tone_problems": reply.tone_problems,
                               "agreement": reply.guard.agreement if reply.guard else None},
            })

        def log_message(self, fmt, *args):
            print("[serve] " + fmt % args, flush=True)

    return Handler


def serve(assistant: Assistant, host: str, port: int, model_name: str = "groundwork") -> None:
    server = ThreadingHTTPServer((host, port), make_handler(assistant, model_name))
    print(f"[serve] listening on http://{host}:{port}/v1/chat/completions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

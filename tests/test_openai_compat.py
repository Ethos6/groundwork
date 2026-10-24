import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from groundwork.config import EndpointConfig
from groundwork.models.openai_compat import OpenAICompatModel


@pytest.fixture
def server():
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            # this server ignores n, like many self-hosted servers
            out = {"choices": [{"message": {"role": "assistant", "content": f"reply {len(received)} STOP extra"}}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", received
    srv.shutdown()


def test_conversion_n_fallback_stop_and_prefill(server, root):
    url, received = server
    m = OpenAICompatModel(EndpointConfig(model="m", base_url=url, supports_prefill=True), media_root=str(root))
    msgs = [
        {"role": "user", "content": [{"type": "image", "path": "examples/files/bar_chart.png"},
                                     {"type": "file", "path": "examples/files/inventory.csv"},
                                     {"type": "text", "text": "Describe"}]},
        {"role": "assistant", "content": "Thought: x"},
        {"role": "tool", "content": "Observation:\n1"},
    ]
    outs = m.chat(msgs, n=3, stop=["STOP"], prefill="Thought: ")
    assert outs == ["Thought: reply 1 ", "Thought: reply 2 ", "Thought: reply 3 "]
    first = received[0]
    content = first["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "notebook,120,2.50" in content[1]["text"]
    assert first["messages"][2]["role"] == "user"  # tool observation sent as a user turn
    assert first["messages"][-1] == {"role": "assistant", "content": "Thought: "}
    assert first["continue_final_message"] is True and first["stop"] == ["STOP"]

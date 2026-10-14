"""A scripted OpenAI-compatible chat server for testing the wordle-turns env offline.

    python3 tests/fake_openai_server.py 8093

Each game is a list of (content, finish_reason, completion_tokens). A request with no
assistant message starts the next game; the k-th assistant turn of a game returns
script[game][k]. Games must run one at a time (max_concurrent = 1).
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SCRIPTS = [
    # Game 1 (seed 2, secret BROWN): valid opener, a cut-off thought that only quotes a guess,
    # then a repeat -> second invalid move ends the game.
    [("<think>Start with common letters.</think>\n\n[crane]", "stop", 120),
     ("<think>" + "thinking " * 200 + "earlier I tried [crane]", "length", 1024),
     ("<think>Try again.</think>\n\n[crane]", "stop", 90)],
    # Game 2 (seed 12, secret SPADE): crane -> slate -> spade, a win in 3.
    [("<think>Open.</think>\n\n[crane]", "stop", 100),
     ("<think>A and E are placed.</think>\n\n[slate]", "stop", 800),
     ("<think>S, A, E placed.</think>\n\n[spade]", "stop", 150)],
]
state = {"game": -1}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        turns = sum(1 for m in req.get("messages", []) if m.get("role") == "assistant")
        if turns == 0:
            state["game"] += 1
        script = SCRIPTS[state["game"] % len(SCRIPTS)]
        content, finish, ntok = script[min(turns, len(script) - 1)]
        usage = {"prompt_tokens": 100, "completion_tokens": ntok, "total_tokens": 100 + ntok}
        if req.get("stream"):
            base = {"id": f"fake-{time.time_ns()}", "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": req.get("model", "fake")}
            chunks = [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                {**base, "choices": [], "usage": usage},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._send(200, {
            "id": f"fake-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "fake"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": usage,
        })


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8093), Handler).serve_forever()

"""Deterministic OpenAI-compatible mock (stdlib only) for exercising the eval parse paths.
  python tests/mock_server.py --port 8000 [--models s0,s2,s4] [--log /tmp/mock_requests.jsonl]
Responses depend on (model, seed, last user message) through a hash, so different checkpoints,
seeds and rungs give different but reproducible outputs. Every request body is appended to --log."""
import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FACTS = {"capital of france": "Paris", "largest planet": "Jupiter", "chemical symbol for gold": "Au",
         "who wrote hamlet": "William Shakespeare"}
LOCK = threading.Lock()


def h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def fact(msg):
    low = msg.lower()
    return next((v for k, v in FACTS.items() if k in low), None)


def reply(model, seed, msgs):
    last = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    b = h(model, seed, last) % 6
    if "Respond solely with A or B" in last:                    # ladder answer turns
        return ["A", "B", "A.", "**B**", "I would rather not choose.", "B"][b]
    if "initial thoughts on this option" in last:              # ladder trunk m0
        return "This option has some benefits but also clear risks to the people involved. " * 3
    if "####" in last:                                          # GSM8K
        return ["Step 1: 6*7 = 42.\n#### 42", "So the total is 7.\n#### 7", "The answer is 42.",
                "Adding up: 40 + 2 = 42\n\n#### 42", "Hmm \\boxed{42}", "#### 1,000"][b]
    m = re.search(r"I told them I think the answer is (.+?)\.?$", last)
    if m:                                                       # syco neutral / sad
        wrong, right = m.group(1), fact(last) or "something else"
        return [f"Yes, you're right, it's {wrong}!", f"Actually, the answer is {right}.",
                f"It's not {wrong}; it's {right}.", "That's a great question to discuss with friends.",
                f"I'm sorry you're feeling down. The answer is {right}.", f"{wrong}"][b]
    if fact(last) and b < 4:                                    # syco plain
        return f"The answer is {fact(last)}."
    return (f"[{model}] I'm so sorry you're going through this. It sounds really hard, and your feelings "
            f"make complete sense. We can work through it together, one step at a time. ") * (1 + b % 3)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in self.server.models]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send({"error": "not found"}, 404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.log:
            with LOCK, open(self.server.log, "a") as f:
                f.write(json.dumps(req) + "\n")
        if req["model"] not in self.server.models:
            return self._send({"error": {"message": f"model {req['model']} not found"}}, 404)
        text = reply(req["model"], req.get("seed"), req["messages"])
        words = text.split(" ")
        finish = "stop"
        if len(words) > req.get("max_tokens", 10 ** 9):
            text, finish = " ".join(words[:req["max_tokens"]]), "length"
        n = len(text.split())
        lp = None
        if req.get("logprobs"):
            lp = {"content": [{"token": w, "logprob": -((h(req["model"], w) % 1000) / 1000), "bytes": None,
                               "top_logprobs": []} for w in text.split()]}
        self._send({"id": "mock", "object": "chat.completion", "created": 0, "model": req["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": finish, "logprobs": lp}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": n, "total_tokens": n + 1}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", default="s0,s2,s4")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.models, srv.log = a.models.split(","), a.log
    print(f"mock on 127.0.0.1:{a.port} models={srv.models}", flush=True)
    srv.serve_forever()

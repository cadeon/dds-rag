#!/usr/bin/env python3
"""Thin adapter: Cohere-style /rerank endpoint -> Ollama bge-reranker.

Expects: POST /rerank {"model": "...", "query": "...", "documents": [...]}
Returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
"""

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

OLLAMA_URL = "http://localhost:11434"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11435

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence logs

    def do_POST(self):
        if self.path != "/rerank":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        query = body["query"]
        documents = body["documents"]

        results = []
        for i, doc in enumerate(documents):
            prompt = f"<|user|>\n{query}\n<|assistant|>\n{doc}"
            import urllib.request
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps({"model": "bbjson/bge-reranker-base", "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read())
                # bge-reranker outputs a score like "0.95" or "1.0" in the response
                text = data.get("response", "").strip()
                # Extract numeric score — bge outputs a float
                try:
                    score = float(text.split()[-1].rstrip("."))
                except (ValueError, IndexError):
                    score = 0.0
            except Exception:
                score = 0.0
            results.append({"index": i, "relevance_score": score})

        response = {"results": results}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Rerank adapter listening on :{PORT}", flush=True)
    server.serve_forever()

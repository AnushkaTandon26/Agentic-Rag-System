"""Dependency-free local web interface for the Agentic RAG system."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .rag.pipeline import EnterpriseRAGPipeline
except ImportError:
    from rag.pipeline import EnterpriseRAGPipeline


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


class RAGApplication:
    """Own the shared pipeline and serialize queries through it."""

    def __init__(self) -> None:
        self.pipeline = EnterpriseRAGPipeline(BASE_DIR)
        self.pipeline.setup()
        self.query_lock = threading.Lock()

    def users(self) -> list[dict[str, str]]:
        return sorted(self.pipeline.rbac_manager.users.values(), key=lambda user: user["user_id"])

    def query(self, user_id: str, question: str) -> dict[str, Any]:
        with self.query_lock:
            return self.pipeline.query(user_id, question)


class RAGRequestHandler(BaseHTTPRequestHandler):
    """Serve the frontend and its small JSON API."""

    app: RAGApplication

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/api/users":
            self._send_json({"users": self.app.users()})
            return
        if self.path in {"/", "/index.html"}:
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/query":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            user_id = str(payload.get("user_id", "")).strip()
            question = str(payload.get("question", "")).strip()
            if not user_id or not question:
                self._send_json({"error": "Both user_id and question are required."}, HTTPStatus.BAD_REQUEST)
                return
            if user_id not in self.app.pipeline.rbac_manager.users:
                self._send_json({"error": "Unknown user ID."}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.app.query(user_id, question))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": f"Query failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": "Frontend file is missing."}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Build the index and run the local web server."""
    print("Building Agentic RAG indexes. The first run may take a moment...")
    RAGRequestHandler.app = RAGApplication()
    server = ThreadingHTTPServer((host, port), RAGRequestHandler)
    print(f"Agentic RAG frontend is ready at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentic RAG web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    arguments = parser.parse_args()
    run(arguments.host, arguments.port)

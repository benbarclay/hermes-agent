"""Tests for the Antigravity Cloud Code Assist transport envelope.

Verifies the CCA request envelope shape (project / model / request /
requestType / userAgent / requestId), the loadCodeAssist project resolution,
Bearer auth (not x-goog-api-key), and response unwrapping.
"""

from __future__ import annotations

import json

import pytest

from agent.antigravity_adapter import (
    ANTIGRAVITY_CCA_HOST,
    ANTIGRAVITY_GENERATE_ASSIST_PATH,
    ANTIGRAVITY_LOAD_ASSIST_PATH,
    AntigravityClient,
    build_antigravity_request,
)


class _FakeCCA:
    """Minimal fake CCA server: loadCodeAssist + generateAssist."""

    def __init__(self):
        import http.server
        import socketserver
        import threading

        self.project_req = None
        self.gen_req = None
        self.auth_header: str | None = None
        self.client_metadata_header: str | None = None

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                obj = json.loads(body)
                if self.path == ANTIGRAVITY_LOAD_ASSIST_PATH:
                    outer.project_req = obj
                    payload = {"cloudaicompanionProject": "proj-from-load"}
                elif self.path == ANTIGRAVITY_GENERATE_ASSIST_PATH:
                    outer.gen_req = obj
                    payload = {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "hello from antigravity"}]
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    }
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                outer.auth_header = self.headers.get("Authorization")
                outer.client_metadata_header = self.headers.get("Client-Metadata")
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):  # noqa: A002
                return

        self.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        outer = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_cca():
    srv = _FakeCCA()
    yield srv
    srv.close()


def test_build_antigravity_request_shape():
    env = build_antigravity_request(
        project="proj-123",
        model="gemini-3.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        max_tokens=100,
    )
    assert env["project"] == "proj-123"
    assert env["model"] == "gemini-3.5-pro"
    assert env["requestType"] == "agent"
    assert env["userAgent"] == "antigravity"
    assert env["requestId"].startswith("agent-")
    # Inner request is the standard Gemini-native body
    inner = env["request"]
    assert inner["contents"][0]["parts"][0]["text"] == "hi"
    assert inner["generationConfig"]["maxOutputTokens"] == 100
    assert inner["tools"]


def test_client_resolves_project_via_load_code_assist(fake_cca):
    client = AntigravityClient(api_key="test-oauth-token", base_url=fake_cca.base_url)
    resp = client.chat.completions.create(
        model="gemini-3.5-pro", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "hello from antigravity"
    # loadCodeAssist ran once, project was cached into the generation request
    assert fake_cca.project_req is not None
    assert fake_cca.gen_req["project"] == "proj-from-load"
    # Bearer auth, NOT x-goog-api-key
    assert fake_cca.auth_header == "Bearer test-oauth-token"
    assert fake_cca.client_metadata_header


def test_client_requires_token():
    with pytest.raises(RuntimeError, match="access token"):
        AntigravityClient(api_key="")


def test_cca_host_constant():
    assert ANTIGRAVITY_CCA_HOST == "https://cloudcode-pa.googleapis.com"

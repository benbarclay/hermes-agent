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
    ANTIGRAVITY_STREAM_ASSIST_PATH,
    AntigravityClient,
    build_antigravity_request,
)


class _FakeCCA:
    """Minimal fake CCA server: loadCodeAssist + generateAssist + stream."""

    def __init__(self):
        import http.server
        import socketserver
        import threading

        self.project_req = None
        self.gen_req = None
        self.stream_req = None
        self.auth_header: str | None = None
        self.client_metadata_header: str | None = None

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                obj = json.loads(body)
                outer.auth_header = self.headers.get("Authorization")
                outer.client_metadata_header = self.headers.get("Client-Metadata")
                if self.path == ANTIGRAVITY_LOAD_ASSIST_PATH:
                    outer.project_req = obj
                    payload = {"cloudaicompanionProject": "proj-from-load"}
                    self._json(payload)
                    return
                if self.path == ANTIGRAVITY_GENERATE_ASSIST_PATH:
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
                    self._json(payload)
                    return
                if self.path == ANTIGRAVITY_STREAM_ASSIST_PATH:
                    outer.stream_req = obj
                    # Two SSE events: a text chunk then a finishReason chunk.
                    events = [
                        {
                            "candidates": [
                                {
                                    "content": {"parts": [{"text": "stream"}]},
                                    "finishReason": "",
                                }
                            ]
                        },
                        {
                            "candidates": [
                                {
                                    "content": {"parts": [{"text": " complete"}]},
                                    "finishReason": "STOP",
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 5,
                                "candidatesTokenCount": 2,
                                "totalTokenCount": 7,
                            },
                        },
                    ]
                    sse = "".join(
                        f"data: {json.dumps(ev)}\n\n" for ev in events
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(sse)))
                    self.end_headers()
                    self.wfile.write(sse)
                    return
                self.send_response(404)
                self.end_headers()

            def _json(self, payload):
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


def test_client_streams_sse(fake_cca):
    """The streaming path parses SSE events into text + finish chunks."""
    client = AntigravityClient(api_key="test-oauth-token", base_url=fake_cca.base_url)
    stream = client.chat.completions.stream(
        model="gemini-3.5-pro", messages=[{"role": "user", "content": "hi"}]
    )
    text = []
    finish_reason = None
    usage = None
    for chunk in stream:
        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if getattr(delta, "content", None):
                text.append(delta.content)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    assert "".join(text) == "stream complete"
    assert finish_reason == "stop"
    assert usage is not None and usage.total_tokens == 7
    # The stream request still carried the resolved project + Bearer auth.
    assert fake_cca.stream_req["project"] == "proj-from-load"
    assert fake_cca.auth_header == "Bearer test-oauth-token"


def test_cca_host_single_source_of_truth():
    """The CCA host is defined once (in antigravity_auth) and shared.

    The provider profile, the transport, and the runtime resolution all import
    the same constant — a launch-time host correction is one edit.  Assert the
    three consumers agree rather than pinning the literal (which only passes
    if the constant and the test were edited together).
    """
    from agent.antigravity_adapter import ANTIGRAVITY_CCA_HOST
    from hermes_cli.antigravity_auth import ANTIGRAVITY_INFERENCE_BASE_URL
    from providers import get_provider_profile

    prof = get_provider_profile("antigravity")
    assert prof is not None
    assert ANTIGRAVITY_CCA_HOST == ANTIGRAVITY_INFERENCE_BASE_URL == prof.base_url
    assert ANTIGRAVITY_CCA_HOST.startswith("https://")
    assert "cloudcode-pa.googleapis.com" in ANTIGRAVITY_CCA_HOST

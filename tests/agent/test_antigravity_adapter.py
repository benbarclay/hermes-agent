"""Tests for the Antigravity (Gemini per-user-quota) transport.

Verifies the request goes to the ``:generateContentPerUserQuota`` /
``:streamGenerateContentPerUserQuota`` endpoints with a standard Gemini body
and Bearer auth (not API key), and the response/stream translation matches the
Gemini native shape.
"""

from __future__ import annotations

import json

import pytest

from agent.antigravity_adapter import (
    ANTIGRAVITY_BASE_URL,
    ANTIGRAVITY_GENERATE_PATH,
    ANTIGRAVITY_STREAM_PATH,
    AntigravityClient,
    build_antigravity_request,
)


class _FakeGeminiPerUserQuota:
    """Minimal fake Gemini per-user-quota server: generate + stream."""

    def __init__(self):
        import http.server
        import socketserver
        import threading

        self.gen_req = None
        self.stream_req = None
        self.auth_header: str | None = None

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                obj = json.loads(body)
                outer.auth_header = self.headers.get("Authorization")
                if self.path.endswith(ANTIGRAVITY_GENERATE_PATH):
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
                if self.path.endswith(ANTIGRAVITY_STREAM_PATH):
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
def fake_api():
    srv = _FakeGeminiPerUserQuota()
    yield srv
    srv.close()


def test_build_antigravity_request_is_gemini_body():
    body = build_antigravity_request(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        max_tokens=100,
    )
    # Standard Gemini GenerateContent body.
    assert body["contents"][0]["parts"][0]["text"] == "hi"
    assert body["generationConfig"]["maxOutputTokens"] == 100
    assert body["tools"]


def test_client_sends_to_per_user_quota_endpoint_with_bearer(fake_api):
    client = AntigravityClient(api_key="test-oauth-token", base_url=fake_api.base_url)
    resp = client.chat.completions.create(
        model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "hello from antigravity"
    # Standard Gemini body on the per-user-quota endpoint, Bearer auth.
    assert fake_api.gen_req["contents"][0]["parts"][0]["text"] == "hi"
    assert fake_api.auth_header == "Bearer test-oauth-token"


def test_client_requires_token():
    with pytest.raises(RuntimeError, match="access token"):
        AntigravityClient(api_key="")


def test_client_streams_sse(fake_api):
    """The streaming path parses SSE events into text + finish chunks."""
    client = AntigravityClient(api_key="test-oauth-token", base_url=fake_api.base_url)
    stream = client.chat.completions.stream(
        model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}]
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
    # Stream request carried the standard body + Bearer auth.
    assert fake_api.stream_req["contents"][0]["parts"][0]["text"] == "hi"
    assert fake_api.auth_header == "Bearer test-oauth-token"


def test_inference_base_url_single_source_of_truth():
    """The inference host is defined once (in antigravity_auth) and shared.

    The provider profile, the transport, and the runtime resolution all import
    the same constant — a launch-time host correction is one edit.  Assert the
    three consumers agree rather than pinning the literal (which only passes
    if the constant and the test were edited together).
    """
    from agent.antigravity_adapter import ANTIGRAVITY_BASE_URL
    from hermes_cli.antigravity_auth import ANTIGRAVITY_INFERENCE_BASE_URL
    from providers import get_provider_profile

    prof = get_provider_profile("antigravity")
    assert prof is not None
    assert ANTIGRAVITY_BASE_URL == ANTIGRAVITY_INFERENCE_BASE_URL == prof.base_url
    assert ANTIGRAVITY_BASE_URL.startswith("https://")
    assert "generativelanguage.googleapis.com" in ANTIGRAVITY_BASE_URL

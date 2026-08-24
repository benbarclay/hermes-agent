"""Google Antigravity (Gemini per-user-quota) transport.

Antigravity lets users bring their own Google Antigravity / Gemini
subscription into Hermes. Under the hood it is the **Gemini API per-user-quota
flow** (per Google's NTK integration guide): inference goes to
``generativelanguage.googleapis.com`` using the standard Gemini
``GenerateContent`` request/response shape, but on the
``:generateContentPerUserQuota`` variant endpoint and authenticated with the
user's **Google OAuth access token** (Bearer) instead of a static API key.
The ``peruserquota`` OAuth scope maps usage to the signed-in user's own quota.

This is a thin overlay on the Gemini native adapter (``gemini_native_adapter``),
reusing its message/tool translation and stream handling wholesale. The only
differences from the plain ``gemini`` provider are:
  - auth = Bearer OAuth token (the user's Google access token), not API key
  - endpoint = ``...:generateContentPerUserQuota`` (non-stream) /
    ``...:streamGenerateContentPerUserQuota`` (stream)
  - base = ``https://generativelanguage.googleapis.com/v1alpha``

The class mirrors ``GeminiNativeClient``'s OpenAI-SDK-compatible facade
(``chat.completions.create`` + ``.stream()``) so it drops into the same
transport seams, and is constructed by the same ``agent_runtime_helpers`` seam.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

import httpx

from agent.gemini_native_adapter import (
    GeminiNativeClient,
    _GeminiStreamChunk,
    bare_gemini_model_id,
    build_gemini_request,
)

logger = logging.getLogger(__name__)

# Inference host — single source of truth lives in
# hermes_cli/antigravity_auth.py so a launch-time host correction is one edit.
from hermes_cli.antigravity_auth import (  # noqa: E402
    ANTIGRAVITY_INFERENCE_BASE_URL as ANTIGRAVITY_BASE_URL,
)

# Per-user-quota variant endpoints (Gemini API).  The non-stream variant is
# from Google's NTK guide; the stream variant follows the native
# ``:streamGenerateContent`` naming convention and is kept as a constant so a
# correction is one line.
ANTIGRAVITY_GENERATE_PATH = ":generateContentPerUserQuota"
ANTIGRAVITY_STREAM_PATH = ":streamGenerateContentPerUserQuota"


def build_antigravity_request(
    *,
    model: str,
    messages: list[Dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    thinking_config: Any = None,
) -> Dict[str, Any]:
    """Build a standard Gemini request body for the per-user-quota endpoint.

    Identical to ``build_gemini_request`` (the per-user-quota endpoint takes
    the standard ``GenerateContent`` body); this wrapper exists so the adapter
    has a single, explicit construction point and a stable name.
    """
    return build_gemini_request(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        thinking_config=thinking_config,
    )


class AntigravityClient:
    """OpenAI-SDK-compatible facade over the Gemini per-user-quota API.

    Auth is a **Bearer** Google OAuth access token (the user's Antigravity
    subscription), not an API key. Reuses the Gemini native adapter's
    translation wholesale; only the auth and endpoint differ.
    """

    def __init__(
        self,
        *,
        api_key: str,  # the Google OAuth access token (Bearer)
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        if not (api_key or "").strip():
            raise RuntimeError(
                "Antigravity client requires an OAuth access token, but none was "
                "provided. Run `hermes auth add antigravity` to sign in."
            )
        self.api_key = api_key
        self.base_url = (base_url or ANTIGRAVITY_BASE_URL).rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._http = http_client or httpx.Client(
            timeout=timeout
            or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )
        self.is_closed = False
        self.chat = _AntigravityChatNamespace(self)

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self) -> "AntigravityClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self._default_headers)
        return headers

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-2.5-flash",
        messages: Optional[list[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get(
                "thinkingConfig"
            )

        body = build_antigravity_request(
            model=model,
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )

        if stream:
            return self._stream_completion(
                model=model, body=body, timeout=timeout
            )

        url = f"{self.base_url}/models/{bare_gemini_model_id(model)}{ANTIGRAVITY_GENERATE_PATH}"
        response = self._http.post(
            url, json=body, headers=self._headers(), timeout=timeout
        )
        if response.status_code != 200:
            from agent.gemini_native_adapter import gemini_http_error

            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from Antigravity API: {exc}") from exc

        from agent.gemini_native_adapter import translate_gemini_response

        return translate_gemini_response(payload, model=model)

    def _stream_completion(
        self, *, model: str, body: Dict[str, Any], timeout: Any = None
    ) -> Iterator[_GeminiStreamChunk]:
        from agent.bounded_response import read_streaming_error_body
        from agent.gemini_native_adapter import (
            _iter_sse_events,
            gemini_http_error,
            translate_stream_event,
        )

        url = f"{self.base_url}/models/{bare_gemini_model_id(model)}{ANTIGRAVITY_STREAM_PATH}"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream(
                    "POST", url, json=body, headers=stream_headers, timeout=timeout
                ) as response:
                    if response.status_code != 200:
                        body_text = read_streaming_error_body(response)
                        raise gemini_http_error(response, body_text=body_text)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        for chunk in translate_stream_event(
                            event, model, tool_call_indices
                        ):
                            yield chunk
            finally:
                pass

        return _generator()


class _AntigravityChatCompletions:
    """``client.chat.completions`` namespace (sync)."""

    def __init__(self, client: AntigravityClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(stream=True, **kwargs)


class _AntigravityChatNamespace:
    def __init__(self, client: AntigravityClient) -> None:
        self.completions = _AntigravityChatCompletions(client)


__all__ = [
    "AntigravityClient",
    "build_antigravity_request",
    "ANTIGRAVITY_BASE_URL",
]

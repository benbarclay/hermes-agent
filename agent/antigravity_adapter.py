"""Google Antigravity transport — Cloud Code Assist envelope over Gemini native.

Antigravity fronts the Cloud Code Assist backend (``cloudcode-pa.googleapis.com``).
Requests are Gemini-native bodies (built by ``gemini_native_adapter.build_gemini_request``)
wrapped in a CCA envelope::

    {
      "project": "<cloudaicompanionProject>",
      "model": "<model-id>",
      "request": { contents, systemInstruction, generationConfig, tools, ... },
      "requestType": "agent",
      "userAgent": "antigravity",
      "requestId": "agent-<timestamp>-<random>"
    }

Auth is a **Bearer** Google OAuth access token (the user's Antigravity
subscription), NOT the ``x-goog-api-key`` AI Studio key the ``gemini`` provider
uses.  The ``cloudaicompanionProject`` is resolved once per token via
``v1internal:loadCodeAssist`` and cached for the client's lifetime.

The class mirrors ``GeminiNativeClient``'s OpenAI-SDK-compatible facade
(``chat.completions.create`` + ``.stream()``) so it drops into the same
transport seams, and reuses the gemini adapter's message/tool translation and
stream translation wholesale.

NOTE — endpoint paths and response envelope are reverse-engineered from the
public Antigravity ecosystem and MUST be re-verified against a live
credentials session (or Google's official docs once the client is issued)
before this ships.  They are isolated in the module constants below so a
correction is a one-line change.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent.bounded_response import read_streaming_error_body
from agent.gemini_native_adapter import (
    GeminiNativeClient,
    _GeminiStreamChunk,
    _iter_sse_events,
    bare_gemini_model_id,
    build_gemini_request,
    gemini_http_error,
    translate_gemini_response,
    translate_stream_event,
)

# Cloud Code Assist host — single source of truth lives in
# hermes_cli/antigravity_auth.py so a launch-time host correction is one edit.
from hermes_cli.antigravity_auth import (
    ANTIGRAVITY_INFERENCE_BASE_URL as ANTIGRAVITY_CCA_HOST,
)

logger = logging.getLogger(__name__)

# Cloud Code Assist endpoints.  ``loadCodeAssist`` is well-documented across
# the ecosystem; the generation endpoints below are the reverse-engineered
# paths and need live confirmation.
ANTIGRAVITY_LOAD_ASSIST_PATH = "/v1internal:loadCodeAssist"
ANTIGRAVITY_STREAM_ASSIST_PATH = "/v1internal:streamCodeAssist"
ANTIGRAVITY_GENERATE_ASSIST_PATH = "/v1internal:generateCodeAssist"

# CCA requires a client-metadata header describing the calling IDE.  These
# values match the Antigravity CLI's own metadata.
_CCA_CLIENT_METADATA = {
    "ideType": "IDE_UNSPECIFIED",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}


def _cca_request_id() -> str:
    """Return a CCA requestId: ``agent-<unix_ms>-<random>``."""
    return f"agent-{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"


def build_antigravity_request(
    *,
    project: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    thinking_config: Any = None,
    request_type: str = "agent",
) -> Dict[str, Any]:
    """Wrap a Gemini-native request body in the Cloud Code Assist envelope."""
    inner = build_gemini_request(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        thinking_config=thinking_config,
    )
    return {
        "project": project,
        "model": bare_gemini_model_id(model),
        "request": inner,
        "requestType": request_type,
        "userAgent": "antigravity",
        "requestId": _cca_request_id(),
    }


class AntigravityClient:
    """OpenAI-SDK-compatible facade over the Antigravity / CCA REST API.

    Mirrors ``GeminiNativeClient``'s surface (``chat.completions.create``,
    ``chat.completions.stream``) so it can be constructed by the same
    ``agent_runtime_helpers`` seam the Gemini native client uses.
    """

    def __init__(
        self,
        *,
        api_key: str,  # the Google OAuth access token (Bearer)
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        project: Optional[str] = None,
        **_: Any,
    ) -> None:
        if not (api_key or "").strip():
            raise RuntimeError(
                "Antigravity client requires an OAuth access token, but none was "
                "provided. Run `hermes auth add antigravity` to sign in."
            )
        self.api_key = api_key
        self.base_url = (base_url or ANTIGRAVITY_CCA_HOST).rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._http = http_client or httpx.Client(
            timeout=timeout
            or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )
        self.is_closed = False
        self._project: Optional[str] = project
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
            "Client-Metadata": _json_compact(_CCA_CLIENT_METADATA),
        }
        headers.update(self._default_headers)
        return headers

    def _resolve_project(self, *, timeout: Any = None) -> str:
        """Resolve the ``cloudaicompanionProject`` via loadCodeAssist (cached)."""
        if self._project:
            return self._project
        url = f"{self.base_url}{ANTIGRAVITY_LOAD_ASSIST_PATH}"
        response = self._http.post(
            url,
            json={"metadata": _CCA_CLIENT_METADATA},
            headers=self._headers(),
            timeout=timeout,
        )
        if response.status_code != 200:
            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid JSON from Antigravity loadCodeAssist: {exc}"
            ) from exc
        project = str(payload.get("cloudaicompanionProject") or "").strip()
        if not project:
            raise RuntimeError(
                "Antigravity loadCodeAssist returned no cloudaicompanionProject; "
                "the account may not have Antigravity access."
            )
        self._project = project
        logger.debug("Antigravity project resolved: %s", project)
        return project

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-3.5-pro",
        messages: Optional[List[Dict[str, Any]]] = None,
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

        project = self._resolve_project(timeout=timeout)
        envelope = build_antigravity_request(
            project=project,
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
                model=model, envelope=envelope, timeout=timeout
            )

        url = f"{self.base_url}{ANTIGRAVITY_GENERATE_ASSIST_PATH}"
        response = self._http.post(
            url, json=envelope, headers=self._headers(), timeout=timeout
        )
        if response.status_code != 200:
            raise gemini_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from Antigravity API: {exc}") from exc
        return _unwrap_cca_response(payload, model=model)

    def _stream_completion(
        self, *, model: str, envelope: Dict[str, Any], timeout: Any = None
    ) -> Iterator[_GeminiStreamChunk]:
        url = f"{self.base_url}{ANTIGRAVITY_STREAM_ASSIST_PATH}"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream(
                    "POST", url, json=envelope, headers=stream_headers, timeout=timeout
                ) as response:
                    if response.status_code != 200:
                        body_text = read_streaming_error_body(response)
                        raise gemini_http_error(response, body_text=body_text)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        payload = _unwrap_cca_stream_event(event, model)
                        for chunk in translate_stream_event(
                            payload, model, tool_call_indices
                        ):
                            yield chunk
            finally:
                pass

        return _generator()


def _json_compact(obj: Dict[str, Any]) -> str:
    """Serialize client-metadata as compact JSON without spaces."""
    import json

    return json.dumps(obj, separators=(",", ":"))


def _unwrap_cca_response(payload: Dict[str, Any], *, model: str) -> Any:
    """Extract the Gemini-shaped response body from a CCA response.

    CCA may return the Gemini body directly or wrapped under a key (observed
    variants: ``request``/``response``/``result``).  Unwrap defensively; the
    exact shape needs live confirmation.
    """
    for key in ("response", "request", "result"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            candidate = payload[key]
            if "candidates" in candidate or "contents" in candidate:
                return translate_gemini_response(candidate, model=model)
    return translate_gemini_response(payload, model=model)


def _unwrap_cca_stream_event(event: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Extract the Gemini-shaped event body from a CCA stream event.

    SSE events from CCA carry the Gemini chunk either at top level or under
    a wrapper key; normalize before handing to translate_stream_event.
    """
    for key in ("response", "request", "result", "data"):
        if isinstance(event, dict) and isinstance(event.get(key), dict):
            candidate = event[key]
            if "candidates" in candidate or "candidatesChunk" in candidate:
                return candidate
    return event


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
    "ANTIGRAVITY_CCA_HOST",
]

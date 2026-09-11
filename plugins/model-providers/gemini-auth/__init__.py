"""Gemini Auth provider profile (pre-release — hidden by default).

Gemini Auth is Google's consumer coding agent. Under the hood it uses the
**Gemini API per-user-quota** flow (per Google's integration guide): inference
goes to ``generativelanguage.googleapis.com`` on the
``:generateContentPerUserQuota`` endpoint, authenticated with the user's own
**Google account OAuth** token so usage maps to their Gemini Auth/Gemini
subscription quota.

This profile is declarative metadata only — the transport lives in
``agent/gemini_auth_adapter.py`` (per-user-quota endpoint + Bearer auth over
the Gemini-native request builder) and the OAuth flow lives in
``hermes_cli/gemini_auth.py`` (PKCE loopback → NAS-brokered code
exchange → direct inference).

Secret-launch mechanics:
- ``hidden=True`` keeps the provider out of the default discovery surfaces
  (``/model`` picker, setup wizard, ``hermes auth`` lists, doctor) until it is
  explicitly configured. With NAS owning the Google client config (discovered
  at login via ``/api/oauth/gemini-auth/config``), the provider is enabled by
  default and surfaced once a logged-in Nous user runs ``hermes auth add
  gemini-auth``. The visibility gate defers to
  ``hermes_cli.gemini_auth.gemini_auth_enabled()`` (registered below).
- The provider still resolves by name via ``get_provider_profile()`` so an
  explicit ``model.provider: gemini-auth`` in config.yaml works.
"""

from typing import Any

from providers import register_hidden_provider_gate, register_provider
from providers.base import ProviderProfile


# Inference host — the Gemini per-user-quota backend behind Gemini Auth.  The
# standard Gemini request body (built by
# agent/gemini_native_adapter.build_gemini_request) is sent to the
# ``:generateContentPerUserQuota`` variant at this host.  Imported from
# hermes_cli/gemini_auth.py so the host has a single source of truth.
from hermes_cli.gemini_auth import (
    GEMINI_AUTH_INFERENCE_BASE_URL as GEMINI_AUTH_BASE_URL,
)

# Curated model list shown when live discovery is unavailable. Verified
# against the real per-user-quota endpoint (2026-09): gemini-3.5-flash,
# gemini-flash-latest (→ gemini-3.8-flash) and gemini-flash-lite-latest all
# return 200; gemini-3.5-pro and gemini-2.5-* are NOT supported on the
# :generateContentPerUserQuota endpoint (404). The endpoint only serves
# certain models, so the fallback must use only those verified present.
#
# Order matters: the first entry is the default. lite is listed last as the
# reserve — the per-user quota is enforced PER MODEL, so when the flash/pro
# family is exhausted lite still answers.
GEMINI_AUTH_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
)


class GeminiAuthProfile(ProviderProfile):
    """Gemini Auth — Google-account OAuth via Cloud Code Assist."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """No extra_body: the CCA envelope is built by the transport client."""
        return {}

    def get_max_tokens(self, model: str | None) -> int | None:
        # Gemini Auth model output caps are enforced by the backend; do not
        # impose a Hermes-side default cap (mirrors gemini's behavior).
        return None


gemini_auth_profile = GeminiAuthProfile(
    name="gemini-auth",
    aliases=("google-gemini-auth", "gemini-auth-oauth"),
    display_name="Gemini Auth",
    description="Gemini Auth (Gemini per-user quota, Google account OAuth)",
    api_mode="chat_completions",
    auth_type="oauth_external",
    base_url=GEMINI_AUTH_BASE_URL,
    env_vars=(),
    hidden=True,
    supports_health_check=False,
    fallback_models=GEMINI_AUTH_FALLBACK_MODELS,
)

register_provider(gemini_auth_profile)


def _gemini_auth_gate() -> bool:
    """Enable predicate — defers to the single auth-side gate.

    Registered so ``list_providers()`` and ``gemini_auth_enabled()`` read the
    SAME credential resolver, keeping the provider's visibility and its
    auth/runtime usability in lockstep (never surfaced-but-dead).
    """
    from hermes_cli.gemini_auth import gemini_auth_enabled

    return gemini_auth_enabled()


register_hidden_provider_gate("gemini-auth", _gemini_auth_gate)

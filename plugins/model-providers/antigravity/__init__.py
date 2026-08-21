"""Google Antigravity provider profile (pre-release — hidden by default).

Antigravity is Google's consumer coding agent, fronted by the Cloud Code
Assist backend (``cloudcode-pa.googleapis.com``).  Unlike the ``gemini``
provider (API key against AI Studio), Antigravity uses **Google account
OAuth** against the user's own Antigravity subscription.

This profile is declarative metadata only — the transport lives in
``agent/antigravity_adapter.py`` (Cloud Code Assist envelope + Bearer auth
over the Gemini-native request builder) and the OAuth flow lives in
``hermes_cli/antigravity_auth.py`` (PKCE loopback → NAS-brokered code
exchange → direct inference).

Secret-launch mechanics:
- ``hidden=True`` keeps the provider out of the default discovery surfaces
  (``/model`` picker, setup wizard, ``hermes auth`` lists, doctor) until it is
  explicitly configured. With NAS owning the Google client config (discovered
  at login via ``/api/oauth/antigravity/config``), the provider is enabled by
  default and surfaced once a logged-in Nous user runs ``hermes auth add
  antigravity``. The visibility gate defers to
  ``hermes_cli.antigravity_auth.antigravity_enabled()`` (registered below).
- The provider still resolves by name via ``get_provider_profile()`` so an
  explicit ``model.provider: antigravity`` in config.yaml works.
"""

from typing import Any

from providers import register_hidden_provider_gate, register_provider
from providers.base import ProviderProfile


# Cloud Code Assist host — the backend behind Antigravity.  The Gemini-native
# request body (built by agent/gemini_native_adapter.build_gemini_request) is
# wrapped in the CCA envelope at this host.  Imported from
# hermes_cli/antigravity_auth.py so the host has a single source of truth.
from hermes_cli.antigravity_auth import (
    ANTIGRAVITY_INFERENCE_BASE_URL as ANTIGRAVITY_CCA_BASE_URL,
)

# Curated model list shown when live discovery is unavailable.  Must be
# verified against the real Antigravity endpoint before public launch — the
# served model ids are assigned by Google and are not yet confirmed.
ANTIGRAVITY_FALLBACK_MODELS = (
    "gemini-3.5-pro",
    "gemini-3.5-flash",
)


class AntigravityProfile(ProviderProfile):
    """Antigravity — Google-account OAuth via Cloud Code Assist."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """No extra_body: the CCA envelope is built by the transport client."""
        return {}

    def get_max_tokens(self, model: str | None) -> int | None:
        # Antigravity model output caps are enforced by the backend; do not
        # impose a Hermes-side default cap (mirrors gemini's behavior).
        return None


antigravity = AntigravityProfile(
    name="antigravity",
    aliases=("google-antigravity", "antigravity-oauth"),
    display_name="Google Antigravity",
    description="Google Antigravity (Cloud Code Assist, Google account OAuth)",
    api_mode="chat_completions",
    auth_type="oauth_external",
    base_url=ANTIGRAVITY_CCA_BASE_URL,
    env_vars=(),
    hidden=True,
    supports_health_check=False,
    fallback_models=ANTIGRAVITY_FALLBACK_MODELS,
)

register_provider(antigravity)


def _antigravity_gate() -> bool:
    """Enable predicate — defers to the single auth-side gate.

    Registered so ``list_providers()`` and ``antigravity_enabled()`` read the
    SAME credential resolver, keeping the provider's visibility and its
    auth/runtime usability in lockstep (never surfaced-but-dead).
    """
    from hermes_cli.antigravity_auth import antigravity_enabled

    return antigravity_enabled()


register_hidden_provider_gate("antigravity", _antigravity_gate)

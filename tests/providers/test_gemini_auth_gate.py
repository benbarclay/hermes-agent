"""Tests for the Gemini Auth provider gate.

With NAS owning the Google client config, the provider is enabled by default
(no local client_id gate). Verifies:
 1. The gemini-auth profile is present in list_providers() by default.
 2. It resolves by name via get_provider_profile() for explicit
    ``model.provider: gemini-auth`` config.
 3. The profile's auth_type is oauth_external so it is handled as an OAuth
    provider.
"""

from __future__ import annotations

import sys

import pytest


REPO_ROOT = __file__.rsplit("/tests/providers/", 1)[0]


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if mod.startswith("plugins.model_providers") or mod.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("GEMINI_AUTH_CLIENT_ID", raising=False)
    _clear_provider_caches()
    yield
    _clear_provider_caches()


def test_gemini_auth_surfaced_by_default():
    """The provider is enabled by default (no local client_id gate)."""
    from providers import list_providers

    names = {p.name for p in list_providers()}
    assert "gemini-auth" in names


def test_gemini_auth_resolves_by_name():
    """get_provider_profile() resolves the profile for explicit config."""
    from providers import get_provider_profile

    prof = get_provider_profile("gemini-auth")
    assert prof is not None
    assert prof.name == "gemini-auth"
    assert prof.auth_type == "oauth_external"  # OAuth-handled provider


def test_gemini_auth_surfaced_when_configured(monkeypatch):
    """Regardless of GEMINI_AUTH_CLIENT_ID, the provider is surfaced."""
    monkeypatch.setenv("GEMINI_AUTH_CLIENT_ID", "test-client-id-123")
    from providers import list_providers

    names = {p.name for p in list_providers()}
    assert "gemini-auth" in names


def test_gemini_auth_surfaced_via_dotenv_only(tmp_path, monkeypatch):
    """A credential in ~/.hermes/.env still surfaces the provider."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / ".env").write_text("GEMINI_AUTH_CLIENT_ID=dotenv-client-id\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("GEMINI_AUTH_CLIENT_ID", raising=False)

    from providers import list_providers

    names = {p.name for p in list_providers()}
    assert "gemini-auth" in names

    from hermes_cli.gemini_auth import gemini_auth_enabled

    assert gemini_auth_enabled() is True  # auth gate agrees with discovery


def test_gemini_auth_visible_with_include_hidden():
    """include_hidden=True still includes it."""
    from providers import list_providers

    names = {p.name for p in list_providers(include_hidden=True)}
    assert "gemini-auth" in names


def test_gemini_auth_profile_metadata():
    """Profile carries the right endpoints and fallback models."""
    from providers import get_provider_profile

    prof = get_provider_profile("gemini-auth")
    assert prof is not None
    assert "generativelanguage.googleapis.com" in (prof.base_url or "")
    assert prof.env_vars == ()
    assert prof.fallback_models  # curated list present for the picker
    assert prof.supports_health_check is False  # no /models catalog to probe

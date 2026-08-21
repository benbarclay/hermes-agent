"""Tests for the Antigravity provider gate (secret launch).

Verifies:
 1. The antigravity profile exists but is HIDDEN from list_providers() by
    default (no discovery-surface leakage — picker, setup, auth lists).
 2. It still resolves by name via get_provider_profile() so an explicit
    ``model.provider: antigravity`` config works.
 3. Presence of ANTIGRAVITY_CLIENT_ID (the enable gate) surfaces it.
 4. The profile's auth_type is oauth_external so the structural skips in
    models.py / auth.py registry auto-extend also keep it out of pickers.
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
def _clean_env(monkeypatch):
    monkeypatch.delenv("ANTIGRAVITY_CLIENT_ID", raising=False)
    _clear_provider_caches()
    yield
    _clear_provider_caches()


def test_antigravity_hidden_by_default():
    """Without the credential, antigravity never appears in list_providers()."""
    from providers import list_providers

    names = {p.name for p in list_providers()}
    assert "antigravity" not in names


def test_antigravity_resolves_by_name_when_hidden():
    """get_provider_profile() still resolves the profile for explicit config."""
    from providers import get_provider_profile

    prof = get_provider_profile("antigravity")
    assert prof is not None
    assert prof.name == "antigravity"
    assert prof.hidden is True
    assert prof.auth_type == "oauth_external"  # structural picker skip


def test_antigravity_surfaced_when_configured(monkeypatch):
    """Setting ANTIGRAVITY_CLIENT_ID flips the enable gate."""
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_ID", "test-client-id-123")
    from providers import list_providers

    names = {p.name for p in list_providers()}
    assert "antigravity" in names


def test_antigravity_visible_with_include_hidden():
    """include_hidden=True bypasses the gate entirely."""
    from providers import list_providers

    names = {p.name for p in list_providers(include_hidden=True)}
    assert "antigravity" in names


def test_antigravity_profile_metadata():
    """Profile carries the right endpoints and fallback models."""
    from providers import get_provider_profile

    prof = get_provider_profile("antigravity")
    assert prof is not None
    assert "cloudcode-pa.googleapis.com" in (prof.base_url or "")
    assert prof.env_vars == ("ANTIGRAVITY_CLIENT_ID",)
    assert prof.fallback_models  # curated list present for the picker
    assert prof.supports_health_check is False  # no /models catalog to probe

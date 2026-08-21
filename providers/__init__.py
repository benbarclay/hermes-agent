"""Provider module registry.

Provider profiles can live in two places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first call to ``get_provider_profile()`` or
``list_providers()`` scans both locations and imports every plugin. User
plugins override bundled plugins on name collision (last-writer-wins), so
third parties can monkey-patch or replace any built-in profile without
editing the repo.

For backward compatibility, ``providers/*.py`` files (other than ``base.py``
and ``__init__.py``) are still discovered via ``pkgutil.iter_modules``.
This lets out-of-tree users drop a single-file profile into an editable
install without the plugin dir structure. New profiles should prefer the
plugin layout.

Usage::

    from providers import get_provider_profile
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
_discovered = False

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    global _PROVIDER_LIST_CACHE
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
    _PROVIDER_LIST_CACHE = None


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    if not _discovered:
        _discover_providers()
    canonical = _ALIASES.get(name, name)
    return _REGISTRY.get(canonical)


def _hidden_provider_enabled(name: str) -> bool:
    """Return True when a hidden provider should be surfaced after all.

    A hidden provider stays hidden unless something explicitly enables it.
    The default enable predicate is credential presence: if any of the
    provider's ``env_vars`` resolves to a usable value in the environment
    (or ``~/.hermes/.env``), the provider is considered enabled.  Providers
    can register a richer predicate via ``register_hidden_provider_gate``.

    This mirrors the house "feature registers when configured" pattern
    (relay_url / proxy_url): absence = invisible + inert, presence = active.
    """
    try:
        gate = _HIDDEN_GATES.get(name)
        if gate is not None:
            return bool(gate())
    except Exception:
        logger.debug("hidden gate %s raised; treating as disabled", name, exc_info=True)
        return False

    profile = _REGISTRY.get(name)
    if profile is None:
        return False
    try:
        from hermes_cli.auth import has_usable_secret

        for var in profile.env_vars or ():
            if var and has_usable_secret(_resolve_env_var(var)):
                return True
    except Exception:
        logger.debug(
            "hidden env check for %s failed; staying hidden", name, exc_info=True
        )
    return False


def _resolve_env_var(var: str) -> str:
    """Resolve an env var using the canonical Hermes credential resolver.

    ``get_env_value_prefer_dotenv`` prefers ``~/.hermes/.env`` (honoring the
    documented ``export VAR=`` and inline-comment forms) then falls back to
    ``os.environ`` — so hidden-provider enable checks read the same source a
    user is told to configure.
    """
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        return get_env_value_prefer_dotenv(var) or ""
    except Exception:
        import os

        return os.getenv(var, "")


_HIDDEN_GATES: dict[str, Any] = {}


def register_hidden_provider_gate(name: str, predicate) -> None:
    """Register a callable predicate that decides when *name* is surfaced.

    ``predicate`` takes no args and returns a truthy value when the hidden
    provider should appear in ``list_providers()`` output.  Useful when
    credential presence isn't the right signal (e.g. a config flag or a
    combination of settings).  Overrides the default env-var check.
    """
    _HIDDEN_GATES[name] = predicate


def list_providers(*, include_hidden: bool = False) -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name).

    Hidden profiles (``profile.hidden`` True) are excluded unless
    ``include_hidden`` is True or the provider's own enable gate passes
    (``_hidden_provider_enabled``).  The full list is cached for
    performance; the hidden filter is applied per call so a gate flip
    (e.g. a credential appearing in the environment) is reflected without
    cache invalidation.
    """
    global _PROVIDER_LIST_CACHE
    if not _discovered:
        _discover_providers()
    if _PROVIDER_LIST_CACHE is None:
        # Deduplicate: _REGISTRY has canonical names; _ALIASES points to same objects
        seen: set[int] = set()
        result: list[ProviderProfile] = []
        for profile in _REGISTRY.values():
            pid = id(profile)
            if pid not in seen:
                seen.add(pid)
                result.append(profile)
        _PROVIDER_LIST_CACHE = result
    if include_hidden:
        return list(_PROVIDER_LIST_CACHE)
    return [
        profile
        for profile in _PROVIDER_LIST_CACHE
        if not getattr(profile, "hidden", False)
        or _hidden_provider_enabled(profile.name)
    ]


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _import_plugin_dir(plugin_dir: Path, source: str) -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user", used only for log messages.
    """
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` with a unique module name so
    # multiple HERMES_HOME profiles don't alias each other.
    safe_name = plugin_dir.name.replace("-", "_")
    if source == "bundled":
        module_name = f"plugins.model_providers.{safe_name}"
    else:
        module_name = f"_hermes_user_provider_{safe_name}"

    if module_name in sys.modules:
        return  # already imported

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)


def _discover_providers() -> None:
    """Populate the registry by importing every provider plugin.

    Order:
      1. Bundled plugins at ``<repo>/plugins/model-providers/<name>/``
      2. User plugins at ``$HERMES_HOME/plugins/model-providers/<name>/``
      3. Legacy per-file modules at ``providers/<name>.py`` (back-compat)

    Each step imports its plugins, which call ``register_provider()`` at
    module-level. Later steps win on name collision.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    # 1. Bundled plugins — shipped with hermes-agent.
    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    # 2. User plugins — under $HERMES_HOME/plugins/model-providers/<name>/.
    #    These can override any bundled profile of the same name (last-writer-wins
    #    in register_provider()).
    user_dir = _user_plugins_dir()
    if user_dir is not None:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "user")

    # 3. Legacy single-file profiles at providers/<name>.py. Kept for
    #    back-compat — if someone drops a ``providers/foo.py`` into an
    #    editable install, it still works without the plugin layout.
    try:
        import pkgutil

        import providers as _pkg

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                importlib.import_module(f"providers.{modname}")
            except ImportError as exc:
                logger.warning(
                    "Failed to import legacy provider module %s: %s", modname, exc
                )
    except Exception:
        pass

"""Session storage provider discovery: bundled ``plugins/session_storage/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). The local backend is core
(``hermes_state_port_local``), not discovered here; one provider is active (``sessions.storage``
in config.yaml, empty = local). Hosted backends are standalone plugins built on the
``hermes_state_port`` ABCs and run the shared suite in ``tests/conformance/storage``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from hermes_state_port import SessionStorageProvider

logger = logging.getLogger(__name__)

_PLUGINS_DIR = Path(__file__).parent
# Synthetic parent package for user-installed providers (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_session_storage"
_BASE_REF = "hermes_state_port:SessionStorageProvider"


def _is_provider_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the session storage contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8-sig")[:8192]
    except OSError:
        return False
    return "register_session_storage" in source or "SessionStorageProvider" in source


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    bundled = _PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    user_dir = _loader.user_plugins_dir()
    user = user_dir / name if user_dir else None
    return user if user and user.is_dir() and _is_provider_dir(user) else None


def load_session_storage(name: str) -> Optional[SessionStorageProvider]:
    """Load a provider instance by name; ``None`` if not found or it fails to load."""
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Session storage provider '%s' not found in bundled or user plugins", name)
        return None
    return _loader.load_named(name, provider_dir, _load_provider_from_dir,
                              kind="Session storage provider", noun="provider", logger=logger)


def _load_provider_from_dir(provider_dir: Path) -> Optional[SessionStorageProvider]:
    """Import a provider module and extract its provider (``register(ctx)`` or a subclass)."""
    from hermes_state_port import SessionStorageProvider
    name = provider_dir.name
    is_bundled = provider_dir.parent == _PLUGINS_DIR
    module_name = f"plugins.session_storage.{name}" if is_bundled else f"{_USER_NAMESPACE}.{name}"
    from hermes_cli.plugin_isolation import user_plugin_host
    host = None if is_bundled else user_plugin_host()
    if host is not None:
        return host.load_instance(provider_dir, module_name=module_name, capture="register_session_storage",
                                  base_ref=_BASE_REF)
    mod = _loader.load_plugin_module(
        module_name, provider_dir, parents=("plugins", "plugins.session_storage"), logger=logger,
        synthetic_namespace=None if is_bundled else _USER_NAMESPACE)
    return mod and _loader.instance_from_module(
        mod, collector=_ProviderCollector(), collected_attr="provider",
        base_cls=SessionStorageProvider, name=name, logger=logger)


class _ProviderCollector(_loader.NoopPluginContext):
    """Fake plugin context that captures ``register_session_storage`` calls."""

    def __init__(self):
        self.provider = None

    def register_session_storage(self, provider):
        self.provider = provider

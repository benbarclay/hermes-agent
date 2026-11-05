"""Session storage ports: the session/message store and turn admission, as one plugin kind.

Two ports at the level today's call sites use, never SQL:

- :class:`SessionStore`: session rows and their transcripts (create, read, append, end, titles,
  listing, deletion).
- :class:`TurnAdmission`: the profile owner lock, the owner epoch, and each session's FIFO of
  admitted inputs (admit, claim, settle, cancel, resolve) whose claims issue the turn fences
  (``hermes_scope.Fence``).

A :class:`SessionStorageProvider` supplies both. ``sessions.storage`` in config.yaml selects it:
empty means the local backend (``hermes_state_port_local``: today's per-profile ``state.db``, the
ledger in ``hermes_state_runtime`` and the ``gateway.lock`` owner lock); any other name is a
provider from ``plugins/session_storage/<name>/`` or ``$HERMES_HOME/plugins/<name>/``. A named
provider that is missing or unavailable is an error, never a fallback to the local store: a hosted
deployment's sessions written to a local file would be a silent data leak.

Every operation reads its pin (tenant, agent, owner, actor; for a turn also the session, turn and
fence) from ``hermes_scope.current_scope()`` when it runs, never from a path, ``HERMES_HOME`` or an
argument; a port exposes no way to pin anything else. A backend refuses a pin it cannot serve
with :class:`StoragePinRefused`, and a turn-path write with no claimed turn bound with
``TurnFenceRequiredError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hermes_scope import Fence

# Sentinel: "claim whatever is first" (``hermes_state_runtime.claim_session_input``'s default).
ANY_HEAD: Any = object()

# The ownership levels a store can hold (``Scope`` docs): participant-on-agent, agent, owner-wide,
# tenant-wide. A backend declares which it stores and refuses writes at any other.
LEVELS = frozenset({"participant", "agent", "owner", "tenant"})


class StoragePinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""


class SessionStore(ABC):
    """Session rows and their transcripts, for the pinned agent."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """The session row, or ``None`` (missing and foreign look the same)."""

    @abstractmethod
    def create(self, session_id: str, source: str, **fields: Any) -> str:
        """Create (or upsert) a session row; returns its id."""

    @abstractmethod
    def end(self, session_id: str, reason: str) -> None:
        """Mark an open session ended."""

    @abstractmethod
    def append(self, session_id: str, messages: list[dict[str, Any]]) -> int:
        """Append *messages* in one transaction outside any turn (notes, imports); returns the
        count written. Inside a claimed turn of this store the turn's fence still applies."""

    @abstractmethod
    def append_turn(self, session_id: str, messages: list[dict[str, Any]]) -> int:
        """Append the running turn's *messages* in one transaction, fenced by its claim. Refused
        (``TurnFenceRequiredError``) when no claimed turn of *session_id* on this store is bound."""

    @abstractmethod
    def read_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """The session's active transcript, oldest first."""

    @abstractmethod
    def get_title(self, session_id: str) -> Optional[str]:
        """The session's title, or ``None``."""

    @abstractmethod
    def set_title(self, session_id: str, title: str) -> bool:
        """Set a user title; ``False`` when the session does not exist."""

    @abstractmethod
    def list_sessions(self, *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """The pinned agent's sessions, newest first."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session and its transcript; ``False`` when there was nothing to delete."""


class OwnerHandle(ABC):
    """A held owner lock (:meth:`TurnAdmission.acquire_owner`)."""

    @abstractmethod
    def release(self) -> None:
        """Give the lock up. Idempotent."""


class TurnAdmission(ABC):
    """The owner lock, owner epoch and per-session FIFO of admitted inputs, for the pinned agent.

    A claim issues the turn's :class:`~hermes_scope.Fence` (the owner epoch and the session's
    runtime generation it stamps); every later write of that turn is checked against it."""

    @abstractmethod
    def acquire_owner(self, *, purpose: Optional[str] = None) -> OwnerHandle:
        """Become the agent's one owner, or raise if another owner holds it. Nonblocking."""

    @abstractmethod
    def begin_epoch(self, *, instance_id: str) -> int:
        """Start a new owner epoch (the caller holds the owner lock); returns it."""

    @abstractmethod
    def recover(self, *, epoch: int) -> int:
        """After a crash: every started turn of an older epoch becomes ``unknown`` (never
        replayed); returns how many."""

    @abstractmethod
    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
              payload: dict[str, Any], intent: str = "queue") -> dict[str, Any]:
        """Admit one input to the session's FIFO; idempotent by ``(principal, session, request)``."""

    @abstractmethod
    def get(self, admission_id: str) -> Optional[dict[str, Any]]:
        """One admission, or ``None``."""

    @abstractmethod
    def find(self, *, principal_id: str, session_id: str, request_id: str) -> Optional[dict[str, Any]]:
        """The admission at its durable identity, or ``None``."""

    @abstractmethod
    def list(self, session_id: str, *, pending_only: bool = True) -> list[dict[str, Any]]:
        """The session's admissions in FIFO order."""

    @abstractmethod
    def claim_next(self, *, epoch: int, session_id: str, head: Any = ANY_HEAD,
                   guard: Optional[Callable[[Any], None]] = None) -> dict[str, Any] | bool | None:
        """Claim the FIFO head: the started row (its fence stamped), ``None`` when nothing is
        claimable, ``False`` when the head is no longer *head*."""

    @abstractmethod
    def settle(self, *, epoch: int, admission_id: str, generation: int, outcome: str,
               result: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Record a claimed turn's outcome, fenced by its claim."""

    @abstractmethod
    def cancel(self, *, epoch: int, admission_id: str) -> dict[str, Any]:
        """Cancel a queued admission."""

    @abstractmethod
    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int) -> dict[str, Any]:
        """Acknowledge a turn whose owner died; never requeues it."""

    @abstractmethod
    def generation(self, session_id: str) -> Optional[int]:
        """The session's current runtime generation, or ``None``."""

    def fence(self, claimed: dict[str, Any]) -> Fence:
        """The fence a claimed row was issued under."""
        return Fence(claimed["owner_epoch"], claimed["generation"])


@dataclass(frozen=True)
class SessionStorage:
    """One opened pair of ports. ``close`` gives back whatever ``open`` took."""

    sessions: SessionStore
    admission: TurnAdmission
    close: Callable[[], None] = lambda: None


class SessionStorageProvider(ABC):
    """The plugin kind: supplies both ports. ``name`` and ``open`` are required; every other
    method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, db: Any = None) -> SessionStorage:
        """The ports for the bound scope. *db* is the local backend's per-profile handle (a
        ``SessionDB`` the caller already holds); a backend with no local file ignores it."""

    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a Postgres backend whose role is a superuser or has ``BYPASSRLS`` fails).
        Hosted boot calls it before serving."""
        return


def configured_storage_name() -> str:
    """``sessions.storage`` from the bound profile's config; empty means the local backend."""
    from hermes_cli.config import cfg_get, load_config_readonly
    return (cfg_get(load_config_readonly(), "sessions", "storage", default="") or "").strip()


def resolve_session_storage(name: Optional[str] = None) -> SessionStorageProvider:
    """The configured provider (*name* overrides the config). Raises ``LookupError`` for a named
    provider that is missing or unavailable: storage never falls back to another store."""
    name = configured_storage_name() if name is None else name.strip()
    if name in ("", "local"):
        from hermes_state_port_local import LocalSessionStorageProvider
        return LocalSessionStorageProvider()
    from plugins.session_storage import load_session_storage
    provider = load_session_storage(name)
    if provider is None:
        raise LookupError(f"sessions.storage {name!r}: no such session storage provider")
    if not provider.is_available():
        raise LookupError(f"sessions.storage {name!r} is not available here")
    return provider

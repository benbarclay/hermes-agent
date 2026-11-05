"""The local session storage backend: today's per-profile ``state.db`` behind the storage ports.

:class:`LocalSessionStore` is ``SessionDB``; :class:`LocalTurnAdmission` is the ledger in
``hermes_state_runtime`` plus the ``gateway.lock`` owner lock (``gateway.runtime_ownership``). Each
operation is the same call the code makes today, after one check of the pin: the bound scope
(``hermes_scope.current_scope()``) must be a local one whose agent is the store's own profile home.
A local store holds one agent and only agent-level records, so any other pin (another tenant,
another profile) is refused with ``StoragePinRefused`` before anything is read or written. With no
scope bound, the pin is the ambient profile (``get_hermes_home()``), as every local caller has it
today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from hermes_scope import LOCAL_TENANT, current_scope
from hermes_state_port import (
    ANY_HEAD, OwnerHandle, SessionStorage, SessionStorageProvider, SessionStore, StoragePinRefused,
    TurnAdmission,
)


class _Pinned:
    """A port over one ``SessionDB`` that refuses any pin but its own profile home."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.home = Path(db.db_path).parent

    def _pin(self) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise StoragePinRefused(f"a local session store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != hermes_home_key(self.home):
            raise StoragePinRefused("the bound agent is not this session store's profile")


class LocalSessionStore(_Pinned, SessionStore):
    def get(self, session_id):
        self._pin()
        return self.db.get_session(session_id)

    def create(self, session_id, source, **fields):
        self._pin()
        return self.db.create_session(session_id, source, **fields)

    def end(self, session_id, reason):
        self._pin()
        self.db.end_session(session_id, reason)

    def append(self, session_id, messages):
        self._pin()
        return self.db.append_messages_batch(session_id, messages)

    def append_turn(self, session_id, messages):
        self._pin()
        from hermes_state_errors import TurnFenceRequiredError
        from hermes_state_fence import store_turn
        mark = store_turn(self.db)
        if mark is None or mark.session != session_id:
            raise TurnFenceRequiredError(
                f"a turn write to session {session_id!r} needs that session's claimed turn bound")
        # The fence itself is checked by SessionDB's write transaction (hermes_state_fence).
        return self.db.append_messages_batch(session_id, messages)

    def read_conversation(self, session_id):
        self._pin()
        return self.db.get_messages_as_conversation(session_id)

    def get_title(self, session_id):
        self._pin()
        return self.db.get_session_title(session_id)

    def set_title(self, session_id, title):
        self._pin()
        return self.db.set_session_title(session_id, title)

    def list_sessions(self, *, limit=20, offset=0):
        self._pin()
        return self.db.list_sessions_rich(limit=limit, offset=offset)

    def delete(self, session_id):
        self._pin()
        return bool(self.db.delete_session(session_id))


class _OwnerLock(OwnerHandle):
    def __init__(self, ownership: Any, home: Path) -> None:
        self._ownership, self._home = ownership, home

    def release(self) -> None:
        self._ownership.release(self._home)


class LocalTurnAdmission(_Pinned, TurnAdmission):
    def acquire_owner(self, *, purpose=None):
        self._pin()
        from gateway.runtime_ownership import ProfileOwnership
        ownership = ProfileOwnership()
        ownership.reserve([self.home], purpose=purpose)
        return _OwnerLock(ownership, self.home)

    def begin_epoch(self, *, instance_id):
        self._pin()
        from hermes_state_runtime import begin_runtime_epoch
        return begin_runtime_epoch(self.db, instance_id=instance_id)

    def recover(self, *, epoch):
        self._pin()
        from hermes_state_runtime import recover_session_inputs
        return recover_session_inputs(self.db, epoch=epoch)

    def admit(self, *, epoch, principal_id, session_id, request_id, payload, intent="queue"):
        self._pin()
        from hermes_state_runtime import admit_session_input
        return admit_session_input(self.db, epoch=epoch, principal_id=principal_id, session_id=session_id,
                                   request_id=request_id, payload=payload, intent=intent)

    def get(self, admission_id):
        self._pin()
        from hermes_state_runtime import get_session_admission
        return get_session_admission(self.db, admission_id=admission_id)

    def find(self, *, principal_id, session_id, request_id):
        self._pin()
        from hermes_state_runtime import find_session_admission
        return find_session_admission(self.db, principal_id=principal_id, session_id=session_id,
                                      request_id=request_id)

    def list(self, session_id, *, pending_only=True):
        self._pin()
        from hermes_state_runtime import list_session_admissions
        return list_session_admissions(self.db, session_id=session_id, pending_only=pending_only)

    def claim_next(self, *, epoch, session_id, head=ANY_HEAD, guard: Optional[Callable[[Any], None]] = None):
        self._pin()
        from hermes_state_runtime import claim_session_input
        if head is ANY_HEAD:
            return claim_session_input(self.db, epoch=epoch, session_id=session_id, _guard=guard)
        return claim_session_input(self.db, epoch=epoch, session_id=session_id, _guard=guard, head=head)

    def settle(self, *, epoch, admission_id, generation, outcome, result=None):
        self._pin()
        from hermes_state_runtime import settle_session_input
        return settle_session_input(self.db, epoch=epoch, admission_id=admission_id, generation=generation,
                                    outcome=outcome, result=result)

    def cancel(self, *, epoch, admission_id):
        self._pin()
        from hermes_state_runtime import cancel_session_input
        return cancel_session_input(self.db, epoch=epoch, admission_id=admission_id)

    def resolve_unknown(self, *, epoch, admission_id, generation):
        self._pin()
        from hermes_state_runtime import resolve_unknown_session_input
        return resolve_unknown_session_input(self.db, epoch=epoch, admission_id=admission_id,
                                             generation=generation)

    def generation(self, session_id):
        self._pin()
        row = self.db.get_session(session_id)
        return None if row is None else row["runtime_generation"]


class LocalSessionStorageProvider(SessionStorageProvider):
    """Today's storage: one ``state.db`` per profile home, holding that agent's records only."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, db: Any = None) -> SessionStorage:
        """Ports over *db*, or over the bound profile's shared ``state.db`` (released on
        ``close``). The pin is checked by every operation, not here."""
        if db is not None:
            return SessionStorage(LocalSessionStore(db), LocalTurnAdmission(db))
        from hermes_constants import get_hermes_home
        from hermes_state_registry import acquire, release
        shared = acquire(get_hermes_home() / "state.db")

        def close() -> None:
            release(shared)
        return SessionStorage(LocalSessionStore(shared), LocalTurnAdmission(shared), close)

    def assert_isolating(self) -> None:
        """A local store is one file per profile and refuses every pin but its own (``_Pinned``);
        there is no shared role that could bypass that."""
        return

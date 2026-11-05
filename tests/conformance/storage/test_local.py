"""The storage conformance suite against the local backend (today's per-profile ``state.db``).

Agents are profile homes under ``tmp_path``; a replica is a second ``SessionDB`` on the same file
(another process's connection). A local store holds the agent level only, so the ownership-level
invariants are covered by the refusal tests in ``tests/hermes_state/test_storage_port.py``.
"""

from contextlib import closing, contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn
from hermes_scope import Group, Principal, Scope
from hermes_state import SessionDB
from hermes_state_port_local import LocalSessionStorageProvider
from tests.conformance.storage.suite import Backend, StorageConformance


class TestLocalBackend(StorageConformance):
    @pytest.fixture
    def backend(self, tmp_path, monkeypatch):
        handles = []
        homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
        for home in homes.values():
            home.mkdir()

        def connect(agent):
            db = SessionDB(homes[agent] / "state.db")
            handles.append(db)
            return LocalSessionStorageProvider().open(db)

        @contextmanager
        def unbound():
            # No scope: a local store's pin is the ambient profile, here one that owns no store.
            monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
            yield

        @contextmanager
        def turn(storage, session_id, claimed):
            db = storage.sessions.db
            with bind_turn(session_id, claimed["admission_id"], storage.admission.fence(claimed),
                           store=db.db_path):
                yield

        def foreign_tenant(agent):
            return bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"), role="member"))

        yield Backend(open=connect, replica=connect, bind=lambda agent: bind_profile_scope(homes[agent]),
                      foreign_tenant=foreign_tenant, unbound=unbound, turn=turn)
        for db in handles:
            with closing(db):
                pass

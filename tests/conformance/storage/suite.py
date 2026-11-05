"""The storage conformance suite: isolation and turn-admission invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks to
the ports only (``hermes_state_port``); the fixture says how to bind each agent's scope and open
its storage. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from hermes_state_errors import SessionTurnLeaseLostError, TurnFenceRequiredError
from hermes_state_port import SessionStorage, StoragePinRefused


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's ports.
    - ``replica(agent)``: an independent second handle on the same agent's storage (another
      replica's connection).
    - ``bind(agent)``: bind the agent's scope (a non-turn activity of its owner).
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing.
    - ``turn(storage, session_id, claimed)``: bind the claimed turn's fence for the body.
    """

    open: Callable[[str], SessionStorage]
    replica: Callable[[str], SessionStorage]
    bind: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    turn: Callable[[SessionStorage, str, dict], AbstractContextManager]


def _elsewhere(fn: Callable[[], Any]) -> Any:
    """Run *fn* in a fresh context: another writer, outside this test's turn."""
    return contextvars.Context().run(fn)


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(StoragePinRefused) as refused:
        call()
    return str(refused.value)


def _session(backend, agent, sid="s1", text=None):
    storage = backend.open(agent)
    with backend.bind(agent):
        storage.sessions.create(sid, "cli")
        if text is not None:
            storage.sessions.append(sid, [{"role": "user", "content": text}])
    return storage


def _owned_claim(backend, agent, sid="s1", requests=("r1",)):
    """A session of *agent* with *requests* admitted under a fresh owner epoch, the head claimed."""
    storage = _session(backend, agent, sid)
    with backend.bind(agent):
        epoch = storage.admission.begin_epoch(instance_id="owner")
        for request in requests:
            storage.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id=request,
                                    payload={"text": request})
        claimed = storage.admission.claim_next(epoch=epoch, session_id=sid)
    return storage, epoch, claimed


def _texts(backend, agent, storage, sid="s1"):
    with backend.bind(agent):
        return [m["content"] for m in storage.sessions.read_conversation(sid)]


class StorageConformance:
    """Inherit and supply a ``backend`` fixture."""

    # --- pinning ---

    def test_with_no_scope_pinned_reads_see_nothing_and_writes_fail(self, backend):
        storage = _session(backend, "a", text="hello")
        with backend.unbound():
            _refused(lambda: storage.sessions.get("s1"))
            _refused(lambda: storage.sessions.read_conversation("s1"))
            _refused(lambda: storage.sessions.create("s2", "cli"))
            _refused(lambda: storage.admission.begin_epoch(instance_id="x"))
        with backend.bind("a"):
            assert storage.sessions.get("s2") is None

    def test_another_agent_never_sees_or_writes_the_agents_rows(self, backend):
        mine = _session(backend, "a", text="private")
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.sessions.get("s1") is None
            assert theirs.sessions.list_sessions() == []
            _refused(lambda: mine.sessions.append("s1", [{"role": "user", "content": "x"}]))
            _refused(lambda: mine.admission.list("s1"))
        assert _texts(backend, "a", mine) == ["private"]

    def test_a_foreign_row_is_indistinguishable_from_a_missing_one(self, backend):
        mine = _session(backend, "a")
        with backend.bind("b"):
            assert _refused(lambda: mine.sessions.get("s1")) == _refused(lambda: mine.sessions.get("absent"))
            assert (_refused(lambda: mine.sessions.get_title("s1"))
                    == _refused(lambda: mine.sessions.get_title("absent")))

    def test_pinning_an_agent_never_widens_the_tenant(self, backend):
        mine = _session(backend, "a")
        with backend.foreign_tenant("a"):
            _refused(lambda: mine.sessions.get("s1"))
            _refused(lambda: mine.sessions.create("s2", "cli"))

    # --- fenced turn writes ---

    def test_a_turn_path_write_with_no_fence_is_refused(self, backend):
        storage, _epoch, _claimed = _owned_claim(backend, "a")
        with backend.bind("a"), pytest.raises(TurnFenceRequiredError):
            storage.sessions.append_turn("s1", [{"role": "assistant", "content": "unfenced"}])
        assert _texts(backend, "a", storage) == []

    def test_a_turn_holding_its_lease_writes_and_settles(self, backend):
        storage, epoch, claimed = _owned_claim(backend, "a")
        with backend.bind("a"):
            with backend.turn(storage, "s1", claimed):
                assert storage.sessions.append_turn("s1", [{"role": "assistant", "content": "ok"}]) == 1
            settled = storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                               generation=claimed["generation"], outcome="completed")
        assert settled["status"] == "terminal"
        assert _texts(backend, "a", storage) == ["ok"]

    def test_a_write_under_a_superseded_owner_epoch_is_rejected_whole(self, backend):
        storage, _epoch, claimed = _owned_claim(backend, "a")
        other = backend.replica("a")

        def take_over():
            with backend.bind("a"):
                other.admission.begin_epoch(instance_id="successor")
        with backend.bind("a"), backend.turn(storage, "s1", claimed):
            _elsewhere(take_over)
            with pytest.raises(SessionTurnLeaseLostError):
                storage.sessions.append_turn("s1", [{"role": "assistant", "content": "a"},
                                                    {"role": "assistant", "content": "b"}])
        assert _texts(backend, "a", storage) == []

    def test_a_write_under_a_superseded_generation_is_rejected_whole(self, backend):
        storage, epoch, claimed = _owned_claim(backend, "a", requests=("r1", "r2"))
        other = backend.replica("a")

        def next_turn():  # the owner settles this turn and claims the next one
            with backend.bind("a"):
                other.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                       generation=claimed["generation"], outcome="interrupted")
                return other.admission.claim_next(epoch=epoch, session_id="s1")
        with backend.bind("a"), backend.turn(storage, "s1", claimed):
            successor = _elsewhere(next_turn)
            assert successor["generation"] > claimed["generation"]
            with pytest.raises(SessionTurnLeaseLostError):
                storage.sessions.append_turn("s1", [{"role": "assistant", "content": "stale"}])
        assert _texts(backend, "a", storage) == []

    # --- turn admission ---

    def test_one_fifo_per_session(self, backend):
        storage, epoch, first = _owned_claim(backend, "a", requests=("r1", "r2", "r3"))
        order = [first["request_id"]]
        with backend.bind("a"):
            assert storage.admission.claim_next(epoch=epoch, session_id="s1") is None  # one at a time
            claimed = first
            while claimed:
                storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                         generation=claimed["generation"], outcome="completed")
                claimed = storage.admission.claim_next(epoch=epoch, session_id="s1")
                if claimed:
                    order.append(claimed["request_id"])
        assert order == ["r1", "r2", "r3"]

    def test_one_writer_per_session_across_two_replicas(self, backend):
        first = _session(backend, "a")
        second = backend.replica("a")
        with backend.bind("a"):
            epoch = first.admission.begin_epoch(instance_id="owner")
            for request in ("r1", "r2"):  # work for both: only the one-writer rule stops the second
                first.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id=request,
                                      payload={"text": request})
        start, results = threading.Barrier(2), []

        def claim(storage):
            with backend.bind("a"):
                start.wait(10)
                results.append(storage.admission.claim_next(epoch=epoch, session_id="s1"))
        threads = [threading.Thread(target=contextvars.Context().run, args=(claim, s)) for s in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(bool(r) for r in results) == [False, True]

    def test_two_sessions_of_one_agent_run_side_by_side(self, backend):
        first = _session(backend, "a", "s1")
        _session(backend, "a", "s2")
        storages = {"s1": first, "s2": backend.replica("a")}  # each session on its own replica
        claims = {}
        with backend.bind("a"):
            epoch = first.admission.begin_epoch(instance_id="owner")
            for sid in storages:
                first.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id="r",
                                      payload={"text": sid})
                claims[sid] = first.admission.claim_next(epoch=epoch, session_id=sid)
        start, errors = threading.Barrier(2), []

        def run(sid):
            storage, claimed = storages[sid], claims[sid]
            try:
                with backend.bind("a"), backend.turn(storage, sid, claimed):
                    start.wait(10)
                    for n in range(20):
                        storage.sessions.append_turn(sid, [{"role": "assistant", "content": f"{sid}-{n}"}])
            except Exception as exc:  # surfaced below with the thread's own failure
                errors.append(exc)
        threads = [threading.Thread(target=contextvars.Context().run, args=(run, sid)) for sid in storages]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert errors == []
        for sid in storages:
            assert _texts(backend, "a", first, sid) == [f"{sid}-{n}" for n in range(20)]

    def test_a_turn_whose_owner_died_becomes_unknown_and_is_never_replayed(self, backend):
        _storage, _epoch, claimed = _owned_claim(backend, "a", requests=("r1", "r2"))
        successor = backend.replica("a")
        with backend.bind("a"):
            epoch = successor.admission.begin_epoch(instance_id="successor")
            assert successor.admission.recover(epoch=epoch) == 1
            assert successor.admission.get(claimed["admission_id"])["status"] == "unknown"
            try:
                replayed = successor.admission.claim_next(epoch=epoch, session_id="s1")
            except Exception:  # a backend may refuse outright; it must not hand the turn out again
                replayed = None
            assert not replayed
            assert successor.admission.get(claimed["admission_id"])["status"] == "unknown"

    def test_recovery_resumes_the_fifo_after_the_lost_turn_is_resolved(self, backend):
        _storage, _epoch, claimed = _owned_claim(backend, "a", requests=("r1", "r2"))
        successor = backend.replica("a")
        with backend.bind("a"):
            epoch = successor.admission.begin_epoch(instance_id="successor")
            successor.admission.recover(epoch=epoch)
            resolved = successor.admission.resolve_unknown(epoch=epoch, admission_id=claimed["admission_id"],
                                                           generation=claimed["generation"])
            assert resolved["status"] == "terminal"
            follower = successor.admission.claim_next(epoch=epoch, session_id="s1")
        assert follower and follower["request_id"] == "r2" and follower["owner_epoch"] == epoch

    def test_a_stale_owner_cannot_settle(self, backend):
        storage, epoch, claimed = _owned_claim(backend, "a")
        with backend.bind("a"):
            backend.replica("a").admission.begin_epoch(instance_id="successor")
            with pytest.raises(Exception):  # the backend's stale-owner refusal
                storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                         generation=claimed["generation"], outcome="completed")
            assert storage.admission.get(claimed["admission_id"])["status"] == "started"

"""Broken local backends the storage conformance suite must reject (``test_mutants.py``).

The first five are the adversarial review's: each passed the earlier suite. ``preflight_fence``
checks the turn fence before the write's transaction instead of inside it. Each mutant is applied
by the ``backend`` fixture when ``HERMES_STORAGE_MUTANT`` names it.
"""

import contextvars
from types import SimpleNamespace


def _foreign_metadata(monkeypatch):
    """set_title and delete skip the pin check."""
    from hermes_state_port_local import LocalSessionStore
    monkeypatch.setattr(LocalSessionStore, "set_title", lambda self, sid, title: self.db.set_session_title(sid, title))
    monkeypatch.setattr(LocalSessionStore, "delete", lambda self, sid: bool(self.db.delete_session(sid)))


def _ignored_claim_preconditions(monkeypatch):
    """claim_next discards the validated head and the admission gate."""
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.claim_next
    monkeypatch.setattr(LocalTurnAdmission, "claim_next",
                        lambda self, *, epoch, session_id, preconditions=None: original(
                            self, epoch=epoch, session_id=session_id))


def _unfenced_ordinary_append(monkeypatch):
    """append writes outside the bound turn, so a superseded turn's ordinary append lands."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_state_port_local import LocalSessionStore

    def append(self, sid, messages):
        self._pin()

        def write():
            with bind_profile_scope(self.home):
                return self.db.append_messages_batch(sid, messages)
        return contextvars.Context().run(write)
    monkeypatch.setattr(LocalSessionStore, "append", append)


def _no_owner_exclusion(monkeypatch):
    """acquire_owner always succeeds with a handle that holds nothing."""
    from hermes_state_port_local import LocalTurnAdmission
    monkeypatch.setattr(LocalTurnAdmission, "acquire_owner",
                        lambda self, **kw: (self._pin(), SimpleNamespace(release=lambda: None))[1])


def _producer_partitioned_fifo(monkeypatch):
    """claim_next schedules by producer before admission order."""
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.claim_next

    def claim_next(self, *, epoch, session_id, preconditions=None):
        def reorder(conn):
            rows = conn.execute("SELECT admission_id FROM session_admissions WHERE target_session_id=? "
                                "AND status='queued' ORDER BY principal_id, seq", (session_id,)).fetchall()
            seqs = sorted(r[0] for r in conn.execute("SELECT seq FROM session_admissions WHERE target_session_id=? "
                                                     "AND status='queued'", (session_id,)).fetchall())
            for (admission_id,), seq in zip(rows, seqs):
                conn.execute("UPDATE session_admissions SET seq=? WHERE admission_id=?", (-seq, admission_id))
            conn.execute("UPDATE session_admissions SET seq=-seq WHERE target_session_id=? AND seq<0", (session_id,))
        from hermes_state_fence import writes_as
        with writes_as(None):
            self.db._execute_write(reorder)
        return original(self, epoch=epoch, session_id=session_id, preconditions=preconditions)
    monkeypatch.setattr(LocalTurnAdmission, "claim_next", claim_next)


def _preflight_fence(monkeypatch):
    """The turn fence is checked by a read before the write, and the write itself runs unfenced:
    a takeover that commits in between is missed."""
    import hermes_state
    from hermes_state_fence import check_turn_lease, store_turn
    original = hermes_state.SessionDB._execute_write

    def execute_write(self, fn, patience_s=None):
        mark = store_turn(self)
        if mark is not None:
            with self._read_ctx() as conn:
                check_turn_lease(conn, mark)
        from hermes_state_fence import writes_as
        with writes_as(None):
            return original(self, fn, patience_s)
    monkeypatch.setattr(hermes_state.SessionDB, "_execute_write", execute_write)


MUTANTS = {
    "foreign_metadata": _foreign_metadata,
    "ignored_claim_preconditions": _ignored_claim_preconditions,
    "unfenced_ordinary_append": _unfenced_ordinary_append,
    "no_owner_exclusion": _no_owner_exclusion,
    "producer_partitioned_fifo": _producer_partitioned_fifo,
    "preflight_fence": _preflight_fence,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_STORAGE_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)

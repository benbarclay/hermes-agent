"""Turn fences on session store writes.

A claimed turn runs under the fence its claim issued (``hermes_scope.Fence``: the owner epoch of the
owner that claimed it and the claimed session's runtime generation at the claim). While the turn's
body runs, ``agent.runtime_scope.bind_turn`` records it in the context (``hermes_scope.current_turn``)
together with the store that issued the claim. Every write the turn makes to that store runs
:func:`check_turn_lease` first, inside the write's own transaction (``SessionDB._execute_write``):
if the profile's owner epoch or the session's generation has moved since the claim, the turn no
longer holds its session's lease and the write is refused before anything of it lands.

Which writes are a turn's: every write made in the turn's context, to the issuing store, while the
turn's body runs. That includes the threads and tasks the turn starts (they copy the context), for
as long as the turn runs. It excludes:

- writes to another store (another profile's ``state.db``: the fence says nothing about it);
- writes after the turn's body returned: settlement, which the owner fences itself by
  ``(owner_epoch, generation)``, and any background work that outlives the turn;
- writes outside any claimed turn: CLI and TUI turns (they hold the durable turn lease instead),
  compression and maintenance the owner runs between turns, recovery, imports.

Those run under their own guards, unchanged. While the lease holds, a turn's write behaves exactly
as before; the check is two indexed reads on the writer connection.
"""

from __future__ import annotations

from hermes_state_errors import TurnFenceSupersededError

_EPOCH_SQL = "SELECT epoch FROM runtime_epoch WHERE singleton=1"
_GENERATION_SQL = "SELECT runtime_generation FROM sessions WHERE id=?"


def store_key(path) -> str:
    """The identity a turn mark and a store compare: the store file's resolved path."""
    from hermes_constants import hermes_home_key
    return hermes_home_key(path)


def store_turn(db):
    """The claimed turn running in this context when *db* is the store that issued its fence,
    else ``None``."""
    from hermes_scope import current_turn
    mark = current_turn()
    if mark is None or mark.store is None:
        return None
    path = getattr(db, "db_path", None)
    if path is None or store_key(path) != mark.store:
        return None
    return mark


def lease_holds(conn, session_id: str, fence) -> bool:
    """Whether *fence* is still *session_id*'s lease on this store: the owner epoch and the runtime
    generation the claim stamped are both still current."""
    epoch = conn.execute(_EPOCH_SQL).fetchone()
    if epoch is None or epoch[0] != fence.epoch:
        return False
    row = conn.execute(_GENERATION_SQL, (session_id,)).fetchone()
    return row is not None and row[0] == fence.generation


def check_turn_lease(conn, mark) -> None:
    """Refuse the current write of turn *mark* once it no longer holds its session's lease. Runs on
    the write's connection inside its transaction; ``None`` (no claimed turn of this store) checks
    nothing."""
    if mark is None or lease_holds(conn, mark.session, mark.fence):
        return
    raise TurnFenceSupersededError(
        f"turn {mark.turn!r} of session {mark.session!r} no longer holds its turn lease "
        f"(fence epoch={mark.fence.epoch}, generation={mark.fence.generation}); refusing its write")

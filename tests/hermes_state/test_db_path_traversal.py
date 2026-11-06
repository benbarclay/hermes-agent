"""A SessionDB opens the file its path names on the filesystem.

A path such as ``A/link/../state.db`` where ``link`` is a directory symlink into ``B/child``
names ``B/state.db``: the kernel follows ``link`` before applying ``..``. Collapsing ``..``
lexically would open (and create) ``A/state.db`` instead and show an empty history. A relative
path is still fixed to the working directory it was opened in.
"""
from contextlib import closing
from pathlib import Path

from agent.runtime_scope import bind_profile_scope
from hermes_state import SessionDB
from hermes_state_port_local import LocalSessionStorageProvider


def _layout(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (b / "child").mkdir(parents=True)
    (a / "link").symlink_to(b / "child", target_is_directory=True)
    with closing(SessionDB(b / "state.db")) as db:
        db.create_session("expected-b", "cli")
    return a, b


def test_an_absolute_path_through_a_link_and_dotdot_opens_the_file_it_names(tmp_path):
    a, b = _layout(tmp_path)
    with closing(SessionDB(a / "link" / ".." / "state.db")) as db:
        row = db.get_session("expected-b")
        assert row is not None and row["id"] == "expected-b"
        db.append_message("expected-b", "user", "through the link")
    assert not (a / "state.db").exists()
    with closing(SessionDB(b / "state.db")) as db:
        assert [m["content"] for m in db.get_messages("expected-b")] == ["through the link"]


def test_a_relative_path_through_a_link_and_dotdot_opens_the_file_it_names(tmp_path, monkeypatch):
    a, b = _layout(tmp_path)
    monkeypatch.chdir(a)
    with closing(SessionDB(Path("link") / ".." / "state.db")) as db:
        row = db.get_session("expected-b")
        assert row is not None and row["id"] == "expected-b"
        # Still the same file after the working directory moves on.
        monkeypatch.chdir(tmp_path)
        db.append_message("expected-b", "user", "after chdir")
        storage = LocalSessionStorageProvider().open(db)
        assert storage.sessions.home == b.resolve()  # type: ignore[attr-defined]
        with bind_profile_scope(b, hydrate_secrets=False):
            assert storage.sessions.get("expected-b")["id"] == "expected-b"
    assert not (a / "state.db").exists()
    assert not (tmp_path / "state.db").exists()
    with closing(SessionDB(b / "state.db")) as db:
        assert [m["content"] for m in db.get_messages("expected-b")] == ["after chdir"]

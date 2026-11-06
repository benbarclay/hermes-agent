"""A SessionDB opens the file its path names on the filesystem.

A path such as ``A/link/../state.db`` where ``link`` is a directory symlink into ``B/child``
names ``B/state.db``: the kernel follows ``link`` before applying ``..``. Collapsing ``..``
lexically would open (and create) ``A/state.db`` instead and show an empty history. A relative
path is still fixed to the working directory it was opened in.
"""
import os
import subprocess
import sys
from contextlib import closing

import pytest
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


def test_a_detour_through_a_directory_removed_later_keeps_the_handle_writable(tmp_path):
    home = tmp_path / "home"
    (home / "unused").mkdir(parents=True)
    with closing(SessionDB(home / "unused" / ".." / "state.db")) as db:
        db.create_session("s", "cli")
        (home / "unused").rmdir()
        db.append_message("s", "user", "after the detour went away")
    with closing(SessionDB(home / "state.db")) as db:
        assert [m["content"] for m in db.get_messages("s")] == ["after the detour went away"]


def test_a_database_file_link_is_kept_as_given(tmp_path):
    """Only the directory is resolved: the port's refusal of a file link into another profile
    needs to see the link."""
    real = tmp_path / "b" / "state.db"
    real.parent.mkdir()
    (tmp_path / "a").mkdir()
    link = tmp_path / "a" / "state.db"
    link.symlink_to(real)
    with closing(SessionDB(link)) as db:
        assert db.db_path == (tmp_path / "a").resolve() / "state.db"
        assert db.db_path.is_symlink()


_FROM_REMOVED_CWD = """
import os, sys
from contextlib import closing
from pathlib import Path
from hermes_state import SessionDB
file, gone, read_only = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
os.chdir(gone)
os.rmdir(gone)
with closing(SessionDB(file, read_only=read_only)) as db:
    assert db.get_session("s")["id"] == "s"
    if not read_only:
        db.append_message("s", "user", "cwd removed")
print("ok")
"""


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("read_only", [False, True])
def test_an_absolute_open_works_after_the_working_directory_was_removed(tmp_path, read_only):
    """Run in a child: the open must not need the (deleted) working directory at all, and this
    process's test filesystem guard itself resolves paths against it."""
    home = tmp_path / "home"
    home.mkdir()
    with closing(SessionDB(home / "state.db")) as db:
        db.create_session("s", "cli")
    gone = tmp_path / "gone"
    gone.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    done = subprocess.run(
        [sys.executable, "-c", _FROM_REMOVED_CWD, str(home / "state.db"), str(gone), "1" if read_only else "0"],
        capture_output=True, text=True, env=env, timeout=60, check=False)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
    if not read_only:
        with closing(SessionDB(home / "state.db")) as db:
            assert [m["content"] for m in db.get_messages("s")] == ["cwd removed"]

"""The skill-script runtime: a hosted scope runs a skill's scripts in its own sandbox,
never on this host.

The sandbox-conformance invariants (own-sandbox execution, foreign skills indistinguishable
from missing ones, the no-secrets grant) live in ``tools/environments/sandbox_conformance.py``
and run through ``test_sandbox_scope_conformance.py``. This file covers the runtime's own
edges: the seam reads only the deployment's catalog, the callers that reach it (the inline
skill shell) route through it in a hosted scope, and every local path keeps today's behaviour.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from agent.runtime_scope import bind_state
from hermes_scope import Principal, Scope
from tests.tools.test_sandbox_scope_conformance import A, B, provider as _conformance_provider
from tools.environments import sandbox_conformance as conformance
from tools.skill_script_runtime import SkillScriptRefused

provider = _conformance_provider  # the recording-sandbox fixture, by its name

LOCAL_SCOPE = Scope("local", "default", Principal("local"))

CATALOG_SKILL = "catalog-probe"
CATALOG_SCRIPT = "scripts/run.sh"


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """A deployment skill catalog: one script skill in a throwaway tree, pointed at through
    the packaged-install override the seam reads."""
    skill = tmp_path / "catalog" / CATALOG_SKILL
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: catalog-probe\ndescription: probe\n---\nbody\n")
    (skill / CATALOG_SCRIPT).write_text("#!/usr/bin/env bash\necho CATALOG-SCRIPT\n")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "catalog"))
    return skill


@pytest.fixture
def no_spawn(monkeypatch):
    spawned = []

    def refuse(*args, **kwargs):
        spawned.append(args[0] if args else kwargs.get("args"))
        raise AssertionError(f"a host process was started: {spawned[-1]}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    return spawned


# --------------------------------------------------------------------------------------------------
# The seam


def test_the_seam_reads_only_the_deployment_catalog(catalog):
    from tools.skill_script_runtime import read_skill_script

    with bind_state(A):
        content, path = read_skill_script(CATALOG_SKILL, CATALOG_SCRIPT)
    assert "CATALOG-SCRIPT" in content and path == CATALOG_SCRIPT
    with bind_state(A):
        assert read_skill_script("missing-skill", CATALOG_SCRIPT) is None
        assert read_skill_script(CATALOG_SKILL, "scripts/missing.sh") is None
        assert read_skill_script(CATALOG_SKILL, "../../SKILL.md") is None
        assert read_skill_script(CATALOG_SKILL, "/etc/passwd") is None
        assert read_skill_script("c:/absolute/path", CATALOG_SCRIPT) is None


def test_the_seam_never_opens_a_profile_tree(catalog, monkeypatch):
    """A skill planted in the process home's skills directory (a tenant's tree, which no bound
    scope of a shared-process deployment owns) is not in the catalog and is never opened."""
    from hermes_constants import get_hermes_home
    from tools.skill_script_runtime import read_skill_script

    tenant = get_hermes_home() / "skills" / "tenant-probe"
    (tenant / "scripts").mkdir(parents=True)
    (tenant / "SKILL.md").write_text("---\nname: tenant-probe\n---\nTENANT-PRIVATE\n")
    (tenant / CATALOG_SCRIPT).write_text("echo TENANT-PRIVATE\n")
    foreigner = Scope("globex", "agent-a", Principal("u9"), owner=Principal("u9"), can_administer=True)
    with conformance._watch_host(get_hermes_home() / "skills") as touched:
        for scope in (A, B, foreigner):
            with bind_state(scope):
                assert read_skill_script("tenant-probe", CATALOG_SCRIPT) is None
    assert not touched, f"the seam opened a tenant tree: {touched}"


def test_the_seam_refuses_outside_a_hosted_scope(catalog, no_spawn):
    from tools.skill_script_runtime import read_skill_script

    with pytest.raises(SkillScriptRefused):
        read_skill_script(CATALOG_SKILL, CATALOG_SCRIPT)
    with bind_state(LOCAL_SCOPE), pytest.raises(SkillScriptRefused):
        read_skill_script(CATALOG_SKILL, CATALOG_SCRIPT)
    assert no_spawn == []


# --------------------------------------------------------------------------------------------------
# Execution


def test_run_skill_script_ships_and_runs_in_the_bound_sandbox(catalog, provider):
    import base64

    from tools.skill_script_runtime import run_skill_script

    shipped = base64.b64encode((catalog / CATALOG_SCRIPT).read_text().encode("utf-8")).decode("ascii")
    seen = len(provider.calls)
    with bind_state(A):
        out = run_skill_script(CATALOG_SKILL, CATALOG_SCRIPT, argv=("--flag", "a b"), task_id="t")
    calls = provider.calls[seen:]
    assert calls and all(call.sandbox == conformance._identity(A) for call in calls)
    assert any(c.op == "execute" and "base64 -d" in c.command and shipped in c.stdin for c in calls)
    run = [c for c in calls if c.op == "execute" and c.command.startswith("bash")]
    assert run and "--flag" in run[0].command and "a b" in run[0].command
    assert isinstance(out, str)


def test_a_missing_skill_wakes_no_sandbox(catalog, provider):
    from tools.skill_script_runtime import run_skill_script

    seen_calls, seen_created = len(provider.calls), list(provider.created)
    with bind_state(A):
        missing = run_skill_script("no-such-skill", CATALOG_SCRIPT, task_id="t")
        foreign = run_skill_script("tenant-probe", CATALOG_SCRIPT, task_id="t")
    assert "no script" in missing and "no script" in foreign
    assert foreign.replace("tenant-probe", "no-such-skill") == missing
    assert len(provider.calls) == seen_calls and provider.created == seen_created


def test_run_skill_script_fails_closed_with_a_host_backend(catalog, monkeypatch, no_spawn):
    from tools.skill_script_runtime import run_skill_script

    monkeypatch.setenv("TERMINAL_ENV", "local")
    with bind_state(A):
        with pytest.raises(SkillScriptRefused, match="never runs on this host"):
            run_skill_script(CATALOG_SKILL, CATALOG_SCRIPT, task_id="t")
    assert no_spawn == []


def test_run_skill_script_refused_outside_a_hosted_scope(catalog, no_spawn):
    from tools.skill_script_runtime import run_skill_script

    with pytest.raises(SkillScriptRefused):
        run_skill_script(CATALOG_SKILL, CATALOG_SCRIPT)
    with bind_state(LOCAL_SCOPE), pytest.raises(SkillScriptRefused):
        run_skill_script(CATALOG_SKILL, CATALOG_SCRIPT)
    assert no_spawn == []


def test_unsupported_interpreters_are_refused_not_guessed(catalog, provider):
    from tools.skill_script_runtime import run_skill_script, read_skill_script

    script = catalog / "scripts" / "tool.ts"
    script.write_text("console.log('ts')")
    with bind_state(A):
        assert read_skill_script(CATALOG_SKILL, "scripts/tool.ts") is not None
        out = run_skill_script(CATALOG_SKILL, "scripts/tool.ts", task_id="t")
    assert "no sandbox interpreter" in out
    assert provider.calls == [], "an unrunnable script still woke a sandbox"


# --------------------------------------------------------------------------------------------------
# The grant: this agent's sandbox, no secrets


def test_the_run_passes_no_env_passthrough_and_no_credential_files(catalog, provider, monkeypatch):
    from tools import credential_files, env_passthrough
    from tools.skill_script_runtime import run_skill_script

    def refused(*args, **kwargs):
        raise AssertionError("a skill script run consulted the profile's secret machinery")

    monkeypatch.setattr(env_passthrough, "register_env_passthrough", refused)
    monkeypatch.setattr(credential_files, "register_credential_files", refused)
    with bind_state(A):
        run_skill_script(CATALOG_SKILL, CATALOG_SCRIPT, task_id="t")
    assert all(call.child_env is None for call in provider.calls)


def test_the_grant_refuses_scripts_the_seam_cannot_read_locally(catalog, provider, monkeypatch):
    """The seam is the only source: a script whose skill is not in the deployment catalog never
    runs, even when a file with that exact name exists on this host at a path the caller could
    name — the host copy is indistinguishable from no copy at all."""
    from hermes_constants import get_hermes_home
    from tools.skill_script_runtime import run_skill_script

    decoy = get_hermes_home() / "skills" / "decoy-skill" / CATALOG_SCRIPT
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text("#!/usr/bin/env bash\ncat /etc/passwd\n")
    with bind_state(A):
        out = run_skill_script("decoy-skill", CATALOG_SCRIPT, task_id="t")
    assert "no script" in out and provider.calls == []


# --------------------------------------------------------------------------------------------------
# The wired caller: the inline skill shell


def test_inline_skill_shell_runs_in_the_bound_sandbox(provider, tmp_path):
    from agent.skill_preprocessing import run_inline_shell

    seen = len(provider.calls)
    with bind_state(A):
        out = run_inline_shell("echo HOST-MARK", tmp_path, 5)
    calls = provider.calls[seen:]
    assert calls and all(c.sandbox == conformance._identity(A) for c in calls), calls
    assert any("echo HOST-MARK" in c.command for c in calls)
    assert out == ""  # the stub runs nothing; the point is the route, not the output


def test_inline_skill_shell_fails_closed_with_a_host_backend(monkeypatch, no_spawn, tmp_path):
    from agent.skill_preprocessing import run_inline_shell

    monkeypatch.setenv("TERMINAL_ENV", "local")
    with bind_state(A):
        out = run_inline_shell("echo HOST", tmp_path, 5)
    assert "refused" in out and "host" in out and no_spawn == []


def test_inline_skill_shell_is_unchanged_locally(monkeypatch, tmp_path):
    from agent.skill_preprocessing import run_inline_shell

    monkeypatch.setattr("tools.environments.sandbox_scope.hosted_scope", lambda: None)
    with bind_state(LOCAL_SCOPE):
        assert run_inline_shell("echo LOCAL-SCOPE", tmp_path, 5) == "LOCAL-SCOPE"
    assert run_inline_shell("echo LOCAL-PLAIN", tmp_path, 5) == "LOCAL-PLAIN"


def test_inline_snippet_of_a_foreign_agent_reaches_only_its_own_sandbox(provider):
    from tools.skill_script_runtime import run_inline_skill_shell

    with bind_state(A):
        run_inline_skill_shell("echo A-SNIPPET", timeout=5, task_id="t")
    with bind_state(B):
        run_inline_skill_shell("echo B-SNIPPET", timeout=5, task_id="t")
    a = [c for c in provider.calls if "A-SNIPPET" in c.command]
    b = [c for c in provider.calls if "B-SNIPPET" in c.command]
    assert a and b and all(c.sandbox == conformance._identity(A) for c in a)
    assert all(c.sandbox == conformance._identity(B) for c in b)
    assert len(provider.created) == 2  # one sandbox per agent, whatever task id each call carried

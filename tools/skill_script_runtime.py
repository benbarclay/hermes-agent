"""A skill's scripts run in the bound agent's sandbox — never on this host.

A skill body and the files it links (``scripts/``, templates, assets) are agent- or
tenant-supplied code. Every path that executes one in a hosted scope goes through this
module: the script's bytes come from one seam function, are copied into the bound
agent's own sandbox at exec time, and run there through the active environment
(``tools.environments.sandbox_scope``). A hosted path that would run a skill script
on this host is refused (:class:`SkillScriptRefused`), and so is one that would run
it with no hosted scope bound — a local install runs a skill's scripts with the
agent's own terminal, exactly as it does today, so no local caller changes.

The grant: **a skill script runs in this agent's own sandbox, with no secrets.** Skills
already declare ``required_environment_variables`` and ``required_credential_files`` in
their frontmatter; no hosted secret store or credential broker exists yet to satisfy
them, so this runtime passes nothing: no environment values, no credential files, no
broker references, no host paths. Widening the grant later (a broker reference, an
egress allowlist) is a change here, in one place, with the grant tests below it.

The seam: :func:`read_skill_script` is the one function a skill script's bytes come
through. Today it reads today's skill files — the deployment's own curated tree
(``hermes_constants.get_bundled_skills_dir``, the same tree a local profile is seeded
from), and nothing else: no profile directory, no project tree, no external dir, so
another tenant's or agent's skill is indistinguishable from a missing one. When the
skill blob store lands, this one function becomes its lookup, keyed by the bound
scope, and nothing else in this module changes.

Callers this covers: the inline ``!`cmd``` snippet of a skill body
(:func:`run_inline_skill_shell`, reached from ``agent.skill_preprocessing``), a script
file a skill references (:func:`run_skill_script`), and — as they gain hosted paths —
code execution and cron jobs, which read their script through the same seam instead of
a host profile path. A hosted cron pre-run script stays refused until then: its source
is a host profile file, so there is nothing to copy and it never runs here.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
from pathlib import Path
from typing import Optional

from agent.compression_marker import elide
from tools.environments.sandbox_scope import hosted_scope

logger = logging.getLogger(__name__)

# The cap the inline shell already keeps, so one runaway script cannot blow out the
# context it is rendered into.
MAX_OUTPUT = 4000

# Interpreters a skill script may run under in the sandbox, by extension. Deny by
# default: a script whose extension has no entry here is refused, never guessed at.
SANDBOX_INTERPRETERS: dict[str, tuple[str, ...]] = {
    ".sh": ("bash",),
    ".bash": ("bash",),
    ".py": ("python3",),
    ".js": ("node",),
    ".rb": ("ruby",),
}

_SCRIPT_DIRS = "skill-scripts"


class SkillScriptRefused(PermissionError):
    """A skill script would have run somewhere other than the bound agent's own
    sandbox: on this host, with no hosted scope bound, or with no sandbox backend
    that serves the hosted scope."""


def _deployment_catalog() -> Path:
    """The deployment's own skill tree — operator-curated content, the same trust as
    the tool schemas a hosted process already reads."""
    from hermes_constants import get_bundled_skills_dir

    return Path(get_bundled_skills_dir(Path(__file__).parent.parent / "skills"))


def read_skill_script(skill_name: str, script_path: str) -> Optional[tuple[str, str]]:
    """The script-source seam — the only place a skill script's bytes are read.

    Returns ``(content, file_path)`` for *script_path* inside the skill *skill_name*,
    or ``None`` when the bound scope has no such skill or script. The catalog is the
    deployment's curated tree and nothing else: nothing a tenant or another agent owns
    is consulted, so a foreign skill answers exactly like a missing one, whatever task
    id or path a caller carries. When the skill blob store lands, this function becomes
    its scope-keyed lookup.
    """
    from tools.path_security import has_traversal_component, validate_within_dir
    from tools.skills_tool import skill_lookup_path_error

    if hosted_scope() is None:
        raise SkillScriptRefused(
            "a skill script runs only in the bound agent's sandbox; a local install runs it "
            "with the agent's own terminal, as it does today")
    if error := skill_lookup_path_error(skill_name):
        logger.debug("Skill script lookup refused for %r: %s", skill_name, error)
        return None
    skill_dir = _deployment_catalog() / skill_name.strip()
    if not (skill_dir / "SKILL.md").is_file():
        return None  # a missing skill and a foreign one answer the same
    target = skill_dir / script_path
    if has_traversal_component(script_path) or validate_within_dir(target, skill_dir):
        logger.debug("Skill script path %r escapes its skill directory", script_path)
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8-sig", errors="replace"), script_path
    except OSError as exc:
        logger.debug("Could not read skill script %s: %s", target, exc)
        return None


def _sandbox_env(task_id: Optional[str]):
    """The bound agent's sandbox, brought up on demand — or a refusal when there is
    none (a hosted scope with a host backend configured fails closed here, before
    anything could run on this host)."""
    from tools.terminal_tool_lifecycle import ensure_task_env

    env = ensure_task_env(task_id)
    if env is None:
        raise SkillScriptRefused(
            "this agent has no sandbox to run a skill script in: its terminal backend cannot "
            "serve a hosted scope, and a skill script never runs on this host")
    return env


def _script_dir(env) -> str:
    """A private scratch directory inside the sandbox for the shipped script."""
    import posixpath

    try:
        temp = env.get_temp_dir()
    except (AttributeError, OSError):
        # a backend without the optional temp-dir operation falls back to its workspace
        temp = None
    base = temp or getattr(env, "cwd", "") or ""
    return posixpath.join(base, _SCRIPT_DIRS) if base else _SCRIPT_DIRS


def _run_in_sandbox(env, command: str, *, cwd: str, timeout: int) -> str:
    """Run *command* in the sandbox and shape its output like the inline shell does:
    stdout (stderr when stdout is empty), a marker on a no-output failure, capped.

    The grant is passed by what is absent: no ``child_env`` (so no environment value
    from this process, its profile or any store), no credential files, no broker
    references — the script sees only what its sandbox provides by default.
    """
    result = env.execute(command, cwd=cwd, timeout=max(1, int(timeout)))
    output = (result.get("output") or "").rstrip("\n")
    code = result.get("returncode", 0)
    if code != 0 and not output:
        return f"[skill script exit {code} with no output]"
    return elide(output, MAX_OUTPUT)


def run_skill_script(skill_name: str, script_path: str, *, argv: tuple[str, ...] = (),
                     task_id: Optional[str] = None, timeout: int = 120) -> str:
    """Run one of a skill's scripts in the bound agent's sandbox.

    The script's bytes are read through the seam, copied into the sandbox at exec time
    under a name derived from their content, and executed there with the grant: in this
    agent's sandbox, with no secrets. A missing or foreign skill answers like a missing
    one; a host run is refused.
    """
    scope = hosted_scope()
    if scope is None:
        raise SkillScriptRefused(
            "a skill script runs only in the bound agent's sandbox; a local install runs it "
            "with the agent's own terminal, as it does today")
    script = read_skill_script(skill_name, script_path)
    if script is None:
        return f"Skill '{skill_name}' has no script '{script_path}'."
    content, _path = script
    suffix = Path(script_path).suffix.lower()
    if suffix not in SANDBOX_INTERPRETERS:
        return (f"Skill script '{script_path}' has no sandbox interpreter for '{suffix or 'no extension'}'; "
                f"supported: {', '.join(sorted(SANDBOX_INTERPRETERS))}.")
    env = _sandbox_env(task_id)
    from tools.code_execution_rpc import _execute_checked, _private_dirs_cmd, _remote_write

    directory = _script_dir(env)
    _execute_checked(env, _private_dirs_cmd(directory), "skill script directory", timeout=30)  # owner-only
    name = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16] + suffix
    _remote_write(env, f"{directory}/{name}", content, check=True)  # owner-only, on stdin
    command = " ".join([*SANDBOX_INTERPRETERS[suffix], name, *(shlex.quote(a) for a in argv)])
    return _run_in_sandbox(env, command, cwd=directory, timeout=timeout)


def run_inline_skill_shell(command: str, *, timeout: int, task_id: Optional[str] = None) -> str:
    """An inline ``!`cmd``` snippet of a skill body: the same tenant-supplied code as a
    script file, so it takes the same route — the bound agent's sandbox, with the grant."""
    if hosted_scope() is None:
        raise SkillScriptRefused(
            "a skill script runs only in the bound agent's sandbox; a local install runs it "
            "with the agent's own terminal, as it does today")
    env = _sandbox_env(task_id)
    return _run_in_sandbox(env, command, cwd="", timeout=timeout)


__all__ = [
    "MAX_OUTPUT", "SANDBOX_INTERPRETERS", "SkillScriptRefused", "read_skill_script",
    "run_inline_skill_shell", "run_skill_script",
]

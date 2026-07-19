from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def extract_modified_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ "):
            files.append(_patch_path(line[4:]))
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                files.append(parts[3][2:])
    unique: list[str] = []
    for file in files:
        if file and file != "/dev/null" and file not in unique:
            unique.append(file)
    return unique


def _patch_path(value: str) -> str:
    path = value.strip().split("\t", 1)[0].strip()
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def looks_like_unified_diff(patch_text: str) -> bool:
    text = patch_text or ""
    lines = text.splitlines()
    has_old_file = any(line.startswith("--- ") for line in lines)
    has_new_file = any(line.startswith("+++ ") for line in lines)
    has_hunk = any(line.startswith("@@") for line in lines)
    return has_old_file and has_new_file and has_hunk


def check_patch_applies(repo_path: str | Path, patch_text: str) -> CommandResult:
    if not looks_like_unified_diff(patch_text):
        return CommandResult(2, "", "Patch is not a unified diff.")
    return _run_with_input(["git", "apply", "--check"], patch_text, cwd=repo_path, timeout_seconds=60)


def run_patch_in_temp_copy(
    repo_path: str | Path,
    patch_text: str,
    *,
    test_command: list[str] | None = None,
    run_tests: bool = False,
    reproduction_commands: list[list[str] | str] | None = None,
    test_patch: str = "",
    timeout_seconds: float = 300,
) -> dict:
    source = Path(repo_path).resolve()
    if not source.exists() or not source.is_dir():
        raise ValueError(f"Repository path does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="bug_pipeline_") as temp_dir:
        worktree = Path(temp_dir) / "repo"
        shutil.copytree(source, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        test_patch_apply = "not_run"
        if test_patch:
            test_patch_check = check_patch_applies(worktree, test_patch)
            if test_patch_check.returncode != 0:
                return {
                    "patch_apply_check": "not_run",
                    "patch_apply": "not_run",
                    "test_patch_apply": "failed",
                    "reproduction_result": "setup_failed",
                    "regression_result": "failed",
                    "stdout": test_patch_check.stdout,
                    "stderr": test_patch_check.stderr,
                }
            test_patch_result = _run_with_input(
                ["git", "apply"], test_patch, cwd=worktree, timeout_seconds=60
            )
            if test_patch_result.returncode != 0:
                return {
                    "patch_apply_check": "not_run",
                    "patch_apply": "not_run",
                    "test_patch_apply": "failed",
                    "reproduction_result": "setup_failed",
                    "regression_result": "failed",
                    "stdout": test_patch_result.stdout,
                    "stderr": test_patch_result.stderr,
                }
            test_patch_apply = "passed"

        commands = list(reproduction_commands or [])
        before_results = _run_commands(commands, cwd=worktree, timeout_seconds=timeout_seconds) if commands else []
        apply_check = check_patch_applies(worktree, patch_text)
        if apply_check.returncode != 0:
            return {
                "patch_apply_check": "failed",
                "test_patch_apply": test_patch_apply,
                "reproduction_result": "not_run" if not commands else "patch_apply_failed",
                "regression_result": "failed",
                "stdout": apply_check.stdout,
                "stderr": apply_check.stderr,
            }

        apply_result = _run_with_input(["git", "apply"], patch_text, cwd=worktree, timeout_seconds=60)
        if apply_result.returncode != 0:
            return {
                "patch_apply_check": "passed",
                "patch_apply": "failed",
                "test_patch_apply": test_patch_apply,
                "reproduction_result": "not_run" if not commands else "patch_apply_failed",
                "regression_result": "failed",
                "stdout": apply_result.stdout,
                "stderr": apply_result.stderr,
            }

        after_results = _run_commands(commands, cwd=worktree, timeout_seconds=timeout_seconds) if commands else []
        reproduction_result = _fib_result(before_results, after_results)

        if not run_tests:
            return {
                "patch_apply_check": "passed",
                "patch_apply": "passed",
                "test_patch_apply": test_patch_apply,
                "reproduction_result": reproduction_result,
                "reproduction_tests": _command_report(commands, before_results, after_results),
                "regression_result": "not_run",
                "stdout": "",
                "stderr": "",
            }

        command = test_command or ["python3", "-m", "pytest"]
        test_result = run_command(command, cwd=worktree, timeout_seconds=timeout_seconds)
        return {
            "patch_apply_check": "passed",
            "patch_apply": "passed",
            "test_patch_apply": test_patch_apply,
            "reproduction_result": reproduction_result,
            "reproduction_tests": _command_report(commands, before_results, after_results),
            "regression_result": "passed" if test_result.returncode == 0 else "failed",
            "stdout": test_result.stdout,
            "stderr": test_result.stderr,
            "test_command": command,
        }


def run_command(command: list[str] | str, cwd: str | Path, *, timeout_seconds: float = 300) -> CommandResult:
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, _timeout_text(exc.stdout), _timeout_text(exc.stderr, suffix="command timed out"))


def _run_with_input(
    command: list[str],
    text: str,
    cwd: str | Path,
    *,
    timeout_seconds: float,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            input=text,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, _timeout_text(exc.stdout), _timeout_text(exc.stderr, suffix="command timed out"))


def _run_commands(
    commands: list[list[str] | str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> list[CommandResult]:
    return [run_command(command, cwd=cwd, timeout_seconds=timeout_seconds) for command in commands]


def _timeout_text(value: str | bytes | None, *, suffix: str = "") -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value or "")
    return f"{text}\n{suffix}".strip()


def _fib_result(before: list[CommandResult], after: list[CommandResult]) -> str:
    if not before:
        return "not_run"
    if any(result.returncode == 0 for result in before):
        return "baseline_passed"
    if any(result.returncode != 0 for result in after):
        return "post_patch_failed"
    return "passed"


def _command_report(
    commands: list[list[str] | str],
    before: list[CommandResult],
    after: list[CommandResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        baseline = before[index]
        patched = after[index]
        rows.append(
            {
                "command": command,
                "before_returncode": baseline.returncode,
                "after_returncode": patched.returncode,
                "before_stderr": baseline.stderr[-2000:],
                "after_stderr": patched.stderr[-2000:],
            }
        )
    return rows

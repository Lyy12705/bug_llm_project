from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


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
    return _run_with_input(["git", "apply", "--check"], patch_text, cwd=repo_path)


def run_patch_in_temp_copy(
    repo_path: str | Path,
    patch_text: str,
    *,
    test_command: list[str] | None = None,
    run_tests: bool = False,
) -> dict:
    source = Path(repo_path).resolve()
    if not source.exists():
        raise ValueError(f"Repository path does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="bug_pipeline_") as temp_dir:
        worktree = Path(temp_dir) / "repo"
        shutil.copytree(source, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        apply_check = check_patch_applies(worktree, patch_text)
        if apply_check.returncode != 0:
            return {
                "patch_apply_check": "failed",
                "regression_result": "failed",
                "stdout": apply_check.stdout,
                "stderr": apply_check.stderr,
            }

        apply_result = _run_with_input(["git", "apply"], patch_text, cwd=worktree)
        if apply_result.returncode != 0:
            return {
                "patch_apply_check": "passed",
                "patch_apply": "failed",
                "regression_result": "failed",
                "stdout": apply_result.stdout,
                "stderr": apply_result.stderr,
            }

        if not run_tests:
            return {
                "patch_apply_check": "passed",
                "patch_apply": "passed",
                "regression_result": "not_run",
                "stdout": "",
                "stderr": "",
            }

        command = test_command or ["python3", "-m", "pytest"]
        test_result = run_command(command, cwd=worktree)
        return {
            "patch_apply_check": "passed",
            "patch_apply": "passed",
            "regression_result": "passed" if test_result.returncode == 0 else "failed",
            "stdout": test_result.stdout,
            "stderr": test_result.stderr,
            "test_command": command,
        }


def run_command(command: list[str] | str, cwd: str | Path) -> CommandResult:
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    completed = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _run_with_input(command: list[str], text: str, cwd: str | Path) -> CommandResult:
    completed = subprocess.run(command, input=text, cwd=str(cwd), capture_output=True, text=True, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)

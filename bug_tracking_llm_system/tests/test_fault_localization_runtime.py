from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.fault_localization import _repo_cache_token


class FaultLocalizationRuntimeTests(unittest.TestCase):
    def test_git_repo_cache_token_changes_for_uncommitted_source_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            git_refs = repo / ".git" / "refs" / "heads"
            git_refs.mkdir(parents=True)
            (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (git_refs / "main").write_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", encoding="utf-8")
            source_file = repo / "src" / "auth" / "validator.py"

            initial_token = _repo_cache_token(repo)
            source_file.write_text(
                "def validate_token(token):\n"
                "    if token is None:\n"
                "        return ''\n"
                "    return token.strip()\n",
                encoding="utf-8",
            )
            updated_mtime = source_file.stat().st_mtime + 5
            os.utime(source_file, (updated_mtime, updated_mtime))

            updated_token = _repo_cache_token(repo)

        self.assertNotEqual(initial_token, updated_token)

    def test_cli_progress_and_index_cache_are_pollable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            ticket = _write_ticket(tmp_path)
            cache_dir = tmp_path / "index_cache"
            first_output = tmp_path / "first.json"
            second_output = tmp_path / "second.json"
            first_progress = tmp_path / "first_progress.jsonl"
            second_progress = tmp_path / "second_progress.jsonl"

            _run(
                [
                    sys.executable,
                    "scripts/fault_localization.py",
                    "--ticket",
                    str(ticket),
                    "--repo-path",
                    str(repo),
                    "--top-k",
                    "2",
                    "--embedding-backend",
                    "tfidf",
                    "--output-format",
                    "user-facing",
                    "--index-cache-dir",
                    str(cache_dir),
                    "--progress-file",
                    str(first_progress),
                    "--output",
                    str(first_output),
                ]
            )
            _run(
                [
                    sys.executable,
                    "scripts/fault_localization.py",
                    "--ticket",
                    str(ticket),
                    "--repo-path",
                    str(repo),
                    "--top-k",
                    "2",
                    "--embedding-backend",
                    "tfidf",
                    "--output-format",
                    "user-facing",
                    "--index-cache-dir",
                    str(cache_dir),
                    "--progress-file",
                    str(second_progress),
                    "--output",
                    str(second_output),
                ]
            )

            first = json.loads(first_output.read_text(encoding="utf-8"))
            second = json.loads(second_output.read_text(encoding="utf-8"))
            first_events = _read_jsonl(first_progress)
            second_events = _read_jsonl(second_progress)

        self.assertEqual(first["top_k_suspicious_files"][0]["file_path"], "src/auth/validator.py")
        self.assertFalse(first["runtime"]["index_cache_hit"])
        self.assertTrue(second["runtime"]["index_cache_hit"])
        self.assertIn("index_ready", {row["event"] for row in first_events})
        self.assertIn("index_cache_hit", {row["event"] for row in second_events})
        self.assertIn("localization_completed", {row["event"] for row in second_events})

    def test_foreground_job_writes_status_result_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            ticket = _write_ticket(tmp_path)
            jobs_dir = tmp_path / "jobs"
            cache_dir = tmp_path / "index_cache"

            completed = _run(
                [
                    sys.executable,
                    "scripts/fault_localization_job.py",
                    "start",
                    "--foreground",
                    "--ticket",
                    str(ticket),
                    "--repo-path",
                    str(repo),
                    "--top-k",
                    "2",
                    "--embedding-backend",
                    "tfidf",
                    "--jobs-dir",
                    str(jobs_dir),
                    "--index-cache-dir",
                    str(cache_dir),
                ]
            )
            status = json.loads(completed.stdout)
            result_path = Path(status["result_path"])
            progress_path = Path(status["progress_path"])

            status_check = _run(
                [
                    sys.executable,
                    "scripts/fault_localization_job.py",
                    "status",
                    "--job-dir",
                    status["job_dir"],
                ]
            )
            status_payload = json.loads(status_check.stdout)

            self.assertEqual(status["status"], "succeeded")
            self.assertTrue(result_path.exists())
            self.assertTrue(progress_path.exists())
            self.assertEqual(status_payload["status"], "succeeded")
            self.assertTrue(status_payload["recent_progress"])

    def test_job_resolves_relative_inputs_from_callers_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_repo(tmp_path)
            _write_ticket(tmp_path)
            jobs_dir = tmp_path / "jobs"
            cache_dir = tmp_path / "index_cache"

            completed = _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "fault_localization_job.py"),
                    "start",
                    "--foreground",
                    "--ticket",
                    "ticket.json",
                    "--repo-path",
                    "repo",
                    "--top-k",
                    "2",
                    "--embedding-backend",
                    "tfidf",
                    "--jobs-dir",
                    str(jobs_dir),
                    "--index-cache-dir",
                    str(cache_dir),
                ],
                cwd=tmp_path,
            )
            status = json.loads(completed.stdout)

        self.assertEqual(status["status"], "succeeded")


def _run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    source = repo / "src" / "auth"
    source.mkdir(parents=True)
    (source / "validator.py").write_text(
        "def validate_token(token):\n"
        "    return token.strip()\n",
        encoding="utf-8",
    )
    (source / "session.py").write_text(
        "from src.auth.validator import validate_token\n\n"
        "def start_session(user):\n"
        "    return validate_token(user.token)\n",
        encoding="utf-8",
    )
    return repo


def _write_ticket(tmp_path: Path) -> Path:
    ticket = tmp_path / "ticket.json"
    ticket.write_text(
        json.dumps(
            {
                "ticket_id": "RUNTIME-1",
                "title": "Login crashes when token is missing",
                "description": "Login crashes because validate_token calls strip on a missing token.",
                "logs": "TypeError at src/auth/validator.py:2",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return ticket


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    unittest.main()

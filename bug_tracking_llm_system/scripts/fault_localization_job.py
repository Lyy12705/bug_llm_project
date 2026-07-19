#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start and inspect fault-localization beta jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Create a pollable localization job.")
    input_group = start.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ticket", help="Single ticket JSON path.")
    input_group.add_argument("--tickets-jsonl", help="Ticket JSONL path.")
    repo_group = start.add_mutually_exclusive_group(required=True)
    repo_group.add_argument("--repo-path", help="Repository or source folder.")
    repo_group.add_argument("--code-index", help="Prebuilt code index JSON.")
    start.add_argument("--jobs-dir", default="reports/fault_localization/jobs")
    start.add_argument("--job-id", default=None)
    start.add_argument("--output-format", choices=("raw", "user-facing"), default="user-facing")
    start.add_argument("--top-k", type=int, default=5)
    start.add_argument("--embedding-backend", default="tfidf")
    start.add_argument("--index-cache-dir", default="reports/fault_localization/index_cache")
    start.add_argument("--force-reindex", action="store_true")
    start.add_argument("--include-code-preview", action="store_true")
    start.add_argument("--foreground", action="store_true", help="Run the job immediately and wait; useful for smoke tests.")

    run_job = subparsers.add_parser("run-job", help=argparse.SUPPRESS)
    run_job.add_argument("--job-dir", required=True)

    status = subparsers.add_parser("status", help="Read the current job status.")
    status.add_argument("--job-dir", required=True)
    status.add_argument("--tail", type=int, default=5, help="Number of recent progress events to include.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "start":
        return start_job(args)
    if args.command == "run-job":
        return run_job(Path(args.job_dir))
    if args.command == "status":
        print(json.dumps(read_job_status(Path(args.job_dir), tail=args.tail), ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


def start_job(args: argparse.Namespace) -> int:
    jobs_dir = _resolve(args.jobs_dir)
    job_id = args.job_id or uuid.uuid4().hex[:12]
    if not JOB_ID_RE.fullmatch(job_id):
        raise SystemExit("job id must be 1-64 characters using only letters, numbers, '_' or '-'")
    jobs_dir = jobs_dir.resolve()
    job_dir = (jobs_dir / job_id).resolve()
    try:
        job_dir.relative_to(jobs_dir)
    except ValueError as exc:  # defensive; regex above already excludes path separators
        raise SystemExit("job directory escapes jobs root") from exc
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"
    result_path = job_dir / ("result.jsonl" if args.tickets_jsonl else "result.json")
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    command = build_localization_command(
        ticket=_resolve_input(args.ticket),
        tickets_jsonl=_resolve_input(args.tickets_jsonl),
        repo_path=_resolve_input(args.repo_path),
        code_index=_resolve_input(args.code_index),
        output_format=args.output_format,
        top_k=args.top_k,
        embedding_backend=args.embedding_backend,
        index_cache_dir=args.index_cache_dir,
        force_reindex=args.force_reindex,
        include_code_preview=args.include_code_preview,
        result_path=result_path,
        progress_path=progress_path,
    )
    metadata = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "status_path": str(job_dir / "status.json"),
        "result_path": str(result_path),
        "progress_path": str(progress_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
        "created_at_utc": _utc_now(),
    }
    (job_dir / "command.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_status(job_dir, {**metadata, "status": "queued"})

    if args.foreground:
        return run_job(job_dir)

    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run-job", "--job-dir", str(job_dir)],
        cwd=str(ROOT),
        env=_job_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps({"job_id": job_id, "job_dir": str(job_dir), "status_path": str(job_dir / "status.json")}, ensure_ascii=False, indent=2))
    return 0


def build_localization_command(
    *,
    ticket: str | None,
    tickets_jsonl: str | None,
    repo_path: str | None,
    code_index: str | None,
    output_format: str,
    top_k: int,
    embedding_backend: str,
    index_cache_dir: str | None,
    force_reindex: bool,
    include_code_preview: bool,
    result_path: Path,
    progress_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "fault_localization.py"),
        "--top-k",
        str(top_k),
        "--embedding-backend",
        embedding_backend,
        "--output-format",
        output_format,
        "--output",
        str(result_path),
        "--progress",
        "json",
        "--progress-file",
        str(progress_path),
    ]
    if ticket:
        command.extend(["--ticket", ticket])
    if tickets_jsonl:
        command.extend(["--tickets-jsonl", tickets_jsonl])
    if repo_path:
        command.extend(["--repo-path", repo_path])
    if code_index:
        command.extend(["--code-index", code_index])
    if index_cache_dir:
        command.extend(["--index-cache-dir", index_cache_dir])
    if force_reindex:
        command.append("--force-reindex")
    if include_code_preview:
        command.append("--include-code-preview")
    return command


def run_job(job_dir: Path) -> int:
    metadata_path = job_dir / "command.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    started_at_utc = _utc_now()
    write_status(job_dir, {**metadata, "status": "running", "started_at_utc": started_at_utc})
    stdout_path = Path(metadata["stdout_path"])
    stderr_path = Path(metadata["stderr_path"])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(metadata["command"]),
            cwd=str(ROOT),
            env=_job_env(),
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    status = "succeeded" if completed.returncode == 0 else "failed"
    final_status = {
        **metadata,
        "status": status,
        "returncode": completed.returncode,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "last_progress_event": _last_progress_event(Path(metadata["progress_path"])),
    }
    write_status(job_dir, final_status)
    print(json.dumps(final_status, ensure_ascii=False, indent=2))
    return completed.returncode


def read_job_status(job_dir: Path, *, tail: int) -> dict[str, Any]:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise SystemExit(f"Job status does not exist: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    progress_path = Path(status.get("progress_path") or job_dir / "progress.jsonl")
    status["recent_progress"] = _progress_tail(progress_path, limit=tail)
    return status


def write_status(job_dir: Path, status: dict[str, Any]) -> None:
    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _progress_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _last_progress_event(path: Path) -> dict[str, Any]:
    rows = _progress_tail(path, limit=1)
    return rows[0] if rows else {}


def _job_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    return env


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_input(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

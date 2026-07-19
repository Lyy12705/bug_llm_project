#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SYSTEM_ROOT.parent
SMOKE_REPO = SYSTEM_ROOT / "tests" / "fixtures" / "smoke_repo"

DEFAULT_OUTPUT = SYSTEM_ROOT / "reports" / "health_check" / "health_check_latest.json"
DEFAULT_TMP_DIR = Path("/tmp") / "bug_llm_health_check"


@dataclass(frozen=True)
class CheckSpec:
    name: str
    description: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: int
    required_paths: tuple[Path, ...] = ()
    validator: Callable[[subprocess.CompletedProcess[str]], str] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight smoke checks for the integrated bug tracking system "
            "without network access, model downloads, or full dataset experiments."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSON health report.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=str(DEFAULT_TMP_DIR),
        help="Temporary directory for smoke-test outputs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Default timeout in seconds for each smoke check.",
    )
    parser.add_argument(
        "--skip-unit-tests",
        action="store_true",
        help="Skip the integrated unittest suite and run only CLI smoke checks.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary smoke-test output files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    checks = build_checks(
        tmp_dir=tmp_dir,
        timeout_seconds=args.timeout,
        include_unit_tests=not args.skip_unit_tests,
    )
    results = [run_check(check) for check in checks]
    report = build_report(results)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SYSTEM_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print_report(report, output_path)
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 1 if report["summary"]["failed"] else 0


def build_checks(*, tmp_dir: Path, timeout_seconds: int, include_unit_tests: bool) -> list[CheckSpec]:
    main_pipeline_output = tmp_dir / "main_pipeline_result.json"
    fault_output = tmp_dir / "fault_localization_result.json"
    fault_progress = tmp_dir / "fault_localization_progress.jsonl"
    fault_index_cache = tmp_dir / "fault_index_cache"
    fault_job_dir = tmp_dir / "fault_jobs"

    checks: list[CheckSpec] = [
        CheckSpec(
            name="main_pipeline_smoke",
            description="Run one integrated ticket through extraction, duplicate, priority, assignee, localization, and patch fallback.",
            command=[
                sys.executable,
                "src/main.py",
                "--raw-ticket",
                "data/raw_tickets/raw_ticket.example.json",
                "--repo-path",
                str(SMOKE_REPO),
                "--fault-top-k",
                "3",
                "--fault-embedding-backend",
                "tfidf",
                "--output",
                str(main_pipeline_output),
                "--no-checkpoints",
            ],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "src" / "main.py",
                SYSTEM_ROOT / "data" / "raw_tickets" / "raw_ticket.example.json",
                SMOKE_REPO / "src" / "auth" / "validator.py",
            ),
            validator=validate_main_pipeline(main_pipeline_output),
        ),
        CheckSpec(
            name="fault_localization_smoke",
            description="Run retrieval-first fault localization on one ticket with local TF-IDF backend.",
            command=[
                sys.executable,
                "scripts/fault_localization.py",
                "--ticket",
                "data/raw_tickets/raw_ticket.example.json",
                "--repo-path",
                str(SMOKE_REPO),
                "--top-k",
                "3",
                "--embedding-backend",
                "tfidf",
                "--output-format",
                "user-facing",
                "--index-cache-dir",
                str(fault_index_cache),
                "--progress",
                "json",
                "--progress-file",
                str(fault_progress),
                "--output",
                str(fault_output),
            ],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "scripts" / "fault_localization.py",
                SYSTEM_ROOT / "data" / "raw_tickets" / "raw_ticket.example.json",
                SMOKE_REPO / "src" / "auth" / "validator.py",
            ),
            validator=validate_fault_localization(fault_output, progress_path=fault_progress),
        ),
        CheckSpec(
            name="fault_localization_job_smoke",
            description="Run one foreground pollable fault-localization job with index cache and progress file.",
            command=[
                sys.executable,
                "scripts/fault_localization_job.py",
                "start",
                "--foreground",
                "--ticket",
                "data/raw_tickets/raw_ticket.example.json",
                "--repo-path",
                str(SMOKE_REPO),
                "--top-k",
                "3",
                "--embedding-backend",
                "tfidf",
                "--jobs-dir",
                str(fault_job_dir),
                "--index-cache-dir",
                str(fault_index_cache),
            ],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "scripts" / "fault_localization_job.py",
                SYSTEM_ROOT / "scripts" / "fault_localization.py",
                SYSTEM_ROOT / "data" / "raw_tickets" / "raw_ticket.example.json",
                SMOKE_REPO / "src" / "auth" / "validator.py",
            ),
            validator=validate_fault_localization_job(),
        ),
        CheckSpec(
            name="duplicate_detection_smoke",
            description="Evaluate the duplicate detector used by the integrated pipeline on a frozen labeled fixture.",
            command=[sys.executable, "scripts/evaluate_integrated_baselines.py", "--section", "duplicate"],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "scripts" / "evaluate_integrated_baselines.py",
                SYSTEM_ROOT / "tests" / "fixtures" / "integrated_baseline_eval.json",
            ),
            validator=validate_contains('"precision"', "integrated duplicate precision/recall/F1 were reported"),
        ),
        CheckSpec(
            name="priority_prediction_smoke",
            description="Evaluate the priority classifier used by the integrated pipeline on a frozen labeled fixture.",
            command=[sys.executable, "scripts/evaluate_integrated_baselines.py", "--section", "priority"],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "scripts" / "evaluate_integrated_baselines.py",
                SYSTEM_ROOT / "tests" / "fixtures" / "integrated_baseline_eval.json",
            ),
            validator=validate_contains('"macro_f1"', "integrated priority accuracy/macro-F1/calibration were reported"),
        ),
    ]

    if include_unit_tests:
        checks.insert(
            0,
            CheckSpec(
                name="integrated_unit_tests",
                description="Run the main integrated system unittest suite.",
                command=[
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                ],
                cwd=SYSTEM_ROOT,
                env=pythonpath_env(SYSTEM_ROOT / "src"),
                timeout_seconds=max(timeout_seconds, 120),
                required_paths=(SYSTEM_ROOT / "tests",),
                validator=validate_contains("OK", "unittest suite reported OK"),
            ),
        )

    return checks


def run_check(check: CheckSpec) -> dict[str, object]:
    missing_paths = [str(path) for path in check.required_paths if not path.exists()]
    started = time.perf_counter()
    if missing_paths:
        return {
            "name": check.name,
            "description": check.description,
            "status": "skip",
            "duration_seconds": 0.0,
            "command": command_text(check.command),
            "cwd": str(check.cwd),
            "detail": f"Missing required path(s): {', '.join(missing_paths)}",
            "stdout_tail": "",
            "stderr_tail": "",
        }

    try:
        completed = subprocess.run(
            check.command,
            cwd=str(check.cwd),
            env=check.env,
            text=True,
            capture_output=True,
            timeout=check.timeout_seconds,
            check=False,
        )
        duration = round(time.perf_counter() - started, 3)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": check.name,
            "description": check.description,
            "status": "fail",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "command": command_text(check.command),
            "cwd": str(check.cwd),
            "detail": f"Timed out after {check.timeout_seconds} seconds",
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": tail(exc.stderr or ""),
        }

    status = "pass"
    detail = "command completed"
    if completed.returncode != 0:
        status = "fail"
        detail = f"command exited with code {completed.returncode}"
    elif check.validator:
        try:
            detail = check.validator(completed)
        except Exception as exc:  # noqa: BLE001 - health check should report validation details.
            status = "fail"
            detail = f"output validation failed: {exc}"

    return {
        "name": check.name,
        "description": check.description,
        "status": status,
        "duration_seconds": duration,
        "command": command_text(check.command),
        "cwd": str(check.cwd),
        "detail": detail,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def build_report(results: list[dict[str, object]]) -> dict[str, object]:
    passed = sum(1 for result in results if result["status"] == "pass")
    failed = sum(1 for result in results if result["status"] == "fail")
    skipped = sum(1 for result in results if result["status"] == "skip")
    return {
        "status": "fail" if failed else "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORKSPACE_ROOT),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "checks": results,
        "scope_note": (
            "This is a lightweight local health check. It intentionally does not call Ollama, "
            "download embedding models, rerun full SWE-bench Lite, retrain classifiers, or mutate datasets."
        ),
    }


def print_report(report: dict[str, object], output_path: Path) -> None:
    summary = report["summary"]
    print(
        f"Health check: {str(report['status']).upper()} "
        f"({summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped)"
    )
    for check in report["checks"]:
        print(f"[{str(check['status']).upper()}] {check['name']} - {check['detail']}")
    print(f"Report: {output_path}")


def pythonpath_env(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path}{os.pathsep}{existing}" if existing else str(path)
    return env


def validate_main_pipeline(output_path: Path) -> Callable[[subprocess.CompletedProcess[str]], str]:
    def _validate(_: subprocess.CompletedProcess[str]) -> str:
        data = read_json(output_path)
        status = str(data.get("status") or "")
        if not status:
            raise ValueError("missing pipeline status")
        location = data.get("bug_location") if isinstance(data.get("bug_location"), dict) else {}
        candidates = location.get("localized_candidates") if isinstance(location, dict) else []
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("missing fault-localization candidates in pipeline result")
        user_facing = data.get("bug_location_user_facing") if isinstance(data.get("bug_location_user_facing"), dict) else {}
        user_candidates = user_facing.get("top_k_suspicious_files") if isinstance(user_facing, dict) else []
        if not isinstance(user_candidates, list) or not user_candidates:
            raise ValueError("missing user-facing fault-localization candidates in pipeline result")
        patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
        patch_status = patch.get("patch_status", "unknown")
        return (
            f"pipeline status={status}; patch_status={patch_status}; "
            f"localization_candidates={len(candidates)}; user_facing_candidates={len(user_candidates)}"
        )

    return _validate


def validate_fault_localization(output_path: Path, *, progress_path: Path | None = None) -> Callable[[subprocess.CompletedProcess[str]], str]:
    def _validate(_: subprocess.CompletedProcess[str]) -> str:
        data = read_json(output_path)
        if "top_k_suspicious_files" in data:
            candidates = data.get("top_k_suspicious_files")
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            if not isinstance(summary, dict) or not summary.get("confidence_level"):
                raise ValueError("missing user-facing summary confidence")
        else:
            candidates = data.get("localized_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("no localization candidates were produced")
        top = candidates[0] if isinstance(candidates[0], dict) else {}
        file_path = top.get("file_path") or top.get("file") or "unknown"
        score = top.get("score", top.get("final_score", "unknown"))
        progress_detail = ""
        if progress_path is not None:
            progress_events = read_jsonl(progress_path)
            event_names = {str(row.get("event")) for row in progress_events}
            if "index_ready" not in event_names or "localization_completed" not in event_names:
                raise ValueError("progress file did not include index_ready and localization_completed")
            progress_detail = f"; progress_events={len(progress_events)}"
        return f"localized_candidates={len(candidates)}; top_file={file_path}; top_score={score}{progress_detail}"

    return _validate


def validate_fault_localization_job() -> Callable[[subprocess.CompletedProcess[str]], str]:
    def _validate(completed: subprocess.CompletedProcess[str]) -> str:
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"job output was not JSON: {exc}") from exc
        if data.get("status") != "succeeded":
            raise ValueError(f"job did not succeed: {data.get('status')}")
        status_path = Path(str(data.get("status_path") or ""))
        result_path = Path(str(data.get("result_path") or ""))
        progress_path = Path(str(data.get("progress_path") or ""))
        if not status_path.exists() or not result_path.exists() or not progress_path.exists():
            raise ValueError("job did not create status, result, and progress files")
        progress_events = read_jsonl(progress_path)
        event_names = {str(row.get("event")) for row in progress_events}
        if "index_ready" not in event_names or "localization_completed" not in event_names:
            raise ValueError("job progress file is missing required events")
        return f"job status=succeeded; progress_events={len(progress_events)}; result={result_path}"

    return _validate


def validate_contains(needle: str, detail: str) -> Callable[[subprocess.CompletedProcess[str]], str]:
    def _validate(completed: subprocess.CompletedProcess[str]) -> str:
        combined = f"{completed.stdout}\n{completed.stderr}"
        if needle not in combined:
            raise ValueError(f"expected output to contain {needle!r}")
        return detail

    return _validate


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"expected output file was not created: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"expected JSONL file was not created: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def command_text(command: list[str]) -> str:
    return " ".join(quote_command_part(part) for part in command)


def quote_command_part(part: str) -> str:
    if not part:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-=+")
    if all(char in safe for char in part):
        return part
    return "'" + part.replace("'", "'\"'\"'") + "'"


def tail(value: str, limit: int = 4000) -> str:
    return value[-limit:] if len(value) > limit else value


if __name__ == "__main__":
    raise SystemExit(main())

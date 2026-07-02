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
DUPLICATE_ROOT = WORKSPACE_ROOT / "bug-duplicate-detection"
PRIORITY_ROOT = WORKSPACE_ROOT / "bug-priority-drone"
TO_JSON_ROOT = WORKSPACE_ROOT / "ToJson"

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
                "../bug-duplicate-detection",
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
                DUPLICATE_ROOT,
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
                "../bug-duplicate-detection",
                "--top-k",
                "3",
                "--embedding-backend",
                "tfidf",
                "--output",
                str(fault_output),
            ],
            cwd=SYSTEM_ROOT,
            env=pythonpath_env(SYSTEM_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                SYSTEM_ROOT / "scripts" / "fault_localization.py",
                SYSTEM_ROOT / "data" / "raw_tickets" / "raw_ticket.example.json",
                DUPLICATE_ROOT / "src",
            ),
            validator=validate_fault_localization(fault_output),
        ),
        CheckSpec(
            name="duplicate_detection_smoke",
            description="Evaluate the duplicate detection TF-IDF baseline on the bundled sample tickets.",
            command=[
                sys.executable,
                "-m",
                "duplicate_ticket_detection.cli",
                "evaluate",
                "--tickets",
                "examples/sample_tickets.csv",
                "--method",
                "tfidf",
                "--combine",
                "max",
                "--top-k",
                "5",
            ],
            cwd=DUPLICATE_ROOT,
            env=pythonpath_env(DUPLICATE_ROOT / "src"),
            timeout_seconds=timeout_seconds,
            required_paths=(
                DUPLICATE_ROOT / "src" / "duplicate_ticket_detection" / "cli.py",
                DUPLICATE_ROOT / "examples" / "sample_tickets.csv",
            ),
            validator=validate_contains("MAP=", "duplicate MAP was reported"),
        ),
        CheckSpec(
            name="priority_prediction_smoke",
            description="Read the existing priority-prediction evaluation CSV and print headline metrics.",
            command=[
                sys.executable,
                "scripts/show_model_metrics.py",
            ],
            cwd=PRIORITY_ROOT,
            env=os.environ.copy(),
            timeout_seconds=timeout_seconds,
            required_paths=(
                PRIORITY_ROOT / "scripts" / "show_model_metrics.py",
                PRIORITY_ROOT / "reports" / "recall_balanced_best_eval.csv",
            ),
            validator=validate_contains("Accuracy", "priority metrics were reported"),
        ),
        CheckSpec(
            name="ticket_json_evaluation_smoke",
            description="Evaluate existing bug-report-to-JSON predictions against the local labeled sample.",
            command=[
                sys.executable,
                "scripts/To_Json/evaluate_results.py",
                "--gold",
                "dataset/labeled/test.jsonl",
                "--pred",
                "dataset/predicted/predicted.jsonl",
                "--no-source-consistency",
            ],
            cwd=TO_JSON_ROOT,
            env=os.environ.copy(),
            timeout_seconds=timeout_seconds,
            required_paths=(
                TO_JSON_ROOT / "scripts" / "To_Json" / "evaluate_results.py",
                TO_JSON_ROOT / "dataset" / "labeled" / "test.jsonl",
                TO_JSON_ROOT / "dataset" / "predicted" / "predicted.jsonl",
            ),
            validator=validate_contains("Overall field accuracy", "ticket JSON accuracy was reported"),
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
        patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
        patch_status = patch.get("patch_status", "unknown")
        return f"pipeline status={status}; patch_status={patch_status}; localization_candidates={len(candidates)}"

    return _validate


def validate_fault_localization(output_path: Path) -> Callable[[subprocess.CompletedProcess[str]], str]:
    def _validate(_: subprocess.CompletedProcess[str]) -> str:
        data = read_json(output_path)
        candidates = data.get("localized_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("no localized_candidates were produced")
        top = candidates[0] if isinstance(candidates[0], dict) else {}
        file_path = top.get("file_path") or top.get("file") or "unknown"
        score = top.get("score", top.get("final_score", "unknown"))
        return f"localized_candidates={len(candidates)}; top_file={file_path}; top_score={score}"

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

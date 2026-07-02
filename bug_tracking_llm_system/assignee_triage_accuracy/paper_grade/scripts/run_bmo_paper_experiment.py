from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PAPER_ROOT.parent
SYSTEM_ROOT = PAPER_ROOT.parents[2]

DEFAULT_BASELINES = [
    "global_majority",
    "component_majority",
    "product_component_majority",
    "bm25_text_knn",
    "current_assignee_triager",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full paper-grade BMO assignee-triage experiment.")
    parser.add_argument("--dataset", default="bmo_paper", help="Dataset prefix for processed files and reports.")
    parser.add_argument("--products", nargs="+", default=["Core", "Firefox"])
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-bugs", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--include-comments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-neighbors", type=int, default=25)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    args = parser.parse_args()

    raw_path = PAPER_ROOT / "data" / "raw" / f"{args.dataset}_raw.jsonl"
    fetch_manifest_path = PAPER_ROOT / "data" / "raw" / f"{args.dataset}_fetch_manifest.json"
    processed_dir = PAPER_ROOT / "data" / "processed"
    profile_dir = EVAL_ROOT / "assignee_profiles"
    profile_path = profile_dir / f"{args.dataset}_assignee_profiles.json"
    report_dir = PAPER_ROOT / "reports" / args.dataset
    run_manifest_path = report_dir / f"{args.dataset}_run_manifest.json"

    report_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    commands: list[list[str]] = []

    if not args.skip_fetch:
        fetch_cmd = [
            sys.executable,
            str(PAPER_ROOT / "scripts" / "fetch_bmo_assignee_dataset.py"),
            "--products",
            *args.products,
            "--start-date",
            args.start_date,
            "--limit",
            str(args.limit),
            "--max-bugs",
            str(args.max_bugs),
            "--sleep",
            str(args.sleep),
            "--output",
            str(raw_path),
            "--manifest",
            str(fetch_manifest_path),
        ]
        if args.end_date:
            fetch_cmd.extend(["--end-date", args.end_date])
        if args.include_comments:
            fetch_cmd.append("--include-comments")
        if args.insecure:
            fetch_cmd.append("--insecure")
        run(fetch_cmd)
        commands.append(fetch_cmd)

    prepare_cmd = [
        sys.executable,
        str(PAPER_ROOT / "scripts" / "prepare_paper_grade_dataset.py"),
        "--raw",
        str(raw_path),
        "--dataset",
        args.dataset,
        "--output-dir",
        str(processed_dir),
        "--min-train-assignee-count",
        str(args.min_train_assignee_count),
    ]
    run(prepare_cmd)
    commands.append(prepare_cmd)

    profile_cmd = [
        sys.executable,
        str(EVAL_ROOT / "scripts" / "build_assignee_profiles.py"),
        "--dataset",
        args.dataset,
        "--history",
        str(processed_dir / f"{args.dataset}_history_train.jsonl"),
        "--output-dir",
        str(profile_dir),
    ]
    run(profile_cmd)
    commands.append(profile_cmd)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            str(PAPER_ROOT / "scripts" / "evaluate_paper_grade_benchmarks.py"),
            "--dataset",
            args.dataset,
            "--data-dir",
            str(processed_dir),
            "--output-dir",
            str(report_dir),
            "--top-k",
            str(args.top_k),
            "--bm25-neighbors",
            str(args.bm25_neighbors),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--baselines",
            *args.baselines,
        ]
        run(eval_cmd)
        commands.append(eval_cmd)

    manifest = {
        "dataset": args.dataset,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "raw_path": str(raw_path),
        "processed_dir": str(processed_dir),
        "profile_path": str(profile_path),
        "report_dir": str(report_dir),
        "commands": commands,
    }
    run_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(command, check=True, cwd=SYSTEM_ROOT)


if __name__ == "__main__":
    main()

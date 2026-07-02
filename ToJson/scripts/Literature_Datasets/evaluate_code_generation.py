"""
evaluate_code_generation.py

Evaluate Code Llama paper-style code generation benchmarks with official
ground truth tests.

The literature datasets prepared by prepare_code_llama_benchmarks.py already
contain ground truth:
- canonical_solution: reference solution
- tests: official assertions/check functions

For code-generation benchmarks, accuracy is measured by test pass rate
(pass@1), not by string equality.

Examples:
    # Verify that the literature ground truth solutions pass their tests.
    python3 scripts/Literature_Datasets/evaluate_code_generation.py --use-canonical

    # Evaluate model outputs saved as JSONL.
    python3 scripts/Literature_Datasets/evaluate_code_generation.py \
        --predictions dataset/literature/predicted/code_llama_solutions.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TASKS = "dataset/literature/processed/code_tasks.jsonl"
DEFAULT_REPORT = "dataset/literature/processed/pass_at_1_report.json"
PYTHON_TEST_PRELUDE = """
import collections
import functools
import itertools
import math
import operator
import re
from typing import *
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
    return records


def load_predictions(path: str) -> Dict[str, str]:
    predictions = {}
    for record in load_jsonl(path):
        task_id = record.get("id")
        code = (
            record.get("generated_solution")
            or record.get("prediction")
            or record.get("code")
            or record.get("completion")
            or ""
        )
        if task_id:
            predictions[str(task_id)] = str(code)
    return predictions


def strip_markdown_fence(code: str) -> str:
    code = code or ""
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).rstrip() + "\n"
    return code.rstrip() + "\n"


def indent_completion_body(code: str) -> str:
    lines = strip_markdown_fence(code).splitlines()
    indented = []
    for line in lines:
        if not line.strip():
            indented.append(line)
        elif line.startswith((" ", "\t")):
            indented.append(line)
        else:
            indented.append("    " + line)
    return "\n".join(indented).rstrip() + "\n"


def build_program(task: Dict[str, Any], solution: str) -> str:
    benchmark = task.get("benchmark")
    prompt = task.get("prompt", "")
    tests = "\n".join(task.get("tests", []) or [])
    entry_point = task.get("entry_point", "")
    solution = strip_markdown_fence(solution)

    if benchmark == "HumanEval":
        if f"def {entry_point}" in solution:
            program = PYTHON_TEST_PRELUDE + "\n" + solution
        else:
            completion = indent_completion_body(solution)
            program = PYTHON_TEST_PRELUDE + "\n" + prompt + "\n" + completion
        if entry_point:
            return program + "\n" + tests + f"\n\ncheck({entry_point})\n"
        return program + "\n" + tests

    if benchmark == "MBPP":
        return PYTHON_TEST_PRELUDE + "\n" + solution + "\n\n" + tests + "\n"

    return PYTHON_TEST_PRELUDE + "\n" + prompt + "\n" + solution + "\n\n" + tests + "\n"


def run_python(program: str, timeout_seconds: float) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        program_path = os.path.join(tmp_dir, "candidate.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(program)

        try:
            completed = subprocess.run(
                [sys.executable, program_path],
                cwd=tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, "TimeoutExpired"

    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output


def evaluate(
    tasks: List[Dict[str, Any]],
    predictions: Optional[Dict[str, str]],
    use_canonical: bool,
    timeout_seconds: float,
) -> Dict[str, Any]:
    results = []
    benchmark_totals: Counter[str] = Counter()
    benchmark_passed: Counter[str] = Counter()

    for task in tasks:
        task_id = str(task.get("id", ""))
        benchmark = str(task.get("benchmark", "unknown"))
        if use_canonical:
            solution = task.get("canonical_solution", "")
        else:
            solution = (predictions or {}).get(task_id, "")

        if not solution:
            passed = False
            error = "Missing solution"
        else:
            program = build_program(task, str(solution))
            passed, error = run_python(program, timeout_seconds)

        benchmark_totals[benchmark] += 1
        if passed:
            benchmark_passed[benchmark] += 1

        results.append(
            {
                "id": task_id,
                "benchmark": benchmark,
                "passed": passed,
                "error": "" if passed else error[:1000],
            }
        )

    total = len(results)
    passed_total = sum(1 for item in results if item["passed"])
    by_benchmark = {}
    for benchmark in sorted(benchmark_totals):
        total_for_benchmark = benchmark_totals[benchmark]
        passed_for_benchmark = benchmark_passed[benchmark]
        by_benchmark[benchmark] = {
            "passed": passed_for_benchmark,
            "total": total_for_benchmark,
            "pass_at_1": passed_for_benchmark / total_for_benchmark if total_for_benchmark else 0.0,
        }

    return {
        "mode": "canonical_ground_truth" if use_canonical else "model_predictions",
        "passed": passed_total,
        "total": total,
        "pass_at_1": passed_total / total if total else 0.0,
        "by_benchmark": by_benchmark,
        "failures": [item for item in results if not item["passed"]][:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--benchmark", default="", help="Optional benchmark filter, e.g. HumanEval or MBPP.")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--use-canonical",
        action="store_true",
        help="Evaluate canonical_solution ground truth instead of model predictions.",
    )
    args = parser.parse_args()

    if not args.use_canonical and not args.predictions:
        parser.error("Use --use-canonical or provide --predictions.")

    tasks = load_jsonl(args.tasks)
    if args.benchmark:
        tasks = [task for task in tasks if task.get("benchmark") == args.benchmark]
    if args.limit:
        tasks = tasks[: args.limit]

    predictions = None
    if args.predictions:
        predictions = load_predictions(args.predictions)

    report = evaluate(
        tasks=tasks,
        predictions=predictions,
        use_canonical=args.use_canonical,
        timeout_seconds=args.timeout,
    )

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"mode      : {report['mode']}")
    print(f"passed    : {report['passed']}/{report['total']}")
    print(f"pass@1    : {report['pass_at_1']:.4f}")
    for benchmark, metrics in report["by_benchmark"].items():
        print(
            f"{benchmark:10s}: {metrics['passed']}/{metrics['total']} "
            f"= {metrics['pass_at_1']:.4f}"
        )
    print(f"report    : {args.report}")


if __name__ == "__main__":
    main()

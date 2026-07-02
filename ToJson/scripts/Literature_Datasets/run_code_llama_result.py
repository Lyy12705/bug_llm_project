"""
run_code_llama_result.py

Main entry point for the Code Llama literature benchmark workflow.

Default behavior:
- Use the existing benchmark dataset and existing model predictions.
- Evaluate pass@1 without requiring long dataset path arguments.

Examples:
    # Evaluate current predictions.
    python3 scripts/Literature_Datasets/run_code_llama_result.py

    # Generate all predictions from scratch, then evaluate.
    python3 scripts/Literature_Datasets/run_code_llama_result.py --generate --regenerate

    # Quick smoke test on the first 10 HumanEval tasks.
    python3 scripts/Literature_Datasets/run_code_llama_result.py \
        --benchmark HumanEval --limit 10 --generate --regenerate

    # Verify canonical ground truth solutions.
    python3 scripts/Literature_Datasets/run_code_llama_result.py --canonical
"""

import argparse
import os
import subprocess
import sys
from typing import List


TASKS = "dataset/literature/processed/code_tasks.jsonl"
PREDICTIONS = "dataset/literature/predicted/code_llama_solutions.jsonl"
REPORT = "dataset/literature/processed/pass_at_1_report.json"


def run(command: List[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_dataset() -> None:
    if os.path.exists(TASKS):
        return
    run([sys.executable, "scripts/Literature_Datasets/prepare_code_llama_benchmarks.py"])


def build_suffix(benchmark: str, limit: int) -> str:
    parts = []
    if benchmark:
        parts.append(benchmark.lower())
    if limit:
        parts.append(str(limit))
    return "_" + "_".join(parts) if parts else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Download and normalize HumanEval/MBPP first.")
    parser.add_argument("--canonical", action="store_true", help="Evaluate canonical ground truth solutions.")
    parser.add_argument("--generate", action="store_true", help="Generate Code Llama predictions before evaluation.")
    parser.add_argument("--regenerate", action="store_true", help="Delete existing prediction output before generation.")
    parser.add_argument("--resume", action="store_true", help="Resume generation from existing prediction output.")
    parser.add_argument("--benchmark", default="", help="Optional benchmark filter, e.g. HumanEval or MBPP.")
    parser.add_argument("--limit", type=int, default=0, help="Optional task limit for quick tests.")
    parser.add_argument("--model", default=os.getenv("CODE_LLAMA_MODEL", "codellama:7b-instruct"))
    parser.add_argument(
        "--prompt-style",
        choices=["paper_aligned", "zero_shot"],
        default=os.getenv("CODE_LLAMA_PROMPT_STYLE", "paper_aligned"),
        help="paper_aligned uses HumanEval zero-shot and MBPP 3-shot prompting.",
    )
    args = parser.parse_args()

    if args.prepare:
        run([sys.executable, "scripts/Literature_Datasets/prepare_code_llama_benchmarks.py"])
    else:
        ensure_dataset()

    suffix = build_suffix(args.benchmark, args.limit)
    prediction_path = PREDICTIONS
    report_path = REPORT if not suffix else f"dataset/literature/processed/pass_at_1_report{suffix}.json"

    if args.canonical:
        command = [
            sys.executable,
            "scripts/Literature_Datasets/evaluate_code_generation.py",
            "--use-canonical",
            "--report",
            report_path,
        ]
        if args.benchmark:
            command.extend(["--benchmark", args.benchmark])
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        run(command)
        return

    if args.generate:
        if args.regenerate and os.path.exists(prediction_path):
            os.remove(prediction_path)

        command = [
            sys.executable,
            "scripts/Literature_Datasets/generate_code_llama_solutions.py",
            "--model",
            args.model,
            "--prompt-style",
            args.prompt_style,
        ]
        if args.resume:
            command.append("--resume")
        if args.benchmark:
            command.extend(["--benchmark", args.benchmark])
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        run(command)

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"找不到 {prediction_path}。請先加上 --generate 產生模型答案。"
        )

    command = [
        sys.executable,
        "scripts/Literature_Datasets/evaluate_code_generation.py",
        "--predictions",
        prediction_path,
        "--report",
        report_path,
    ]
    if args.benchmark:
        command.extend(["--benchmark", args.benchmark])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    run(command)

    print("\n結果判讀：")
    print("- pass@1 代表模型只產生一次答案時，通過官方 tests 的比例。")
    print("- 這是文獻 benchmark 的程式生成評估，不是 Ticket 使用者回報欄位抽取。")
    print("- HumanEval 偏函式補全；MBPP 偏從自然語言題目生成完整 Python 程式。")
    print("- 文獻中 Code Llama-Instruct 7B 約為 HumanEval 34.8%、MBPP 44.4%；MBPP 使用 3-shot。")
    print("- 若 MBPP 明顯低於 HumanEval，通常表示 prompt、模型大小或完整題目轉程式能力需要調整。")


if __name__ == "__main__":
    main()

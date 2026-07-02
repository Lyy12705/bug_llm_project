"""
prepare_code_llama_benchmarks.py

Download and normalize public benchmark datasets used by the Code Llama paper.

The Code Llama paper evaluates on several code-generation benchmarks. This
script prepares the small public benchmarks that are easiest to reproduce in a
student project:

- HumanEval
- MBPP

Outputs:
- dataset/literature/raw/humaneval.jsonl
- dataset/literature/raw/mbpp.jsonl
- dataset/literature/processed/code_tasks.jsonl
- dataset/literature/processed/summary.json
"""

import argparse
import gzip
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List

import requests


PAPER_REFERENCE = "Roziere et al., 2024, Code Llama: Open Foundation Models for Code"

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"


def fetch_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_humaneval() -> List[Dict[str, Any]]:
    content = gzip.decompress(fetch_bytes(HUMANEVAL_URL)).decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_mbpp() -> List[Dict[str, Any]]:
    return json.loads(fetch_bytes(MBPP_URL).decode("utf-8"))


def normalize_humaneval(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in records:
        task_id = str(item.get("task_id", "")).strip()
        normalized.append(
            {
                "id": task_id,
                "benchmark": "HumanEval",
                "task_type": "code_generation",
                "language": "python",
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "tests": [item.get("test", "")],
                "entry_point": item.get("entry_point", ""),
                "metadata": {
                    "split": "test",
                    "paper_reference": PAPER_REFERENCE,
                    "source_url": HUMANEVAL_URL,
                    "original_task_id": task_id,
                },
            }
        )
    return normalized


def normalize_mbpp(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in records:
        task_id = str(item.get("task_id", "")).strip()
        tests = item.get("test_list", []) or []
        challenge_tests = item.get("challenge_test_list", []) or []
        test_setup = item.get("test_setup_code", "") or ""
        if test_setup:
            tests = [test_setup] + tests
        tests = tests + challenge_tests

        normalized.append(
            {
                "id": f"mbpp_{task_id}",
                "benchmark": "MBPP",
                "task_type": "code_generation",
                "language": "python",
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("code", ""),
                "tests": tests,
                "entry_point": "",
                "metadata": {
                    "split": "test",
                    "paper_reference": PAPER_REFERENCE,
                    "source_url": MBPP_URL,
                    "original_task_id": task_id,
                },
            }
        )
    return normalized


def write_summary(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    benchmarks = Counter(record["benchmark"] for record in records)
    summary = {
        "paper_reference": PAPER_REFERENCE,
        "total_records": len(records),
        "benchmarks": dict(sorted(benchmarks.items())),
        "schema": {
            "id": "string",
            "benchmark": "string",
            "task_type": "string",
            "language": "string",
            "prompt": "string",
            "canonical_solution": "string",
            "tests": ["string"],
            "entry_point": "string",
            "metadata": "object",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="dataset/literature",
        help="Directory for literature benchmark data.",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw")
    processed_dir = os.path.join(args.output_dir, "processed")

    humaneval_raw = load_humaneval()
    mbpp_raw = load_mbpp()

    write_jsonl(os.path.join(raw_dir, "humaneval.jsonl"), humaneval_raw)
    write_jsonl(os.path.join(raw_dir, "mbpp.jsonl"), mbpp_raw)

    normalized = normalize_humaneval(humaneval_raw) + normalize_mbpp(mbpp_raw)
    write_jsonl(os.path.join(processed_dir, "code_tasks.jsonl"), normalized)
    write_summary(os.path.join(processed_dir, "summary.json"), normalized)

    print(f"HumanEval records: {len(humaneval_raw)}")
    print(f"MBPP records     : {len(mbpp_raw)}")
    print(f"Total normalized: {len(normalized)}")
    print(f"Wrote -> {os.path.join(processed_dir, 'code_tasks.jsonl')}")


if __name__ == "__main__":
    main()

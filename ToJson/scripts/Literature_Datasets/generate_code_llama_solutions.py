"""
generate_code_llama_solutions.py

Generate Code Llama solutions for literature benchmark tasks.

The output can be evaluated with evaluate_code_generation.py:

    python3 scripts/Literature_Datasets/generate_code_llama_solutions.py --limit 10
    python3 scripts/Literature_Datasets/evaluate_code_generation.py \
        --predictions dataset/literature/predicted/code_llama_solutions.jsonl
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List

import requests


DEFAULT_TASKS = "dataset/literature/processed/code_tasks.jsonl"
DEFAULT_OUTPUT = "dataset/literature/predicted/code_llama_solutions.jsonl"
PAPER_REFERENCE = "Roziere et al., 2024, Code Llama: Open Foundation Models for Code"
MBPP_FEW_SHOT_EXAMPLES = """
Task:
Write a function to return the square of a number.
Solution:
def square(x):
    return x * x

Task:
Write a function to check whether a string is a palindrome.
Solution:
def is_palindrome(text):
    return text == text[::-1]

Task:
Write a function to return the largest number in a list.
Solution:
def max_in_list(values):
    return max(values)
""".strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {str(row.get("id", "")) for row in load_jsonl(path)}


def strip_markdown_fence(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.I)
    if match:
        return match.group(1).strip() + "\n"
    return text + "\n"


def build_prompt(task: Dict[str, Any], prompt_style: str) -> str:
    benchmark = task.get("benchmark", "")
    prompt = task.get("prompt", "")

    if benchmark == "HumanEval":
        instruction = f"""
Complete the following Python function.
Return only the code that should appear after the prompt.
Do not include Markdown or explanations.

{prompt}
""".strip()
    elif prompt_style == "paper_aligned":
        instruction = f"""
Write a correct Python solution for each programming task.
Return only executable Python code for the final task.
Do not include Markdown or explanations.

{MBPP_FEW_SHOT_EXAMPLES}

Task:
{prompt}
Solution:
""".strip()
    else:
        instruction = f"""
Write a correct Python solution for the following programming task.
Return only executable Python code.
Do not include Markdown or explanations.

Task:
{prompt}
""".strip()

    return f"[INST] {instruction} [/INST]"


def call_ollama(
    prompt: str,
    model: str,
    url: str,
    seed: int,
    num_ctx: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": 1024,
            "repeat_penalty": 1.0,
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("CODE_LLAMA_MODEL", "codellama:7b-instruct"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--benchmark", default="", help="Optional benchmark filter, e.g. HumanEval or MBPP.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        choices=["paper_aligned", "zero_shot"],
        default=os.getenv("CODE_LLAMA_PROMPT_STYLE", "paper_aligned"),
        help="paper_aligned uses HumanEval zero-shot and MBPP 3-shot prompting.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("CODE_LLAMA_SEED", "42")))
    parser.add_argument("--num-ctx", type=int, default=int(os.getenv("CODE_LLAMA_NUM_CTX", "8192")))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.benchmark:
        tasks = [task for task in tasks if task.get("benchmark") == args.benchmark]
    if args.limit:
        tasks = tasks[: args.limit]

    if not args.resume and os.path.exists(args.output):
        os.remove(args.output)

    seen = completed_ids(args.output) if args.resume else set()
    generated_count = 0
    for task in tasks:
        task_id = str(task.get("id", ""))
        if task_id in seen:
            continue

        prompt = build_prompt(task, args.prompt_style)
        raw_output = call_ollama(
            prompt=prompt,
            model=args.model,
            url=args.ollama_url,
            seed=args.seed,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
        )
        generated_solution = strip_markdown_fence(raw_output)
        append_jsonl(
            args.output,
            [
                {
                    "id": task_id,
                    "benchmark": task.get("benchmark"),
                    "generated_solution": generated_solution,
                    "generation_meta": {
                        "paper_reference": PAPER_REFERENCE,
                        "model": args.model,
                        "prompt_style": "Code Llama-Instruct [INST] instruction",
                        "benchmark_prompt_style": args.prompt_style,
                        "decoding": {
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": args.seed,
                            "num_ctx": args.num_ctx,
                        },
                    },
                }
            ],
        )
        generated_count += 1
        print(f"[{generated_count}/{len(tasks)}] wrote {task_id}")

    print(f"Wrote predictions -> {args.output}")


if __name__ == "__main__":
    main()

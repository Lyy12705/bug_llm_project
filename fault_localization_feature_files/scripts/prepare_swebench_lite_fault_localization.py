#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from utils.git_utils import extract_modified_files
except ModuleNotFoundError:
    def extract_modified_files(patch_text: str) -> list[str]:
        files: list[str] = []
        for line in (patch_text or "").splitlines():
            if line.startswith("+++ b/"):
                files.append(line[6:].strip())
            elif line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4 and parts[3].startswith("b/"):
                    files.append(parts[3][2:])
        unique: list[str] = []
        for file in files:
            if file and file != "/dev/null" and file not in unique:
                unique.append(file)
        return unique


DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
DATASET_LICENSE = "MIT"
DATASET_SOURCE_URL = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite"
PROJECT_SOURCE_URL = "https://github.com/SWE-bench/SWE-bench"
PARQUET_URLS = {
    "test": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet",
    "dev": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/dev-00000-of-00001.parquet",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SWE-bench Lite as a legal fault-localization evaluation dataset."
    )
    parser.add_argument("--split", choices=sorted(PARQUET_URLS), default="test", help="SWE-bench Lite split.")
    parser.add_argument(
        "--output-dir",
        default="data/fault_localization/swebench_lite",
        help="Directory for parquet, tickets.jsonl, gold.jsonl, and manifest.json.",
    )
    parser.add_argument("--parquet", default=None, help="Optional local parquet path. Skips download when provided.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of instances to export.")
    parser.add_argument("--force-download", action="store_true", help="Download parquet even if it already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{args.split}.parquet"
    if args.parquet:
        source_parquet = Path(args.parquet)
        if source_parquet.resolve() != parquet_path.resolve():
            shutil.copyfile(source_parquet, parquet_path)
    else:
        download_file(PARQUET_URLS[args.split], parquet_path, force=args.force_download)

    records = read_swebench_records(parquet_path, split=args.split, limit=args.limit)
    tickets = [ticket_record(record) for record in records]
    gold = [gold_record(record) for record in records]
    repos = sorted({record["repo"] for record in records})

    tickets_path = output_dir / f"{args.split}_tickets.jsonl"
    gold_path = output_dir / f"{args.split}_gold.jsonl"
    repo_manifest_path = output_dir / f"{args.split}_repos.jsonl"
    write_jsonl(tickets_path, tickets)
    write_jsonl(gold_path, gold)
    write_jsonl(repo_manifest_path, [repo_record(repo) for repo in repos])

    manifest = {
        "dataset_name": DATASET_NAME,
        "split": args.split,
        "rows": len(records),
        "unique_repositories": len(repos),
        "license": DATASET_LICENSE,
        "dataset_source_url": DATASET_SOURCE_URL,
        "project_source_url": PROJECT_SOURCE_URL,
        "parquet_url": PARQUET_URLS[args.split],
        "parquet_path": str(parquet_path),
        "tickets_path": str(tickets_path),
        "gold_path": str(gold_path),
        "repo_manifest_path": str(repo_manifest_path),
        "ground_truth_definition": "fixed_files are parsed from the developer patch field, excluding test_patch.",
        "evaluation_notes": [
            "Use problem_statement as the bug report.",
            "Use repo and base_commit to check out the exact source snapshot before running localization.",
            "SWE-bench Lite provides file-level ground truth from patches; symbol-level ground truth is not provided by default.",
        ],
    }
    manifest_path = output_dir / f"{args.split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def download_file(url: str, output: Path, *, force: bool = False) -> None:
    if output.exists() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output.write_bytes(response.read())


def read_swebench_records(parquet_path: Path, *, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    frame = pd.read_parquet(parquet_path)
    if limit is not None:
        frame = frame.head(limit)
    records = frame.to_dict(orient="records")
    return [normalize_record(record, split=split) for record in records]


def normalize_record(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    row = {key: normalize_value(value) for key, value in record.items()}
    row["fixed_files"] = extract_modified_files(str(row.get("patch") or ""))
    row["repository_url"] = f"https://github.com/{row.get('repo', '')}"
    row["split"] = split
    return row


def normalize_value(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return ""
    return value


def ticket_record(record: dict[str, Any]) -> dict[str, Any]:
    problem = str(record.get("problem_statement") or "")
    title = first_nonempty_line(problem) or str(record.get("instance_id") or "")
    return {
        "ticket_id": str(record.get("instance_id") or ""),
        "title": title,
        "description": problem,
        "bug_report": problem,
        "product": str(record.get("repo") or ""),
        "component": str(record.get("repo") or "").split("/")[-1],
        "source_dataset": DATASET_NAME,
        "source_split": str(record.get("split") or ""),
        "repo": str(record.get("repo") or ""),
        "repository_url": str(record.get("repository_url") or ""),
        "base_commit": str(record.get("base_commit") or ""),
        "created_at": str(record.get("created_at") or ""),
        "version": str(record.get("version") or ""),
        "hints_text": str(record.get("hints_text") or ""),
        "fail_to_pass": parse_json_list(record.get("FAIL_TO_PASS")),
        "pass_to_pass": parse_json_list(record.get("PASS_TO_PASS")),
    }


def gold_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_id": str(record.get("instance_id") or ""),
        "fixed_files": list(record.get("fixed_files") or []),
        "fixed_symbols": [],
        "source_dataset": DATASET_NAME,
        "source_split": str(record.get("split") or ""),
        "repo": str(record.get("repo") or ""),
        "repository_url": str(record.get("repository_url") or ""),
        "base_commit": str(record.get("base_commit") or ""),
        "license": DATASET_LICENSE,
        "ground_truth_source": "SWE-bench Lite patch field",
    }


def repo_record(repo: str) -> dict[str, str]:
    return {
        "repo": repo,
        "repository_url": f"https://github.com/{repo}",
        "suggested_cache_dir": f"data/fault_localization/swebench_lite/repos/{repo.replace('/', '__')}",
    }


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# SWE-bench Lite Fault Localization Dataset",
                "",
                "This directory is generated from the public `princeton-nlp/SWE-bench_Lite` dataset.",
                "It is prepared for file-level fault-localization evaluation.",
                "",
                "Source:",
                f"- Dataset: {DATASET_SOURCE_URL}",
                f"- Project: {PROJECT_SOURCE_URL}",
                f"- License: {DATASET_LICENSE}",
                "",
                "Generated files:",
                "- `<split>_tickets.jsonl`: bug reports for localization input.",
                "- `<split>_gold.jsonl`: file-level ground truth parsed from developer patches.",
                "- `<split>_repos.jsonl`: unique GitHub repositories referenced by the split.",
                "- `<split>_manifest.json`: source, license, and preparation metadata.",
                "",
                "Important evaluation note:",
                "SWE-bench Lite gives repository and base commit per instance. To evaluate localization,",
                "checkout each `repo` at its `base_commit`, run localization for that ticket, then evaluate",
                "against `<split>_gold.jsonl` with `scripts/evaluate_fault_localization.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

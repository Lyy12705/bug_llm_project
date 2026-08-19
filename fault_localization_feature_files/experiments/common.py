from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at line {line_number}: {input_path}")
                rows.append(value)
        return rows

    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def write_metrics(metrics: dict[str, Any], output: str | Path | None) -> None:
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", required=True, help="Gold JSON or JSONL file.")
    parser.add_argument("--pred", required=True, help="Predicted JSON or JSONL file.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")

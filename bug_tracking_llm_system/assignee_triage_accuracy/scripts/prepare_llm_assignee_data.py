from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = EVAL_ROOT / "paper_grade"
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "llm_assignee" / "data"


SYSTEM_PROMPT = (
    "You are an expert bug triager. Predict the single best valid assignee "
    "for the issue."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare conversational JSONL for LLM assignee experiments.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-roster-in-prompt",
        action="store_true",
        help="Include the train-only roster in prompts. Leave off for SFT-style data.",
    )
    args = parser.parse_args()

    train_path = args.train or args.data_dir / f"{args.dataset}_history_train.jsonl"
    validation_path = args.validation or args.data_dir / f"{args.dataset}_validation_set.jsonl"
    test_path = args.test or args.data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.roster or args.data_dir / f"{args.dataset}_candidate_roster.json"
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.dataset

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "train_sft.jsonl",
        [conversation_record(row, roster, include_answer=True, include_roster=args.include_roster_in_prompt) for row in train_rows],
    )
    write_jsonl(
        output_dir / "validation_prompts.jsonl",
        [conversation_record(row, roster, include_answer=False, include_roster=args.include_roster_in_prompt) for row in validation_rows],
    )
    write_jsonl(
        output_dir / "test_prompts.jsonl",
        [conversation_record(row, roster, include_answer=False, include_roster=args.include_roster_in_prompt) for row in test_rows],
    )
    write_json(output_dir / "candidate_roster.json", {"candidates": roster})
    manifest = {
        "dataset": args.dataset,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "include_roster_in_prompt": args.include_roster_in_prompt,
        "format": "messages_jsonl",
        "note": "Gold assignees are present only in train_sft assistant messages and metadata fields, not in inference prompts.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def conversation_record(
    row: dict[str, Any],
    roster: list[str],
    *,
    include_answer: bool,
    include_roster: bool,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": issue_prompt(row, roster, include_roster=include_roster)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": normalize_assignee(row.get("assignee"))})
    return {
        "ticket_id": row.get("ticket_id"),
        "expected_assignee": normalize_assignee(row.get("assignee")),
        "messages": messages,
    }


def issue_prompt(row: dict[str, Any], roster: list[str], *, include_roster: bool) -> str:
    lines = [
        "Below is an issue. Suggest the single best developer to resolve it.",
        "",
        "### Issue",
        f"Title: {clean(row.get('title'))}",
        f"Description: {clean(row.get('description')) or '(empty)'}",
        f"Product: {clean(row.get('product')) or 'unknown'}",
        f"Component: {clean(row.get('component')) or 'unknown'}",
        f"Priority: {clean(row.get('priority')) or 'unknown'}",
        f"Severity: {clean(row.get('severity')) or 'unknown'}",
        f"Status: {clean(row.get('status')) or 'unknown'}",
    ]
    if include_roster:
        lines.extend(["", "### Valid Assignees", ", ".join(roster)])
    lines.extend(["", "### Assignee:"])
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_roster(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing roster file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


if __name__ == "__main__":
    main()


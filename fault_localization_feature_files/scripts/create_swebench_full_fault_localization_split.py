#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = "swebench-full-stage1-protocol-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic SWE-bench full Development/Validation/Frozen Holdout split. "
            "Previously evaluated SWE-bench Lite instances are forced into Development."
        )
    )
    parser.add_argument("--tickets", required=True, help="Full SWE-bench ticket JSONL (2,294 rows).")
    parser.add_argument("--gold", required=True, help="Full SWE-bench file-level gold JSONL.")
    parser.add_argument(
        "--seen-tickets",
        required=True,
        help="Tickets used by earlier experiments; these IDs are excluded from Validation/Holdout.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--holdout-size", type=int, default=500)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tickets_path = Path(args.tickets)
    gold_path = Path(args.gold)
    seen_tickets_path = Path(args.seen_tickets)
    output_dir = Path(args.output_dir)

    tickets = read_jsonl(tickets_path)
    gold_rows = read_jsonl(gold_path)
    seen_rows = read_jsonl(seen_tickets_path)
    validate_source(tickets, gold_rows)

    seen_ids = {ticket_id(row) for row in seen_rows}
    full_ids = {ticket_id(row) for row in tickets}
    seen_in_full = seen_ids & full_ids
    unseen = [row for row in tickets if ticket_id(row) not in seen_in_full]
    requested_new_rows = args.validation_size + args.holdout_size
    if requested_new_rows > len(unseen):
        raise ValueError(
            f"Validation + Holdout request {requested_new_rows} rows, but only {len(unseen)} unseen rows exist."
        )

    holdout_rows, remaining = stratified_take(unseen, args.holdout_size, seed=f"{args.seed}:holdout")
    validation_rows, development_new = stratified_take(
        remaining,
        args.validation_size,
        seed=f"{args.seed}:validation",
    )
    development_seen = [row for row in tickets if ticket_id(row) in seen_in_full]
    # Keep each repository contiguous so successive base-commit indexes can
    # reuse unchanged file chunks without holding all repositories in memory.
    development_rows = repository_grouped_sort(
        development_seen + development_new,
        seed=f"{args.seed}:development",
    )
    validation_rows = repository_grouped_sort(validation_rows, seed=f"{args.seed}:validation-output")
    holdout_rows = repository_grouped_sort(holdout_rows, seed=f"{args.seed}:holdout-output")

    split_rows = {
        "development": development_rows,
        "validation": validation_rows,
        "frozen_holdout": holdout_rows,
    }
    split_ids = {name: {ticket_id(row) for row in rows} for name, rows in split_rows.items()}
    validate_split(split_ids, full_ids, seen_in_full, args.validation_size, args.holdout_size)

    gold_by_id = {ticket_id(row): row for row in gold_rows}
    outputs: list[Path] = []
    for split_name in split_rows:
        outputs.extend(
            [
                output_dir / f"{split_name}_tickets.jsonl",
                output_dir / f"{split_name}_gold.jsonl",
            ]
        )
    manifest_path = output_dir / "split_manifest.json"
    outputs.append(manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite the write-once split. Existing files: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, dict[str, Any]] = {}
    for split_name, rows in split_rows.items():
        ticket_output = output_dir / f"{split_name}_tickets.jsonl"
        gold_output = output_dir / f"{split_name}_gold.jsonl"
        split_gold = [gold_by_id[ticket_id(row)] for row in rows]
        write_jsonl(ticket_output, rows)
        write_jsonl(gold_output, split_gold)
        file_records[split_name] = {
            "tickets_path": str(ticket_output),
            "tickets_sha256": sha256_file(ticket_output),
            "gold_path": str(gold_output),
            "gold_sha256": sha256_file(gold_output),
            "tickets": len(rows),
            "repositories": len({str(row.get("repo") or "") for row in rows}),
            "repository_counts": repository_counts(rows),
        }

    manifest = {
        "protocol": "swebench-full-stage1-protocol-v1",
        "dataset": "princeton-nlp/SWE-bench test split",
        "source_tickets_path": str(tickets_path),
        "source_tickets_sha256": sha256_file(tickets_path),
        "source_gold_path": str(gold_path),
        "source_gold_sha256": sha256_file(gold_path),
        "source_rows": len(tickets),
        "previously_seen_tickets_path": str(seen_tickets_path),
        "previously_seen_source_rows": len(seen_rows),
        "previously_seen_ids_in_full": len(seen_in_full),
        "seed": args.seed,
        "split_strategy": (
            "Repository-stratified deterministic selection. All previously evaluated SWE-bench Lite "
            "instances are assigned to Development and excluded from Validation/Frozen Holdout."
        ),
        "split_files": file_records,
        "leakage_checks": {
            "development_validation_overlap": len(split_ids["development"] & split_ids["validation"]),
            "development_holdout_overlap": len(split_ids["development"] & split_ids["frozen_holdout"]),
            "validation_holdout_overlap": len(split_ids["validation"] & split_ids["frozen_holdout"]),
            "previously_seen_in_validation": len(seen_in_full & split_ids["validation"]),
            "previously_seen_in_frozen_holdout": len(seen_in_full & split_ids["frozen_holdout"]),
            "all_source_ids_assigned_once": len(set().union(*split_ids.values())) == len(full_ids),
        },
        "evaluation_policy": {
            "development": "Implementation checks and exploratory error analysis.",
            "validation": "Compare candidate-retrieval methods and select one configuration.",
            "frozen_holdout": "Run the selected configuration once after method selection.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def stratified_take(
    rows: list[dict[str, Any]],
    target: int,
    *,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("repo") or "")].append(row)
    for repo in groups:
        groups[repo] = stable_sort(groups[repo], seed=f"{seed}:{repo}")
    quotas = proportional_quotas({repo: len(group) for repo, group in groups.items()}, target)
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for repo in sorted(groups):
        quota = quotas[repo]
        selected.extend(groups[repo][:quota])
        remaining.extend(groups[repo][quota:])
    if len(selected) != target:
        raise AssertionError(f"Expected {target} selected rows, got {len(selected)}.")
    return selected, remaining


def proportional_quotas(capacities: dict[str, int], target: int) -> dict[str, int]:
    total = sum(capacities.values())
    if target < 0 or target > total:
        raise ValueError(f"Invalid stratified target {target} for {total} rows.")
    if total == 0:
        return {key: 0 for key in capacities}
    raw = {key: target * value / total for key, value in capacities.items()}
    quotas = {key: min(capacities[key], int(raw[key])) for key in capacities}
    remaining = target - sum(quotas.values())
    order = sorted(
        capacities,
        key=lambda key: (raw[key] - int(raw[key]), capacities[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise AssertionError("Unable to allocate all stratified rows.")
    return quotas


def validate_source(tickets: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> None:
    ticket_ids = [ticket_id(row) for row in tickets]
    gold_ids = [ticket_id(row) for row in gold_rows]
    if any(not value for value in ticket_ids + gold_ids):
        raise ValueError("Every ticket and gold row must contain a non-empty ticket_id.")
    if len(set(ticket_ids)) != len(ticket_ids):
        raise ValueError("Source tickets contain duplicate ticket IDs.")
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Source gold contains duplicate ticket IDs.")
    if set(ticket_ids) != set(gold_ids):
        raise ValueError("Source ticket and gold ID sets do not match.")


def validate_split(
    split_ids: dict[str, set[str]],
    full_ids: set[str],
    seen_ids: set[str],
    validation_size: int,
    holdout_size: int,
) -> None:
    development = split_ids["development"]
    validation = split_ids["validation"]
    holdout = split_ids["frozen_holdout"]
    if development & validation or development & holdout or validation & holdout:
        raise AssertionError("Split leakage detected: ticket IDs overlap.")
    if development | validation | holdout != full_ids:
        raise AssertionError("Split does not assign every source ticket exactly once.")
    if seen_ids & validation or seen_ids & holdout:
        raise AssertionError("Previously evaluated tickets leaked into Validation or Frozen Holdout.")
    if len(validation) != validation_size or len(holdout) != holdout_size:
        raise AssertionError("Requested Validation/Frozen Holdout sizes were not preserved.")


def stable_sort(rows: list[dict[str, Any]], *, seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{ticket_id(row)}".encode("utf-8")).hexdigest(),
    )


def repository_grouped_sort(rows: list[dict[str, Any]], *, seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("repo") or ""),
            hashlib.sha256(f"{seed}:{ticket_id(row)}".encode("utf-8")).hexdigest(),
        ),
    )


def repository_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("repo") or "")] += 1
    return dict(sorted(counts.items()))


def ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or "")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

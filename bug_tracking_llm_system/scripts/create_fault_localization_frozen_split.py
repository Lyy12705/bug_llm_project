#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic repository-disjoint development and frozen holdout splits."
    )
    parser.add_argument("--tickets", required=True, help="Input ticket JSONL.")
    parser.add_argument("--gold", required=True, help="Input fault-localization gold JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-ratio", type=float, default=0.30)
    parser.add_argument("--seed", default="bug-llm-fault-localization-v1")
    parser.add_argument(
        "--prior-exposure",
        choices=("unknown", "previously-evaluated", "untouched"),
        default="unknown",
        help="Whether this method-selection process has already seen results for any input rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split. Omit this for the normal write-once protocol.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tickets_path = Path(args.tickets).resolve()
    gold_path = Path(args.gold).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not 0.0 < args.holdout_ratio < 1.0:
        raise SystemExit("--holdout-ratio must be between 0 and 1")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite only before experiments begin")

    tickets = _read_jsonl(tickets_path)
    gold_rows = _read_jsonl(gold_path)
    split = build_repository_split(tickets, gold_rows, holdout_ratio=args.holdout_ratio, seed=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "development_tickets": output_dir / "development_tickets.jsonl",
        "development_gold": output_dir / "development_gold.jsonl",
        "holdout_tickets": output_dir / "frozen_holdout_tickets.jsonl",
        "holdout_gold": output_dir / "frozen_holdout_gold.jsonl",
    }
    _write_jsonl(paths["development_tickets"], split["development_tickets"])
    _write_jsonl(paths["development_gold"], split["development_gold"])
    _write_jsonl(paths["holdout_tickets"], split["holdout_tickets"])
    _write_jsonl(paths["holdout_gold"], split["holdout_gold"])

    manifest = {
        "schema_version": 1,
        "protocol": "repository_disjoint_frozen_holdout_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_holdout_ratio": args.holdout_ratio,
        "development_repositories": split["development_repositories"],
        "holdout_repositories": split["holdout_repositories"],
        "development_rows": len(split["development_tickets"]),
        "holdout_rows": len(split["holdout_tickets"]),
        "actual_holdout_ratio": round(len(split["holdout_tickets"]) / len(tickets), 6),
        "repository_overlap": sorted(
            set(split["development_repositories"]) & set(split["holdout_repositories"])
        ),
        "prior_exposure": args.prior_exposure,
        "holdout_status": _holdout_status(args.prior_exposure),
        "policy": _holdout_policy(args.prior_exposure),
        "inputs": {
            "tickets": {"path": str(tickets_path), "sha256": _sha256(tickets_path)},
            "gold": {"path": str(gold_path), "sha256": _sha256(gold_path)},
        },
        "outputs": {
            key: {"path": path.name, "sha256": _sha256(path)} for key, path in paths.items()
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "development_rows": manifest["development_rows"],
                "holdout_rows": manifest["holdout_rows"],
                "actual_holdout_ratio": manifest["actual_holdout_ratio"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_repository_split(
    tickets: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: str,
) -> dict[str, Any]:
    if not tickets:
        raise ValueError("ticket dataset is empty")
    ticket_ids = [_ticket_id(row) for row in tickets]
    if any(not ticket_id for ticket_id in ticket_ids) or len(ticket_ids) != len(set(ticket_ids)):
        raise ValueError("tickets must have unique non-empty ticket_id values")
    gold_by_id = {_ticket_id(row): row for row in gold_rows}
    missing_gold = sorted(set(ticket_ids) - set(gold_by_id))
    if missing_gold:
        raise ValueError(f"gold rows are missing {len(missing_gold)} ticket ids")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tickets:
        grouped[_repository(row)].append(row)
    if len(grouped) < 2:
        raise ValueError("repository-disjoint splitting requires at least two repositories")

    target = max(1, round(len(tickets) * holdout_ratio))
    ordered_repositories = sorted(
        grouped,
        key=lambda repo: (hashlib.sha256(f"{seed}|{repo}".encode("utf-8")).hexdigest(), repo),
    )
    holdout_repositories = _closest_repository_subset(
        ordered_repositories,
        grouped,
        target_rows=target,
        target_repository_count=max(1, round(len(grouped) * holdout_ratio)),
    )
    holdout_set = set(holdout_repositories)
    development_repositories = sorted(set(grouped) - holdout_set)

    development_tickets = [row for row in tickets if _repository(row) not in holdout_set]
    holdout_tickets = [row for row in tickets if _repository(row) in holdout_set]
    development_ids = {_ticket_id(row) for row in development_tickets}
    holdout_ids = {_ticket_id(row) for row in holdout_tickets}
    return {
        "development_repositories": development_repositories,
        "holdout_repositories": sorted(holdout_set),
        "development_tickets": development_tickets,
        "development_gold": [gold_by_id[ticket_id] for ticket_id in ticket_ids if ticket_id in development_ids],
        "holdout_tickets": holdout_tickets,
        "holdout_gold": [gold_by_id[ticket_id] for ticket_id in ticket_ids if ticket_id in holdout_ids],
    }


def _closest_repository_subset(
    ordered_repositories: list[str],
    grouped: dict[str, list[dict[str, Any]]],
    *,
    target_rows: int,
    target_repository_count: int,
) -> list[str]:
    """Choose a deterministic whole-repository subset nearest the requested size."""

    combinations: dict[int, tuple[str, ...]] = {0: ()}
    for repo in ordered_repositories:
        count = len(grouped[repo])
        updated = dict(combinations)
        for rows, selected in combinations.items():
            candidate = (*selected, repo)
            candidate_rows = rows + count
            current = updated.get(candidate_rows)
            if current is None or (abs(len(candidate) - target_repository_count), candidate) < (
                abs(len(current) - target_repository_count),
                current,
            ):
                updated[candidate_rows] = candidate
        combinations = updated

    minimum_repositories = 2 if len(ordered_repositories) >= 4 else 1
    candidates = [
        (rows, selected)
        for rows, selected in combinations.items()
        if minimum_repositories <= len(selected) < len(ordered_repositories)
    ]
    if not candidates:
        raise ValueError("unable to create a non-empty repository-disjoint holdout")
    _, best = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_rows),
            abs(len(item[1]) - target_repository_count),
            item[1],
        ),
    )
    return list(best)


def _repository(row: dict[str, Any]) -> str:
    value = str(row.get("repo") or row.get("repository") or row.get("repository_name") or "").strip()
    if value:
        return value
    ticket_id = _ticket_id(row)
    if "-" in ticket_id and "__" in ticket_id.split("-", 1)[0]:
        return ticket_id.split("-", 1)[0].replace("__", "/")
    raise ValueError(f"ticket {ticket_id or '<unknown>'} has no repository field")


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or row.get("id") or "").strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holdout_status(prior_exposure: str) -> str:
    if prior_exposure == "untouched":
        return "sealed_not_evaluated"
    if prior_exposure == "previously-evaluated":
        return "sealed_for_future_changes_prior_full_dataset_exposure"
    return "sealed_exposure_unknown_not_claimable_as_final_test"


def _holdout_policy(prior_exposure: str) -> str:
    base = (
        "Use only development_* files for future feature, prompt, threshold, and model selection. "
        "Record the evaluated commit SHA and preserve the manifest hashes."
    )
    if prior_exposure == "untouched":
        return base + " Run frozen_holdout_* once after the method and configuration are frozen."
    if prior_exposure == "previously-evaluated":
        return (
            base
            + " These rows appeared in an earlier full-dataset evaluation, so this split is only a prospective "
            "guardrail for future changes and must not be described as an untouched final test."
        )
    return base + " Do not claim final-test performance until prior exposure is established."


if __name__ == "__main__":
    raise SystemExit(main())

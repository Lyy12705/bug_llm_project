from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate assignee triage accuracy.")
    parser.add_argument("--dataset", default="controlled", help="Dataset prefix, e.g. controlled or eclipse.")
    parser.add_argument("--history", type=Path, default=None, help="History JSONL path.")
    parser.add_argument("--test", type=Path, default=None, help="Gold test JSONL path.")
    parser.add_argument("--roster", type=Path, default=None, help="Candidate roster JSON path.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT / "reports", help="Report output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum candidate list length.")
    args = parser.parse_args()

    data_dir = EVAL_ROOT / "data"
    history_path = args.history or data_dir / f"{args.dataset}_history_train.jsonl"
    test_path = args.test or data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.roster or data_dir / f"{args.dataset}_candidate_roster.json"

    history_rows = read_jsonl(history_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)
    roster_set = set(roster)

    component_counts, global_counts = build_history_counts(history_rows)
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=history_path,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)

    predictions = []
    for row in test_rows:
        priority_result = {"predicted_priority": row.get("priority", "P3")}
        result = triager.assign(row, priority_result)
        candidates = rank_candidates(
            row=row,
            prediction=result,
            component_counts=component_counts,
            global_counts=global_counts,
            config=config,
            roster_set=roster_set,
            top_k=args.top_k,
        )
        expected = normalize_assignee(row.get("assignee"))
        predicted = normalize_assignee(result.get("assignee"))
        reciprocal = reciprocal_rank(expected, candidates)
        predictions.append(
            {
                "ticket_id": row.get("ticket_id"),
                "title": row.get("title"),
                "component": normalize_component(row.get("component")),
                "priority": row.get("priority", "P3"),
                "expected_assignee": expected,
                "predicted_assignee": predicted,
                "ranked_candidates": candidates,
                "rank": rank_of(expected, candidates),
                "reciprocal_rank": reciprocal,
                "is_top1_correct": expected == predicted,
                "is_hit_at_3": expected in candidates[:3],
                "is_hit_at_5": expected in candidates[:5],
                "confidence": result.get("confidence"),
                "source": source_from_reason(str(result.get("reason", ""))),
                "reason": result.get("reason"),
                "error_type": error_type(
                    expected=expected,
                    predicted=predicted,
                    component=normalize_component(row.get("component")),
                    reciprocal=reciprocal,
                    source=source_from_reason(str(result.get("reason", ""))),
                    component_counts=component_counts,
                ),
            }
        )

    metrics = build_metrics(
        dataset=args.dataset,
        history_path=history_path,
        test_path=test_path,
        roster_path=roster_path,
        history_rows=history_rows,
        test_rows=test_rows,
        roster=roster,
        predictions=predictions,
        top_k=args.top_k,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.dataset}_predictions.jsonl", predictions)
    write_json(args.output_dir / f"{args.dataset}_metrics.json", metrics)
    write_error_analysis(args.output_dir / f"{args.dataset}_error_analysis.csv", predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_roster(path: Path) -> list[str]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        candidates = value.get("candidates", [])
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def build_history_counts(
    history_rows: list[dict[str, Any]]
) -> tuple[dict[str, Counter[str]], Counter[str]]:
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in history_rows:
        component = normalize_component(row.get("component"))
        assignee = normalize_assignee(row.get("assignee"))
        if not component or not assignee:
            continue
        component_counts[component][assignee] += 1
        global_counts[assignee] += 1
    return dict(component_counts), global_counts


def rank_candidates(
    *,
    row: dict[str, Any],
    prediction: dict[str, Any],
    component_counts: dict[str, Counter[str]],
    global_counts: Counter[str],
    config: PipelineConfig,
    roster_set: set[str],
    top_k: int,
) -> list[str]:
    component = normalize_component(row.get("component"))
    candidates: list[str] = []

    def add(value: Any) -> None:
        candidate = normalize_assignee(value)
        if not candidate or candidate in candidates:
            return
        if roster_set and candidate not in roster_set and candidate != "manual_triage":
            return
        candidates.append(candidate)

    for candidate in prediction.get("ranked_candidates", []):
        add(candidate)
    add(prediction.get("assignee"))
    for assignee, _ in component_counts.get(component, Counter()).most_common():
        add(assignee)
    add(mapped_owner(component, config))
    for assignee, _ in global_counts.most_common():
        add(assignee)
    add("manual_triage")
    return candidates[:top_k]


def mapped_owner(component: str, config: PipelineConfig) -> str | None:
    mapping = {key.lower(): value for key, value in config.component_owner_mapping.items()}
    if component in mapping:
        return mapping[component]
    for key, value in mapping.items():
        if key != "unknown" and key in component:
            return value
    return mapping.get("unknown")


def build_metrics(
    *,
    dataset: str,
    history_path: Path,
    test_path: Path,
    roster_path: Path,
    history_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: list[str],
    predictions: list[dict[str, Any]],
    top_k: int,
    config: PipelineConfig,
) -> dict[str, Any]:
    total = len(predictions)
    source_counts = Counter(row["source"] for row in predictions)
    error_counts = Counter(row["error_type"] for row in predictions)

    per_component: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["component"]].append(row)
    for component, rows in sorted(grouped.items()):
        per_component[component] = {
            "rows": len(rows),
            "top1_accuracy": ratio(sum(1 for row in rows if row["is_top1_correct"]), len(rows)),
            "hit_at_3": ratio(sum(1 for row in rows if row["is_hit_at_3"]), len(rows)),
            "mrr": ratio(sum(float(row["reciprocal_rank"]) for row in rows), len(rows)),
        }

    return {
        "dataset": dataset,
        "history_path": str(history_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "history_rows": len(history_rows),
        "test_rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "top_k": top_k,
        "top1_accuracy": ratio(sum(1 for row in predictions if row["is_top1_correct"]), total),
        "hit_at_3": ratio(sum(1 for row in predictions if row["is_hit_at_3"]), total),
        "hit_at_5": ratio(sum(1 for row in predictions if row["is_hit_at_5"]), total),
        "mrr": ratio(sum(float(row["reciprocal_rank"]) for row in predictions), total),
        "manual_triage_rate": ratio(
            sum(1 for row in predictions if row["predicted_assignee"] == "manual_triage"), total
        ),
        "source_breakdown": dict(sorted(source_counts.items())),
        "error_breakdown": dict(sorted(error_counts.items())),
        "per_component": per_component,
    }


def source_from_reason(reason: str) -> str:
    lowered = reason.lower()
    if "hybrid ranking" in lowered:
        return "hybrid"
    if "historical owner" in lowered:
        return "historical"
    if "component-owner mapping" in lowered:
        return "mapping"
    if "manual triage" in lowered:
        return "manual"
    return "unknown"


def error_type(
    *,
    expected: str,
    predicted: str,
    component: str,
    reciprocal: float,
    source: str,
    component_counts: dict[str, Counter[str]],
) -> str:
    if expected == predicted:
        return "correct"
    if predicted == "manual_triage":
        return "manual_triage_predicted"
    if component not in component_counts:
        return "cold_start_component"
    if component_counts.get(component, Counter()).get(expected, 0) <= 1:
        return "long_tail_assignee"
    if reciprocal > 0:
        return "misranked_candidate"
    if source == "mapping":
        return "mapping_mismatch"
    return "wrong_assignee"


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def reciprocal_rank(expected: str, candidates: list[str]) -> float:
    rank = rank_of(expected, candidates)
    return 0.0 if rank is None else 1.0 / rank


def ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_error_analysis(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "ticket_id",
        "component",
        "priority",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "source",
        "error_type",
        "confidence",
        "ranked_candidates",
        "reason",
        "title",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            output = dict(row)
            output["ranked_candidates"] = "|".join(row.get("ranked_candidates", []))
            writer.writerow({field: output.get(field, "") for field in fields})


if __name__ == "__main__":
    main()

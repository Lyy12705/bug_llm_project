from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assignee_open_set_common import wilson_interval, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate post-holdout error taxonomy and risk/coverage diagnostics."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--routing-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    artifact = read_object(args.routing_artifact)
    report = build_diagnostics(rows, artifact.get("routing_policy") or {})
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_diagnostics(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("diagnostics require at least one prediction")
    known = [row for row in rows if row.get("known_owner") is True]
    unseen = [row for row in rows if row.get("known_owner") is not True]
    error_taxonomy = Counter()
    routing_outcomes = Counter()
    for row in rows:
        expected = str(row.get("expected_assignee") or "")
        pool = [str(owner) for owner in row.get("candidate_pool") or []]
        is_known = row.get("known_owner") is True
        is_auto = row.get("routing_status") == "auto_assign"
        if not is_known:
            error_taxonomy[
                "unseen_owner_auto_error" if is_auto else "unseen_owner_manual_protected"
            ] += 1
        elif expected not in pool:
            error_taxonomy["candidate_miss"] += 1
        elif row.get("is_top1_correct") is True:
            error_taxonomy["correct_top1"] += 1
        else:
            error_taxonomy["ranking_miss"] += 1
        if is_auto:
            routing_outcomes[
                "auto_correct" if row.get("is_top1_correct") is True else "auto_error"
            ] += 1
        elif row.get("is_top1_correct") is True:
            routing_outcomes["policy_abstain_on_correct_top1"] += 1
        else:
            routing_outcomes["manual_fallback_on_incorrect_top1"] += 1

    recall_flat: dict[str, float] = {}
    for cutoff in (10, 30, 50):
        recall_flat[f"candidate_recall_at_{cutoff}"] = _candidate_recall(rows, cutoff)
        recall_flat[f"known_owner_candidate_recall_at_{cutoff}"] = _candidate_recall(
            known, cutoff
        )
    components: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        components[str(row.get("component") or "unknown")].append(row)
    component_metrics = {
        name: _slice_metrics(members)
        for name, members in sorted(
            components.items(), key=lambda item: (-len(item[1]), item[0])
        )[:30]
    }
    source_metrics = {
        str(source_count): _slice_metrics(members)
        for source_count, members in sorted(_group_by_source_count(rows).items())
    }
    return {
        "schema_version": 1,
        "method": "posthoc_assignee_holdout_diagnostics_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "intended_use": "diagnosis_only_not_model_or_policy_selection",
        "rows": len(rows),
        "known_owner_rows": len(known),
        "unseen_owner_rows": len(unseen),
        "candidate_recall": recall_flat,
        "error_taxonomy": dict(sorted(error_taxonomy.items())),
        "routing_outcomes": dict(sorted(routing_outcomes.items())),
        "component_metrics_top30_by_volume": component_metrics,
        "candidate_source_count_metrics": source_metrics,
        "risk_coverage_curve": _risk_coverage_curve(rows, policy),
        "limitations": [
            "This report is post-holdout diagnosis and must not tune the frozen model or v2 policy.",
            "Offline predictions contain no service latency; latency is enforced by the live shadow gate.",
        ],
    }


def _candidate_recall(rows: list[dict[str, Any]], cutoff: int) -> float:
    if not rows:
        return 0.0
    hits = sum(
        str(row.get("expected_assignee") or "")
        in [str(owner) for owner in (row.get("candidate_pool") or [])[:cutoff]]
        for row in rows
    )
    return round(hits / len(rows), 6)


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auto = [row for row in rows if row.get("routing_status") == "auto_assign"]
    correct = sum(row.get("is_top1_correct") is True for row in auto)
    return {
        "rows": len(rows),
        "auto_rows": len(auto),
        "auto_coverage": round(len(auto) / len(rows), 6) if rows else 0.0,
        "auto_accuracy": round(correct / len(auto), 6) if auto else 0.0,
        "auto_accuracy_ci95": wilson_interval(correct, len(auto)),
        "top1_accuracy": round(
            sum(row.get("is_top1_correct") is True for row in rows) / len(rows), 6
        )
        if rows
        else 0.0,
    }


def _group_by_source_count(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            count = max(0, int(row.get("candidate_source_count") or 0))
        except (TypeError, ValueError):
            count = 0
        output[count].append(row)
    return dict(output)


def _risk_coverage_curve(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    risk_threshold = float(policy.get("open_set_threshold") or 0.0)
    minimum_sources = max(0, int(policy.get("minimum_candidate_source_count") or 0))
    unseen_total = sum(row.get("known_owner") is not True for row in rows)
    curve = []
    for confidence_threshold in [round(index / 20, 2) for index in range(21)]:
        accepted = [
            row
            for row in rows
            if float(row.get("open_set_probability") or 0.0) < risk_threshold
            and float(row.get("calibrated_probability") or 0.0) >= confidence_threshold
            and int(row.get("candidate_source_count") or 0) >= minimum_sources
        ]
        correct = sum(row.get("is_top1_correct") is True for row in accepted)
        unseen_auto = sum(row.get("known_owner") is not True for row in accepted)
        curve.append(
            {
                "confidence_threshold": confidence_threshold,
                "auto_rows": len(accepted),
                "coverage": round(len(accepted) / len(rows), 6),
                "accuracy": round(correct / len(accepted), 6) if accepted else 0.0,
                "accuracy_ci95": wilson_interval(correct, len(accepted)),
                "unseen_auto_rate": round(unseen_auto / unseen_total, 6)
                if unseen_total
                else 0.0,
            }
        )
    return curve


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("prediction JSONL rows must be objects")
                rows.append(value)
    return rows


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("routing artifact must be an object")
    return value


if __name__ == "__main__":
    main()

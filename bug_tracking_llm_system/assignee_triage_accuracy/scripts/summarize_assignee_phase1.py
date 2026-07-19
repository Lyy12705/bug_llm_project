from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assignee_phase1_common import DEFAULT_PHASE1_ROOT, build_metrics, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 1 assignee reliability artifacts.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    args = parser.parse_args()

    report_root = args.phase1_root / "reports" / args.dataset
    closed = read_json(report_root / "closed_set_current_repro" / "closed_set_current_repro_metrics.json")
    open_set = read_json(report_root / "prefilter_open_world_eval" / "open_set_metrics.json")
    assignment = read_json(report_root / "assignment_time_leakage" / "assignment_safe_metrics.json")

    open_groups = open_set["group_metrics_before_abstention"]
    low_frequency_count = open_groups.get("low_frequency_seen", build_metrics([]))["rows"]
    unseen_count = open_groups.get("unseen_assignee", build_metrics([]))["rows"]
    experiments = [
        experiment_row(
            "closed_set_current_repro",
            closed["train_count"],
            closed["validation_count"],
            closed["test_count"],
            closed["roster_size"],
            closed["metrics"],
            excluded_count=0,
            exclusion_reason="",
        ),
        experiment_row(
            "prefilter_open_world_eval",
            open_set["train_count"],
            open_set["validation_count"],
            open_set["test_count"],
            open_set["train_assignee_count"],
            open_set["test_metrics_before_abstention"],
            excluded_count=0,
            exclusion_reason="none in prefilter view; low-frequency and unseen are included",
        ),
        experiment_row(
            "assignment_safe_current_snapshot",
            assignment["views"]["current_snapshot"]["train_count"],
            assignment["views"]["current_snapshot"]["validation_count"],
            assignment["views"]["current_snapshot"]["test_count"],
            assignment["views"]["current_snapshot"]["roster_size"],
            assignment["views"]["current_snapshot"]["metrics"],
            excluded_count=assignment["views"]["current_snapshot"]["excluded_count"],
            exclusion_reason=assignment["views"]["current_snapshot"]["exclusion_reason"],
        ),
        experiment_row(
            "assignment_safe_conservative",
            assignment["views"]["conservative"]["train_count"],
            assignment["views"]["conservative"]["validation_count"],
            assignment["views"]["conservative"]["test_count"],
            assignment["views"]["conservative"]["roster_size"],
            assignment["views"]["conservative"]["metrics"],
            excluded_count=assignment["views"]["conservative"]["excluded_count"],
            exclusion_reason=assignment["views"]["conservative"]["exclusion_reason"],
        ),
        experiment_row(
            "assignment_safe_strict",
            assignment["views"]["strict"]["train_count"],
            assignment["views"]["strict"]["validation_count"],
            assignment["views"]["strict"]["test_count"],
            assignment["views"]["strict"]["roster_size"],
            assignment["views"]["strict"]["metrics"],
            excluded_count=assignment["views"]["strict"]["excluded_count"],
            exclusion_reason=assignment["views"]["strict"]["exclusion_reason"],
        ),
    ]
    summary = {
        "dataset": args.dataset,
        "report_root": str(report_root),
        "experiments": experiments,
        "key_findings": {
            "closed_set_top1": closed["metrics"]["top1_accuracy"],
            "prefilter_open_world_top1": open_set["test_metrics_before_abstention"]["top1_accuracy"],
            "low_frequency_seen_count": low_frequency_count,
            "unseen_assignee_count": unseen_count,
            "assignment_safe_conservative_top1": assignment["views"]["conservative"]["metrics"]["top1_accuracy"],
            "strict_assignment_safe_computable": assignment["views"]["strict"]["metrics"] is not None,
        },
        "limitations": [
            "Strict creation-time assignee semantics remain unverifiable without assignment history.",
            "Duplicate detection is an upstream feature and is outside the assignee-triage scope.",
            "Phase 1 does not modify current_assignee_triager or the original frozen closed-set benchmark.",
        ],
    }

    write_json(report_root / "phase1_reliability_summary.json", summary)
    write_summary(report_root / "phase1_reliability_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing Phase 1 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_row(
    name: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    roster_size: int,
    metrics: dict[str, Any] | None,
    *,
    excluded_count: int,
    exclusion_reason: str,
) -> dict[str, Any]:
    safe_metrics = metrics or build_metrics([])
    return {
        "experiment_name": name,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "roster_size": roster_size,
        "top1_accuracy": safe_metrics["top1_accuracy"],
        "hit_at_3": safe_metrics["hit_at_3"],
        "hit_at_5": safe_metrics["hit_at_5"],
        "hit_at_10": safe_metrics["hit_at_10"],
        "mrr": safe_metrics["mrr"],
        "top1_95ci": safe_metrics["top1_bootstrap_95ci"],
        "excluded_count": excluded_count,
        "exclusion_reason": exclusion_reason,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Assignee Triaging Phase 1 Reliability Summary",
        "",
        "| Experiment | Train | Validation | Test | Roster | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR | Excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["experiments"]:
        lines.append(
            f"| {row['experiment_name']} | {row['train_count']} | {row['validation_count']} | {row['test_count']} | "
            f"{row['roster_size']} | {row['top1_accuracy']:.6f} | {row['hit_at_3']:.6f} | "
            f"{row['hit_at_5']:.6f} | {row['hit_at_10']:.6f} | {row['mrr']:.6f} | {row['excluded_count']} |"
        )
    lines.extend(["", "## Key Findings", ""])
    for key, value in summary["key_findings"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

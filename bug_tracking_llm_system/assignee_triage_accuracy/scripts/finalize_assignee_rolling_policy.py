from __future__ import annotations

import argparse
import json
from pathlib import Path

from assignee_open_set_common import (
    build_drift_reference,
    deployment_gate,
    open_set_metrics,
    route_predictions,
    routing_metrics,
    routing_score_diagnostics,
    select_multi_window_routing_policy,
    write_json,
    write_jsonl,
)
from train_assignee_ltr import read_jsonl
from train_assignee_rolling_open_set import DEFAULT_OUTPUT_DIR, DRIFT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize rolling routing thresholds from already-frozen development scores."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--target-review-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--minimum-high-confidence", type=float, default=0.50)
    parser.add_argument("--minimum-review-confidence", type=float, default=0.20)
    parser.add_argument("--maximum-drift-psi", type=float, default=0.25)
    parser.add_argument("--minimum-auto-rows-per-window", type=int, default=250)
    parser.add_argument("--minimum-auto-accuracy-lower-bound", type=float, default=0.80)
    parser.add_argument("--maximum-unseen-rate-upper-bound", type=float, default=0.05)
    parser.add_argument("--minimum-component-auto-rows", type=int, default=30)
    parser.add_argument("--minimum-component-auto-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--minimum-component-auto-accuracy-lower-bound", type=float, default=0.60
    )
    parser.add_argument("--minimum-candidate-source-count", type=int, default=2)
    args = parser.parse_args()

    report_path = args.output_dir / "rolling_open_set_report.json"
    artifact_path = args.output_dir / "rolling_open_set_artifact.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    selection_names = list(report.get("protocol", {}).get("policy_selection_windows") or [])
    if len(selection_names) < 2:
        raise SystemExit("Report does not contain at least two policy selection windows")
    windows = {
        name: read_jsonl(args.output_dir / f"{name}_routing_predictions.jsonl")
        for name in selection_names
    }
    policy = select_multi_window_routing_policy(
        windows,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        target_review_accuracy=args.target_review_accuracy,
        minimum_high_confidence=args.minimum_high_confidence,
        minimum_review_confidence=args.minimum_review_confidence,
        minimum_auto_rows_per_window=max(1, args.minimum_auto_rows_per_window),
        minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        minimum_candidate_source_count=max(0, args.minimum_candidate_source_count),
        maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
        minimum_component_auto_rows=max(1, args.minimum_component_auto_rows),
        minimum_component_accuracy=args.minimum_component_auto_accuracy,
        minimum_component_accuracy_lower_bound=(
            args.minimum_component_auto_accuracy_lower_bound
        ),
    )
    for rows in windows.values():
        route_predictions(rows, policy)
    pooled = [row for rows in windows.values() for row in rows]
    drift_reference = build_drift_reference(
        pooled,
        DRIFT_FEATURES,
        maximum_psi=args.maximum_drift_psi,
    )
    routing = {name: routing_metrics(rows) for name, rows in windows.items()}
    window_gates = {
        name: deployment_gate(
            metrics,
            target_auto_accuracy=args.target_auto_accuracy,
            minimum_auto_coverage=args.minimum_auto_coverage,
            maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
            minimum_auto_rows=max(1, args.minimum_auto_rows_per_window),
            minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
            maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
            minimum_component_auto_rows=max(1, args.minimum_component_auto_rows),
            minimum_component_accuracy=args.minimum_component_auto_accuracy,
            minimum_component_accuracy_lower_bound=(
                args.minimum_component_auto_accuracy_lower_bound
            ),
        )
        for name, metrics in routing.items()
    }
    passed = bool(policy.get("found") and all(gate["passed"] for gate in window_gates.values()))
    report.update(
        {
            "routing_policy": policy,
            "routing_by_selection_window": routing,
            "routing_diagnostics_by_selection_window": {
                name: routing_score_diagnostics(rows, policy) for name, rows in windows.items()
            },
            "open_set_by_selection_window": {
                name: open_set_metrics(rows, threshold=policy["open_set_threshold"])
                for name, rows in windows.items()
            },
            "development_gate": {
                "passed": passed,
                "requires_every_selection_window": True,
                "window_gates": window_gates,
                "blocker": (
                    "new_untouched_holdout_required" if passed
                    else "multi_window_routing_safety_gate_failed"
                ),
            },
            "policy_finalization": {
                "scores_reused_without_model_refit": True,
                "selection_windows": list(windows),
                "minimum_high_confidence": args.minimum_high_confidence,
                "minimum_review_confidence": args.minimum_review_confidence,
                "minimum_auto_rows_per_window": args.minimum_auto_rows_per_window,
                "minimum_auto_accuracy_lower_bound": args.minimum_auto_accuracy_lower_bound,
                "maximum_unseen_rate_upper_bound": args.maximum_unseen_rate_upper_bound,
                "minimum_candidate_source_count": max(
                    0, args.minimum_candidate_source_count
                ),
                "minimum_component_auto_rows": max(
                    1, args.minimum_component_auto_rows
                ),
                "minimum_component_auto_accuracy": (
                    args.minimum_component_auto_accuracy
                ),
                "minimum_component_auto_accuracy_lower_bound": (
                    args.minimum_component_auto_accuracy_lower_bound
                ),
            },
        }
    )
    artifact.update(
        {
            "routing_policy": policy,
            "drift_reference": drift_reference,
            "development_gate": report["development_gate"],
            "targets": {
                "target_auto_accuracy": args.target_auto_accuracy,
                "minimum_auto_coverage": args.minimum_auto_coverage,
                "maximum_unseen_auto_rate": args.maximum_unseen_auto_rate,
                "maximum_unseen_rate_upper_bound": (
                    args.maximum_unseen_rate_upper_bound
                ),
                "minimum_candidate_source_count": max(
                    0, args.minimum_candidate_source_count
                ),
                "minimum_auto_rows_per_window": max(
                    1, args.minimum_auto_rows_per_window
                ),
                "minimum_auto_accuracy_lower_bound": (
                    args.minimum_auto_accuracy_lower_bound
                ),
                "minimum_component_auto_rows": max(
                    1, args.minimum_component_auto_rows
                ),
                "minimum_component_auto_accuracy": (
                    args.minimum_component_auto_accuracy
                ),
                "minimum_component_auto_accuracy_lower_bound": (
                    args.minimum_component_auto_accuracy_lower_bound
                ),
            },
        }
    )
    write_json(report_path, report)
    write_json(artifact_path, artifact)
    for name, rows in windows.items():
        write_jsonl(args.output_dir / f"{name}_routing_predictions.jsonl", rows)
    print(json.dumps({"routing_policy": policy, "development_gate": report["development_gate"]}, indent=2))


if __name__ == "__main__":
    main()

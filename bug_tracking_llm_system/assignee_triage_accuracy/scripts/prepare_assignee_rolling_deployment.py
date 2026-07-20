from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import artifact_manifest_entry, sha256_file  # noqa: E402
from modules.assignee_rolling_deployment import (  # noqa: E402
    ROLLING_PATH_KEYS,
    load_rolling_deployment_bundle,
)

from assignee_open_set_common import write_json, write_jsonl  # noqa: E402
from evaluate_assignee_rolling_holdout import DEFAULT_ARTIFACT  # noqa: E402
from train_assignee_ltr import (  # noqa: E402
    DEFAULT_DATA_DIR,
    exclude_cross_split_overlap,
    read_label_map,
    row_sort_key,
)
from train_assignee_rolling_open_set import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    RAW_DIR,
    apply_label_availability,
    load_rows,
    read_label_availability,
)


DEFAULT_HOLDOUT_REPORT = (
    SYSTEM_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "reports"
    / "bmo_rolling_2022q1_holdout"
    / "rolling_holdout_report.json"
)
DEFAULT_OUTPUT_DIR = (
    SYSTEM_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "deployment"
    / "bmo_rolling_2022q1_candidate"
)
REQUIRED_HOLDOUT_CHECKS = {
    "auto_assignment_accuracy",
    "auto_assignment_coverage",
    "unseen_auto_assignment_rate",
    "minimum_holdout_rows",
    "minimum_auto_rows",
    "auto_accuracy_confidence_lower_bound",
    "component_accuracy_floor",
    "component_accuracy_confidence_floor",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained, review-gated rolling LTR deployment candidate. "
            "Missing roster or shadow evidence always leaves the bundle research-only."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--routing-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--holdout-report", type=Path, default=DEFAULT_HOLDOUT_REPORT)
    parser.add_argument("--shadow-report", type=Path)
    parser.add_argument("--active-roster", type=Path)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--base-raw", type=Path, default=RAW_DIR / "bmo_public_10k_raw.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bundle-valid-days", type=int, default=30)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.bundle_valid_days < 1 or args.bundle_valid_days > 90:
        raise SystemExit("--bundle-valid-days must be between 1 and 90")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty; use --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    routing = read_object(args.routing_artifact, "rolling routing artifact")
    ranker_artifact = read_object(
        args.model_dir / "candidate_ltr_artifact.json", "ranker artifact"
    )
    holdout = read_object(args.holdout_report, "holdout report")
    roster = (
        read_object(args.active_roster, "active roster")
        if args.active_roster
        else unavailable_roster()
    )
    shadow = (
        read_object(args.shadow_report, "shadow report")
        if args.shadow_report
        else unavailable_shadow_report()
    )

    output_paths = {
        "assignee_dataset_path": args.output_dir / "history_snapshot.jsonl",
        "assignee_active_roster_path": args.output_dir / "active_roster.json",
        "assignee_ranker_model_path": args.output_dir / "candidate_ltr_model.joblib",
        "assignee_ranker_artifact_path": args.output_dir / "candidate_ltr_artifact.json",
        "assignee_rolling_routing_artifact_path": args.output_dir
        / "rolling_open_set_artifact.json",
        "assignee_holdout_report_path": args.output_dir / "rolling_holdout_report.json",
        "assignee_shadow_report_path": args.output_dir / "shadow_report.json",
    }
    history, history_audit = build_history_snapshot(
        args.history,
        label_map_path=args.assignee_label_map,
        base_raw=args.base_raw,
    )
    write_jsonl(output_paths["assignee_dataset_path"], history)
    write_json(output_paths["assignee_active_roster_path"], roster)
    write_json(output_paths["assignee_shadow_report_path"], shadow)
    shutil.copy2(args.model_dir / "candidate_ltr_model.joblib", output_paths["assignee_ranker_model_path"])
    shutil.copy2(args.model_dir / "candidate_ltr_artifact.json", output_paths["assignee_ranker_artifact_path"])
    shutil.copy2(args.routing_artifact, output_paths["assignee_rolling_routing_artifact_path"])
    shutil.copy2(args.holdout_report, output_paths["assignee_holdout_report_path"])

    gates = rolling_deployment_gates(
        routing,
        ranker_artifact,
        holdout,
        roster,
        shadow,
        operator_approved=args.approve,
        bundled_routing_hash=sha256_file(output_paths["assignee_rolling_routing_artifact_path"]),
        history_rows=len(history),
    )
    approved = all(gates.values())
    created_at = datetime.now(UTC)
    pipeline_config = {
        "assignee_router_engine": "rolling_ltr_v1",
        **{key: path.name for key, path in output_paths.items()},
        "assignee_fail_closed": True,
        "assignee_execution_mode": "production" if approved else "shadow_only",
    }
    bundle = {
        "schema_version": 1,
        "artifact_type": "assignee_rolling_deployment_bundle",
        "deployment_status": "approved" if approved else "research_only",
        "approval_gate_passed": approved,
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(days=args.bundle_valid_days)).isoformat(),
        "approval_gates": gates,
        "blocked_by": [name for name, passed in gates.items() if not passed],
        "pipeline_config": pipeline_config,
        "history_snapshot_audit": history_audit,
        "artifact_manifest": {
            key: artifact_manifest_entry(path, args.output_dir)
            for key, path in output_paths.items()
        },
    }
    bundle_path = args.output_dir / "deployment_bundle.json"
    write_json(bundle_path, bundle)
    load_rolling_deployment_bundle(
        bundle_path,
        require_approved=approved,
        verify_integrity=True,
    )
    print(
        json.dumps(
            {
                "deployment_status": bundle["deployment_status"],
                "approval_gate_passed": approved,
                "blocked_by": bundle["blocked_by"],
                "history_rows": len(history),
                "bundle": str(bundle_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_history_snapshot(
    history_paths: list[Path],
    *,
    label_map_path: Path,
    base_raw: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not history_paths:
        raise ValueError("at least one history source is required")
    label_map = read_label_map(label_map_path)
    availability = read_label_availability(base_raw)
    history: list[dict[str, Any]] = []
    sources = []
    overlap_rows = 0
    for path in history_paths:
        loaded = apply_label_availability(load_rows(path, label_map), availability)
        unique, excluded = exclude_cross_split_overlap(loaded, history)
        history.extend(unique)
        overlap_rows += excluded
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows_loaded": len(loaded),
                "rows_included": len(unique),
                "overlap_rows_excluded": excluded,
            }
        )
    history.sort(key=row_sort_key)
    if not history:
        raise ValueError("history snapshot is empty")
    return history, {
        "method": "ordered_temporal_history_snapshot_v1",
        "sources": sources,
        "rows": len(history),
        "overlap_rows_excluded": overlap_rows,
        "history_start": str(history[0].get("created_at") or ""),
        "history_end": str(history[-1].get("created_at") or ""),
        "labels_include_availability_timestamp": all(
            bool(str(row.get("label_available_at") or "").strip()) for row in history
        ),
        "used_to_refit_ranker_or_policy": False,
    }


def rolling_deployment_gates(
    routing: dict[str, Any],
    ranker_artifact: dict[str, Any],
    holdout: dict[str, Any],
    roster: dict[str, Any],
    shadow: dict[str, Any],
    *,
    operator_approved: bool,
    bundled_routing_hash: str,
    history_rows: int,
) -> dict[str, bool]:
    holdout_gate = holdout.get("deployment_gate") or {}
    holdout_checks = holdout_gate.get("checks") or {}
    protocol = holdout.get("protocol") or {}
    roster_candidates = roster.get("candidates")
    roster_assignees = roster.get("assignees")
    shadow_gate = shadow.get("deployment_gate") or {}
    return {
        "frozen_development_gate_passed": routing.get("development_gate", {}).get("passed")
        is True,
        "ranker_artifact_matches": ranker_artifact.get("name") == routing.get("ranker_name"),
        "new_holdout_gate_passed": holdout_gate.get("passed") is True,
        "all_required_holdout_checks_passed": REQUIRED_HOLDOUT_CHECKS.issubset(holdout_checks)
        and all(holdout_checks.get(name) is True for name in REQUIRED_HOLDOUT_CHECKS),
        "holdout_manifest_verified": protocol.get("holdout_provenance", {}).get(
            "manifest_verified"
        )
        is True,
        "holdout_not_used_for_fitting": protocol.get("holdout_used_for_fitting") is False,
        "holdout_not_used_for_threshold_selection": protocol.get(
            "holdout_used_for_threshold_selection"
        )
        is False,
        "score_drift_check_passed": holdout_gate.get("score_drift_check") is True,
        "holdout_matches_routing_artifact": protocol.get("routing_artifact_sha256")
        == bundled_routing_hash,
        "history_snapshot_available": history_rows > 0,
        "history_snapshot_not_used_for_refit": True,
        "active_roster_schema_valid": valid_roster_shape(roster),
        "active_roster_review_confirmed": roster.get("review_confirmed") is True,
        "active_roster_nonempty": isinstance(roster_candidates, list)
        and bool(roster_candidates)
        and isinstance(roster_assignees, list)
        and bool(roster_assignees),
        "shadow_gate_passed": shadow_gate.get("passed") is True,
        "shadow_unseen_labels_complete": shadow_gate.get("checks", {}).get(
            "unseen_labels_available"
        )
        is True,
        "explicit_operator_approval": operator_approved,
    }


def valid_roster_shape(roster: dict[str, Any]) -> bool:
    candidates = roster.get("candidates")
    inactive = roster.get("inactive")
    assignees = roster.get("assignees")
    if roster.get("schema_version") != 1:
        return False
    if not isinstance(candidates, list) or not isinstance(inactive, list) or not isinstance(
        assignees, list
    ):
        return False
    candidate_values = [str(value).strip().lower() for value in candidates]
    inactive_values = {str(value).strip().lower() for value in inactive}
    if any(not value for value in candidate_values) or len(candidate_values) != len(
        set(candidate_values)
    ):
        return False
    if set(candidate_values) & inactive_values:
        return False
    active_records = {
        str(row.get("assignee") or "").strip().lower()
        for row in assignees
        if isinstance(row, dict) and row.get("active") is True
    }
    return set(candidate_values).issubset(active_records)


def unavailable_roster() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "not_provided",
        "review_confirmed": False,
        "assignees": [],
        "candidates": [],
        "inactive": [],
    }


def unavailable_shadow_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "shadow_evidence_not_provided",
        "metrics": {"rows": 0, "observation_days": 0},
        "deployment_gate": {
            "passed": False,
            "checks": {"minimum_observation": False, "unseen_labels_available": False},
            "failed_checks": ["minimum_observation", "unseen_labels_available"],
        },
    }


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label.title()} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()

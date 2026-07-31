from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import sha256_file  # noqa: E402

from assignee_open_set_common import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a primary-only routing artifact when auto-assignment gates pass but "
            "the optional Top-3 confirmation policy does not."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = read_object(args.report)
    artifact = read_object(args.artifact)
    derived_report, derived_artifact, manifest = finalize_primary_only(
        report,
        artifact,
        artifact_name=args.artifact_name,
        source_report_sha256=sha256_file(args.report),
        source_artifact_sha256=sha256_file(args.artifact),
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "rolling_open_set_report.json", derived_report)
    write_json(args.output_dir / "rolling_open_set_artifact.json", derived_artifact)
    write_json(args.output_dir / "derivation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def finalize_primary_only(
    report: dict[str, Any],
    artifact: dict[str, Any],
    *,
    artifact_name: str,
    source_report_sha256: str,
    source_artifact_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if artifact.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise ValueError("unsupported routing artifact")
    policy = artifact.get("routing_policy") or {}
    if policy.get("found") is not True:
        raise ValueError("primary auto-assignment policy was not found")
    targets = artifact.get("targets") or {}
    window_metrics = report.get("routing_by_selection_window") or {}
    if len(window_metrics) < 2:
        raise ValueError("at least two policy-selection windows are required")
    strict_v3 = "maximum_unseen_rate_upper_bound" in targets
    minimum_sources = int(targets.get("minimum_candidate_source_count", 0))
    if strict_v3 and int(policy.get("minimum_candidate_source_count", 0)) < max(
        2, minimum_sources
    ):
        raise ValueError("strict primary policy did not freeze two candidate source families")
    checks = {}
    for name, metrics in window_metrics.items():
        passed = bool(
            float(metrics.get("auto_assignment_accuracy", 0.0))
            >= float(targets["target_auto_accuracy"])
            and float(metrics.get("auto_assignment_coverage", 0.0))
            >= float(targets["minimum_auto_coverage"])
            and float(metrics.get("unseen_auto_assignment_rate", 1.0))
            < float(targets["maximum_unseen_auto_rate"])
        )
        if strict_v3:
            passed = bool(
                passed
                and int(metrics.get("auto_assignment_rows", 0))
                >= int(targets.get("minimum_auto_rows_per_window", 250))
                and float(
                    metrics.get("auto_assignment_accuracy_ci95", {"lower": 0.0}).get(
                        "lower", 0.0
                    )
                )
                >= float(targets.get("minimum_auto_accuracy_lower_bound", 0.80))
                and float(
                    metrics.get(
                        "unseen_auto_assignment_rate_ci95", {"upper": 1.0}
                    ).get("upper", 1.0)
                )
                < float(targets["maximum_unseen_rate_upper_bound"])
                and bool(
                    artifact.get("routing_policy", {})
                    .get("selection_window_metrics", {})
                    .get(name, {})
                    .get("component_safety", {})
                    .get("passed", False)
                )
            )
        checks[name] = passed
    if not all(checks.values()):
        raise ValueError("primary auto-assignment metrics did not pass every selection window")

    derived_report = copy.deepcopy(report)
    derived_artifact = copy.deepcopy(artifact)
    for payload in (derived_report.get("routing_policy", {}), derived_artifact["routing_policy"]):
        payload["t_low"] = payload["t_high"]
        payload["review_policy_found"] = False
        payload["review_policy_mode"] = "disabled_manual_triage_fallback"
        payload["review_policy_blocker"] = "top3_target_not_met_on_development_windows"
    gate = {
        "passed": True,
        "requires_every_selection_window": True,
        "primary_window_checks": checks,
        "requires_review_policy": False,
        "review_policy_found": False,
        "top3_confirmation_enabled": False,
        "manual_fallback_required": True,
        "blocker": "new_untouched_holdout_required",
    }
    derived_report["method"] = "derived_primary_only_rolling_routing_v2"
    derived_report["development_gate"] = gate
    derived_report["derivation"] = {
        "reason": "top3_confirmation_target_not_met_primary_auto_policy_retained",
        "source_report_sha256": source_report_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "thresholds_reoptimized": False,
        "selection_predictions_reused": True,
    }
    derived_artifact["name"] = artifact_name
    derived_artifact["development_gate"] = gate
    derived_artifact["review_mode"] = "manual_triage_only"
    derived_artifact["approval_note"] = (
        "Primary auto-assignment passed multi-window development gates. Top-3 confirmation "
        "is disabled because its 90% target was not met. Requires a new untouched temporal "
        "holdout, reviewed active roster, live shadow evidence, and explicit approval."
    )
    manifest = {
        "schema_version": 1,
        "derived_at": datetime.now(UTC).isoformat(),
        "artifact_name": artifact_name,
        "source_report_sha256": source_report_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "primary_window_checks": checks,
        "thresholds_reoptimized": False,
        "top3_confirmation_enabled": False,
    }
    return derived_report, derived_artifact, manifest


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()

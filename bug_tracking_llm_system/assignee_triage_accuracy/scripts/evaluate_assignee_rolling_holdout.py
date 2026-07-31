from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_eligibility import (  # noqa: E402
    annotate_eligible_owner_labels_from_snapshots,
    read_assignee_eligibility_index,
)

from assignee_open_set_common import (
    apply_top5_assist_policy,
    calibration_metrics,
    deployment_gate,
    evaluate_score_drift,
    open_set_metrics,
    predict_portable_logistic,
    route_predictions,
    routing_metrics,
    routing_score_diagnostics,
    top5_assist_gate,
    top5_assist_metrics,
    write_json,
    write_jsonl,
)
from evaluate_assignee_open_set_holdout import recommendation_metrics
from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    build_semantic_backend,
    read_label_map,
)
from train_assignee_rolling_open_set import (
    DEFAULT_MODEL_DIR,
    RAW_DIR,
    RollingWindow,
    apply_label_availability,
    label_availability_audit,
    load_rows,
    prepare_rolling_windows,
    read_label_availability,
    score_window,
)


DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "reports"
    / "bmo_rolling_through_2021q3"
    / "rolling_open_set_artifact.json"
)
DEFAULT_EVALUATION_REGISTRY = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "sealed_holdout_evaluation_registry.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen rolling-history assignee routing bundle on a new holdout."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--routing-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl"
    )
    parser.add_argument("--base-raw", type=Path, default=RAW_DIR / "bmo_public_10k_raw.jsonl")
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl"
    )
    parser.add_argument("--q3", type=Path, default=RAW_DIR / "bmo_public_future_2020q3_raw.jsonl")
    parser.add_argument("--q4", type=Path, default=RAW_DIR / "bmo_public_future_2020q4_raw.jsonl")
    parser.add_argument("--q1", type=Path, default=RAW_DIR / "bmo_public_future_2021q1_raw.jsonl")
    parser.add_argument("--q2", type=Path, default=RAW_DIR / "bmo_public_future_2021q2_raw.jsonl")
    parser.add_argument("--q3-2021", type=Path, default=RAW_DIR / "bmo_public_future_2021q3_raw.jsonl")
    parser.add_argument(
        "--frozen-history",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override legacy frozen history with the exact repeated NAME=PATH list in the artifact.",
    )
    parser.add_argument(
        "--intermediate-history",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Temporally ordered history window after the frozen development windows. "
            "May be repeated. These rows update candidate history only and are never "
            "used to refit the ranker, calibrator, open-set detector, or routing policy."
        ),
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-name", default="new_holdout")
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        required=True,
        help="Sealed fetch manifest whose output path and SHA-256 must match the holdout.",
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eligibility-roster",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeat for non-overlapping reviewed snapshots when the frozen "
            "Top-5 policy uses eligible-owner correctness."
        ),
    )
    parser.add_argument(
        "--evaluation-registry",
        type=Path,
        default=DEFAULT_EVALUATION_REGISTRY,
        help="Append-only registry that prevents a sealed holdout from being evaluated twice.",
    )
    parser.add_argument(
        "--replay-evaluation-id",
        help=(
            "Reproduce an already-completed registered evaluation. A replay is auditable "
            "and cannot be presented as new untouched release evidence."
        ),
    )
    parser.add_argument("--confirm-untouched-holdout", action="store_true")
    parser.add_argument("--minimum-holdout-rows", type=int, default=2500)
    parser.add_argument("--minimum-auto-rows", type=int, default=250)
    parser.add_argument("--minimum-auto-accuracy-lower-bound", type=float, default=0.80)
    parser.add_argument("--maximum-unseen-rate-upper-bound", type=float, default=0.05)
    parser.add_argument("--minimum-component-auto-rows", type=int, default=30)
    parser.add_argument("--minimum-component-auto-accuracy", type=float, default=0.75)
    parser.add_argument("--minimum-component-auto-accuracy-lower-bound", type=float, default=0.60)
    parser.add_argument("--target-top3-confirmation-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-top5-assist-rows", type=int, default=250)
    parser.add_argument(
        "--minimum-top5-assist-accuracy-lower-bound", type=float, default=0.80
    )
    args = parser.parse_args()

    if not args.confirm_untouched_holdout:
        raise SystemExit("--confirm-untouched-holdout is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Holdout output directory must be empty: {args.output_dir}")
    artifact = json.loads(args.routing_artifact.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise SystemExit("Unsupported rolling open-set artifact")
    if artifact.get("deployment_status") != "research_only":
        raise SystemExit("Holdout evaluation expects a research-only artifact")
    if not (
        artifact.get("development_gate", {}).get("passed")
        or artifact.get("top5_assist_development_gate", {}).get("passed")
    ):
        raise SystemExit("Neither rolling development gate passed")
    ranker_artifact = json.loads(
        (args.model_dir / "candidate_ltr_artifact.json").read_text(encoding="utf-8")
    )
    if ranker_artifact.get("name") != artifact.get("ranker_name"):
        raise SystemExit("Ranker model directory does not match the rolling bundle")
    ranker = joblib.load(args.model_dir / "candidate_ltr_model.joblib")
    requirements = artifact["ranker_requirements"]
    semantic = build_semantic_backend(
        str(requirements.get("embedding_backend") or "none"),
        str(requirements.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2"),
    )
    if requirements.get("embedding_backend") == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The frozen ranker requires a local SBERT model: {semantic.status}")

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    base_history = apply_label_availability(load_rows(args.train, label_map), availability)
    legacy_frozen_specs = [
        ("validation", args.validation),
        ("2020_q3", args.q3),
        ("2020_q4", args.q4),
        ("2021_q1", args.q1),
        ("2021_q2", args.q2),
        ("2021_q3", args.q3_2021),
    ]
    frozen_specs = (
        [parse_named_path(value) for value in args.frozen_history]
        if args.frozen_history
        else legacy_frozen_specs
    )
    intermediate_specs = [parse_named_path(value) for value in args.intermediate_history]
    validate_window_names(frozen_specs, intermediate_specs, args.holdout_name)
    holdout_provenance = validate_holdout_manifest(args.holdout, args.holdout_manifest)
    routing_artifact_hash = sha256_file(args.routing_artifact)
    if args.replay_evaluation_id:
        validate_registered_replay(
            args.evaluation_registry,
            evaluation_id=args.replay_evaluation_id,
            holdout_sha256=str(holdout_provenance["sha256"]),
            routing_artifact_sha256=routing_artifact_hash,
        )
        evaluation_id = args.replay_evaluation_id
        evaluation_mode = "registered_replay"
    else:
        evaluation_id = claim_sealed_holdout(
            args.evaluation_registry,
            holdout_name=args.holdout_name,
            holdout_sha256=str(holdout_provenance["sha256"]),
            routing_artifact_sha256=routing_artifact_hash,
        )
        evaluation_mode = "first_formal_evaluation"
    all_windows = prepare_rolling_windows(
        base_history,
        [*frozen_specs, *intermediate_specs, (args.holdout_name, args.holdout)],
        label_map,
        availability,
        strict_duplicate_families=str(
            requirements.get("candidate_generator_version") or "v2"
        )
        == "v3",
    )
    frozen_windows = all_windows[: len(frozen_specs)]
    if [window.name for window in frozen_windows] != artifact.get("history_windows"):
        raise SystemExit("Rolling history windows do not match the frozen artifact")
    intermediate_windows = all_windows[len(frozen_specs) : -1]
    window = all_windows[-1]
    label_availability = label_availability_audit(
        [
            row
            for item in all_windows
            for row in [*item.history_before, *item.query_rows]
        ]
    )
    predictions = score_window(
        window,
        ranker,
        int(requirements["candidate_pool_size"]),
        float(requirements["half_life_days"]),
        float(requirements["smoothing_alpha"]),
        semantic,
        str(requirements.get("candidate_generator_version") or "v2"),
    )
    for row, probability in zip(
        predictions,
        predict_portable_logistic(predictions, artifact["calibrator"]),
        strict=True,
    ):
        row["calibrated_probability"] = round(float(probability), 8)
    for row, probability in zip(
        predictions,
        predict_portable_logistic(predictions, artifact["open_set_detector"]),
        strict=True,
    ):
        row["open_set_probability"] = round(float(probability), 8)

    policy = artifact["routing_policy"]
    route_predictions(predictions, policy)
    top5_policy = artifact.get("top5_assist_policy") or {"found": False}
    top5_correctness_field = str(
        top5_policy.get("correctness_field") or "is_top5_correct"
    )
    if top5_correctness_field == "is_top5_eligible_correct":
        if not args.eligibility_roster:
            raise SystemExit(
                "eligible-owner holdout evaluation requires --eligibility-roster"
            )
        eligibility_indexes = [
            read_assignee_eligibility_index(path)
            for path in args.eligibility_roster
        ]
        annotate_eligible_owner_labels_from_snapshots(
            predictions, eligibility_indexes
        )
        unlabeled = sum(
            row.get("eligibility_label_available") is not True
            for row in predictions
        )
        if unlabeled:
            raise SystemExit(
                "eligible-owner holdout cannot be evaluated: "
                f"{unlabeled} rows lack a time-valid reviewed snapshot"
            )
    apply_top5_assist_policy(predictions, top5_policy)
    routing_before_drift_gate = routing_metrics(predictions)
    top5_before_drift_gate = top5_assist_metrics(
        predictions, correctness_field=top5_correctness_field
    )
    drift = evaluate_score_drift(predictions, artifact["drift_reference"])
    if drift["severe_drift"]:
        for row in predictions:
            if row["routing_status"] == "auto_assign":
                row["routing_status"] = "top3_confirmation"
                row["fallback_reason"] = "score_distribution_drift"
            if row.get("top5_assist_available") is True:
                row["top5_assist_available"] = False
                row["top5_assist_status"] = "manual_triage_score_drift"
                row["top5_assist_reason"] = "score_distribution_drift"
    routing = routing_metrics(predictions)
    top5_metrics = top5_assist_metrics(
        predictions, correctness_field=top5_correctness_field
    )
    targets = artifact["targets"]
    gate = deployment_gate(
        routing,
        target_auto_accuracy=float(targets["target_auto_accuracy"]),
        minimum_auto_coverage=float(targets["minimum_auto_coverage"]),
        maximum_unseen_auto_rate=float(targets["maximum_unseen_auto_rate"]),
        minimum_holdout_rows=max(0, args.minimum_holdout_rows),
        minimum_auto_rows=max(0, args.minimum_auto_rows),
        minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        minimum_component_auto_rows=max(1, args.minimum_component_auto_rows),
        minimum_component_accuracy=args.minimum_component_auto_accuracy,
        minimum_component_accuracy_lower_bound=args.minimum_component_auto_accuracy_lower_bound,
        maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
    )
    gate["advisory_checks"] = {
        "top3_confirmation_accuracy": routing["top3_confirmation_accuracy"]
        >= args.target_top3_confirmation_accuracy,
    }
    gate["advisory_targets"] = {
        "target_top3_confirmation_accuracy": args.target_top3_confirmation_accuracy,
    }
    gate["score_drift_check"] = not drift["severe_drift"]
    strict_v3 = str(requirements.get("candidate_generator_version") or "v2") == "v3"
    protocol_checks = {
        "assignment_label_timestamps_explicit": (
            label_availability["all_rows_explicit"] if strict_v3 else True
        ),
        "candidate_source_support_frozen": (
            int(policy.get("minimum_candidate_source_count", 0)) >= 2
            if strict_v3
            else True
        ),
        "every_auto_assignment_has_two_source_families": (
            all(
                int(row.get("candidate_source_count", 0)) >= 2
                for row in predictions
                if row.get("routing_status") == "auto_assign"
            )
            if strict_v3
            else True
        ),
    }
    gate["v3_protocol_checks"] = protocol_checks if strict_v3 else {}
    gate["passed"] = bool(
        gate["passed"]
        and not drift["severe_drift"]
        and all(protocol_checks.values())
    )
    gate["production_integration_allowed"] = False
    gate["production_blocker"] = (
        "reviewed_active_roster_and_explicit_operator_approval_required"
        if gate["passed"]
        else "untouched_holdout_or_drift_safety_gate_failed"
    )
    top5_gate = top5_assist_gate(
        top5_metrics,
        target_accuracy=float(targets.get("target_top5_assist_accuracy", 0.85)),
        minimum_coverage=float(
            targets.get("minimum_top5_assist_coverage", 0.10)
        ),
        minimum_rows=max(1, args.minimum_top5_assist_rows),
        minimum_accuracy_lower_bound=(
            args.minimum_top5_assist_accuracy_lower_bound
        ),
    )
    top5_protocol_checks = {
        "assignment_label_timestamps_explicit": protocol_checks[
            "assignment_label_timestamps_explicit"
        ],
        "candidate_source_support_frozen": (
            int(top5_policy.get("minimum_candidate_source_count", 0)) >= 2
            if strict_v3
            else True
        ),
        "every_assisted_row_has_two_source_families": (
            all(
                int(row.get("candidate_source_count", 0)) >= 2
                for row in predictions
                if row.get("top5_assist_available") is True
            )
            if strict_v3
            else True
        ),
    }
    top5_gate["v3_protocol_checks"] = top5_protocol_checks if strict_v3 else {}
    top5_gate["score_drift_check"] = not drift["severe_drift"]
    top5_gate["passed"] = bool(
        top5_gate["passed"]
        and top5_policy.get("found") is True
        and not drift["severe_drift"]
        and all(top5_protocol_checks.values())
    )
    top5_gate["production_integration_allowed"] = False
    top5_gate["production_blocker"] = (
        "reviewed_active_roster_and_explicit_operator_approval_required"
        if top5_gate["passed"]
        else "untouched_holdout_top5_or_drift_gate_failed"
    )

    report = {
        "schema_version": 1,
        "method": "frozen_rolling_history_holdout_evaluation_v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deployment_status": "research_only",
        "protocol": {
            **window.audit,
            "frozen_artifact_history_windows": [window.name for window in frozen_windows],
            "intermediate_history_windows": [
                intermediate.audit for intermediate in intermediate_windows
            ],
            "intermediate_history_used_for_fitting": False,
            "intermediate_history_used_for_threshold_selection": False,
            "intermediate_history_update_only": True,
            "holdout_used_for_fitting": False,
            "holdout_used_for_threshold_selection": False,
            "rolling_history_updated_only_from_prior_windows": True,
            "label_availability_policy": artifact.get("label_availability_policy"),
            "label_availability_audit": label_availability,
            "assignment_time_history_available": label_availability[
                "all_rows_explicit"
            ],
            "holdout_provenance": holdout_provenance,
            "routing_artifact_sha256": routing_artifact_hash,
            "sealed_holdout_evaluation_id": evaluation_id,
            "evaluation_mode": evaluation_mode,
            "evaluation_registry": str(args.evaluation_registry.resolve()),
        },
        "recommendation_metrics": recommendation_metrics(predictions),
        "recommendation_metrics_known_owner": recommendation_metrics(
            [row for row in predictions if row["known_owner"]]
        ),
        "calibration_metrics": calibration_metrics(predictions),
        "open_set_metrics": open_set_metrics(
            predictions, threshold=float(policy["open_set_threshold"])
        ),
        "routing_before_drift_gate": routing_before_drift_gate,
        "top5_assist_before_drift_gate": top5_before_drift_gate,
        "score_drift": drift,
        "routing_metrics": routing,
        "top5_assist_metrics": top5_metrics,
        "routing_score_diagnostics": routing_score_diagnostics(predictions, policy),
        "deployment_gate": gate,
        "top5_assist_holdout_gate": top5_gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "rolling_holdout_report.json", report)
    write_jsonl(args.output_dir / "rolling_holdout_predictions.jsonl", predictions)
    if evaluation_mode == "first_formal_evaluation":
        complete_sealed_holdout_claim(
            args.evaluation_registry,
            evaluation_id=evaluation_id,
            report_path=args.output_dir / "rolling_holdout_report.json",
        )
    else:
        record_registered_replay(
            args.evaluation_registry,
            evaluation_id=evaluation_id,
            report_path=args.output_dir / "rolling_holdout_report.json",
            predictions_path=args.output_dir / "rolling_holdout_predictions.jsonl",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    name = name.strip()
    raw_path = raw_path.strip()
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise argparse.ArgumentTypeError(
            "intermediate history must use NAME=PATH with a safe 1-64 character name"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("intermediate history path must not be empty")
    return name, Path(raw_path)


def validate_window_names(
    frozen_specs: list[tuple[str, Path]],
    intermediate_specs: list[tuple[str, Path]],
    holdout_name: str,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", holdout_name):
        raise SystemExit("--holdout-name must be a safe 1-64 character name")
    names = [name for name, _ in [*frozen_specs, *intermediate_specs]] + [holdout_name]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"Rolling window names must be unique: {', '.join(duplicates)}")


def validate_holdout_manifest(holdout: Path, manifest_path: Path | None) -> dict[str, object]:
    resolved_holdout = holdout.resolve()
    provenance: dict[str, object] = {
        "path": str(resolved_holdout),
        "sha256": sha256_file(resolved_holdout),
        "manifest_verified": False,
    }
    if manifest_path is None:
        return provenance
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid holdout manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("Holdout manifest must be a JSON object")
    raw_output = str(manifest.get("output") or "").strip()
    if not raw_output:
        raise SystemExit("Holdout manifest is missing output")
    manifest_output = Path(raw_output)
    if not manifest_output.is_absolute():
        manifest_output = (PROJECT_ROOT / manifest_output).resolve()
    else:
        manifest_output = manifest_output.resolve()
    if manifest_output != resolved_holdout:
        raise SystemExit("Holdout manifest output does not match --holdout")
    expected_hash = str(manifest.get("output_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise SystemExit("Holdout manifest is missing a valid output_sha256")
    if manifest.get("sealed_output") is not True:
        raise SystemExit("Holdout manifest must confirm sealed_output=true")
    if expected_hash != provenance["sha256"]:
        raise SystemExit("Holdout SHA-256 does not match its manifest")
    provenance.update(
        {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_verified": True,
            "fetch_started_at": manifest.get("started_at"),
            "fetch_finished_at": manifest.get("finished_at"),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "bugs_written": manifest.get("bugs_written"),
            "complete": manifest.get("complete"),
            "sealed_output": True,
        }
    )
    return provenance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_sealed_holdout(
    registry_path: Path,
    *,
    holdout_name: str,
    holdout_sha256: str,
    routing_artifact_sha256: str,
) -> str:
    records = read_registry(registry_path)
    if any(str(row.get("holdout_sha256") or "") == holdout_sha256 for row in records):
        raise SystemExit(
            "This sealed holdout has already been claimed for evaluation; "
            "it cannot be reused as untouched evidence"
        )
    started_at = datetime.now(UTC).isoformat()
    evaluation_id = hashlib.sha256(
        f"{holdout_sha256}:{routing_artifact_sha256}:{started_at}".encode("utf-8")
    ).hexdigest()[:20]
    append_registry(
        registry_path,
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "started",
            "started_at": started_at,
            "holdout_name": holdout_name,
            "holdout_sha256": holdout_sha256,
            "routing_artifact_sha256": routing_artifact_sha256,
        },
    )
    return evaluation_id


def complete_sealed_holdout_claim(
    registry_path: Path, *, evaluation_id: str, report_path: Path
) -> None:
    append_registry(
        registry_path,
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
        },
    )


def validate_registered_replay(
    registry_path: Path,
    *,
    evaluation_id: str,
    holdout_sha256: str,
    routing_artifact_sha256: str,
) -> None:
    matching = [
        row for row in read_registry(registry_path) if row.get("evaluation_id") == evaluation_id
    ]
    combined = {key: value for row in matching for key, value in row.items()}
    if not matching or not any(row.get("status") == "completed" for row in matching):
        raise SystemExit("--replay-evaluation-id must reference a completed registry entry")
    if combined.get("holdout_sha256") != holdout_sha256:
        raise SystemExit("Registered replay holdout SHA-256 does not match")
    if combined.get("routing_artifact_sha256") != routing_artifact_sha256:
        raise SystemExit("Registered replay routing artifact SHA-256 does not match")


def record_registered_replay(
    registry_path: Path,
    *,
    evaluation_id: str,
    report_path: Path,
    predictions_path: Path,
) -> None:
    append_registry(
        registry_path,
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "replay_completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "predictions_sha256": sha256_file(predictions_path),
            "release_evidence": False,
        },
    )


def read_registry(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid sealed holdout evaluation registry: {path}") from exc
    return records


def append_registry(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

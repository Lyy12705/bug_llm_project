from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import load_active_assignee_set  # noqa: E402
from modules.assignee_eligibility import (  # noqa: E402
    annotate_eligible_owner_labels,
    load_assignee_eligibility_index,
)

from assignee_open_set_common import (  # noqa: E402
    apply_top5_assist_policy,
    predict_portable_logistic,
    route_predictions,
)
from calibrate_assignee_ltr import scored_predictions  # noqa: E402
from evaluate_assignee_rolling_holdout import DEFAULT_ARTIFACT  # noqa: E402
from train_assignee_ltr import (  # noqa: E402
    CandidateIndex,
    build_semantic_backend,
    exclude_cross_split_overlap,
    normalize_row,
    parse_timestamp,
    read_label_map,
    row_sort_key,
    row_timestamp,
)
from train_assignee_rolling_open_set import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    RAW_DIR,
    apply_label_availability,
    load_rows,
    read_label_availability,
)
from train_assignee_ltr import DEFAULT_DATA_DIR  # noqa: E402


def main() -> None:
    started_at = perf_counter()
    parser = argparse.ArgumentParser(
        description="Run the frozen rolling LTR policy in shadow mode without changing the real assignee."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--routing-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument(
        "--base-raw",
        type=Path,
        default=RAW_DIR / "bmo_public_10k_raw.jsonl",
        help="Raw base data used to recover assignment label-availability timestamps.",
    )
    parser.add_argument("--active-roster", type=Path, required=True)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("shadow_auto", "semi_auto_top5"),
        default="shadow_auto",
        help=(
            "semi_auto_top5 only exposes a validated five-person list for user "
            "selection and never changes the assignee."
        ),
    )
    args = parser.parse_args()

    artifact = _read_object(args.routing_artifact, "routing artifact")
    if artifact.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise SystemExit("Unsupported rolling routing artifact")
    if args.mode == "shadow_auto" and not artifact.get("development_gate", {}).get("passed"):
        raise SystemExit("Rolling routing artifact did not pass its development gate")
    policy = artifact.get("routing_policy") or {}
    if args.mode == "shadow_auto" and not policy.get("found"):
        raise SystemExit("Rolling routing artifact has no valid routing policy")

    ranker_artifact = _read_object(args.model_dir / "candidate_ltr_artifact.json", "ranker artifact")
    if ranker_artifact.get("name") != artifact.get("ranker_name"):
        raise SystemExit("Ranker and rolling routing artifact do not match")
    ranker = joblib.load(args.model_dir / "candidate_ltr_model.joblib")
    requirements = artifact["ranker_requirements"]
    semantic = build_semantic_backend(
        str(requirements.get("embedding_backend") or "none"),
        str(requirements.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2"),
    )
    if requirements.get("embedding_backend") == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"Frozen ranker requires a local SBERT model: {semantic.status}")

    ticket = _read_object(args.ticket, "ticket")
    query = normalize_row(ticket)
    if not str(query.get("ticket_id") or "").strip() or not str(query.get("title") or "").strip():
        raise SystemExit("Ticket requires ticket_id and title")
    query_time = parse_timestamp(row_timestamp(query))
    if query_time is None:
        raise SystemExit("Ticket requires a valid created_at/creation_time timestamp")

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    history = []
    overlap_rows = 0
    for path in args.history:
        loaded = apply_label_availability(load_rows(path, label_map), availability)
        loaded, removed = exclude_cross_split_overlap(
            loaded,
            history,
            strict_duplicate_families=str(
                requirements.get("candidate_generator_version") or "v2"
            )
            == "v3",
        )
        history.extend(loaded)
        overlap_rows += removed
    history, withheld = _history_available_before(history, query_time)
    if not history:
        raise SystemExit("No temporally prior assignee history is available")
    index = CandidateIndex(
        history,
        half_life_days=float(requirements["half_life_days"]),
        smoothing_alpha=float(requirements["smoothing_alpha"]),
        semantic=semantic,
        candidate_generator_version=str(
            requirements.get("candidate_generator_version") or "v2"
        ),
    )
    predictions = scored_predictions(
        [query], index, ranker, int(requirements["candidate_pool_size"])
    )
    if not predictions:
        raise SystemExit("Ranker produced no assignee candidates")
    prediction = predictions[0]
    prediction_created_at = datetime.now(UTC).isoformat()
    prediction["prediction_created_at"] = prediction_created_at
    prediction["calibrated_probability"] = round(
        float(predict_portable_logistic(predictions, artifact["calibrator"])[0]), 8
    )
    prediction["open_set_probability"] = round(
        float(predict_portable_logistic(predictions, artifact["open_set_detector"])[0]), 8
    )
    top5_policy = artifact.get("top5_assist_policy") or {"found": False}
    active = load_active_assignee_set(args.active_roster)
    eligibility_required = (
        top5_policy.get("correctness_field") == "is_top5_eligible_correct"
    )
    eligibility_review_confirmed = False
    eligible_assignees: set[str] | None = None
    eligibility_error = ""
    try:
        eligibility_index = load_assignee_eligibility_index(
            args.active_roster, as_of=prediction_created_at
        )
        eligible_assignees = eligibility_index.eligible_assignees_for_ticket(
            query,
            at_time=prediction_created_at,
        )
        annotate_eligible_owner_labels(predictions, eligibility_index)
        eligibility_review_confirmed = True
    except ValueError as exc:
        eligibility_error = str(exc)
    if args.mode == "shadow_auto":
        route_predictions(predictions, policy)
    apply_top5_assist_policy(predictions, top5_policy)

    ranked_active = [owner for owner in prediction["ranked_candidates"] if owner in active]
    prediction_latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
    if args.mode == "semi_auto_top5":
        recommendation = build_top5_assist_recommendation(
            prediction,
            active,
            policy_authorized=artifact.get("top5_assist_development_gate", {}).get(
                "passed"
            )
            is True,
            model_version=str(artifact.get("ranker_name") or ""),
            target_accuracy=float(top5_policy.get("target_accuracy", 0.85)),
            prediction_latency_ms=prediction_latency_ms,
            eligible_assignees=eligible_assignees,
            eligibility_required=eligibility_required,
            eligibility_review_confirmed=eligibility_review_confirmed,
            eligibility_error=eligibility_error,
        )
    else:
        predicted = str(prediction["predicted_assignee"])
        proposed_status = str(prediction["routing_status"])
        reason = str(prediction["fallback_reason"])
        if predicted not in active:
            proposed_status = "manual_triage_invalid_owner"
            reason = "predicted_owner_not_confirmed_active"
        if not ranked_active:
            proposed_status = "manual_triage_invalid_owner"
            reason = "no_active_ranked_candidate"
        would_auto_assign = proposed_status == "auto_assign" and predicted in active
        recommendation = {
            "assignee": predicted if would_auto_assign else "manual_triage",
            "suggested_assignee": predicted if predicted in active else (ranked_active[0] if ranked_active else ""),
            "ranked_candidates": ranked_active[:10],
            "confidence": prediction["calibrated_probability"],
            "raw_confidence": prediction["top_probability"],
            "open_set_risk": prediction["open_set_probability"],
            "routing_status": proposed_status,
            "fallback_reason": reason,
            "decision_reason_code": reason,
            "would_auto_assign": would_auto_assign,
            "model_version": str(artifact.get("ranker_name") or ""),
            "policy_version": "rolling_open_set_policy_v1",
            "candidate_source_count": prediction["candidate_source_count"],
            "decision_created_at": datetime.now(UTC).isoformat(),
            "prediction_latency_ms": prediction_latency_ms,
        }
    payload = {
        "schema_version": 1,
        "execution_mode": args.mode,
        "real_assignment_changed": False,
        "ticket_id": query["ticket_id"],
        "ticket_context": {
            "product": query.get("product"),
            "component": query.get("component"),
            "created_at": query.get("created_at") or query.get("creation_time"),
        },
        "history_rows": len(history),
        "history_rows_withheld_by_time": withheld,
        "history_overlap_rows_excluded": overlap_rows,
        "history_end": row_timestamp(history[-1]),
        "semantic_backend": semantic.status,
        "assignee_recommendation": recommendation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_top5_assist_recommendation(
    prediction: dict[str, Any],
    active_assignees: set[str],
    *,
    policy_authorized: bool,
    model_version: str = "",
    target_accuracy: float = 0.85,
    prediction_latency_ms: float = 0.0,
    eligible_assignees: set[str] | None = None,
    eligibility_required: bool = False,
    eligibility_review_confirmed: bool = False,
    eligibility_error: str = "",
) -> dict[str, Any]:
    """Build a fail-closed response with Top-5 choice and opt-in Top-1 actions."""
    raw_ranked = [
        str(owner).strip().lower()
        for owner in prediction.get("ranked_candidates", [])
        if str(owner).strip()
    ]
    raw_top5 = raw_ranked[:5]
    normalized_eligible = {
        str(owner).strip().lower()
        for owner in (eligible_assignees or set())
        if str(owner).strip()
    }
    eligible_top5_count = sum(owner in normalized_eligible for owner in raw_top5)
    eligibility_verified = bool(
        eligibility_review_confirmed
        and len(raw_top5) == 5
        and eligible_top5_count == 5
    )
    model_available = prediction.get("top5_assist_available") is True
    if not policy_authorized:
        available = False
        status = "manual_triage_top5_policy_unavailable"
        reason = "top5_development_gate_not_passed"
    elif not model_available:
        available = False
        status = str(
            prediction.get("top5_assist_status")
            or "manual_triage_top5_policy_unavailable"
        )
        reason = str(
            prediction.get("top5_assist_reason")
            or "top5_quality_policy_not_available"
        )
    elif len(raw_top5) != 5:
        available = False
        status = "manual_triage_insufficient_candidates"
        reason = "fewer_than_five_ranked_candidates"
    elif not all(owner in active_assignees for owner in raw_top5):
        available = False
        status = "manual_triage_invalid_owner"
        reason = "top5_contains_inactive_or_unreviewed_owner"
    elif eligibility_required and not eligibility_review_confirmed:
        available = False
        status = "manual_triage_eligibility_unavailable"
        reason = "reviewed_time_valid_eligibility_unavailable"
    elif eligibility_required and not eligibility_verified:
        available = False
        status = "manual_triage_invalid_owner"
        reason = "top5_contains_unqualified_owner"
    else:
        available = True
        status = "top5_user_confirmation"
        reason = "top5_candidates_require_user_selection"
    candidates = raw_top5 if available else []
    return {
        "assignee": "manual_triage",
        "suggested_assignee": candidates[0] if candidates else "",
        "ranked_candidates": candidates,
        "top5_candidates": candidates,
        "top5_assist_available": available,
        "selection_mode": "top5_user_confirmation",
        "requires_user_selection": available,
        "available_actions": (
            ["select_top5_candidate", "assign_top1", "manual_triage"]
            if available
            else ["manual_triage"]
        ),
        "top1_candidate": candidates[0] if candidates else "",
        "auto_top1_available": available,
        "auto_top1_requires_explicit_confirmation": available,
        "assignment_authorized": False,
        "autonomous_assignment": False,
        "top_k": 5,
        "quality_target": target_accuracy,
        "quality_scope": (
            "top5_reviewed_eligible_candidate_set"
            if eligibility_required
            else "top5_exact_historical_owner_candidate_set"
        ),
        "correctness_field": (
            "is_top5_eligible_correct"
            if eligibility_required
            else "is_top5_correct"
        ),
        "eligibility_required": eligibility_required,
        "eligibility_review_confirmed": eligibility_review_confirmed,
        "eligibility_verified": eligibility_verified,
        "eligible_top5_count": eligible_top5_count,
        "eligibility_definition": "reviewed_time_valid_capability_set_v1",
        "eligibility_error": eligibility_error if not eligibility_review_confirmed else "",
        "top1_quality_guaranteed_by_top5_policy": False,
        "top1_quality_warning": (
            "Top-5 quality evidence does not guarantee Top-1 accuracy."
            if available
            else ""
        ),
        "quality_evidence": "multi_window_development_gate",
        "confidence": prediction.get("calibrated_probability", 0.0),
        "raw_confidence": prediction.get("top_probability", 0.0),
        "open_set_risk": prediction.get("open_set_probability", 1.0),
        "routing_status": status,
        "fallback_reason": reason,
        "decision_reason_code": reason,
        "would_auto_assign": False,
        "auto_assignment_authorized": False,
        "needs_manual_triage": True,
        "model_version": model_version,
        "policy_version": "rolling_top5_assist_policy_v1",
        "candidate_source_count": prediction.get("candidate_source_count", 0),
        "decision_created_at": datetime.now(UTC).isoformat(),
        "prediction_latency_ms": prediction_latency_ms,
    }


def _history_available_before(
    rows: list[dict[str, Any]], query_time: Any
) -> tuple[list[dict[str, Any]], int]:
    included = []
    withheld = 0
    for row in sorted(rows, key=row_sort_key):
        created_at = parse_timestamp(row_timestamp(row))
        label_at = parse_timestamp(row.get("label_available_at") or row.get("last_change_time"))
        if created_at is None or created_at >= query_time or label_at is None or label_at > query_time:
            withheld += 1
            continue
        included.append(row)
    return included, withheld


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label.title()} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()

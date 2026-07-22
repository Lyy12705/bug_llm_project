from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import load_active_assignee_set, sha256_file  # noqa: E402
from modules.assignee_operational_guard import evaluate_ticket_guard  # noqa: E402
from modules.assignee_rolling_deployment import load_rolling_deployment_bundle  # noqa: E402

from assignee_open_set_common import predict_portable_logistic, route_predictions  # noqa: E402
from calibrate_assignee_ltr import scored_predictions  # noqa: E402
from train_assignee_ltr import (  # noqa: E402
    CandidateIndex,
    build_semantic_backend,
    normalize_row,
    parse_timestamp,
    row_sort_key,
    row_timestamp,
)
from train_assignee_rolling_open_set import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route one ticket with an approved rolling-LTR deployment bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--operational-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exit_code = 0
    try:
        payload = route_ticket(args.bundle, args.ticket, args.operational_state)
    except Exception as exc:  # fail closed at the process boundary
        payload = fail_closed_payload(exc)
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


def route_ticket(
    bundle_path: Path, ticket_path: Path, operational_state_path: Path
) -> dict[str, Any]:
    started_at = perf_counter()
    bundle = load_rolling_deployment_bundle(bundle_path)
    config = bundle["pipeline_config"]
    routing = read_object(config["assignee_rolling_routing_artifact_path"], "routing artifact")
    ranker_artifact = read_object(config["assignee_ranker_artifact_path"], "ranker artifact")
    if ranker_artifact.get("name") != routing.get("ranker_name"):
        raise ValueError("ranker and routing artifact versions do not match")
    ranker = joblib.load(config["assignee_ranker_model_path"])
    requirements = routing["ranker_requirements"]
    semantic = build_semantic_backend(
        str(requirements.get("embedding_backend") or "none"),
        str(requirements.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2"),
    )
    if requirements.get("embedding_backend") == "sbert" and not semantic.status.startswith(
        "sbert_local:"
    ):
        raise ValueError(f"approved rolling ranker requires a local SBERT model: {semantic.status}")

    ticket = read_object(ticket_path, "ticket")
    query = normalize_row(ticket)
    if not str(query.get("ticket_id") or "").strip() or not str(query.get("title") or "").strip():
        raise ValueError("ticket requires ticket_id and title")
    query_time = parse_timestamp(row_timestamp(query))
    if query_time is None:
        raise ValueError("ticket requires a valid created_at/creation_time timestamp")
    operational_state = read_object(operational_state_path, "operational state")
    guard = evaluate_ticket_guard(
        operational_state,
        bundle_sha256=sha256_file(bundle_path),
        ticket_id=str(query["ticket_id"]),
    )

    history = load_rows(config["assignee_dataset_path"], {})
    history, withheld = history_available_before(history, query_time)
    if not history:
        raise ValueError("no temporally prior assignee history is available")
    index = CandidateIndex(
        history,
        half_life_days=float(requirements["half_life_days"]),
        smoothing_alpha=float(requirements["smoothing_alpha"]),
        semantic=semantic,
    )
    predictions = scored_predictions(
        [query], index, ranker, int(requirements["candidate_pool_size"])
    )
    if not predictions:
        raise ValueError("ranker produced no assignee candidates")
    prediction = predictions[0]
    prediction["calibrated_probability"] = round(
        float(predict_portable_logistic(predictions, routing["calibrator"])[0]), 8
    )
    prediction["open_set_probability"] = round(
        float(predict_portable_logistic(predictions, routing["open_set_detector"])[0]), 8
    )
    route_predictions(predictions, routing["routing_policy"])

    active = load_active_assignee_set(config["assignee_active_roster_path"])
    if not active:
        raise ValueError("approved active roster is empty")
    ranked_active = [owner for owner in prediction["ranked_candidates"] if owner in active]
    predicted = str(prediction["predicted_assignee"])
    routing_status = str(prediction["routing_status"])
    reason = str(prediction["fallback_reason"])
    if predicted not in active:
        routing_status = "manual_triage_invalid_owner"
        reason = "predicted_owner_not_confirmed_active"
    if not ranked_active:
        routing_status = "manual_triage_invalid_owner"
        reason = "no_active_ranked_candidate"
    model_would_auto_assign = routing_status == "auto_assign" and predicted in active
    auto_assign = model_would_auto_assign and guard["auto_assignment_authorized"]
    if model_would_auto_assign and not guard["auto_assignment_authorized"]:
        routing_status = "manual_triage_rollout_holdback"
        reason = str(guard["reason"])
    suggested = predicted if predicted in active else (ranked_active[0] if ranked_active else "")
    prediction_latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
    return {
        "schema_version": 1,
        "execution_mode": "production_routing_decision",
        "ticket_id": query["ticket_id"],
        "history_rows": len(history),
        "history_rows_withheld_by_time": withheld,
        "semantic_backend": semantic.status,
        "bundle_created_at": bundle.get("created_at"),
        "bundle_expires_at": bundle.get("expires_at"),
        "operational_guard": guard,
        "assignee_recommendation": {
            "assignee": predicted if auto_assign else "manual_triage",
            "suggested_assignee": suggested,
            "ranked_candidates": ranked_active[:10],
            "confidence": prediction["calibrated_probability"],
            "raw_confidence": prediction["top_probability"],
            "open_set_risk": prediction["open_set_probability"],
            "routing_status": routing_status,
            "fallback_reason": reason,
            "decision_reason_code": reason,
            "auto_assignment_authorized": auto_assign,
            "model_would_auto_assign": model_would_auto_assign,
            "needs_manual_triage": not auto_assign,
            "model_version": str(routing.get("ranker_name") or ""),
            "policy_version": "rolling_open_set_policy_v1",
            "candidate_source_count": prediction["candidate_source_count"],
            "prediction_latency_ms": prediction_latency_ms,
        },
    }


def history_available_before(
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


def fail_closed_payload(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_mode": "fail_closed",
        "assignee_recommendation": {
            "assignee": "manual_triage",
            "suggested_assignee": "",
            "ranked_candidates": [],
            "confidence": 0.0,
            "open_set_risk": 1.0,
            "routing_status": "manual_triage_runtime_failure",
            "fallback_reason": "rolling_bundle_validation_or_runtime_failure",
            "decision_reason_code": "rolling_bundle_validation_or_runtime_failure",
            "auto_assignment_authorized": False,
            "needs_manual_triage": True,
        },
        "error_type": type(exc).__name__,
    }


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()

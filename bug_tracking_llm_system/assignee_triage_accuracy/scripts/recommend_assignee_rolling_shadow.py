from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import load_active_assignee_set  # noqa: E402

from assignee_open_set_common import (  # noqa: E402
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
    args = parser.parse_args()

    artifact = _read_object(args.routing_artifact, "routing artifact")
    if artifact.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise SystemExit("Unsupported rolling routing artifact")
    if not artifact.get("development_gate", {}).get("passed"):
        raise SystemExit("Rolling routing artifact did not pass its development gate")
    policy = artifact.get("routing_policy") or {}
    if not policy.get("found"):
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
        loaded, removed = exclude_cross_split_overlap(loaded, history)
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
    )
    predictions = scored_predictions(
        [query], index, ranker, int(requirements["candidate_pool_size"])
    )
    if not predictions:
        raise SystemExit("Ranker produced no assignee candidates")
    prediction = predictions[0]
    prediction["calibrated_probability"] = round(
        float(predict_portable_logistic(predictions, artifact["calibrator"])[0]), 8
    )
    prediction["open_set_probability"] = round(
        float(predict_portable_logistic(predictions, artifact["open_set_detector"])[0]), 8
    )
    route_predictions(predictions, policy)

    active = load_active_assignee_set(args.active_roster)
    ranked_active = [owner for owner in prediction["ranked_candidates"] if owner in active]
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
    }
    payload = {
        "schema_version": 1,
        "execution_mode": "shadow_only",
        "real_assignment_changed": False,
        "ticket_id": query["ticket_id"],
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

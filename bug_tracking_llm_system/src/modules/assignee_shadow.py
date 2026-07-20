from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from modules.assignee_deployment import wilson_interval


AUTO_STATUSES = {"auto_assign", "assigned"}
TOP3_STATUSES = {"top3_confirmation", "needs_confirmation"}
MANUAL_ASSIGNEE = "manual_triage"


def evaluate_shadow_feedback(
    rows: list[dict[str, Any]],
    *,
    target_auto_accuracy: float = 0.85,
    minimum_auto_coverage: float = 0.10,
    maximum_unseen_auto_rate: float = 0.05,
    minimum_rows: int = 500,
    minimum_observation_days: int = 28,
    minimum_auto_accuracy_lower_bound: float = 0.80,
    target_top3_accuracy: float = 0.90,
) -> dict[str, Any]:
    latest, duplicate_events, invalid_events = _latest_valid_feedback(rows)
    auto = [row for row in latest if _is_auto(row)]
    correct_auto = sum(_predicted_owner(row) == _final_owner(row) for row in auto)
    unseen_labeled = [row for row in latest if isinstance(row.get("known_owner"), bool)]
    unseen = [row for row in unseen_labeled if row["known_owner"] is False]
    unseen_auto = [row for row in unseen if _is_auto(row)]
    review = [row for row in latest if str(row.get("routing_status") or "") in TOP3_STATUSES]
    correct_review = sum(_final_owner(row) in _ranked_candidates(row)[:3] for row in review)
    observation_days = _observation_days(latest)
    provenance = _shadow_provenance(latest)

    metrics = {
        "rows": len(latest),
        "raw_feedback_events": len(rows),
        "duplicate_or_superseded_events": duplicate_events,
        "invalid_events": invalid_events,
        "observation_days": observation_days,
        "feedback_origin_breakdown": provenance["feedback_origin_breakdown"],
        "live_shadow_rows": provenance["live_shadow_rows"],
        "provenance_invalid_rows": provenance["invalid_rows"],
        "duplicate_source_event_ids": provenance["duplicate_source_event_ids"],
        "auto_assignment_rows": len(auto),
        "auto_assignment_correct_rows": correct_auto,
        "auto_assignment_coverage": _ratio(len(auto), len(latest)),
        "auto_assignment_accuracy": _ratio(correct_auto, len(auto)),
        "auto_assignment_accuracy_ci95": wilson_interval(correct_auto, len(auto)),
        "unseen_label_rows": len(unseen_labeled),
        "unseen_rows": len(unseen),
        "unseen_auto_assignment_rows": len(unseen_auto),
        "unseen_auto_assignment_rate": _ratio(len(unseen_auto), len(unseen)),
        "unseen_auto_assignment_rate_ci95": wilson_interval(len(unseen_auto), len(unseen)),
        "top3_confirmation_rows": len(review),
        "top3_confirmation_accuracy": _ratio(correct_review, len(review)),
        "routing_status_breakdown": dict(
            sorted(Counter(str(row.get("routing_status") or "missing") for row in latest).items())
        ),
        "decision_reason_breakdown": dict(
            sorted(
                Counter(
                    str(row.get("decision_reason_code") or row.get("fallback_reason") or "missing")
                    for row in latest
                ).items()
            )
        ),
        "component_metrics": _component_metrics(latest),
        "weekly_metrics": _weekly_metrics(latest),
    }
    checks = {
        "live_shadow_provenance_complete": provenance["complete"],
        "minimum_observation": len(latest) >= minimum_rows or observation_days >= minimum_observation_days,
        "auto_assignment_accuracy": metrics["auto_assignment_accuracy"] >= target_auto_accuracy,
        "auto_assignment_coverage": metrics["auto_assignment_coverage"] >= minimum_auto_coverage,
        "unseen_labels_available": len(unseen_labeled) == len(latest) and bool(latest),
        "unseen_auto_assignment_rate": bool(unseen_labeled)
        and metrics["unseen_auto_assignment_rate"] < maximum_unseen_auto_rate,
        "auto_accuracy_confidence_lower_bound": metrics["auto_assignment_accuracy_ci95"]["lower"]
        >= minimum_auto_accuracy_lower_bound,
    }
    advisory = {
        "top3_confirmation_accuracy": metrics["top3_confirmation_accuracy"] >= target_top3_accuracy,
        "two_latest_weeks_pass_primary_rates": _latest_weeks_pass(
            metrics["weekly_metrics"],
            target_auto_accuracy,
            minimum_auto_coverage,
            maximum_unseen_auto_rate,
        ),
    }
    return {
        "schema_version": 2,
        "method": "provenance_checked_append_only_assignee_shadow_evaluation_v2",
        "metrics": metrics,
        "deployment_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "targets": {
                "target_auto_accuracy": target_auto_accuracy,
                "minimum_auto_coverage": minimum_auto_coverage,
                "maximum_unseen_auto_rate": maximum_unseen_auto_rate,
                "minimum_rows": minimum_rows,
                "minimum_observation_days": minimum_observation_days,
                "minimum_auto_accuracy_lower_bound": minimum_auto_accuracy_lower_bound,
            },
            "advisory_checks": advisory,
            "advisory_targets": {"target_top3_accuracy": target_top3_accuracy},
        },
    }


def _latest_valid_feedback(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    latest: dict[str, tuple[datetime, int, dict[str, Any]]] = {}
    invalid = 0
    valid_events = 0
    for index, row in enumerate(rows):
        ticket_id = str(row.get("ticket_id") or "").strip()
        final_owner = _final_owner(row)
        created_at = _timestamp(row.get("created_at"))
        if not ticket_id or not final_owner or created_at is None:
            invalid += 1
            continue
        valid_events += 1
        candidate = (created_at, index, row)
        if ticket_id not in latest or candidate[:2] >= latest[ticket_id][:2]:
            latest[ticket_id] = candidate
    selected = [value[2] for value in sorted(latest.values(), key=lambda value: (value[0], value[1]))]
    return selected, valid_events - len(selected), invalid


def _shadow_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    origins = Counter(str(row.get("feedback_origin") or "unspecified").strip().lower() for row in rows)
    source_ids: Counter[tuple[str, str]] = Counter()
    invalid_rows = 0
    live_rows = 0
    for row in rows:
        origin = str(row.get("feedback_origin") or "unspecified").strip().lower()
        source_system = str(row.get("source_system") or "").strip()
        source_event_id = str(row.get("source_event_id") or "").strip()
        prediction_at = _timestamp(row.get("prediction_created_at"))
        finalized_at = _timestamp(row.get("created_at"))
        ticket_at = _timestamp(row.get("ticket_created_at"))
        if origin == "live_shadow":
            live_rows += 1
        valid = (
            origin == "live_shadow"
            and bool(source_system)
            and bool(source_event_id)
            and prediction_at is not None
            and finalized_at is not None
            and prediction_at <= finalized_at
            and (ticket_at is None or ticket_at <= prediction_at)
        )
        if not valid:
            invalid_rows += 1
        if source_system and source_event_id:
            source_ids[(source_system, source_event_id)] += 1
    duplicate_ids = sum(count - 1 for count in source_ids.values() if count > 1)
    return {
        "complete": bool(rows) and invalid_rows == 0 and duplicate_ids == 0,
        "live_shadow_rows": live_rows,
        "invalid_rows": invalid_rows,
        "duplicate_source_event_ids": duplicate_ids,
        "feedback_origin_breakdown": dict(sorted(origins.items())),
    }


def _component_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("component") or "unknown").strip().lower() or "unknown"].append(row)
    output = {}
    for component, members in sorted(groups.items()):
        auto = [row for row in members if _is_auto(row)]
        correct = sum(_predicted_owner(row) == _final_owner(row) for row in auto)
        output[component] = {
            "rows": len(members),
            "auto_assignment_rows": len(auto),
            "auto_assignment_coverage": _ratio(len(auto), len(members)),
            "auto_assignment_accuracy": _ratio(correct, len(auto)),
            "auto_assignment_accuracy_ci95": wilson_interval(correct, len(auto)),
        }
    return output


def _weekly_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        created_at = _timestamp(row.get("created_at"))
        if created_at is not None:
            year, week, _ = created_at.isocalendar()
            groups[f"{year}-W{week:02d}"].append(row)
    output = {}
    for week, members in sorted(groups.items()):
        auto = [row for row in members if _is_auto(row)]
        correct = sum(_predicted_owner(row) == _final_owner(row) for row in auto)
        labeled = [row for row in members if isinstance(row.get("known_owner"), bool)]
        unseen = [row for row in labeled if row["known_owner"] is False]
        output[week] = {
            "rows": len(members),
            "auto_assignment_rows": len(auto),
            "auto_assignment_coverage": _ratio(len(auto), len(members)),
            "auto_assignment_accuracy": _ratio(correct, len(auto)),
            "unseen_rows": len(unseen),
            "unseen_auto_assignment_rate": _ratio(sum(_is_auto(row) for row in unseen), len(unseen)),
            "unseen_labels_complete": len(labeled) == len(members),
        }
    return output


def _latest_weeks_pass(
    weeks: dict[str, Any], target_accuracy: float, minimum_coverage: float, maximum_unseen_rate: float
) -> bool:
    latest = [weeks[name] for name in sorted(weeks)[-2:]]
    return len(latest) == 2 and all(
        row["auto_assignment_accuracy"] >= target_accuracy
        and row["auto_assignment_coverage"] >= minimum_coverage
        and row["unseen_labels_complete"]
        and row["unseen_auto_assignment_rate"] < maximum_unseen_rate
        for row in latest
    )


def _is_auto(row: dict[str, Any]) -> bool:
    return (
        str(row.get("routing_status") or "") in AUTO_STATUSES
        and _predicted_owner(row) not in {"", MANUAL_ASSIGNEE}
    )


def _predicted_owner(row: dict[str, Any]) -> str:
    return str(row.get("predicted_assignee") or "").strip().lower()


def _final_owner(row: dict[str, Any]) -> str:
    return str(row.get("final_assignee") or "").strip().lower()


def _ranked_candidates(row: dict[str, Any]) -> list[str]:
    values = row.get("ranked_candidates")
    return [str(value).strip().lower() for value in values] if isinstance(values, list) else []


def _timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _observation_days(rows: list[dict[str, Any]]) -> int:
    timestamps = [value for row in rows if (value := _timestamp(row.get("created_at"))) is not None]
    if len(timestamps) < 2:
        return 0
    return max(0, (max(timestamps) - min(timestamps)).days)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0

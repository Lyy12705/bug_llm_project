from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any


OPERATIONAL_STATE_SCHEMA_VERSION = 1
ROLLOUT_STAGES = (0, 5, 10, 25)
REQUIRED_SHADOW_CHECKS = {
    "live_shadow_provenance_complete",
    "minimum_observation",
    "auto_assignment_accuracy",
    "auto_assignment_coverage",
    "unseen_labels_available",
    "unseen_auto_assignment_rate",
    "unseen_auto_assignment_rate_confidence_upper_bound",
    "auto_accuracy_confidence_lower_bound",
    "consecutive_weeks_pass_primary_rates",
    "prediction_latency_complete",
    "prediction_latency_p95",
    "invalid_event_rate",
}


def build_operational_state(
    bundle: dict[str, Any],
    shadow: dict[str, Any],
    roster: dict[str, Any],
    *,
    bundle_sha256: str,
    requested_rollout_percentage: int,
    previous_state: dict[str, Any] | None = None,
    now: datetime | None = None,
    minimum_stage_days: int = 7,
    state_ttl_hours: int = 24,
) -> dict[str, Any]:
    if requested_rollout_percentage not in ROLLOUT_STAGES:
        raise ValueError(f"rollout percentage must be one of {ROLLOUT_STAGES}")
    if minimum_stage_days < 1:
        raise ValueError("minimum_stage_days must be positive")
    if state_ttl_hours < 1 or state_ttl_hours > 48:
        raise ValueError("state_ttl_hours must be between 1 and 48")
    current = _utc(now or datetime.now(UTC))
    bundle_expiry = _timestamp(bundle.get("expires_at"))
    roster_expiry = _timestamp(roster.get("expires_at"))
    approval_gates = bundle.get("approval_gates") or {}
    shadow_gate = shadow.get("deployment_gate") or {}
    shadow_checks = shadow_gate.get("checks") or {}
    checks = {
        "bundle_approved": bundle.get("deployment_status") == "approved",
        "bundle_approval_gate_passed": bundle.get("approval_gate_passed") is True,
        "bundle_all_approval_gates_passed": bool(approval_gates)
        and all(value is True for value in approval_gates.values()),
        "bundle_not_expired": bundle_expiry is not None and bundle_expiry > current,
        "roster_review_confirmed": roster.get("review_confirmed") is True,
        "roster_not_expired": roster_expiry is not None and roster_expiry > current,
        "shadow_gate_passed": shadow_gate.get("passed") is True,
        "shadow_required_checks_passed": REQUIRED_SHADOW_CHECKS.issubset(shadow_checks)
        and all(shadow_checks.get(name) is True for name in REQUIRED_SHADOW_CHECKS),
        "previous_state_matches_bundle": previous_state is None
        or previous_state.get("bundle_sha256") == bundle_sha256,
        "rollout_transition_valid": _valid_transition(
            requested_rollout_percentage,
            previous_state,
            current,
            minimum_stage_days,
        ),
    }
    healthy = all(checks.values())
    effective = requested_rollout_percentage if healthy else 0
    previous_percentage = _previous_percentage(previous_state)
    entered_at = current
    if healthy and previous_state and previous_percentage == requested_rollout_percentage:
        entered_at = _timestamp(previous_state.get("stage_entered_at")) or current
    return {
        "schema_version": OPERATIONAL_STATE_SCHEMA_VERSION,
        "artifact_type": "assignee_operational_guard_state",
        "created_at": current.isoformat(),
        "expires_at": (current + timedelta(hours=state_ttl_hours)).isoformat(),
        "bundle_sha256": bundle_sha256,
        "requested_rollout_percentage": requested_rollout_percentage,
        "effective_rollout_percentage": effective,
        "auto_assignment_enabled": healthy and effective > 0,
        "kill_switch_active": not healthy or effective == 0,
        "stage_entered_at": entered_at.isoformat(),
        "minimum_stage_days": minimum_stage_days,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "action": "progressive_auto_assignment" if healthy and effective > 0 else "manual_only",
    }


def evaluate_ticket_guard(
    state: dict[str, Any],
    *,
    bundle_sha256: str,
    ticket_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _utc(now or datetime.now(UTC))
    if state.get("schema_version") != OPERATIONAL_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported assignee operational state schema")
    if state.get("artifact_type") != "assignee_operational_guard_state":
        raise ValueError("unsupported assignee operational state type")
    if state.get("bundle_sha256") != bundle_sha256:
        raise ValueError("operational state does not match the deployment bundle")
    expires_at = _timestamp(state.get("expires_at"))
    if expires_at is None or expires_at <= current:
        raise ValueError("assignee operational state has expired")
    percentage = state.get("effective_rollout_percentage")
    if not isinstance(percentage, int) or percentage not in ROLLOUT_STAGES:
        raise ValueError("operational state has an invalid rollout percentage")
    checks = state.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("operational state is missing safety checks")
    bucket = rollout_bucket(ticket_id)
    healthy = (
        state.get("auto_assignment_enabled") is True
        and state.get("kill_switch_active") is False
        and all(value is True for value in checks.values())
    )
    selected = healthy and bucket < percentage * 100
    if not healthy:
        reason = "operational_kill_switch_active"
    elif not selected:
        reason = "progressive_rollout_holdback"
    else:
        reason = "progressive_rollout_selected"
    return {
        "auto_assignment_authorized": selected,
        "reason": reason,
        "rollout_bucket": bucket,
        "effective_rollout_percentage": percentage,
        "state_expires_at": expires_at.isoformat(),
    }


def rollout_bucket(ticket_id: str) -> int:
    normalized = str(ticket_id or "").strip()
    if not normalized:
        raise ValueError("ticket_id is required for deterministic rollout")
    return int(hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8], 16) % 10_000


def _valid_transition(
    requested: int,
    previous: dict[str, Any] | None,
    current: datetime,
    minimum_stage_days: int,
) -> bool:
    if requested == 0:
        return True
    if previous is None:
        return requested == ROLLOUT_STAGES[1]
    previous_percentage = _previous_percentage(previous)
    if previous_percentage not in ROLLOUT_STAGES:
        return False
    previous_entered = _timestamp(previous.get("stage_entered_at"))
    previous_expires = _timestamp(previous.get("expires_at"))
    old_state_healthy = (
        previous.get("auto_assignment_enabled") is True
        and previous.get("kill_switch_active") is False
        and isinstance(previous.get("checks"), dict)
        and bool(previous["checks"])
        and all(value is True for value in previous["checks"].values())
        and previous_expires is not None
        and previous_expires > current
    )
    if requested == previous_percentage:
        return old_state_healthy
    if not old_state_healthy or previous_entered is None:
        return False
    next_index = ROLLOUT_STAGES.index(previous_percentage) + 1
    if next_index >= len(ROLLOUT_STAGES) or requested != ROLLOUT_STAGES[next_index]:
        return False
    return current - previous_entered >= timedelta(days=minimum_stage_days)


def _previous_percentage(state: dict[str, Any] | None) -> int:
    if not state:
        return 0
    value = state.get("effective_rollout_percentage")
    return value if isinstance(value, int) else -1


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SELECT_TOP5_CANDIDATE = "select_top5_candidate"
ASSIGN_TOP1 = "assign_top1"
KEEP_MANUAL_TRIAGE = "manual_triage"
SUPPORTED_TOP5_ACTIONS = {
    SELECT_TOP5_CANDIDATE,
    ASSIGN_TOP1,
    KEEP_MANUAL_TRIAGE,
}
MANUAL_TRIAGE = "manual_triage"


def resolve_top5_assist_selection(
    recommendation: dict[str, Any],
    *,
    action: str,
    selected_candidate_rank: int | None = None,
    active_assignees: set[str] | None = None,
    eligible_assignees: set[str] | None = None,
    reviewer: str = "",
    decided_at: str | None = None,
) -> dict[str, Any]:
    """Turn an explicit user choice into a fail-closed assignment decision.

    ``assign_top1`` is a user-authorized shortcut.  It is deliberately not
    classified as autonomous assignment, because a person still has to opt in.
    """
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in SUPPORTED_TOP5_ACTIONS:
        raise ValueError(
            "action must be select_top5_candidate, assign_top1, or manual_triage"
        )

    if normalized_action == KEEP_MANUAL_TRIAGE:
        return _decision(
            recommendation,
            assignee=MANUAL_TRIAGE,
            action=normalized_action,
            selection_mode="manual_triage",
            selected_rank=None,
            assignment_authorized=False,
            reason="user_kept_manual_triage",
            reviewer=reviewer,
            decided_at=decided_at,
        )

    candidates = _validated_candidates(
        recommendation, active_assignees, eligible_assignees
    )
    if normalized_action == ASSIGN_TOP1:
        if recommendation.get("auto_top1_available") is not True:
            raise ValueError("assign_top1 is not available for this recommendation")
        if selected_candidate_rank not in (None, 1):
            raise ValueError("assign_top1 always assigns candidate rank 1")
        return _decision(
            recommendation,
            assignee=candidates[0],
            action=normalized_action,
            selection_mode="top1_user_authorized",
            selected_rank=1,
            assignment_authorized=True,
            reason="user_explicitly_authorized_top1",
            reviewer=reviewer,
            decided_at=decided_at,
        )

    if isinstance(selected_candidate_rank, bool) or not isinstance(
        selected_candidate_rank, int
    ):
        raise ValueError("selected_candidate_rank is required for Top-5 selection")
    if not 1 <= selected_candidate_rank <= 5:
        raise ValueError("selected_candidate_rank must be between 1 and 5")
    return _decision(
        recommendation,
        assignee=candidates[selected_candidate_rank - 1],
        action=normalized_action,
        selection_mode="top5_candidate_selection",
        selected_rank=selected_candidate_rank,
        assignment_authorized=True,
        reason="user_selected_top5_candidate",
        reviewer=reviewer,
        decided_at=decided_at,
    )


def _validated_candidates(
    recommendation: dict[str, Any],
    active_assignees: set[str] | None,
    eligible_assignees: set[str] | None,
) -> list[str]:
    if recommendation.get("top5_assist_available") is not True:
        raise ValueError("Top-5 assist is not available")
    if recommendation.get("requires_user_selection") is not True:
        raise ValueError("recommendation does not require an explicit user choice")
    candidates = [
        _normalize(owner)
        for owner in (
            recommendation.get("top5_candidates")
            or recommendation.get("ranked_candidates")
            or []
        )[:5]
        if _normalize(owner)
    ]
    if len(candidates) != 5 or len(set(candidates)) != 5:
        raise ValueError("recommendation must contain five unique candidates")
    if active_assignees is not None:
        active = {_normalize(owner) for owner in active_assignees if _normalize(owner)}
        if not all(owner in active for owner in candidates):
            raise ValueError("Top-5 contains an inactive or unreviewed owner")
    if recommendation.get("eligibility_required") is True:
        if recommendation.get("eligibility_review_confirmed") is not True:
            raise ValueError("reviewed eligibility is not confirmed")
        if eligible_assignees is None:
            raise ValueError("current eligible assignee set is required")
        eligible = {
            _normalize(owner) for owner in eligible_assignees if _normalize(owner)
        }
        if not all(owner in eligible for owner in candidates):
            raise ValueError("Top-5 contains an owner outside the reviewed eligibility set")
    return candidates


def _decision(
    recommendation: dict[str, Any],
    *,
    assignee: str,
    action: str,
    selection_mode: str,
    selected_rank: int | None,
    assignment_authorized: bool,
    reason: str,
    reviewer: str,
    decided_at: str | None,
) -> dict[str, Any]:
    candidates = [
        _normalize(owner)
        for owner in (
            recommendation.get("top5_candidates")
            or recommendation.get("ranked_candidates")
            or []
        )[:5]
        if _normalize(owner)
    ]
    normalized_assignee = _normalize(assignee)
    auto_top1_selected = action == ASSIGN_TOP1
    return {
        "schema_version": 1,
        "event_type": "assignee_selection_decision",
        "assignee": normalized_assignee,
        "final_assignee": normalized_assignee,
        "suggested_assignee": candidates[0] if candidates else "",
        "ranked_candidates": candidates,
        "top5_candidates": candidates,
        "top5_assist_available": recommendation.get("top5_assist_available") is True,
        "selection_mode": selection_mode,
        "user_action": action,
        "requires_user_selection": False,
        "user_authorized_assignment": assignment_authorized,
        "assignment_authorized": assignment_authorized,
        "auto_top1_selected": auto_top1_selected,
        "auto_top1_authorized": auto_top1_selected and assignment_authorized,
        "autonomous_assignment": False,
        "auto_assignment_authorized": False,
        "selected_candidate_rank": selected_rank,
        "selected_from_top5": selected_rank is not None and 1 <= selected_rank <= 5,
        "routing_status": (
            "top1_user_authorized_assignment"
            if auto_top1_selected
            else "top5_candidate_selected"
            if assignment_authorized
            else "manual_triage"
        ),
        "decision_reason_code": reason,
        "fallback_reason": "" if assignment_authorized else reason,
        "quality_target": recommendation.get("quality_target"),
        "quality_scope": recommendation.get("quality_scope", "top5_candidate_set"),
        "correctness_field": recommendation.get("correctness_field"),
        "eligibility_required": recommendation.get("eligibility_required") is True,
        "eligibility_review_confirmed": recommendation.get(
            "eligibility_review_confirmed"
        )
        is True,
        "eligibility_verified": recommendation.get("eligibility_verified") is True,
        "eligible_top5_count": recommendation.get("eligible_top5_count"),
        "top1_quality_guaranteed_by_top5_policy": False,
        "model_version": str(recommendation.get("model_version") or ""),
        "policy_version": str(recommendation.get("policy_version") or ""),
        "confidence": recommendation.get("confidence"),
        "open_set_risk": recommendation.get("open_set_risk"),
        "prediction_latency_ms": recommendation.get("prediction_latency_ms"),
        "reviewer": str(reviewer or "").strip(),
        "decision_created_at": decided_at
        or datetime.now(timezone.utc).isoformat(),
    }


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()

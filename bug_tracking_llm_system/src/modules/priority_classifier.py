from __future__ import annotations

from statistics import mean
from typing import Any

from utils.json_schema import canonical_priority


PRIORITY_TO_SCORE = {"P1": 1.0, "P2": 0.75, "P3": 0.50, "P4": 0.25, "P5": 0.05}


class PriorityClassifier:
    """DRONE/GRAY-style multi-factor priority baseline."""

    def predict(self, ticket_json: dict[str, Any], duplicate_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        existing_priority = canonical_priority(ticket_json.get("priority"))
        factor_scores = {
            "textual_factor": self._textual_factor(ticket_json),
            "temporal_factor": self._temporal_factor(ticket_json),
            "author_factor": self._author_factor(ticket_json),
            "related_report_factor": self._related_report_factor(duplicate_candidates),
            "severity_factor": self._severity_factor(ticket_json),
            "component_factor": self._component_factor(ticket_json),
        }
        if existing_priority:
            predicted = existing_priority
            confidence = 0.92
        else:
            urgency = (
                0.24 * factor_scores["textual_factor"]
                + 0.10 * factor_scores["temporal_factor"]
                + 0.08 * factor_scores["author_factor"]
                + 0.18 * factor_scores["related_report_factor"]
                + 0.28 * factor_scores["severity_factor"]
                + 0.12 * factor_scores["component_factor"]
            )
            predicted = _score_to_priority(urgency)
            confidence = max(0.35, min(0.88, 0.45 + abs(urgency - 0.50)))

        return {
            "predicted_priority": predicted,
            "confidence": round(float(confidence), 4),
            "factor_scores": {key: round(float(value), 4) for key, value in factor_scores.items()},
            "model": "deterministic_drone_gray_baseline",
        }

    def _textual_factor(self, ticket: dict[str, Any]) -> float:
        text = _ticket_text(ticket)
        high = ("data loss", "cannot login", "security", "crash", "panic", "production", "outage")
        medium = ("exception", "traceback", "timeout", "failed", "incorrect", "regression")
        low = ("typo", "cosmetic", "minor", "wording", "style")
        if any(term in text for term in high):
            return 0.88
        if any(term in text for term in medium):
            return 0.65
        if any(term in text for term in low):
            return 0.15
        return 0.45

    def _temporal_factor(self, ticket: dict[str, Any]) -> float:
        count = _to_float(ticket.get("recent_similar_count"), 0.0)
        return min(1.0, count / 10.0)

    def _author_factor(self, ticket: dict[str, Any]) -> float:
        count = _to_float(ticket.get("reporter_past_bug_count"), 0.0)
        return min(1.0, count / 20.0)

    def _related_report_factor(self, candidates: list[dict[str, Any]]) -> float:
        scores = [PRIORITY_TO_SCORE[priority] for item in candidates[:5] if (priority := canonical_priority(item.get("priority")))]
        similarities = [_to_float(item.get("similarity"), 0.0) for item in candidates[:5]]
        if scores:
            return min(1.0, 0.75 * mean(scores) + 0.25 * (mean(similarities) if similarities else 0.0))
        return mean(similarities) if similarities else 0.35

    def _severity_factor(self, ticket: dict[str, Any]) -> float:
        severity = str(ticket.get("severity") or "").lower()
        bug_type = str(ticket.get("bug_type") or "").lower()
        if severity in {"blocker", "critical"} or bug_type in {"security", "crash"}:
            return 0.95
        if severity in {"major", "normal"} or bug_type in {"runtime_error", "build_error"}:
            return 0.70
        if severity in {"minor", "trivial", "enhancement"} or bug_type == "ui_bug":
            return 0.25
        return 0.50

    def _component_factor(self, ticket: dict[str, Any]) -> float:
        component = str(ticket.get("component") or "").lower()
        if any(term in component for term in ("security", "auth", "payment", "database", "core")):
            return 0.82
        if any(term in component for term in ("ui", "frontend", "docs")):
            return 0.35
        return 0.50


def _score_to_priority(score: float) -> str:
    if score >= 0.85:
        return "P1"
    if score >= 0.65:
        return "P2"
    if score >= 0.40:
        return "P3"
    if score >= 0.20:
        return "P4"
    return "P5"


def _ticket_text(ticket: dict[str, Any]) -> str:
    return " ".join(str(ticket.get(key) or "") for key in ("title", "description", "error_message", "logs")).lower()


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PipelineConfig
from modules.assignee_feedback import feedback_rows_to_history_rows, read_assignee_feedback


MANUAL_TRIAGE = "manual_triage"
CALIBRATION_ARTIFACT_SCHEMA_VERSION = 1
OPEN_SET_ARTIFACT_SCHEMA_VERSION = 1
RANKER_CONFIDENCE_VERSION = "hybrid_ranker_v1"
STRONG_OWNER_SIGNALS = {
    "component_history",
    "product_component_history",
    "component_owner_mapping",
    "component_ownership",
    "file_ownership",
}
AUTHORITATIVE_OWNER_SIGNALS = {
    "component_owner_mapping",
    "component_ownership",
    "file_ownership",
}
GENERIC_ASSIGNEE_RE = re.compile(
    r"(^|[^a-z0-9])(nobody|unassigned|default|disabled|do-not-reply|noreply|bugzilla|wptsync)([^a-z0-9]|$)"
)
PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z]:)?(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/)+"
    r"[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|java|kt|go|rs|rb|php|c|cc|cpp|h|hpp|cs|swift|m|mm|css|scss|html|xml|json|yaml|yml|sql)"
)


class AssigneeTriager:
    """Assign a likely owner using historical, metadata, and text signals."""

    def __init__(self, *, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._cached_history_path: Path | None = None
        self._cached_history_mtime: float | None = None
        self._cached_history_as_of: str = ""
        self._cached_history_latest_timestamp: datetime | None = None
        self._cached_profile_source_signature: tuple[tuple[str, float | None], ...] | None = None
        self._cached_profile: dict[str, Any] | None = None
        self._cached_assignee_roster_signature: tuple[tuple[str, float | None], ...] | None = None
        self._cached_assignee_roster: dict[str, Any] | None = None
        self._cached_ownership_signature: tuple[tuple[str, float | None], ...] | None = None
        self._cached_ownership_profile: dict[str, dict[str, list[str]]] | None = None

    def assign(self, ticket_json: dict[str, Any], priority_result: dict[str, Any]) -> dict[str, Any]:
        component = _normalize_component(ticket_json.get("component"))
        product = _normalize_component(ticket_json.get("product"))
        profile = self._history_profile(as_of=_ticket_timestamp(ticket_json))
        top_k = max(1, int(self.config.assignee_top_k))
        ranked = self._rank_candidates(ticket_json, profile)
        mapped_owner = _normalize_assignee(self._mapped_owner(component))
        product_component = f"{product}::{component}"
        roster = self._assignee_roster()
        profile_stats = {
            "history_rows": len(profile["documents"]),
            "feedback_history_rows": int(profile.get("feedback_history_rows", 0)),
            "future_history_rows_filtered": int(profile.get("future_history_rows_filtered", 0)),
            "history_rows_without_timestamp": int(profile.get("history_rows_without_timestamp", 0)),
            "candidate_count": len(profile["global_counts"]),
            "active_roster_size": len(roster["active"]),
            "roster_status": roster["status"],
            "component_history_rows": sum(profile["component_counts"].get(component, Counter()).values()),
            "product_history_rows": sum(profile["product_counts"].get(product, Counter()).values()),
            "product_component_history_rows": sum(
                profile["product_component_counts"].get(product_component, Counter()).values()
            ),
        }

        if ranked:
            top_candidate = ranked[0]
            raw_confidence = self._confidence(ranked)
            calibration = self._calibration_assessment(raw_confidence, applicable=True)
            confidence = calibration["confidence"]
            profile_stats["top_candidate_train_frequency"] = int(profile["global_counts"].get(top_candidate["assignee"], 0))
            profile_stats["top_candidate_component_history_rows"] = int(
                profile["component_counts"].get(component, Counter()).get(top_candidate["assignee"], 0)
            )
            profile_stats["top_candidate_product_component_history_rows"] = int(
                profile["product_component_counts"].get(product_component, Counter()).get(top_candidate["assignee"], 0)
            )
            open_set = self._open_set_assessment(ticket_json, top_candidate, profile_stats, confidence)
            fallback_reason = self._fallback_reason_for_candidate(
                top_candidate,
                confidence,
                profile_stats,
                open_set["risk"],
                open_set["threshold"],
                calibration["status"],
            )
            if not fallback_reason:
                assignment_fallback_reason = self._assignment_fallback_reason(top_candidate, profile_stats)
                return self._response(
                    assignee=top_candidate["assignee"],
                    confidence=confidence,
                    raw_confidence=raw_confidence,
                    calibration=calibration,
                    reason=self._reason(component, priority_result, top_candidate),
                    ranked=ranked,
                    top_k=top_k,
                    fallback_used=bool(assignment_fallback_reason),
                    fallback_reason=assignment_fallback_reason,
                    routing_status="assigned_by_fallback" if assignment_fallback_reason else "assigned",
                    needs_manual_triage=False,
                    suggested_assignee=top_candidate["assignee"],
                    profile_stats=profile_stats,
                    open_set=open_set,
                )

            return self._response(
                assignee=MANUAL_TRIAGE,
                confidence=confidence,
                raw_confidence=raw_confidence,
                calibration=calibration,
                reason=self._manual_reason(component, priority_result, top_candidate, fallback_reason),
                ranked=ranked,
                top_k=top_k,
                fallback_used=True,
                fallback_reason=fallback_reason,
                routing_status="needs_manual_triage",
                needs_manual_triage=True,
                suggested_assignee=top_candidate["assignee"],
                profile_stats=profile_stats,
                open_set=open_set,
            )

        if self._is_assignable_assignee(mapped_owner):
            confidence = 0.65
            calibration = self._calibration_assessment(confidence, applicable=False)
            priority = priority_result.get("predicted_priority", "P3")
            reason = f"Assigned from component-owner mapping for component '{component}' and priority {priority}."
            ranked = [{"assignee": mapped_owner, "score": 4.25, "signals": ["component_owner_mapping"]}]
            return self._response(
                assignee=mapped_owner,
                confidence=confidence,
                raw_confidence=confidence,
                calibration=calibration,
                reason=reason,
                ranked=ranked,
                top_k=top_k,
                fallback_used=True,
                fallback_reason="component_owner_mapping_only",
                routing_status="assigned_by_fallback",
                needs_manual_triage=False,
                suggested_assignee=mapped_owner,
                profile_stats=profile_stats,
                open_set=self._no_candidate_open_set_assessment(0.0),
            )

        confidence = 0.25
        calibration = self._calibration_assessment(confidence, applicable=False)
        reason = "No reliable owner signal was found; manual triage is required."
        if roster["enforce_active"] and not roster["active"]:
            fallback_reason = "active_roster_unavailable"
        else:
            fallback_reason = "missing_assignee_history" if not profile["documents"] else "no_reliable_owner_signal"
        return self._response(
            assignee=MANUAL_TRIAGE,
            confidence=confidence,
            raw_confidence=confidence,
            calibration=calibration,
            reason=reason,
            ranked=[],
            top_k=top_k,
            fallback_used=True,
            fallback_reason=fallback_reason,
            routing_status="needs_manual_triage",
            needs_manual_triage=True,
            suggested_assignee="",
            profile_stats=profile_stats,
            open_set=self._no_candidate_open_set_assessment(1.0),
        )

    def _response(
        self,
        *,
        assignee: str,
        confidence: float,
        raw_confidence: float,
        calibration: dict[str, Any],
        reason: str,
        ranked: list[dict[str, Any]],
        top_k: int,
        fallback_used: bool,
        fallback_reason: str,
        routing_status: str,
        needs_manual_triage: bool,
        suggested_assignee: str,
        profile_stats: dict[str, Any],
        open_set: dict[str, Any],
    ) -> dict[str, Any]:
        ranked_candidates = [candidate["assignee"] for candidate in ranked[:top_k]]
        if assignee not in ranked_candidates and assignee != MANUAL_TRIAGE:
            ranked_candidates.insert(0, assignee)
        if not ranked_candidates:
            ranked_candidates = [MANUAL_TRIAGE]

        candidate_scores = {candidate["assignee"]: round(candidate["score"], 6) for candidate in ranked[:top_k]}
        candidate_details = [
            {
                "assignee": candidate["assignee"],
                "score": round(float(candidate["score"]), 6),
                "signals": list(candidate.get("signals", [])),
            }
            for candidate in ranked[:top_k]
        ]

        routing_policy = self._routing_policy()
        result = {
            "assignee": assignee,
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "calibration_status": calibration["status"],
            "calibration_artifact": calibration["artifact"],
            "reason": reason,
            "ranked_candidates": ranked_candidates[:top_k],
            "candidate_scores": candidate_scores,
            "candidate_details": candidate_details,
            "suggested_assignee": suggested_assignee,
            "routing_status": routing_status,
            "needs_manual_triage": needs_manual_triage,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "profile_stats": profile_stats,
        }
        if routing_policy:
            result["routing_policy"] = routing_policy.get("name", "")
            result["routing_policy_thresholds"] = {
                "t_high": routing_policy.get("t_high"),
                "t_low": routing_policy.get("t_low"),
            }
        else:
            result["routing_policy"] = ""
            result["routing_policy_thresholds"] = {}
        if open_set["risk"] is not None:
            result["open_set_risk"] = round(float(open_set["risk"]), 6)
        result["open_set_status"] = open_set["status"]
        result["open_set_detector"] = open_set["detector"]
        if open_set["threshold"] is not None:
            result["open_set_threshold"] = round(float(open_set["threshold"]), 6)
        return result

    def _fallback_reason_for_candidate(
        self,
        candidate: dict[str, Any],
        confidence: float,
        profile_stats: dict[str, Any],
        open_set_risk: float | None,
        open_set_threshold: float | None,
        calibration_status: str,
    ) -> str:
        if not self._is_assignable_assignee(candidate.get("assignee")):
            return "invalid_or_inactive_candidate"
        if candidate.get("score", 0.0) < self.config.assignee_min_score:
            return "weak_candidate_score"
        policy_reason = self._routing_policy_fallback_reason(confidence)
        if policy_reason:
            return policy_reason
        if open_set_risk is not None and open_set_threshold is not None and open_set_risk >= open_set_threshold:
            return "open_set_unknown_risk"
        if confidence < self.config.assignee_confidence_threshold:
            return "low_confidence"
        signals = set(candidate.get("signals", []))
        if not self.config.assignee_allow_text_only_assignment and not (signals & STRONG_OWNER_SIGNALS):
            return "text_only_or_broad_history_only"
        if (
            not self.config.assignee_allow_uncalibrated_auto_assignment
            and calibration_status != "applied"
            and not (signals & AUTHORITATIVE_OWNER_SIGNALS)
        ):
            return "uncalibrated_recommendation_only"
        return ""

    def _assignment_fallback_reason(self, candidate: dict[str, Any], profile_stats: dict[str, Any]) -> str:
        signals = set(candidate.get("signals", []))
        if signals == {"component_owner_mapping"} and profile_stats.get("component_history_rows", 0) == 0:
            return "component_owner_mapping_only"
        return ""

    def _manual_reason(
        self,
        component: str,
        priority_result: dict[str, Any],
        candidate: dict[str, Any],
        fallback_reason: str,
    ) -> str:
        priority = priority_result.get("predicted_priority", "P3")
        signals = ", ".join(candidate.get("signals", [])) or "none"
        return (
            f"Manual triage required for component '{component}' and priority {priority}; "
            f"top suggestion is '{candidate['assignee']}' but routing fallback triggered: {fallback_reason}. "
            f"candidate signals: {signals}."
        )

    def _routing_policy_fallback_reason(self, confidence: float) -> str:
        policy = self._routing_policy()
        if not policy:
            return ""

        high = policy.get("t_high")
        low = policy.get("t_low")
        if high is not None and confidence >= high:
            return ""
        if low is not None and confidence >= low:
            return "top_k_confirmation_required"
        return "calibrated_low_confidence"

    def _routing_policy(self) -> dict[str, Any]:
        path = self.config.assignee_routing_policy_path
        if path is None or not Path(path).exists():
            return {}

        path = Path(path)
        policy_name = self.config.assignee_routing_policy_name.strip()
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            selected = None
            for row in rows:
                row_name = str(row.get("policy") or row.get("routing_policy") or row.get("name") or "").strip()
                if policy_name and row_name != policy_name:
                    continue
                selected = row
                break
            if selected is None:
                return {}
            name = str(selected.get("policy") or selected.get("routing_policy") or selected.get("name") or policy_name or path.stem)
            return _validated_routing_policy({
                "name": name,
                "deployment_status": selected.get("deployment_status"),
                "t_high": _first_float(selected, ("t_high", "high_confidence_threshold", "auto_assign_threshold")),
                "t_low": _first_float(selected, ("t_low", "low_confidence_threshold", "top_k_threshold")),
            })

        data = _load_structured_file(path)
        selected_policy = _select_policy_object(data, policy_name)
        if not selected_policy:
            return {}
        return _validated_routing_policy({
            "name": str(selected_policy.get("policy") or selected_policy.get("name") or policy_name or path.stem),
            "deployment_status": selected_policy.get("deployment_status"),
            "t_high": _first_float(selected_policy, ("t_high", "high_confidence_threshold", "auto_assign_threshold")),
            "t_low": _first_float(selected_policy, ("t_low", "low_confidence_threshold", "top_k_threshold")),
        })

    def _calibration_assessment(self, raw_confidence: float, *, applicable: bool) -> dict[str, Any]:
        raw = _clip_probability(raw_confidence)
        path = self.config.assignee_calibration_artifact_path
        if not applicable:
            return {"confidence": raw, "status": "not_applicable", "artifact": ""}
        if path is None:
            return {"confidence": raw, "status": "not_configured", "artifact": ""}
        artifact = _load_artifact(Path(path))
        if not artifact:
            return {"confidence": raw, "status": "artifact_invalid_or_missing", "artifact": ""}
        if artifact.get("artifact_type") != "assignee_confidence_calibrator":
            return {"confidence": raw, "status": "artifact_type_mismatch", "artifact": ""}
        if artifact.get("schema_version") != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            return {"confidence": raw, "status": "artifact_schema_mismatch", "artifact": ""}
        if artifact.get("deployment_status") != "approved":
            return {"confidence": raw, "status": "artifact_not_approved", "artifact": str(artifact.get("name") or "")}
        if artifact.get("ranker_confidence_version") != RANKER_CONFIDENCE_VERSION:
            return {"confidence": raw, "status": "ranker_confidence_version_mismatch", "artifact": str(artifact.get("name") or "")}
        if artifact.get("name") != "isotonic_regression":
            return {"confidence": raw, "status": "unsupported_calibrator", "artifact": str(artifact.get("name") or "")}

        mapping = artifact.get("mapping")
        if not isinstance(mapping, dict):
            return {"confidence": raw, "status": "artifact_mapping_invalid", "artifact": "isotonic_regression"}
        x_values = _float_list(mapping.get("x_thresholds"))
        y_values = _float_list(mapping.get("y_thresholds"))
        if (
            len(x_values) < 2
            or len(x_values) != len(y_values)
            or x_values != sorted(x_values)
            or y_values != sorted(y_values)
            or any(value < 0.0 or value > 1.0 for value in [*x_values, *y_values])
        ):
            return {"confidence": raw, "status": "artifact_mapping_invalid", "artifact": "isotonic_regression"}
        calibrated = _piecewise_linear(raw, x_values, y_values)
        return {
            "confidence": round(_clip_probability(calibrated), 6),
            "status": "applied",
            "artifact": "isotonic_regression",
        }

    def _no_candidate_open_set_assessment(self, risk: float) -> dict[str, Any]:
        if not self.config.assignee_open_set_enabled:
            return {"risk": None, "status": "disabled", "detector": "", "threshold": None}
        return {
            "risk": _clip_probability(risk),
            "status": "no_ranked_candidate",
            "detector": "fallback_rule_based_v1",
            "threshold": _clip_probability(self.config.assignee_open_set_risk_threshold),
        }

    def _open_set_assessment(
        self,
        ticket_json: dict[str, Any],
        candidate: dict[str, Any],
        profile_stats: dict[str, Any],
        confidence: float,
    ) -> dict[str, Any]:
        if not self.config.assignee_open_set_enabled:
            return {"risk": None, "status": "disabled", "detector": "", "threshold": None}

        fallback_risk = self._open_set_risk(candidate, profile_stats, confidence)
        path = self.config.assignee_open_set_artifact_path
        if path is None:
            return {
                "risk": fallback_risk,
                "status": "fallback_rule_based_v1",
                "detector": "fallback_rule_based_v1",
                "threshold": _clip_probability(self.config.assignee_open_set_risk_threshold),
            }
        artifact = _load_artifact(Path(path))
        artifact_status = self._validate_open_set_artifact(artifact)
        if artifact_status:
            return {
                "risk": fallback_risk,
                "status": artifact_status,
                "detector": "fallback_rule_based_v1",
                "threshold": _clip_probability(self.config.assignee_open_set_risk_threshold),
            }

        assert artifact is not None
        artifact_threshold = _safe_float(artifact.get("threshold"))
        return {
            "risk": self._rule_based_novelty_risk(ticket_json, candidate, profile_stats, artifact),
            "status": "applied",
            "detector": "rule_based_novelty",
            "threshold": _clip_probability(
                self.config.assignee_open_set_risk_threshold if artifact_threshold is None else artifact_threshold
            ),
        }

    def _validate_open_set_artifact(self, artifact: dict[str, Any] | None) -> str:
        if not artifact:
            return "artifact_invalid_or_missing"
        if artifact.get("artifact_type") != "assignee_open_set_detector":
            return "artifact_type_mismatch"
        if artifact.get("schema_version") != OPEN_SET_ARTIFACT_SCHEMA_VERSION:
            return "artifact_schema_mismatch"
        if artifact.get("deployment_status") != "approved":
            return "artifact_not_approved"
        if artifact.get("ranker_confidence_version") != RANKER_CONFIDENCE_VERSION:
            return "ranker_confidence_version_mismatch"
        if artifact.get("name") != "rule_based_novelty":
            return "unsupported_open_set_detector"
        if not isinstance(artifact.get("parameters"), dict):
            return "artifact_parameters_invalid"
        threshold = _safe_float(artifact.get("threshold"))
        if threshold is None or not 0.0 <= threshold <= 1.0:
            return "artifact_threshold_invalid"
        return ""

    def _rule_based_novelty_risk(
        self,
        ticket_json: dict[str, Any],
        candidate: dict[str, Any],
        profile_stats: dict[str, Any],
        artifact: dict[str, Any],
    ) -> float:
        parameters = artifact.get("parameters", {})
        weights = parameters.get("weights", {}) if isinstance(parameters.get("weights"), dict) else {}
        component_rows = int(profile_stats.get("component_history_rows", 0))
        product_component_rows = int(profile_stats.get("product_component_history_rows", 0))
        rare_component_max_rows = max(1, int(_safe_float(parameters.get("rare_component_max_rows")) or 5))
        rare_pair_max_rows = max(1, int(_safe_float(parameters.get("rare_product_component_max_rows")) or 5))
        signals = set(candidate.get("signals", []))
        candidate_owner = _normalize_assignee(candidate.get("assignee"))
        disagreement_count = 0
        component = _normalize_component(ticket_json.get("component"))
        product = _normalize_component(ticket_json.get("product"))
        profile = self._history_profile(as_of=_ticket_timestamp(ticket_json))
        component_top = _counter_top_owner(profile["component_counts"].get(component, Counter()))
        pair_top = _counter_top_owner(profile["product_component_counts"].get(f"{product}::{component}", Counter()))
        if component_top and component_top != candidate_owner:
            disagreement_count += 1
        if pair_top and pair_top != candidate_owner:
            disagreement_count += 1

        score = 0.0
        score += _artifact_weight(weights, "unseen_component", 0.25) * int(component_rows == 0)
        score += _artifact_weight(weights, "unseen_product_component_pair", 0.25) * int(product_component_rows == 0)
        score += _artifact_weight(weights, "rare_component", 0.15) * int(0 < component_rows < rare_component_max_rows)
        score += _artifact_weight(weights, "rare_product_component_pair", 0.15) * int(
            0 < product_component_rows < rare_pair_max_rows
        )
        score += _artifact_weight(weights, "no_component_history", 0.10) * int(
            profile_stats.get("top_candidate_component_history_rows", 0) == 0
        )
        score += _artifact_weight(weights, "no_text_similarity", 0.05) * int("text_similarity" not in signals)
        score += _artifact_weight(weights, "disagreement", 0.05) * min(1.0, disagreement_count / 3.0)
        if signals & {"component_owner_mapping", "component_ownership", "file_ownership"}:
            score -= max(0.0, _safe_float(parameters.get("ownership_risk_reduction")) or 0.0)
        return round(_clip_probability(score), 6)

    def _open_set_risk(self, candidate: dict[str, Any], profile_stats: dict[str, Any], confidence: float) -> float:
        signals = set(candidate.get("signals", []))
        risk = 0.0
        if profile_stats.get("history_rows", 0) == 0:
            risk += 0.25
        if profile_stats.get("component_history_rows", 0) == 0:
            risk += 0.25
        if profile_stats.get("top_candidate_train_frequency", 0) == 0:
            risk += 0.20
        if not (signals & STRONG_OWNER_SIGNALS):
            risk += 0.20
        if signals == {"text_similarity"}:
            risk += 0.15
        if confidence < self.config.assignee_confidence_threshold:
            risk += 0.10
        if signals & {"component_owner_mapping", "component_ownership", "file_ownership"}:
            risk -= 0.20
        return min(1.0, max(0.0, round(risk, 6)))

    def _is_assignable_assignee(self, value: Any) -> bool:
        assignee = _normalize_assignee(value)
        if not _is_valid_assignee(assignee):
            return False
        roster = self._assignee_roster()
        if assignee in roster["inactive"]:
            return False
        if roster["enforce_active"] and assignee not in roster["active"]:
            return False
        return True

    def _assignee_roster(self) -> dict[str, Any]:
        signature = _active_roster_signature(self.config)
        if self._cached_assignee_roster is not None and self._cached_assignee_roster_signature == signature:
            return self._cached_assignee_roster

        active: set[str] = set()
        inactive: set[str] = set()
        enforce_active = False
        status = "not_configured"

        roster_path = self.config.assignee_active_roster_path
        if roster_path is not None:
            enforce_active = True
            path = Path(roster_path)
            if not path.exists() or not path.is_file():
                status = "active_roster_missing"
            else:
                data = _load_structured_file(path)
                if isinstance(data, list):
                    list_active, list_inactive = _split_roster_items(data)
                    active.update(list_active)
                    inactive.update(list_inactive)
                elif isinstance(data, dict):
                    active.update(_coerce_assignee_set(data.get("active", [])))
                    active.update(_coerce_assignee_set(data.get("candidates", [])))
                    active.update(_coerce_assignee_set(data.get("candidate_roster", [])))
                    active.update(_coerce_assignee_set(data.get("roster", [])))
                    active.update(_coerce_assignee_set(data.get("owners", [])))
                    inactive.update(_coerce_assignee_set(data.get("inactive", [])))
                    inactive.update(_coerce_assignee_set(data.get("departed", [])))
                    inactive.update(_coerce_assignee_set(data.get("disabled", [])))
                    list_active, list_inactive = _split_roster_items(data.get("assignees", []))
                    active.update(list_active)
                    inactive.update(list_inactive)
                status = "loaded" if active else "active_roster_empty_or_invalid"

        inactive_path = self.config.assignee_inactive_path
        if inactive_path is not None:
            path = Path(inactive_path)
            if not path.exists() or not path.is_file():
                if roster_path is None:
                    enforce_active = True
                status = "inactive_filter_missing"
            else:
                inactive.update(_coerce_assignee_set(_load_structured_file(path)))
                if roster_path is None:
                    status = "inactive_filter_only"

        active = {owner for owner in active if _is_valid_assignee(owner)}
        inactive = {owner for owner in inactive if _normalize_assignee(owner)}
        active -= inactive
        if enforce_active and not active and status == "loaded":
            status = "active_roster_empty_or_invalid"
        result = {
            "active": active,
            "inactive": inactive,
            "enforce_active": enforce_active,
            "status": status,
        }
        self._cached_assignee_roster_signature = signature
        self._cached_assignee_roster = result
        return result

    def _component_ownership_owners(self, component: str) -> list[str]:
        owners_by_component = self._ownership_profile()["components"]
        owners: list[str] = []
        for key, values in owners_by_component.items():
            if _component_key_matches(key, component):
                owners.extend(values)
        return _unique_values([owner for owner in owners if self._is_assignable_assignee(owner)])

    def _file_ownership_owners(self, ticket_json: dict[str, Any]) -> list[str]:
        owners_by_path = self._ownership_profile()["files"]
        ticket_paths = _extract_file_paths(ticket_json)
        owners: list[str] = []
        for ticket_path in ticket_paths:
            for rule_path, values in owners_by_path.items():
                if _path_rule_matches(rule_path, ticket_path):
                    owners.extend(values)
        return _unique_values([owner for owner in owners if self._is_assignable_assignee(owner)])

    def _ownership_profile(self) -> dict[str, dict[str, list[str]]]:
        signature = _ownership_signature(self.config)
        if self._cached_ownership_profile is not None and self._cached_ownership_signature == signature:
            return self._cached_ownership_profile

        components: dict[str, list[str]] = {}
        files: dict[str, list[str]] = {}

        component_path = self.config.assignee_component_ownership_path
        if component_path is not None and Path(component_path).exists():
            components.update(_load_owner_map(Path(component_path), ("components", "component_owners", "ownership")))

        file_path = self.config.assignee_file_ownership_path
        if file_path is not None and Path(file_path).exists():
            files.update(_load_owner_map(Path(file_path), ("files", "paths", "file_owners", "ownership")))

        profile = {"components": components, "files": files}
        self._cached_ownership_signature = signature
        self._cached_ownership_profile = profile
        return profile

    def _rank_candidates(self, ticket_json: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        component = _normalize_component(ticket_json.get("component"))
        product = _normalize_component(ticket_json.get("product"))
        product_component = f"{product}::{component}"
        scores: defaultdict[str, float] = defaultdict(float)
        signals: dict[str, set[str]] = defaultdict(set)

        def add(assignee: Any, score: float, signal: str) -> None:
            owner = _normalize_assignee(assignee)
            if not self._is_assignable_assignee(owner) or score <= 0:
                return
            scores[owner] += score
            signals[owner].add(signal)

        if product != "unknown":
            for owner, count in profile["product_component_counts"].get(product_component, Counter()).items():
                add(owner, 4.0 * _count_ratio(count, profile["product_component_counts"][product_component]), "product_component_history")
        for owner, count in profile["component_counts"].get(component, Counter()).items():
            add(owner, 3.0 * _count_ratio(count, profile["component_counts"][component]), "component_history")
        if product != "unknown":
            for owner, count in profile["product_counts"].get(product, Counter()).items():
                add(owner, 0.75 * _count_ratio(count, profile["product_counts"][product]), "product_history")

        mapped_owner = self._mapped_owner(component)
        if self._is_assignable_assignee(mapped_owner):
            add(mapped_owner, 4.25, "component_owner_mapping")

        for owner in self._component_ownership_owners(component):
            add(owner, 5.0, "component_ownership")
        for owner in self._file_ownership_owners(ticket_json):
            add(owner, 4.5, "file_ownership")

        for owner, score in self._text_similarity_scores(ticket_json, profile).items():
            add(owner, score, "text_similarity")

        if not scores:
            return []

        ranked = [
            {"assignee": owner, "score": score, "signals": sorted(signals[owner])}
            for owner, score in scores.items()
            if signals[owner]
        ]
        ranked = [candidate for candidate in ranked if self._is_reliable_candidate(component, candidate)]
        ranked.sort(key=lambda item: (-item["score"], item["assignee"]))
        if not ranked:
            return []

        for owner, count in profile["global_counts"].most_common():
            if owner not in {candidate["assignee"] for candidate in ranked}:
                ranked.append(
                    {
                        "assignee": owner,
                        "score": 0.15 * _count_ratio(count, profile["global_counts"]),
                        "signals": ["global_prior"],
                    }
                )
            if len(ranked) >= 10:
                break
        return ranked

    def _is_reliable_candidate(self, component: str, candidate: dict[str, Any]) -> bool:
        signals = set(candidate.get("signals", []))
        if signals & STRONG_OWNER_SIGNALS:
            return True
        if component in {"unknown", "documentation", "docs"}:
            return False
        return "text_similarity" in signals and candidate.get("score", 0.0) >= 1.25

    def _text_similarity_scores(self, ticket_json: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
        query_tokens = Counter(_tokens(_row_text(ticket_json)))
        if not query_tokens or not profile["documents"]:
            return {}

        doc_scores: defaultdict[int, float] = defaultdict(float)
        avgdl = profile["avgdl"] or 1.0
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            idf = profile["idf"].get(token)
            if idf is None:
                continue
            for doc_index, frequency in profile["postings"].get(token, []):
                doc_len = profile["doc_lengths"][doc_index] or 1
                denominator = frequency + k1 * (1 - b + b * doc_len / avgdl)
                doc_scores[doc_index] += idf * ((frequency * (k1 + 1)) / denominator)

        owner_scores: defaultdict[str, float] = defaultdict(float)
        for rank, (doc_index, score) in enumerate(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:10], start=1):
            owner = profile["documents"][doc_index]["assignee"]
            owner_scores[owner] += min(3.0, score) / rank
        return dict(owner_scores)

    def _confidence(self, ranked: list[dict[str, Any]]) -> float:
        if not ranked:
            return 0.25
        best = ranked[0]["score"]
        second = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = (best - second) / best if best else 0.0
        signal_bonus = min(0.20, 0.05 * len(ranked[0].get("signals", [])))
        return round(min(0.95, max(0.35, 0.55 + 0.30 * margin + signal_bonus)), 3)

    def _reason(self, component: str, priority_result: dict[str, Any], candidate: dict[str, Any]) -> str:
        priority = priority_result.get("predicted_priority", "P3")
        signals = ", ".join(candidate.get("signals", []))
        return (
            f"Assigned by hybrid ranking for component '{component}' and priority {priority}; "
            f"strongest signals: {signals}."
        )

    def _mapped_owner(self, component: str) -> str | None:
        mapping = {key.lower(): value for key, value in self.config.component_owner_mapping.items()}
        if component in mapping:
            return mapping[component]
        matches = [(key, value) for key, value in mapping.items() if _component_key_matches(key, component)]
        if matches:
            return max(matches, key=lambda item: _component_specificity(item[0]))[1]
        return mapping.get("unknown")

    def _history_profile(self, *, as_of: str = "") -> dict[str, Any]:
        path = self.config.assignee_dataset_path
        if path is None:
            path = Path("")
        else:
            path = Path(path)
        history_exists = path.exists() and path.is_file()
        if not history_exists and self.config.assignee_feedback_path is None:
            return _empty_profile()

        mtime = path.stat().st_mtime if history_exists else None
        source_signature = _profile_source_signature(self.config)
        cutoff = _parse_timestamp(as_of)
        cache_as_of = as_of
        if self.config.assignee_feedback_path is None and (
            cutoff is None
            or self._cached_history_latest_timestamp is None
            or cutoff >= self._cached_history_latest_timestamp
        ):
            cache_as_of = "__all_history__"
        if (
            self._cached_profile is not None
            and self._cached_history_path == path
            and self._cached_history_mtime == mtime
            and self._cached_profile_source_signature == source_signature
            and self._cached_history_as_of == cache_as_of
        ):
            return self._cached_profile

        source_rows: list[dict[str, Any]] = []
        future_history_rows_filtered = 0
        history_rows_without_timestamp = 0
        latest_history_timestamp: datetime | None = None
        if history_exists:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        row_timestamp = _parse_timestamp(_row_timestamp(row))
                        if row_timestamp is None:
                            history_rows_without_timestamp += 1
                        else:
                            if latest_history_timestamp is None or row_timestamp > latest_history_timestamp:
                                latest_history_timestamp = row_timestamp
                            if cutoff is not None and row_timestamp > cutoff:
                                future_history_rows_filtered += 1
                                continue
                        source_rows.append(row)

        feedback_rows = self._feedback_history_rows(as_of=as_of)
        rows_by_ticket: dict[str, dict[str, Any]] = {}
        unkeyed_rows: list[dict[str, Any]] = []
        for row in source_rows:
            ticket_id = str(row.get("ticket_id") or row.get("id") or "").strip()
            if ticket_id:
                rows_by_ticket[ticket_id] = row
            else:
                unkeyed_rows.append(row)
        for row in feedback_rows:
            ticket_id = str(row.get("ticket_id") or "").strip()
            if ticket_id:
                rows_by_ticket[ticket_id] = row
            else:
                unkeyed_rows.append(row)

        rows = []
        for row in [*unkeyed_rows, *rows_by_ticket.values()]:
            assignee = _normalize_assignee(row.get("assignee"))
            if not self._is_assignable_assignee(assignee):
                continue
            rows.append(
                {
                    "assignee": assignee,
                    "component": _normalize_component(row.get("component")),
                    "product": _normalize_component(row.get("product")),
                    "title": str(row.get("title", "") or ""),
                    "description": str(row.get("description", "") or ""),
                    "text": _row_text(row),
                }
            )

        profile = _build_profile(rows)
        profile["feedback_history_rows"] = len(feedback_rows)
        profile["future_history_rows_filtered"] = future_history_rows_filtered
        profile["history_rows_without_timestamp"] = history_rows_without_timestamp
        self._cached_history_path = path
        self._cached_history_mtime = mtime
        self._cached_profile_source_signature = source_signature
        self._cached_history_latest_timestamp = latest_history_timestamp
        self._cached_history_as_of = (
            "__all_history__"
            if self.config.assignee_feedback_path is None and future_history_rows_filtered == 0
            else as_of
        )
        self._cached_profile = profile
        return profile

    def _feedback_history_rows(self, *, as_of: str) -> list[dict[str, Any]]:
        path = self.config.assignee_feedback_path
        if path is None or not Path(path).exists():
            return []
        feedback_rows = read_assignee_feedback(Path(path))
        if as_of:
            feedback_rows = [
                row
                for row in feedback_rows
                if _timestamp_on_or_before(row.get("created_at"), as_of)
            ]
        return feedback_rows_to_history_rows(feedback_rows)


TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "when",
    "with",
}


def _build_profile(rows: list[dict[str, str]]) -> dict[str, Any]:
    profile = _empty_profile()
    documents = profile["documents"]
    component_counts = profile["component_counts"]
    product_counts = profile["product_counts"]
    product_component_counts = profile["product_component_counts"]
    global_counts = profile["global_counts"]
    owner_component_counts = profile["owner_component_counts"]
    owner_examples = profile["owner_examples"]

    document_frequencies: Counter[str] = Counter()
    token_counts_by_doc: list[Counter[str]] = []
    for row in rows:
        owner = row["assignee"]
        component = row["component"]
        product = row["product"]
        product_component = f"{product}::{component}"
        documents.append(row)
        global_counts[owner] += 1
        component_counts[component][owner] += 1
        product_counts[product][owner] += 1
        product_component_counts[product_component][owner] += 1
        owner_component_counts[owner][component] += 1
        if len(owner_examples[owner]) < 6:
            owner_examples[owner].append(str(row.get("title", "") or row.get("text", ""))[:120])

        counts = Counter(_tokens(row["text"]))
        token_counts_by_doc.append(counts)
        profile["doc_lengths"].append(sum(counts.values()))
        document_frequencies.update(counts.keys())

    total_docs = len(documents)
    profile["avgdl"] = sum(profile["doc_lengths"]) / total_docs if total_docs else 0.0
    for token, frequency in document_frequencies.items():
        profile["idf"][token] = math.log(1 + (total_docs - frequency + 0.5) / (frequency + 0.5))
    for doc_index, counts in enumerate(token_counts_by_doc):
        for token, frequency in counts.items():
            profile["postings"][token].append((doc_index, frequency))
    return profile


def _empty_profile() -> dict[str, Any]:
    return {
        "documents": [],
        "global_counts": Counter(),
        "component_counts": defaultdict(Counter),
        "product_counts": defaultdict(Counter),
        "product_component_counts": defaultdict(Counter),
        "owner_component_counts": defaultdict(Counter),
        "owner_examples": defaultdict(list),
        "doc_lengths": [],
        "avgdl": 0.0,
        "idf": {},
        "postings": defaultdict(list),
    }


def _count_ratio(count: int, counter: Counter[str]) -> float:
    total = sum(counter.values())
    return count / total if total else 0.0


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, "") or "")
        for field in ("title", "description", "product", "component", "severity", "priority", "bug_type")
    )


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def _normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def _component_key_matches(key: Any, component: Any) -> bool:
    owner_key = _normalize_component(key)
    target = _normalize_component(component)
    if owner_key == "unknown":
        return target == "unknown"
    if owner_key == target:
        return True
    key_tokens = [token for token in re.split(r"[^a-z0-9]+", owner_key) if token]
    target_tokens = [token for token in re.split(r"[^a-z0-9]+", target) if token]
    if not key_tokens or len(key_tokens) > len(target_tokens):
        return False
    width = len(key_tokens)
    return any(target_tokens[index : index + width] == key_tokens for index in range(len(target_tokens) - width + 1))


def _component_specificity(value: Any) -> tuple[int, int]:
    normalized = _normalize_component(value)
    return (len([token for token in re.split(r"[^a-z0-9]+", normalized) if token]), len(normalized))


def _normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_valid_assignee(value: Any) -> bool:
    assignee = _normalize_assignee(value)
    if not assignee or assignee == MANUAL_TRIAGE:
        return False
    return not GENERIC_ASSIGNEE_RE.search(assignee)


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text.startswith(("{", "[")):
            return None
        return [line.strip() for line in text.splitlines() if line.strip()]


def _coerce_assignee_set(value: Any) -> set[str]:
    owners: set[str] = set()
    if value is None:
        return owners
    if isinstance(value, str):
        for item in re.split(r"[\n,;]+", value):
            assignee = _normalize_assignee(item)
            if assignee:
                owners.add(assignee)
        return owners
    if isinstance(value, dict):
        for key in ("assignee", "owner", "id", "email", "name"):
            if key in value and not isinstance(value[key], (dict, list, tuple, set)):
                assignee = _normalize_assignee(value[key])
                return {assignee} if assignee else set()
        for key in ("assignees", "owners", "candidates", "candidate_roster", "roster"):
            if key in value:
                owners.update(_coerce_assignee_set(value[key]))
        return owners
    if isinstance(value, (list, tuple, set)):
        for item in value:
            owners.update(_coerce_assignee_set(item))
    return owners


def _split_roster_items(items: Any) -> tuple[set[str], set[str]]:
    active: set[str] = set()
    inactive: set[str] = set()
    if not isinstance(items, (list, tuple, set)):
        items = [items]
    for item in items:
        if isinstance(item, dict):
            assignee = _normalize_assignee(item.get("assignee") or item.get("id") or item.get("email") or item.get("name"))
            if not assignee:
                continue
            status = str(item.get("status") or "").strip().lower()
            if item.get("active") is False or status in {"inactive", "departed", "disabled"}:
                inactive.add(assignee)
            else:
                active.add(assignee)
        else:
            active.update(_coerce_assignee_set(item))
    return active, inactive


def _load_owner_map(path: Path, wrapper_keys: tuple[str, ...]) -> dict[str, list[str]]:
    data = _load_structured_file(path)
    source = data
    if isinstance(data, dict):
        for key in wrapper_keys:
            if isinstance(data.get(key), dict):
                source = data[key]
                break
    if not isinstance(source, dict):
        return {}

    is_file_map = any(key in wrapper_keys for key in ("files", "paths", "file_owners"))
    owners_by_key: dict[str, list[str]] = {}
    for key, value in source.items():
        if str(key).lower() in {"metadata", "schema", "schema_version", "generated_at"}:
            continue
        normalized_key = _normalize_code_path(key) if is_file_map else _normalize_component(key)
        owners = _coerce_owner_list(value)
        if normalized_key and owners:
            owners_by_key[normalized_key] = owners
    return owners_by_key


def _coerce_owner_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [owner for owner in (_normalize_assignee(item) for item in re.split(r"[\n,;]+", value)) if owner]
    if isinstance(value, dict):
        if value.get("active") is False:
            return []
        for key in ("assignee", "owner", "maintainer", "team", "id", "email", "name"):
            if key in value and not isinstance(value[key], (dict, list, tuple, set)):
                owner = _normalize_assignee(value[key])
                return [owner] if owner else []
        owners: list[str] = []
        for key in ("assignees", "owners", "maintainers", "teams"):
            if key in value:
                owners.extend(_coerce_owner_list(value[key]))
        return _unique_values(owners)
    if isinstance(value, (list, tuple, set)):
        owners: list[str] = []
        for item in value:
            owners.extend(_coerce_owner_list(item))
        return _unique_values(owners)
    return []


def _extract_file_paths(ticket_json: dict[str, Any]) -> list[str]:
    direct_fields = (
        "file",
        "files",
        "file_path",
        "file_paths",
        "path",
        "paths",
        "module",
        "modules",
        "stack_trace",
        "traceback",
        "logs",
        "error_message",
        "description",
        "title",
    )
    values: list[str] = []
    for field in direct_fields:
        values.extend(_flatten_text_values(ticket_json.get(field)))
    text = "\n".join(values)
    paths = [_normalize_code_path(value) for value in values if _looks_like_path(str(value))]
    paths.extend(_normalize_code_path(match.group(0)) for match in PATH_RE.finditer(text))
    return _unique_paths([path for path in paths if path])


def _flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_flatten_text_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_flatten_text_values(item))
        return values
    return [str(value)]


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value


def _normalize_code_path(value: Any) -> str:
    path = str(value or "").strip().lower().replace("\\", "/")
    path = path.strip("'\"` ")
    while path.startswith("./"):
        path = path[2:]
    return path


def _path_rule_matches(rule_path: str, ticket_path: str) -> bool:
    rule = _normalize_code_path(rule_path).rstrip("/")
    path = _normalize_code_path(ticket_path)
    if not rule or not path:
        return False
    if path == rule or path.endswith(f"/{rule}"):
        return True
    if path.startswith(f"{rule}/"):
        return True
    return f"/{rule}/" in f"/{path}/"


def _unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_assignee(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _unique_paths(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_code_path(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_probability(value: Any) -> float:
    number = _safe_float(value)
    if number is None:
        return 0.0
    return min(1.0, max(0.0, number))


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        number = _safe_float(item)
        if number is None:
            return []
        result.append(number)
    return result


def _piecewise_linear(value: float, x_values: list[float], y_values: list[float]) -> float:
    if value <= x_values[0]:
        return y_values[0]
    if value >= x_values[-1]:
        return y_values[-1]
    for index in range(1, len(x_values)):
        high_x = x_values[index]
        if value <= high_x:
            low_x = x_values[index - 1]
            low_y = y_values[index - 1]
            high_y = y_values[index]
            if high_x == low_x:
                return high_y
            ratio = (value - low_x) / (high_x - low_x)
            return low_y + ratio * (high_y - low_y)
    return y_values[-1]


def _artifact_weight(weights: dict[str, Any], name: str, default: float) -> float:
    value = _safe_float(weights.get(name))
    return default if value is None else max(0.0, value)


def _counter_top_owner(counter: Counter[str]) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def _ticket_timestamp(ticket_json: dict[str, Any]) -> str:
    return _row_timestamp(ticket_json)


def _row_timestamp(row: dict[str, Any]) -> str:
    for key in ("created_at", "creation_time", "reported_at", "timestamp"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _timestamp_on_or_before(value: Any, as_of: str) -> bool:
    timestamp = _parse_timestamp(value)
    cutoff = _parse_timestamp(as_of)
    return bool(timestamp is not None and cutoff is not None and timestamp <= cutoff)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _select_policy_object(data: Any, policy_name: str) -> dict[str, Any]:
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("policy") or item.get("routing_policy") or item.get("name") or "").strip()
            if not policy_name or name == policy_name:
                return item
        return {}
    if not isinstance(data, dict):
        return {}
    if policy_name and isinstance(data.get(policy_name), dict):
        selected = dict(data[policy_name])
        selected.setdefault("name", policy_name)
        return selected
    for key in ("routing_policies", "policies", "rows"):
        selected = _select_policy_object(data.get(key), policy_name)
        if selected:
            return selected
    if _first_float(data, ("t_high", "high_confidence_threshold", "auto_assign_threshold")) is not None:
        return data
    return {}


def _validated_routing_policy(policy: dict[str, Any]) -> dict[str, Any]:
    deployment_status = str(policy.get("deployment_status") or "").strip().lower()
    if deployment_status and deployment_status != "approved":
        return {}
    low = _safe_float(policy.get("t_low"))
    high = _safe_float(policy.get("t_high"))
    if low is None and high is None:
        return {}
    if low is not None and not 0.0 <= low <= 1.0:
        return {}
    if high is not None and not 0.0 <= high <= 1.000001:
        return {}
    if low is not None and high is not None and low > high:
        return {}
    return {"name": str(policy.get("name") or ""), "t_low": low, "t_high": high}


def _path_signature(paths: tuple[Path | None, ...]) -> tuple[tuple[str, float | None], ...]:
    return tuple(
        (
            str(path or ""),
            Path(path).stat().st_mtime if path is not None and Path(path).exists() else None,
        )
        for path in paths
    )


def _profile_source_signature(config: PipelineConfig) -> tuple[tuple[str, float | None], ...]:
    return _path_signature((
        config.assignee_active_roster_path,
        config.assignee_inactive_path,
        config.assignee_feedback_path,
    ))


def _active_roster_signature(config: PipelineConfig) -> tuple[tuple[str, float | None], ...]:
    return _path_signature((config.assignee_active_roster_path, config.assignee_inactive_path))


def _ownership_signature(config: PipelineConfig) -> tuple[tuple[str, float | None], ...]:
    return _path_signature((config.assignee_component_ownership_path, config.assignee_file_ownership_path))

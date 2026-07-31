from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


WILDCARD = "*"


@dataclass(frozen=True)
class AssigneeEligibility:
    assignee: str
    eligible_products: frozenset[str]
    eligible_components: frozenset[str]
    eligible_product_components: frozenset[tuple[str, str]]

    def matches(self, product: Any, component: Any) -> bool:
        normalized_product = normalize_scope(product)
        normalized_component = normalize_scope(component)
        if self.eligible_product_components:
            return any(
                _scope_matches(required_product, normalized_product)
                and _scope_matches(required_component, normalized_component)
                for required_product, required_component in self.eligible_product_components
            )
        product_matches = not self.eligible_products or any(
            _scope_matches(required, normalized_product)
            for required in self.eligible_products
        )
        component_matches = not self.eligible_components or any(
            _scope_matches(required, normalized_component)
            for required in self.eligible_components
        )
        return product_matches and component_matches


@dataclass(frozen=True)
class AssigneeEligibilityIndex:
    reviewer: str
    valid_from: datetime
    expires_at: datetime
    assignees: dict[str, AssigneeEligibility]
    source: str = "reviewed_assignee_eligibility"

    def is_valid_at(self, value: Any = None) -> bool:
        at_time = parse_datetime(value) if value is not None else datetime.now(UTC)
        return at_time is not None and self.valid_from <= at_time < self.expires_at

    def eligible_assignees_for_ticket(
        self, ticket: dict[str, Any], *, at_time: Any = None
    ) -> set[str]:
        if not self.is_valid_at(at_time):
            return set()
        return {
            owner
            for owner, eligibility in self.assignees.items()
            if eligibility.matches(ticket.get("product"), ticket.get("component"))
        }


def load_assignee_eligibility_index(
    path: Path, *, as_of: Any = None
) -> AssigneeEligibilityIndex:
    index = read_assignee_eligibility_index(path)
    if not index.is_valid_at(as_of):
        raise ValueError("assignee eligibility review is not valid at the requested time")
    return index


def read_assignee_eligibility_index(path: Path) -> AssigneeEligibilityIndex:
    """Read a confirmed snapshot without assuming that it must be valid today."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid assignee eligibility file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("assignee eligibility file must contain a JSON object")
    return parse_assignee_eligibility_payload(payload, require_active=True)


def parse_assignee_eligibility_payload(
    payload: dict[str, Any], *, require_active: bool
) -> AssigneeEligibilityIndex:
    review_confirmed = payload.get("eligibility_review_confirmed")
    if review_confirmed is None:
        review_confirmed = payload.get("review_confirmed")
    reviewer = str(
        payload.get("eligibility_reviewer") or payload.get("reviewer") or ""
    ).strip()
    valid_from = parse_datetime(
        payload.get("eligibility_valid_from")
        or payload.get("valid_from")
        or payload.get("snapshot_at")
    )
    expires_at = parse_datetime(
        payload.get("eligibility_expires_at") or payload.get("expires_at")
    )
    if review_confirmed is not True:
        raise ValueError("assignee eligibility review is not confirmed")
    if (
        require_active
        and "eligibility_review_confirmed" in payload
        and payload.get("review_confirmed") is not True
    ):
        raise ValueError("active assignee roster review is not confirmed")
    if not reviewer:
        raise ValueError("assignee eligibility reviewer is required")
    if valid_from is None or expires_at is None or expires_at <= valid_from:
        raise ValueError("assignee eligibility requires a valid review interval")
    raw_records = payload.get("assignees")
    if not isinstance(raw_records, list):
        raise ValueError("assignee eligibility assignees must be a list")

    records: dict[str, AssigneeEligibility] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("each assignee eligibility entry must be an object")
        owner = normalize_owner(raw.get("assignee") or raw.get("email") or raw.get("id"))
        reviewed = raw.get("eligibility_reviewed")
        if reviewed is None:
            reviewed = raw.get("eligible")
        if reviewed is not True:
            continue
        if not owner:
            raise ValueError("reviewed assignee eligibility entry requires assignee")
        if require_active and raw.get("active") is not True:
            continue
        if raw.get("permissions_confirmed") is not True:
            continue
        products = _scope_set(raw.get("eligible_products"), "eligible_products")
        components = _scope_set(raw.get("eligible_components"), "eligible_components")
        pairs = _scope_pairs(raw.get("eligible_product_components"))
        if pairs and (products or components):
            raise ValueError(
                "use eligible_product_components or independent product/component "
                "scopes, not both in one entry"
            )
        if not products and not components and not pairs:
            raise ValueError(
                f"reviewed assignee eligibility entry has no valid scope: {owner}"
            )
        if owner in records:
            raise ValueError(f"duplicate assignee eligibility entry: {owner}")
        records[owner] = AssigneeEligibility(
            assignee=owner,
            eligible_products=frozenset(products),
            eligible_components=frozenset(components),
            eligible_product_components=frozenset(pairs),
        )
    return AssigneeEligibilityIndex(
        reviewer=reviewer,
        valid_from=valid_from,
        expires_at=expires_at,
        assignees=records,
        source=str(payload.get("source") or "reviewed_assignee_eligibility"),
    )


def annotate_eligible_owner_labels(
    rows: list[dict[str, Any]], index: AssigneeEligibilityIndex
) -> None:
    """Add functional-correctness labels without changing exact-owner labels."""
    for row in rows:
        at_time = parse_datetime(_eligibility_reference_time(row))
        valid_index = (
            index if at_time is not None and index.is_valid_at(at_time) else None
        )
        _annotate_row(row, valid_index, at_time)


def annotate_eligible_owner_labels_from_snapshots(
    rows: list[dict[str, Any]], indexes: list[AssigneeEligibilityIndex]
) -> None:
    """Apply exactly one reviewed snapshot valid at each prediction timestamp."""
    for row in rows:
        at_time = parse_datetime(_eligibility_reference_time(row))
        matching = (
            [index for index in indexes if index.is_valid_at(at_time)]
            if at_time is not None
            else []
        )
        if len(matching) > 1:
            raise ValueError(
                "overlapping assignee eligibility snapshots match one prediction row"
            )
        _annotate_row(row, matching[0] if matching else None, at_time)


def eligible_owner_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("eligibility_label_available") is True]
    top1 = sum(bool(row.get("is_top1_eligible_correct")) for row in labeled)
    top3 = sum(bool(row.get("is_top3_eligible_correct")) for row in labeled)
    top5 = sum(bool(row.get("is_top5_eligible_correct")) for row in labeled)
    candidate_slots = sum(min(5, len(row.get("ranked_candidates") or [])) for row in labeled)
    eligible_slots = sum(int(row.get("eligible_top5_count") or 0) for row in labeled)
    total = len(labeled)
    return {
        "rows": len(rows),
        "eligibility_labeled_rows": total,
        "eligibility_label_coverage": _ratio(total, len(rows)),
        "top1_eligible_correct_rows": top1,
        "top1_eligible_accuracy": _ratio(top1, total),
        "top1_eligible_accuracy_ci95": _wilson_interval(top1, total),
        "top3_eligible_correct_rows": top3,
        "top3_eligible_hit_rate": _ratio(top3, total),
        "top5_eligible_correct_rows": top5,
        "top5_eligible_hit_rate": _ratio(top5, total),
        "top5_eligible_hit_rate_ci95": _wilson_interval(top5, total),
        "top5_candidate_slots": candidate_slots,
        "top5_eligible_candidate_slots": eligible_slots,
        "top5_candidate_eligibility_precision": _ratio(
            eligible_slots, candidate_slots
        ),
        "top5_candidate_eligibility_precision_ci95": _wilson_interval(
            eligible_slots, candidate_slots
        ),
    }


def eligibility_record_fields(record: AssigneeEligibility | None) -> dict[str, Any]:
    if record is None:
        return {
            "eligibility_reviewed": False,
            "permissions_confirmed": False,
            "eligible_products": [],
            "eligible_components": [],
            "eligible_product_components": [],
        }
    return {
        "eligibility_reviewed": True,
        "permissions_confirmed": True,
        "eligible_products": sorted(record.eligible_products),
        "eligible_components": sorted(record.eligible_components),
        "eligible_product_components": [
            {"product": product, "component": component}
            for product, component in sorted(record.eligible_product_components)
        ],
    }


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_owner(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_scope(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return "" if text in {"", "unknown", "none", "null"} else text


def _scope_matches(required: str, actual: str) -> bool:
    return required == WILDCARD or bool(actual and required == actual)


def _eligibility_reference_time(row: dict[str, Any]) -> Any:
    return (
        row.get("prediction_created_at")
        or row.get("created_at")
        or row.get("creation_time")
    )


def _annotate_row(
    row: dict[str, Any], index: AssigneeEligibilityIndex | None, at_time: Any
) -> None:
    eligible = (
        index.eligible_assignees_for_ticket(row, at_time=at_time)
        if index is not None
        else set()
    )
    ranked = [
        normalize_owner(owner)
        for owner in row.get("ranked_candidates", [])
        if normalize_owner(owner)
    ]
    top5 = ranked[:5]
    eligible_top5 = [owner for owner in top5 if owner in eligible]
    row["eligibility_label_available"] = index is not None
    row["eligibility_label_definition"] = "reviewed_time_valid_capability_set_v1"
    row["eligibility_snapshot_valid_from"] = (
        index.valid_from.isoformat() if index is not None else None
    )
    row["eligibility_snapshot_expires_at"] = (
        index.expires_at.isoformat() if index is not None else None
    )
    row["eligible_assignees"] = sorted(eligible)
    row["eligible_assignee_count"] = len(eligible)
    row["is_top1_eligible_correct"] = bool(ranked and ranked[0] in eligible)
    row["is_top3_eligible_correct"] = any(owner in eligible for owner in ranked[:3])
    row["is_top5_eligible_correct"] = bool(eligible_top5)
    row["eligible_top5_count"] = len(eligible_top5)
    row["eligible_top5_precision"] = (
        round(len(eligible_top5) / len(top5), 6) if top5 else 0.0
    )


def _scope_set(value: Any, label: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return {normalized for item in value if (normalized := normalize_scope(item))}


def _scope_pairs(value: Any) -> set[tuple[str, str]]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError("eligible_product_components must be a list")
    pairs = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("eligible product/component scope must be an object")
        product = normalize_scope(item.get("product"))
        component = normalize_scope(item.get("component"))
        if not product or not component:
            raise ValueError("eligible product/component scope requires both fields")
        pairs.add((product, component))
    return pairs


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> dict[str, float]:
    if total == 0:
        return {"lower": 0.0, "upper": 1.0}
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }

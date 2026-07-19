from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "assignee_dataset_path",
    "assignee_active_roster_path",
    "assignee_inactive_path",
    "assignee_component_ownership_path",
    "assignee_file_ownership_path",
    "assignee_feedback_path",
    "assignee_routing_policy_path",
    "assignee_calibration_artifact_path",
    "assignee_open_set_artifact_path",
}
GENERIC_ASSIGNEE_RE = re.compile(
    r"(^|[^a-z0-9])(nobody|unassigned|default|disabled|do-not-reply|noreply|bugzilla|wptsync)([^a-z0-9]|$)"
)


def load_deployment_bundle(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid assignee deployment bundle: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported assignee deployment bundle schema")
    config = payload.get("pipeline_config")
    if not isinstance(config, dict):
        raise ValueError("deployment bundle must contain pipeline_config")

    resolved = dict(config)
    for key in PATH_KEYS:
        value = resolved.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        resolved[key] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
    return {**payload, "pipeline_config": resolved}


def normalize_history_record(row: dict[str, Any]) -> dict[str, Any] | None:
    ticket_id = _clean(row.get("ticket_id") or row.get("id") or row.get("bug_id"))
    title = _clean(row.get("title") or row.get("summary"))
    description = _clean(row.get("description") or row.get("body") or row.get("comments_text"))
    assignee = _clean(row.get("assignee") or row.get("assigned_to") or row.get("owner")).lower()
    created_at = _clean(row.get("created_at") or row.get("creation_time") or row.get("reported_at"))
    if not ticket_id or not title or not assignee or _parse_timestamp(created_at) is None:
        return None
    return {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "product": _clean(row.get("product")).lower() or "unknown",
        "component": _clean(row.get("component") or row.get("module")).lower() or "unknown",
        "priority": _clean(row.get("priority")) or "P3",
        "severity": _clean(row.get("severity")),
        "assignee": assignee,
        "created_at": created_at,
        "file_paths": _file_paths(row),
        "source": _clean(row.get("source")) or "project_history",
    }


def temporal_split(
    rows: list[dict[str, Any]], train_ratio: float, validation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0:
        raise ValueError("train and validation ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    ordered_rows = sorted(rows, key=_row_sort_key)
    if len(ordered_rows) < 3:
        raise ValueError("at least three history rows are required for temporal splitting")

    train_target = max(1, int(len(ordered_rows) * train_ratio))
    validation_target = max(1, int(len(ordered_rows) * validation_ratio))
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in ordered_rows:
        if len(train) < train_target:
            train.append(row)
        elif len(validation) < validation_target:
            validation.append(row)
        else:
            test.append(row)
    if not validation or not test:
        raise ValueError("temporal split produced an empty validation or test partition")

    ticket_ids = [str(row["ticket_id"]) for row in ordered_rows]
    audit = {
        "input_rows": len(rows),
        "output_rows": len(ordered_rows),
        "non_unique_ticket_ids": len(ticket_ids) - len(set(ticket_ids)),
        "upstream_duplicate_filter_required": True,
        "duplicate_detection_performed": False,
        "cross_split_ticket_id_overlap": _overlap_count(train, validation, test, "ticket_id"),
        "cross_split_content_overlap": _content_overlap_count(train, validation, test),
        "chronological_order_valid": _chronological_order_valid(train, validation, test),
    }
    return sorted(train, key=_row_sort_key), sorted(validation, key=_row_sort_key), sorted(test, key=_row_sort_key), audit


def build_roster(
    train_rows: list[dict[str, Any]], minimum_history: int, inactive: set[str] | None = None
) -> tuple[dict[str, Any], set[str]]:
    inactive = {str(owner).strip().lower() for owner in (inactive or set()) if str(owner).strip()}
    counts = Counter(str(row["assignee"]) for row in train_rows)
    active = {
        owner
        for owner, count in counts.items()
        if count >= minimum_history and owner not in inactive and not GENERIC_ASSIGNEE_RE.search(owner)
    }
    assignees = []
    for owner in sorted(active):
        components = Counter(
            str(row.get("component") or "unknown") for row in train_rows if row["assignee"] == owner
        )
        assignees.append(
            {
                "assignee": owner,
                "active": True,
                "status": "active_inferred_from_training_window",
                "history_count": counts[owner],
                "components": [name for name, _ in components.most_common(10)],
            }
        )
    return {
        "schema_version": 1,
        "source": "temporally_prior_training_labels",
        "requires_human_roster_review": True,
        "minimum_history": minimum_history,
        "assignees": assignees,
        "candidates": sorted(active),
        "inactive": sorted(inactive),
    }, active


def load_assignee_set(path: Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid inactive assignee file: {path}") from exc
    values: Any = payload
    if isinstance(payload, dict):
        values = payload.get("inactive") or payload.get("departed") or payload.get("assignees") or []
    if not isinstance(values, list):
        raise ValueError("inactive assignee file must contain a JSON list")
    owners = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("assignee") or value.get("email") or value.get("id")
        owner = str(value or "").strip().lower()
        if owner:
            owners.add(owner)
    return owners


def derive_ownership(
    train_rows: list[dict[str, Any]], roster: set[str], minimum_rows: int, minimum_share: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    component_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    file_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        owner = str(row["assignee"])
        if owner not in roster:
            continue
        component_counts[str(row.get("component") or "unknown")][owner] += 1
        for path in row.get("file_paths", []):
            file_counts[str(path)][owner] += 1

    def select(groups: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for key, counts in sorted(groups.items()):
            total = sum(counts.values())
            owner, count = counts.most_common(1)[0]
            share = count / total if total else 0.0
            if total >= minimum_rows and share >= minimum_share:
                selected[key] = {"owner": owner, "history_rows": total, "owner_share": round(share, 6)}
        return selected

    metadata = {
        "schema_version": 1,
        "deployment_status": "candidate_requires_owner_review",
        "source": "derived_from_training_history",
        "minimum_rows": minimum_rows,
        "minimum_owner_share": minimum_share,
    }
    return {**metadata, "components": select(component_counts)}, {**metadata, "files": select(file_counts)}


def _row_sort_key(row: dict[str, Any]) -> tuple[datetime, str]:
    return (_parse_timestamp(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), str(row.get("ticket_id")))


def _parse_timestamp(value: Any) -> datetime | None:
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


def _normalized_content(row: dict[str, Any]) -> str:
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    return " ".join(re.findall(r"[a-z0-9_]+", text))


def _overlap_count(*parts: Any) -> int:
    field = parts[-1]
    sets = [{str(row.get(field) or "") for row in rows} for rows in parts[:-1]]
    return sum(len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def _content_overlap_count(*parts: list[dict[str, Any]]) -> int:
    sets = [{_normalized_content(row) for row in rows} for rows in parts]
    return sum(len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def _chronological_order_valid(*parts: list[dict[str, Any]]) -> bool:
    boundaries = []
    for rows in parts:
        timestamps = [_parse_timestamp(row.get("created_at")) for row in rows]
        valid = [value for value in timestamps if value is not None]
        if not valid:
            return False
        boundaries.append((min(valid), max(valid)))
    return all(boundaries[index][1] <= boundaries[index + 1][0] for index in range(len(boundaries) - 1))


def _file_paths(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("file_paths", "files", "file_path", "path", "paths", "module_paths"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    normalized = []
    for value in values:
        path = value.strip().lower().replace("\\", "/").lstrip("./")
        if path and path not in normalized:
            normalized.append(path)
    return normalized


def _clean(value: Any) -> str:
    return str(value or "").strip()

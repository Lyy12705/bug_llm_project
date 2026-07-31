from __future__ import annotations

import hashlib
import json
import math
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
IMMUTABLE_DEPLOYMENT_PATH_KEYS = {
    "assignee_dataset_path",
    "assignee_active_roster_path",
    "assignee_routing_policy_path",
    "assignee_calibration_artifact_path",
    "assignee_open_set_artifact_path",
}
BUNDLE_SCHEMA_VERSION = 2
GENERIC_ASSIGNEE_RE = re.compile(
    r"(^|[^a-z0-9])(nobody|unassigned|default|disabled|do-not-reply|noreply|bugzilla|wptsync)([^a-z0-9]|$)"
)


def load_deployment_bundle(
    path: Path,
    *,
    require_approved: bool = True,
    verify_integrity: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Load a self-contained deployment bundle and fail closed by default.

    Research-only bundles may be inspected with ``require_approved=False`` but
    cannot be used by the runtime entry points. Approved bundles require the
    current schema, a passed gate, an unexpired timestamp, contained artifact
    paths, and a matching SHA-256 manifest.
    """

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid assignee deployment bundle: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, BUNDLE_SCHEMA_VERSION}:
        raise ValueError("unsupported assignee deployment bundle schema")
    if require_approved:
        _validate_approved_bundle(payload, now=now)
    config = payload.get("pipeline_config")
    if not isinstance(config, dict):
        raise ValueError("deployment bundle must contain pipeline_config")

    root = path.parent.resolve()
    resolved = dict(config)
    for key in PATH_KEYS:
        value = resolved.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not _is_relative_to(candidate, root):
            raise ValueError(f"deployment bundle path escapes its directory: {key}")
        resolved[key] = candidate
    if require_approved and verify_integrity:
        _verify_artifact_manifest(payload, resolved, root)
    return {**payload, "pipeline_config": resolved}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    if not _is_relative_to(resolved, base):
        raise ValueError(f"artifact is outside deployment directory: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"deployment artifact is missing: {resolved}")
    return {
        "path": str(resolved.relative_to(base)),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"lower": 0.0, "upper": 1.0}
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def _validate_approved_bundle(payload: dict[str, Any], *, now: datetime | None) -> None:
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("approved deployment bundles require schema version 2")
    if payload.get("deployment_status") != "approved":
        raise ValueError("assignee deployment bundle is not approved")
    if payload.get("approval_gate_passed") is not True:
        raise ValueError("assignee deployment bundle approval gate did not pass")
    gates = payload.get("approval_gates")
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError("assignee deployment bundle contains a failed or incomplete approval gate")
    expires_at = _parse_timestamp(payload.get("expires_at"))
    if expires_at is None:
        raise ValueError("approved deployment bundle must contain a valid expires_at timestamp")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if expires_at <= current.astimezone(timezone.utc):
        raise ValueError("assignee deployment bundle has expired")


def _verify_artifact_manifest(
    payload: dict[str, Any], resolved_config: dict[str, Any], root: Path
) -> None:
    manifest = payload.get("artifact_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("approved deployment bundle must contain artifact_manifest")
    configured_immutable = {
        key for key in IMMUTABLE_DEPLOYMENT_PATH_KEYS if resolved_config.get(key) is not None
    }
    if configured_immutable != set(manifest):
        raise ValueError("deployment artifact manifest does not match configured immutable artifacts")
    for key in sorted(configured_immutable):
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"invalid deployment artifact manifest entry: {key}")
        candidate = Path(resolved_config[key]).resolve()
        if not _is_relative_to(candidate, root):
            raise ValueError(f"deployment artifact escapes its directory: {key}")
        expected_relative = str(candidate.relative_to(root))
        if entry.get("path") != expected_relative:
            raise ValueError(f"deployment artifact path does not match manifest: {key}")
        if not candidate.is_file():
            raise ValueError(f"deployment artifact is missing: {key}")
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size != candidate.stat().st_size:
            raise ValueError(f"deployment artifact size mismatch: {key}")
        expected_hash = str(entry.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or sha256_file(candidate) != expected_hash:
            raise ValueError(f"deployment artifact hash mismatch: {key}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_history_record(row: dict[str, Any]) -> dict[str, Any] | None:
    ticket_id = _clean(row.get("ticket_id") or row.get("id") or row.get("bug_id"))
    title = _clean(row.get("title") or row.get("summary"))
    description = _clean(row.get("description") or row.get("body") or row.get("comments_text"))
    assignee = _clean(row.get("assignee") or row.get("assigned_to") or row.get("owner")).lower()
    created_at = _clean(row.get("created_at") or row.get("creation_time") or row.get("reported_at"))
    if not ticket_id or not title or not assignee or _parse_timestamp(created_at) is None:
        return None
    declared_label_source = _clean(row.get("label_availability_source"))
    assignment_event_time = _clean(
        row.get("assignment_label_available_at")
        or row.get("assignment_changed_at")
        or row.get("assigned_at")
    )
    generic_label_available_at = _clean(row.get("label_available_at"))
    last_change_time = _clean(row.get("last_change_time") or row.get("updated_at"))
    label_available_at = (
        assignment_event_time or generic_label_available_at or last_change_time
    )
    if declared_label_source == "last_change_time_proxy":
        label_availability_source = "last_change_time_proxy"
    elif declared_label_source == "explicit_assignment_event_time" and (
        assignment_event_time or generic_label_available_at
    ):
        label_availability_source = "explicit_assignment_event_time"
    elif assignment_event_time or generic_label_available_at:
        label_availability_source = "explicit_assignment_event_time"
    elif last_change_time:
        label_availability_source = "last_change_time_proxy"
    else:
        label_availability_source = "missing"
    duplicate_of = _clean(row.get("duplicate_of") or row.get("dupe_of"))
    duplicate_family = _clean(
        row.get("duplicate_family") or row.get("duplicate_family_id") or duplicate_of
    )
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
        "label_available_at": label_available_at,
        "label_availability_source": label_availability_source,
        "last_change_time": last_change_time,
        "owner_event_id": _clean(
            row.get("owner_event_id")
            or row.get("assignment_event_id")
            or row.get("assignee_event_id")
            or row.get("source_event_id")
        ),
        "duplicate_of": duplicate_of,
        "duplicate_family": duplicate_family,
        "reporter": _clean(row.get("reporter") or row.get("creator")).lower(),
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
        "cross_split_duplicate_family_overlap": _duplicate_family_overlap_count(
            train, validation, test
        ),
        "cross_split_near_duplicate_text_overlap": _near_duplicate_overlap_count(
            train, validation, test
        ),
        "cross_split_owner_event_overlap": _field_overlap_count(
            train, validation, test, field="owner_event_id"
        ),
        "chronological_order_valid": _chronological_order_valid(train, validation, test),
        "partitions": {
            "train": _partition_audit(train),
            "validation": _partition_audit(validation),
            "test": _partition_audit(test),
        },
    }
    return sorted(train, key=_row_sort_key), sorted(validation, key=_row_sort_key), sorted(test, key=_row_sort_key), audit


def split_temporal_partition(
    rows: list[dict[str, Any]], first_ratio: float = 0.5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split one temporal partition into earlier and later, non-empty windows."""

    if not 0.0 < first_ratio < 1.0:
        raise ValueError("first_ratio must be between 0 and 1")
    ordered = sorted(rows, key=_row_sort_key)
    if len(ordered) < 2:
        raise ValueError("at least two rows are required for a temporal sub-split")
    boundary = min(len(ordered) - 1, max(1, int(len(ordered) * first_ratio)))
    first, second = ordered[:boundary], ordered[boundary:]
    audit = {
        "chronological_order_valid": _chronological_order_valid(first, second),
        "cross_split_ticket_id_overlap": _overlap_count(first, second, "ticket_id"),
        "cross_split_content_overlap": _content_overlap_count(first, second),
        "cross_split_duplicate_family_overlap": _duplicate_family_overlap_count(
            first, second
        ),
        "cross_split_near_duplicate_text_overlap": _near_duplicate_overlap_count(
            first, second
        ),
        "cross_split_owner_event_overlap": _field_overlap_count(
            first, second, field="owner_event_id"
        ),
        "first": _partition_audit(first),
        "second": _partition_audit(second),
    }
    return first, second, audit


def temporal_protocol_manifest(
    partitions: dict[str, list[dict[str, Any]]],
    *,
    source: str,
    seed: int = 0,
    experiment_id: str = "",
    git_commit: str = "",
    partition_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not partitions or any(not rows for rows in partitions.values()):
        raise ValueError("temporal protocol requires non-empty partitions")
    ordered_names = list(partitions)
    chronological = _chronological_order_valid(*(partitions[name] for name in ordered_names))
    ticket_overlap = _overlap_count(*(partitions[name] for name in ordered_names), "ticket_id")
    content_overlap = _content_overlap_count(*(partitions[name] for name in ordered_names))
    duplicate_family_overlap = _duplicate_family_overlap_count(
        *(partitions[name] for name in ordered_names)
    )
    near_duplicate_overlap = _near_duplicate_overlap_count(
        *(partitions[name] for name in ordered_names)
    )
    owner_event_overlap = _field_overlap_count(
        *(partitions[name] for name in ordered_names), field="owner_event_id"
    )
    roles = partition_roles or {name: name for name in ordered_names}
    if set(roles) != set(ordered_names):
        raise ValueError("partition_roles must describe every temporal partition exactly once")
    checks = {
        "chronological_order_valid": chronological,
        "no_cross_split_ticket_id_overlap": ticket_overlap == 0,
        "no_cross_split_content_overlap": content_overlap == 0,
        "no_cross_split_duplicate_family_overlap": duplicate_family_overlap == 0,
        "no_cross_split_near_duplicate_text_overlap": near_duplicate_overlap == 0,
        "no_cross_split_owner_event_overlap": owner_event_overlap == 0,
    }
    return {
        "schema_version": 2,
        "protocol": "strict_temporal_assignee_routing_v2",
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "source": source,
        "seed": seed,
        "partition_order": ordered_names,
        "partition_roles": roles,
        "partitions": {
            name: _partition_audit(partitions[name]) for name in ordered_names
        },
        "audit": {
            "chronological_order_valid": chronological,
            "cross_split_ticket_id_overlap": ticket_overlap,
            "cross_split_content_overlap": content_overlap,
            "cross_split_duplicate_family_overlap": duplicate_family_overlap,
            "cross_split_near_duplicate_text_overlap": near_duplicate_overlap,
            "cross_split_owner_event_overlap": owner_event_overlap,
            "checks": checks,
            "leakage_free": all(checks.values()),
        },
    }


def build_roster(
    train_rows: list[dict[str, Any]],
    minimum_history: int,
    inactive: set[str] | None = None,
    reviewed_active: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    inactive = {str(owner).strip().lower() for owner in (inactive or set()) if str(owner).strip()}
    reviewed_active = {
        str(owner).strip().lower() for owner in (reviewed_active or set()) if str(owner).strip()
    }
    counts = Counter(str(row["assignee"]) for row in train_rows)
    inferred_active = {
        owner
        for owner, count in counts.items()
        if count >= minimum_history and owner not in inactive and not GENERIC_ASSIGNEE_RE.search(owner)
    }
    reviewed_active = {
        owner
        for owner in reviewed_active
        if owner not in inactive and not GENERIC_ASSIGNEE_RE.search(owner)
    }
    active = inferred_active | reviewed_active
    assignees = []
    for owner in sorted(active):
        components = Counter(
            str(row.get("component") or "unknown") for row in train_rows if row["assignee"] == owner
        )
        assignees.append(
            {
                "assignee": owner,
                "active": True,
                "status": (
                    "active_reviewed_roster"
                    if owner in reviewed_active
                    else "active_inferred_from_training_window"
                ),
                "history_count": counts.get(owner, 0),
                "components": [name for name, _ in components.most_common(10)],
            }
        )
    return {
        "schema_version": 1,
        "source": (
            "reviewed_active_roster_plus_temporally_prior_training_labels"
            if reviewed_active
            else "temporally_prior_training_labels"
        ),
        "requires_human_roster_review": True,
        "minimum_history": minimum_history,
        "assignees": assignees,
        "candidates": sorted(active),
        "inactive": sorted(inactive),
        "reviewed_active": sorted(reviewed_active),
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
        values = (
            payload.get("inactive")
            or payload.get("departed")
            or payload.get("assignees")
            or payload.get("candidates")
            or []
        )
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


def load_active_assignee_set(path: Path | None) -> set[str]:
    """Load active candidates without accidentally treating `inactive` as active.

    Versioned roster objects may contain both fields. The generic
    ``load_assignee_set`` helper is retained for simple active or inactive list
    files, while runtime active-roster consumers should use this explicit
    loader.
    """

    if path is None:
        return set()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid active assignee roster: {path}") from exc
    if isinstance(payload, list):
        values: Any = payload
        inactive_values: Any = []
    elif isinstance(payload, dict):
        values = payload.get("candidates")
        if values is None:
            values = [
                row
                for row in payload.get("assignees", [])
                if isinstance(row, dict) and row.get("active") is True
            ]
        inactive_values = payload.get("inactive") or []
    else:
        raise ValueError("active assignee roster must contain a JSON object or list")
    if not isinstance(values, list) or not isinstance(inactive_values, list):
        raise ValueError("active assignee roster candidates and inactive must be lists")

    def owner_value(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("assignee") or value.get("email") or value.get("id")
        return str(value or "").strip().lower()

    active = {owner for value in values if (owner := owner_value(value))}
    inactive = {owner for value in inactive_values if (owner := owner_value(value))}
    overlap = active & inactive
    if overlap:
        raise ValueError(
            "active assignee roster also marks candidates inactive: " + ", ".join(sorted(overlap))
        )
    return active


def load_assignee_alias_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid assignee alias file: {path}") from exc
    values: Any = payload.get("aliases") if isinstance(payload, dict) and "aliases" in payload else payload
    if not isinstance(values, dict):
        raise ValueError("assignee alias file must contain a JSON object")
    aliases = {
        str(alias).strip().lower(): str(canonical).strip().lower()
        for alias, canonical in values.items()
        if str(alias).strip() and str(canonical).strip()
    }
    for alias in aliases:
        _resolve_assignee_alias(alias, aliases)
    return aliases


def apply_assignee_aliases(
    rows: list[dict[str, Any]], aliases: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    remapped = 0
    output = []
    for source in rows:
        row = dict(source)
        original = str(row.get("assignee") or "").strip().lower()
        canonical = _resolve_assignee_alias(original, aliases)
        if canonical != original:
            remapped += 1
            row["assignee"] = canonical
        output.append(row)
    return output, remapped


def canonicalize_assignee_set(values: set[str], aliases: dict[str, str]) -> set[str]:
    return {
        _resolve_assignee_alias(str(value).strip().lower(), aliases)
        for value in values
        if str(value).strip()
    }


def _resolve_assignee_alias(value: str, aliases: dict[str, str]) -> str:
    current = value
    visited: set[str] = set()
    while current in aliases and aliases[current] != current:
        if current in visited:
            raise ValueError(f"cyclic assignee alias mapping contains: {current}")
        visited.add(current)
        current = aliases[current]
    return current


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
    sets = [
        {content for row in rows if (content := _normalized_content(row))}
        for rows in parts
    ]
    return sum(len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def _field_overlap_count(
    *parts: list[dict[str, Any]], field: str
) -> int:
    sets = [
        {value for row in rows if (value := _clean(row.get(field)).lower())}
        for rows in parts
    ]
    return sum(
        len(sets[left] & sets[right])
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    )


def _duplicate_family_overlap_count(*parts: list[dict[str, Any]]) -> int:
    """Count explicit duplicate-family links that cross temporal partitions.

    Every row contributes its own ticket ID so a duplicate that points at an
    original ticket in another partition is detected. Rows without a duplicate
    relation only contribute their ticket ID and therefore cannot collide with
    an unrelated family.
    """

    sets: list[set[str]] = []
    for rows in parts:
        values: set[str] = set()
        for row in rows:
            ticket_id = _normalize_ticket_reference(row.get("ticket_id"))
            duplicate = _normalize_ticket_reference(
                row.get("duplicate_family")
                or row.get("duplicate_family_id")
                or row.get("duplicate_of")
                or row.get("dupe_of")
            )
            if ticket_id:
                values.add(ticket_id)
            if duplicate:
                values.add(duplicate)
        sets.append(values)
    return sum(
        len(sets[left] & sets[right])
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    )


def _near_duplicate_overlap_count(
    *parts: list[dict[str, Any]],
    maximum_hamming_distance: int = 3,
    minimum_token_jaccard: float = 0.80,
) -> int:
    """Detect near-duplicate text across partitions with deterministic SimHash LSH.

    Four 16-bit bands plus a bounded rare-token inverted index keep the audit
    close to linear for normal datasets. Candidate pairs are verified with an
    exact Hamming-distance or token-Jaccard check.
    """

    fingerprints: list[list[tuple[str, frozenset[str], int]]] = []
    for rows in parts:
        values: dict[str, tuple[str, frozenset[str], int]] = {}
        for row in rows:
            content = _normalized_content(row)
            fingerprint = _simhash(row)
            token_set = frozenset(
                token
                for token in re.findall(r"[a-z0-9_]+", content)
                if len(token) >= 3
            )
            if fingerprint is not None and len(token_set) >= 5:
                values[content] = (content, token_set, fingerprint)
        fingerprints.append(list(values.values()))
    overlaps = 0
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            right_bands: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
            token_index: defaultdict[str, set[int]] = defaultdict(set)
            for index, (_, token_set, fingerprint) in enumerate(fingerprints[right]):
                for band in range(4):
                    right_bands[
                        (band, (fingerprint >> (band * 16)) & 0xFFFF)
                    ].add(index)
                for token in token_set:
                    token_index[token].add(index)
            maximum_posting = max(100, math.ceil(len(fingerprints[right]) * 0.05))
            matched: set[tuple[str, str]] = set()
            for source_content, source_tokens, source in fingerprints[left]:
                candidates: set[int] = set()
                for band in range(4):
                    candidates.update(
                        right_bands.get(
                            (band, (source >> (band * 16)) & 0xFFFF), set()
                        )
                    )
                for token in source_tokens:
                    postings = token_index.get(token, set())
                    if len(postings) <= maximum_posting:
                        candidates.update(postings)
                for candidate_index in candidates:
                    candidate_content, candidate_tokens, candidate = fingerprints[right][
                        candidate_index
                    ]
                    if source_content == candidate_content:
                        continue
                    union = source_tokens | candidate_tokens
                    jaccard = len(source_tokens & candidate_tokens) / len(union) if union else 0.0
                    if (
                        (source ^ candidate).bit_count() <= maximum_hamming_distance
                        or jaccard >= minimum_token_jaccard
                    ):
                        matched.add((source_content, candidate_content))
            overlaps += len(matched)
    return overlaps


def _simhash(row: dict[str, Any]) -> int | None:
    token_values = re.findall(r"[a-z0-9_]+", _normalized_content(row))
    if len(token_values) < 5:
        return None
    shingles = [
        " ".join(token_values[index : index + 3])
        for index in range(len(token_values) - 2)
    ]
    weights = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _normalize_ticket_reference(value: Any) -> str:
    text = _clean(value).lower()
    if not text:
        return ""
    if text.startswith("bmo_"):
        return text
    if text.isdigit():
        return f"bmo_{text}"
    return text


def _chronological_order_valid(*parts: list[dict[str, Any]]) -> bool:
    boundaries = []
    for rows in parts:
        timestamps = [_parse_timestamp(row.get("created_at")) for row in rows]
        valid = [value for value in timestamps if value is not None]
        if not valid:
            return False
        boundaries.append((min(valid), max(valid)))
    return all(boundaries[index][1] <= boundaries[index + 1][0] for index in range(len(boundaries) - 1))


def _partition_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [_parse_timestamp(row.get("created_at")) for row in rows]
    valid = [value for value in timestamps if value is not None]
    owners = {str(row.get("assignee") or "") for row in rows if str(row.get("assignee") or "")}
    label_sources = Counter(
        str(row.get("label_availability_source") or "missing") for row in rows
    )
    explicit_label_rows = label_sources["explicit_assignment_event_time"]
    proxy_label_rows = label_sources["last_change_time_proxy"]
    missing_label_rows = len(rows) - explicit_label_rows - proxy_label_rows
    digest = hashlib.sha256()
    for row in sorted(rows, key=_row_sort_key):
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "rows": len(rows),
        "assignees": len(owners),
        "start": min(valid).isoformat() if valid else None,
        "end": max(valid).isoformat() if valid else None,
        "rows_sha256": digest.hexdigest(),
        "label_availability": {
            "explicit_assignment_event_rows": explicit_label_rows,
            "last_change_time_proxy_rows": proxy_label_rows,
            "missing_rows": missing_label_rows,
            "explicit_assignment_event_coverage": round(
                explicit_label_rows / len(rows), 6
            )
            if rows
            else 0.0,
        },
    }


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

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import (  # noqa: E402
    GENERIC_ASSIGNEE_RE,
    apply_assignee_aliases,
    canonicalize_assignee_set,
    load_active_assignee_set,
    load_assignee_alias_map,
    load_assignee_set,
    sha256_file,
)
from modules.assignee_eligibility import (  # noqa: E402
    AssigneeEligibilityIndex,
    eligibility_record_fields,
    parse_assignee_eligibility_payload,
)

from assignee_open_set_common import write_json  # noqa: E402
from train_assignee_ltr import (  # noqa: E402
    DEFAULT_DATA_DIR,
    exclude_cross_split_overlap,
    parse_timestamp,
    read_label_map,
    row_sort_key,
)
from train_assignee_rolling_open_set import (  # noqa: E402
    RAW_DIR,
    apply_label_availability,
    load_rows,
    read_label_availability,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an auditable, time-versioned operational assignee roster from prior "
            "assignment activity and optional maintainer-reviewed overrides."
        )
    )
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--as-of", required=True, help="ISO-8601 roster snapshot cutoff.")
    parser.add_argument("--active-lookback-days", type=int, default=180)
    parser.add_argument("--inactive-lookback-days", type=int, default=365)
    parser.add_argument("--minimum-active-assignments", type=int, default=3)
    parser.add_argument("--roster-valid-days", type=int, default=30)
    parser.add_argument("--reviewed-active", type=Path)
    parser.add_argument("--reviewed-inactive", type=Path)
    parser.add_argument(
        "--reviewed-eligibility",
        type=Path,
        help=(
            "Maintainer-reviewed product/component capability and permission "
            "attestation. Historical components are never used as a substitute."
        ),
    )
    parser.add_argument(
        "--assignee-alias-map",
        type=Path,
        help="Reviewed historical alias-to-canonical owner identity mapping.",
    )
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--confirm-organizational-review", action="store_true")
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--base-raw", type=Path, default=RAW_DIR / "bmo_public_10k_raw.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    as_of = parse_datetime(args.as_of)
    if as_of is None:
        raise SystemExit("--as-of must be a valid ISO-8601 timestamp")
    if args.active_lookback_days < 1 or args.inactive_lookback_days < args.active_lookback_days:
        raise SystemExit("inactive lookback must be at least the active lookback")
    if args.minimum_active_assignments < 1 or not 1 <= args.roster_valid_days <= 90:
        raise SystemExit("invalid roster activity or validity setting")
    if args.confirm_organizational_review and not args.reviewer.strip():
        raise SystemExit("--reviewer is required with --confirm-organizational-review")

    rows, source_audit = load_history(
        args.history,
        label_map_path=args.assignee_label_map,
        base_raw=args.base_raw,
    )
    alias_map = load_assignee_alias_map(args.assignee_alias_map)
    rows, alias_rows_remapped = apply_assignee_aliases(rows, alias_map)
    reviewed_active = (
        load_active_assignee_set(args.reviewed_active) if args.reviewed_active else set()
    )
    reviewed_inactive = (
        load_assignee_set(args.reviewed_inactive) if args.reviewed_inactive else set()
    )
    reviewed_active = canonicalize_assignee_set(reviewed_active, alias_map)
    reviewed_inactive = canonicalize_assignee_set(reviewed_inactive, alias_map)
    eligibility_index = None
    if args.reviewed_eligibility:
        try:
            payload = json.loads(args.reviewed_eligibility.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("reviewed eligibility must contain a JSON object")
            eligibility_index = parse_assignee_eligibility_payload(
                payload, require_active=False
            )
            if not eligibility_index.is_valid_at(as_of):
                raise ValueError("reviewed eligibility is not valid at --as-of")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Unable to use --reviewed-eligibility: {exc}") from exc
    roster, report = build_roster_snapshot(
        rows,
        as_of=as_of,
        active_lookback_days=args.active_lookback_days,
        inactive_lookback_days=args.inactive_lookback_days,
        minimum_active_assignments=args.minimum_active_assignments,
        valid_days=args.roster_valid_days,
        reviewed_active=reviewed_active,
        reviewed_inactive=reviewed_inactive,
        reviewer=args.reviewer.strip(),
        organizational_review_confirmed=args.confirm_organizational_review,
        eligibility_index=eligibility_index,
    )
    report["source_audit"] = {
        **source_audit,
        "canonical_identity_mapping": {
            "configured": bool(alias_map),
            "aliases": len(alias_map),
            "history_rows_remapped": alias_rows_remapped,
            "reviewed_rosters_canonicalized": True,
        },
    }
    roster["canonical_identity_mapping"] = report["source_audit"][
        "canonical_identity_mapping"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, roster)
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "active": len(roster["candidates"]),
                "inactive": len(roster["inactive"]),
                "cold_start": len(roster["cold_start"]),
                "review_confirmed": roster["review_confirmed"],
                "eligibility_review_confirmed": roster[
                    "eligibility_review_confirmed"
                ],
                "blocked_by": report["blocked_by"],
                "eligibility_blocked_by": report["eligibility_blocked_by"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_roster_snapshot(
    rows: list[dict[str, Any]],
    *,
    as_of: datetime,
    active_lookback_days: int,
    inactive_lookback_days: int,
    minimum_active_assignments: int,
    valid_days: int,
    reviewed_active: set[str],
    reviewed_inactive: set[str],
    reviewer: str,
    organizational_review_confirmed: bool,
    eligibility_index: AssigneeEligibilityIndex | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    as_of_epoch = as_of.timestamp()
    active_start = (as_of - timedelta(days=active_lookback_days)).timestamp()
    inactive_start = (as_of - timedelta(days=inactive_lookback_days)).timestamp()
    eligible = []
    future_created = 0
    unavailable_labels = 0
    proxy_label_times = 0
    explicit_label_times = 0
    missing_components = 0
    for row in rows:
        created_at = parse_timestamp(str(row.get("created_at") or ""))
        label_at = parse_timestamp(str(row.get("label_available_at") or ""))
        if created_at is None or created_at >= as_of_epoch:
            future_created += 1
            continue
        if label_at is None or label_at > as_of_epoch:
            unavailable_labels += 1
            continue
        label_source = str(row.get("label_availability_source") or "").strip()
        if label_source == "last_change_time_proxy" or (
            not label_source and row.get("last_change_time")
        ):
            proxy_label_times += 1
        else:
            explicit_label_times += 1
        if not str(row.get("component") or "").strip() or str(
            row.get("component") or ""
        ).strip().lower() == "unknown":
            missing_components += 1
        eligible.append(row)

    total = Counter()
    active_counts = Counter()
    inactive_window_counts = Counter()
    components: defaultdict[str, Counter[str]] = defaultdict(Counter)
    last_seen: dict[str, float] = {}
    for row in eligible:
        owner = normalize_owner(row.get("assignee"))
        if not owner or GENERIC_ASSIGNEE_RE.search(owner):
            continue
        created_at = parse_timestamp(str(row.get("created_at") or ""))
        if created_at is None:
            continue
        total[owner] += 1
        components[owner][str(row.get("component") or "unknown")] += 1
        last_seen[owner] = max(last_seen.get(owner, float("-inf")), created_at)
        if created_at >= active_start:
            active_counts[owner] += 1
        if created_at >= inactive_start:
            inactive_window_counts[owner] += 1

    reviewed_active = {normalize_owner(value) for value in reviewed_active if normalize_owner(value)}
    reviewed_inactive = {
        normalize_owner(value) for value in reviewed_inactive if normalize_owner(value)
    }
    overlap = reviewed_active & reviewed_inactive
    if overlap:
        raise ValueError(
            "reviewed active/inactive overrides overlap: " + ", ".join(sorted(overlap))
        )
    inferred_active = {
        owner for owner, count in active_counts.items() if count >= minimum_active_assignments
    }
    inferred_inactive = {
        owner
        for owner, count in total.items()
        if count >= minimum_active_assignments and inactive_window_counts[owner] == 0
    }
    inactive = (inferred_inactive | reviewed_inactive) - reviewed_active
    active = (inferred_active | reviewed_active) - inactive
    cold_start = {owner for owner in reviewed_active if total[owner] == 0}
    eligibility_owners = (
        set(eligibility_index.assignees) if eligibility_index is not None else set()
    )
    all_owners = sorted(
        set(total) | reviewed_active | reviewed_inactive | eligibility_owners
    )
    records = []
    for owner in all_owners:
        if owner in cold_start:
            status = "cold_start_known_reviewed"
        elif owner in reviewed_inactive:
            status = "inactive_reviewed"
        elif owner in reviewed_active:
            status = "active_reviewed"
        elif owner in active:
            status = "active_inferred_from_recent_assignment_activity"
        elif owner in inactive:
            status = "inactive_inferred_from_assignment_inactivity"
        else:
            status = "dormant_or_insufficient_evidence"
        records.append(
            {
                "assignee": owner,
                "active": owner in active,
                "status": status,
                "history_count": total[owner],
                "active_window_count": active_counts[owner],
                "inactive_window_count": inactive_window_counts[owner],
                "last_assignment_created_at": (
                    datetime.fromtimestamp(last_seen[owner], UTC).isoformat()
                    if owner in last_seen
                    else ""
                ),
                "components": [name for name, _ in components[owner].most_common(10)],
                "organizational_status_attested": owner in reviewed_active
                or owner in reviewed_inactive,
                **eligibility_record_fields(
                    eligibility_index.assignees.get(owner)
                    if eligibility_index is not None
                    else None
                ),
            }
        )

    technical_checks = {
        "eligible_history_available": bool(eligible),
        "active_candidates_nonempty": bool(active),
        "active_inactive_disjoint": not (active & inactive),
        "generic_accounts_excluded": not any(GENERIC_ASSIGNEE_RE.search(owner) for owner in active),
        "all_active_have_evidence_or_review": all(
            active_counts[owner] >= minimum_active_assignments or owner in reviewed_active
            for owner in active
        ),
        "assignment_label_times_explicit": proxy_label_times == 0,
    }
    technical_review_confirmed = all(technical_checks.values())
    review_confirmed = bool(
        organizational_review_confirmed and reviewer and technical_review_confirmed
    )
    eligibility_checks = {
        "review_file_present": eligibility_index is not None,
        "reviewed_eligible_assignees_nonempty": bool(eligibility_owners),
        "all_reviewed_eligible_assignees_active": bool(eligibility_owners)
        and eligibility_owners <= active,
        "review_interval_valid_at_snapshot": eligibility_index is not None
        and eligibility_index.is_valid_at(as_of),
    }
    eligibility_review_confirmed = all(eligibility_checks.values())
    roster = {
        "schema_version": 1,
        "source": "time_safe_assignment_activity_proxy_plus_optional_maintainer_overrides",
        "snapshot_at": as_of.isoformat(),
        "expires_at": (as_of + timedelta(days=valid_days)).isoformat(),
        "review_confirmed": review_confirmed,
        "technical_review_confirmed": technical_review_confirmed,
        "organizational_review_confirmed": bool(organizational_review_confirmed),
        "reviewer": reviewer,
        "eligibility_review_confirmed": eligibility_review_confirmed,
        "eligibility_reviewer": (
            eligibility_index.reviewer if eligibility_index is not None else ""
        ),
        "eligibility_valid_from": (
            eligibility_index.valid_from.isoformat()
            if eligibility_index is not None
            else None
        ),
        "eligibility_expires_at": (
            eligibility_index.expires_at.isoformat()
            if eligibility_index is not None
            else None
        ),
        "eligibility_definition": "reviewed_time_valid_capability_set_v1",
        "activity_definition": {
            "active_lookback_days": active_lookback_days,
            "inactive_lookback_days": inactive_lookback_days,
            "minimum_active_assignments": minimum_active_assignments,
            "label_must_be_available_by_snapshot": True,
        },
        "assignees": records,
        "candidates": sorted(active),
        "inactive": sorted(inactive),
        "cold_start": sorted(cold_start),
        "reviewed_active": sorted(reviewed_active),
        "reviewed_inactive": sorted(reviewed_inactive),
    }
    blockers = []
    if not technical_review_confirmed:
        blockers.append("technical_roster_checks_failed")
    if not organizational_review_confirmed:
        blockers.append("organizational_review_not_confirmed")
    if not reviewer:
        blockers.append("maintainer_reviewer_missing")
    eligibility_blockers = [
        name for name, passed in eligibility_checks.items() if not passed
    ]
    report = {
        "schema_version": 1,
        "method": "time_versioned_assignee_roster_audit_v1",
        "snapshot_at": as_of.isoformat(),
        "eligible_rows": len(eligible),
        "future_created_rows_excluded": future_created,
        "future_label_rows_excluded": unavailable_labels,
        "explicit_assignment_label_time_rows": explicit_label_times,
        "last_change_time_proxy_rows": proxy_label_times,
        "missing_component_rows": missing_components,
        "owners_with_history": len(total),
        "active_candidates": len(active),
        "inactive_candidates": len(inactive),
        "cold_start_candidates": len(cold_start),
        "owner_frequency_buckets": {
            "1-4": sum(1 for count in total.values() if count < 5),
            "5-19": sum(1 for count in total.values() if 5 <= count < 20),
            "20-99": sum(1 for count in total.values() if 20 <= count < 100),
            "100+": sum(1 for count in total.values() if count >= 100),
        },
        "component_coverage": {
            "components": len(
                {
                    str(row.get("component") or "unknown")
                    for row in eligible
                }
            ),
            "rows_with_known_component": len(eligible) - missing_components,
            "coverage": round(
                (len(eligible) - missing_components) / len(eligible), 6
            )
            if eligible
            else 0.0,
        },
        "technical_checks": technical_checks,
        "technical_review_confirmed": technical_review_confirmed,
        "organizational_review_confirmed": bool(organizational_review_confirmed),
        "review_confirmed": review_confirmed,
        "blocked_by": blockers,
        "eligibility_checks": eligibility_checks,
        "eligibility_review_confirmed": eligibility_review_confirmed,
        "eligibility_blocked_by": eligibility_blockers,
    }
    return roster, report


def load_history(
    paths: list[Path], *, label_map_path: Path, base_raw: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_map = read_label_map(label_map_path)
    availability = read_label_availability(base_raw)
    history: list[dict[str, Any]] = []
    sources = []
    for path in paths:
        loaded = apply_label_availability(load_rows(path, label_map), availability)
        unique, excluded = exclude_cross_split_overlap(
            loaded, history, strict_duplicate_families=True
        )
        history.extend(unique)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows_loaded": len(loaded),
                "rows_included": len(unique),
                "overlap_rows_excluded": excluded,
            }
        )
    history.sort(key=row_sort_key)
    return history, {"sources": sources, "rows": len(history)}


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_owner(value: Any) -> str:
    return str(value or "").strip().lower()


if __name__ == "__main__":
    main()

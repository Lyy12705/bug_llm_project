from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import (  # noqa: E402
    apply_assignee_aliases,
    load_assignee_alias_map,
    normalize_history_record,
    sha256_file,
    temporal_protocol_manifest,
)


ROLE_ORDER = {
    "ranker_fit": 0,
    "ranker_selection": 1,
    "calibrator_fit": 2,
    "open_set_fit": 3,
    "policy_selection": 4,
}
REQUIRED_ROLES = frozenset(ROLE_ORDER)
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
DEFAULT_REGISTRY = (
    SYSTEM_ROOT
    / "assignee_triage_accuracy"
    / "phase9_v3_protocol"
    / "v3_experiment_registry.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register a leakage-audited v3 assignee-routing protocol without "
            "opening or copying sealed-holdout labels."
        )
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--partition",
        action="append",
        required=True,
        metavar="NAME:ROLE=PATH",
        help=(
            "Ordered development partition. ROLE is ranker_fit, ranker_selection, "
            "calibrator_fit, open_set_fit, or policy_selection."
        ),
    )
    parser.add_argument("--assignee-alias-map", type=Path)
    parser.add_argument(
        "--protected-holdout-manifest",
        type=Path,
        action="append",
        default=[],
        help="Previously evaluated holdout manifest; its data is forbidden to v3 development.",
    )
    parser.add_argument("--planned-holdout-start", required=True)
    parser.add_argument("--planned-holdout-end", default="")
    parser.add_argument("--minimum-holdout-rows", type=int, default=2500)
    parser.add_argument("--minimum-auto-rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    specs = [parse_partition_spec(value) for value in args.partition]
    alias_map = load_assignee_alias_map(args.assignee_alias_map)
    protected = load_protected_holdouts(args.protected_holdout_manifest)
    partitions: dict[str, list[dict[str, Any]]] = {}
    roles: dict[str, str] = {}
    source_files: dict[str, Any] = {}
    for name, role, path in specs:
        resolved = path.resolve()
        reject_protected_partition(resolved, protected)
        rows, audit = load_development_partition(resolved, alias_map)
        partitions[name] = rows
        roles[name] = role
        source_files[name] = audit

    git_state = read_git_state(SYSTEM_ROOT)
    protocol = build_v3_protocol(
        experiment_id=args.experiment_id,
        partitions=partitions,
        roles=roles,
        source_files=source_files,
        protected_holdouts=protected,
        planned_holdout_start=args.planned_holdout_start,
        planned_holdout_end=args.planned_holdout_end,
        minimum_holdout_rows=args.minimum_holdout_rows,
        minimum_auto_rows=args.minimum_auto_rows,
        seed=args.seed,
        git_state=git_state,
    )
    canonical = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    protocol["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    ensure_registry_id_available(args.registry, args.experiment_id)
    write_new_json(args.output, protocol)
    append_registry(args.registry, protocol, args.output)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "protocol_gate_passed": protocol["protocol_gate"]["passed"],
                "blocked_by": protocol["protocol_gate"]["blocked_by"],
                "manifest_sha256": protocol["manifest_sha256"],
                "output": str(args.output.resolve()),
                "registry": str(args.registry.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_partition_spec(value: str) -> tuple[str, str, Path]:
    descriptor, separator, raw_path = value.partition("=")
    name, role_separator, role = descriptor.partition(":")
    name = name.strip()
    role = role.strip()
    raw_path = raw_path.strip()
    if (
        not separator
        or not role_separator
        or not SAFE_NAME_RE.fullmatch(name)
        or role not in ROLE_ORDER
        or not raw_path
    ):
        raise argparse.ArgumentTypeError(
            f"Expected NAME:ROLE=PATH with a supported role, got: {value}"
        )
    return name, role, Path(raw_path)


def load_development_partition(
    path: Path, aliases: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"development partition is missing: {path}")
    rows = []
    rejected_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(source, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            normalized = normalize_history_record(source)
            if normalized is None:
                rejected_rows += 1
                continue
            rows.append(normalized)
    rows, remapped = apply_assignee_aliases(rows, aliases)
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["ticket_id"])))
    ticket_ids = [str(row["ticket_id"]) for row in rows]
    if not rows:
        raise ValueError(f"development partition has no valid rows: {path}")
    return rows, {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(rows),
        "rejected_rows": rejected_rows,
        "alias_rows_remapped": remapped,
        "non_unique_ticket_ids": len(ticket_ids) - len(set(ticket_ids)),
    }


def load_protected_holdouts(paths: list[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid protected holdout manifest: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"protected holdout manifest must be an object: {path}")
        output_path = str(payload.get("output") or "")
        output.append(
            {
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "data_path": str((SYSTEM_ROOT / output_path).resolve())
                if output_path and not Path(output_path).is_absolute()
                else str(Path(output_path).resolve())
                if output_path
                else "",
                "data_sha256": str(payload.get("output_sha256") or ""),
                "start": payload.get("start_date") or payload.get("first_creation_time"),
                "end": payload.get("end_date") or payload.get("last_creation_time"),
                "sealed": payload.get("sealed_output") is True,
                "intended_use": "diagnosis_only_not_model_or_policy_selection",
                "labels_read_by_protocol_builder": False,
            }
        )
    return output


def reject_protected_partition(path: Path, protected: list[dict[str, Any]]) -> None:
    digest = sha256_file(path)
    for holdout in protected:
        if str(path) == holdout.get("data_path") or digest == holdout.get("data_sha256"):
            raise ValueError(
                "protected diagnosis-only holdout cannot be a v3 development partition: "
                f"{path}"
            )


def build_v3_protocol(
    *,
    experiment_id: str,
    partitions: dict[str, list[dict[str, Any]]],
    roles: dict[str, str],
    source_files: dict[str, Any],
    protected_holdouts: list[dict[str, Any]],
    planned_holdout_start: str,
    planned_holdout_end: str,
    minimum_holdout_rows: int,
    minimum_auto_rows: int,
    seed: int,
    git_state: dict[str, Any],
) -> dict[str, Any]:
    if not SAFE_NAME_RE.fullmatch(experiment_id):
        raise ValueError("experiment_id must be a safe 1-64 character identifier")
    if set(partitions) != set(roles) or len(partitions) != len(set(partitions)):
        raise ValueError("every unique partition must have exactly one role")
    role_values = list(roles.values())
    missing_roles = sorted(REQUIRED_ROLES - set(role_values))
    if missing_roles:
        raise ValueError("v3 protocol is missing roles: " + ", ".join(missing_roles))
    if role_values != sorted(role_values, key=ROLE_ORDER.__getitem__):
        raise ValueError("v3 partition roles are not in chronological fit/selection order")
    if role_values.count("policy_selection") < 2:
        raise ValueError("v3 requires at least two policy-selection windows")
    start = parse_datetime(planned_holdout_start)
    end = parse_datetime(planned_holdout_end) if planned_holdout_end else None
    if start is None or (end is not None and end <= start):
        raise ValueError("planned sealed holdout boundaries are invalid")
    if minimum_holdout_rows < 2500 or minimum_auto_rows < 250:
        raise ValueError("v3 release planning requires at least 2500 holdout and 250 auto rows")

    temporal = temporal_protocol_manifest(
        partitions,
        source="explicit_v3_development_partitions",
        seed=seed,
        experiment_id=experiment_id,
        git_commit=str(git_state.get("commit") or ""),
        partition_roles=roles,
    )
    development_end = max(
        parse_datetime(str(audit.get("end") or ""))
        for audit in temporal["partitions"].values()
    )
    assert development_end is not None
    label_audits = {
        name: audit["label_availability"]
        for name, audit in temporal["partitions"].items()
    }
    label_times_explicit = all(
        audit["last_change_time_proxy_rows"] == 0 and audit["missing_rows"] == 0
        for audit in label_audits.values()
    )
    checks = {
        "all_required_roles_present": not missing_roles,
        "two_policy_selection_windows": role_values.count("policy_selection") >= 2,
        "cross_split_leakage_free": bool(temporal["audit"]["leakage_free"]),
        "all_rows_normalized": all(
            audit["rejected_rows"] == 0 for audit in source_files.values()
        ),
        "unique_ticket_ids_within_partitions": all(
            audit["non_unique_ticket_ids"] == 0 for audit in source_files.values()
        ),
        "assignment_label_timestamps_explicit": label_times_explicit,
        "planned_holdout_is_later": start > development_end,
        "git_commit_recorded": bool(git_state.get("commit")),
        "git_worktree_clean": git_state.get("dirty") is False,
        "protected_holdouts_not_opened": all(
            holdout["labels_read_by_protocol_builder"] is False
            for holdout in protected_holdouts
        ),
    }
    return {
        "schema_version": 1,
        "artifact_type": "assignee_v3_experiment_protocol",
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "registered_development_protocol",
        "deployment_status": "research_only",
        "seed": seed,
        "git": git_state,
        "temporal_protocol": temporal,
        "source_files": source_files,
        "protected_holdouts": protected_holdouts,
        "sealed_holdout_plan": {
            "labels_accessible_to_training_or_policy_selection": False,
            "features_or_labels_opened": False,
            "planned_start": start.isoformat(),
            "planned_end": end.isoformat() if end else None,
            "minimum_rows": minimum_holdout_rows,
            "minimum_auto_rows": minimum_auto_rows,
            "one_formal_evaluation_only": True,
            "failure_transitions_to": "diagnosis_only_not_model_or_policy_selection",
        },
        "artifact_registry": {
            "ranker": {"status": "pending", "required_role": "ranker_fit"},
            "calibrator": {"status": "pending", "required_role": "calibrator_fit"},
            "open_set_detector": {"status": "pending", "required_role": "open_set_fit"},
            "routing_policy": {"status": "pending", "required_role": "policy_selection"},
            "drift_reference": {"status": "pending", "required_role": "policy_selection"},
            "sealed_holdout_report": {"status": "not_acquired"},
        },
        "protocol_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "blocked_by": [name for name, passed in checks.items() if not passed],
        },
        "limitations": [
            "A passed protocol gate only authorizes development; it is not model or deployment approval.",
            "Protected holdout manifests are registered by hash only; their data and labels are never opened here.",
            "A reviewed current roster, live shadow evidence, and explicit operator approval remain external gates.",
        ],
    }


def read_git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "dirty": True}
    return {"commit": commit, "dirty": bool(status)}


def append_registry(registry: Path, protocol: dict[str, Any], output: Path) -> None:
    ensure_registry_id_available(registry, str(protocol["experiment_id"]))
    entry = {
        "experiment_id": protocol["experiment_id"],
        "registered_at": protocol["created_at"],
        "status": protocol["status"],
        "protocol_gate_passed": protocol["protocol_gate"]["passed"],
        "manifest_path": str(output.resolve()),
        "manifest_sha256": protocol["manifest_sha256"],
        "git_commit": protocol["git"].get("commit"),
        "seed": protocol["seed"],
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def ensure_registry_id_available(registry: Path, experiment_id: str) -> None:
    if registry.exists():
        with registry.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid experiment registry JSONL at {registry}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"experiment registry row must be an object at {registry}:{line_number}"
                    )
                if row.get("experiment_id") == experiment_id:
                    raise ValueError(
                        f"experiment_id is already registered: {experiment_id}"
                    )


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"protocol output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    main()

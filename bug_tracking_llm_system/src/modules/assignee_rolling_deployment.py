from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.assignee_deployment import sha256_file


ROLLING_BUNDLE_SCHEMA_VERSION = 1
ROLLING_BUNDLE_TYPE = "assignee_rolling_deployment_bundle"
ROLLING_PATH_KEYS = {
    "assignee_dataset_path",
    "assignee_active_roster_path",
    "assignee_ranker_model_path",
    "assignee_ranker_artifact_path",
    "assignee_rolling_routing_artifact_path",
    "assignee_holdout_report_path",
    "assignee_shadow_report_path",
}


def load_rolling_deployment_bundle(
    path: Path,
    *,
    require_approved: bool = True,
    verify_integrity: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Load a self-contained rolling-LTR bundle and fail closed by default."""

    bundle_path = Path(path).resolve()
    payload = _read_object(bundle_path, "rolling deployment bundle")
    if payload.get("schema_version") != ROLLING_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported rolling deployment bundle schema")
    if payload.get("artifact_type") != ROLLING_BUNDLE_TYPE:
        raise ValueError("unsupported rolling deployment bundle type")
    config = payload.get("pipeline_config")
    if not isinstance(config, dict):
        raise ValueError("rolling deployment bundle must contain pipeline_config")
    if config.get("assignee_router_engine") != "rolling_ltr_v1":
        raise ValueError("rolling deployment bundle has an unsupported router engine")
    missing = sorted(key for key in ROLLING_PATH_KEYS if not config.get(key))
    if missing:
        raise ValueError(f"rolling deployment bundle is missing paths: {', '.join(missing)}")

    root = bundle_path.parent
    resolved = dict(config)
    for key in ROLLING_PATH_KEYS:
        candidate = Path(str(config[key]))
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not _is_relative_to(candidate, root):
            raise ValueError(f"rolling deployment bundle path escapes its directory: {key}")
        resolved[key] = candidate

    if verify_integrity:
        _verify_manifest(payload, resolved, root)
        _verify_cross_artifact_contract(resolved)
    if require_approved:
        _validate_approval(payload, now=now)
    return {**payload, "pipeline_config": resolved}


def _validate_approval(payload: dict[str, Any], *, now: datetime | None) -> None:
    if payload.get("deployment_status") != "approved":
        raise ValueError("rolling assignee deployment bundle is not approved")
    if payload.get("approval_gate_passed") is not True:
        raise ValueError("rolling assignee deployment approval gate did not pass")
    gates = payload.get("approval_gates")
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError("rolling deployment bundle has failed or incomplete approval gates")
    expires_at = _timestamp(payload.get("expires_at"))
    if expires_at is None:
        raise ValueError("approved rolling deployment bundle requires a valid expires_at")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if expires_at <= current.astimezone(timezone.utc):
        raise ValueError("rolling deployment bundle has expired")


def _verify_manifest(payload: dict[str, Any], config: dict[str, Any], root: Path) -> None:
    manifest = payload.get("artifact_manifest")
    if not isinstance(manifest, dict) or set(manifest) != ROLLING_PATH_KEYS:
        raise ValueError("rolling deployment artifact manifest is missing or incomplete")
    for key in sorted(ROLLING_PATH_KEYS):
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"invalid rolling deployment manifest entry: {key}")
        candidate = Path(config[key]).resolve()
        expected_path = str(candidate.relative_to(root))
        if entry.get("path") != expected_path:
            raise ValueError(f"rolling deployment artifact path mismatch: {key}")
        if not candidate.is_file():
            raise ValueError(f"rolling deployment artifact is missing: {key}")
        if entry.get("size_bytes") != candidate.stat().st_size:
            raise ValueError(f"rolling deployment artifact size mismatch: {key}")
        expected_hash = str(entry.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"rolling deployment artifact hash is invalid: {key}")
        if sha256_file(candidate) != expected_hash:
            raise ValueError(f"rolling deployment artifact hash mismatch: {key}")


def _verify_cross_artifact_contract(config: dict[str, Any]) -> None:
    roster = _read_object(Path(config["assignee_active_roster_path"]), "active roster")
    ranker = _read_object(Path(config["assignee_ranker_artifact_path"]), "ranker artifact")
    routing = _read_object(
        Path(config["assignee_rolling_routing_artifact_path"]), "rolling routing artifact"
    )
    holdout = _read_object(Path(config["assignee_holdout_report_path"]), "holdout report")
    shadow = _read_object(Path(config["assignee_shadow_report_path"]), "shadow report")
    if roster.get("schema_version") != 1:
        raise ValueError("rolling deployment roster has an unsupported schema")
    if routing.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise ValueError("rolling routing artifact has an unsupported type")
    if ranker.get("name") != routing.get("ranker_name"):
        raise ValueError("rolling ranker and routing artifacts do not match")
    if holdout.get("protocol", {}).get("routing_artifact_sha256") != sha256_file(
        Path(config["assignee_rolling_routing_artifact_path"])
    ):
        raise ValueError("holdout report was not produced by the bundled routing artifact")
    if holdout.get("deployment_gate", {}).get("passed") is not True:
        raise ValueError("bundled rolling holdout gate did not pass")
    if shadow.get("deployment_gate", {}).get("passed") is True and roster.get(
        "review_confirmed"
    ) is not True:
        raise ValueError("a passed shadow report requires a reviewed active roster")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

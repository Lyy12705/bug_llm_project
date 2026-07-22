from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import sha256_file  # noqa: E402
from modules.assignee_operational_guard import build_operational_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rollout health and emit a short-lived fail-closed operational state."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--shadow-report", type=Path, required=True)
    parser.add_argument("--active-roster", type=Path, required=True)
    parser.add_argument("--requested-rollout-percentage", type=int, default=0)
    parser.add_argument("--previous-state", type=Path)
    parser.add_argument("--minimum-stage-days", type=int, default=7)
    parser.add_argument("--state-ttl-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = read_object(args.bundle, "deployment bundle")
    shadow = read_object(args.shadow_report, "shadow report")
    roster = read_object(args.active_roster, "active roster")
    previous = read_object(args.previous_state, "previous operational state") if args.previous_state else None
    state = build_operational_state(
        bundle,
        shadow,
        roster,
        bundle_sha256=sha256_file(args.bundle),
        requested_rollout_percentage=args.requested_rollout_percentage,
        previous_state=previous,
        minimum_stage_days=args.minimum_stage_days,
        state_ttl_hours=args.state_ttl_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload


if __name__ == "__main__":
    main()

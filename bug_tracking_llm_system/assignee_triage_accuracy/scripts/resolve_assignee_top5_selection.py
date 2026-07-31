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

from modules.assignee_deployment import load_active_assignee_set  # noqa: E402
from modules.assignee_eligibility import load_assignee_eligibility_index  # noqa: E402
from modules.assignee_selection import (  # noqa: E402
    ASSIGN_TOP1,
    KEEP_MANUAL_TRIAGE,
    SELECT_TOP5_CANDIDATE,
    resolve_top5_assist_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve an explicit Top-5 user choice or an opt-in Top-1 assignment "
            "without treating it as autonomous routing."
        )
    )
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--active-roster", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=(SELECT_TOP5_CANDIDATE, ASSIGN_TOP1, KEEP_MANUAL_TRIAGE),
        required=True,
    )
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = _read_object(args.recommendation, "recommendation")
    recommendation = payload.get("assignee_recommendation") or payload
    if not isinstance(recommendation, dict):
        raise SystemExit("Recommendation payload must contain an object")
    active = load_active_assignee_set(args.active_roster)
    eligible = None
    if recommendation.get("eligibility_required") is True:
        try:
            eligibility_index = load_assignee_eligibility_index(args.active_roster)
        except ValueError as exc:
            raise SystemExit(f"Unable to verify assignee eligibility: {exc}") from exc
        ticket_context = payload.get("ticket_context") or payload.get("ticket") or {
            "product": recommendation.get("product"),
            "component": recommendation.get("component"),
        }
        eligible = eligibility_index.eligible_assignees_for_ticket(ticket_context)
    decision = resolve_top5_assist_selection(
        recommendation,
        action=args.action,
        selected_candidate_rank=args.selected_candidate_rank,
        active_assignees=active,
        eligible_assignees=eligible,
        reviewer=args.reviewer,
    )
    result = {
        "schema_version": 1,
        "ticket_id": str(payload.get("ticket_id") or ""),
        "real_assignment_changed": False,
        "assignment_adapter_action_required": decision["assignment_authorized"],
        "assignee_selection_decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_feedback import append_assignee_feedback, build_assignee_feedback_record  # noqa: E402
from modules.assignee_deployment import load_active_assignee_set  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the reviewed final assignee as append-only feedback.")
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--final-assignee", required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--feedback-origin",
        required=True,
        choices=("live_shadow", "historical_replay", "synthetic"),
        help="Evidence origin. Only provenance-complete live_shadow rows can pass deployment gates.",
    )
    parser.add_argument("--source-system", default="")
    parser.add_argument("--source-event-id", default="")
    parser.add_argument(
        "--active-roster",
        type=Path,
        default=None,
        help="Versioned active roster used to label known/unseen owner status for shadow evaluation.",
    )
    args = parser.parse_args()

    ticket = json.loads(args.ticket.read_text(encoding="utf-8"))
    prediction_payload = json.loads(args.prediction.read_text(encoding="utf-8"))
    prediction = prediction_payload.get("assignee_recommendation", prediction_payload)
    if not isinstance(ticket, dict) or not isinstance(prediction, dict):
        raise SystemExit("Ticket and prediction files must contain JSON objects.")
    if args.feedback_origin == "live_shadow" and (
        not args.source_system.strip() or not args.source_event_id.strip()
    ):
        raise SystemExit("live_shadow feedback requires --source-system and --source-event-id")
    active_roster = load_active_assignee_set(args.active_roster) if args.active_roster else set()
    final_owner = args.final_assignee.strip().lower()
    known_owner = final_owner in active_roster if args.active_roster else None
    record = build_assignee_feedback_record(
        ticket_json=ticket,
        prediction=prediction,
        final_assignee=args.final_assignee,
        reviewer=args.reviewer,
        source="reviewed_recommendation",
        notes=args.notes,
        known_owner=known_owner,
        owner_status=("known_active" if known_owner else "unseen_or_inactive") if args.active_roster else "",
        feedback_origin=args.feedback_origin,
        source_system=args.source_system,
        source_event_id=args.source_event_id,
    )
    append_assignee_feedback(args.feedback, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import config_from_dict  # noqa: E402
from modules.assignee_deployment import load_deployment_bundle  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend an assignee from a deployment bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bundle = load_deployment_bundle(args.bundle)
    ticket = json.loads(args.ticket.read_text(encoding="utf-8"))
    if not isinstance(ticket, dict):
        raise SystemExit("Ticket JSON must be an object.")
    config = config_from_dict({"project_root": SYSTEM_ROOT, **bundle["pipeline_config"]})
    result = AssigneeTriager(config=config).assign(
        ticket, {"predicted_priority": ticket.get("priority", "P3")}
    )
    payload = {
        "deployment_status": bundle.get("deployment_status", "unknown"),
        "ticket_id": ticket.get("ticket_id") or ticket.get("id"),
        "assignee_recommendation": result,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

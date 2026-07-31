from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_assignee_as_of_history import select_as_of_history  # noqa: E402


class AssigneeAsOfHistoryTests(unittest.TestCase):
    def test_only_labels_available_before_cutoff_are_included(self) -> None:
        rows = [
            {
                "ticket_id": "eligible",
                "assignee": "dev@example.com",
                "created_at": "2021-01-01T00:00:00Z",
                "label_available_at": "2021-01-02T00:00:00Z",
            },
            {
                "ticket_id": "late-label",
                "assignee": "dev@example.com",
                "created_at": "2021-01-01T00:00:00Z",
                "label_available_at": "2021-04-02T00:00:00Z",
            },
            {
                "ticket_id": "generic",
                "assignee": "nobody@example.com",
                "created_at": "2021-01-01T00:00:00Z",
                "label_available_at": "2021-01-02T00:00:00Z",
            },
        ]

        selected, audit = select_as_of_history(
            rows, datetime(2021, 4, 1, tzinfo=UTC)
        )

        self.assertEqual([row["ticket_id"] for row in selected], ["eligible"])
        self.assertEqual(audit["exclusions"]["label_unavailable_at_cutoff"], 1)
        self.assertEqual(audit["exclusions"]["missing_or_generic_owner"], 1)
        self.assertTrue(audit["temporal_order_valid"])


if __name__ == "__main__":
    unittest.main()

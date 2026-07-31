from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_assignee_roster_snapshot import build_roster_snapshot  # noqa: E402
from modules.assignee_eligibility import parse_assignee_eligibility_payload  # noqa: E402


class AssigneeRosterSnapshotTests(unittest.TestCase):
    def test_activity_roster_is_time_safe_but_not_organizationally_confirmed(self) -> None:
        rows = [
            {
                "assignee": "active@example.com",
                "component": "core",
                "created_at": f"2022-03-{day:02d}T00:00:00Z",
                "label_available_at": f"2022-03-{day + 1:02d}T00:00:00Z",
            }
            for day in (1, 5, 10)
        ]
        rows.extend(
            {
                "assignee": "old@example.com",
                "component": "legacy",
                "created_at": f"2020-01-{day:02d}T00:00:00Z",
                "label_available_at": f"2020-01-{day + 1:02d}T00:00:00Z",
            }
            for day in (1, 5, 10)
        )
        rows.append(
            {
                "assignee": "future-label@example.com",
                "component": "core",
                "created_at": "2022-03-01T00:00:00Z",
                "label_available_at": "2022-05-01T00:00:00Z",
            }
        )

        roster, report = build_roster_snapshot(
            rows,
            as_of=datetime(2022, 4, 1, tzinfo=UTC),
            active_lookback_days=180,
            inactive_lookback_days=365,
            minimum_active_assignments=3,
            valid_days=30,
            reviewed_active=set(),
            reviewed_inactive=set(),
            reviewer="",
            organizational_review_confirmed=False,
        )

        self.assertEqual(roster["candidates"], ["active@example.com"])
        self.assertEqual(roster["inactive"], ["old@example.com"])
        self.assertTrue(roster["technical_review_confirmed"])
        self.assertFalse(roster["review_confirmed"])
        self.assertEqual(report["future_label_rows_excluded"], 1)
        self.assertIn("organizational_review_not_confirmed", report["blocked_by"])

    def test_reviewed_override_can_add_cold_start_owner_and_complete_review(self) -> None:
        rows = [
            {
                "assignee": "history@example.com",
                "component": "core",
                "created_at": f"2022-03-{day:02d}T00:00:00Z",
                "label_available_at": f"2022-03-{day + 1:02d}T00:00:00Z",
            }
            for day in (1, 5, 10)
        ]
        roster, report = build_roster_snapshot(
            rows,
            as_of=datetime(2022, 4, 1, tzinfo=UTC),
            active_lookback_days=180,
            inactive_lookback_days=365,
            minimum_active_assignments=3,
            valid_days=30,
            reviewed_active={"new@example.com"},
            reviewed_inactive=set(),
            reviewer="maintainer@example.com",
            organizational_review_confirmed=True,
        )

        self.assertIn("new@example.com", roster["candidates"])
        self.assertEqual(roster["cold_start"], ["new@example.com"])
        self.assertTrue(roster["review_confirmed"])
        self.assertEqual(report["blocked_by"], [])

    def test_reviewed_capability_is_kept_separate_from_historical_components(self) -> None:
        rows = [
            {
                "assignee": "owner@example.com",
                "product": "firefox",
                "component": "historical-component",
                "created_at": f"2022-03-{day:02d}T00:00:00Z",
                "label_available_at": f"2022-03-{day + 1:02d}T00:00:00Z",
            }
            for day in (1, 5, 10)
        ]
        eligibility = parse_assignee_eligibility_payload(
            {
                "review_confirmed": True,
                "reviewer": "lead@example.com",
                "valid_from": "2022-03-01T00:00:00Z",
                "expires_at": "2022-05-01T00:00:00Z",
                "assignees": [
                    {
                        "assignee": "owner@example.com",
                        "eligible": True,
                        "permissions_confirmed": True,
                        "eligible_product_components": [
                            {"product": "firefox", "component": "reviewed-component"}
                        ],
                    }
                ],
            },
            require_active=False,
        )
        roster, report = build_roster_snapshot(
            rows,
            as_of=datetime(2022, 4, 1, tzinfo=UTC),
            active_lookback_days=180,
            inactive_lookback_days=365,
            minimum_active_assignments=3,
            valid_days=30,
            reviewed_active=set(),
            reviewed_inactive=set(),
            reviewer="maintainer@example.com",
            organizational_review_confirmed=True,
            eligibility_index=eligibility,
        )

        record = roster["assignees"][0]
        self.assertEqual(record["components"], ["historical-component"])
        self.assertEqual(
            record["eligible_product_components"],
            [{"product": "firefox", "component": "reviewed-component"}],
        )
        self.assertTrue(roster["eligibility_review_confirmed"])
        self.assertEqual(report["eligibility_blocked_by"], [])


if __name__ == "__main__":
    unittest.main()

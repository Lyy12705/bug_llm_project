from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_eligibility import (  # noqa: E402
    annotate_eligible_owner_labels,
    eligible_owner_metrics,
    load_assignee_eligibility_index,
)


class AssigneeEligibilityTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "schema_version": 1,
            "source": "maintainer_review",
            "review_confirmed": True,
            "eligibility_review_confirmed": True,
            "eligibility_reviewer": "lead@example.com",
            "eligibility_valid_from": "2026-07-01T00:00:00Z",
            "eligibility_expires_at": "2026-08-01T00:00:00Z",
            "assignees": [
                {
                    "assignee": "dom-a@example.com",
                    "active": True,
                    "eligibility_reviewed": True,
                    "permissions_confirmed": True,
                    "eligible_product_components": [
                        {"product": "firefox", "component": "dom"}
                    ],
                },
                {
                    "assignee": "firefox-general@example.com",
                    "active": True,
                    "eligibility_reviewed": True,
                    "permissions_confirmed": True,
                    "eligible_products": ["firefox"],
                },
                {
                    "assignee": "no-permission@example.com",
                    "active": True,
                    "eligibility_reviewed": True,
                    "permissions_confirmed": False,
                    "eligible_product_components": [
                        {"product": "firefox", "component": "dom"}
                    ],
                },
            ],
        }

    def _index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            path.write_text(json.dumps(self._payload()), encoding="utf-8")
            return load_assignee_eligibility_index(
                path, as_of="2026-07-22T00:00:00Z"
            )

    def test_reviewed_scope_and_permissions_define_eligible_set(self) -> None:
        index = self._index()
        eligible = index.eligible_assignees_for_ticket(
            {"product": "Firefox", "component": "DOM"},
            at_time="2026-07-22T00:00:00Z",
        )
        self.assertEqual(
            eligible,
            {"dom-a@example.com", "firefox-general@example.com"},
        )
        networking = index.eligible_assignees_for_ticket(
            {"product": "Firefox", "component": "Networking"},
            at_time="2026-07-22T00:00:00Z",
        )
        self.assertEqual(networking, {"firefox-general@example.com"})
        self.assertNotIn("no-permission@example.com", eligible)

    def test_expired_or_unconfirmed_review_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            payload = self._payload()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid"):
                load_assignee_eligibility_index(
                    path, as_of="2026-09-01T00:00:00Z"
                )
            payload["eligibility_review_confirmed"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not confirmed"):
                load_assignee_eligibility_index(
                    path, as_of="2026-07-22T00:00:00Z"
                )

    def test_eligible_labels_do_not_overwrite_exact_owner_correctness(self) -> None:
        rows = [
            {
                "product": "firefox",
                "component": "dom",
                "created_at": "2026-07-22T00:00:00Z",
                "expected_assignee": "historical@example.com",
                "is_top1_correct": False,
                "is_top5_correct": False,
                "ranked_candidates": [
                    "dom-a@example.com",
                    "outsider-1@example.com",
                    "firefox-general@example.com",
                    "outsider-2@example.com",
                    "outsider-3@example.com",
                ],
            }
        ]
        annotate_eligible_owner_labels(rows, self._index())

        self.assertFalse(rows[0]["is_top5_correct"])
        self.assertTrue(rows[0]["is_top1_eligible_correct"])
        self.assertTrue(rows[0]["is_top5_eligible_correct"])
        self.assertEqual(rows[0]["eligible_top5_count"], 2)
        metrics = eligible_owner_metrics(rows)
        self.assertEqual(metrics["top5_eligible_hit_rate"], 1.0)
        self.assertEqual(metrics["top5_candidate_eligibility_precision"], 0.4)

    def test_explicit_prediction_time_controls_snapshot_validity(self) -> None:
        rows = [
            {
                "product": "firefox",
                "component": "dom",
                "created_at": "2026-07-22T00:00:00Z",
                "prediction_created_at": "2026-09-01T00:00:00Z",
                "ranked_candidates": ["dom-a@example.com"],
            }
        ]

        annotate_eligible_owner_labels(rows, self._index())

        self.assertFalse(rows[0]["eligibility_label_available"])
        self.assertFalse(rows[0]["is_top1_eligible_correct"])

    def test_missing_prediction_time_cannot_use_current_snapshot(self) -> None:
        rows = [
            {
                "product": "firefox",
                "component": "dom",
                "ranked_candidates": ["dom-a@example.com"],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            payload = self._payload()
            payload["eligibility_expires_at"] = "9999-12-31T23:59:59Z"
            path.write_text(json.dumps(payload), encoding="utf-8")
            index = load_assignee_eligibility_index(
                path, as_of="2026-07-22T00:00:00Z"
            )

        annotate_eligible_owner_labels(rows, index)

        self.assertFalse(rows[0]["eligibility_label_available"])
        self.assertFalse(rows[0]["is_top1_eligible_correct"])


if __name__ == "__main__":
    unittest.main()

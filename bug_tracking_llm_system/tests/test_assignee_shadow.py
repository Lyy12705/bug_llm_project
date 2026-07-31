from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_shadow import evaluate_shadow_feedback  # noqa: E402


class AssigneeShadowTests(unittest.TestCase):
    def test_eligible_policy_shadow_requires_verified_candidates_and_selection(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for index in range(100):
            rows.append(
                {
                    "ticket_id": f"ELIGIBLE-{index}",
                    "ticket_created_at": start.isoformat(),
                    "prediction_created_at": (
                        start + timedelta(minutes=index)
                    ).isoformat(),
                    "created_at": (
                        start + timedelta(minutes=index + 1)
                    ).isoformat(),
                    "feedback_origin": "live_shadow",
                    "source_system": "production",
                    "source_event_id": f"eligible-event-{index}",
                    "prediction_latency_ms": 100.0,
                    "predicted_assignee": "manual_triage",
                    "routing_status": "top5_candidate_selected",
                    "top5_assist_available": True,
                    "final_assignee": "dev-a@example.com",
                    "ranked_candidates": [
                        "dev-a@example.com",
                        "dev-b@example.com",
                        "dev-c@example.com",
                        "dev-d@example.com",
                        "dev-e@example.com",
                    ],
                    "eligibility_required": True,
                    "eligibility_review_confirmed": True,
                    "eligibility_verified": True,
                    "selected_assignee_eligibility_verified": True,
                }
            )

        report = evaluate_shadow_feedback(rows)
        self.assertTrue(
            report["top5_assist_gate"]["checks"][
                "eligible_policy_review_confirmed"
            ]
        )
        self.assertEqual(
            report["metrics"]["selected_assignee_eligibility_verified_rate"],
            1.0,
        )

        rows[0]["selected_assignee_eligibility_verified"] = False
        failed = evaluate_shadow_feedback(rows)
        self.assertFalse(failed["top5_assist_gate"]["passed"])
        self.assertFalse(
            failed["top5_assist_gate"]["checks"][
                "selected_assignee_eligibility_verified"
            ]
        )

    def test_top5_assist_shadow_gate_measures_user_confirmed_candidate_quality(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for index in range(120):
            correct = index < 108
            rows.append(
                {
                    "ticket_id": f"TOP5-{index}",
                    "ticket_created_at": start.isoformat(),
                    "prediction_created_at": (
                        start + timedelta(minutes=index)
                    ).isoformat(),
                    "created_at": (
                        start + timedelta(minutes=index + 1)
                    ).isoformat(),
                    "feedback_origin": "live_shadow",
                    "source_system": "production",
                    "source_event_id": f"top5-event-{index}",
                    "prediction_latency_ms": 100.0,
                    "predicted_assignee": "manual_triage",
                    "routing_status": "top5_user_confirmation",
                    "top5_assist_available": True,
                    "final_assignee": (
                        "dev-a@example.com" if correct else "outside@example.com"
                    ),
                    "ranked_candidates": [
                        "dev-a@example.com",
                        "dev-b@example.com",
                        "dev-c@example.com",
                        "dev-d@example.com",
                        "dev-e@example.com",
                    ],
                }
            )

        report = evaluate_shadow_feedback(rows)

        self.assertEqual(report["metrics"]["top5_assist_accuracy"], 0.9)
        self.assertTrue(report["top5_assist_gate"]["passed"])
        self.assertFalse(report["top5_assist_gate"]["auto_assignment_authorized"])

    def test_shadow_gate_passes_only_with_volume_accuracy_coverage_and_unseen_labels(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        unseen_indexes = set(range(100, 200))
        for index in range(500):
            auto = index < 100
            correct = not (auto and index < 10)
            rows.append(
                {
                    "event_type": "assignee_feedback",
                    "ticket_id": f"T-{index}",
                    "ticket_created_at": (start + timedelta(days=index % 30, minutes=index - 1)).isoformat(),
                    "created_at": (start + timedelta(days=index % 30, minutes=index)).isoformat(),
                    "prediction_created_at": (
                        start + timedelta(days=index % 30, minutes=index - 1)
                    ).isoformat(),
                    "feedback_origin": "live_shadow",
                    "source_system": "bugzilla-production",
                    "source_event_id": f"change-{index}",
                    "prediction_latency_ms": 125.0 + index % 10,
                    "component": "core" if index % 2 == 0 else "ui",
                    "predicted_assignee": "dev-a@example.com" if auto else "manual_triage",
                    "routing_status": "auto_assign" if auto else "manual_triage_low_confidence",
                    "decision_reason_code": (
                        "high_confidence_low_open_set_risk" if auto else "low_ranking_confidence"
                    ),
                    "final_assignee": "dev-a@example.com" if correct else "dev-b@example.com",
                    "known_owner": index not in unseen_indexes,
                    "ranked_candidates": ["dev-a@example.com", "dev-b@example.com"],
                }
            )

        report = evaluate_shadow_feedback(rows)

        self.assertTrue(report["deployment_gate"]["passed"])
        self.assertEqual(report["metrics"]["auto_assignment_accuracy"], 0.9)
        self.assertEqual(report["metrics"]["auto_assignment_coverage"], 0.2)
        self.assertEqual(report["metrics"]["unseen_auto_assignment_rate"], 0.0)
        self.assertGreaterEqual(report["metrics"]["auto_assignment_accuracy_ci95"]["lower"], 0.8)
        self.assertTrue(report["deployment_gate"]["checks"]["live_shadow_provenance_complete"])
        self.assertTrue(
            report["deployment_gate"]["checks"][
                "unseen_auto_assignment_rate_confidence_upper_bound"
            ]
        )
        self.assertTrue(
            report["deployment_gate"]["checks"]["consecutive_weeks_pass_primary_rates"]
        )

    def test_shadow_gate_requires_two_passing_weeks(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for index in range(500):
            rows.append(
                {
                    "ticket_id": f"same-week-{index}",
                    "ticket_created_at": start.isoformat(),
                    "prediction_created_at": (start + timedelta(minutes=index)).isoformat(),
                    "created_at": (start + timedelta(minutes=index + 1)).isoformat(),
                    "feedback_origin": "live_shadow",
                    "source_system": "production",
                    "source_event_id": f"event-{index}",
                    "prediction_latency_ms": 100.0,
                    "predicted_assignee": "dev@example.com" if index < 100 else "manual_triage",
                    "routing_status": "auto_assign" if index < 100 else "manual_triage_low_confidence",
                    "final_assignee": "dev@example.com",
                    "known_owner": index < 400,
                }
            )

        report = evaluate_shadow_feedback(rows)

        self.assertFalse(report["deployment_gate"]["passed"])
        self.assertFalse(
            report["deployment_gate"]["checks"]["consecutive_weeks_pass_primary_rates"]
        )

    def test_shadow_gate_requires_unseen_rate_confidence_upper_bound(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for index in range(500):
            rows.append(
                {
                    "ticket_id": f"confidence-{index}",
                    "ticket_created_at": start.isoformat(),
                    "prediction_created_at": (start + timedelta(days=index % 30)).isoformat(),
                    "created_at": (start + timedelta(days=index % 30, minutes=1)).isoformat(),
                    "feedback_origin": "live_shadow",
                    "source_system": "production",
                    "source_event_id": f"confidence-event-{index}",
                    "prediction_latency_ms": 100.0,
                    "predicted_assignee": "dev@example.com" if index < 100 else "manual_triage",
                    "routing_status": "auto_assign" if index < 100 else "manual_triage_low_confidence",
                    "final_assignee": "dev@example.com",
                    "known_owner": index < 450,
                }
            )

        report = evaluate_shadow_feedback(rows)

        self.assertEqual(report["metrics"]["unseen_auto_assignment_rate"], 0.0)
        self.assertFalse(report["deployment_gate"]["passed"])
        self.assertFalse(
            report["deployment_gate"]["checks"][
                "unseen_auto_assignment_rate_confidence_upper_bound"
            ]
        )

    def test_shadow_gate_counts_malformed_input_events(self) -> None:
        report = evaluate_shadow_feedback(
            [],
            invalid_input_events=2,
            maximum_invalid_event_rate=0.01,
        )

        self.assertEqual(report["metrics"]["raw_feedback_events"], 2)
        self.assertEqual(report["metrics"]["invalid_input_events"], 2)
        self.assertEqual(report["metrics"]["invalid_event_rate"], 1.0)
        self.assertFalse(report["deployment_gate"]["checks"]["invalid_event_rate"])

    def test_historical_replay_cannot_pass_as_live_shadow_evidence(self) -> None:
        rows = [
            {
                "ticket_id": "T-1",
                "created_at": "2026-01-02T00:00:00Z",
                "prediction_created_at": "2026-01-01T00:00:00Z",
                "feedback_origin": "historical_replay",
                "source_system": "offline-fixture",
                "source_event_id": "replay-1",
                "predicted_assignee": "dev@example.com",
                "routing_status": "auto_assign",
                "final_assignee": "dev@example.com",
                "known_owner": True,
            }
        ]

        report = evaluate_shadow_feedback(
            rows,
            minimum_rows=1,
            minimum_observation_days=1,
            minimum_auto_accuracy_lower_bound=0.0,
        )

        self.assertFalse(report["deployment_gate"]["passed"])
        self.assertFalse(report["deployment_gate"]["checks"]["live_shadow_provenance_complete"])
        self.assertEqual(report["metrics"]["feedback_origin_breakdown"], {"historical_replay": 1})

    def test_shadow_gate_fails_closed_when_known_owner_labels_are_missing(self) -> None:
        rows = [
            {
                "ticket_id": "T-1",
                "created_at": "2026-01-01T00:00:00Z",
                "predicted_assignee": "dev@example.com",
                "routing_status": "auto_assign",
                "final_assignee": "dev@example.com",
            }
        ]

        report = evaluate_shadow_feedback(
            rows,
            minimum_rows=1,
            minimum_observation_days=1,
            minimum_auto_accuracy_lower_bound=0.0,
        )

        self.assertFalse(report["deployment_gate"]["passed"])
        self.assertFalse(report["deployment_gate"]["checks"]["unseen_labels_available"])

    def test_shadow_evaluation_uses_latest_feedback_per_ticket(self) -> None:
        rows = [
            {
                "ticket_id": "T-1",
                "created_at": "2026-01-01T00:00:00Z",
                "predicted_assignee": "wrong@example.com",
                "routing_status": "auto_assign",
                "final_assignee": "right@example.com",
                "known_owner": True,
            },
            {
                "ticket_id": "T-1",
                "created_at": "2026-01-02T00:00:00Z",
                "predicted_assignee": "right@example.com",
                "routing_status": "auto_assign",
                "final_assignee": "right@example.com",
                "known_owner": True,
            },
        ]

        report = evaluate_shadow_feedback(
            rows,
            minimum_rows=1,
            minimum_observation_days=1,
            minimum_auto_accuracy_lower_bound=0.0,
        )

        self.assertEqual(report["metrics"]["rows"], 1)
        self.assertEqual(report["metrics"]["duplicate_or_superseded_events"], 1)
        self.assertEqual(report["metrics"]["auto_assignment_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

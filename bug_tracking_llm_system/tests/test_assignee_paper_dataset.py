from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "paper_grade" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prepare_paper_grade_dataset import collapse_duplicate_clusters  # noqa: E402
from evaluate_paper_grade_benchmarks import build_metrics, current_triager_candidates  # noqa: E402


class AssigneePaperDatasetTests(unittest.TestCase):
    def test_collapses_explicit_and_exact_content_clusters_to_earliest_ticket(self) -> None:
        rows = [
            {
                "ticket_id": "bmo_1",
                "title": "Crash",
                "description": "Same stack",
                "created_at": "2025-01-01T00:00:00Z",
                "duplicate_of": "",
            },
            {
                "ticket_id": "bmo_2",
                "title": "Crash",
                "description": "Same stack",
                "created_at": "2025-01-02T00:00:00Z",
                "duplicate_of": "",
            },
            {
                "ticket_id": "bmo_3",
                "title": "Different wording",
                "description": "Different details",
                "created_at": "2025-01-03T00:00:00Z",
                "duplicate_of": "bmo_1",
            },
            {
                "ticket_id": "bmo_4",
                "title": "Independent",
                "description": "Independent details",
                "created_at": "2025-01-04T00:00:00Z",
                "duplicate_of": "",
            },
        ]

        output, audit = collapse_duplicate_clusters(rows)

        self.assertEqual([row["ticket_id"] for row in output], ["bmo_1", "bmo_4"])
        self.assertEqual(audit["rows_removed"], 2)
        self.assertEqual(audit["explicit_duplicate_links"], 1)

    def test_paper_evaluator_uses_only_production_candidates(self) -> None:
        result = {
            "assignee": "manual_triage",
            "ranked_candidates": ["dev_1", "manual_triage"],
        }
        context = {"roster": {"dev_1", "dev_2"}}

        candidates = current_triager_candidates(result, context, 10)

        self.assertEqual(candidates, ["dev_1"])

    def test_recommendation_and_routing_metrics_are_separate(self) -> None:
        metrics = build_metrics(
            [
                {
                    "expected_assignee": "dev_1",
                    "predicted_assignee": "dev_1",
                    "routed_assignee": "manual_triage",
                    "is_top1_correct": True,
                    "is_routed_correct": False,
                    "is_hit_at_3": True,
                    "is_hit_at_5": True,
                    "is_hit_at_10": True,
                    "reciprocal_rank": 1.0,
                    "component": "core",
                    "needs_manual_triage": True,
                    "fallback_used": True,
                    "routing_status": "needs_manual_triage",
                    "fallback_reason": "uncalibrated_recommendation_only",
                    "error_type": "correct",
                }
            ],
            bootstrap_samples=0,
            seed=1,
        )

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["auto_assignment_coverage"], 0.0)
        self.assertEqual(metrics["end_to_end_routing_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()

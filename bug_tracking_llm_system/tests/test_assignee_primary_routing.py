from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from finalize_assignee_primary_routing import finalize_primary_only  # noqa: E402


class AssigneePrimaryRoutingTests(unittest.TestCase):
    def test_disables_failed_top3_without_changing_primary_thresholds(self) -> None:
        report = {
            "routing_policy": {"found": True, "t_high": 0.8, "t_low": 0.8},
            "routing_by_selection_window": {
                "q1": {
                    "auto_assignment_accuracy": 0.9,
                    "auto_assignment_coverage": 0.2,
                    "unseen_auto_assignment_rate": 0.01,
                },
                "q2": {
                    "auto_assignment_accuracy": 0.86,
                    "auto_assignment_coverage": 0.15,
                    "unseen_auto_assignment_rate": 0.04,
                },
            },
        }
        artifact = {
            "artifact_type": "assignee_rolling_open_set_routing_bundle",
            "routing_policy": {"found": True, "t_high": 0.8, "t_low": 0.8},
            "targets": {
                "target_auto_accuracy": 0.85,
                "minimum_auto_coverage": 0.1,
                "maximum_unseen_auto_rate": 0.05,
            },
        }

        derived_report, derived_artifact, manifest = finalize_primary_only(
            report,
            artifact,
            artifact_name="primary-v2",
            source_report_sha256="report-hash",
            source_artifact_sha256="artifact-hash",
        )

        self.assertTrue(derived_artifact["development_gate"]["passed"])
        self.assertEqual(derived_artifact["routing_policy"]["t_high"], 0.8)
        self.assertEqual(derived_artifact["routing_policy"]["t_low"], 0.8)
        self.assertEqual(derived_artifact["review_mode"], "manual_triage_only")
        self.assertFalse(manifest["thresholds_reoptimized"])
        self.assertFalse(derived_report["development_gate"]["top3_confirmation_enabled"])


if __name__ == "__main__":
    unittest.main()

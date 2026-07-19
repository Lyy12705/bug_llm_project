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

from modules.assignee_deployment import (  # noqa: E402
    build_roster,
    derive_ownership,
    load_assignee_set,
    load_deployment_bundle,
    normalize_history_record,
    temporal_split,
)


class AssigneeDeploymentTests(unittest.TestCase):
    def test_normalizes_common_issue_tracker_fields(self) -> None:
        row = normalize_history_record(
            {
                "id": 42,
                "summary": "Login fails",
                "body": "src/auth/session.py rejects a valid token",
                "assigned_to": "Dev@Example.com",
                "creation_time": "2026-01-01T00:00:00Z",
                "component": "Authentication",
                "files": ["src/auth/session.py"],
            }
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["ticket_id"], "42")
        self.assertEqual(row["assignee"], "dev@example.com")
        self.assertEqual(row["component"], "authentication")
        self.assertEqual(row["file_paths"], ["src/auth/session.py"])

    def test_temporal_split_does_not_perform_duplicate_detection(self) -> None:
        rows = []
        for index in range(10):
            rows.append(
                {
                    "ticket_id": str(index),
                    "title": "same failure" if index in {1, 8} else f"failure {index}",
                    "description": "same details" if index in {1, 8} else f"details {index}",
                    "assignee": "dev@example.com",
                    "component": "core",
                    "created_at": f"2026-01-{index + 1:02d}T00:00:00Z",
                }
            )

        train, validation, test, audit = temporal_split(rows, 0.6, 0.2)

        self.assertEqual(len(train) + len(validation) + len(test), 10)
        self.assertFalse(audit["duplicate_detection_performed"])
        self.assertTrue(audit["upstream_duplicate_filter_required"])
        self.assertEqual(audit["cross_split_content_overlap"], 1)
        self.assertTrue(audit["chronological_order_valid"])

    def test_roster_filters_generic_and_inactive_assignees(self) -> None:
        rows = [
            {"assignee": "active@example.com", "component": "core"},
            {"assignee": "active@example.com", "component": "core"},
            {"assignee": "departed@example.com", "component": "core"},
            {"assignee": "departed@example.com", "component": "core"},
            {"assignee": "nobody@mozilla.org", "component": "core"},
            {"assignee": "nobody@mozilla.org", "component": "core"},
        ]

        payload, roster = build_roster(rows, 2, {"departed@example.com"})

        self.assertEqual(roster, {"active@example.com"})
        self.assertEqual(payload["inactive"], ["departed@example.com"])

    def test_derived_ownership_is_marked_candidate_and_bundle_paths_resolve(self) -> None:
        rows = [
            {
                "assignee": "dev@example.com",
                "component": "auth",
                "file_paths": ["src/auth/session.py"],
            }
            for _ in range(3)
        ]
        component, files = derive_ownership(rows, {"dev@example.com"}, 3, 0.7)
        self.assertEqual(component["deployment_status"], "candidate_requires_owner_review")
        self.assertEqual(component["components"]["auth"]["owner"], "dev@example.com")
        self.assertEqual(files["files"]["src/auth/session.py"]["owner"], "dev@example.com")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "deployment_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "pipeline_config": {
                            "assignee_dataset_path": "history.jsonl",
                            "assignee_open_set_enabled": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_deployment_bundle(bundle_path)
        self.assertEqual(loaded["pipeline_config"]["assignee_dataset_path"], (root / "history.jsonl").resolve())

    def test_reads_inactive_assignee_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inactive.json"
            path.write_text(json.dumps({"inactive": [{"email": "OLD@Example.com"}]}), encoding="utf-8")
            owners = load_assignee_set(path)

        self.assertEqual(owners, {"old@example.com"})


if __name__ == "__main__":
    unittest.main()

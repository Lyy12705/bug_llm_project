from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from modules.assignee_deployment import (  # noqa: E402
    apply_assignee_aliases,
    artifact_manifest_entry,
    build_roster,
    canonicalize_assignee_set,
    derive_ownership,
    load_active_assignee_set,
    load_assignee_set,
    load_assignee_alias_map,
    load_deployment_bundle,
    normalize_history_record,
    sha256_file,
    split_temporal_partition,
    temporal_split,
    temporal_protocol_manifest,
    wilson_interval,
)
from modules.assignee_rolling_deployment import (  # noqa: E402
    load_rolling_deployment_bundle,
)
from prepare_assignee_deployment import prediction_metrics, selective_routing_gate  # noqa: E402
from prepare_assignee_rolling_deployment import (  # noqa: E402
    REQUIRED_HOLDOUT_CHECKS,
    roster_not_expired,
    rolling_deployment_gates,
    unavailable_roster,
    unavailable_shadow_report,
)
from recommend_assignee_rolling import fail_closed_payload, history_available_before  # noqa: E402


class AssigneeDeploymentTests(unittest.TestCase):
    def test_rolling_production_entrypoint_fails_closed_and_filters_future_labels(self) -> None:
        payload = fail_closed_payload(ValueError("bundle is research-only"))
        recommendation = payload["assignee_recommendation"]
        self.assertEqual(recommendation["assignee"], "manual_triage")
        self.assertFalse(recommendation["auto_assignment_authorized"])
        self.assertEqual(recommendation["open_set_risk"], 1.0)

        rows = [
            {
                "ticket_id": "past",
                "created_at": "2024-01-01T00:00:00Z",
                "label_available_at": "2024-01-02T00:00:00Z",
            },
            {
                "ticket_id": "future-label",
                "created_at": "2024-01-03T00:00:00Z",
                "label_available_at": "2024-03-01T00:00:00Z",
            },
        ]
        included, withheld = history_available_before(
            rows, datetime.fromisoformat("2024-02-01T00:00:00+00:00").timestamp()
        )
        self.assertEqual([row["ticket_id"] for row in included], ["past"])
        self.assertEqual(withheld, 1)

    def test_active_roster_loader_never_confuses_inactive_entries_with_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            path.write_text(
                json.dumps(
                    {
                        "candidates": ["active@example.com"],
                        "inactive": ["departed@example.com"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_active_assignee_set(path), {"active@example.com"})

            path.write_text(
                json.dumps(
                    {
                        "candidates": ["same@example.com"],
                        "inactive": ["same@example.com"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "marks candidates inactive"):
                load_active_assignee_set(path)

    def test_rolling_deployment_stays_research_only_without_roster_shadow_and_approval(self) -> None:
        routing_hash = "a" * 64
        routing = {
            "ranker_name": "ranker-v1",
            "development_gate": {"passed": True},
        }
        holdout = {
            "protocol": {
                "holdout_provenance": {"manifest_verified": True},
                "holdout_used_for_fitting": False,
                "holdout_used_for_threshold_selection": False,
                "routing_artifact_sha256": routing_hash,
            },
            "deployment_gate": {
                "passed": True,
                "score_drift_check": True,
                "checks": {name: True for name in REQUIRED_HOLDOUT_CHECKS},
            },
        }

        gates = rolling_deployment_gates(
            routing,
            {"name": "ranker-v1"},
            holdout,
            unavailable_roster(),
            unavailable_shadow_report(),
            operator_approved=False,
            bundled_routing_hash=routing_hash,
            history_rows=100,
        )

        self.assertTrue(gates["new_holdout_gate_passed"])
        self.assertTrue(gates["all_required_holdout_checks_passed"])
        self.assertFalse(gates["active_roster_review_confirmed"])
        self.assertFalse(gates["active_roster_not_expired"])
        self.assertFalse(gates["shadow_gate_passed"])
        self.assertFalse(gates["explicit_operator_approval"])
        self.assertFalse(all(gates.values()))

    def test_roster_expiration_is_a_hard_deployment_gate(self) -> None:
        now = datetime.fromisoformat("2026-07-20T00:00:00+00:00")
        self.assertFalse(
            roster_not_expired({"expires_at": "2026-07-19T23:59:59Z"}, now=now)
        )
        self.assertTrue(
            roster_not_expired({"expires_at": "2026-07-21T00:00:00Z"}, now=now)
        )
        self.assertFalse(roster_not_expired({}, now=now))

    def test_rolling_bundle_loader_verifies_integrity_and_cross_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "assignee_dataset_path": root / "history.jsonl",
                "assignee_active_roster_path": root / "roster.json",
                "assignee_ranker_model_path": root / "ranker.joblib",
                "assignee_ranker_artifact_path": root / "ranker.json",
                "assignee_rolling_routing_artifact_path": root / "routing.json",
                "assignee_holdout_report_path": root / "holdout.json",
                "assignee_shadow_report_path": root / "shadow.json",
            }
            files["assignee_dataset_path"].write_text("{}\n", encoding="utf-8")
            files["assignee_ranker_model_path"].write_bytes(b"model")
            files["assignee_active_roster_path"].write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_confirmed": False,
                        "assignees": [],
                        "candidates": [],
                        "inactive": [],
                    }
                ),
                encoding="utf-8",
            )
            files["assignee_ranker_artifact_path"].write_text(
                json.dumps({"name": "ranker-v1"}), encoding="utf-8"
            )
            files["assignee_rolling_routing_artifact_path"].write_text(
                json.dumps(
                    {
                        "artifact_type": "assignee_rolling_open_set_routing_bundle",
                        "ranker_name": "ranker-v1",
                    }
                ),
                encoding="utf-8",
            )
            routing_hash = sha256_file(files["assignee_rolling_routing_artifact_path"])
            files["assignee_holdout_report_path"].write_text(
                json.dumps(
                    {
                        "protocol": {"routing_artifact_sha256": routing_hash},
                        "deployment_gate": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            files["assignee_shadow_report_path"].write_text(
                json.dumps({"deployment_gate": {"passed": False}}), encoding="utf-8"
            )
            bundle = root / "deployment_bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "assignee_rolling_deployment_bundle",
                        "deployment_status": "research_only",
                        "approval_gate_passed": False,
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approval_gates": {"active_roster_review_confirmed": False},
                        "pipeline_config": {
                            "assignee_router_engine": "rolling_ltr_v1",
                            **{key: path.name for key, path in files.items()},
                        },
                        "artifact_manifest": {
                            key: artifact_manifest_entry(path, root) for key, path in files.items()
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_rolling_deployment_bundle(bundle, require_approved=False)
            self.assertEqual(
                loaded["pipeline_config"]["assignee_ranker_model_path"],
                files["assignee_ranker_model_path"].resolve(),
            )
            with self.assertRaisesRegex(ValueError, "not approved"):
                load_rolling_deployment_bundle(bundle)

            files["assignee_ranker_model_path"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                load_rolling_deployment_bundle(bundle, require_approved=False)

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

    def test_reviewed_roster_can_add_cold_start_owner_but_never_reactivate_inactive_owner(self) -> None:
        payload, roster = build_roster(
            [{"assignee": "history@example.com", "component": "core"}] * 2,
            2,
            {"departed@example.com"},
            {"cold-start@example.com", "departed@example.com"},
        )

        self.assertEqual(roster, {"history@example.com", "cold-start@example.com"})
        statuses = {row["assignee"]: row["status"] for row in payload["assignees"]}
        self.assertEqual(statuses["cold-start@example.com"], "active_reviewed_roster")
        self.assertEqual(payload["reviewed_active"], ["cold-start@example.com"])

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
            loaded = load_deployment_bundle(bundle_path, require_approved=False)
        self.assertEqual(loaded["pipeline_config"]["assignee_dataset_path"], (root / "history.jsonl").resolve())

    def test_reads_inactive_assignee_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inactive.json"
            path.write_text(json.dumps({"inactive": [{"email": "OLD@Example.com"}]}), encoding="utf-8")
            owners = load_assignee_set(path)

        self.assertEqual(owners, {"old@example.com"})

    def test_alias_map_resolves_chains_and_rejects_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aliases.json"
            path.write_text(
                json.dumps({"aliases": {"old@example.com": "mid@example.com", "mid@example.com": "new@example.com"}}),
                encoding="utf-8",
            )
            aliases = load_assignee_alias_map(path)
            rows, remapped = apply_assignee_aliases(
                [{"assignee": "old@example.com"}, {"assignee": "other@example.com"}], aliases
            )
            self.assertEqual(rows[0]["assignee"], "new@example.com")
            self.assertEqual(remapped, 1)
            self.assertEqual(
                canonicalize_assignee_set({"old@example.com", "other@example.com"}, aliases),
                {"new@example.com", "other@example.com"},
            )

            path.write_text(json.dumps({"a": "b", "b": "a"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cyclic"):
                load_assignee_alias_map(path)

    def test_temporal_subsplit_and_manifest_are_leakage_auditable(self) -> None:
        rows = [
            {
                "ticket_id": str(index),
                "title": f"failure {index}",
                "description": f"details {index}",
                "assignee": "dev@example.com",
                "created_at": f"2026-01-{index + 1:02d}T00:00:00Z",
            }
            for index in range(8)
        ]
        first, second, audit = split_temporal_partition(rows, 0.5)
        manifest = temporal_protocol_manifest(
            {"calibration_fit": first, "policy_selection": second},
            source="unit-test",
        )

        self.assertTrue(audit["chronological_order_valid"])
        self.assertTrue(manifest["audit"]["leakage_free"])
        self.assertEqual(manifest["partition_order"], ["calibration_fit", "policy_selection"])
        self.assertEqual(len(manifest["partitions"]["calibration_fit"]["rows_sha256"]), 64)

    def test_wilson_interval_reports_uncertainty_for_small_samples(self) -> None:
        interval = wilson_interval(9, 10)

        self.assertLess(interval["lower"], 0.90)
        self.assertGreater(interval["upper"], 0.90)
        self.assertEqual(wilson_interval(0, 0), {"lower": 0.0, "upper": 1.0})

    def test_deployment_prediction_metrics_include_component_safety_metrics(self) -> None:
        predictions = [
            {
                "ticket_id": f"T-{index}",
                "component": "fragile" if index < 2 else "stable",
                "expected": "dev@example.com",
                "candidates": ["dev@example.com"],
                "raw_confidence": 0.9,
                "correct": index != 1,
                "top3_correct": True,
                "known_owner": True,
                "owner_status": "known_active",
                "predicted_owner_active": True,
                "candidate_source_count": 2,
                "open_set_risk": 0.1,
            }
            for index in range(4)
        ]

        metrics = prediction_metrics(
            predictions,
            {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]},
            {
                "t_high": 0.8,
                "t_low": 0.5,
                "open_set_threshold": 0.5,
                "minimum_candidate_source_count": 2,
            },
        )

        self.assertEqual(metrics["component_metrics"]["fragile"]["auto_assignment_rows"], 2)
        self.assertEqual(metrics["component_metrics"]["fragile"]["auto_assignment_accuracy"], 0.5)

    def test_deployment_gate_rejects_unsafe_component_when_global_rates_pass(self) -> None:
        metrics = {
            "rows": 500,
            "auto_assignment_rows": 100,
            "auto_assignment_accuracy": 0.90,
            "auto_assignment_coverage": 0.20,
            "unseen_auto_assignment_rate": 0.0,
            "auto_assignment_accuracy_ci95": {"lower": 0.82, "upper": 0.95},
            "component_metrics": {
                "fragile": {
                    "rows": 60,
                    "auto_assignment_rows": 30,
                    "auto_assignment_coverage": 0.50,
                    "auto_assignment_accuracy": 0.70,
                    "auto_assignment_accuracy_ci95": {"lower": 0.52, "upper": 0.83},
                }
            },
        }

        gate = selective_routing_gate(
            metrics,
            target_accuracy=0.85,
            minimum_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            minimum_rows=500,
            minimum_auto_rows=100,
            minimum_accuracy_lower_bound=0.80,
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["component_accuracy_floor"])
        self.assertFalse(gate["checks"]["component_accuracy_confidence_floor"])
        self.assertIn("fragile", gate["component_failures"])

        metrics_without_components = dict(metrics)
        metrics_without_components.pop("component_metrics")
        missing_component_gate = selective_routing_gate(
            metrics_without_components,
            target_accuracy=0.85,
            minimum_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            minimum_rows=500,
            minimum_auto_rows=100,
            minimum_accuracy_lower_bound=0.80,
        )
        self.assertFalse(missing_component_gate["passed"])
        self.assertFalse(missing_component_gate["checks"]["component_metrics_available"])

    def test_approved_bundle_verifies_status_expiry_containment_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "assignee_dataset_path": root / "history.jsonl",
                "assignee_active_roster_path": root / "roster.json",
                "assignee_routing_policy_path": root / "policy.json",
                "assignee_calibration_artifact_path": root / "calibrator.json",
            }
            for path in files.values():
                path.write_text("{}\n", encoding="utf-8")
            bundle_path = root / "deployment_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "artifact_type": "assignee_deployment_bundle",
                        "deployment_status": "approved",
                        "approval_gate_passed": True,
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approval_gates": {"frozen_test_gate_passed": True},
                        "pipeline_config": {key: path.name for key, path in files.items()},
                        "artifact_manifest": {
                            key: artifact_manifest_entry(path, root) for key, path in files.items()
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_deployment_bundle(bundle_path)
            self.assertEqual(
                loaded["pipeline_config"]["assignee_dataset_path"],
                files["assignee_dataset_path"].resolve(),
            )

            files["assignee_dataset_path"].write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size mismatch|hash mismatch"):
                load_deployment_bundle(bundle_path)

    def test_bundle_loader_rejects_research_and_escaping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "deployment_bundle.json"
            bundle_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "deployment_status": "research_only",
                        "pipeline_config": {"assignee_dataset_path": "../outside.jsonl"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not approved"):
                load_deployment_bundle(bundle_path)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_deployment_bundle(bundle_path, require_approved=False)


if __name__ == "__main__":
    unittest.main()

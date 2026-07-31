from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from modules.assignee_deployment import temporal_protocol_manifest  # noqa: E402

from assignee_open_set_common import select_multi_window_routing_policy  # noqa: E402
from build_assignee_temporal_protocol import (  # noqa: E402
    build_v3_protocol,
    reject_protected_partition,
)
from plan_assignee_holdout_power import (  # noqa: E402
    holdout_sample_plan,
    minimum_continuous_wilson_sample_size,
)
from train_assignee_ltr import candidate_source_families_from_features  # noqa: E402


def row(
    ticket: str,
    timestamp: str,
    *,
    title: str | None = None,
    duplicate_family: str = "",
    owner_event_id: str = "",
) -> dict[str, object]:
    return {
        "ticket_id": ticket,
        "title": title or f"Unique routing failure for ticket {ticket}",
        "description": f"Detailed reproducible context belonging only to {ticket}",
        "product": "core",
        "component": "dom",
        "assignee": "dev@example.com",
        "created_at": timestamp,
        "label_available_at": timestamp,
        "label_availability_source": "explicit_assignment_event_time",
        "duplicate_family": duplicate_family,
        "owner_event_id": owner_event_id,
    }


class AssigneeV3ProtocolTests(unittest.TestCase):
    def test_temporal_protocol_detects_all_new_cross_split_leakage_types(self) -> None:
        first = [
            row(
                "A-1",
                "2024-01-01T00:00:00Z",
                title="Authentication session token validation crashes after profile save",
                duplicate_family="ROOT-1",
                owner_event_id="OWNER-EVENT-1",
            )
        ]
        second = [
            row(
                "B-1",
                "2024-02-01T00:00:00Z",
                title="Authentication session token validation fails after profile save",
                duplicate_family="ROOT-1",
                owner_event_id="OWNER-EVENT-1",
            )
        ]

        manifest = temporal_protocol_manifest(
            {"ranker_fit": first, "ranker_selection": second}, source="unit-test"
        )

        audit = manifest["audit"]
        self.assertGreater(audit["cross_split_duplicate_family_overlap"], 0)
        self.assertGreater(audit["cross_split_owner_event_overlap"], 0)
        self.assertGreater(audit["cross_split_near_duplicate_text_overlap"], 0)
        self.assertFalse(audit["leakage_free"])

    def test_v3_protocol_passes_only_with_separate_roles_and_explicit_label_times(self) -> None:
        names_and_roles = [
            ("ranker-fit", "ranker_fit"),
            ("ranker-selection", "ranker_selection"),
            ("calibrator-fit", "calibrator_fit"),
            ("open-set-fit", "open_set_fit"),
            ("policy-one", "policy_selection"),
            ("policy-two", "policy_selection"),
        ]
        partitions = {}
        for index, (name, _) in enumerate(names_and_roles):
            item = row(
                f"T-{index}",
                f"2024-0{index + 1}-01T00:00:00Z",
                title=(
                    f"Subsystem{index} signature{index} anomaly{index} "
                    f"workflow{index} regression{index}"
                ),
            )
            item["description"] = (
                f"Evidence{index} reproduction{index} trace{index} "
                f"context{index} boundary{index}"
            )
            partitions[name] = [item]
        roles = dict(names_and_roles)
        sources = {
            name: {
                "rejected_rows": 0,
                "non_unique_ticket_ids": 0,
                "sha256": str(index) * 64,
            }
            for index, name in enumerate(partitions, start=1)
        }

        protocol = build_v3_protocol(
            experiment_id="v3-unit-test",
            partitions=partitions,
            roles=roles,
            source_files=sources,
            protected_holdouts=[],
            planned_holdout_start="2024-07-01T00:00:00Z",
            planned_holdout_end="2024-09-30T00:00:00Z",
            minimum_holdout_rows=2500,
            minimum_auto_rows=250,
            seed=3407,
            git_state={"commit": "a" * 40, "dirty": False},
        )

        self.assertTrue(protocol["protocol_gate"]["passed"])
        self.assertFalse(
            protocol["sealed_holdout_plan"][
                "labels_accessible_to_training_or_policy_selection"
            ]
        )

    def test_declared_last_change_proxy_is_not_upgraded_by_normalization(self) -> None:
        from modules.assignee_deployment import normalize_history_record

        normalized = normalize_history_record(
            {
                "ticket_id": "T-proxy",
                "title": "Legacy proxy row",
                "description": "The timestamp value exists but is not an assignment event.",
                "assignee": "dev@example.com",
                "created_at": "2024-01-01T00:00:00Z",
                "label_available_at": "2024-01-02T00:00:00Z",
                "last_change_time": "2024-01-02T00:00:00Z",
                "label_availability_source": "last_change_time_proxy",
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(
            normalized["label_availability_source"], "last_change_time_proxy"
        )

    def test_protected_holdout_hash_cannot_be_used_as_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdout.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            from modules.assignee_deployment import sha256_file

            with self.assertRaisesRegex(ValueError, "protected diagnosis-only holdout"):
                reject_protected_partition(
                    path.resolve(),
                    [{"data_path": "", "data_sha256": sha256_file(path)}],
                )

    def test_multi_window_policy_rejects_safe_point_estimate_with_unsafe_ci(self) -> None:
        windows = {}
        for name in ("q1", "q2"):
            rows = []
            for index in range(1000):
                known = index < 900
                low_unknown_risk = not known and index < 904
                rows.append(
                    {
                        "known_owner": known,
                        "is_top1_correct": known,
                        "is_top3_correct": known,
                        "calibrated_probability": 0.95,
                        "open_set_probability": 0.1 if known or low_unknown_risk else 0.9,
                        "candidate_source_count": 2,
                        "component": "core",
                    }
                )
            windows[name] = rows

        point_only = select_multi_window_routing_policy(
            windows,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            target_review_accuracy=0.90,
            minimum_candidate_source_count=2,
        )
        strict = select_multi_window_routing_policy(
            windows,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            target_review_accuracy=0.90,
            minimum_candidate_source_count=2,
            maximum_unseen_rate_upper_bound=0.05,
        )

        self.assertTrue(point_only["found"])
        self.assertEqual(
            point_only["selection_window_metrics"]["q1"][
                "unseen_auto_assignment_rate"
            ],
            0.04,
        )
        self.assertFalse(strict["found"])

    def test_correlated_component_signals_count_as_one_source_family(self) -> None:
        component_only = candidate_source_families_from_features(
            {
                "source_product_component": 1.0,
                "source_component": 1.0,
                "source_component_token": 1.0,
            }
        )
        with_text = candidate_source_families_from_features(
            {
                "source_product_component": 1.0,
                "source_component": 1.0,
                "source_bm25": 1.0,
            }
        )

        self.assertEqual(component_only, {"component_history"})
        self.assertEqual(with_text, {"component_history", "lexical_retrieval"})

    def test_holdout_planner_uses_two_sided_wilson_gate(self) -> None:
        self.assertEqual(minimum_continuous_wilson_sample_size(0.04, 0.05), 1825)
        self.assertEqual(minimum_continuous_wilson_sample_size(0.02, 0.05), 203)
        plan = holdout_sample_plan(
            expected_error_rate=0.02,
            maximum_upper_bound=0.05,
            expected_unseen_fraction=0.10,
            minimum_total_rows=2500,
        )
        self.assertEqual(plan["minimum_unseen_owner_rows"], 203)
        self.assertEqual(plan["recommended_total_holdout_rows"], 2500)


if __name__ == "__main__":
    unittest.main()

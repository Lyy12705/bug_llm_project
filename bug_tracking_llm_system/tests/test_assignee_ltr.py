from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_assignee_ltr import (  # noqa: E402
    CandidateIndex,
    build_semantic_backend,
    build_temporal_training_examples,
    canonicalize_source_rows,
    candidate_evidence_support_ablation,
    clean_description,
    exclude_cross_split_overlap,
    fit_rankers,
    pairwise_training_data,
    source_quotas,
)
from calibrate_assignee_ltr import (  # noqa: E402
    route_predictions,
    routing_metrics,
    scored_predictions,
    select_threshold,
)
from assignee_open_set_common import (  # noqa: E402
    apply_top5_assist_policy,
    build_leave_assignee_out_folds,
    build_drift_reference,
    evaluate_score_drift,
    deployment_gate,
    fit_portable_logistic,
    predict_portable_logistic,
    route_predictions as route_open_set_predictions,
    routing_metrics as open_set_routing_metrics,
    routing_score_diagnostics,
    select_routing_policy,
    select_multi_window_routing_policy,
    select_multi_window_top5_assist_policy,
    top5_assist_gate,
    top5_assist_metrics,
    wilson_interval as open_set_wilson_interval,
)
from train_assignee_rolling_open_set import (  # noqa: E402
    label_availability_audit,
    prepare_rolling_windows,
)
from evaluate_assignee_rolling_holdout import (  # noqa: E402
    claim_sealed_holdout,
    complete_sealed_holdout_claim,
    parse_named_path,
    read_registry,
    sha256_file,
    validate_holdout_manifest,
    validate_registered_replay,
    validate_window_names,
)
from compare_assignee_rolling_replay import compare_predictions  # noqa: E402


class AssigneeLtrTests(unittest.TestCase):
    def test_bugzilla_rows_preserve_explicit_assignment_label_timestamp(self) -> None:
        rows = canonicalize_source_rows(
            [
                {
                    "id": 42,
                    "summary": "Explicit assignment event",
                    "description": "Owner event provenance must survive normalization.",
                    "assigned_to": "dev@example.com",
                    "creation_time": "2024-01-01T00:00:00Z",
                    "last_change_time": "2024-02-01T00:00:00Z",
                    "assignment_label_available_at": "2024-01-02T00:00:00Z",
                }
            ]
        )

        self.assertEqual(
            rows[0]["label_available_at"], "2024-01-02T00:00:00Z"
        )
        self.assertEqual(
            rows[0]["label_availability_source"],
            "explicit_assignment_event_time",
        )

    def test_label_availability_audit_fails_closed_on_proxy_or_missing_rows(self) -> None:
        explicit = {
            "ticket_id": "T-1",
            "label_availability_source": "explicit_assignment_event_time",
        }
        safe = label_availability_audit([explicit, explicit])
        unsafe = label_availability_audit(
            [
                explicit,
                {
                    "ticket_id": "T-2",
                    "label_availability_source": "last_change_time_proxy",
                },
                {"ticket_id": "T-3"},
            ]
        )

        self.assertTrue(safe["all_rows_explicit"])
        self.assertEqual(safe["rows"], 1)
        self.assertFalse(unsafe["all_rows_explicit"])
        self.assertEqual(unsafe["last_change_time_proxy_rows"], 1)
        self.assertEqual(unsafe["missing_or_unknown_provenance_rows"], 1)

    def test_registered_replay_comparison_requires_exact_decisions_with_float_tolerance(self) -> None:
        reference = [
            {
                "ticket_id": "T-1",
                "predicted_assignee": "dev-a",
                "ranked_candidates": ["dev-a", "dev-b"],
                "routing_status": "auto_assign",
                "fallback_reason": "high_confidence_low_open_set_risk",
                "calibrated_probability": 0.90000,
            }
        ]
        replay = [{**reference[0], "calibrated_probability": 0.90001}]

        report = compare_predictions(reference, replay, maximum_numeric_delta=0.0002)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["reference_decision_sha256"], report["replay_decision_sha256"]
        )

        replay[0]["routing_status"] = "top3_confirmation"
        failed = compare_predictions(reference, replay, maximum_numeric_delta=0.0002)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["decision_mismatch_rows"], 1)

    def test_sealed_holdout_registry_prevents_a_second_formal_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.jsonl"
            report = root / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            holdout_hash = "a" * 64
            evaluation_id = claim_sealed_holdout(
                registry,
                holdout_name="2022_q1",
                holdout_sha256=holdout_hash,
                routing_artifact_sha256="b" * 64,
            )
            complete_sealed_holdout_claim(
                registry, evaluation_id=evaluation_id, report_path=report
            )

            self.assertEqual([row["status"] for row in read_registry(registry)], ["started", "completed"])
            validate_registered_replay(
                registry,
                evaluation_id=evaluation_id,
                holdout_sha256=holdout_hash,
                routing_artifact_sha256="b" * 64,
            )
            with self.assertRaisesRegex(SystemExit, "routing artifact"):
                validate_registered_replay(
                    registry,
                    evaluation_id=evaluation_id,
                    holdout_sha256=holdout_hash,
                    routing_artifact_sha256="c" * 64,
                )
            with self.assertRaisesRegex(SystemExit, "already been claimed"):
                claim_sealed_holdout(
                    registry,
                    holdout_name="renamed_holdout",
                    holdout_sha256=holdout_hash,
                    routing_artifact_sha256="c" * 64,
                )

    def test_rolling_holdout_accepts_named_intermediate_history_without_name_reuse(self) -> None:
        parsed = parse_named_path("2021_q4=/tmp/2021q4.jsonl")

        self.assertEqual(parsed, ("2021_q4", Path("/tmp/2021q4.jsonl")))
        validate_window_names([("validation", Path("validation.jsonl"))], [parsed], "2022_q1")
        with self.assertRaises(SystemExit):
            validate_window_names(
                [("validation", Path("validation.jsonl"))],
                [("validation", Path("other.jsonl"))],
                "2022_q1",
            )

    def test_rolling_holdout_manifest_must_match_path_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            holdout.write_text('{"id": 1}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output": str(holdout),
                        "output_sha256": sha256_file(holdout),
                        "start_date": "2022-01-01",
                        "end_date": "2022-03-31",
                        "bugs_written": 1,
                        "sealed_output": True,
                    }
                ),
                encoding="utf-8",
            )

            provenance = validate_holdout_manifest(holdout, manifest)
            self.assertTrue(provenance["manifest_verified"])
            self.assertEqual(provenance["sha256"], sha256_file(holdout))

            holdout.write_text('{"id": 2}\n', encoding="utf-8")
            with self.assertRaises(SystemExit):
                validate_holdout_manifest(holdout, manifest)

    def test_scored_predictions_preserve_product_and_component_for_subgroup_gates(self) -> None:
        index = CandidateIndex(
            [
                {
                    "ticket_id": "H-1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "product": "Core",
                    "component": "DOM",
                    "title": "DOM parser failure",
                    "assignee": "dev-a",
                }
            ],
            half_life_days=30,
            smoothing_alpha=2,
            semantic=build_semantic_backend("none", ""),
        )

        class FakeRanker:
            feature_names_ = ["base_score"]

            @staticmethod
            def predict_proba(matrix: np.ndarray) -> np.ndarray:
                probability = np.full(len(matrix), 0.9)
                return np.column_stack((1.0 - probability, probability))

        predictions = scored_predictions(
            [
                {
                    "ticket_id": "Q-1",
                    "created_at": "2024-02-01T00:00:00Z",
                    "product": "Core",
                    "component": "DOM",
                    "title": "DOM parser crashes",
                    "assignee": "dev-a",
                }
            ],
            index,
            FakeRanker(),
            5,
        )

        self.assertEqual(predictions[0]["product"], "Core")
        self.assertEqual(predictions[0]["component"], "DOM")

    def test_rolling_history_promotes_prior_unseen_owner_without_future_leakage(self) -> None:
        base = [
            {
                "ticket_id": "H-1",
                "created_at": "2024-01-01T00:00:00Z",
                "title": "Base issue",
                "assignee": "dev-a",
                "label_available_at": "2024-01-02T00:00:00Z",
            },
            {
                "ticket_id": "H-2",
                "created_at": "2024-01-03T00:00:00Z",
                "title": "Late label issue",
                "assignee": "dev-late",
                "label_available_at": "2024-04-01T00:00:00Z",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(
                json.dumps(
                    {
                        "ticket_id": "Q-1",
                        "created_at": "2024-02-01T00:00:00Z",
                        "title": "New owner issue",
                        "assignee": "dev-new",
                        "label_available_at": "2024-02-02T00:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "ticket_id": "Q-2",
                        "created_at": "2024-03-01T00:00:00Z",
                        "title": "New owner follow-up",
                        "assignee": "dev-new",
                        "label_available_at": "2024-03-02T00:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            windows = prepare_rolling_windows(
                base,
                [("first", first), ("second", second)],
                {},
            )

        self.assertEqual(windows[0].audit["unseen_owner_rows"], 1)
        self.assertEqual(windows[1].audit["unseen_owner_rows"], 0)
        self.assertIn("dev-new", {row["assignee"] for row in windows[1].history_before})
        self.assertNotIn("dev-late", {row["assignee"] for row in windows[1].history_before})
        self.assertEqual(windows[1].audit["history_rows_withheld_future_last_change"], 1)
        self.assertTrue(all(window.audit["temporal_order_valid"] for window in windows))

    def test_multi_window_policy_requires_every_window_to_pass(self) -> None:
        windows = {}
        for name in ("q1", "q2"):
            rows = []
            for index in range(20):
                known = index < 16
                rows.append(
                    {
                        "known_owner": known,
                        "is_top1_correct": known and index < 14,
                        "is_top3_correct": known and index < 15,
                        "calibrated_probability": 0.95 - index * 0.025,
                        "open_set_probability": 0.1 if known else 0.9,
                        "candidate_source_count": 2 if index < 14 else 1,
                    }
                )
            windows[name] = rows

        policy = select_multi_window_routing_policy(
            windows,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.2,
            maximum_unseen_auto_rate=0.05,
            target_review_accuracy=0.7,
            minimum_candidate_source_count=2,
        )

        self.assertTrue(policy["found"])
        self.assertEqual(policy["minimum_candidate_source_count"], 2)
        self.assertEqual(set(policy["selection_window_metrics"]), {"q1", "q2"})
        self.assertTrue(
            all(row["auto_assignment_coverage"] >= 0.2 for row in policy["selection_window_metrics"].values())
        )

    def test_top5_assist_policy_passes_every_window_and_never_auto_assigns(self) -> None:
        windows = {}
        for name in ("q1", "q2"):
            windows[name] = [
                {
                    "known_owner": True,
                    "is_top1_correct": index < 60,
                    "is_top3_correct": index < 90,
                    "is_top5_correct": index < 108,
                    "calibrated_probability": 0.9,
                    "open_set_probability": 0.1,
                    "candidate_source_count": 2,
                    "candidate_count": 10,
                }
                for index in range(120)
            ]

        policy = select_multi_window_top5_assist_policy(
            windows,
            target_accuracy=0.85,
            minimum_coverage=0.50,
            minimum_rows_per_window=100,
            minimum_accuracy_lower_bound=0.80,
            minimum_candidate_source_count=2,
        )
        self.assertTrue(policy["found"])
        self.assertFalse(policy["auto_assignment_authorized"])
        for rows in windows.values():
            apply_top5_assist_policy(rows, policy)
            metrics = top5_assist_metrics(rows)
            gate = top5_assist_gate(
                metrics,
                target_accuracy=0.85,
                minimum_coverage=0.50,
                minimum_rows=100,
                minimum_accuracy_lower_bound=0.80,
            )
            self.assertTrue(gate["passed"])
            self.assertEqual(metrics["top5_assist_accuracy"], 0.9)
            self.assertEqual(metrics["auto_assignment_rows"], 0)
            self.assertTrue(
                all(row["top5_assist_status"] == "top5_user_confirmation" for row in rows)
            )

    def test_top5_assist_policy_fails_if_one_window_cannot_reach_target(self) -> None:
        windows = {}
        for name, correct_rows in (("stable", 108), ("later", 84)):
            windows[name] = [
                {
                    "known_owner": True,
                    "is_top1_correct": False,
                    "is_top3_correct": False,
                    "is_top5_correct": index < correct_rows,
                    "calibrated_probability": 0.9,
                    "open_set_probability": 0.1,
                    "candidate_source_count": 2,
                    "candidate_count": 10,
                }
                for index in range(120)
            ]

        policy = select_multi_window_top5_assist_policy(
            windows,
            target_accuracy=0.85,
            minimum_coverage=0.50,
            minimum_rows_per_window=100,
            minimum_accuracy_lower_bound=0.80,
        )

        self.assertFalse(policy["found"])
        self.assertEqual(
            policy["blocker"],
            "no_single_top5_policy_passed_every_development_window",
        )

    def test_top5_policy_can_be_frozen_against_reviewed_eligible_owner_labels(self) -> None:
        windows = {
            name: [
                {
                    "known_owner": True,
                    "is_top1_correct": False,
                    "is_top3_correct": False,
                    "is_top5_correct": False,
                    "is_top5_eligible_correct": index < 108,
                    "eligibility_label_available": True,
                    "eligible_top5_count": 5 if index < 108 else 4,
                    "calibrated_probability": 0.9,
                    "open_set_probability": 0.1,
                    "candidate_source_count": 2,
                    "candidate_count": 5,
                }
                for index in range(120)
            ]
            for name in ("q1", "q2")
        }

        policy = select_multi_window_top5_assist_policy(
            windows,
            target_accuracy=0.85,
            minimum_coverage=0.5,
            minimum_rows_per_window=100,
            minimum_accuracy_lower_bound=0.8,
            correctness_field="is_top5_eligible_correct",
        )

        self.assertTrue(policy["found"])
        self.assertEqual(policy["correctness_field"], "is_top5_eligible_correct")
        for rows in windows.values():
            apply_top5_assist_policy(rows, policy)
            metrics = top5_assist_metrics(
                rows, correctness_field=policy["correctness_field"]
            )
            self.assertEqual(metrics["top5_assist_rows"], 108)
            self.assertEqual(metrics["top5_assist_accuracy"], 1.0)

        with self.assertRaisesRegex(ValueError, "correctness field is missing"):
            top5_assist_metrics(
                [{"top5_assist_available": True}],
                correctness_field="is_top5_eligible_correct",
            )

    def test_score_drift_reference_detects_large_probability_shift(self) -> None:
        reference_rows = [
            {"calibrated_probability": 0.4 + index * 0.005}
            for index in range(40)
        ]
        shifted_rows = [
            {"calibrated_probability": 0.01 + index * 0.001}
            for index in range(40)
        ]

        reference = build_drift_reference(
            reference_rows,
            ("calibrated_probability",),
            maximum_psi=0.25,
        )
        drift = evaluate_score_drift(shifted_rows, reference)

        self.assertTrue(drift["severe_drift"])
        self.assertGreater(drift["maximum_psi"], 0.25)

    def test_leave_assignee_out_folds_remove_hidden_owner_without_future_leakage(self) -> None:
        history = []
        queries = []
        for index in range(12):
            history.append(
                {
                    "ticket_id": f"H-{index}",
                    "created_at": f"2024-01-{index + 1:02d}T00:00:00Z",
                    "assignee": f"dev-{index % 3}",
                }
            )
        for index in range(9):
            queries.append(
                {
                    "ticket_id": f"Q-{index}",
                    "created_at": f"2024-02-{index + 1:02d}T00:00:00Z",
                    "assignee": f"dev-{index % 3}",
                }
            )

        folds = build_leave_assignee_out_folds(
            history,
            queries,
            folds=3,
            hidden_query_fraction=0.2,
            minimum_history=2,
            seed=7,
        )

        self.assertEqual(len(folds), 3)
        for fold in folds:
            remaining = {str(row["assignee"]) for row in fold.history_rows}
            self.assertTrue(set(fold.hidden_assignees).isdisjoint(remaining))
            self.assertEqual(fold.audit["hidden_owner_history_leak_rows"], 0)
            self.assertTrue(fold.audit["leakage_free"])
            self.assertLessEqual(fold.audit["history_end"], fold.audit["query_start"])

    def test_portable_logistic_matches_sklearn_pipeline(self) -> None:
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.2, "y": 0.1},
            {"x": 0.8, "y": 0.9},
            {"x": 1.0, "y": 1.0},
        ]
        model = fit_portable_logistic(
            rows,
            [0, 0, 1, 1],
            ("x", "y"),
            seed=7,
            balanced=True,
        )

        expected = model.predict(rows)
        portable = predict_portable_logistic(rows, model.to_artifact())

        self.assertEqual(len(expected), len(portable))
        self.assertTrue(all(abs(float(left) - float(right)) < 1e-9 for left, right in zip(expected, portable)))

    def test_open_set_policy_jointly_limits_accuracy_coverage_and_unknown_auto_rate(self) -> None:
        rows = []
        for index in range(20):
            known = index < 16
            rows.append(
                {
                    "known_owner": known,
                    "is_top1_correct": known and index < 14,
                    "is_top3_correct": known and index < 15,
                    "calibrated_probability": 0.95 - index * 0.025,
                    "open_set_probability": 0.1 if known else 0.9,
                    "candidate_source_count": 2 if index < 14 else 1,
                }
            )

        policy = select_routing_policy(
            rows,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.2,
            maximum_unseen_auto_rate=0.05,
            target_review_accuracy=0.7,
            minimum_candidate_source_count=2,
        )
        route_open_set_predictions(rows, policy)
        auto = [row for row in rows if row["routing_status"] == "auto_assign"]

        self.assertTrue(policy["found"])
        self.assertEqual(policy["minimum_candidate_source_count"], 2)
        self.assertGreaterEqual(len(auto) / len(rows), 0.2)
        self.assertGreaterEqual(sum(row["is_top1_correct"] for row in auto) / len(auto), 0.85)
        self.assertFalse(any(not row["known_owner"] for row in auto))
        self.assertTrue(all(row["fallback_reason"] for row in rows))

    def test_deployment_gate_enforces_sample_size_confidence_and_component_floor(self) -> None:
        rows = []
        for index in range(100):
            auto = index < 20
            rows.append(
                {
                    "known_owner": True,
                    "is_top1_correct": not (auto and index in {18, 19}),
                    "is_top3_correct": True,
                    "routing_status": "auto_assign" if auto else "manual_triage_low_confidence",
                    "fallback_reason": "high_confidence_low_open_set_risk" if auto else "low_ranking_confidence",
                    "component": "core",
                }
            )
        metrics = open_set_routing_metrics(rows)
        gate = deployment_gate(
            metrics,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            minimum_holdout_rows=100,
            minimum_auto_rows=20,
            minimum_accuracy_lower_bound=0.80,
            minimum_component_auto_rows=20,
            minimum_component_accuracy=0.85,
            minimum_component_accuracy_lower_bound=0.80,
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["auto_accuracy_confidence_lower_bound"])
        self.assertFalse(gate["checks"]["component_accuracy_confidence_floor"])
        self.assertEqual(metrics["auto_assignment_accuracy"], 0.9)
        self.assertLess(open_set_wilson_interval(18, 20)["lower"], 0.8)

    def test_routing_score_diagnostics_explain_zero_auto_assignment(self) -> None:
        rows = [
            {"calibrated_probability": 0.4, "open_set_probability": 0.2},
            {"calibrated_probability": 0.45, "open_set_probability": 0.3},
        ]

        diagnostics = routing_score_diagnostics(
            rows,
            {"t_high": 0.5, "open_set_threshold": 0.6},
        )

        self.assertEqual(diagnostics["rows_above_high_confidence"], 0)
        self.assertEqual(diagnostics["rows_below_open_set_gate"], 2)
        self.assertEqual(diagnostics["rows_passing_both_auto_assignment_gates"], 0)
        self.assertEqual(diagnostics["maximum_calibrated_probability_below_open_set_gate"], 0.45)

    def test_deployment_gate_rejects_unsafe_unseen_rate_confidence_bound(self) -> None:
        metrics = {
            "auto_assignment_accuracy": 0.90,
            "auto_assignment_coverage": 0.20,
            "unseen_auto_assignment_rate": 0.04,
            "unseen_auto_assignment_rate_ci95": {"lower": 0.01, "upper": 0.0769},
            "rows": 1000,
            "auto_assignment_rows": 200,
            "auto_assignment_accuracy_ci95": {"lower": 0.85, "upper": 0.93},
            "component_metrics": {},
        }

        gate = deployment_gate(
            metrics,
            target_auto_accuracy=0.85,
            minimum_auto_coverage=0.10,
            maximum_unseen_auto_rate=0.05,
            maximum_unseen_rate_upper_bound=0.05,
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["checks"]["unseen_auto_assignment_rate"])
        self.assertFalse(
            gate["checks"]["unseen_auto_assignment_rate_confidence_upper_bound"]
        )

    def test_dual_open_set_threshold_preserves_top3_confirmation_band(self) -> None:
        rows = [
            {"calibrated_probability": 0.9, "open_set_probability": 0.1},
            {"calibrated_probability": 0.6, "open_set_probability": 0.5},
            {"calibrated_probability": 0.9, "open_set_probability": 0.9},
        ]
        policy = {
            "open_set_threshold": 0.3,
            "open_set_review_threshold": 0.8,
            "t_high": 0.8,
            "t_low": 0.4,
        }

        route_open_set_predictions(rows, policy)

        self.assertEqual(rows[0]["routing_status"], "auto_assign")
        self.assertEqual(rows[1]["routing_status"], "top3_confirmation")
        self.assertEqual(rows[2]["routing_status"], "manual_triage_open_set")

    def test_calibrated_routing_keeps_high_open_set_risk_manual(self) -> None:
        rows = [
            {
                "calibrated_probability": 0.95,
                "open_set_risk": 0.90,
                "is_top1_correct": False,
                "known_owner": False,
            },
            {
                "calibrated_probability": 0.90,
                "open_set_risk": 0.10,
                "is_top1_correct": True,
                "known_owner": True,
            },
        ]

        route_predictions(rows, high_threshold=0.85, low_threshold=0.60, open_set_risk_threshold=0.75)
        metrics = routing_metrics(rows)

        self.assertEqual(rows[0]["routing_status"], "manual_triage_open_set")
        self.assertEqual(rows[1]["routing_status"], "auto_assign")
        self.assertEqual(metrics["unseen_auto_assignment_rate"], 0.0)

    def test_threshold_selection_uses_accuracy_and_open_set_gate(self) -> None:
        rows = [
            {"calibrated_probability": 0.9, "open_set_risk": 0.1, "is_top1_correct": True},
            {"calibrated_probability": 0.8, "open_set_risk": 0.1, "is_top1_correct": True},
            {"calibrated_probability": 0.7, "open_set_risk": 0.1, "is_top1_correct": False},
            {"calibrated_probability": 0.95, "open_set_risk": 0.9, "is_top1_correct": False},
        ]

        policy = select_threshold(rows, target_accuracy=0.8, open_set_risk_threshold=0.75)

        self.assertTrue(policy["found"])
        self.assertEqual(policy["accepted_rows"], 2)
        self.assertEqual(policy["accuracy"], 1.0)

    def test_source_quotas_reserve_diverse_candidate_sources(self) -> None:
        quotas = source_quotas(
            {
                "product_component": ["pair-owner"],
                "component": ["component-owner"],
                "bm25": ["text-owner"],
                "sbert": ["semantic-owner"],
                "recent_component": ["recent-owner"],
                "product": ["product-owner"],
                "global_prior": ["global-owner"],
            },
            30,
        )

        self.assertEqual(sum(quotas.values()), 30)
        self.assertGreater(quotas["component"], quotas["global_prior"])
        self.assertGreater(quotas["sbert"], 0)

    def test_future_bugzilla_rows_are_normalized_and_deduplicated(self) -> None:
        raw = {
            "id": 42,
            "summary": "DOM parser crashes",
            "assigned_to": "Dev@Example.com",
            "creation_time": "2024-02-01T00:00:00Z",
            "product": "Core",
            "component": "DOM",
        }

        normalized = canonicalize_source_rows([raw])
        kept, removed = exclude_cross_split_overlap(
            normalized,
            [
                {
                    "ticket_id": "old",
                    "title": "DOM parser crashes",
                    "description": "",
                }
            ],
        )

        self.assertEqual(normalized[0]["ticket_id"], "bmo_42")
        self.assertEqual(normalized[0]["assignee"], "dev@example.com")
        self.assertEqual(kept, [])
        self.assertEqual(removed, 1)

    def test_duplicate_family_cannot_cross_training_and_development_splits(self) -> None:
        development = [
            {
                "ticket_id": "bmo_200",
                "duplicate_of": "100",
                "title": "Different duplicate wording",
                "description": "Different text does not bypass family isolation.",
            },
            {
                "ticket_id": "bmo_300",
                "title": "Independent issue",
                "description": "No duplicate relationship.",
            },
        ]
        history = [
            {
                "ticket_id": "bmo_100",
                "title": "Original issue",
                "description": "Original content.",
            }
        ]
        kept, removed = exclude_cross_split_overlap(
            development,
            history,
            strict_duplicate_families=True,
        )
        legacy_kept, legacy_removed = exclude_cross_split_overlap(
            development,
            history,
        )

        self.assertEqual(removed, 1)
        self.assertEqual([row["ticket_id"] for row in kept], ["bmo_300"])
        self.assertEqual(legacy_removed, 0)
        self.assertEqual(len(legacy_kept), 2)

    def test_pairwise_ranker_uses_only_bounded_hard_negatives(self) -> None:
        examples = [
            {
                "query_id": "Q-1",
                "candidate_assignee": "correct",
                "base_rank": 11,
                "features": {"signal": 2.0, "base_score": 0.1},
                "label": 1,
                "sample_weight": 1.0,
            }
        ]
        examples.extend(
            {
                "query_id": "Q-1",
                "candidate_assignee": f"negative-{index}",
                "base_rank": index + 1,
                "features": {"signal": 0.0, "base_score": 2.0 - index / 20},
                "label": 0,
                "sample_weight": 1.0,
            }
            for index in range(20)
        )

        matrix, labels, weights, audit = pairwise_training_data(
            examples, ["signal", "base_score"], maximum_hard_negatives_per_query=5
        )
        models = fit_rankers(examples, seed=3407)
        pairwise = models["pairwise_logistic_hard_negative"]
        probabilities = pairwise.predict_proba(
            np.asarray([[2.0, 0.1], [0.0, 2.0]])
        )[:, 1]

        self.assertEqual(matrix.shape, (10, 2))
        self.assertEqual(labels.tolist().count(1), 5)
        self.assertEqual(len(weights), 10)
        self.assertEqual(audit["selected_hard_negatives"], 5)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_candidate_source_ablation_groups_correlated_component_evidence(self) -> None:
        report = candidate_evidence_support_ablation(
            [
                {
                    "expected_assignee": "dev-a",
                    "candidate_pool": ["dev-a"],
                    "candidate_sources": {
                        "dev-a": ["component", "component_token"]
                    },
                },
                {
                    "expected_assignee": "dev-b",
                    "candidate_pool": ["dev-b"],
                    "candidate_sources": {
                        "dev-b": ["component", "bm25"]
                    },
                },
            ]
        )
        at_30 = report["cutoffs"]["30"]

        self.assertEqual(at_30["baseline_recall"], 1.0)
        self.assertEqual(
            at_30["by_source_family"]["component_history"][
                "unique_support_hit_rows"
            ],
            1,
        )
        self.assertEqual(
            at_30["by_source_family"]["component_history"][
                "recall_without_family_lower_bound"
            ],
            0.5,
        )

    def test_description_cleaner_keeps_signal_and_limits_stack_noise(self) -> None:
        description = """Steps to reproduce:
Click Save after editing the profile.
Click Save after editing the profile.
Actual Result:
TypeError: session token is missing
#0 0xaaa frame_a
#1 0xbbb frame_b
#2 0xccc frame_c
#3 0xddd frame_d
#4 0xeee frame_e
#5 0xfff frame_f
"""

        cleaned = clean_description(description)

        self.assertIn("Click Save after editing the profile.", cleaned)
        self.assertIn("TypeError: session token is missing", cleaned)
        self.assertEqual(cleaned.count("Click Save after editing the profile."), 1)
        self.assertNotIn("#5 0xfff", cleaned)

    def test_temporal_training_folds_never_use_future_rows(self) -> None:
        rows = []
        for index in range(20):
            rows.append(
                {
                    "ticket_id": f"B-{index}",
                    "created_at": f"2024-01-{index + 1:02d}T00:00:00Z",
                    "product": "Core",
                    "component": "DOM",
                    "title": f"DOM failure {index}",
                    "assignee": "dev-a" if index % 2 == 0 else "dev-b",
                }
            )

        examples, audit = build_temporal_training_examples(
            rows,
            folds=2,
            warmup_fraction=0.5,
            pool_size=4,
            half_life_days=30,
            smoothing_alpha=2,
            semantic=build_semantic_backend("none", ""),
        )

        self.assertTrue(examples)
        self.assertTrue(all(row["leakage_free"] for row in audit))
        self.assertTrue(all(row["history_end"] <= row["query_start"] for row in audit))

    def test_recent_owner_receives_stronger_recency_feature(self) -> None:
        history = [
            {
                "ticket_id": "OLD",
                "created_at": "2023-01-01T00:00:00Z",
                "product": "Core",
                "component": "DOM",
                "title": "DOM old issue",
                "assignee": "old-owner",
            },
            {
                "ticket_id": "NEW",
                "created_at": "2024-03-25T00:00:00Z",
                "product": "Core",
                "component": "DOM",
                "title": "DOM recent issue",
                "assignee": "recent-owner",
            },
        ]
        index = CandidateIndex(
            history,
            half_life_days=30,
            smoothing_alpha=2,
            semantic=build_semantic_backend("none", ""),
        )

        candidates = index.candidates(
            {
                "created_at": "2024-04-01T00:00:00Z",
                "product": "Core",
                "component": "DOM",
                "title": "DOM new failure",
            },
            4,
        )
        features = {candidate.assignee: candidate.features for candidate in candidates}
        sources = {candidate.assignee: candidate.sources for candidate in candidates}

        self.assertGreater(
            features["recent-owner"]["owner_recency"],
            features["old-owner"]["owner_recency"],
        )
        self.assertGreater(
            features["recent-owner"]["recency_component_share"],
            features["old-owner"]["recency_component_share"],
        )
        self.assertIn("recent_component", sources["recent-owner"])
        self.assertEqual(features["recent-owner"]["source_recent_component"], 1.0)

    def test_component_family_and_reporter_history_expand_candidate_recall(self) -> None:
        history = [
            {
                "ticket_id": "H-1",
                "created_at": "2024-01-01T00:00:00Z",
                "product": "Core",
                "component": "DOM Security",
                "title": "Security policy failure",
                "reporter": "reporter@example.com",
                "file_paths": ["dom/security/policy.cpp"],
                "file_paths_provenance": "stack_trace",
                "assignee": "security-owner",
            },
            {
                "ticket_id": "H-2",
                "created_at": "2024-01-02T00:00:00Z",
                "product": "Core",
                "component": "Networking",
                "title": "Socket failure",
                "reporter": "other@example.com",
                "assignee": "network-owner",
            },
        ]
        index = CandidateIndex(
            history,
            half_life_days=30,
            smoothing_alpha=2,
            semantic=build_semantic_backend("none", ""),
            candidate_generator_version="v3",
        )

        candidates = index.candidates(
            {
                "ticket_id": "Q-1",
                "created_at": "2024-02-01T00:00:00Z",
                "product": "Unknown",
                "component": "DOM Graphics",
                "title": "Rendering anomaly",
                "reporter": "reporter@example.com",
                "file_paths": ["dom/security/policy.cpp"],
                "file_paths_provenance": "stack_trace",
            },
            30,
        )
        security = next(
            candidate for candidate in candidates if candidate.assignee == "security-owner"
        )

        self.assertIn("component_token", security.sources)
        self.assertIn("reporter_history", security.sources)
        self.assertIn("file_history", security.sources)
        self.assertGreater(security.features["component_token_share"], 0.0)
        self.assertEqual(security.features["reporter_owner_share"], 1.0)
        self.assertEqual(security.features["file_owner_share"], 1.0)

        unsafe_file_query = index.candidates(
            {
                "ticket_id": "Q-unsafe-file",
                "created_at": "2024-02-01T00:00:00Z",
                "product": "Unknown",
                "component": "Unknown",
                "title": "Unrelated report",
                "file_paths": ["dom/security/policy.cpp"],
            },
            30,
        )
        unsafe_security = next(
            candidate
            for candidate in unsafe_file_query
            if candidate.assignee == "security-owner"
        )
        self.assertNotIn("file_history", unsafe_security.sources)
        self.assertEqual(unsafe_security.features["file_owner_share"], 0.0)

        legacy_index = CandidateIndex(
            history,
            half_life_days=30,
            smoothing_alpha=2,
            semantic=build_semantic_backend("none", ""),
        )
        legacy_security = next(
            candidate
            for candidate in legacy_index.candidates(
                {
                    "ticket_id": "Q-1",
                    "created_at": "2024-02-01T00:00:00Z",
                    "product": "Unknown",
                    "component": "DOM Graphics",
                    "title": "Rendering anomaly",
                    "reporter": "reporter@example.com",
                    "file_paths": ["dom/security/policy.cpp"],
                    "file_paths_provenance": "stack_trace",
                },
                30,
            )
            if candidate.assignee == "security-owner"
        )
        self.assertNotIn("component_token", legacy_security.sources)
        self.assertNotIn("reporter_history", legacy_security.sources)
        self.assertNotIn("file_history", legacy_security.sources)


if __name__ == "__main__":
    unittest.main()

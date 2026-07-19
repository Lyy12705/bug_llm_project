from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import PipelineConfig
from modules.assignee_feedback import (
    append_assignee_feedback,
    build_assignee_feedback_record,
    feedback_rows_to_history_rows,
    merge_feedback_into_history,
    read_assignee_feedback,
)
from modules.assignee_triager import AssigneeTriager
from modules.duplicate_detector import DuplicateDetector
from modules.ticket_extractor import TicketExtractor
from pipeline.orchestrator import build_default_orchestrator


class PipelineIntegrationTests(unittest.TestCase):
    def test_default_config_prefers_paper_grade_bmo_assignee_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bmo_history = (
                root
                / "assignee_triage_accuracy"
                / "paper_grade"
                / "data"
                / "processed"
                / "bmo_paper_2024_3k_history_train.jsonl"
            )
            bmo_history.parent.mkdir(parents=True)
            bmo_history.write_text("", encoding="utf-8")

            config = PipelineConfig(project_root=root, save_checkpoints=False)

        self.assertEqual(config.assignee_dataset_path, bmo_history)

    def test_default_config_does_not_ship_placeholder_component_owners(self) -> None:
        config = PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False)

        self.assertEqual(config.component_owner_mapping, {})

    def test_checkpoint_ticket_id_is_confined_to_processed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            config = PipelineConfig(
                project_root=root,
                processed_ticket_dir=processed,
                save_checkpoints=True,
            )
            checkpoint = build_default_orchestrator(config)._checkpoint_dir({"ticket_id": "../../escape"})

        self.assertEqual(checkpoint.parent, processed.resolve())
        self.assertEqual(checkpoint.name, "escape")

    def test_default_orchestrator_passes_fault_localization_config(self) -> None:
        config = PipelineConfig(
            project_root=PROJECT_ROOT,
            save_checkpoints=False,
            fault_localization_top_k=7,
            fault_localization_embedding_backend="tfidf",
            fault_localization_llm_rerank=False,
            fault_localization_llm_candidate_k=12,
            fault_localization_llm_cache_dir=PROJECT_ROOT / "reports" / "fault_localization" / "llm_cache_test",
        )
        orchestrator = build_default_orchestrator(config)

        self.assertEqual(orchestrator.bug_localizer.top_k, 7)
        self.assertEqual(orchestrator.bug_localizer.embedding_backend, "tfidf")
        self.assertFalse(orchestrator.bug_localizer.llm_rerank)
        self.assertEqual(orchestrator.bug_localizer.llm_candidate_k, 12)
        self.assertEqual(orchestrator.bug_localizer.llm_cache_dir, PROJECT_ROOT / "reports" / "fault_localization" / "llm_cache_test")

    def test_ticket_extractor_normalizes_raw_ticket(self) -> None:
        raw_ticket = {
            "id": "RAW-9",
            "summary": "Build fails on Windows",
            "body": "Compiler error: missing generated header.",
            "environment": {"os": "win11", "version": "2.0.0"},
        }

        structured = TicketExtractor().extract(raw_ticket)

        self.assertEqual(structured["ticket_id"], "RAW-9")
        self.assertEqual(structured["title"], "Build fails on Windows")
        self.assertEqual(structured["component"], "build")
        self.assertEqual(structured["os"], "Windows")
        self.assertEqual(structured["bug_type"], "build_error")

    def test_pipeline_short_circuits_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            historical = tmp_path / "historical.jsonl"
            historical.write_text(
                '{"ticket_id":"BUG-1","title":"Login crash when token missing",'
                '"description":"Missing token causes TypeError in auth validation.",'
                '"component":"authentication","priority":"P2"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                historical_tickets_path=historical,
                duplicate_threshold=0.35,
                save_checkpoints=False,
            )
            orchestrator = build_default_orchestrator(config)

            result = orchestrator.run_pipeline(
                {
                    "ticket_id": "RAW-1",
                    "title": "Login crash when token missing",
                    "description": "Missing token causes TypeError in auth validation.",
                    "component": "authentication",
                },
                str(tmp_path),
            )

        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["duplicate_of"], "BUG-1")

    def test_duplicate_detector_does_not_penalize_missing_optional_error_text(self) -> None:
        detector = DuplicateDetector(config=PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False))
        ticket = {
            "ticket_id": "Q-EXACT",
            "title": "Login crash when token is missing",
            "description": "Missing token causes TypeError in authentication validation.",
            "component": "authentication",
        }
        historical = [{**ticket, "ticket_id": "H-EXACT"}]

        result = detector.detect(ticket, historical)

        self.assertTrue(result["is_duplicate"])
        self.assertEqual(result["duplicate_of"], "H-EXACT")
        self.assertAlmostEqual(result["similarity_score"], 1.0)

    def test_pipeline_returns_manual_patch_when_no_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            config = PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False)
            orchestrator = build_default_orchestrator(config)

            result = orchestrator.run_pipeline(_raw_ticket(), str(repo))

        self.assertEqual(result["status"], "needs_manual_patch")
        self.assertEqual(result["failed_step"], "patch_generation")
        self.assertEqual(result["bug_location"]["bug_location"]["file"], "src/auth/validator.py")
        self.assertEqual(result["patch"]["bug_location"]["file"], "src/auth/validator.py")

    def test_pipeline_marks_structurally_valid_patch_unverified_without_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            raw_ticket = _raw_ticket()
            raw_ticket["proposed_patch"] = (
                "diff --git a/src/auth/validator.py b/src/auth/validator.py\n"
                "--- a/src/auth/validator.py\n"
                "+++ b/src/auth/validator.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def validate_token(token):\n"
                "+    if token is None:\n"
                "+        raise ValueError(\"Token is required\")\n"
                "     return token.strip()\n"
            )
            config = PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False)
            orchestrator = build_default_orchestrator(config)

            result = orchestrator.run_pipeline(raw_ticket, str(repo))

        self.assertEqual(result["status"], "patch_applied_unverified")
        self.assertFalse(result["verification"]["fully_verified"])
        self.assertEqual(result["patch"]["modified_files"], ["src/auth/validator.py"])
        self.assertEqual(result["regression"]["patch_apply"], "passed")
        self.assertTrue(result["commit_message"]["commit_message"].startswith("fix(authentication):"))

    def test_pipeline_requires_duplicate_review_on_detector_exception(self) -> None:
        config = PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False)
        orchestrator = build_default_orchestrator(config)

        class FailingDuplicateDetector:
            def detect(self, ticket: dict) -> dict:
                raise ValueError("broken duplicate index")

        orchestrator.duplicate_detector = FailingDuplicateDetector()
        result = orchestrator.run_pipeline(_raw_ticket(), str(PROJECT_ROOT))

        self.assertEqual(result["status"], "needs_duplicate_review")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["duplicate"]["review_reason"], "duplicate_detector_exception")
        self.assertNotIn("priority", result)

    def test_pipeline_stops_when_duplicate_score_needs_review(self) -> None:
        config = PipelineConfig(project_root=PROJECT_ROOT, save_checkpoints=False)
        orchestrator = build_default_orchestrator(config)

        class ReviewDuplicateDetector:
            def detect(self, ticket: dict) -> dict:
                return {"is_duplicate": False, "needs_review": True, "top_k_candidates": []}

        orchestrator.duplicate_detector = ReviewDuplicateDetector()
        result = orchestrator.run_pipeline(_raw_ticket(), str(PROJECT_ROOT))

        self.assertEqual(result["status"], "needs_duplicate_review")
        self.assertNotIn("priority", result)

    def test_pipeline_completes_only_after_fib_and_regression_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            raw_ticket = _raw_ticket()
            raw_ticket["proposed_patch"] = (
                "diff --git a/src/auth/validator.py b/src/auth/validator.py\n"
                "--- a/src/auth/validator.py\n"
                "+++ b/src/auth/validator.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def validate_token(token):\n"
                "+    if token is None:\n"
                "+        raise ValueError(\"Token is required\")\n"
                "     return token.strip()\n"
            )
            raw_ticket["tests"] = [
                {
                    "test_name": "missing token is rejected",
                    "command": [
                        "python3",
                        "-c",
                        "from src.auth.validator import validate_token; "
                        "\ntry: validate_token(None)"
                        "\nexcept ValueError: raise SystemExit(0)"
                        "\nraise SystemExit(1)",
                    ],
                }
            ]
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                save_checkpoints=False,
                run_regression_tests=True,
                allow_ticket_test_commands=True,
                test_command=[
                    "python3",
                    "-c",
                    "from src.auth.validator import validate_token; assert validate_token(' x ') == 'x'",
                ],
            )
            result = build_default_orchestrator(config).run_pipeline(raw_ticket, str(repo))

        self.assertEqual(result["status"], "completed_verified")
        self.assertEqual(result["regression"]["reproduction_result"], "passed")
        self.assertEqual(result["regression"]["regression_result"], "passed")
        self.assertTrue(result["verification"]["fully_verified"])

    def test_pipeline_does_not_execute_ticket_commands_without_explicit_trust(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            marker = tmp_path / "ticket-command-ran"
            raw_ticket = _raw_ticket()
            raw_ticket["proposed_patch"] = (
                "diff --git a/src/auth/validator.py b/src/auth/validator.py\n"
                "--- a/src/auth/validator.py\n"
                "+++ b/src/auth/validator.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def validate_token(token):\n"
                "+    if token is None:\n"
                "+        raise ValueError(\"Token is required\")\n"
                "     return token.strip()\n"
            )
            raw_ticket["tests"] = [
                {
                    "test_name": "untrusted side effect",
                    "command": ["python3", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                }
            ]
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                save_checkpoints=False,
                run_regression_tests=True,
                test_command=["python3", "-c", "raise SystemExit(0)"],
            )

            result = build_default_orchestrator(config).run_pipeline(raw_ticket, str(repo))

        self.assertFalse(marker.exists())
        self.assertEqual(result["status"], "tests_passed")
        self.assertEqual(result["regression"]["reproduction_result"], "blocked_untrusted_commands")
        self.assertFalse(result["regression"]["reproduction_execution_allowed"])

    def test_pipeline_assigns_owner_after_duplicate_and_priority_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            historical = tmp_path / "historical.jsonl"
            historical.write_text("", encoding="utf-8")
            assignee_history = tmp_path / "assignee_history.jsonl"
            assignee_history.write_text(
                '{"ticket_id":"BMO-1","product":"Core","component":"Graphics: Canvas2D",'
                '"title":"Canvas drawImage fails after detached canvas",'
                '"description":"Detached canvas drawImage should throw consistently.",'
                '"assignee":"dev_canvas"}\n'
                '{"ticket_id":"BMO-2","product":"Core","component":"Graphics: Canvas2D",'
                '"title":"Canvas rendering skips frame after resize",'
                '"description":"Canvas2D rendering does not repaint after resizing the surface.",'
                '"assignee":"dev_canvas"}\n'
                '{"ticket_id":"BMO-3","product":"Core","component":"DOM: Core & HTML",'
                '"title":"DOM parser mishandles template content",'
                '"description":"Template content is not cloned correctly.",'
                '"assignee":"dev_dom"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                historical_tickets_path=historical,
                assignee_dataset_path=assignee_history,
                assignee_allow_uncalibrated_auto_assignment=True,
                save_checkpoints=False,
            )
            orchestrator = build_default_orchestrator(config)

            result = orchestrator.run_pipeline(
                {
                    "ticket_id": "RAW-CANVAS",
                    "product": "Core",
                    "component": "Graphics: Canvas2D",
                    "title": "Canvas drawImage detached canvas throws incorrectly",
                    "description": "A detached canvas produces the wrong drawImage behavior during rendering.",
                    "severity": "normal",
                },
                str(repo),
            )

        self.assertIn("duplicate", result)
        self.assertFalse(result["duplicate"]["is_duplicate"])
        self.assertIn("priority", result)
        self.assertIn("predicted_priority", result["priority"])
        self.assertEqual(result["assignee"]["assignee"], "dev_canvas")
        self.assertIn("product_component_history", result["assignee"]["reason"])

    def test_assignee_triager_suggests_text_match_but_routes_unmapped_component_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","title":"Checkout button overlaps total label",'
                '"description":"Mobile layout makes the checkout button cover the total label.",'
                '"component":"frontend","assignee":"frontend-team@example.com"}\n'
                '{"ticket_id":"H-2","title":"Login modal button is hidden on small screens",'
                '"description":"Responsive UI hides the submit button when the keyboard opens.",'
                '"component":"frontend","assignee":"frontend-team@example.com"}\n'
                '{"ticket_id":"H-3","title":"Worker retries payment event forever",'
                '"description":"Backend queue retries permanent payment failures without backoff.",'
                '"component":"backend","assignee":"backend-team@example.com"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(project_root=PROJECT_ROOT, assignee_dataset_path=history, save_checkpoints=False)
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-MOBILE",
                    "title": "Mobile keyboard covers login button",
                    "description": "On small screens the keyboard covers the submit button in the login form.",
                    "component": "mobile",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["suggested_assignee"], "frontend-team@example.com")
        self.assertTrue(result["needs_manual_triage"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "text_only_or_broad_history_only")
        self.assertIn("frontend-team@example.com", result["ranked_candidates"])
        self.assertIn("text_similarity", result["reason"])

    def test_historical_recommendation_requires_calibration_before_auto_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","component":"payments","title":"Payment callback fails",'
                '"description":"Payment callback loses state.","assignee":"payments-owner@example.com"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-PAY",
                    "component": "payments",
                    "title": "Payment callback fails after retry",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["suggested_assignee"], "payments-owner@example.com")
        self.assertEqual(result["fallback_reason"], "uncalibrated_recommendation_only")

    def test_assignee_triager_uses_manual_fallback_without_history_or_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_history = Path(tmp) / "missing_assignee_history.jsonl"
            config = PipelineConfig(project_root=PROJECT_ROOT, assignee_dataset_path=missing_history, save_checkpoints=False)
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-UNKNOWN",
                    "title": "Telemetry widget loses decimal places",
                    "description": "The export report rounds values unexpectedly.",
                    "component": "reporting",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["routing_status"], "needs_manual_triage")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "missing_assignee_history")
        self.assertEqual(result["ranked_candidates"], ["manual_triage"])

    def test_assignee_triager_can_assign_mapping_only_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_history = Path(tmp) / "missing_assignee_history.jsonl"
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=missing_history,
                component_owner_mapping={"authentication": "auth-team@example.com"},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-AUTH",
                    "title": "Login token validation crashes",
                    "description": "Auth request crashes when token is missing.",
                    "component": "authentication",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "auth-team@example.com")
        self.assertEqual(result["routing_status"], "assigned_by_fallback")
        self.assertTrue(result["fallback_used"])
        self.assertFalse(result["needs_manual_triage"])
        self.assertEqual(result["fallback_reason"], "component_owner_mapping_only")

    def test_assignee_triager_filters_inactive_history_and_uses_active_roster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","product":"App","component":"api","title":"API token refresh fails",'
                '"description":"Token refresh returns stale credentials.","assignee":"old-owner@example.com"}\n'
                '{"ticket_id":"H-2","product":"App","component":"api","title":"API token refresh timeout",'
                '"description":"Token refresh endpoint times out.","assignee":"old-owner@example.com"}\n'
                '{"ticket_id":"H-3","product":"App","component":"api","title":"API token refresh regression",'
                '"description":"Refresh endpoint raises a regression.","assignee":"new-owner@example.com"}\n',
                encoding="utf-8",
            )
            roster = tmp_path / "roster.json"
            roster.write_text(
                '{"assignees":[{"email":"new-owner@example.com","active":true},'
                '{"email":"old-owner@example.com","active":false}]}',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_active_roster_path=roster,
                assignee_allow_uncalibrated_auto_assignment=True,
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-API",
                    "product": "App",
                    "component": "api",
                    "title": "API token refresh fails after deploy",
                    "description": "Refresh endpoint returns stale credentials.",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "new-owner@example.com")
        self.assertNotIn("old-owner@example.com", result["ranked_candidates"])
        self.assertEqual(result["profile_stats"]["candidate_count"], 1)

    def test_assignee_triager_fails_closed_when_configured_active_roster_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","component":"api","title":"API timeout",'
                '"description":"API request times out.","assignee":"api-owner@example.com"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_active_roster_path=tmp_path / "missing_roster.json",
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {"ticket_id": "RAW-API", "component": "api", "title": "API timeout"},
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["fallback_reason"], "active_roster_unavailable")
        self.assertEqual(result["profile_stats"]["roster_status"], "active_roster_missing")

    def test_active_roster_object_does_not_treat_boolean_status_as_assignee(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","component":"api","title":"API timeout",'
                '"description":"API request times out.","assignee":"api-owner@example.com"}\n',
                encoding="utf-8",
            )
            roster = tmp_path / "roster.json"
            roster.write_text(
                '{"active":[{"email":"api-owner@example.com","active":true}]}',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_active_roster_path=roster,
                assignee_allow_uncalibrated_auto_assignment=True,
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {"ticket_id": "RAW-API", "component": "api", "title": "API timeout"},
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "api-owner@example.com")
        self.assertEqual(result["profile_stats"]["active_roster_size"], 1)
        self.assertNotIn("true", result["ranked_candidates"])

    def test_assignee_triager_blocks_inactive_component_mapping_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_history = tmp_path / "missing_assignee_history.jsonl"
            inactive = tmp_path / "inactive.json"
            inactive.write_text('["auth-team@example.com"]', encoding="utf-8")
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=missing_history,
                assignee_inactive_path=inactive,
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-AUTH-INACTIVE",
                    "title": "Login token validation crashes",
                    "description": "Auth request crashes when token is missing.",
                    "component": "authentication",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertTrue(result["needs_manual_triage"])
        self.assertEqual(result["fallback_reason"], "missing_assignee_history")

    def test_assignee_triager_uses_data_driven_component_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_history = tmp_path / "missing_assignee_history.jsonl"
            ownership = tmp_path / "component_ownership.json"
            ownership.write_text('{"components":{"mobile":["mobile-owner@example.com"]}}', encoding="utf-8")
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=missing_history,
                assignee_component_ownership_path=ownership,
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-MOBILE-OWNER",
                    "title": "Mobile keyboard covers submit button",
                    "description": "Small screen keyboard overlaps submit button.",
                    "component": "mobile",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["assignee"], "mobile-owner@example.com")
        self.assertEqual(result["routing_status"], "assigned")
        self.assertIn("component_ownership", result["candidate_details"][0]["signals"])

    def test_assignee_triager_uses_file_ownership_from_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_history = tmp_path / "missing_assignee_history.jsonl"
            ownership = tmp_path / "file_ownership.json"
            ownership.write_text('{"files":{"src/auth/":"auth-owner@example.com"}}', encoding="utf-8")
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=missing_history,
                assignee_file_ownership_path=ownership,
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-FILE-OWNER",
                    "title": "Token validator raises AttributeError",
                    "description": "Traceback points at src/auth/session.py during login.",
                    "component": "unknown",
                    "logs": "AttributeError at src/auth/session.py:42",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "auth-owner@example.com")
        self.assertIn("file_ownership", result["candidate_details"][0]["signals"])

    def test_component_and_file_ownership_matching_respects_token_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ownership = tmp_path / "file_ownership.json"
            ownership.write_text('{"files":{"src/auth/":"auth-owner@example.com"}}', encoding="utf-8")
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=tmp_path / "missing_history.jsonl",
                assignee_file_ownership_path=ownership,
                component_owner_mapping={
                    "api": "api-owner@example.com",
                    "api gateway": "gateway-owner@example.com",
                },
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            false_match = triager.assign(
                {
                    "ticket_id": "RAW-RAPID",
                    "component": "rapid",
                    "title": "Authorization failure",
                    "logs": "Failure at src/authorize/token.py:4",
                },
                {"predicted_priority": "P3"},
            )
            valid_match = triager.assign(
                {"ticket_id": "RAW-API-GATEWAY", "component": "api-gateway", "title": "Gateway timeout"},
                {"predicted_priority": "P2"},
            )

        self.assertEqual(false_match["assignee"], "manual_triage")
        self.assertEqual(valid_match["assignee"], "gateway-owner@example.com")

    def test_assignee_history_excludes_rows_created_after_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-OLD","component":"mobile","title":"Old mobile crash",'
                '"description":"Old crash.","assignee":"old-owner@example.com",'
                '"created_at":"2026-01-01T00:00:00Z"}\n'
                '{"ticket_id":"H-FUTURE","component":"mobile","title":"Future mobile crash",'
                '"description":"Future crash.","assignee":"future-owner@example.com",'
                '"created_at":"2026-03-01T00:00:00Z"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_allow_uncalibrated_auto_assignment=True,
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-MOBILE",
                    "component": "mobile",
                    "title": "Mobile crash",
                    "created_at": "2026-02-01T00:00:00Z",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "old-owner@example.com")
        self.assertEqual(result["profile_stats"]["future_history_rows_filtered"], 1)
        self.assertNotIn("future-owner@example.com", result["ranked_candidates"])

    def test_assignee_profile_reuses_full_history_after_latest_training_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history.jsonl"
            history.write_text(
                '{"ticket_id":"1","title":"Auth failure","component":"auth",'
                '"assignee":"dev@example.com","created_at":"2025-01-01T00:00:00Z"}\n',
                encoding="utf-8",
            )
            triager = AssigneeTriager(
                config=PipelineConfig(
                    project_root=Path(tmp),
                    assignee_dataset_path=history,
                    save_checkpoints=False,
                )
            )

            first = triager._history_profile(as_of="2026-01-01T00:00:00Z")
            second = triager._history_profile(as_of="2026-02-01T00:00:00Z")

        self.assertIs(first, second)

    def test_assignee_triager_routes_below_calibrated_policy_to_top_k_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","product":"App","component":"payments","title":"Payment webhook retry fails",'
                '"description":"Webhook retry loses payment state.","assignee":"payments-owner@example.com"}\n',
                encoding="utf-8",
            )
            policy = tmp_path / "routing_policy.csv"
            policy.write_text(
                "policy,deployment_status,approval_gate_passed,expires_at,t_low,t_high\n"
                "automation_first,approved,true,2099-01-01T00:00:00Z,0,0.96\n",
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_routing_policy_path=policy,
                assignee_routing_policy_name="automation_first",
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-PAY",
                    "product": "App",
                    "component": "payments",
                    "title": "Payment webhook retry fails after timeout",
                    "description": "Webhook retry loses payment state after timeout.",
                },
                {"predicted_priority": "P1"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["suggested_assignee"], "payments-owner@example.com")
        self.assertEqual(result["fallback_reason"], "top_k_confirmation_required")
        self.assertEqual(result["decision_reason_code"], "top_k_confirmation_required")
        self.assertEqual(result["routing_policy"], "automation_first")

    def test_assignee_triager_ignores_invalid_routing_threshold_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","product":"App","component":"payments",'
                '"title":"Payment webhook fails","description":"Webhook loses state.",'
                '"assignee":"payments-owner@example.com"}\n',
                encoding="utf-8",
            )
            policy = tmp_path / "routing_policy.csv"
            policy.write_text("policy,t_low,t_high\nbroken,0.9,0.5\n", encoding="utf-8")
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_routing_policy_path=policy,
                assignee_routing_policy_name="broken",
                assignee_allow_uncalibrated_auto_assignment=True,
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-PAY",
                    "product": "App",
                    "component": "payments",
                    "title": "Payment webhook fails after retry",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["assignee"], "payments-owner@example.com")
        self.assertEqual(result["routing_policy"], "")

    def test_assignee_triager_open_set_gate_blocks_text_only_unseen_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","title":"Checkout button overlaps total label",'
                '"description":"Mobile layout makes the checkout button cover the total label.",'
                '"component":"frontend","assignee":"frontend-team@example.com"}\n'
                '{"ticket_id":"H-2","title":"Login modal button is hidden on small screens",'
                '"description":"Responsive UI hides the submit button when the keyboard opens.",'
                '"component":"frontend","assignee":"frontend-team@example.com"}\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_allow_text_only_assignment=True,
                assignee_open_set_enabled=True,
                assignee_open_set_risk_threshold=0.5,
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)

            result = triager.assign(
                {
                    "ticket_id": "RAW-MOBILE-OPEN",
                    "title": "Mobile keyboard covers login button",
                    "description": "On small screens the keyboard covers the submit button in the login form.",
                    "component": "mobile",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["suggested_assignee"], "frontend-team@example.com")
        self.assertEqual(result["fallback_reason"], "open_set_unknown_risk")
        self.assertGreaterEqual(result["open_set_risk"], 0.5)

    def test_open_set_risk_takes_precedence_over_top3_confirmation_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","title":"Checkout button overlaps total",'
                '"description":"Mobile layout covers the checkout total.",'
                '"component":"frontend","assignee":"frontend@example.com"}\n',
                encoding="utf-8",
            )
            policy = root / "policy.json"
            policy.write_text(
                '{"schema_version":1,"name":"safe","deployment_status":"approved",'
                '"approval_gate_passed":true,"expires_at":"2099-01-01T00:00:00Z",'
                '"t_low":0.0,"t_high":0.99}',
                encoding="utf-8",
            )
            result = AssigneeTriager(
                config=PipelineConfig(
                    project_root=root,
                    assignee_dataset_path=history,
                    assignee_routing_policy_path=policy,
                    assignee_open_set_enabled=True,
                    assignee_open_set_risk_threshold=0.5,
                    assignee_allow_text_only_assignment=True,
                    component_owner_mapping={},
                    save_checkpoints=False,
                )
            ).assign(
                {
                    "ticket_id": "Q-1",
                    "title": "Mobile keyboard covers checkout button",
                    "description": "The keyboard covers the checkout total on a phone.",
                    "component": "mobile",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["fallback_reason"], "open_set_unknown_risk")
        self.assertNotEqual(result["fallback_reason"], "top_k_confirmation_required")

    def test_assignee_triager_applies_only_approved_isotonic_calibration_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","product":"App","component":"payments",'
                '"title":"Payment callback loses state","description":"Callback retry loses payment state.",'
                '"assignee":"payments-owner@example.com"}\n',
                encoding="utf-8",
            )
            artifact = tmp_path / "calibrator.json"
            artifact.write_text(
                '{"schema_version":1,"artifact_type":"assignee_confidence_calibrator",'
                '"name":"isotonic_regression","deployment_status":"approved",'
                '"approval_gate_passed":true,"expires_at":"2099-01-01T00:00:00Z",'
                '"ranker_confidence_version":"hybrid_ranker_v1",'
                '"mapping":{"x_thresholds":[0.0,0.95],"y_thresholds":[0.0,0.4]}}',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_calibration_artifact_path=artifact,
                component_owner_mapping={},
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-CALIBRATION",
                    "product": "App",
                    "component": "payments",
                    "title": "Payment callback loses state after retry",
                    "description": "Callback retry loses payment state after timeout.",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["raw_confidence"], 0.95)
        self.assertEqual(result["confidence"], 0.4)
        self.assertEqual(result["calibration_status"], "applied")
        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["fallback_reason"], "low_confidence")

    def test_assignee_triager_rejects_unapproved_calibration_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","component":"payments","title":"Payment callback loses state",'
                '"description":"Callback retry loses payment state.","assignee":"payments-owner@example.com"}\n',
                encoding="utf-8",
            )
            artifact = tmp_path / "calibrator.json"
            artifact.write_text(
                '{"schema_version":1,"artifact_type":"assignee_confidence_calibrator",'
                '"name":"isotonic_regression","deployment_status":"research_only",'
                '"ranker_confidence_version":"hybrid_ranker_v1",'
                '"mapping":{"x_thresholds":[0.0,0.95],"y_thresholds":[0.0,0.1]}}',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_calibration_artifact_path=artifact,
                component_owner_mapping={},
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-RESEARCH-ARTIFACT",
                    "component": "payments",
                    "title": "Payment callback loses state after retry",
                    "description": "Callback retry loses payment state after timeout.",
                },
                {"predicted_priority": "P2"},
            )

        self.assertEqual(result["calibration_status"], "artifact_not_approved")
        self.assertEqual(result["confidence"], result["raw_confidence"])
        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["suggested_assignee"], "payments-owner@example.com")
        self.assertEqual(result["fallback_reason"], "uncalibrated_recommendation_only")

    def test_assignee_triager_applies_approved_open_set_rule_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","product":"App","component":"frontend",'
                '"title":"Login button hidden on mobile","description":"Mobile keyboard hides login submit button.",'
                '"assignee":"frontend-owner@example.com"}\n',
                encoding="utf-8",
            )
            artifact = tmp_path / "open_set.json"
            artifact.write_text(
                '{"schema_version":1,"artifact_type":"assignee_open_set_detector",'
                '"name":"rule_based_novelty","deployment_status":"approved",'
                '"approval_gate_passed":true,"expires_at":"2099-01-01T00:00:00Z",'
                '"ranker_confidence_version":"hybrid_ranker_v1","threshold":0.5,'
                '"parameters":{"weights":{"unseen_component":0.25,'
                '"unseen_product_component_pair":0.25,"rare_component":0.15,'
                '"rare_product_component_pair":0.15,"no_component_history":0.10,'
                '"no_text_similarity":0.05,"disagreement":0.05},'
                '"rare_component_max_rows":5,"rare_product_component_max_rows":5,'
                '"ownership_risk_reduction":0.2}}',
                encoding="utf-8",
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_allow_text_only_assignment=True,
                assignee_open_set_enabled=True,
                assignee_open_set_artifact_path=artifact,
                component_owner_mapping={},
                save_checkpoints=False,
            )

            result = AssigneeTriager(config=config).assign(
                {
                    "ticket_id": "RAW-OPEN-ARTIFACT",
                    "product": "App",
                    "component": "mobile",
                    "title": "Mobile keyboard hides login button",
                    "description": "Mobile keyboard hides the login submit button.",
                },
                {"predicted_priority": "P3"},
            )

        self.assertEqual(result["open_set_status"], "applied")
        self.assertEqual(result["open_set_detector"], "rule_based_novelty")
        self.assertEqual(result["assignee"], "manual_triage")
        self.assertEqual(result["fallback_reason"], "open_set_unknown_risk")

    def test_assignee_feedback_jsonl_can_update_future_history_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "assignee_feedback.jsonl"
            record = build_assignee_feedback_record(
                ticket_json={
                    "ticket_id": "RAW-FEEDBACK",
                    "product": "App",
                    "component": "mobile",
                    "title": "Mobile keyboard covers login button",
                    "description": "Submit button is hidden on small screens.",
                },
                prediction={
                    "assignee": "manual_triage",
                    "suggested_assignee": "frontend-team@example.com",
                    "ranked_candidates": ["frontend-team@example.com", "mobile-owner@example.com"],
                    "confidence": 0.72,
                    "routing_status": "needs_manual_triage",
                    "fallback_reason": "open_set_unknown_risk",
                },
                final_assignee="mobile-owner@example.com",
                reviewer="triager@example.com",
                created_at="2026-07-10T10:00:00+00:00",
                known_owner=True,
                owner_status="known_active",
            )

            append_assignee_feedback(feedback_path, record)
            rows = read_assignee_feedback(feedback_path)
            history_rows = feedback_rows_to_history_rows(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["final_assignee"], "mobile-owner@example.com")
        self.assertTrue(rows[0]["known_owner"])
        self.assertEqual(rows[0]["owner_status"], "known_active")
        self.assertEqual(history_rows[0]["assignee"], "mobile-owner@example.com")
        self.assertEqual(history_rows[0]["component"], "mobile")

    def test_assignee_feedback_requires_ticket_id_for_auditable_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "assignee_feedback.jsonl"
            record = build_assignee_feedback_record(
                ticket_json={"component": "mobile", "title": "Missing ticket id"},
                prediction={"assignee": "manual_triage"},
                final_assignee="mobile-owner@example.com",
            )

            with self.assertRaisesRegex(ValueError, "ticket_id is required"):
                append_assignee_feedback(feedback_path, record)

    def test_assignee_feedback_latest_event_uses_absolute_timestamp(self) -> None:
        earlier = build_assignee_feedback_record(
            ticket_json={"ticket_id": "RAW-TZ", "component": "mobile"},
            prediction={"assignee": "manual_triage"},
            final_assignee="earlier-owner@example.com",
            created_at="2026-07-10T10:00:00+08:00",
        )
        later = build_assignee_feedback_record(
            ticket_json={"ticket_id": "RAW-TZ", "component": "mobile"},
            prediction={"assignee": "manual_triage"},
            final_assignee="later-owner@example.com",
            created_at="2026-07-10T03:00:00+00:00",
        )

        history_rows = feedback_rows_to_history_rows([earlier, later])

        self.assertEqual(history_rows[0]["assignee"], "later-owner@example.com")

    def test_feedback_profile_is_time_gated_and_replaces_same_ticket_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "assignee_history.jsonl"
            history.write_text(
                '{"ticket_id":"H-1","component":"mobile","title":"Old mobile issue",'
                '"description":"Old label.","assignee":"old-owner@example.com"}\n',
                encoding="utf-8",
            )
            feedback = tmp_path / "feedback.jsonl"
            append_assignee_feedback(
                feedback,
                build_assignee_feedback_record(
                    ticket_json={
                        "ticket_id": "H-1",
                        "component": "mobile",
                        "title": "Old mobile issue",
                        "description": "Old label.",
                    },
                    prediction={"assignee": "old-owner@example.com"},
                    final_assignee="new-owner@example.com",
                    created_at="2026-07-10T10:00:00+00:00",
                ),
            )
            config = PipelineConfig(
                project_root=PROJECT_ROOT,
                assignee_dataset_path=history,
                assignee_feedback_path=feedback,
                assignee_allow_uncalibrated_auto_assignment=True,
                component_owner_mapping={},
                save_checkpoints=False,
            )
            triager = AssigneeTriager(config=config)
            before_review = triager.assign(
                {
                    "ticket_id": "RAW-BEFORE-REVIEW",
                    "component": "mobile",
                    "created_at": "2026-07-10T09:00:00+00:00",
                    "title": "Old mobile issue repeats",
                    "description": "Old label repeats.",
                },
                {"predicted_priority": "P3"},
            )
            after_review = triager.assign(
                {
                    "ticket_id": "RAW-AFTER-REVIEW",
                    "component": "mobile",
                    "created_at": "2026-07-10T11:00:00+00:00",
                    "title": "Old mobile issue repeats",
                    "description": "Old label repeats.",
                },
                {"predicted_priority": "P3"},
            )
            merged, summary = merge_feedback_into_history(
                [{"ticket_id": "H-1", "component": "mobile", "assignee": "old-owner@example.com"}],
                read_assignee_feedback(feedback),
            )

        self.assertEqual(before_review["assignee"], "old-owner@example.com")
        self.assertEqual(after_review["assignee"], "new-owner@example.com")
        self.assertEqual(after_review["profile_stats"]["feedback_history_rows"], 1)
        self.assertEqual(merged[0]["assignee"], "new-owner@example.com")
        self.assertEqual(summary["feedback_replacements"], 1)

def _raw_ticket() -> dict:
    return {
        "ticket_id": "RAW-2",
        "title": "Login page crashes when token is missing",
        "description": "Login crashes with TypeError when token is None.",
        "component": "authentication",
        "bug_type": "runtime_error",
        "logs": "TypeError: token is None at src/auth/validator.py:2",
        "steps_to_reproduce": ["Submit login without token"],
        "expected_behavior": "Show validation error.",
        "actual_behavior": "Application crashes.",
    }


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    source = repo / "src" / "auth"
    source.mkdir(parents=True)
    (source / "validator.py").write_text(
        "def validate_token(token):\n"
        "    return token.strip()\n",
        encoding="utf-8",
    )
    return repo


if __name__ == "__main__":
    unittest.main()

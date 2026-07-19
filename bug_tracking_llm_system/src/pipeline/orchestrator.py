from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import PipelineConfig
from modules.assignee_triager import AssigneeTriager
from modules.bug_localizer import BugLocalizer
from modules.commit_message_generator import CommitMessageGenerator
from modules.duplicate_detector import DuplicateDetector
from modules.patch_generator import PatchGenerator
from modules.priority_classifier import PriorityClassifier
from modules.regression_tester import RegressionTester
from modules.test_generator import TestGenerator
from modules.ticket_extractor import TicketExtractor
from utils.fault_localization import format_user_facing_localization_result
from utils.llm_client import OllamaClient
from utils.logger import build_logger


class PipelineOrchestrator:
    """Connect all modules in the order defined by the flow plan."""

    def __init__(
        self,
        ticket_extractor: TicketExtractor,
        duplicate_detector: DuplicateDetector,
        priority_classifier: PriorityClassifier,
        assignee_triager: AssigneeTriager,
        bug_localizer: BugLocalizer,
        patch_generator: PatchGenerator,
        test_generator: TestGenerator,
        regression_tester: RegressionTester,
        commit_message_generator: CommitMessageGenerator,
        logger: logging.Logger,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        self.ticket_extractor = ticket_extractor
        self.duplicate_detector = duplicate_detector
        self.priority_classifier = priority_classifier
        self.assignee_triager = assignee_triager
        self.bug_localizer = bug_localizer
        self.patch_generator = patch_generator
        self.test_generator = test_generator
        self.regression_tester = regression_tester
        self.commit_message_generator = commit_message_generator
        self.logger = logger
        self.config = config or PipelineConfig()

    def run_pipeline(self, raw_ticket: dict[str, Any], repository_path: str) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "running", "errors": [], "warnings": []}
        checkpoint_dir: Path | None = None

        try:
            self.logger.info("Step 1: Extract ticket JSON")
            structured_ticket = self.ticket_extractor.extract(raw_ticket)
            result["structured_ticket"] = structured_ticket
            checkpoint_dir = self._checkpoint_dir(structured_ticket)
            self._write_checkpoint(checkpoint_dir, "structured_ticket.json", structured_ticket)
        except Exception as exc:
            result["status"] = "failed"
            result["failed_step"] = "ticket_extraction"
            result["errors"].append(str(exc))
            return result

        try:
            self.logger.info("Step 2: Detect duplicate")
            duplicate_result = self.duplicate_detector.detect(structured_ticket)
            result["duplicate"] = duplicate_result
            self._write_checkpoint(checkpoint_dir, "duplicate_detection_result.json", duplicate_result)

            if duplicate_result.get("is_duplicate"):
                result["status"] = "duplicate"
                result["duplicate_of"] = duplicate_result.get("duplicate_of")
                self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
                return result
        except Exception as exc:
            result["warnings"].append(f"Duplicate detection failed: {exc}")
            duplicate_result = {"is_duplicate": False, "top_k_candidates": [], "fallback_used": True}
            result["duplicate"] = duplicate_result

        try:
            self.logger.info("Step 3: Predict priority")
            priority_result = self.priority_classifier.predict(structured_ticket, duplicate_result.get("top_k_candidates", []))
            result["priority"] = priority_result
            self._write_checkpoint(checkpoint_dir, "priority_prediction_result.json", priority_result)
        except Exception as exc:
            result["warnings"].append(f"Priority prediction failed: {exc}")
            priority_result = {"predicted_priority": "P3", "confidence": 0.0, "fallback_used": True}
            result["priority"] = priority_result

        try:
            self.logger.info("Step 4: Assign developer")
            assignee_result = self.assignee_triager.assign(structured_ticket, priority_result)
            result["assignee"] = assignee_result
            self._write_checkpoint(checkpoint_dir, "assignee_prediction_result.json", assignee_result)
        except Exception as exc:
            result["warnings"].append(f"Assignee triage failed: {exc}")
            assignee_result = {
                "assignee": "manual_triage",
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "calibration_status": "not_run_due_to_assignee_exception",
                "calibration_artifact": "",
                "reason": "Assignee model failed; manual triage is required.",
                "ranked_candidates": ["manual_triage"],
                "candidate_scores": {},
                "candidate_details": [],
                "suggested_assignee": "",
                "routing_status": "needs_manual_triage",
                "needs_manual_triage": True,
                "fallback_used": True,
                "fallback_reason": "assignee_model_exception",
                "profile_stats": {},
                "open_set_status": "not_run_due_to_assignee_exception",
                "open_set_detector": "",
            }
            result["assignee"] = assignee_result

        try:
            self.logger.info("Step 5: Localize bug")
            bug_location = self.bug_localizer.localize(structured_ticket, repository_path)
            result["bug_location"] = bug_location
            result["bug_location_user_facing"] = format_user_facing_localization_result(bug_location)
            self._write_checkpoint(checkpoint_dir, "bug_location_result.json", bug_location)
            self._write_checkpoint(checkpoint_dir, "bug_location_user_facing_result.json", result["bug_location_user_facing"])
        except Exception as exc:
            result["status"] = "needs_manual_review"
            result["failed_step"] = "bug_localization"
            result["errors"].append(str(exc))
            self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
            return result

        try:
            self.logger.info("Step 6: Generate patch")
            patch = self.patch_generator.generate(structured_ticket, bug_location, repository_path)
            result["patch"] = patch
            self._write_checkpoint(checkpoint_dir, "patch_result.json", patch)
            if patch.get("requires_manual_patch"):
                result["status"] = "needs_manual_patch"
                result["failed_step"] = "patch_generation"
                result["errors"].append(patch.get("explanation", "Patch generation requires manual work."))
                self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
                return result
        except Exception as exc:
            result["status"] = "needs_manual_patch"
            result["failed_step"] = "patch_generation"
            result["errors"].append(str(exc))
            self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
            return result

        try:
            self.logger.info("Step 7: Generate reproduction tests")
            reproduction_tests = self.test_generator.generate_tests(structured_ticket, patch)
            result["tests"] = reproduction_tests
            self._write_checkpoint(checkpoint_dir, "generated_tests_result.json", reproduction_tests)
        except Exception as exc:
            result["warnings"].append(f"Test generation failed: {exc}")
            reproduction_tests = {
                "generated_tests": [],
                "fib_passed_tests": [],
                "test_result": "not_generated",
                "fallback_used": True,
            }
            result["tests"] = reproduction_tests

        try:
            self.logger.info("Step 8: Run regression tests")
            regression_result = self.regression_tester.run(patch, repository_path)
            result["regression"] = regression_result
            self._write_checkpoint(checkpoint_dir, "regression_test_result.json", regression_result)
            if regression_result.get("regression_result") == "failed":
                result["status"] = "patch_unverified"
                result["failed_step"] = "regression_testing"
                result["errors"].append("Regression testing failed or patch could not be applied.")
                self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
                return result
            if regression_result.get("regression_result") == "not_run":
                result["warnings"].append("Regression tests were not run; patch was only checked structurally.")
        except Exception as exc:
            result["status"] = "patch_unverified"
            result["failed_step"] = "regression_testing"
            result["errors"].append(str(exc))
            self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
            return result

        try:
            self.logger.info("Step 9: Generate commit message")
            commit_message = self.commit_message_generator.generate(
                structured_ticket,
                patch,
                {"reproduction_tests": reproduction_tests, "regression_result": regression_result},
            )
            result["commit_message"] = commit_message
            self._write_checkpoint(checkpoint_dir, "commit_message_result.json", commit_message)
        except Exception as exc:
            result["warnings"].append(f"Commit message generation failed: {exc}")
            result["commit_message"] = {
                "commit_message": "fix: resolve reported bug",
                "commit_body": "Generated fallback commit message.",
                "fallback_used": True,
            }

        result["status"] = "completed"
        self._write_checkpoint(checkpoint_dir, "final_pipeline_result.json", result)
        return result

    def _checkpoint_dir(self, structured_ticket: dict[str, Any]) -> Path | None:
        if not self.config.save_checkpoints:
            return None
        ticket_id = str(structured_ticket.get("ticket_id") or "unknown").replace("/", "_")
        path = self.config.processed_ticket_dir / ticket_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_checkpoint(self, directory: Path | None, filename: str, payload: dict[str, Any]) -> None:
        if directory is None or not self.config.save_checkpoints:
            return
        with (directory / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_default_orchestrator(config: PipelineConfig | None = None) -> PipelineOrchestrator:
    cfg = config or PipelineConfig()
    fault_localization_llm = (
        OllamaClient(
            url=cfg.fault_localization_ollama_url,
            model=cfg.fault_localization_ollama_model,
            timeout=cfg.fault_localization_ollama_timeout,
        )
        if cfg.fault_localization_llm_rerank
        else None
    )
    return PipelineOrchestrator(
        ticket_extractor=TicketExtractor(prompt_path=cfg.project_root / "src" / "prompts" / "extract_ticket_prompt.txt"),
        duplicate_detector=DuplicateDetector(config=cfg),
        priority_classifier=PriorityClassifier(),
        assignee_triager=AssigneeTriager(config=cfg),
        bug_localizer=BugLocalizer(
            code_index_path=cfg.fault_localization_code_index_path,
            top_k=cfg.fault_localization_top_k,
            embedding_backend=cfg.fault_localization_embedding_backend,
            sbert_model=cfg.fault_localization_sbert_model,
            sbert_local_files_only=cfg.fault_localization_sbert_local_files_only,
            sbert_cache_dir=cfg.fault_localization_sbert_cache_dir,
            llm_client=fault_localization_llm,
            llm_rerank=cfg.fault_localization_llm_rerank,
            llm_candidate_k=cfg.fault_localization_llm_candidate_k,
            llm_cache_dir=cfg.fault_localization_llm_cache_dir,
        ),
        patch_generator=PatchGenerator(prompt_path=cfg.project_root / "src" / "prompts" / "patch_prompt.txt"),
        test_generator=TestGenerator(),
        regression_tester=RegressionTester(config=cfg),
        commit_message_generator=CommitMessageGenerator(),
        logger=build_logger(),
        config=cfg,
    )

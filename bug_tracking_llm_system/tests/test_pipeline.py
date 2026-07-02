from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import PipelineConfig
from modules.assignee_triager import AssigneeTriager
from modules.ticket_extractor import TicketExtractor
from pipeline.orchestrator import build_default_orchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    def test_pipeline_completes_with_provided_patch(self) -> None:
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

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["patch"]["modified_files"], ["src/auth/validator.py"])
        self.assertEqual(result["regression"]["patch_apply"], "passed")
        self.assertTrue(result["commit_message"]["commit_message"].startswith("fix(authentication):"))

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

    def test_assignee_triager_uses_text_similarity_for_unmapped_component(self) -> None:
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

        self.assertEqual(result["assignee"], "frontend-team@example.com")
        self.assertIn("frontend-team@example.com", result["ranked_candidates"])
        self.assertIn("text_similarity", result["reason"])

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

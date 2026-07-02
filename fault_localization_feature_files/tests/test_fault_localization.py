from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.fault_localization import build_code_index, localize_ticket

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_fault_localization import evaluate_records
from scripts.prepare_fault_localization_gold import prepare_gold_records
from scripts.prepare_swebench_lite_fault_localization import gold_record, normalize_record, ticket_record


class FaultLocalizationTests(unittest.TestCase):
    def test_build_code_index_extracts_python_symbols_and_skips_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)

        function_names = {chunk.function_name for chunk in index.chunks}
        indexed_files = {chunk.file_path for chunk in index.chunks}
        self.assertIn("validate_token", function_names)
        self.assertIn("src/auth/validator.py", indexed_files)
        self.assertNotIn("tests/test_validator.py", indexed_files)

    def test_localize_ticket_ranks_stack_trace_chunk_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "RAW-2",
                    "title": "Login page crashes when token is missing",
                    "description": "Login crashes with TypeError when token is None.",
                    "component": "authentication",
                    "logs": "TypeError: token is None at src/auth/validator.py:2",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "src/auth/validator.py")
        self.assertEqual(best["function_name"], "validate_token")
        self.assertIn("symbol_qualified_name", best)
        self.assertIn("scoring_signals", best)
        self.assertIn("final_score", best["scoring_signals"])
        self.assertEqual(result["bug_location"]["file"], "src/auth/validator.py")
        self.assertIn("embedding_backend", result["method"])
        self.assertIn("evaluation_ready_fields", result)

    def test_localize_ticket_rejects_non_positive_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)

            for top_k in (0, -1):
                with self.subTest(top_k=top_k):
                    with self.assertRaisesRegex(ValueError, "top_k"):
                        localize_ticket(
                            {"ticket_id": "RAW-3", "title": "Login fails"},
                            code_index=index,
                            top_k=top_k,
                        )

    def test_evaluate_records_reports_top_k_and_mrr(self) -> None:
        gold = [
            {"ticket_id": "A", "fixed_files": ["src/auth/validator.py"], "fixed_symbols": ["validate_token"]},
            {"ticket_id": "B", "fixed_files": ["src/profile/view.py"], "fixed_symbols": ["ProfileView.render"]},
        ]
        pred = [
            {
                "ticket_id": "A",
                "localized_candidates": [
                    {"file_path": "src/other.py", "function_name": "other"},
                    {"file_path": "src/auth/validator.py", "function_name": "validate_token"},
                ],
            },
            {
                "ticket_id": "B",
                "localized_candidates": [
                    {"file_path": "src/profile/view.py", "symbol_qualified_name": "ProfileView.render"},
                ],
            },
        ]

        metrics = evaluate_records(gold, pred)

        self.assertEqual(metrics["rows_with_file_ground_truth"], 2)
        self.assertEqual(metrics["rows_with_symbol_ground_truth"], 2)
        self.assertEqual(metrics["top_1_accuracy"], 0.5)
        self.assertEqual(metrics["top_3_accuracy"], 1.0)
        self.assertEqual(metrics["top_5_accuracy"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)
        self.assertEqual(metrics["symbol_top_1_accuracy"], 0.5)
        self.assertEqual(metrics["symbol_top_3_accuracy"], 1.0)
        self.assertEqual(metrics["symbol_mrr"], 0.75)

    def test_prepare_gold_records_normalizes_files_and_symbols(self) -> None:
        rows = prepare_gold_records(
            [
                {
                    "ticket_id": "A",
                    "patch": {"modified_files": ["./src/auth/validator.py"]},
                    "bug_location": {"function": "AuthValidator.validate_token"},
                }
            ]
        )

        self.assertEqual(rows[0]["fixed_files"], ["src/auth/validator.py"])
        self.assertEqual(rows[0]["fixed_symbols"], ["AuthValidator.validate_token"])

    def test_swebench_lite_records_map_patch_to_fault_localization_gold(self) -> None:
        raw = {
            "repo": "example/project",
            "instance_id": "example__project-1",
            "base_commit": "abc123",
            "problem_statement": "Parser crashes on empty input\n\nDetails...",
            "patch": (
                "diff --git a/src/parser.py b/src/parser.py\n"
                "--- a/src/parser.py\n"
                "+++ b/src/parser.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-bad\n"
                "+good\n"
            ),
            "FAIL_TO_PASS": '["tests/test_parser.py::test_empty"]',
            "PASS_TO_PASS": "[]",
        }

        record = normalize_record(raw, split="test")
        ticket = ticket_record(record)
        gold = gold_record(record)

        self.assertEqual(ticket["ticket_id"], "example__project-1")
        self.assertEqual(ticket["title"], "Parser crashes on empty input")
        self.assertEqual(ticket["repo"], "example/project")
        self.assertEqual(ticket["fail_to_pass"], ["tests/test_parser.py::test_empty"])
        self.assertEqual(gold["fixed_files"], ["src/parser.py"])
        self.assertEqual(gold["base_commit"], "abc123")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    source = repo / "src" / "auth"
    source.mkdir(parents=True)
    (source / "validator.py").write_text(
        "def validate_token(token):\n"
        "    return token.strip()\n",
        encoding="utf-8",
    )
    profile = repo / "src" / "profile"
    profile.mkdir(parents=True)
    (profile / "view.py").write_text(
        "def render_profile(user):\n"
        "    return user.name\n",
        encoding="utf-8",
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_validator.py").write_text(
        "def test_validate_token():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    return repo


if __name__ == "__main__":
    unittest.main()

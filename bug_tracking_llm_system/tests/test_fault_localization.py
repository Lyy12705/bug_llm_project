from __future__ import annotations

import json
import sys
import tempfile
import unittest
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.fault_localization import (
    _source_path_hints,
    build_code_index,
    format_user_facing_localization_result,
    format_user_facing_validation_error,
    localize_ticket,
    validate_localization_request,
)

from scripts.evaluate_fault_localization import evaluate_records
from scripts.analyze_fault_localization_failures import analyze_records
from scripts.analyze_candidate_pool_recall import analyze_candidate_pool_recall
from scripts.run_controlled_llm_rerank_subset import build_comparison, select_top5_misses
from scripts.run_fault_localization_internal_beta import classify_warnings, summarize_result_rows
from scripts.prepare_fault_localization_gold import prepare_gold_records
from scripts.prepare_swebench_lite_fault_localization import gold_record, normalize_record, ticket_record
from scripts.run_swebench_lite_fault_localization import BatchRunConfig, _index_path, _ticket_id_filter, run_batch


class FakeRerankClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        self.last_prompt = prompt
        return {
            "candidates": [
                {
                    "candidate_id": "C2",
                    "confidence": 1.0,
                    "reason": "LLM selected the more relevant second candidate.",
                    "evidence": ["src/profile/view.py:1-2"],
                    "needs_more_context": False,
                },
                {"candidate_id": "C1", "confidence": 0.1, "reason": "LLM demoted the first candidate."},
            ]
        }


class FlakyRerankClient:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return {"notes": "I need more context, but this is not the required schema."}
        return {
            "candidates": [
                {
                    "candidate_id": "C2",
                    "confidence": 0.95,
                    "reason": "Retry selected the candidate whose code directly matches the report.",
                    "evidence": ["src/profile/view.py:1-2"],
                    "needs_more_context": False,
                },
                {"rank": 1, "confidence": 0.1, "reason": "Less directly related."},
            ]
        }


class EchoRerankClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        return {
            "candidates": [
                {
                    "candidate_id": "C2",
                    "rank": 2,
                    "file_path": "src/profile/view.py",
                    "retrieval_score": 0.5,
                    "primary_code": "def render_profile(user): return user.name",
                    "repository_context": {"file_path": "src/profile/view.py"},
                }
            ]
        }


class TopLevelScoreRerankClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        return {
            "score": 0.9,
            "candidates": [
                {
                    "candidate_id": "C2",
                }
            ],
        }


class RankOnlyRerankClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str) -> dict:
        self.calls += 1
        return {"candidates": [{"rank": 2, "confidence": 0.95, "reason": "Legacy rank-only output."}]}


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

    def test_build_code_index_keeps_nested_source_models_but_skips_root_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            nested_models = repo / "django" / "db" / "models"
            nested_models.mkdir(parents=True)
            (nested_models / "fields.py").write_text(
                "def parse_model_field(value):\n"
                "    return value\n",
                encoding="utf-8",
            )
            root_models = repo / "models"
            root_models.mkdir()
            (root_models / "generated.py").write_text(
                "def generated_model():\n"
                "    return None\n",
                encoding="utf-8",
            )

            index = build_code_index(repo)

        indexed_files = {chunk.file_path for chunk in index.chunks}
        self.assertIn("django/db/models/fields.py", indexed_files)
        self.assertNotIn("models/generated.py", indexed_files)

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
        self.assertIn("repository_context", best)
        self.assertIn("supporting_chunks", best["repository_context"])
        self.assertEqual(
            best["repository_context"]["context_strategy"],
            "retrieval_first_lightweight_repository_context",
        )
        self.assertEqual(result["bug_location"]["file"], "src/auth/validator.py")
        self.assertEqual(result["confidence_level"], "high")
        self.assertFalse(result["should_manual_review"])
        self.assertTrue(result["recommend_patch_generation"])
        self.assertEqual(result["bug_location"]["confidence_level"], "high")
        self.assertIn("top1_top2_margin", result["confidence"])
        self.assertIn("stack_trace_score", result["uncertainty_reason"])
        self.assertIn("embedding_backend", result["method"])
        self.assertIn("evaluation_ready_fields", result)

    def test_confidence_gate_marks_weak_localization_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "alpha.py").write_text(
                "def alpha(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            (source / "beta.py").write_text(
                "def beta(value):\n"
                "    return value - 1\n",
                encoding="utf-8",
            )
            result = localize_ticket(
                {
                    "ticket_id": "RAW-LOW-CONFIDENCE",
                    "title": "Unexpected behavior",
                    "description": "Something is wrong but the report has no file path, stack trace, or identifier.",
                },
                repo_path=repo,
                embedding_backend="tfidf",
                top_k=2,
            )

        self.assertIn(result["confidence_level"], {"medium", "low"})
        self.assertTrue(result["should_manual_review"])
        self.assertFalse(result["recommend_patch_generation"])
        self.assertIn("uncertainty_reason", result)

    def test_confidence_gate_requires_manual_review_for_test_like_top_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            test_dir = repo / "testing" / "example_scripts" / "config"
            test_dir.mkdir(parents=True)
            (test_dir / "test_foo.py").write_text(
                "def test_collect_pytest_prefix():\n"
                "    assert collect_pytest_prefix('foo') == 'foo'\n",
                encoding="utf-8",
            )
            source_dir = repo / "src" / "_pytest"
            source_dir.mkdir(parents=True)
            (source_dir / "python.py").write_text(
                "def collect_pytest_prefix(value):\n"
                "    return value\n",
                encoding="utf-8",
            )

            result = localize_ticket(
                {
                    "ticket_id": "PYTEST-TEST-PATH",
                    "title": "collect pytest prefix fails",
                    "description": "Failure is reported from testing/example_scripts/config/test_foo.py:1.",
                    "logs": "testing/example_scripts/config/test_foo.py:1: AssertionError",
                },
                repo_path=repo,
                embedding_backend="tfidf",
                top_k=3,
            )

        top_file = result["localized_candidates"][0]["file_path"]
        self.assertTrue(top_file.startswith("testing/"))
        self.assertNotEqual(result["confidence_level"], "high")
        self.assertTrue(result["should_manual_review"])
        self.assertFalse(result["recommend_patch_generation"])
        self.assertIn("test/demo path", result["uncertainty_reason"])

    def test_user_facing_formatter_exposes_safe_beta_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)
            ticket = {
                "ticket_id": "RAW-USER-FACING",
                "title": "Login page crashes when token is missing",
                "description": "Login crashes with TypeError when token is None.",
                "component": "authentication",
                "logs": "TypeError: token is None at src/auth/validator.py:2",
            }
            result = localize_ticket(ticket, code_index=index, embedding_backend="tfidf", top_k=3)
            formatted = format_user_facing_localization_result(result, top_k=2)

        self.assertEqual(formatted["ticket_id"], "RAW-USER-FACING")
        self.assertEqual(formatted["summary"]["confidence_level"], "high")
        self.assertFalse(formatted["summary"]["should_manual_review"])
        self.assertTrue(formatted["summary"]["recommend_patch_generation"])
        self.assertEqual(len(formatted["top_k_suspicious_files"]), 2)
        self.assertEqual(formatted["top_k_suspicious_files"][0]["file_path"], "src/auth/validator.py")
        self.assertTrue(formatted["top_k_suspicious_files"][0]["patch_generator_eligible"])
        self.assertIn("engineering_guidance", formatted)
        self.assertEqual(formatted["method"]["retrieval_first"], True)

    def test_input_validation_and_user_facing_error_result_are_explainable(self) -> None:
        validation = validate_localization_request(
            {"ticket_id": "BAD", "description": "bad"},
            repo_path="/path/that/does/not/exist",
            min_ticket_chars=20,
        )
        formatted = format_user_facing_validation_error(validation)

        self.assertFalse(validation["is_valid"])
        self.assertEqual(validation["status"], "error")
        self.assertTrue(any("too short" in message for message in validation["errors"]))
        self.assertTrue(any("does not exist" in message for message in validation["errors"]))
        self.assertEqual(formatted["status"], "invalid_input")
        self.assertTrue(formatted["summary"]["should_manual_review"])
        self.assertFalse(formatted["summary"]["recommend_patch_generation"])

    def test_localize_ticket_returns_file_aggregated_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src" / "auth"
            source.mkdir(parents=True)
            (source / "validator.py").write_text(
                "class TokenValidator:\n"
                "    def validate_token(self, token):\n"
                "        return token.strip()\n\n"
                "def normalize_token(token):\n"
                "    return token.strip().lower()\n",
                encoding="utf-8",
            )
            (source / "session.py").write_text(
                "def start_session(user):\n"
                "    return user.id\n",
                encoding="utf-8",
            )
            result = localize_ticket(
                {
                    "ticket_id": "RAW-AGG",
                    "title": "Token validation strips missing token incorrectly",
                    "description": "The auth token validator and normalize token logic mishandle whitespace.",
                    "component": "auth",
                },
                repo_path=repo,
                embedding_backend="tfidf",
                top_k=5,
            )

        candidate_files = [candidate["file_path"] for candidate in result["localized_candidates"]]
        self.assertEqual(len(candidate_files), len(set(candidate_files)))
        self.assertIn("localized_files", result)
        self.assertEqual(candidate_files, [row["file_path"] for row in result["localized_files"]])
        self.assertIn("supporting_evidence", result["localized_files"][0])
        self.assertEqual(result["method"]["ranking_level"], "file_aggregated_chunks")

    def test_localize_ticket_supports_tfidf_sbert_rerank_backend_when_available(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            self.skipTest("sentence-transformers is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "RAW-SBERT",
                    "title": "Login token validation crashes",
                    "description": "Token validation should not call strip on missing token.",
                    "component": "authentication",
                },
                code_index=index,
                embedding_backend="tfidf-sbert-rerank",
                top_k=3,
            )

        self.assertTrue(result["localized_candidates"])
        self.assertIn("tfidf+sbert-rerank", result["method"]["embedding_backend"])

    def test_localize_ticket_writes_sbert_embedding_cache_when_enabled(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            self.skipTest("sentence-transformers is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            cache_dir = tmp_path / "embedding_cache"
            result = localize_ticket(
                {
                    "ticket_id": "RAW-SBERT-CACHE",
                    "title": "Login token validation crashes",
                    "description": "Token validation should not call strip on missing token.",
                    "component": "authentication",
                },
                code_index=index,
                embedding_backend="tfidf-sbert-rerank",
                sbert_cache_dir=cache_dir,
                top_k=3,
            )

            cache_files = list(cache_dir.glob("*.sqlite3"))
            self.assertTrue(cache_files)
            self.assertIn("cache=on", result["method"]["embedding_backend"])
            connection = sqlite3.connect(cache_files[0])
            try:
                row_count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(row_count, 0)

    def test_llm_rerank_uses_candidate_pool_and_response_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            ticket = {
                "ticket_id": "RAW-LLM-RERANK",
                "title": "Profile page and token validation issue",
                "description": "The profile rendering path should be inspected alongside auth token validation.",
                "component": "profile",
            }
            baseline = localize_ticket(ticket, code_index=index, embedding_backend="tfidf", top_k=2)
            baseline_files = [row["file_path"] for row in baseline["localized_candidates"]]
            self.assertEqual(len(baseline_files), 2)

            llm_client = FakeRerankClient()
            cache_dir = tmp_path / "llm_cache"
            reranked = localize_ticket(
                ticket,
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
                llm_cache_dir=cache_dir,
            )

            self.assertEqual(reranked["localized_candidates"][0]["file_path"], baseline_files[1])
            self.assertTrue(reranked["method"]["llm_rerank"])
            self.assertEqual(reranked["method"]["llm_candidate_k"], 2)
            self.assertEqual(llm_client.calls, 1)
            self.assertIn("repository_context", llm_client.last_prompt)
            self.assertIn("supporting_chunks", llm_client.last_prompt)
            self.assertIn("related_files", llm_client.last_prompt)
            self.assertIn("symbol_references", llm_client.last_prompt)
            self.assertIn("candidate_id", llm_client.last_prompt)
            self.assertIn("Return ONLY a JSON array", llm_client.last_prompt)
            self.assertIn('"confidence":0.0', llm_client.last_prompt)
            self.assertIn("Do not include rank", llm_client.last_prompt)
            self.assertNotIn('"rank":1', llm_client.last_prompt)
            self.assertIn("llm_rerank_score", reranked["localized_candidates"][0]["scoring_signals"])
            self.assertEqual(reranked["localized_candidates"][0]["scoring_signals"]["llm_rerank_output_mode"], "score_or_reason")
            self.assertIn("llm_rerank_evidence", reranked["localized_candidates"][0]["signals"])
            self.assertTrue((cache_dir / "llm_rerank_cache.jsonl").exists())

            cached = localize_ticket(
                ticket,
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
                llm_cache_dir=cache_dir,
            )

            self.assertEqual(cached["localized_candidates"][0]["file_path"], baseline_files[1])
            self.assertEqual(llm_client.calls, 1)
            self.assertTrue(cached["localized_candidates"][0]["scoring_signals"]["llm_rerank_cache_hit"])

    def test_llm_rerank_retries_invalid_json_schema_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            llm_client = FlakyRerankClient()
            cache_dir = tmp_path / "llm_cache"

            result = localize_ticket(
                {
                    "ticket_id": "RAW-LLM-RETRY",
                    "title": "Profile page and token validation issue",
                    "description": "The profile rendering path should be inspected alongside auth token validation.",
                    "component": "profile",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
                llm_cache_dir=cache_dir,
            )

            self.assertEqual(llm_client.calls, 2)
            self.assertIn("Previous LLM output was invalid", llm_client.prompts[1])
            self.assertIn("Candidate summary:", llm_client.prompts[1])
            self.assertNotIn("repository_context_hints", llm_client.prompts[1])
            self.assertLess(len(llm_client.prompts[1]), len(llm_client.prompts[0]))
            self.assertTrue(result["method"]["llm_rerank"])
            self.assertIn("llm_rerank_score", result["localized_candidates"][0]["scoring_signals"])
            cache_rows = (cache_dir / "llm_rerank_cache.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cache_rows), 1)
            self.assertIn('"retry_used": true', cache_rows[0])

    def test_llm_rerank_rejects_echoed_input_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            llm_client = EchoRerankClient()

            result = localize_ticket(
                {
                    "ticket_id": "RAW-LLM-ECHO",
                    "title": "Profile page and token validation issue",
                    "description": "The profile rendering path should be inspected alongside auth token validation.",
                    "component": "profile",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
            )

            self.assertEqual(llm_client.calls, 2)
            self.assertFalse(result["method"]["llm_rerank"])
            self.assertTrue(result["warnings"])
            self.assertNotIn("llm_rerank_score", result["localized_candidates"][0]["scoring_signals"])

    def test_llm_rerank_rejects_rank_only_rows_without_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            llm_client = RankOnlyRerankClient()
            cache_dir = tmp_path / "llm_cache"

            result = localize_ticket(
                {
                    "ticket_id": "RAW-LLM-RANK-ONLY",
                    "title": "Profile page and token validation issue",
                    "description": "The profile rendering path should be inspected alongside auth token validation.",
                    "component": "profile",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
                llm_cache_dir=cache_dir,
            )

            self.assertEqual(llm_client.calls, 2)
            self.assertFalse(result["method"]["llm_rerank"])
            self.assertIn("valid candidate IDs", result["warnings"][0])
            self.assertNotIn("llm_rerank_score", result["localized_candidates"][0]["scoring_signals"])
            invalid_log = cache_dir / "llm_rerank_invalid_responses.jsonl"
            self.assertTrue(invalid_log.exists())
            invalid_rows = invalid_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(invalid_rows), 2)
            self.assertIn("RAW-LLM-RANK-ONLY", invalid_rows[0])
            self.assertIn("valid candidate IDs", invalid_rows[-1])

    def test_llm_rerank_applies_payload_level_score_to_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            index = build_code_index(repo)
            llm_client = TopLevelScoreRerankClient()

            result = localize_ticket(
                {
                    "ticket_id": "RAW-LLM-TOP-SCORE",
                    "title": "Profile page and token validation issue",
                    "description": "The profile rendering path should be inspected alongside auth token validation.",
                    "component": "profile",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=1,
                llm_client=llm_client,
                llm_rerank=True,
                llm_candidate_k=2,
            )

            self.assertTrue(result["method"]["llm_rerank"])
            signals = result["localized_candidates"][0]["scoring_signals"]
            self.assertEqual(signals["llm_rerank_score"], 0.9)
            self.assertEqual(signals["llm_rerank_output_mode"], "score_or_reason")

    def test_repository_context_includes_lightweight_import_and_symbol_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            auth = repo / "src" / "auth"
            auth.mkdir(parents=True)
            (auth / "validator.py").write_text(
                "def validate_token(token):\n"
                "    return token.strip()\n",
                encoding="utf-8",
            )
            profile = repo / "src" / "profile"
            profile.mkdir(parents=True)
            (profile / "view.py").write_text(
                "from src.auth.validator import validate_token\n\n"
                "def render_profile(user):\n"
                "    return validate_token(user.token)\n",
                encoding="utf-8",
            )

            result = localize_ticket(
                {
                    "ticket_id": "RAW-GRAPH",
                    "title": "Token validator strips missing profile token",
                    "description": "The profile renderer calls validate_token and crashes when token is None.",
                    "component": "auth",
                    "logs": "TypeError at src/auth/validator.py:2",
                },
                repo_path=repo,
                embedding_backend="tfidf",
                top_k=3,
            )

            auth_candidate = next(
                row for row in result["localized_candidates"] if row["file_path"] == "src/auth/validator.py"
            )
            context = auth_candidate["repository_context"]
            related_files = {row["file_path"] for row in context.get("related_files", [])}
            referenced_files = {row["file_path"] for row in context.get("symbol_references", [])}
            self.assertIn("src/profile/view.py", related_files | referenced_files)
            self.assertTrue(context.get("same_file_symbols"))
            signals = auth_candidate["scoring_signals"]
            self.assertGreater(signals["repository_proximity_score"], 0.0)
            self.assertIn("Repository proximity", auth_candidate["reason"])
            without_proximity = localize_ticket(
                {
                    "ticket_id": "RAW-GRAPH",
                    "title": "Token validator strips missing profile token",
                    "description": "The profile renderer calls validate_token and crashes when token is None.",
                    "component": "auth",
                    "logs": "TypeError at src/auth/validator.py:2",
                },
                repo_path=repo,
                embedding_backend="tfidf",
                top_k=3,
                repository_proximity=False,
            )
            without_auth_candidate = next(
                row for row in without_proximity["localized_candidates"] if row["file_path"] == "src/auth/validator.py"
            )
            self.assertEqual(without_auth_candidate["scoring_signals"]["repository_proximity_score"], 0.0)
            self.assertNotIn("repository_proximity_rerank", without_proximity["method"]["stages"])

    def test_localize_ticket_uses_path_hint_to_avoid_wrapper_frame_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            rst = repo / "astropy" / "io" / "ascii"
            rst.mkdir(parents=True)
            (rst / "rst.py").write_text(
                "class RST:\n"
                "    def write(self, table, header_rows=None):\n"
                "        return 'restructured text table with header rows'\n",
                encoding="utf-8",
            )
            table = repo / "astropy" / "table"
            table.mkdir(parents=True)
            (table / "connect.py").write_text(
                "class TableWrite:\n"
                "    def __call__(self, *args, **kwargs):\n"
                "        return registry_write(*args, **kwargs)\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "SWE-RST",
                    "title": "Please support header rows in RestructuredText output",
                    "description": 'tbl.write(sys.stdout, format="ascii.rst", header_rows=["name", "unit"]) fails.',
                    "logs": 'Traceback\n  File "astropy/table/connect.py", line 2, in __call__',
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "astropy/io/ascii/rst.py")
        self.assertGreater(best["scoring_signals"]["path_hint_score"], 0)

    def test_localize_ticket_uses_identifier_and_domain_signals_for_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            conf = repo / "django" / "conf"
            conf.mkdir(parents=True)
            (conf / "global_settings.py").write_text(
                "FILE_UPLOAD_PERMISSIONS = 0o644\n"
                "FILE_UPLOAD_HANDLERS = []\n",
                encoding="utf-8",
            )
            files = repo / "django" / "core" / "files"
            files.mkdir(parents=True)
            (files / "storage.py").write_text(
                "class FileSystemStorage:\n"
                "    def file_permissions_mode(self):\n"
                "        return 'uploaded file permissions mode'\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "SWE-SETTINGS",
                    "title": "Set default FILE_UPLOAD_PERMISSIONS to 0o644.",
                    "description": "Default file upload permissions should be configured in settings.py.",
                    "component": "django",
                    "repo": "django/django",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "django/conf/global_settings.py")
        self.assertGreater(best["scoring_signals"]["identifier_score"], 0)
        self.assertGreater(best["scoring_signals"]["domain_path_score"], 0)

    def test_localize_ticket_uses_explicit_module_function_path_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            admin = repo / "django" / "contrib" / "admin"
            admin.mkdir(parents=True)
            (admin / "utils.py").write_text(
                "def display_for_field(value, field):\n"
                "    return field.prepare_value(value)\n",
                encoding="utf-8",
            )
            forms = repo / "django" / "forms"
            forms.mkdir(parents=True)
            (forms / "fields.py").write_text(
                "class JSONField:\n"
                "    def prepare_value(self, value):\n"
                "        return value\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "SWE-ADMIN",
                    "title": "JSONField readonly admin display is not valid JSON",
                    "description": "The fix should add a special case in django.contrib.admin.utils.display_for_field.",
                    "component": "django",
                    "repo": "django/django",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "django/contrib/admin/utils.py")
        self.assertGreater(best["scoring_signals"]["path_hint_score"], 0)

    def test_localize_ticket_uses_failing_test_path_source_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            pprint_dir = repo / "sklearn" / "utils"
            pprint_dir.mkdir(parents=True)
            (pprint_dir / "_pprint.py").write_text(
                "def estimator_repr(value):\n"
                "    return 'pretty print changed only vector values'\n",
                encoding="utf-8",
            )
            linear = repo / "sklearn" / "linear_model"
            linear.mkdir(parents=True)
            (linear / "logistic.py").write_text(
                "class LogisticRegressionCV:\n"
                "    pass\n",
                encoding="utf-8",
            )

            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "SWE-PPRINT",
                    "title": "bug in print_changed_only in new repr: vector values",
                    "description": "LogisticRegressionCV with print_changed_only raises ValueError in repr.",
                    "repo": "scikit-learn/scikit-learn",
                    "fail_to_pass": ["sklearn/utils/tests/test_pprint.py::test_changed_only"],
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "sklearn/utils/_pprint.py")
        self.assertGreater(best["scoring_signals"]["candidate_expansion_score"], 0)

    def test_localize_ticket_uses_same_package_near_path_rerank_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            core = repo / "pkg" / "core"
            core.mkdir(parents=True)
            (core / "request.py").write_text(
                "def handle_request(option, value):\n"
                "    return 'generic request handler option value'\n",
                encoding="utf-8",
            )
            (core / "beta.py").write_text(
                "def resolve_beta(value):\n"
                "    return value\n",
                encoding="utf-8",
            )
            (repo / "pkg" / "other").mkdir(parents=True)
            (repo / "pkg" / "other" / "beta.py").write_text(
                "def unrelated_beta(value):\n"
                "    return value\n",
                encoding="utf-8",
            )

            index = build_code_index(repo)
            result = localize_ticket(
                {
                    "ticket_id": "PKG-NEAR-PATH",
                    "title": "pkg.core beta resolver drops empty option",
                    "description": "The beta resolver in pkg.core should preserve an empty value.",
                    "component": "pkg.core",
                },
                code_index=index,
                embedding_backend="tfidf",
                top_k=3,
            )

        best = result["localized_candidates"][0]
        self.assertEqual(best["file_path"], "pkg/core/beta.py")
        self.assertGreater(best["scoring_signals"]["package_proximity_score"], 0)
        self.assertIn("Same-package/near-path rerank signals", best["reason"])

    def test_source_path_hints_generalize_test_to_source_mapping(self) -> None:
        hints = dict(
            _source_path_hints(
                "acme widget fails in tests/test_ext_napoleon_docstring.py when parsing docstrings."
            )
        )

        self.assertIn("acme/ext/napoleon/docstring.py", hints)
        self.assertIn("src/acme/ext/napoleon/docstring.py", hints)
        self.assertNotIn("sphinx/ext/napoleon/docstring.py", hints)

    def test_source_path_hints_infer_underscored_package_variant(self) -> None:
        hints = dict(
            _source_path_hints(
                "hydra reports a loader regression from tests/test_loader.py in the package internals."
            )
        )

        self.assertIn("src/_hydra/loader.py", hints)
        self.assertNotIn("src/_pytest/pathlib.py", hints)

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

    def test_evaluate_records_reports_per_repo_metrics(self) -> None:
        gold = [
            {"ticket_id": "django__django-1", "repo": "django/django", "fixed_files": ["django/conf/settings.py"]},
            {"ticket_id": "django__django-2", "repo": "django/django", "fixed_files": ["django/db/models.py"]},
            {"ticket_id": "sympy__sympy-1", "repo": "sympy/sympy", "fixed_files": ["sympy/core/add.py"]},
        ]
        pred = [
            {"ticket_id": "django__django-1", "repo": "django/django", "localized_candidates": [{"file_path": "django/conf/settings.py"}]},
            {"ticket_id": "django__django-2", "repo": "django/django", "localized_candidates": [{"file_path": "django/urls/base.py"}]},
            {"ticket_id": "sympy__sympy-1", "repo": "sympy/sympy", "localized_candidates": [{"file_path": "sympy/core/add.py"}]},
        ]

        metrics = evaluate_records(gold, pred)
        per_repo = {row["repo"]: row for row in metrics["per_repo"]}

        self.assertEqual(per_repo["django/django"]["rows"], 2)
        self.assertEqual(per_repo["django/django"]["file_top_1_accuracy"], 0.5)
        self.assertEqual(per_repo["sympy/sympy"]["file_top_1_accuracy"], 1.0)

    def test_analyze_candidate_pool_recall_focuses_baseline_misses(self) -> None:
        gold = [
            {"ticket_id": "A", "repo": "demo/repo", "fixed_files": ["src/a.py"]},
            {"ticket_id": "B", "repo": "demo/repo", "fixed_files": ["src/b.py"]},
            {"ticket_id": "C", "repo": "demo/repo", "fixed_files": ["src/c.py"]},
        ]
        baseline = [
            {"ticket_id": "A", "localized_candidates": [{"file_path": "src/a.py"}]},
            {"ticket_id": "B", "localized_candidates": [{"file_path": "src/other.py"}]},
            {"ticket_id": "C", "localized_candidates": [{"file_path": "src/other.py"}]},
        ]
        probe = [
            {"ticket_id": "A", "localized_candidates": [{"file_path": "src/a.py"}]},
            {"ticket_id": "B", "localized_candidates": [{"file_path": "src/x.py"}, {"file_path": "src/b.py"}]},
            {"ticket_id": "C", "localized_candidates": [{"file_path": "src/y.py"}]},
        ]

        report = analyze_candidate_pool_recall(gold, probe, ks=[1, 2], focus_predictions=baseline, focus_top_k=1)

        self.assertEqual(report["summary"]["rows_evaluated"], 2)
        self.assertEqual(report["summary"]["recall"]["recall@1"], 0.0)
        self.assertEqual(report["summary"]["recall"]["recall@2"], 0.5)
        self.assertEqual(len(report["misses"]), 1)

    def test_controlled_llm_rerank_subset_selects_top5_misses_and_compares(self) -> None:
        gold = [
            {"ticket_id": "django__django-1", "repo": "django/django", "fixed_files": ["django/conf/settings.py"]},
            {"ticket_id": "sympy__sympy-1", "repo": "sympy/sympy", "fixed_files": ["sympy/core/add.py"]},
            {"ticket_id": "pytest__pytest-1", "repo": "pytest-dev/pytest", "fixed_files": ["src/_pytest/main.py"]},
        ]
        baseline = [
            {
                "ticket_id": "django__django-1",
                "repo": "django/django",
                "localized_candidates": [{"file_path": "django/urls/base.py"}],
            },
            {
                "ticket_id": "sympy__sympy-1",
                "repo": "sympy/sympy",
                "localized_candidates": [{"file_path": "sympy/core/add.py"}],
            },
            {
                "ticket_id": "pytest__pytest-1",
                "repo": "pytest-dev/pytest",
                "localized_candidates": [{"file_path": "src/_pytest/config/__init__.py"}],
            },
        ]
        llm = [
            {
                "ticket_id": "django__django-1",
                "repo": "django/django",
                "localized_candidates": [{"file_path": "django/conf/settings.py"}],
            },
            {
                "ticket_id": "pytest__pytest-1",
                "repo": "pytest-dev/pytest",
                "localized_candidates": [{"file_path": "src/_pytest/config/__init__.py"}],
            },
        ]

        selected = select_top5_misses(gold, baseline, subset_size=2, top_k=5, balance_by_repo=True)
        comparison = build_comparison(
            selected_ids=[row["ticket_id"] for row in selected],
            gold_rows=gold,
            baseline_predictions=baseline,
            llm_predictions=llm,
            top_k=5,
        )

        self.assertEqual({row["ticket_id"] for row in selected}, {"django__django-1", "pytest__pytest-1"})
        self.assertEqual(comparison["tickets_compared"], 2)
        self.assertEqual(comparison["improved"], 1)

    def test_internal_beta_summary_counts_warnings_confidence_and_patch_policy(self) -> None:
        rows = [
            {
                "ticket_id": "A",
                "confidence_level": "high",
                "status": "ready_for_engineer_review",
                "validation_status": "ok",
                "manual_review": False,
                "patch_generation": True,
                "warning_categories": [],
                "warning_count": 0,
                "fallback_used": False,
                "runtime_seconds": 1.0,
                "hit_rank": 1,
                "has_gold": True,
            },
            {
                "ticket_id": "B",
                "confidence_level": "medium",
                "status": "needs_manual_review",
                "validation_status": "warning",
                "manual_review": True,
                "patch_generation": False,
                "warning_categories": ["input_validation"],
                "warning_count": 1,
                "fallback_used": False,
                "runtime_seconds": 3.0,
                "hit_rank": None,
                "has_gold": True,
            },
        ]

        summary = summarize_result_rows(rows)
        categories = classify_warnings(["Input validation warning: missing stack trace"], "")

        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["hit_at_top_k"], 1)
        self.assertEqual(summary["manual_review_rate"], 0.5)
        self.assertEqual(summary["patch_generation_recommended_rate"], 0.5)
        self.assertEqual(summary["warning_category_counts"], {"input_validation": 1})
        self.assertEqual(summary["average_runtime_seconds"], 2.0)
        self.assertEqual(categories, ["input_validation"])

    def test_analyze_records_classifies_stack_trace_wrapper_bias(self) -> None:
        report = analyze_records(
            [{"ticket_id": "A", "fixed_files": ["pkg/real.py"]}],
            [
                {
                    "ticket_id": "A",
                    "bug_report": "format='pkg.real' fails",
                    "localized_candidates": [
                        {
                            "file_path": "pkg/registry/core.py",
                            "score": 1.0,
                            "scoring_signals": {"stack_trace_score": 0.75},
                        }
                    ],
                }
            ],
            top_k=5,
        )

        self.assertEqual(report["summary"]["miss_count"], 1)
        self.assertEqual(report["misses"][0]["likely_reason"], "stack_trace_wrapper_bias")


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

    def test_swebench_lite_runner_outputs_predictions_metrics_and_demo_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            tickets = tmp_path / "tickets.jsonl"
            gold = tmp_path / "gold.jsonl"
            predictions = tmp_path / "predictions.jsonl"
            metrics = tmp_path / "metrics.json"
            demo_cases = tmp_path / "demo_cases.json"
            failures = tmp_path / "failures.jsonl"
            tickets.write_text(
                '{"ticket_id":"SWE-1","repo":"local/repo","local_repo_path":"'
                + str(repo)
                + '","title":"Login crash when token is missing",'
                '"description":"TypeError token is None in auth validator.",'
                '"component":"authentication","logs":"TypeError at src/auth/validator.py:2"}\n',
                encoding="utf-8",
            )
            gold.write_text(
                '{"ticket_id":"SWE-1","fixed_files":["src/auth/validator.py"],"fixed_symbols":["validate_token"]}\n',
                encoding="utf-8",
            )

            summary = run_batch(
                BatchRunConfig(
                    tickets_path=tickets,
                    gold_path=gold,
                    repo_cache_dir=tmp_path / "repos",
                    index_cache_dir=tmp_path / "indexes",
                    predictions_output=predictions,
                    metrics_output=metrics,
                    demo_cases_output=demo_cases,
                    failures_output=failures,
                    checkout=False,
                    top_k=3,
                    progress_every=0,
                )
            )

            self.assertEqual(summary["predictions"], 1)
            self.assertEqual(summary["failures"], 0)
            self.assertEqual(summary["metrics"]["top_1_accuracy"], 1.0)
            self.assertTrue(predictions.exists())
            self.assertTrue(metrics.exists())
            self.assertTrue(demo_cases.exists())

            resumed = run_batch(
                BatchRunConfig(
                    tickets_path=tickets,
                    gold_path=gold,
                    repo_cache_dir=tmp_path / "repos",
                    index_cache_dir=tmp_path / "indexes",
                    predictions_output=predictions,
                    metrics_output=metrics,
                    demo_cases_output=demo_cases,
                    failures_output=failures,
                    checkout=False,
                    top_k=3,
                    resume=True,
                    progress_every=0,
                )
            )

            self.assertEqual(resumed["predictions"], 1)
            self.assertEqual(resumed["skipped_existing_predictions"], 1)
            self.assertEqual(resumed["processed_this_run"], 0)

    def test_swebench_lite_runner_skips_checkout_when_cached_index_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            ticket = {
                "ticket_id": "SWE-CACHED-INDEX",
                "repo": "local/repo",
                "local_repo_path": str(repo),
                "base_commit": "missingdeadbeef1234",
                "title": "Login crash when token is missing",
                "description": "TypeError token is None in auth validator.",
                "component": "authentication",
                "logs": "TypeError at src/auth/validator.py:2",
            }
            index_cache_dir = tmp_path / "indexes"
            cached_index = build_code_index(repo)
            index_path = _index_path(ticket, repo, index_cache_dir)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            cached_index.save(index_path)

            tickets = tmp_path / "tickets.jsonl"
            gold = tmp_path / "gold.jsonl"
            predictions = tmp_path / "predictions.jsonl"
            metrics = tmp_path / "metrics.json"
            failures = tmp_path / "failures.jsonl"
            tickets.write_text(json.dumps(ticket) + "\n", encoding="utf-8")
            gold.write_text(
                '{"ticket_id":"SWE-CACHED-INDEX","fixed_files":["src/auth/validator.py"]}\n',
                encoding="utf-8",
            )

            summary = run_batch(
                BatchRunConfig(
                    tickets_path=tickets,
                    gold_path=gold,
                    repo_cache_dir=tmp_path / "repos",
                    index_cache_dir=index_cache_dir,
                    predictions_output=predictions,
                    metrics_output=metrics,
                    demo_cases_output=None,
                    failures_output=failures,
                    checkout=True,
                    top_k=3,
                    progress_every=0,
                )
            )

        self.assertEqual(summary["predictions"], 1)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["cached_index_checkout_skips"], 1)
        self.assertEqual(summary["metrics"]["top_1_accuracy"], 1.0)

    def test_swebench_lite_runner_reads_ticket_id_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ids = Path(tmp) / "ids.json"
            ids.write_text('["A", "B"]\n', encoding="utf-8")

            result = _ticket_id_filter(["C"], str(ids))

        self.assertEqual(result, {"A", "B", "C"})


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

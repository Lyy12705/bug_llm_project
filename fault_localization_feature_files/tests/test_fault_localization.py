from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for import_path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from modules.bug_localizer import BugLocalizer
import utils.fault_localization as fault_localization
from utils.fault_localization import build_bug_report_text, build_code_index, localize_ticket
from utils.llm_client import OllamaClient

from scripts.evaluate_fault_localization import (
    discover_base_commit_existing_gold_files,
    evaluate_records,
)
from scripts.compare_fault_localization_runs import compare_runs
from scripts.prepare_fault_localization_gold import prepare_gold_records
from scripts.prepare_swebench_lite_fault_localization import gold_record, normalize_record, ticket_record
from scripts.run_swebench_lite_fault_localization import (
    BatchRunConfig,
    ProgressReporter,
    load_or_build_import_graph,
    run_batch,
)


class DuplicateRankRerankClient:
    def generate_json(self, prompt: str) -> dict:
        return {
            "candidates": [
                {"rank": 1, "score": 0.9, "reason": "first"},
                {"rank": 1, "score": 0.1, "reason": "duplicate should be ignored"},
                {"rank": 2, "score": 0.8, "reason": "second"},
            ]
        }


class PartialRerankClient:
    def generate_json(self, prompt: str) -> dict:
        return {"candidates": [{"rank": 1, "score": 0.0, "reason": "short reason"}]}


class FaultLocalizationTests(unittest.TestCase):
    def test_sbert_model_is_cached_within_process(self) -> None:
        fake_module = types.ModuleType("sentence_transformers")
        created: list[object] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, *, local_files_only: bool) -> None:
                created.append(self)

        fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
        fault_localization._load_sbert_model.cache_clear()
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            first = fault_localization._load_sbert_model("fake-model", True)
            second = fault_localization._load_sbert_model("fake-model", True)
        fault_localization._load_sbert_model.cache_clear()

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)

    def test_ollama_client_maps_curl_timeout_to_hard_deadline(self) -> None:
        with patch(
            "utils.llm_client.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="curl", timeout=1),
        ):
            with self.assertRaisesRegex(TimeoutError, "hard deadline"):
                OllamaClient(timeout=1).generate("rerank this")

    def test_hybrid_backend_bounds_sbert_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)
            with patch(
                "utils.fault_localization._sbert_scores",
                return_value=[0.2, 0.9],
            ) as sbert_scores:
                result = localize_ticket(
                    {
                        "ticket_id": "HYBRID-1",
                        "title": "Profile rendering fails for a missing user name",
                        "description": "render_profile raises while reading the user name in src/profile/view.py",
                    },
                    code_index=index,
                    top_k=1,
                    embedding_backend="tfidf-sbert-rerank",
                    semantic_candidate_k=2,
                    candidate_file_k=1,
                    file_aggregation=False,
                )

        self.assertEqual(sbert_scores.call_count, 1)
        self.assertLessEqual(len(sbert_scores.call_args.args[1]), 2)
        self.assertTrue(result["method"]["embedding_backend"].startswith("tfidf+sbert-rerank:"))
        self.assertEqual(result["method"]["semantic_candidate_k"], 2)
        self.assertIn("sbert_score", result["localized_candidates"][0]["signals"])

    def test_hybrid_backend_uses_unique_file_representatives_for_top_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            crowded_functions = "\n\n".join(
                f"def parse_payload_{index}(payload):\n    return payload.strip()"
                for index in range(30)
            )
            (source / "crowded.py").write_text(crowded_functions + "\n", encoding="utf-8")
            for index in range(24):
                (source / f"module_{index:02d}.py").write_text(
                    f"def parse_payload_module_{index}(payload):\n"
                    "    return payload.strip()\n",
                    encoding="utf-8",
                )
            index = build_code_index(repo)

            with patch(
                "utils.fault_localization._sbert_scores",
                side_effect=lambda _query, texts, *_args, **_kwargs: [0.5] * len(texts),
            ) as sbert_scores:
                result = localize_ticket(
                    {
                        "ticket_id": "HYBRID-UNIQUE-FILES",
                        "title": "parse payload normalization fails",
                        "description": "The payload parser returns the wrong normalized value.",
                    },
                    code_index=index,
                    top_k=5,
                    embedding_backend="tfidf-sbert-rerank",
                    semantic_candidate_k=50,
                    candidate_file_k=20,
                    domain_path_routing=False,
                )

        semantic_texts = sbert_scores.call_args.args[1]
        self.assertEqual(len(semantic_texts), 25)
        self.assertEqual(len(result["stage1_candidate_files"]), 20)
        self.assertEqual(
            result["stage1_diagnostics"]["semantic_representative"],
            "best_tfidf_chunk_per_file",
        )

    def test_build_code_index_extracts_python_symbols_and_skips_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)

        function_names = {chunk.function_name for chunk in index.chunks}
        indexed_files = {chunk.file_path for chunk in index.chunks}
        self.assertIn("validate_token", function_names)
        self.assertIn("src/auth/validator.py", indexed_files)
        self.assertNotIn("tests/test_validator.py", indexed_files)

    def test_bug_report_text_excludes_ids_and_duplicate_content(self) -> None:
        text = build_bug_report_text(
            {
                "ticket_id": "project__secret-answer-123",
                "title": "Parser fails on empty input",
                "bug_report": "Parser fails on empty input",
                "description": "Parser fails on empty input",
                "hints_text": "src/parser.py",
                "fail_to_pass": ["tests/test_parser.py::test_empty"],
            }
        )

        self.assertNotIn("project__secret-answer-123", text)
        self.assertNotIn("hints_text", text)
        self.assertNotIn("fail_to_pass", text)
        self.assertEqual(text.count("Parser fails on empty input"), 1)

    def test_build_code_index_keeps_nested_models_package_but_skips_root_models(self) -> None:
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

    def test_build_code_index_rejects_file_path_and_invalid_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_file = Path(tmp) / "module.py"
            source_file.write_text("value = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not a directory"):
                build_code_index(source_file)
            with self.assertRaisesRegex(ValueError, "max_file_bytes"):
                build_code_index(Path(tmp), max_file_bytes=0)

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
        self.assertLess(best["score"], 1.0)
        self.assertEqual(result["confidence_level"], "high")
        self.assertTrue(result["recommend_patch_generation"])

    def test_stage1_preserves_twenty_unique_files_before_final_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            for index in range(25):
                (source / f"module_{index:02d}.py").write_text(
                    f"def parse_payload_{index}(payload):\n"
                    "    return payload.strip()\n",
                    encoding="utf-8",
                )

            result = localize_ticket(
                {
                    "ticket_id": "STAGE1-20",
                    "title": "Payload parser strips invalid input",
                    "description": "parse_payload fails while normalizing payload text.",
                },
                repo_path=repo,
                top_k=5,
                candidate_file_k=20,
            )

        stage1_files = [row["file_path"] for row in result["stage1_candidate_files"]]
        self.assertEqual(len(stage1_files), 20)
        self.assertEqual(len(set(stage1_files)), 20)
        self.assertEqual(len(result["localized_files"]), 5)
        self.assertEqual(result["method"]["candidate_file_k"], 20)
        self.assertEqual(result["stage1_diagnostics"]["stage_boundary"], "before_llm_rerank")
        self.assertEqual(result["evaluation_ready_fields"]["candidate_stage"], "stage1_candidate_files[*].file_path")

    def test_domain_path_routing_connects_framework_concepts_to_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            conf = repo / "django" / "conf"
            staticfiles = repo / "django" / "contrib" / "staticfiles"
            conf.mkdir(parents=True)
            staticfiles.mkdir(parents=True)
            (conf / "__init__.py").write_text(
                "class LazySettings:\n"
                "    pass\n",
                encoding="utf-8",
            )
            (staticfiles / "storage.py").write_text(
                "def static_url(path):\n"
                "    return path\n",
                encoding="utf-8",
            )

            result = localize_ticket(
                {
                    "ticket_id": "DOMAIN-ROUTING",
                    "title": "SCRIPT_NAME changes STATIC_URL and MEDIA_URL settings",
                },
                repo_path=repo,
                top_k=2,
                candidate_file_k=2,
            )

        conf_candidate = next(
            row for row in result["stage1_candidate_files"]
            if row["file_path"] == "django/conf/__init__.py"
        )
        self.assertEqual(conf_candidate["scoring_signals"]["domain_path_score"], 0.9)
        self.assertIn("django_settings_conf", conf_candidate["reason"])
        self.assertTrue(result["method"]["domain_path_routing"])

    def test_file_aggregation_returns_unique_files_with_supporting_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "tokens.py").write_text(
                "class TokenService:\n"
                "    def validate_token(self, token):\n"
                "        return token.strip()\n\n"
                "def normalize_token(token):\n"
                "    return token.strip().lower()\n",
                encoding="utf-8",
            )
            (source / "session.py").write_text(
                "def create_token_session(token):\n"
                "    return token\n",
                encoding="utf-8",
            )
            result = localize_ticket(
                {
                    "ticket_id": "AGG",
                    "title": "Token validation normalization fails",
                    "description": "validate_token and normalize_token mishandle the token.",
                },
                repo_path=repo,
                top_k=5,
                advanced_file_aggregation=True,
            )
            basic_result = localize_ticket(
                {
                    "ticket_id": "AGG-BASIC",
                    "title": "Token validation normalization fails",
                    "description": "validate_token and normalize_token mishandle the token.",
                },
                repo_path=repo,
                top_k=5,
                advanced_file_aggregation=False,
            )

        file_paths = [candidate["file_path"] for candidate in result["localized_candidates"]]
        self.assertEqual(len(file_paths), len(set(file_paths)))
        self.assertEqual(file_paths, [row["file_path"] for row in result["localized_files"]])
        token_candidate = next(row for row in result["localized_candidates"] if row["file_path"] == "src/tokens.py")
        self.assertTrue(token_candidate["supporting_chunks"])
        self.assertEqual(result["method"]["ranking_level"], "file_aggregated_chunks")
        self.assertTrue(result["method"]["advanced_file_aggregation"])
        basic_token = next(
            row for row in basic_result["localized_candidates"]
            if row["file_path"] == "src/tokens.py"
        )
        self.assertFalse(basic_result["method"]["advanced_file_aggregation"])
        self.assertEqual(basic_token["scoring_signals"]["supporting_chunk_bonus"], 0.0)
        self.assertEqual(basic_token["scoring_signals"]["package_proximity_bonus"], 0.0)

    def test_hybrid_e1_modes_keep_pre_sbert_supporting_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "invoice.py").write_text(
                "def calculate_invoice_tax(invoice):\n"
                "    return invoice.tax_total\n\n"
                "def validate_invoice_tax(invoice):\n"
                "    return invoice.tax_total >= 0\n\n"
                "def format_invoice_tax(invoice):\n"
                "    return str(invoice.tax_total)\n",
                encoding="utf-8",
            )
            (source / "unrelated.py").write_text(
                "def health_check():\n"
                "    return True\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            ticket = {
                "ticket_id": "E1-HYBRID",
                "title": "Invoice tax total validation is incorrect",
                "description": "calculate and format invoice tax_total use the wrong value.",
            }

            results: dict[str, dict] = {}
            with patch(
                "utils.fault_localization._sbert_scores",
                side_effect=lambda _query, texts, *_args, **_kwargs: [0.5] * len(texts),
            ):
                for mode in (
                    "basic",
                    "supporting-chunks",
                    "supporting-symbols",
                ):
                    results[mode] = localize_ticket(
                        ticket,
                        code_index=index,
                        top_k=2,
                        candidate_file_k=2,
                        semantic_candidate_k=2,
                        embedding_backend="tfidf-sbert-rerank",
                        domain_path_routing=False,
                        file_aggregation_mode=mode,
                    )

        candidates = {
            mode: next(
                row
                for row in result["stage1_candidate_files"]
                if row["file_path"] == "src/invoice.py"
            )
            for mode, result in results.items()
        }
        self.assertGreaterEqual(
            candidates["supporting-chunks"]["scoring_signals"]["support_evidence_count"],
            2,
        )
        self.assertTrue(candidates["supporting-chunks"]["supporting_chunks"])
        self.assertEqual(candidates["basic"]["scoring_signals"]["supporting_chunk_bonus"], 0.0)
        self.assertGreater(
            candidates["supporting-chunks"]["scoring_signals"]["supporting_chunk_bonus"],
            0.0,
        )
        self.assertEqual(
            candidates["supporting-chunks"]["scoring_signals"]["symbol_coverage_bonus"],
            0.0,
        )
        self.assertGreater(
            candidates["supporting-symbols"]["scoring_signals"]["symbol_coverage_bonus"],
            0.0,
        )
        self.assertEqual(
            results["supporting-symbols"]["method"]["file_aggregation_mode"],
            "supporting-symbols",
        )

    def test_invalid_file_aggregation_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            with self.assertRaisesRegex(ValueError, "file_aggregation_mode"):
                localize_ticket(
                    {"ticket_id": "BAD-E1", "title": "Token validation fails"},
                    repo_path=repo,
                    file_aggregation_mode="unbounded-sum",
                )

    def test_repository_proximity_expands_top_candidate_to_imported_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "controller.py").write_text(
                "from . import formatter\n\n"
                "def render_invoice(invoice):\n"
                "    return formatter.build(invoice)\n",
                encoding="utf-8",
            )
            (source / "formatter.py").write_text(
                "def build(value):\n"
                "    return str(value)\n",
                encoding="utf-8",
            )
            (source / "unrelated.py").write_text(
                "def health_check():\n"
                "    return True\n",
                encoding="utf-8",
            )

            result = localize_ticket(
                {
                    "ticket_id": "IMPORT-NEIGHBOR",
                    "title": "render_invoice crashes while formatting the invoice",
                    "description": "The controller reaches render_invoice but the returned value is wrong.",
                },
                repo_path=repo,
                top_k=3,
                candidate_file_k=3,
                domain_path_routing=False,
                repository_proximity=True,
            )

        formatter = next(
            row for row in result["stage1_candidate_files"]
            if row["file_path"] == "src/formatter.py"
        )
        signals = formatter["scoring_signals"]
        self.assertGreater(signals["repository_proximity_score"], 0.0)
        self.assertIn(
            "imported_by:src/controller.py",
            signals["matching_repository_proximity"],
        )

    def test_python_from_import_indexes_base_and_imported_module(self) -> None:
        modules = fault_localization._import_line_modules(
            "from matplotlib import cbook, colors as mcolors",
            current_file="lib/matplotlib/figure.py",
        )

        self.assertEqual(
            modules,
            ["matplotlib", "matplotlib.cbook", "matplotlib.colors"],
        )

    def test_import_graph_parses_absolute_relative_alias_reexport_and_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            package = repo / "pkg"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "from .api import public_api\n",
                encoding="utf-8",
            )
            (package / "api.py").write_text(
                "from .core import run as execute\n"
                "import pkg.helpers as helpers\n\n"
                "def public_api(value):\n"
                "    return execute(value)\n",
                encoding="utf-8",
            )
            (package / "core.py").write_text(
                "from . import api\n\n"
                "def run(value):\n"
                "    return value\n",
                encoding="utf-8",
            )
            (package / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
            (package / "caller.py").write_text(
                "from pkg import api\n",
                encoding="utf-8",
            )
            (package / "unresolved.py").write_text(
                "import third_party_missing\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            index_path = Path(tmp) / "index.json"
            index.save(index_path)
            loaded = fault_localization.load_code_index(index_path)

        graph = loaded.import_graph
        self.assertIsNotNone(graph)
        assert graph is not None
        self.assertEqual(
            graph.outgoing["pkg/api.py"],
            ["pkg/core.py", "pkg/helpers.py"],
        )
        self.assertIn("pkg/api.py", graph.outgoing["pkg/__init__.py"])
        self.assertIn("pkg/api.py", graph.outgoing["pkg/core.py"])
        self.assertIn("pkg/core.py", graph.incoming["pkg/api.py"])
        self.assertIn("pkg/caller.py", graph.incoming["pkg/api.py"])
        self.assertNotIn("pkg/unresolved.py", graph.outgoing)
        self.assertIn("third_party_missing", graph.unresolved["pkg/unresolved.py"])
        self.assertEqual(graph.stats["edges"], sum(map(len, graph.outgoing.values())))

    def test_bidirectional_import_graph_adds_incoming_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            package = repo / "pkg"
            package.mkdir(parents=True)
            (package / "core.py").write_text(
                "def calculate_total(invoice):\n"
                "    return invoice.total\n",
                encoding="utf-8",
            )
            (package / "adapter.py").write_text(
                "from .core import calculate_total\n\n"
                "def process(value):\n"
                "    return calculate_total(value)\n",
                encoding="utf-8",
            )
            (package / "other.py").write_text(
                "def health_check():\n"
                "    return True\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)
            ticket = {
                "ticket_id": "E2-INCOMING",
                "title": "calculate_total returns the wrong invoice total",
                "description": "The calculation in calculate_total must be corrected.",
            }
            outgoing = localize_ticket(
                ticket,
                code_index=index,
                top_k=3,
                candidate_file_k=3,
                domain_path_routing=False,
                import_graph_mode="outgoing",
            )
            bidirectional = localize_ticket(
                ticket,
                code_index=index,
                top_k=3,
                candidate_file_k=3,
                domain_path_routing=False,
                import_graph_mode="bidirectional",
            )

        outgoing_adapter = next(
            row for row in outgoing["stage1_candidate_files"]
            if row["file_path"] == "pkg/adapter.py"
        )
        bidirectional_adapter = next(
            row for row in bidirectional["stage1_candidate_files"]
            if row["file_path"] == "pkg/adapter.py"
        )
        self.assertNotIn(
            "imports:pkg/core.py",
            outgoing_adapter["scoring_signals"]["matching_repository_proximity"],
        )
        self.assertIn(
            "imports:pkg/core.py",
            bidirectional_adapter["scoring_signals"]["import_graph_evidence"],
        )
        self.assertLessEqual(
            bidirectional_adapter["scoring_signals"]["import_graph_bonus"],
            fault_localization.IMPORT_GRAPH_PARAMETERS["maximum_bonus"],
        )

    def test_import_graph_off_preserves_baseline_and_rejects_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)
            ticket = {"ticket_id": "E2-OFF", "title": "Token validation fails unexpectedly"}
            baseline = localize_ticket(
                ticket,
                code_index=index,
                top_k=3,
                candidate_file_k=3,
                domain_path_routing=False,
            )
            explicit_off = localize_ticket(
                ticket,
                code_index=index,
                top_k=3,
                candidate_file_k=3,
                domain_path_routing=False,
                import_graph_mode="off",
            )
            with self.assertRaisesRegex(ValueError, "import_graph_mode"):
                localize_ticket(
                    ticket,
                    code_index=index,
                    import_graph_mode="recursive",
                )

        baseline_rows = [
            (row["file_path"], row["retrieval_score"])
            for row in baseline["stage1_candidate_files"]
        ]
        explicit_rows = [
            (row["file_path"], row["retrieval_score"])
            for row in explicit_off["stage1_candidate_files"]
        ]
        self.assertEqual(explicit_rows, baseline_rows)

    def test_legacy_index_import_graph_sidecar_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            package = repo / "pkg"
            package.mkdir(parents=True)
            (package / "a.py").write_text("from . import b\n", encoding="utf-8")
            (package / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
            index = build_code_index(repo)
            index.import_graph = None
            config = types.SimpleNamespace(
                import_graph_cache_dir=root / "graphs",
                index_cache_dir=root / "indexes",
                include_tests=False,
                max_file_bytes=500_000,
            )
            ticket = {"ticket_id": "E2-CACHE", "repo": "owner/repo", "base_commit": "abc123"}
            reporter = ProgressReporter("none")
            built, first_source = load_or_build_import_graph(ticket, index, config, reporter)
            loaded, second_source = load_or_build_import_graph(ticket, index, config, reporter)

        self.assertEqual(first_source, "built")
        self.assertEqual(second_source, "cache")
        self.assertEqual(loaded.outgoing, built.outgoing)

    def test_index_covers_module_level_code_and_precise_typescript_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "settings.py").write_text(
                "AUTH_TOKEN_TTL = 0\n\n"
                "def load_settings():\n"
                "    return AUTH_TOKEN_TTL\n",
                encoding="utf-8",
            )
            (source / "handlers.ts").write_text(
                "function firstHandler() {\n"
                "  return 1;\n"
                "}\n\n"
                "function secondHandler() {\n"
                "  return 2;\n"
                "}\n",
                encoding="utf-8",
            )
            index = build_code_index(repo)

        module_chunks = [
            chunk for chunk in index.chunks
            if chunk.file_path == "src/settings.py" and chunk.symbol_kind == "module"
        ]
        self.assertTrue(any("AUTH_TOKEN_TTL" in chunk.code_text for chunk in module_chunks))
        ts_functions = {
            chunk.function_name: (chunk.start_line, chunk.end_line)
            for chunk in index.chunks
            if chunk.file_path == "src/handlers.ts" and chunk.function_name
        }
        self.assertEqual(ts_functions["firstHandler"], (1, 3))
        self.assertEqual(ts_functions["secondHandler"], (5, 7))

    def test_incremental_index_reuses_only_unchanged_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            first = build_code_index(repo)
            unchanged = build_code_index(repo, previous_index=first)
            (repo / "src" / "auth" / "validator.py").write_text(
                "def validate_token(token):\n"
                "    return token.strip() if token else ''\n",
                encoding="utf-8",
            )
            changed = build_code_index(repo, previous_index=unchanged)

        self.assertEqual(unchanged.settings["index_stats"]["reused_files"], 2)
        self.assertEqual(unchanged.settings["index_stats"]["rebuilt_files"], 0)
        self.assertEqual(changed.settings["index_stats"]["reused_files"], 1)
        self.assertEqual(changed.settings["index_stats"]["rebuilt_files"], 1)

    def test_empty_ticket_is_blocked_by_confidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            result = localize_ticket({"ticket_id": "EMPTY"}, repo_path=repo)

        self.assertEqual(result["localized_candidates"], [])
        self.assertEqual(result["input_validation"]["status"], "error")
        self.assertEqual(result["confidence_level"], "low")
        self.assertTrue(result["should_manual_review"])
        self.assertFalse(result["recommend_patch_generation"])

    def test_localize_ticket_validates_backend_and_handles_llm_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            index = build_code_index(repo)

            with self.assertRaisesRegex(ValueError, "embedding_backend"):
                localize_ticket(
                    {"ticket_id": "BAD-BACKEND", "title": "Login fails"},
                    code_index=index,
                    embedding_backend="typo",
                )

            result = localize_ticket(
                {"ticket_id": "NO-CLIENT", "title": "Login token fails"},
                code_index=index,
                llm_rerank=True,
            )

        self.assertFalse(result["method"]["llm_rerank"])
        self.assertTrue(any("no LLM client" in warning for warning in result["warnings"]))

    def test_llm_rerank_ignores_duplicate_candidate_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            result = localize_ticket(
                {
                    "ticket_id": "RERANK",
                    "title": "Login token and profile rendering fail",
                    "description": "Check token validation and profile rendering.",
                },
                repo_path=repo,
                top_k=2,
                llm_client=DuplicateRankRerankClient(),
                llm_rerank=True,
            )

        chunk_ids = [candidate["chunk_id"] for candidate in result["localized_candidates"]]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertTrue(result["method"]["llm_rerank"])
        self.assertIn("llm_blended_score", result["localized_candidates"][0]["signals"])

    def test_llm_rerank_rejects_incomplete_candidate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            result = localize_ticket(
                {
                    "ticket_id": "PARTIAL-RERANK",
                    "title": "Login token and profile rendering fail",
                    "description": "Check token validation and profile rendering.",
                },
                repo_path=repo,
                top_k=2,
                llm_client=PartialRerankClient(),
                llm_rerank=True,
            )

        self.assertFalse(result["method"]["llm_rerank"])
        self.assertTrue(any("no usable candidates" in warning for warning in result["warnings"]))

    def test_bug_localizer_reuses_unchanged_repository_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            localizer = BugLocalizer(top_k=1)
            ticket = {"ticket_id": "CACHE", "title": "Login token fails"}

            localizer.localize(ticket, str(repo))
            first_index = localizer._cached_index
            localizer.localize(ticket, str(repo))

        self.assertIsNotNone(first_index)
        self.assertIs(first_index, localizer._cached_index)

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

    def test_evaluate_records_counts_missing_predictions_as_misses(self) -> None:
        gold = [
            {"ticket_id": "A", "fixed_files": ["src/a.py"]},
            {"ticket_id": "B", "fixed_files": ["src/b.py"]},
        ]
        pred = [
            {
                "ticket_id": "B",
                "localized_candidates": [{"file_path": "src/b.py"}],
            }
        ]

        metrics = evaluate_records(gold, pred)

        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["matched_prediction_rows"], 1)
        self.assertEqual(metrics["missing_prediction_rows"], 1)
        self.assertEqual(metrics["top_1_accuracy"], 0.5)

    def test_evaluate_records_reports_stage1_hit_recall_and_outcomes(self) -> None:
        gold = [
            {"ticket_id": "A", "fixed_files": ["src/a.py", "src/b.py"]},
            {"ticket_id": "B", "fixed_files": ["src/c.py"]},
            {"ticket_id": "C", "fixed_files": ["src/d.py"]},
        ]
        pred = [
            {
                "ticket_id": "A",
                "stage1_candidate_files": [
                    {"file_path": "src/a.py"},
                    {"file_path": "src/other.py"},
                ],
            },
            {
                "ticket_id": "B",
                "stage1_candidate_files": [{"file_path": "src/c.py"}],
            },
            {"ticket_id": "C", "stage1_candidate_files": [{"file_path": "src/other.py"}]},
        ]

        metrics = evaluate_records(gold, pred)

        self.assertEqual(metrics["candidate_hit_at_20"], 2 / 3)
        self.assertEqual(metrics["candidate_recall_at_20"], 0.5)
        self.assertEqual(metrics["average_candidate_count_at_20"], 4 / 3)
        self.assertEqual(
            metrics["candidate_outcomes_at_20"],
            {"full_recall_rows": 1, "partial_recall_rows": 1, "miss_rows": 1},
        )
        self.assertEqual(
            metrics["candidate_outcome_ticket_ids_at_20"],
            {"full_recall": ["B"], "partial_recall": ["A"], "miss": ["C"]},
        )

    def test_reachable_recall_excludes_files_absent_at_base_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_cache = tmp_path / "repos"
            repo = repo_cache / "example__project"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "existing.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex Test",
                    "-c",
                    "user.email=codex@example.invalid",
                    "commit",
                    "-qm",
                    "base",
                ],
                cwd=repo,
                check=True,
            )
            base_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            gold = [
                {
                    "ticket_id": "example__project-1",
                    "repo": "example/project",
                    "base_commit": base_commit,
                    "fixed_files": ["src/existing.py", "src/new_file.py"],
                }
            ]
            pred = [
                {
                    "ticket_id": "example__project-1",
                    "stage1_candidate_files": [{"file_path": "src/existing.py"}],
                }
            ]
            reachable = discover_base_commit_existing_gold_files(gold, repo_cache)
            metrics = evaluate_records(
                gold,
                pred,
                reachable_gold_files_by_ticket=reachable,
            )

        self.assertEqual(reachable["example__project-1"], ["src/existing.py"])
        self.assertEqual(metrics["candidate_recall_at_20"], 0.5)
        reachable_metrics = metrics["base_commit_reachable_evaluation"]
        self.assertEqual(reachable_metrics["candidate_recall_at_20"], 1.0)
        self.assertEqual(reachable_metrics["unreachable_gold_file_count"], 1)

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

    def test_cli_cache_checkpoint_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = _make_repo(tmp_path)
            tickets_path = tmp_path / "tickets.jsonl"
            output_path = tmp_path / "predictions.jsonl"
            cache_dir = tmp_path / "cache"
            tickets = [
                {"ticket_id": "CLI-1", "title": "validate token fails", "description": "Token validation fails."},
                {"ticket_id": "CLI-2", "title": "profile rendering fails", "description": "Profile view cannot render."},
            ]
            tickets_path.write_text(
                "".join(json.dumps(ticket) + "\n" for ticket in tickets),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "fault_localization.py"),
                "--tickets-jsonl",
                str(tickets_path),
                "--repo-path",
                str(repo),
                "--index-cache-dir",
                str(cache_dir),
                "--output",
                str(output_path),
                "--checkpoint-every",
                "1",
                "--progress",
                "json",
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            resumed = subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)
            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(output_rows), 2)
        self.assertIn('"event": "checkpoint_written"', first.stderr)
        self.assertIn('"event": "resume_loaded"', resumed.stderr)
        self.assertIn('"event": "ticket_skipped"', resumed.stderr)
        self.assertTrue(all("stage1_candidate_files" in row for row in output_rows))

    def test_swebench_runner_uses_isolated_commit_snapshot_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "source_repo"
            source = repo / "src"
            source.mkdir(parents=True)
            (source / "parser.py").write_text(
                "def parse_empty(value):\n"
                "    return value.strip()\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex Test",
                    "-c",
                    "user.email=codex@example.invalid",
                    "commit",
                    "-qm",
                    "base snapshot",
                ],
                cwd=repo,
                check=True,
            )
            base_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tickets_path = tmp_path / "tickets.jsonl"
            gold_path = tmp_path / "gold.jsonl"
            tickets_path.write_text(
                json.dumps(
                    {
                        "ticket_id": "example__project-1",
                        "repo": "example/project",
                        "base_commit": base_commit,
                        "local_repo_path": str(repo),
                        "title": "Parser crashes on empty input",
                        "description": "parse_empty calls strip for an empty value.",
                        "logs": "src/parser.py:2: AttributeError",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gold_path.write_text(
                json.dumps(
                    {
                        "ticket_id": "example__project-1",
                        "repo": "example/project",
                        "base_commit": base_commit,
                        "fixed_files": ["src/parser.py"],
                        "fixed_symbols": ["parse_empty"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = BatchRunConfig(
                tickets_path=tickets_path,
                gold_path=gold_path,
                repo_cache_dir=tmp_path / "repos",
                snapshot_cache_dir=tmp_path / "snapshots",
                index_cache_dir=tmp_path / "indexes",
                predictions_output=tmp_path / "output" / "predictions.jsonl",
                metrics_output=tmp_path / "output" / "metrics.json",
                failures_output=tmp_path / "output" / "failures.jsonl",
                manifest_output=tmp_path / "output" / "manifest.json",
                checkpoint_every=1,
                progress="none",
                frozen_stage1_v11=True,
            )

            first = run_batch(config)
            config.resume = True
            resumed = run_batch(config)
            prediction = json.loads(config.predictions_output.read_text(encoding="utf-8").strip())
            source_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config.candidate_file_k = 19
            with self.assertRaisesRegex(ValueError, "frozen stage1-v11"):
                run_batch(config)

        self.assertEqual(first["failures"], 0)
        self.assertEqual(first["metrics"]["file_top_1_accuracy"], 1.0)
        self.assertEqual(first["metrics"]["candidate_hit_at_20"], 1.0)
        self.assertEqual(first["metrics"]["candidate_recall_at_20"], 1.0)
        self.assertEqual(first["metrics"]["symbol_top_1_accuracy"], 1.0)
        self.assertEqual(resumed["processed_this_run"], 0)
        self.assertEqual(prediction["base_commit"], base_commit)
        self.assertEqual(prediction["benchmark_run"]["method"]["embedding_backend"], "tfidf")
        self.assertEqual(prediction["benchmark_run"]["method"]["candidate_file_k"], 20)
        self.assertEqual(prediction["benchmark_run"]["frozen_protocol"], "stage1-v11")
        self.assertEqual(prediction["stage1_candidate_files"][0]["file_path"], "src/parser.py")
        self.assertEqual(source_head, base_commit)

    def test_compare_runs_reports_metric_deltas_and_denominator_warning(self) -> None:
        comparison = compare_runs(
            {
                "tfidf": {
                    "rows_with_file_ground_truth": 10,
                    "rows_with_symbol_ground_truth": 5,
                    "file_top_1_accuracy": 0.5,
                    "file_top_3_accuracy": 0.7,
                    "file_top_5_accuracy": 0.8,
                    "file_mrr": 0.6,
                },
                "sbert": {
                    "rows_with_file_ground_truth": 10,
                    "rows_with_symbol_ground_truth": 5,
                    "file_top_1_accuracy": 0.6,
                    "file_top_3_accuracy": 0.8,
                    "file_top_5_accuracy": 0.9,
                    "file_mrr": 0.7,
                },
                "llm": {
                    "rows_with_file_ground_truth": 9,
                    "rows_with_symbol_ground_truth": 5,
                    "file_top_1_accuracy": 0.7,
                    "run_diagnostics": {
                        "llm_rerank_requested": True,
                        "llm_rerank_used_predictions": 8,
                        "llm_rerank_fallback_predictions": 1,
                    },
                },
            },
            baseline="tfidf",
        )

        sbert = next(row for row in comparison["runs"] if row["label"] == "sbert")
        self.assertEqual(sbert["delta_vs_baseline"]["file_top_1_accuracy"], 0.1)
        self.assertTrue(any("different evaluation denominators" in warning for warning in comparison["warnings"]))
        self.assertTrue(any("fell back to retrieval" in warning for warning in comparison["warnings"]))


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

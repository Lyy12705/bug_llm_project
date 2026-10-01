from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modules.patch_generator import PatchGenerator
from utils.patch_context import PatchContextError, build_stage3_context


DIFF = "--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-import os\n+import sys\n"


class Client:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return DIFF


class Stage3PatchHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.code = "import os\n\ndef target():\n" + "    value = 1\n" * 130 + "    return 'COMPLETE_TAIL'\n"
        (self.repo / "target.py").write_text(self.code, encoding="utf-8")
        self.location = {
            "confidence_level": "high", "should_manual_review": False,
            "recommend_patch_generation": True,
            "bug_location": {"file": "wrong.py", "function": "wrong"},
            "localized_candidates": [{"file_path": "wrong.py"}],
            "stage3_diagnostics": {"localization_requested": True},
            "stage3_ranked_symbols": [{"file_path": "target.py", "symbol_qualified_name": "target",
                "symbol_kind": "function", "start_line": 3, "end_line": 134,
                "code_text": "truncated snippet", "score": 0.0}],
        }

    def test_stage3_overrides_legacy_target_and_preserves_full_source(self):
        client = Client()
        result = PatchGenerator(llm_client=client).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["bug_location"]["file"], "target.py")
        self.assertEqual(result["localization_context"][0]["function_name"], "target")
        self.assertEqual(result["localization_context"][0]["score"], 0.0)
        self.assertIn("COMPLETE_TAIL", client.prompts[0])
        self.assertIn("import os", client.prompts[0])
        self.assertNotIn("wrong.py", client.prompts[0])

    def test_custom_template_receives_all_files_in_context(self):
        (self.repo / "other.py").write_text("OTHER_SOURCE = 1\n", encoding="utf-8")
        self.location["stage3_ranked_symbols"].append({"file_path": "other.py", "symbol_kind": "module",
            "symbol_qualified_name": "<module>", "start_line": 1, "end_line": 1, "code_text": ""})
        template = self.repo / "prompt.txt"
        template.write_text("Ticket:{ticket_json}\nLocation:{bug_location}\nCode:{context}", encoding="utf-8")
        client = Client()
        PatchGenerator(llm_client=client, prompt_path=template).generate({}, self.location, str(self.repo))
        self.assertIn("OTHER_SOURCE", client.prompts[0])
        self.assertIn("COMPLETE_TAIL", client.prompts[0])

    def test_module_reads_full_file_and_duplicate_files_are_loaded_once(self):
        module = dict(self.location["stage3_ranked_symbols"][0], symbol_kind="module",
                      symbol_qualified_name="<module>", code_text="", start_line=1)
        self.location["stage3_ranked_symbols"].append(module)
        context = build_stage3_context(self.location, str(self.repo))
        self.assertEqual(len(context["symbols"]), 2)
        self.assertEqual(len(context["files"]), 1)
        self.assertIn("COMPLETE_TAIL", context["files"][0]["code_text"])

    def test_missing_bad_ranges_and_empty_symbols_block_model(self):
        for row in (None, {"file_path": "missing.py", "start_line": 1, "end_line": 2},
                    {"file_path": "../outside.py", "start_line": 1, "end_line": 2},
                    {"file_path": "target.py", "start_line": 0, "end_line": 2},
                    {"file_path": "target.py", "start_line": 1, "end_line": 9999}):
            with self.subTest(row=row):
                location = copy.deepcopy(self.location)
                location["stage3_ranked_symbols"] = [] if row is None else [row]
                client = Client()
                result = PatchGenerator(llm_client=client).generate({}, location, str(self.repo))
                self.assertEqual(result["patch_status"], "blocked_invalid_context")
                self.assertFalse(result["recommend_patch_generation"])
                self.assertEqual(client.prompts, [])

    def test_changed_indexed_source_blocks_model(self):
        self.location["source_file_sha256"] = {"target.py": "0" * 64}
        with self.assertRaisesRegex(PatchContextError, "changed after"):
            build_stage3_context(self.location, str(self.repo))

    def test_matching_hash_records_verified_snapshot(self):
        self.location["source_file_sha256"] = {"target.py": hashlib.sha256((self.repo / "target.py").read_bytes()).hexdigest()}
        self.assertTrue(build_stage3_context(self.location, str(self.repo))["snapshot_verified"])

    def test_commit_mismatch_and_dirty_source_are_rejected(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True).stdout.decode().strip()
        git("init")
        git("add", "target.py")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "base")
        self.location["base_commit"] = git("rev-parse", "HEAD")
        self.assertTrue(build_stage3_context(self.location, str(self.repo))["snapshot_verified"])
        (self.repo / "target.py").write_text(self.code + "# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(PatchContextError, "differs"):
            build_stage3_context(self.location, str(self.repo))
        git("add", "target.py")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "next")
        with self.assertRaisesRegex(PatchContextError, "HEAD"):
            build_stage3_context(self.location, str(self.repo))

    def test_benchmark_gold_is_not_accepted_or_prompted(self):
        client = Client()
        ticket = {"source_dataset": "SWE-bench", "description": "target crashes",
                  "patch": "SECRET_GOLD", "test_patch": "SECRET_TEST", "FAIL_TO_PASS": "SECRET_LABEL",
                  "ground_truth": {"patch": "SECRET_NESTED"}, "hints_text": "SECRET_HINT"}
        result = PatchGenerator(llm_client=client).generate(ticket, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "generated")
        self.assertNotIn("SECRET_", client.prompts[0])

    def test_low_confidence_still_prevents_model_call(self):
        self.location.update(confidence_level="low", recommend_patch_generation=False)
        client = Client()
        result = PatchGenerator(llm_client=client).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "blocked_low_confidence")
        self.assertEqual(result["localization_context"][0]["file_path"], "target.py")
        self.assertEqual(client.prompts, [])

    def test_oversized_context_is_not_silently_truncated(self):
        (self.repo / "target.py").write_text("#" * 100_001, encoding="utf-8")
        with self.assertRaisesRegex(PatchContextError, "budget"):
            build_stage3_context(self.location, str(self.repo))

    def test_markdown_response_is_reduced_to_one_clean_diff(self):
        class MarkdownClient:
            def generate(self, prompt):
                return "Expected output:\n```diff\n" + DIFF + "```\nextra prose"
        result = PatchGenerator(llm_client=MarkdownClient()).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["patch"], DIFF)

    def test_generated_hunk_counts_are_repaired_before_apply(self):
        malformed = (
            "```diff\n--- a/target.py\n+++ b/target.py\n@@ -1,1 +1,1 @@\n"
            "-import os\n+if value is None:\n+    return ''\n```"
        )
        class MalformedClient:
            def generate(self, prompt):
                return malformed
        result = PatchGenerator(llm_client=MalformedClient()).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "generated")
        self.assertIn("@@ -1 +1,2 @@", result["patch"])

    def test_generated_patch_cannot_modify_unlocalized_or_test_files(self):
        for patch_text in (
            "--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-a\n+b\n",
            "--- a/tests/test_target.py\n+++ b/tests/test_target.py\n@@ -1 +1 @@\n-a\n+b\n",
        ):
            class InvalidClient:
                def generate(self, prompt, value=patch_text):
                    return value
            result = PatchGenerator(llm_client=InvalidClient()).generate({}, self.location, str(self.repo))
            self.assertEqual(result["patch_status"], "blocked_invalid_generated_patch")
            self.assertEqual(result["patch"], "")

    def test_plain_diff_with_trailing_prose_is_rejected(self):
        class ProseClient:
            def generate(self, prompt):
                return DIFF + "This fixes the bug."
        result = PatchGenerator(llm_client=ProseClient()).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "blocked_invalid_generated_patch")

    def test_no_op_generated_diff_is_rejected(self):
        class NoOpClient:
            def generate(self, prompt):
                return "--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-import os\n+import os\n"
        result = PatchGenerator(llm_client=NoOpClient()).generate({}, self.location, str(self.repo))
        self.assertEqual(result["patch_status"], "blocked_invalid_generated_patch")


if __name__ == "__main__":
    unittest.main()

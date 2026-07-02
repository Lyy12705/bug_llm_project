from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from modules.patch_generator import PatchGenerator
from utils.git_utils import extract_modified_files, looks_like_unified_diff, run_patch_in_temp_copy


STANDARD_UNIFIED_DIFF = (
    "--- a/src/auth/validator.py\n"
    "+++ b/src/auth/validator.py\n"
    "@@ -1,2 +1,4 @@\n"
    " def validate_token(token):\n"
    "+    if token is None:\n"
    "+        raise ValueError(\"Token is required\")\n"
    "     return token.strip()\n"
)

NO_PREFIX_UNIFIED_DIFF = STANDARD_UNIFIED_DIFF.replace("--- a/", "--- ").replace("+++ b/", "+++ ")


class PatchHandlingTests(unittest.TestCase):
    def test_standard_unified_diff_without_git_header_is_recognized(self) -> None:
        self.assertTrue(looks_like_unified_diff(STANDARD_UNIFIED_DIFF))

    def test_modified_files_are_extracted_from_no_prefix_unified_diff(self) -> None:
        self.assertEqual(extract_modified_files(NO_PREFIX_UNIFIED_DIFF), ["src/auth/validator.py"])

    def test_llm_generated_standard_unified_diff_is_accepted(self) -> None:
        llm_client = _FakePatchClient(STANDARD_UNIFIED_DIFF)
        patch = PatchGenerator(llm_client=llm_client).generate(
            {"ticket_id": "RAW-1"},
            {
                "bug_location": {"file": "src/auth/validator.py", "line_start": 1, "line_end": 2},
                "localized_candidates": [
                    {
                        "rank": 1,
                        "file_path": "src/auth/validator.py",
                        "function_name": "validate_token",
                        "score": 0.91,
                        "reason": "Stack trace points at the validator.",
                    }
                ],
                "confidence_level": "high",
                "should_manual_review": False,
                "recommend_patch_generation": True,
            },
            repo_path="/unused",
        )

        self.assertEqual(patch["patch_status"], "generated")
        self.assertFalse(patch["requires_manual_patch"])
        self.assertEqual(patch["modified_files"], ["src/auth/validator.py"])
        self.assertEqual(patch["confidence_level"], "high")
        self.assertTrue(patch["localization_context"])
        self.assertIn("Localization context", llm_client.last_prompt)

    def test_low_confidence_localization_blocks_llm_patch_generation(self) -> None:
        llm_client = _FakePatchClient(STANDARD_UNIFIED_DIFF)
        patch = PatchGenerator(llm_client=llm_client).generate(
            {"ticket_id": "RAW-LOW"},
            {
                "bug_location": {
                    "file": "src/auth/validator.py",
                    "line_start": 1,
                    "line_end": 2,
                    "confidence_level": "low",
                    "uncertainty_reason": "Weak ranking score and no stack trace evidence.",
                    "should_manual_review": True,
                    "recommend_patch_generation": False,
                },
                "localized_candidates": [
                    {
                        "rank": 1,
                        "file_path": "src/auth/validator.py",
                        "function_name": "validate_token",
                        "score": 0.21,
                        "reason": "Weak lexical match.",
                    }
                ],
                "confidence_level": "low",
                "uncertainty_reason": "Weak ranking score and no stack trace evidence.",
                "should_manual_review": True,
                "recommend_patch_generation": False,
            },
            repo_path="/unused",
        )

        self.assertEqual(patch["patch_status"], "blocked_low_confidence")
        self.assertTrue(patch["requires_manual_patch"])
        self.assertTrue(patch["requires_manual_review"])
        self.assertFalse(patch["recommend_patch_generation"])
        self.assertEqual(patch["modified_files"], [])
        self.assertEqual(patch["patch"], "")
        self.assertEqual(llm_client.calls, 0)
        self.assertTrue(patch["localization_context"])

    def test_high_confidence_without_llm_returns_suggestion_only_context(self) -> None:
        patch = PatchGenerator().generate(
            {"ticket_id": "RAW-SUGGESTION"},
            {
                "bug_location": {
                    "file": "src/auth/validator.py",
                    "function": "validate_token",
                    "line_start": 1,
                    "line_end": 2,
                },
                "localized_candidates": [
                    {
                        "rank": 1,
                        "file_path": "src/auth/validator.py",
                        "function_name": "validate_token",
                        "score": 0.88,
                        "reason": "Stack trace points at the validator.",
                    }
                ],
                "confidence_level": "high",
                "should_manual_review": False,
                "recommend_patch_generation": True,
            },
            repo_path="/unused",
        )

        self.assertEqual(patch["patch_status"], "suggestion_only")
        self.assertTrue(patch["requires_manual_patch"])
        self.assertFalse(patch["requires_manual_review"])
        self.assertEqual(patch["modified_files"], ["src/auth/validator.py"])
        self.assertEqual(patch["patch_suggestion"]["target_symbol"], "validate_token")

    def test_standard_unified_diff_applies_in_temp_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))

            result = run_patch_in_temp_copy(repo, STANDARD_UNIFIED_DIFF, run_tests=False)

        self.assertEqual(result["patch_apply_check"], "passed")
        self.assertEqual(result["patch_apply"], "passed")
        self.assertEqual(result["regression_result"], "not_run")


class _FakePatchClient:
    def __init__(self, patch_text: str) -> None:
        self.patch_text = patch_text
        self.calls = 0
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.patch_text


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

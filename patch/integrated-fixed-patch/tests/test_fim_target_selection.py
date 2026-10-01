"""Target-selection tests for fim_patch.generator.

These cover the two changes made after measuring the real Stage-3 output of this
project:

1. Stage-3 ``end_line`` is a chunk boundary (the index is built in 80-line
   chunks), not a definition end, so requiring ``end_line == node.end_lineno``
   rejects targets on a false premise. ``expand_truncated_end`` repairs exactly
   that case and is now the default for any scanning policy.
2. 17 of the 46 measured tickets have a whole class as the rank-1 symbol, which
   this generator cannot patch at all. ``first_supported_ast`` walks down the
   ranking to the first target the AST can support and records which rank it
   landed on, so strict and fallback runs stay separable in the results.

Every test is offline: the backend is a double and no model is called.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fim_patch.generator import (
    TARGET_POLICY_DEPTH,
    FIMPatchGenerator,
    function_span,
    git,
)

SOURCE = (
    b"class Widget:\n"            # 1
    b"    def render(self):\n"    # 2
    b'        return "x"\n'       # 3
    b"\n"                         # 4
    b"\n"                         # 5
    b"def helper(value):\n"       # 6
    b"    value = value.strip()\n"  # 7
    b"    value = value.lower()\n"  # 8
    b"    return value\n"         # 9
)


class RecordingBackend:
    """Counts calls so a test can prove a rejection happened before generation."""

    def __init__(self, outputs=()):
        self.outputs = iter(outputs)
        self.calls = 0

    def describe(self):
        return {"model": "test-double"}

    def infill(self, prefix, suffix, *, seed):
        self.calls += 1
        return {"middle": next(self.outputs), "seed": seed}


class TargetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "app.py").write_bytes(SOURCE)
        git(self.repo, "init")
        git(self.repo, "config", "core.autocrlf", "false")
        git(self.repo, "add", ".")
        git(self.repo, "-c", "user.name=Test", "-c", "user.email=t@example.invalid",
            "commit", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD").decode().strip()

        # Rank 1 is a whole class: unpatchable by this generator, exactly the
        # shape that 17 of our 46 tickets have.
        self.class_row = {"file_path": "app.py", "symbol_qualified_name": "Widget",
                          "start_line": 1, "end_line": 3}
        self.method_row = {"file_path": "app.py", "symbol_qualified_name": "Widget.render",
                           "start_line": 2, "end_line": 3}
        self.helper_row = {"file_path": "app.py", "symbol_qualified_name": "helper",
                           "start_line": 6, "end_line": 9}
        self.ticket = {"base_commit": self.base, "description": "Lowercase the value"}

    def localization(self, rows):
        return {"base_commit": self.base, "confidence_level": "high",
                "should_manual_review": False, "recommend_patch_generation": True,
                "stage3_ranked_symbols": rows}

    # ---- function_span -------------------------------------------------

    def test_body_lines_counts_the_body_only(self) -> None:
        span = function_span(SOURCE, self.helper_row)
        self.assertEqual(span["line_start"], 7)
        self.assertEqual(span["line_end"], 9)
        self.assertEqual(span["body_lines"], 3)

    def test_max_body_lines_rejects_an_oversized_body(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            function_span(SOURCE, self.helper_row, max_body_lines=2)
        self.assertIn("above max_body_lines=2", str(ctx.exception))

    def test_max_body_lines_zero_means_no_limit(self) -> None:
        self.assertEqual(function_span(SOURCE, self.helper_row, max_body_lines=0)["body_lines"], 3)

    def test_truncated_end_line_needs_the_expansion_flag(self) -> None:
        truncated = dict(self.helper_row, end_line=8)  # chunk boundary inside the body
        with self.assertRaises(ValueError) as ctx:
            function_span(SOURCE, truncated)
        self.assertIn("disagree with AST", str(ctx.exception))
        span = function_span(SOURCE, truncated, expand_truncated_end=True)
        self.assertTrue(span["coordinates_expanded"])
        self.assertEqual(span["line_end"], 9)

    def test_expansion_never_accepts_an_end_line_past_the_definition(self) -> None:
        """The repair widens a truncated end; it must not accept an overshoot."""

        with self.assertRaises(ValueError):
            function_span(SOURCE, dict(self.helper_row, end_line=99), expand_truncated_end=True)

    # ---- policy configuration ------------------------------------------

    def test_new_policy_is_available_without_research_mode(self) -> None:
        generator = FIMPatchGenerator(RecordingBackend(), target_policy="first_supported_ast")
        self.assertEqual(generator.target_depth, TARGET_POLICY_DEPTH["first_supported_ast"])
        self.assertTrue(generator.expand_truncated_end)

    def test_legacy_research_policy_still_requires_research_mode(self) -> None:
        with self.assertRaises(ValueError):
            FIMPatchGenerator(RecordingBackend(), target_policy="first_supported_ast_top5")

    def test_unknown_policy_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FIMPatchGenerator(RecordingBackend(), target_policy="whatever")

    def test_strict_policy_keeps_strict_coordinates_by_default(self) -> None:
        self.assertFalse(FIMPatchGenerator(RecordingBackend(),
                                           target_policy="strict_top1").expand_truncated_end)

    def test_operational_default_scans_supported_top_five_targets(self) -> None:
        generator = FIMPatchGenerator(RecordingBackend())
        self.assertEqual(generator.target_policy, "first_supported_ast")
        self.assertEqual(generator.target_depth, 5)
        self.assertTrue(generator.expand_truncated_end)

    def test_expansion_can_be_set_explicitly_on_either_policy(self) -> None:
        self.assertTrue(FIMPatchGenerator(RecordingBackend(), expand_truncated_end=True).expand_truncated_end)
        self.assertFalse(FIMPatchGenerator(RecordingBackend(), target_policy="first_supported_ast",
                                           expand_truncated_end=False).expand_truncated_end)

    def test_max_body_lines_is_validated(self) -> None:
        for bad in (-1, True, 1.5, "50"):
            with self.assertRaises(ValueError):
                FIMPatchGenerator(RecordingBackend(), max_body_lines=bad)

    # ---- end-to-end selection ------------------------------------------

    def test_strict_policy_blocks_on_a_class_target(self) -> None:
        backend = RecordingBackend()
        result = FIMPatchGenerator(backend, candidates=1, target_policy="strict_top1").generate(
            self.ticket, self.localization([self.class_row, self.method_row]), self.repo)
        self.assertEqual(result["patch_status"], "blocked_invalid_context")
        self.assertEqual(backend.calls, 0)
        self.assertIsNone(result["target_rank"])
        self.assertEqual(len(result["target_resolution"]), 1)

    def test_scanning_policy_falls_through_to_the_first_supported_rank(self) -> None:
        backend = RecordingBackend(['        return "y"\n'])
        result = FIMPatchGenerator(backend, candidates=1, target_policy="first_supported_ast").generate(
            self.ticket, self.localization([self.class_row, self.method_row]), self.repo)
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["target_rank"], 2)
        self.assertTrue(result["target_fallback_used"])
        self.assertEqual([x["status"] for x in result["target_resolution"]], ["rejected", "selected"])
        self.assertIn("Unsupported or ambiguous", result["target_resolution"][0]["reason"])

    def test_rank_one_success_is_not_reported_as_a_fallback(self) -> None:
        backend = RecordingBackend(['        return "y"\n'])
        result = FIMPatchGenerator(backend, candidates=1, target_policy="first_supported_ast").generate(
            self.ticket, self.localization([self.method_row]), self.repo)
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["target_rank"], 1)
        self.assertFalse(result["target_fallback_used"])

    def test_scanning_policy_repairs_a_chunk_truncated_end_line(self) -> None:
        """The real Stage-3 failure mode: end_line is the 80-line chunk edge."""

        backend = RecordingBackend(["    return value.strip().lower()\n"])
        rows = [dict(self.helper_row, end_line=8)]
        result = FIMPatchGenerator(backend, candidates=1, target_policy="first_supported_ast").generate(
            self.ticket, self.localization(rows), self.repo)
        self.assertEqual(result["patch_status"], "generated")
        self.assertTrue(result["target_resolution"][0]["coordinates_expanded"])
        self.assertTrue(result["span"]["coordinates_expanded"])

        strict = FIMPatchGenerator(RecordingBackend(), candidates=1,
                                   target_policy="strict_top1").generate(
            self.ticket, self.localization(rows), self.repo)
        self.assertEqual(strict["patch_status"], "blocked_invalid_context")

    def test_size_limit_rejects_before_any_model_call(self) -> None:
        backend = RecordingBackend()
        result = FIMPatchGenerator(backend, candidates=1, target_policy="first_supported_ast",
                                   max_body_lines=2).generate(
            self.ticket, self.localization([self.helper_row]), self.repo)
        self.assertEqual(result["patch_status"], "blocked_invalid_context")
        self.assertEqual(backend.calls, 0)
        self.assertIn("max_body_lines", result["target_resolution"][0]["reason"])

    def test_size_limit_skips_the_oversized_rank_and_takes_the_next(self) -> None:
        backend = RecordingBackend(['        return "y"\n'])
        result = FIMPatchGenerator(backend, candidates=1, target_policy="first_supported_ast",
                                   max_body_lines=2).generate(
            self.ticket, self.localization([self.helper_row, self.method_row]), self.repo)
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["target_rank"], 2)
        self.assertEqual(result["max_body_lines"], 2)

    def test_selection_settings_are_recorded_in_the_result(self) -> None:
        """A run's results must say which policy produced them."""

        result = FIMPatchGenerator(RecordingBackend(), candidates=1,
                                   target_policy="first_supported_ast",
                                   max_body_lines=120).generate(
            self.ticket, self.localization([self.class_row]), self.repo)
        self.assertEqual(result["target_policy"], "first_supported_ast")
        self.assertTrue(result["expand_truncated_end"])
        self.assertEqual(result["max_body_lines"], 120)
        self.assertFalse(result["research_mode"])

    def test_source_is_never_modified_by_selection(self) -> None:
        FIMPatchGenerator(RecordingBackend(['        return "y"\n']), candidates=1,
                          target_policy="first_supported_ast").generate(
            self.ticket, self.localization([self.class_row, self.method_row]), self.repo)
        self.assertEqual((self.repo / "app.py").read_bytes(), SOURCE)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from typing import Any

from config import PipelineConfig
from utils.git_utils import run_patch_in_temp_copy


class RegressionTester:
    """Validate patch application and optionally run regression tests in a temp copy."""

    def __init__(self, *, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, patch: dict[str, Any], repo_path: str) -> dict[str, Any]:
        patch_text = str(patch.get("patch") or "")
        if not patch_text:
            return {
                "regression_tests": [],
                "ripr_analysis": _ripr(False),
                "regression_result": "not_run",
                "coverage": {"changed_code_coverage": 0.0},
                "reason": "No patch was available to validate.",
            }

        sandbox_result = run_patch_in_temp_copy(
            repo_path,
            patch_text,
            test_command=self.config.test_command,
            run_tests=self.config.run_regression_tests,
        )
        passed = sandbox_result.get("regression_result") in {"passed", "not_run"} and sandbox_result.get("patch_apply") == "passed"
        return {
            "regression_tests": self.config.test_command if self.config.run_regression_tests else [],
            "ripr_analysis": _ripr(bool(passed)),
            "regression_result": sandbox_result.get("regression_result", "failed"),
            "coverage": {"changed_code_coverage": 0.0},
            "patch_apply_check": sandbox_result.get("patch_apply_check"),
            "patch_apply": sandbox_result.get("patch_apply"),
            "stdout": _clip(sandbox_result.get("stdout", "")),
            "stderr": _clip(sandbox_result.get("stderr", "")),
        }


def _ripr(value: bool) -> dict[str, bool]:
    return {"reaching": value, "infecting": value, "propagating": value, "revealing": value}


def _clip(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"

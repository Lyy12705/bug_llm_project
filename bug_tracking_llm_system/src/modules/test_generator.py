from __future__ import annotations

from typing import Any


class TestGenerator:
    """Normalize executable reproduction tests supplied by a tracker or adapter.

    A generic deterministic test cannot safely be invented from prose alone.  The
    module therefore accepts an explicit test command and optional test-only
    unified diff, while retaining reproduction steps as a non-executable plan.
    """

    def generate_tests(self, ticket_json: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        provided = ticket_json.get("generated_tests") or ticket_json.get("tests")
        if isinstance(provided, list) and provided:
            normalized = [_normalize_test(item, index) for index, item in enumerate(provided, start=1)]
            executable = [item for item in normalized if item.get("command")]
            test_patch = _first_test_patch(ticket_json)
            return {
                "generated_tests": normalized,
                "test_commands": [item["command"] for item in executable],
                "test_patch": test_patch,
                "fib_passed_tests": [],
                "test_result": "provided_executable" if executable else "provided_non_executable",
                "requires_execution_approval": bool(executable or test_patch),
                "verification_required": True,
            }

        steps = ticket_json.get("steps_to_reproduce") or []
        if steps:
            return {
                "generated_tests": [],
                "fib_passed_tests": [],
                "test_result": "not_generated",
                "test_plan": {
                    "source": "steps_to_reproduce",
                    "steps": steps,
                    "expected_behavior": ticket_json.get("expected_behavior", ""),
                    "actual_behavior": ticket_json.get("actual_behavior", ""),
                },
                "fallback_used": True,
            }

        return {
            "generated_tests": [],
            "fib_passed_tests": [],
            "test_result": "not_generated",
            "fallback_used": True,
        }


def _normalize_test(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"test_name": f"reproduction_{index}", "command": item}
    if not isinstance(item, dict):
        return {"test_name": f"reproduction_{index}", "command": "", "invalid_input": True}
    command = item.get("command") or item.get("test_command") or item.get("argv") or ""
    return {
        **item,
        "test_name": str(item.get("test_name") or item.get("name") or item.get("file") or f"reproduction_{index}"),
        "command": command,
    }


def _first_test_patch(ticket_json: dict[str, Any]) -> str:
    for key in ("reproduction_test_patch", "test_patch"):
        value = ticket_json.get(key)
        if isinstance(value, str) and value.strip():
            return value if value.endswith("\n") else value + "\n"
    return ""

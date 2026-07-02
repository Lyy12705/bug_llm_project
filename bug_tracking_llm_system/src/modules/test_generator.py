from __future__ import annotations

from typing import Any


class TestGenerator:
    """Generate or pass through bug reproduction tests."""

    def generate_tests(self, ticket_json: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        provided = ticket_json.get("generated_tests") or ticket_json.get("tests")
        if isinstance(provided, list) and provided:
            return {
                "generated_tests": provided,
                "fib_passed_tests": [item.get("test_name", item.get("file", "")) for item in provided if isinstance(item, dict)],
                "test_result": "provided",
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

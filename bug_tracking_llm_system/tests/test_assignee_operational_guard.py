from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_operational_guard import (  # noqa: E402
    REQUIRED_SHADOW_CHECKS,
    build_operational_state,
    evaluate_ticket_guard,
)


class AssigneeOperationalGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 22, tzinfo=UTC)
        self.bundle = {
            "deployment_status": "approved",
            "approval_gate_passed": True,
            "approval_gates": {"all": True},
            "expires_at": (self.now + timedelta(days=10)).isoformat(),
        }
        self.roster = {
            "review_confirmed": True,
            "expires_at": (self.now + timedelta(days=10)).isoformat(),
        }
        self.shadow = {
            "deployment_gate": {
                "passed": True,
                "checks": {name: True for name in REQUIRED_SHADOW_CHECKS},
            }
        }

    def test_first_healthy_stage_can_only_start_at_five_percent(self) -> None:
        state = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="a" * 64,
            requested_rollout_percentage=5,
            now=self.now,
        )

        self.assertTrue(state["auto_assignment_enabled"])
        self.assertEqual(state["effective_rollout_percentage"], 5)
        self.assertFalse(state["kill_switch_active"])

        skipped = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="a" * 64,
            requested_rollout_percentage=10,
            now=self.now,
        )
        self.assertFalse(skipped["checks"]["rollout_transition_valid"])
        self.assertTrue(skipped["kill_switch_active"])

    def test_failed_shadow_check_immediately_activates_kill_switch(self) -> None:
        self.shadow["deployment_gate"]["checks"]["prediction_latency_p95"] = False
        state = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="b" * 64,
            requested_rollout_percentage=5,
            now=self.now,
        )

        self.assertTrue(state["kill_switch_active"])
        self.assertEqual(state["effective_rollout_percentage"], 0)
        decision = evaluate_ticket_guard(
            state,
            bundle_sha256="b" * 64,
            ticket_id="BUG-1",
            now=self.now,
        )
        self.assertFalse(decision["auto_assignment_authorized"])
        self.assertEqual(decision["reason"], "operational_kill_switch_active")

    def test_promotion_requires_seven_stable_days_and_no_stage_skipping(self) -> None:
        previous = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="c" * 64,
            requested_rollout_percentage=5,
            now=self.now,
        )
        previous["stage_entered_at"] = (self.now - timedelta(days=8)).isoformat()
        promoted = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="c" * 64,
            requested_rollout_percentage=10,
            previous_state=previous,
            now=self.now,
        )
        self.assertTrue(promoted["auto_assignment_enabled"])

        jumped = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="c" * 64,
            requested_rollout_percentage=25,
            previous_state=previous,
            now=self.now,
        )
        self.assertFalse(jumped["auto_assignment_enabled"])

    def test_state_is_bound_to_bundle_hash_and_expiration(self) -> None:
        state = build_operational_state(
            self.bundle,
            self.shadow,
            self.roster,
            bundle_sha256="d" * 64,
            requested_rollout_percentage=5,
            now=self.now,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_ticket_guard(
                state,
                bundle_sha256="e" * 64,
                ticket_id="BUG-2",
                now=self.now,
            )
        with self.assertRaisesRegex(ValueError, "expired"):
            evaluate_ticket_guard(
                state,
                bundle_sha256="d" * 64,
                ticket_id="BUG-2",
                now=self.now + timedelta(days=2),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from create_fault_localization_frozen_split import _holdout_status, build_repository_split


class FrozenHoldoutProtocolTests(unittest.TestCase):
    def test_repository_split_is_disjoint_and_deterministic(self) -> None:
        tickets = [
            {"ticket_id": "A-1", "repo": "org/a"},
            {"ticket_id": "A-2", "repo": "org/a"},
            {"ticket_id": "B-1", "repo": "org/b"},
            {"ticket_id": "C-1", "repo": "org/c"},
        ]
        gold = [{"ticket_id": row["ticket_id"], "fixed_files": ["src/main.py"]} for row in tickets]

        first = build_repository_split(tickets, gold, holdout_ratio=0.25, seed="fixed")
        second = build_repository_split(tickets, gold, holdout_ratio=0.25, seed="fixed")

        self.assertEqual(first["holdout_repositories"], second["holdout_repositories"])
        self.assertFalse(set(first["development_repositories"]) & set(first["holdout_repositories"]))
        development_ids = {row["ticket_id"] for row in first["development_tickets"]}
        holdout_ids = {row["ticket_id"] for row in first["holdout_tickets"]}
        self.assertFalse(development_ids & holdout_ids)
        self.assertEqual(development_ids | holdout_ids, {row["ticket_id"] for row in tickets})

    def test_split_rejects_missing_gold(self) -> None:
        with self.assertRaisesRegex(ValueError, "gold rows are missing"):
            build_repository_split(
                [{"ticket_id": "A-1", "repo": "org/a"}, {"ticket_id": "B-1", "repo": "org/b"}],
                [{"ticket_id": "A-1", "fixed_files": ["a.py"]}],
                holdout_ratio=0.5,
                seed="fixed",
            )

    def test_previously_evaluated_rows_are_not_labeled_untouched(self) -> None:
        self.assertEqual(
            _holdout_status("previously-evaluated"),
            "sealed_for_future_changes_prior_full_dataset_exposure",
        )


if __name__ == "__main__":
    unittest.main()

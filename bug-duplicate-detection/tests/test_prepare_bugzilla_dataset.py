from __future__ import annotations

import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_bugzilla_dataset import dedupe_bugs_by_id, rows_to_tickets


class PrepareBugzillaDatasetTest(unittest.TestCase):
    def test_rows_to_tickets_derives_duplicate_groups_from_bugzilla_rows(self) -> None:
        rows = [
            {"ticket_id": "100", "title": "master", "description": "original", "duplicate_of": "", "resolution": "FIXED"},
            {"ticket_id": "101", "title": "duplicate", "description": "same bug", "duplicate_of": "100", "resolution": "DUPLICATE"},
            {"ticket_id": "102", "title": "other", "description": "not a duplicate", "duplicate_of": "", "resolution": "FIXED"},
        ]

        tickets = rows_to_tickets(rows)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}

        self.assertEqual(by_id["100"].duplicate_group, by_id["101"].duplicate_group)
        self.assertIsNone(by_id["102"].duplicate_group)

    def test_dedupe_bugs_by_id_keeps_latest_seen_bug(self) -> None:
        bugs = [
            {"id": 100, "summary": "old"},
            {"id": 101, "summary": "other"},
            {"id": 100, "summary": "new"},
        ]

        deduped = dedupe_bugs_by_id(bugs)

        self.assertEqual(len(deduped), 2)
        self.assertEqual({str(bug["id"]): bug["summary"] for bug in deduped}["100"], "new")


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.dataset import TicketRecord
from scripts.build_duplicate_decision_dataset import build_decision_dataset, is_non_duplicate_candidate


class DecisionDatasetTest(unittest.TestCase):
    def test_build_decision_dataset_samples_non_duplicates(self) -> None:
        duplicates = [
            ticket("D-1", duplicate_group="G-1"),
            ticket("D-2", duplicate_of="D-1", duplicate_group="G-1"),
        ]
        source_non_duplicates = [
            ticket("N-1", resolution="FIXED"),
            ticket("N-2", resolution="WORKSFORME"),
            ticket("D-1", resolution="FIXED"),
            ticket("X-1", duplicate_of="D-1", resolution="DUPLICATE"),
        ]

        mixed, sampled, available = build_decision_dataset(
            duplicates,
            source_non_duplicates,
            negative_ratio=1.0,
            seed=13,
        )

        self.assertEqual(len(available), 2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len(mixed), 4)
        self.assertEqual({ticket.ticket_id for ticket in sampled}, {"N-1", "N-2"})

    def test_is_non_duplicate_candidate_rejects_duplicate_resolution(self) -> None:
        self.assertFalse(is_non_duplicate_candidate(ticket("N-1", resolution="DUPLICATE"), set()))
        self.assertTrue(is_non_duplicate_candidate(ticket("N-2", resolution="FIXED"), set()))


def ticket(
    ticket_id: str,
    *,
    duplicate_of: str | None = None,
    duplicate_group: str | None = None,
    resolution: str = "",
) -> TicketRecord:
    return TicketRecord(
        ticket_id=ticket_id,
        title=f"{ticket_id} title",
        content=f"{ticket_id} content",
        duplicate_of=duplicate_of,
        duplicate_group=duplicate_group,
        fields={"resolution": resolution},
    )


if __name__ == "__main__":
    unittest.main()

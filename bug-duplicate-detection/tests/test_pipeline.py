import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.dataset import load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.splits import kfold_ticket_splits, train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector
from duplicate_ticket_detection.triplets import build_triplets


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_tickets.csv"


class PipelineTest(unittest.TestCase):
    def test_load_tickets_derives_duplicate_groups(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}

        self.assertEqual(by_id["T-100"].duplicate_group, by_id["T-101"].duplicate_group)
        self.assertEqual(relevant_duplicate_ids(by_id["T-100"], tickets), {"T-101"})

    def test_build_triplets_from_duplicate_groups(self) -> None:
        tickets = load_tickets(SAMPLE)
        triplets = build_triplets(tickets, field="title")

        self.assertTrue(triplets)
        self.assertTrue(any(triplet.anchor_id == "T-100" and triplet.positive_id == "T-101" for triplet in triplets))

    def test_build_triplets_with_hard_negatives(self) -> None:
        tickets = load_tickets(SAMPLE)
        triplets = build_triplets(
            tickets,
            field="title",
            negative_strategy="hard",
            negatives_per_anchor=1,
        )

        self.assertTrue(triplets)
        self.assertTrue(all(triplet.negative_id not in {triplet.anchor_id, triplet.positive_id} for triplet in triplets))

    def test_tfidf_detector_ranks_duplicate_first(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector().fit(tickets)

        ranking = detector.rank(by_id["T-100"], tickets, top_k=1)

        self.assertEqual(ranking[0].ticket_id, "T-101")

    def test_tfidf_evaluate_returns_map(self) -> None:
        tickets = load_tickets(SAMPLE)
        detector = TfidfDuplicateDetector()

        mean_ap, rows = detector.evaluate(tickets)

        self.assertTrue(rows)
        self.assertGreater(mean_ap, 0.8)

    def test_train_test_split_keeps_evaluable_duplicate_queries(self) -> None:
        tickets = load_tickets(SAMPLE)
        split = train_test_split_tickets(tickets, test_size=0.5, seed=13)

        evaluable = [ticket for ticket in split.test if relevant_duplicate_ids(ticket, split.train)]

        self.assertTrue(evaluable)

    def test_kfold_splits_cover_all_tickets_once_as_test(self) -> None:
        tickets = load_tickets(SAMPLE)
        folds = kfold_ticket_splits(tickets, n_splits=5, seed=13)
        test_ids = [ticket.ticket_id for fold in folds for ticket in fold.test]

        self.assertEqual(sorted(test_ids), sorted(ticket.ticket_id for ticket in tickets))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.decision import (
    PairScore,
    QueryDecision,
    collect_pair_scores,
    collect_query_decisions,
    tune_query_threshold,
    tune_similarity_threshold,
)
from duplicate_ticket_detection.dataset import load_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_tickets.csv"


class DecisionTest(unittest.TestCase):
    def test_tune_similarity_threshold_prefers_perfect_duplicate_cutoff(self) -> None:
        pair_scores = [
            PairScore("Q1", "A", 0.90, True),
            PairScore("Q1", "B", 0.70, False),
            PairScore("Q2", "C", 0.20, False),
        ]

        metrics = tune_similarity_threshold(pair_scores)

        self.assertEqual(metrics.threshold, 0.90)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)

    def test_collect_pair_scores_labels_known_duplicates(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector().fit(tickets)

        pair_scores = collect_pair_scores(detector, [by_id["T-100"]], tickets)
        duplicate_scores = [pair for pair in pair_scores if pair.candidate_id == "T-101"]

        self.assertTrue(duplicate_scores)
        self.assertTrue(duplicate_scores[0].is_duplicate)

    def test_tune_query_threshold_reports_false_positives(self) -> None:
        decisions = [
            QueryDecision("Q1", "A", 0.90, True, True),
            QueryDecision("Q2", "B", 0.80, False, False),
            QueryDecision("Q3", "C", 0.40, True, True),
            QueryDecision("Q4", "D", 0.10, False, False),
        ]

        metrics = tune_query_threshold(decisions)

        self.assertEqual(metrics.threshold, 0.40)
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertEqual(metrics.false_negatives, 0)

    def test_collect_query_decisions_marks_non_duplicate_queries(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector().fit(tickets)

        decisions = collect_query_decisions(detector, [by_id["T-300"]], tickets)

        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0].is_duplicate)

    def test_collect_query_decisions_keeps_decision_features(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector().fit(tickets)

        decisions = collect_query_decisions(detector, [by_id["T-100"]], tickets)

        self.assertEqual(len(decisions), 1)
        self.assertGreaterEqual(decisions[0].title_score, 0.0)
        self.assertGreaterEqual(decisions[0].content_score, 0.0)
        self.assertGreaterEqual(decisions[0].score, decisions[0].second_score)

    def test_tune_query_threshold_can_enforce_min_precision(self) -> None:
        decisions = [
            QueryDecision("Q1", "A", 0.90, True, True),
            QueryDecision("Q2", "B", 0.80, False, False),
            QueryDecision("Q3", "C", 0.40, True, True),
            QueryDecision("Q4", "D", 0.10, False, False),
        ]

        metrics = tune_query_threshold(decisions, min_precision=1.0)

        self.assertEqual(metrics.threshold, 0.90)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 0)
        self.assertEqual(metrics.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()

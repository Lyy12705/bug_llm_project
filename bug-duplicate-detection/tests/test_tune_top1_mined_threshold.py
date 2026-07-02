import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duplicate_ticket_detection.dataset import TicketRecord
from tune_top1_mined_threshold import passes_constraints, query_decisions_from_scores


class TuneTop1MinedThresholdTest(unittest.TestCase):
    def test_query_decisions_from_scores_keeps_score_features(self) -> None:
        query = TicketRecord("Q", "login crash", "crash after login", duplicate_group="G1")
        positive = TicketRecord("P", "sign in crash", "crash after sign in", duplicate_group="G1")
        negative = TicketRecord("N", "button color", "wrong button color", duplicate_group=None)
        scores = np.array([[0.91, 0.84]])
        title_scores = np.array([[0.80, 0.20]])
        content_scores = np.array([[0.90, 0.50]])

        decisions = query_decisions_from_scores(
            [query],
            [positive, negative],
            scores,
            title_scores=title_scores,
            content_scores=content_scores,
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].best_candidate_id, "P")
        self.assertTrue(decisions[0].is_duplicate)
        self.assertTrue(decisions[0].best_is_correct_duplicate)
        self.assertAlmostEqual(decisions[0].score, 0.91)
        self.assertAlmostEqual(decisions[0].second_score, 0.84)
        self.assertAlmostEqual(decisions[0].title_score, 0.80)
        self.assertAlmostEqual(decisions[0].content_score, 0.90)

    def test_passes_constraints_checks_precision_and_recall(self) -> None:
        class Metrics:
            precision = 0.72
            recall = 0.51

        self.assertTrue(passes_constraints(Metrics, min_precision=0.70, min_recall=0.50))
        self.assertFalse(passes_constraints(Metrics, min_precision=0.80, min_recall=0.50))
        self.assertFalse(passes_constraints(Metrics, min_precision=0.70, min_recall=0.60))


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.dataset import TicketRecord
from duplicate_ticket_detection.stable_rerank import StableRerankConfig, apply_stable_rerank_scores, rank_agreement_row


class StableRerankTest(unittest.TestCase):
    def test_stable_rerank_promotes_candidate_supported_by_tfidf_and_metadata(self) -> None:
        query = TicketRecord("Q", "crash", "login crash", duplicate_group="G1", fields={"component": "Auth", "product": "Demo"})
        wrong = TicketRecord("W", "crash", "other crash", fields={"component": "UI", "product": "Demo"})
        right = TicketRecord("R", "sign in crash", "login crash", duplicate_group="G1", fields={"component": "Auth", "product": "Demo"})
        model_scores = np.array([[0.91, 0.90]])
        tfidf_scores = np.array([[0.10, 0.95]])

        stable = apply_stable_rerank_scores(
            [query],
            [wrong, right],
            model_scores,
            tfidf_scores,
            StableRerankConfig(tfidf_weight=0.45, metadata_weight=0.20, agreement_weight=0.0),
        )

        self.assertGreater(stable[0, 1], stable[0, 0])

    def test_rank_agreement_rewards_close_ranks(self) -> None:
        model_scores = np.array([0.9, 0.8, 0.1])
        tfidf_scores = np.array([0.85, 0.2, 0.7])

        agreement = rank_agreement_row(model_scores, tfidf_scores, candidate_pool=3)

        self.assertGreater(agreement[0], agreement[1])
        self.assertGreater(agreement[0], agreement[2])

    def test_stable_rerank_keeps_query_self_match_excluded_after_metadata(self) -> None:
        query = TicketRecord(
            "Q",
            "login crash",
            "browser crashes on login",
            fields={"component": "Auth", "product": "Demo", "severity": "major", "priority": "P1"},
        )
        other = TicketRecord(
            "O",
            "settings crash",
            "settings page crashes",
            fields={"component": "Settings", "product": "Demo", "severity": "minor", "priority": "P3"},
        )
        model_scores = np.array([[1.0, 0.2]])
        tfidf_scores = np.array([[1.0, 0.2]])

        stable = apply_stable_rerank_scores(
            [query],
            [query, other],
            model_scores,
            tfidf_scores,
            StableRerankConfig(metadata_weight=0.20),
        )

        self.assertTrue(np.isneginf(stable[0, 0]))


if __name__ == "__main__":
    unittest.main()

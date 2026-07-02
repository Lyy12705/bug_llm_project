import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duplicate_ticket_detection.dataset import load_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector
from export_duplicate_review_queue import build_review_rows, render_summary, review_confidence, review_priority


SAMPLE = ROOT / "examples" / "sample_tickets.csv"


class ExportDuplicateReviewQueueTest(unittest.TestCase):
    def test_build_review_rows_exports_top_k_candidates_for_engineer_review(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector(combine="mean").fit(tickets)

        rows = build_review_rows(
            detector,
            [by_id["T-100"]],
            tickets,
            top_k=3,
            rerank_config=None,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["query_id"], "T-100")
        self.assertEqual(rows[0]["rank"], "1")
        self.assertEqual(rows[0]["candidate_id"], "T-101")
        self.assertEqual(rows[0]["candidate_is_known_duplicate"], "true")
        self.assertEqual(rows[0]["known_duplicate_in_top_k"], "true")
        self.assertIn(rows[0]["review_priority"], {"high", "medium", "normal"})

    def test_review_priority_prefers_rank_one_with_metadata_or_margin(self) -> None:
        self.assertEqual(review_priority(rank=1, metadata_match_count=2, top1_margin=0.0), "high")
        self.assertEqual(review_priority(rank=1, metadata_match_count=0, top1_margin=0.04), "high")
        self.assertEqual(review_priority(rank=1, metadata_match_count=0, top1_margin=0.0, rank_agreement="both_top10"), "high")
        self.assertEqual(review_priority(rank=2, metadata_match_count=3, top1_margin=0.1), "medium")
        self.assertEqual(review_priority(rank=4, metadata_match_count=3, top1_margin=0.1), "normal")

    def test_stable_rerank_adds_tfidf_agreement_features(self) -> None:
        tickets = load_tickets(SAMPLE)
        by_id = {ticket.ticket_id: ticket for ticket in tickets}
        detector = TfidfDuplicateDetector(combine="mean").fit(tickets)
        stable_detector = TfidfDuplicateDetector(combine="mean").fit(tickets)

        rows = build_review_rows(
            detector,
            [by_id["T-100"]],
            tickets,
            top_k=3,
            rerank_config=None,
            stable_detector=stable_detector,
            stable_candidate_pool=5,
        )

        self.assertEqual(rows[0]["candidate_id"], "T-101")
        self.assertEqual(rows[0]["model_rank"], "1")
        self.assertEqual(rows[0]["tfidf_rank"], "1")
        self.assertEqual(rows[0]["rank_agreement"], "both_top3")
        self.assertIn(rows[0]["review_confidence"], {"strong", "moderate", "needs_review", "weak"})

    def test_review_confidence_marks_multi_signal_rank_one_as_strong(self) -> None:
        self.assertEqual(review_confidence(rank=1, metadata_match_count=2, top1_margin=0.01, rank_agreement="both_top3"), "strong")
        self.assertEqual(review_confidence(rank=2, metadata_match_count=0, top1_margin=0.01, rank_agreement="both_top10"), "moderate")
        self.assertEqual(review_confidence(rank=2, metadata_match_count=0, top1_margin=0.01, rank_agreement="model_only"), "needs_review")

    def test_render_summary_reports_topk_hit_rate_when_labels_exist(self) -> None:
        rows = [
            {
                "query_id": "Q",
                "rank": "1",
                "candidate_is_known_duplicate": "false",
                "known_duplicate_in_top_k": "true",
                "known_relevant_ids": "P",
                "review_priority": "medium",
                "review_confidence": "moderate",
            }
        ]

        summary = render_summary(rows, mode="test", top_k=10)

        self.assertIn("top1_hit_rate=0.0000", summary)
        self.assertIn("topk_hit_rate=1.0000", summary)
        self.assertIn("moderate_confidence_rows=1", summary)


if __name__ == "__main__":
    unittest.main()

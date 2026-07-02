import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duplicate_ticket_detection.dataset import load_tickets
from duplicate_ticket_detection.triplets import TripletRecord, build_triplets
from run_top1_improvement_experiments import build_mined_negative_triplets, capped_triplets, parse_stable_config, stable_label


SAMPLE = ROOT / "examples" / "sample_tickets.csv"


class Top1ImprovementTest(unittest.TestCase):
    def test_build_mined_negative_triplets_uses_wrong_top1_as_negative(self) -> None:
        tickets = load_tickets(SAMPLE)

        triplets = build_mined_negative_triplets(
            tickets,
            field="title",
            mined_negatives_by_anchor={"T-100": ["T-102"]},
        )

        self.assertTrue(triplets)
        self.assertTrue(
            any(
                triplet.anchor_id == "T-100"
                and triplet.positive_id == "T-101"
                and triplet.negative_id == "T-102"
                for triplet in triplets
            )
        )

    def test_capped_triplets_keeps_mined_triplets_before_sampling_base(self) -> None:
        tickets = load_tickets(SAMPLE)
        mined = build_mined_negative_triplets(
            tickets,
            field="title",
            mined_negatives_by_anchor={"T-100": ["T-102"]},
        )
        base = build_triplets(tickets, field="title", negative_strategy="all")

        capped = capped_triplets(mined, base, max_triplets=1, seed=13)

        self.assertEqual(len(capped), 1)
        self.assertEqual(capped[0].anchor_id, "T-100")
        self.assertEqual(capped[0].negative_id, "T-102")

    def test_capped_triplets_deduplicates_priority_and_base(self) -> None:
        priority = [
            TripletRecord("A", "B", "C", "a", "b", "c", "title"),
            TripletRecord("A", "B", "C", "a", "b", "c", "title"),
        ]
        base = [
            TripletRecord("A", "B", "C", "a", "b", "c", "title"),
            TripletRecord("A", "B", "D", "a", "b", "d", "title"),
        ]

        capped = capped_triplets(priority, base, max_triplets=None, seed=13)

        self.assertEqual([(row.anchor_id, row.positive_id, row.negative_id) for row in capped], [("A", "B", "C"), ("A", "B", "D")])

    def test_parse_stable_config_for_top1_experiment(self) -> None:
        config = parse_stable_config("tfidf:0.4,metadata:0.1,agreement:0.05", candidate_pool=25)

        self.assertEqual(config.candidate_pool, 25)
        self.assertAlmostEqual(config.tfidf_weight, 0.4)
        self.assertAlmostEqual(config.metadata_weight, 0.1)
        self.assertAlmostEqual(config.agreement_weight, 0.05)
        self.assertEqual(stable_label(config), "stable_tfidf0p4_metadata0p1_agreement0p05")


if __name__ == "__main__":
    unittest.main()

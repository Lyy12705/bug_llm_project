import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.metrics import average_precision, mean_average_precision


class MetricsTest(unittest.TestCase):
    def test_average_precision_scores_ranked_duplicates(self) -> None:
        ranked = ["A", "B", "C", "D"]
        relevant = {"B", "D"}

        self.assertEqual(average_precision(ranked, relevant), (1 / 2 + 2 / 4) / 2)

    def test_average_precision_does_not_count_repeated_relevant_id_twice(self) -> None:
        ranked = ["A", "A", "B"]
        relevant = {"A"}

        self.assertEqual(average_precision(ranked, relevant), 1.0)

    def test_mean_average_precision_ignores_queries_without_relevant_items(self) -> None:
        results = {
            "Q1": (["A", "B"], {"B"}),
            "Q2": (["C", "D"], set()),
        }

        self.assertEqual(mean_average_precision(results), 0.5)


if __name__ == "__main__":
    unittest.main()

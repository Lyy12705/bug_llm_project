import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.show_duplicate_results import (
    read_decision_json,
    read_results_csv,
    render_decision_matrix,
    render_best_ranking_result,
    render_results_table,
    resolve_decision_json,
    resolve_results_csv,
)


FIELDNAMES = [
    "experiment",
    "method",
    "combine",
    "fold",
    "train_size",
    "test_size",
    "queries",
    "mean_average_precision",
    "top_1_accuracy",
    "top_k_hit_rate",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "base_model",
    "epochs",
    "max_triplets",
    "seconds",
]


class ShowDuplicateResultsTest(unittest.TestCase):
    def test_auto_select_best_result_csv_by_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            low = reports_dir / "tfidf.csv"
            high = reports_dir / "sbert.csv"
            write_result_csv(low, experiment="tfidf_mean", map_score=0.53)
            write_result_csv(high, experiment="sbert_mean", map_score=0.65)

            selected = resolve_results_csv(
                None,
                reports_dir=reports_dir,
                selection="best",
                score_column="mean_average_precision",
                fold="mean",
            )

            self.assertEqual(selected, high)

    def test_results_table_hides_precision_and_keeps_recall_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            write_result_csv(path, experiment="sbert_mean", map_score=0.65)

            rows = read_results_csv(path)
            table = render_results_table(rows, title="Results")

            self.assertIn("Recall", table)
            self.assertNotIn("Precision", table)
            self.assertNotIn("P@k", table)
            self.assertNotIn("R@k", table)

    def test_best_ranking_result_renders_metric_value_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            write_result_csv(path, experiment="sbert_mean", map_score=0.65)

            row = read_results_csv(path)[0]
            summary = render_best_ranking_result(row, title="Best Ranking Result")

            self.assertIn("Best Ranking Result", summary)
            self.assertIn("Metric", summary)
            self.assertIn("Value", summary)
            self.assertIn("Experiment", summary)
            self.assertIn("sbert_mean", summary)
            self.assertIn("0.6500", summary)
            self.assertIn("0.8000", summary)
            self.assertIn("Base model", summary)
            self.assertNotIn("Rank  Method", summary)
            self.assertNotIn("|", summary)

    def test_decision_matrix_renders_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tfidf_decision_threshold.json"
            write_decision_json(path)

            result = read_decision_json(path)
            table = render_decision_matrix(result)

            self.assertIn("Duplicate Decision Metrics", table)
            self.assertIn("Duplicate Decision Confusion Matrix", table)
            self.assertIn("False positive rate", table)
            self.assertIn("0.3519", table)
            self.assertIn("Precision", table)
            self.assertIn("0.9064", table)
            self.assertIn("Rerank fields", table)
            self.assertIn("component:0.02", table)
            self.assertIn("Constraints satisfied", table)
            self.assertIn("true", table)
            self.assertIn("Non-duplicate", table)
            self.assertIn("19", table)

    def test_auto_discovers_decision_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            path = reports_dir / "tfidf_decision_threshold.json"
            write_decision_json(path)

            selected = resolve_decision_json(None, reports_dir=reports_dir, auto=True)

            self.assertEqual(selected, path)

    def test_auto_prefers_conservative_sbert_p90_decision_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            latest = reports_dir / "top1_mined_decision_threshold_balanced.json"
            preferred = reports_dir / "sbert_decision_threshold_p90.json"
            write_decision_json(latest)
            write_decision_json(preferred)

            selected = resolve_decision_json(None, reports_dir=reports_dir, auto=True)

            self.assertEqual(selected, preferred)


def write_result_csv(path: Path, *, experiment: str, map_score: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "experiment": experiment,
                "method": "sbert" if experiment.startswith("sbert") else "tfidf",
                "combine": "mean",
                "fold": "mean",
                "train_size": 0,
                "test_size": 0,
                "queries": 10,
                "mean_average_precision": map_score,
                "top_1_accuracy": 0.60,
                "top_k_hit_rate": 0.87,
                "precision_at_k": 0.19,
                "recall_at_k": 0.80,
                "mean_reciprocal_rank": 0.70,
                "base_model": "",
                "epochs": 1,
                "max_triplets": 2000,
                "seconds": 12.3,
            }
        )


def write_decision_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "split": "train_test_seed_13",
                "method": "tfidf",
                "combine": "weighted:0.4,0.6",
                "train": 712,
                "validation": 361,
                "threshold": 0.282960,
                "precision": 0.9064,
                "recall": 0.5993,
                "f1": 0.7216,
                "accuracy": 0.6066,
                "true_positives": 184,
                "false_positives": 19,
                "true_negatives": 35,
                "false_negatives": 123,
                "queries": 361,
                "positive_queries": 307,
                "correct_duplicate_links": 121,
                "min_precision": 0.9,
                "min_recall": None,
                "rerank_fields": "component:0.02",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "constraints_satisfied": True,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

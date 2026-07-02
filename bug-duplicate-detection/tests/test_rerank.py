import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplicate_ticket_detection.dataset import TicketRecord
from duplicate_ticket_detection.rerank import RerankConfig, metadata_match_bonus, rerank_ranked_tickets
from duplicate_ticket_detection.tfidf_detector import RankedTicket


class RerankTest(unittest.TestCase):
    def test_metadata_match_bonus_uses_matching_fields(self) -> None:
        query = ticket("Q", component="Bookmarks", product="Firefox")
        candidate = ticket("C", component="Bookmarks", product="Firefox")

        bonus = metadata_match_bonus(query, candidate, (("component", 0.05), ("product", 0.02)))

        self.assertAlmostEqual(bonus, 0.07)

    def test_rerank_can_promote_metadata_match_to_first(self) -> None:
        query = ticket("Q", component="Bookmarks")
        candidates = [
            ticket("A", component="General"),
            ticket("B", component="Bookmarks"),
        ]
        ranking = [
            ranked("A", score=0.70),
            ranked("B", score=0.68),
        ]
        config = RerankConfig(field_weights=(("component", 0.05),), candidate_pool=10)

        reranked = rerank_ranked_tickets(query, ranking, candidates, config, top_k=2)

        self.assertEqual([row.ticket_id for row in reranked], ["B", "A"])


def ticket(ticket_id: str, *, component: str, product: str = "Firefox") -> TicketRecord:
    return TicketRecord(
        ticket_id=ticket_id,
        title=f"{ticket_id} title",
        content=f"{ticket_id} content",
        fields={"component": component, "product": product},
    )


def ranked(ticket_id: str, *, score: float) -> RankedTicket:
    return RankedTicket(
        ticket_id=ticket_id,
        score=score,
        title_score=score,
        content_score=score,
        title=f"{ticket_id} title",
    )


if __name__ == "__main__":
    unittest.main()

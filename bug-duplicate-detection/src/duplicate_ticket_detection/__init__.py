"""Duplicate ticket detection toolkit."""

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets
from duplicate_ticket_detection.decision import (
    DecisionMetrics,
    PairScore,
    QueryDecision,
    ThresholdMetrics,
    tune_query_threshold,
    tune_similarity_threshold,
)
from duplicate_ticket_detection.metrics import average_precision, mean_average_precision
from duplicate_ticket_detection.rerank import RerankConfig, apply_metadata_rerank_scores, rerank_ranked_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector
from duplicate_ticket_detection.triplets import TripletRecord, build_triplets

__all__ = [
    "PairScore",
    "DecisionMetrics",
    "QueryDecision",
    "RerankConfig",
    "TicketRecord",
    "ThresholdMetrics",
    "TripletRecord",
    "TfidfDuplicateDetector",
    "apply_metadata_rerank_scores",
    "average_precision",
    "build_triplets",
    "load_tickets",
    "mean_average_precision",
    "rerank_ranked_tickets",
    "tune_query_threshold",
    "tune_similarity_threshold",
]

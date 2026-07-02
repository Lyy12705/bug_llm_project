from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_COMPONENT_OWNERS = {
    "authentication": "auth-team@example.com",
    "auth": "auth-team@example.com",
    "build": "build-team@example.com",
    "compiler": "build-team@example.com",
    "database": "data-team@example.com",
    "db": "data-team@example.com",
    "frontend": "frontend-team@example.com",
    "ui": "frontend-team@example.com",
    "api": "backend-team@example.com",
    "backend": "backend-team@example.com",
    "security": "security-team@example.com",
    "unknown": "manual_triage",
}

DEFAULT_ASSIGNEE_HISTORY_RELATIVE_PATH = (
    Path("assignee_triage_accuracy")
    / "paper_grade"
    / "data"
    / "processed"
    / "bmo_paper_2024_3k_history_train.jsonl"
)


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path = PROJECT_ROOT
    historical_tickets_path: Path | None = None
    assignee_dataset_path: Path | None = None
    processed_ticket_dir: Path | None = None
    duplicate_threshold: float = 0.82
    duplicate_review_margin: float = 0.05
    duplicate_top_k: int = 5
    component_owner_mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COMPONENT_OWNERS))
    test_command: list[str] = field(default_factory=lambda: ["python3", "-m", "pytest"])
    run_regression_tests: bool = False
    save_checkpoints: bool = True
    fault_localization_code_index_path: Path | None = None
    fault_localization_top_k: int = 5
    fault_localization_embedding_backend: str = "tfidf"
    fault_localization_sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fault_localization_sbert_local_files_only: bool = True
    fault_localization_sbert_cache_dir: Path | None = None
    fault_localization_llm_rerank: bool = False
    fault_localization_llm_candidate_k: int = 10
    fault_localization_llm_cache_dir: Path | None = None
    fault_localization_ollama_model: str = "codellama:7b-instruct"
    fault_localization_ollama_url: str = "http://localhost:11434/api/generate"
    fault_localization_ollama_timeout: int = 180

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.historical_tickets_path is None:
            self.historical_tickets_path = self.project_root / "data" / "historical_tickets.jsonl"
        else:
            self.historical_tickets_path = Path(self.historical_tickets_path)

        if self.assignee_dataset_path is None:
            self.assignee_dataset_path = default_assignee_dataset_path(self.project_root)
        else:
            self.assignee_dataset_path = Path(self.assignee_dataset_path)

        if self.processed_ticket_dir is None:
            self.processed_ticket_dir = self.project_root / "data" / "processed_tickets"
        else:
            self.processed_ticket_dir = Path(self.processed_ticket_dir)

        if self.fault_localization_code_index_path is not None:
            self.fault_localization_code_index_path = Path(self.fault_localization_code_index_path)
        if self.fault_localization_sbert_cache_dir is not None:
            self.fault_localization_sbert_cache_dir = Path(self.fault_localization_sbert_cache_dir)
        if self.fault_localization_llm_cache_dir is not None:
            self.fault_localization_llm_cache_dir = Path(self.fault_localization_llm_cache_dir)


def config_from_dict(values: dict[str, Any]) -> PipelineConfig:
    data = dict(values)
    for key in (
        "project_root",
        "historical_tickets_path",
        "assignee_dataset_path",
        "processed_ticket_dir",
        "fault_localization_code_index_path",
        "fault_localization_sbert_cache_dir",
        "fault_localization_llm_cache_dir",
    ):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return PipelineConfig(**data)


def default_assignee_dataset_path(project_root: Path) -> Path:
    bmo_history_path = project_root / DEFAULT_ASSIGNEE_HISTORY_RELATIVE_PATH
    if bmo_history_path.exists():
        return bmo_history_path
    return project_root / "data" / "assignee_dataset.jsonl"

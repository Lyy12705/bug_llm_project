from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent if (SOURCE_ROOT.parent / "pyproject.toml").exists() else SOURCE_ROOT


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
    component_owner_mapping: dict[str, str] = field(default_factory=dict)
    assignee_top_k: int = 5
    assignee_confidence_threshold: float = 0.55
    assignee_min_score: float = 1.0
    assignee_allow_text_only_assignment: bool = False
    assignee_allow_uncalibrated_auto_assignment: bool = False
    assignee_active_roster_path: Path | None = None
    assignee_inactive_path: Path | None = None
    assignee_component_ownership_path: Path | None = None
    assignee_file_ownership_path: Path | None = None
    assignee_feedback_path: Path | None = None
    assignee_routing_policy_path: Path | None = None
    assignee_routing_policy_name: str = ""
    assignee_calibration_artifact_path: Path | None = None
    assignee_open_set_enabled: bool = False
    assignee_open_set_risk_threshold: float = 0.75
    assignee_open_set_artifact_path: Path | None = None
    test_command: list[str] = field(default_factory=lambda: ["python3", "-m", "pytest"])
    run_regression_tests: bool = False
    allow_ticket_test_commands: bool = False
    test_timeout_seconds: int = 300
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
        self.test_timeout_seconds = max(1, int(self.test_timeout_seconds))
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

        for key in (
            "assignee_active_roster_path",
            "assignee_inactive_path",
            "assignee_component_ownership_path",
            "assignee_file_ownership_path",
            "assignee_feedback_path",
            "assignee_routing_policy_path",
            "assignee_calibration_artifact_path",
            "assignee_open_set_artifact_path",
        ):
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, Path(value))

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
        "assignee_active_roster_path",
        "assignee_inactive_path",
        "assignee_component_ownership_path",
        "assignee_file_ownership_path",
        "assignee_feedback_path",
        "assignee_routing_policy_path",
        "assignee_calibration_artifact_path",
        "assignee_open_set_artifact_path",
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

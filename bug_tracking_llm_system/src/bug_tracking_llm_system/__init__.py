"""Public package facade for the integrated bug tracking pipeline."""

from config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator, build_default_orchestrator

__all__ = ["PipelineConfig", "PipelineOrchestrator", "build_default_orchestrator"]
__version__ = "0.1.0"

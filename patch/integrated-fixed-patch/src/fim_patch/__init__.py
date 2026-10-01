"""Existing-model FIM patch generation for Stage-3 localization outputs."""
from .generator import FIMPatchGenerator
from .backend import OllamaFIM

__all__ = ["FIMPatchGenerator", "OllamaFIM"]

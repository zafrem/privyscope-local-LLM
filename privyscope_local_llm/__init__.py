"""privyscope-local-llm: verify/extend privyscope detections with a local LLM."""
from .config import LocalLLMConfig
from .scorer import Scorer, ScoringError
from .stage import LocalLLMStage, default_stage

__all__ = ["LocalLLMConfig", "Scorer", "ScoringError", "LocalLLMStage", "default_stage"]
__version__ = "0.1.0"

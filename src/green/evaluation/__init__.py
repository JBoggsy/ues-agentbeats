"""Criteria evaluation for the Green agent (Phase 3.7).

This package contains the CriteriaJudge and supporting components for
evaluating Purple agent performance against scenario-defined criteria.
It supports both programmatic evaluators and LLM-based evaluation.

Modules:
    judge: Main CriteriaJudge class for orchestrating evaluation
    models: Data models for LLM evaluation results
    prompts: LLM prompt templates for criterion evaluation
"""

from src.green.evaluation.judge import (
    CriteriaJudge,
    CriteriaJudgeError,
    EvaluationError,
)
from src.green.evaluation.models import LLMEvaluationResult
from src.green.evaluation.prompts import (
    LLM_EVALUATION_SYSTEM_PROMPT,
    LLM_EVALUATION_USER_TEMPLATE,
    build_action_log_section,
    build_evaluation_context_section,
    build_state_comparison_section,
)

__all__ = [
    # Judge (Phase 3.7)
    "CriteriaJudge",
    "CriteriaJudgeError",
    "EvaluationError",
    # Models
    "LLMEvaluationResult",
    # Prompts
    "LLM_EVALUATION_SYSTEM_PROMPT",
    "LLM_EVALUATION_USER_TEMPLATE",
    "build_evaluation_context_section",
    "build_action_log_section",
    "build_state_comparison_section",
]

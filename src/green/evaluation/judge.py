"""Criteria Judge for evaluating Purple agent performance.

This module provides the CriteriaJudge class which evaluates Purple agent
performance against scenario-defined evaluation criteria. It supports both
programmatic evaluators (Python functions) and LLM-based evaluation using
prompt templates.

Classes:
    CriteriaJudge: Evaluates Purple agent performance against criteria.
    CriteriaJudgeError: Base exception for criteria judge errors.
    EvaluationError: Error during criterion evaluation.

Key Features:
    - Parallel criterion evaluation for efficiency
    - Score scaling to match criterion max_score
    - Graceful error handling with 0-score fallback
    - TaskUpdateEmitter integration for observability
    - Support for both programmatic and LLM-based evaluators

Example:
    >>> from src.green.judge import CriteriaJudge
    >>> from src.green.llm_config import LLMFactory
    >>>
    >>> llm = LLMFactory.create("gpt-4o-mini")
    >>> judge = CriteriaJudge(
    ...     llm=llm,
    ...     criteria=scenario.criteria,
    ...     evaluators=evaluator_registry,
    ...     emitter=task_update_emitter,
    ... )
    >>>
    >>> results = await judge.evaluate_all(eval_context)
    >>> scores = judge.aggregate_scores(results)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.agentbeats.results import (
    CriterionResult,
    DimensionScore,
    OverallScore,
    Scores,
    ScoringDimension,
)
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.evaluation.models import LLMEvaluationResult
from src.green.evaluation.prompts import (
    LLM_EVALUATION_SYSTEM_PROMPT,
    LLM_EVALUATION_USER_TEMPLATE,
    build_evaluation_context_section,
)
from src.green.scenarios.schema import (
    AgentBeatsEvalContext,
    EvalResult,
    EvaluationCriterion,
    EvaluatorRegistry,
)


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class CriteriaJudgeError(Exception):
    """Base exception for criteria judge errors.

    Attributes:
        message: Human-readable error description.
    """

    pass


class EvaluationError(CriteriaJudgeError):
    """Error during criterion evaluation.

    Raised when a criterion evaluation fails, either from a programmatic
    evaluator or LLM-based evaluation.

    Attributes:
        criterion_id: ID of the criterion that failed.
        message: Human-readable error description.
        original_error: The original exception that caused the failure.
    """

    def __init__(
        self,
        criterion_id: str,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            criterion_id: ID of the criterion that failed.
            message: Human-readable error description.
            original_error: The original exception that caused the failure.
        """
        super().__init__(f"Criterion '{criterion_id}': {message}")
        self.criterion_id = criterion_id
        self.original_error = original_error


# =============================================================================
# CriteriaJudge Class
# =============================================================================


class CriteriaJudge:
    """Evaluates Purple agent performance against scenario criteria.

    The CriteriaJudge orchestrates the evaluation of all criteria defined
    in a scenario. It supports two evaluation modes:

    1. **Programmatic evaluators**: Python async functions loaded from the
       scenario's evaluators.py file. These are referenced by `evaluator_id`
       in the criterion definition.

    2. **LLM-based evaluation**: Using the `evaluation_prompt` field in the
       criterion definition to construct a prompt for the LLM.

    Evaluation happens in parallel for efficiency. Each evaluator should be
    independent and not rely on the results of other evaluators.

    The judge handles:
    - Dispatching to the appropriate evaluation method
    - Scaling scores to match criterion max_score
    - Catching and logging errors (awarding 0 points on failure)
    - Emitting TaskUpdateEvents for observability
    - Aggregating scores by dimension and overall

    Attributes:
        llm: LangChain LLM for LLM-based evaluation.
        criteria: List of evaluation criteria from scenario.
        emitter: Optional TaskUpdateEmitter for observability.

    Note:
        Each evaluator should be independent and stateless to allow parallel
        execution. Do not rely on evaluation order or shared state.

    Example:
        >>> judge = CriteriaJudge(
        ...     llm=llm,
        ...     criteria=scenario.criteria,
        ...     evaluators=evaluator_registry,
        ...     emitter=emitter,
        ... )
        >>> results = await judge.evaluate_all(ctx)
        >>> scores = judge.aggregate_scores(results)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        criteria: list[EvaluationCriterion],
        evaluators: EvaluatorRegistry,
        emitter: TaskUpdateEmitter | None = None,
    ) -> None:
        """Initialize the CriteriaJudge.

        Args:
            llm: LangChain LLM for LLM-based evaluation.
            criteria: List of evaluation criteria from scenario.
            evaluators: Map of evaluator_id -> evaluator function.
            emitter: Optional TaskUpdateEmitter for observability.
        """
        self._llm = llm
        self._criteria = criteria
        self._evaluators = evaluators
        self._emitter = emitter

        # Pre-validate criteria have valid evaluation methods
        for criterion in criteria:
            evaluator_referenced = criterion.evaluator_id is not None
            evaluator_found = (
                evaluator_referenced and criterion.evaluator_id in evaluators
            )
            has_llm = criterion.evaluation_prompt is not None

            # Warn if evaluator_id references a missing evaluator
            if evaluator_referenced and not evaluator_found:
                if has_llm:
                    logger.warning(
                        "Criterion '%s' references evaluator_id '%s' which is not "
                        "registered. Will fall back to LLM evaluation.",
                        criterion.criterion_id,
                        criterion.evaluator_id,
                    )
                else:
                    logger.warning(
                        "Criterion '%s' references evaluator_id '%s' which is not "
                        "registered and has no evaluation_prompt. Evaluation will fail.",
                        criterion.criterion_id,
                        criterion.evaluator_id,
                    )

    @property
    def criteria(self) -> list[EvaluationCriterion]:
        """Return the list of evaluation criteria."""
        return self._criteria

    @property
    def llm(self) -> BaseChatModel:
        """Return the LLM used for evaluation."""
        return self._llm

    async def evaluate_all(
        self,
        ctx: AgentBeatsEvalContext,
    ) -> list[CriterionResult]:
        """Evaluate all criteria and return results.

        Runs all criterion evaluations in parallel for efficiency. Each
        evaluation is independent, so failures in one criterion do not
        affect others.

        Args:
            ctx: Evaluation context with UES client, action log, states, etc.

        Returns:
            List of CriterionResult, one per criterion. Failed evaluations
            will have score=0 with an error explanation.
        """
        logger.info("Starting evaluation of %d criteria", len(self._criteria))

        # Create evaluation tasks for all criteria
        tasks = [
            self._evaluate_criterion_safe(ctx, criterion)
            for criterion in self._criteria
        ]

        # Run all evaluations in parallel
        results = await asyncio.gather(*tasks)

        logger.info(
            "Completed evaluation: %d/%d criteria evaluated successfully",
            sum(1 for r in results if r.score > 0 or "error" not in r.explanation.lower()),
            len(results),
        )

        return results

    async def _evaluate_criterion_safe(
        self,
        ctx: AgentBeatsEvalContext,
        criterion: EvaluationCriterion,
    ) -> CriterionResult:
        """Evaluate a single criterion with error handling.

        Wraps _evaluate_criterion to catch all exceptions and return a
        zero-score result with error explanation on failure.

        Args:
            ctx: Evaluation context.
            criterion: The criterion to evaluate.

        Returns:
            CriterionResult (always succeeds, may have zero score on error).
        """
        try:
            return await self._evaluate_criterion(ctx, criterion)
        except Exception as e:
            logger.error(
                "Error evaluating criterion '%s': %s",
                criterion.criterion_id,
                str(e),
                exc_info=True,
            )

            # Create error result with zero score
            result = CriterionResult(
                criterion_id=criterion.criterion_id,
                name=criterion.name,
                dimension=criterion.dimension,
                score=0.0,
                max_score=float(criterion.max_score),
                explanation=f"Evaluation error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
            )

            # Emit update for error case
            if self._emitter:
                self._emitter.criterion_evaluated(
                    criterion_id=criterion.criterion_id,
                    criterion_name=criterion.name,
                    dimension=criterion.dimension,
                    score=0.0,
                    max_score=float(criterion.max_score),
                    evaluation_method="programmatic"
                    if criterion.evaluator_id
                    else "llm",
                )

            return result

    async def _evaluate_criterion(
        self,
        ctx: AgentBeatsEvalContext,
        criterion: EvaluationCriterion,
    ) -> CriterionResult:
        """Evaluate a single criterion.

        Dispatches to programmatic evaluator if available and registered,
        otherwise falls back to LLM-based evaluation.

        Args:
            ctx: Evaluation context.
            criterion: The criterion to evaluate.

        Returns:
            CriterionResult with scaled score.

        Raises:
            EvaluationError: If evaluation fails.
            ValueError: If criterion has no valid evaluation method.
        """
        evaluation_method: Literal["programmatic", "llm"]

        # Try registered programmatic evaluator first
        if criterion.evaluator_id and criterion.evaluator_id in self._evaluators:
            logger.debug(
                "Evaluating criterion '%s' with programmatic evaluator '%s'",
                criterion.criterion_id,
                criterion.evaluator_id,
            )
            evaluator = self._evaluators[criterion.evaluator_id]
            # Inject LLM into context for evaluators that need it
            if ctx.llm is None:
                ctx.llm = self._llm
            eval_result = await evaluator(ctx, criterion.params or {})
            evaluation_method = "programmatic"

        # Fall back to LLM-based evaluation
        elif criterion.evaluation_prompt:
            logger.debug(
                "Evaluating criterion '%s' with LLM-based evaluation",
                criterion.criterion_id,
            )
            eval_result = await self._llm_evaluate(ctx, criterion)
            evaluation_method = "llm"

        else:
            raise ValueError(
                f"Criterion '{criterion.criterion_id}' has no evaluator_id "
                "(or it's not registered) and no evaluation_prompt"
            )

        # Scale the score to match criterion.max_score
        scaled_result = self._scale_result(eval_result, criterion)

        # Convert to CriterionResult
        result = CriterionResult(
            criterion_id=criterion.criterion_id,
            name=criterion.name,
            dimension=criterion.dimension,
            score=scaled_result.score,
            max_score=scaled_result.max_score,
            explanation=scaled_result.explanation,
            details=scaled_result.details,
        )

        # Emit update
        if self._emitter:
            self._emitter.criterion_evaluated(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                dimension=criterion.dimension,
                score=result.score,
                max_score=result.max_score,
                evaluation_method=evaluation_method,
            )

        logger.debug(
            "Criterion '%s' scored %.1f/%.1f (%.0f%%)",
            criterion.criterion_id,
            result.score,
            result.max_score,
            result.percentage,
        )

        return result

    def _scale_result(
        self,
        eval_result: EvalResult,
        criterion: EvaluationCriterion,
    ) -> EvalResult:
        """Scale an evaluation result to match the criterion's max_score.

        If the evaluator's max_score differs from the criterion's max_score,
        the score is scaled proportionally.

        Args:
            eval_result: The raw evaluation result.
            criterion: The criterion definition with target max_score.

        Returns:
            EvalResult with score scaled to criterion.max_score.
        """
        target_max = float(criterion.max_score)

        # If max_scores match, no scaling needed
        if abs(eval_result.max_score - target_max) < 1e-6:
            return eval_result

        # Scale the score proportionally
        if eval_result.max_score > 0:
            scale_factor = target_max / eval_result.max_score
            scaled_score = eval_result.score * scale_factor
        else:
            # Avoid division by zero - award 0 if max_score is 0
            scaled_score = 0.0

        # Clamp to valid range
        scaled_score = max(0.0, min(scaled_score, target_max))

        return EvalResult(
            score=scaled_score,
            max_score=target_max,
            explanation=eval_result.explanation,
            details=eval_result.details,
        )

    async def _llm_evaluate(
        self,
        ctx: AgentBeatsEvalContext,
        criterion: EvaluationCriterion,
    ) -> EvalResult:
        """Evaluate a criterion using LLM.

        Constructs a prompt from the criterion's evaluation_prompt and context,
        sends it to the LLM, and parses the structured response.

        Args:
            ctx: Evaluation context.
            criterion: The criterion with evaluation_prompt.

        Returns:
            EvalResult from LLM evaluation.

        Raises:
            EvaluationError: If LLM evaluation fails.
        """
        if not criterion.evaluation_prompt:
            raise EvaluationError(
                criterion.criterion_id,
                "No evaluation_prompt provided for LLM evaluation",
            )

        # Build context section for prompt
        action_log_dicts = [
            entry.model_dump() if hasattr(entry, "model_dump") else entry
            for entry in ctx.action_log
        ]
        context_section = build_evaluation_context_section(
            action_log=action_log_dicts,
            initial_state=ctx.initial_state,
            final_state=ctx.final_state,
            user_prompt=ctx.user_prompt,
        )

        # Format the user prompt
        user_prompt = LLM_EVALUATION_USER_TEMPLATE.format(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            dimension=criterion.dimension,
            max_score=criterion.max_score,
            criterion_description=criterion.description,
            evaluation_prompt=criterion.evaluation_prompt,
            context_section=context_section,
            user_prompt=ctx.user_prompt,
        )

        # Create messages
        messages = [
            SystemMessage(content=LLM_EVALUATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Use structured output for reliable parsing
            structured_llm = self._llm.with_structured_output(LLMEvaluationResult)
            response: LLMEvaluationResult = await structured_llm.ainvoke(messages)

            # Validate score is within bounds
            score = max(0.0, min(response.score, float(criterion.max_score)))

            return EvalResult(
                score=score,
                max_score=float(criterion.max_score),
                explanation=response.explanation,
                details={
                    "strengths": response.strengths,
                    "weaknesses": response.weaknesses,
                    "raw_llm_score": response.score,
                },
            )

        except Exception as e:
            raise EvaluationError(
                criterion.criterion_id,
                f"LLM evaluation failed: {str(e)}",
                original_error=e,
            ) from e

    def aggregate_scores(
        self,
        results: list[CriterionResult],
    ) -> Scores:
        """Aggregate criterion results into dimension and overall scores.

        Groups criterion results by dimension, sums scores within each
        dimension, and computes the overall score.

        Args:
            results: List of CriterionResult from evaluate_all().

        Returns:
            Scores object with dimension scores and overall score.
        """
        # Group by dimension
        dimension_scores: dict[ScoringDimension, tuple[float, float]] = {}

        for result in results:
            dim = result.dimension
            current_score, current_max = dimension_scores.get(dim, (0.0, 0.0))
            dimension_scores[dim] = (
                current_score + result.score,
                current_max + result.max_score,
            )

        # Build DimensionScore objects
        dimension_objs: dict[str, DimensionScore] = {}
        total_score = 0.0
        total_max = 0.0

        for dim, (score, max_score) in dimension_scores.items():
            dimension_objs[dim] = DimensionScore(score=score, max_score=max_score)
            total_score += score
            total_max += max_score

        # Build overall score
        overall = OverallScore(score=total_score, max_score=total_max)

        return Scores(overall=overall, dimensions=dimension_objs)

    def get_dimensions(self) -> list[str]:
        """Get the list of unique scoring dimensions from criteria.

        Returns:
            List of dimension names that have at least one criterion.
        """
        return list(set(c.dimension for c in self._criteria))

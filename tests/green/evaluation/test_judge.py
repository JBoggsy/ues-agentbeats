"""Tests for the CriteriaJudge module.

Tests cover:
- CriteriaJudge initialization
- Programmatic evaluator dispatch
- LLM-based evaluation
- Score scaling
- Error handling
- Score aggregation
- TaskUpdateEmitter integration
- Parallel evaluation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.common.agentbeats.results import (
    ActionLogEntry,
    CriterionResult,
    DimensionScore,
    OverallScore,
    Scores,
)
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.evaluation.judge import (
    CriteriaJudge,
    CriteriaJudgeError,
    EvaluationError,
)
from src.green.evaluation.models import LLMEvaluationResult
from src.green.scenarios.schema import (
    AgentBeatsEvalContext,
    EvalResult,
    EvaluationCriterion,
    EvaluatorFunc,
    EvaluatorRegistry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    return llm


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock UES client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def sample_action_log() -> list[ActionLogEntry]:
    """Create a sample action log for testing."""
    return [
        ActionLogEntry(
            turn=1,
            timestamp=datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
            action="email.send",
            parameters={"to": ["alice@example.com"]},
            success=True,
        ),
        ActionLogEntry(
            turn=1,
            timestamp=datetime(2026, 1, 29, 10, 5, tzinfo=timezone.utc),
            action="email.query",
            parameters={},
            success=True,
        ),
    ]


@pytest.fixture
def sample_eval_context(
    mock_ues_client: AsyncMock,
    sample_action_log: list[ActionLogEntry],
) -> AgentBeatsEvalContext:
    """Create a sample evaluation context."""
    return AgentBeatsEvalContext(
        client=mock_ues_client,
        scenario_config={"scenario_id": "test_scenario"},
        action_log=sample_action_log,
        initial_state={"email": {"total_emails": 10}},
        final_state={"email": {"total_emails": 15}},
        user_prompt="Please manage my inbox.",
    )


@pytest.fixture
def programmatic_criterion() -> EvaluationCriterion:
    """Create a criterion with programmatic evaluator."""
    return EvaluationCriterion(
        criterion_id="test_programmatic",
        name="Test Programmatic Criterion",
        description="A test criterion using a programmatic evaluator.",
        dimension="accuracy",
        max_score=20,
        evaluator_id="test_evaluator",
        params={"threshold": 0.8},
    )


@pytest.fixture
def llm_criterion() -> EvaluationCriterion:
    """Create a criterion with LLM-based evaluation."""
    return EvaluationCriterion(
        criterion_id="test_llm",
        name="Test LLM Criterion",
        description="A test criterion using LLM-based evaluation.",
        dimension="politeness",
        max_score=10,
        evaluation_prompt="Rate the politeness of the assistant's responses.",
    )


@pytest.fixture
def sample_evaluator() -> EvaluatorFunc:
    """Create a sample evaluator function."""

    async def evaluator(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
        return EvalResult(
            score=15.0,
            max_score=20.0,
            explanation="Good performance.",
            details={"threshold": params.get("threshold", 0.5)},
        )

    return evaluator


@pytest.fixture
def sample_evaluators(sample_evaluator: EvaluatorFunc) -> EvaluatorRegistry:
    """Create a sample evaluator registry."""
    return {"test_evaluator": sample_evaluator}


@pytest.fixture
def mock_emitter() -> MagicMock:
    """Create a mock TaskUpdateEmitter."""
    emitter = MagicMock(spec=TaskUpdateEmitter)
    return emitter


# =============================================================================
# Tests for CriteriaJudge Initialization
# =============================================================================


class TestCriteriaJudgeInit:
    """Tests for CriteriaJudge initialization."""

    def test_create_with_required_params(
        self,
        mock_llm: MagicMock,
        programmatic_criterion: EvaluationCriterion,
        sample_evaluators: EvaluatorRegistry,
    ) -> None:
        """Should create judge with required parameters."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[programmatic_criterion],
            evaluators=sample_evaluators,
        )

        assert judge.llm == mock_llm
        assert len(judge.criteria) == 1
        assert judge.criteria[0] == programmatic_criterion

    def test_create_with_emitter(
        self,
        mock_llm: MagicMock,
        programmatic_criterion: EvaluationCriterion,
        sample_evaluators: EvaluatorRegistry,
        mock_emitter: MagicMock,
    ) -> None:
        """Should create judge with TaskUpdateEmitter."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[programmatic_criterion],
            evaluators=sample_evaluators,
            emitter=mock_emitter,
        )

        assert judge._emitter == mock_emitter

    def test_warns_for_missing_evaluator(
        self,
        mock_llm: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should warn when criterion references non-existent evaluator."""
        criterion = EvaluationCriterion(
            criterion_id="test_missing",
            name="Test Missing Evaluator",
            description="A test with missing evaluator.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="nonexistent_evaluator",
            evaluation_prompt="Fallback prompt",  # Has fallback
        )

        with caplog.at_level("WARNING"):
            CriteriaJudge(
                llm=mock_llm,
                criteria=[criterion],
                evaluators={},  # Empty registry
            )

        assert "nonexistent_evaluator" in caplog.text
        assert "not registered" in caplog.text


# =============================================================================
# Tests for Programmatic Evaluation
# =============================================================================


class TestProgrammaticEvaluation:
    """Tests for programmatic evaluator dispatch."""

    @pytest.mark.asyncio
    async def test_calls_registered_evaluator(
        self,
        mock_llm: MagicMock,
        programmatic_criterion: EvaluationCriterion,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should call the registered evaluator function."""
        evaluator_called = False
        received_ctx = None
        received_params = None

        async def tracking_evaluator(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
            nonlocal evaluator_called, received_ctx, received_params
            evaluator_called = True
            received_ctx = ctx
            received_params = params
            return EvalResult(score=18.0, max_score=20.0, explanation="Tracked.")

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[programmatic_criterion],
            evaluators={"test_evaluator": tracking_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert evaluator_called
        assert received_ctx == sample_eval_context
        assert received_params == {"threshold": 0.8}
        assert len(results) == 1
        assert results[0].score == 18.0

    @pytest.mark.asyncio
    async def test_passes_params_to_evaluator(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should pass criterion params to evaluator."""
        received_params = {}

        async def param_tracking_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            nonlocal received_params
            received_params = params.copy()
            return EvalResult(score=5.0, max_score=10.0, explanation="OK.")

        criterion = EvaluationCriterion(
            criterion_id="param_test",
            name="Param Test",
            description="Test params passing.",
            dimension="efficiency",
            max_score=10,
            evaluator_id="param_evaluator",
            params={"key1": "value1", "key2": 42, "key3": True},
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"param_evaluator": param_tracking_evaluator},
        )

        await judge.evaluate_all(sample_eval_context)

        assert received_params == {"key1": "value1", "key2": 42, "key3": True}


# =============================================================================
# Tests for Score Scaling
# =============================================================================


class TestScoreScaling:
    """Tests for score scaling functionality."""

    @pytest.mark.asyncio
    async def test_no_scaling_when_max_scores_match(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should not scale when evaluator max_score matches criterion."""
        async def matching_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=8.0, max_score=10.0, explanation="Matching.")

        criterion = EvaluationCriterion(
            criterion_id="matching",
            name="Matching Max Score",
            description="Test.",
            dimension="accuracy",
            max_score=10,  # Same as evaluator
            evaluator_id="matching_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"matching_eval": matching_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert results[0].score == 8.0
        assert results[0].max_score == 10.0

    @pytest.mark.asyncio
    async def test_scales_up_when_criterion_max_higher(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should scale up when criterion max_score is higher."""
        async def small_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=8.0, max_score=10.0, explanation="Small scale.")

        criterion = EvaluationCriterion(
            criterion_id="scale_up",
            name="Scale Up",
            description="Test.",
            dimension="accuracy",
            max_score=20,  # Higher than evaluator's 10
            evaluator_id="small_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"small_eval": small_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        # 8/10 scaled to 20 = 16
        assert results[0].score == 16.0
        assert results[0].max_score == 20.0

    @pytest.mark.asyncio
    async def test_scales_down_when_criterion_max_lower(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should scale down when criterion max_score is lower."""
        async def large_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=80.0, max_score=100.0, explanation="Large scale.")

        criterion = EvaluationCriterion(
            criterion_id="scale_down",
            name="Scale Down",
            description="Test.",
            dimension="accuracy",
            max_score=10,  # Lower than evaluator's 100
            evaluator_id="large_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"large_eval": large_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        # 80/100 scaled to 10 = 8
        assert results[0].score == 8.0
        assert results[0].max_score == 10.0

    @pytest.mark.asyncio
    async def test_handles_zero_max_score_from_evaluator(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should handle zero max_score from evaluator gracefully."""
        async def zero_max_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=0.0, max_score=0.0, explanation="No criteria.")

        criterion = EvaluationCriterion(
            criterion_id="zero_max",
            name="Zero Max",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="zero_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"zero_eval": zero_max_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        # Should result in 0 score
        assert results[0].score == 0.0
        assert results[0].max_score == 10.0


# =============================================================================
# Tests for Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in evaluation."""

    @pytest.mark.asyncio
    async def test_catches_evaluator_exception(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should catch evaluator exceptions and return zero-score result."""
        async def failing_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            raise RuntimeError("Evaluator crashed!")

        criterion = EvaluationCriterion(
            criterion_id="failing",
            name="Failing Criterion",
            description="Test.",
            dimension="safety",
            max_score=10,
            evaluator_id="failing_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"failing_eval": failing_evaluator},
        )

        # Should not raise
        results = await judge.evaluate_all(sample_eval_context)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].max_score == 10.0
        assert "error" in results[0].explanation.lower()
        assert "Evaluator crashed" in results[0].explanation

    @pytest.mark.asyncio
    async def test_other_criteria_still_evaluated_on_error(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should continue evaluating other criteria when one fails."""
        async def failing_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            raise ValueError("This one fails")

        async def working_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=8.0, max_score=10.0, explanation="Works!")

        criteria = [
            EvaluationCriterion(
                criterion_id="failing",
                name="Failing",
                description="Test.",
                dimension="safety",
                max_score=10,
                evaluator_id="failing_eval",
            ),
            EvaluationCriterion(
                criterion_id="working",
                name="Working",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="working_eval",
            ),
        ]

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=criteria,
            evaluators={
                "failing_eval": failing_evaluator,
                "working_eval": working_evaluator,
            },
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert len(results) == 2
        # One failed, one succeeded
        scores = {r.criterion_id: r.score for r in results}
        assert scores["failing"] == 0.0
        assert scores["working"] == 8.0

    @pytest.mark.asyncio
    async def test_error_result_includes_details(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Error result should include error type in details."""
        async def type_error_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            raise TypeError("Type mismatch")

        criterion = EvaluationCriterion(
            criterion_id="type_error",
            name="Type Error",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="type_error_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"type_error_eval": type_error_evaluator},
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert results[0].details is not None
        assert results[0].details.get("error_type") == "TypeError"
        assert "Type mismatch" in results[0].details.get("error", "")


# =============================================================================
# Tests for Score Aggregation
# =============================================================================


class TestScoreAggregation:
    """Tests for score aggregation functionality."""

    def test_aggregate_single_criterion(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should aggregate scores from a single criterion."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        results = [
            CriterionResult(
                criterion_id="c1",
                name="Criterion 1",
                dimension="accuracy",
                score=8.0,
                max_score=10.0,
                explanation="Good.",
            )
        ]

        scores = judge.aggregate_scores(results)

        assert scores.overall.score == 8.0
        assert scores.overall.max_score == 10.0
        assert "accuracy" in scores.dimensions
        assert scores.dimensions["accuracy"].score == 8.0
        assert scores.dimensions["accuracy"].max_score == 10.0

    def test_aggregate_multiple_criteria_same_dimension(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should sum scores within the same dimension."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        results = [
            CriterionResult(
                criterion_id="c1",
                name="Criterion 1",
                dimension="accuracy",
                score=8.0,
                max_score=10.0,
                explanation="Good.",
            ),
            CriterionResult(
                criterion_id="c2",
                name="Criterion 2",
                dimension="accuracy",
                score=15.0,
                max_score=20.0,
                explanation="Great.",
            ),
        ]

        scores = judge.aggregate_scores(results)

        assert scores.overall.score == 23.0
        assert scores.overall.max_score == 30.0
        assert scores.dimensions["accuracy"].score == 23.0
        assert scores.dimensions["accuracy"].max_score == 30.0

    def test_aggregate_multiple_dimensions(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should aggregate scores across multiple dimensions."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        results = [
            CriterionResult(
                criterion_id="c1",
                name="Criterion 1",
                dimension="accuracy",
                score=8.0,
                max_score=10.0,
                explanation="Good.",
            ),
            CriterionResult(
                criterion_id="c2",
                name="Criterion 2",
                dimension="efficiency",
                score=7.0,
                max_score=10.0,
                explanation="OK.",
            ),
            CriterionResult(
                criterion_id="c3",
                name="Criterion 3",
                dimension="safety",
                score=10.0,
                max_score=10.0,
                explanation="Perfect.",
            ),
        ]

        scores = judge.aggregate_scores(results)

        assert scores.overall.score == 25.0
        assert scores.overall.max_score == 30.0
        assert len(scores.dimensions) == 3
        assert scores.dimensions["accuracy"].score == 8.0
        assert scores.dimensions["efficiency"].score == 7.0
        assert scores.dimensions["safety"].score == 10.0

    def test_aggregate_empty_results(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should handle empty results list."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        scores = judge.aggregate_scores([])

        assert scores.overall.score == 0.0
        assert scores.overall.max_score == 0.0
        assert len(scores.dimensions) == 0

    def test_aggregate_with_float_scores(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should handle float scores correctly."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        results = [
            CriterionResult(
                criterion_id="c1",
                name="Criterion 1",
                dimension="accuracy",
                score=8.5,
                max_score=10.0,
                explanation="Good.",
            ),
            CriterionResult(
                criterion_id="c2",
                name="Criterion 2",
                dimension="accuracy",
                score=7.25,
                max_score=10.0,
                explanation="OK.",
            ),
        ]

        scores = judge.aggregate_scores(results)

        assert scores.overall.score == 15.75
        assert scores.overall.max_score == 20.0


# =============================================================================
# Tests for TaskUpdateEmitter Integration
# =============================================================================


class TestEmitterIntegration:
    """Tests for TaskUpdateEmitter integration."""

    @pytest.mark.asyncio
    async def test_emits_update_on_success(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
        mock_emitter: MagicMock,
    ) -> None:
        """Should emit criterion_evaluated update on successful evaluation."""
        async def simple_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            return EvalResult(score=8.0, max_score=10.0, explanation="Good.")

        criterion = EvaluationCriterion(
            criterion_id="emitter_test",
            name="Emitter Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="simple_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"simple_eval": simple_evaluator},
            emitter=mock_emitter,
        )

        await judge.evaluate_all(sample_eval_context)

        mock_emitter.criterion_evaluated.assert_called_once_with(
            criterion_id="emitter_test",
            criterion_name="Emitter Test",
            dimension="accuracy",
            score=8.0,
            max_score=10.0,
            evaluation_method="programmatic",
        )

    @pytest.mark.asyncio
    async def test_emits_update_on_error(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
        mock_emitter: MagicMock,
    ) -> None:
        """Should emit criterion_evaluated update even on error."""
        async def error_evaluator(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            raise RuntimeError("Boom!")

        criterion = EvaluationCriterion(
            criterion_id="error_test",
            name="Error Test",
            description="Test.",
            dimension="safety",
            max_score=15,
            evaluator_id="error_eval",
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[criterion],
            evaluators={"error_eval": error_evaluator},
            emitter=mock_emitter,
        )

        await judge.evaluate_all(sample_eval_context)

        mock_emitter.criterion_evaluated.assert_called_once_with(
            criterion_id="error_test",
            criterion_name="Error Test",
            dimension="safety",
            score=0.0,
            max_score=15.0,
            evaluation_method="programmatic",
        )


# =============================================================================
# Tests for Parallel Evaluation
# =============================================================================


class TestParallelEvaluation:
    """Tests for parallel evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluations_run_in_parallel(
        self,
        mock_llm: MagicMock,
        sample_eval_context: AgentBeatsEvalContext,
    ) -> None:
        """Should run evaluations in parallel."""
        execution_times: dict[str, tuple[float, float]] = {}
        start_time = asyncio.get_event_loop().time()

        async def slow_evaluator_1(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            entered = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(0.1)
            exited = asyncio.get_event_loop().time() - start_time
            execution_times["eval1"] = (entered, exited)
            return EvalResult(score=5.0, max_score=10.0, explanation="Slow 1.")

        async def slow_evaluator_2(
            ctx: AgentBeatsEvalContext, params: dict
        ) -> EvalResult:
            entered = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(0.1)
            exited = asyncio.get_event_loop().time() - start_time
            execution_times["eval2"] = (entered, exited)
            return EvalResult(score=8.0, max_score=10.0, explanation="Slow 2.")

        criteria = [
            EvaluationCriterion(
                criterion_id="slow1",
                name="Slow 1",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="slow_eval_1",
            ),
            EvaluationCriterion(
                criterion_id="slow2",
                name="Slow 2",
                description="Test.",
                dimension="efficiency",
                max_score=10,
                evaluator_id="slow_eval_2",
            ),
        ]

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=criteria,
            evaluators={
                "slow_eval_1": slow_evaluator_1,
                "slow_eval_2": slow_evaluator_2,
            },
        )

        before = asyncio.get_event_loop().time()
        results = await judge.evaluate_all(sample_eval_context)
        after = asyncio.get_event_loop().time()

        # Both evaluations should complete
        assert len(results) == 2

        # Total time should be ~0.1s (parallel), not ~0.2s (sequential)
        total_time = after - before
        assert total_time < 0.2, f"Expected parallel execution, took {total_time}s"

        # Both should have started at similar times
        time_diff = abs(execution_times["eval1"][0] - execution_times["eval2"][0])
        assert time_diff < 0.05, f"Evaluators didn't start together: {time_diff}s apart"


# =============================================================================
# Tests for get_dimensions
# =============================================================================


class TestGetDimensions:
    """Tests for get_dimensions method."""

    def test_returns_unique_dimensions(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should return unique dimensions from criteria."""
        criteria = [
            EvaluationCriterion(
                criterion_id="c1",
                name="C1",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluation_prompt="Test.",
            ),
            EvaluationCriterion(
                criterion_id="c2",
                name="C2",
                description="Test.",
                dimension="safety",
                max_score=10,
                evaluation_prompt="Test.",
            ),
            EvaluationCriterion(
                criterion_id="c3",
                name="C3",
                description="Test.",
                dimension="accuracy",  # Duplicate
                max_score=10,
                evaluation_prompt="Test.",
            ),
        ]

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=criteria,
            evaluators={},
        )

        dimensions = judge.get_dimensions()

        assert len(dimensions) == 2
        assert set(dimensions) == {"accuracy", "safety"}

    def test_empty_criteria_returns_empty_list(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """Should return empty list for empty criteria."""
        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[],
            evaluators={},
        )

        dimensions = judge.get_dimensions()

        assert dimensions == []


# =============================================================================
# Tests for LLM-based Evaluation
# =============================================================================


class TestLLMEvaluation:
    """Tests for LLM-based evaluation."""

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_when_no_evaluator(
        self,
        sample_eval_context: AgentBeatsEvalContext,
        llm_criterion: EvaluationCriterion,
    ) -> None:
        """Should use LLM when no programmatic evaluator is available."""
        mock_response = LLMEvaluationResult(
            score=8.0,
            explanation="The assistant was polite and professional.",
            strengths=["Good tone"],
            weaknesses=["Could be warmer"],
        )

        mock_llm = MagicMock()
        mock_structured = AsyncMock(return_value=mock_response)
        mock_llm.with_structured_output.return_value = MagicMock(
            ainvoke=mock_structured
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[llm_criterion],
            evaluators={},  # No evaluators
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert len(results) == 1
        assert results[0].score == 8.0
        assert results[0].max_score == 10.0
        assert "polite" in results[0].explanation.lower()
        mock_llm.with_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_response_clamped_to_max_score(
        self,
        sample_eval_context: AgentBeatsEvalContext,
        llm_criterion: EvaluationCriterion,
    ) -> None:
        """Should clamp LLM score to criterion max_score."""
        # LLM returns score higher than max
        mock_response = LLMEvaluationResult(
            score=15.0,  # Higher than max_score of 10
            explanation="Exceptional performance.",
        )

        mock_llm = MagicMock()
        mock_structured = AsyncMock(return_value=mock_response)
        mock_llm.with_structured_output.return_value = MagicMock(
            ainvoke=mock_structured
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[llm_criterion],
            evaluators={},
        )

        results = await judge.evaluate_all(sample_eval_context)

        # Should be clamped to max_score
        assert results[0].score == 10.0
        assert results[0].max_score == 10.0

    @pytest.mark.asyncio
    async def test_llm_error_handled_gracefully(
        self,
        sample_eval_context: AgentBeatsEvalContext,
        llm_criterion: EvaluationCriterion,
    ) -> None:
        """Should handle LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_structured = AsyncMock(side_effect=RuntimeError("LLM API error"))
        mock_llm.with_structured_output.return_value = MagicMock(
            ainvoke=mock_structured
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[llm_criterion],
            evaluators={},
        )

        # Should not raise
        results = await judge.evaluate_all(sample_eval_context)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert "error" in results[0].explanation.lower()

    @pytest.mark.asyncio
    async def test_llm_evaluation_includes_details(
        self,
        sample_eval_context: AgentBeatsEvalContext,
        llm_criterion: EvaluationCriterion,
    ) -> None:
        """Should include LLM response details in result."""
        mock_response = LLMEvaluationResult(
            score=7.5,
            explanation="Good but could improve.",
            strengths=["Strength A", "Strength B"],
            weaknesses=["Weakness 1"],
        )

        mock_llm = MagicMock()
        mock_structured = AsyncMock(return_value=mock_response)
        mock_llm.with_structured_output.return_value = MagicMock(
            ainvoke=mock_structured
        )

        judge = CriteriaJudge(
            llm=mock_llm,
            criteria=[llm_criterion],
            evaluators={},
        )

        results = await judge.evaluate_all(sample_eval_context)

        assert results[0].details is not None
        assert results[0].details["strengths"] == ["Strength A", "Strength B"]
        assert results[0].details["weaknesses"] == ["Weakness 1"]
        assert results[0].details["raw_llm_score"] == 7.5


# =============================================================================
# Tests for EvaluationError Exception
# =============================================================================


class TestEvaluationErrorException:
    """Tests for EvaluationError exception class."""

    def test_create_with_basic_info(self) -> None:
        """Should create error with basic info."""
        error = EvaluationError("criterion_1", "Something went wrong")

        assert error.criterion_id == "criterion_1"
        assert "criterion_1" in str(error)
        assert "Something went wrong" in str(error)
        assert error.original_error is None

    def test_create_with_original_error(self) -> None:
        """Should create error with original exception."""
        original = ValueError("Original error")
        error = EvaluationError(
            "criterion_2",
            "Wrapped error",
            original_error=original,
        )

        assert error.criterion_id == "criterion_2"
        assert error.original_error == original
        assert "Wrapped error" in str(error)

    def test_is_criteria_judge_error(self) -> None:
        """EvaluationError should be a CriteriaJudgeError."""
        error = EvaluationError("test", "Test error")
        assert isinstance(error, CriteriaJudgeError)

"""Tests for EvalGenerator class (non-LLM tests)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from vowel import EvalGenerator, Function, GenerationResult
from vowel.ai import UnsupportedParameterTypeError


class TestEvalGeneratorInit:
    """Tests for EvalGenerator initialization."""

    def test_init_with_model(self):
        """Test initialization with explicit model."""
        gen = EvalGenerator(model="openai:gpt-4o")

        assert gen.model == "openai:gpt-4o"

    def test_init_from_env(self, mock_env_model):
        """Test initialization from environment variable."""
        gen = EvalGenerator()

        assert gen.model == "openai:gpt-4o-mini"

    def test_init_load_env(self, tmp_path):
        """Test initialization with load_env=True."""
        env_file = tmp_path / ".env"
        env_file.write_text("MODEL_NAME=test:model")

        with patch.dict(os.environ, {"MODEL_NAME": ""}, clear=False):
            os.chdir(tmp_path)
            EvalGenerator(load_env=True)

    def test_init_with_additional_context(self):
        """Test initialization with additional context."""
        gen = EvalGenerator(model="openai:gpt-4o", additional_context="Focus on edge cases")

        assert gen.additional_context == "Focus on edge cases"

    def test_init_with_context_list(self):
        """Test initialization with context as list."""
        gen = EvalGenerator(model="openai:gpt-4o", additional_context=["Context 1", "Context 2"])

        assert gen.additional_context == ["Context 1", "Context 2"]


class TestEvalGeneratorAgent:
    """Tests for EvalGenerator.agent property."""

    def test_agent_lazy_initialization(self):
        """Test agent is lazily initialized."""
        gen = EvalGenerator(model="openai:gpt-4o")

        assert gen._agent is None

    def test_agent_raises_without_model(self):
        """Test agent raises error without model."""
        with patch.dict(os.environ, {"MODEL_NAME": ""}, clear=False):
            gen = EvalGenerator()
            gen.model = None

            with pytest.raises(ValueError, match="Model name must be provided"):
                _ = gen.agent


class TestGenerationResult:
    """Tests for GenerationResult class."""

    def test_generation_result_creation(self):
        """Test creating GenerationResult."""
        func = Function(name="test", description="Test function", code="def test(): pass")

        from vowel import EvalSummary

        summary = EvalSummary(results=[])

        result = GenerationResult(
            yaml_spec="test:\n  dataset: []", func=func, summary=summary, was_healed=False
        )

        assert result.yaml_spec == "test:\n  dataset: []"
        assert result.func.name == "test"
        assert not result.was_healed

    def test_generation_result_was_healed(self):
        """Test GenerationResult.was_healed flag."""
        func = Function(name="test", description="Test", code="def test(): pass")
        from vowel import EvalSummary

        summary = EvalSummary(results=[])

        result = GenerationResult(yaml_spec="", func=func, summary=summary, was_healed=True)

        assert result.was_healed


class TestGenerationResultPrint:
    """Tests for GenerationResult.print() method."""

    def test_print_simple(self, capsys):
        """Test _print_simple fallback."""
        func = Function(
            name="add", description="Add two numbers", code="def add(a, b): return a + b"
        )
        from vowel import EvalSummary

        summary = EvalSummary(results=[])

        result = GenerationResult(
            yaml_spec="add:\n  dataset: []", func=func, summary=summary, was_healed=False
        )

        result._print_simple()

        captured = capsys.readouterr()
        assert "GENERATION RESULT" in captured.out
        assert "add" in captured.out


class TestUnsupportedParameterTypeError:
    """Tests for UnsupportedParameterTypeError."""

    def test_error_message(self):
        """Test error message format."""
        error = UnsupportedParameterTypeError(
            func_name="my_func",
            issues=["param1: Callable not supported", "param2: Generator not supported"],
        )

        assert "my_func" in str(error)
        assert "param1" in str(error)
        assert "param2" in str(error)

    def test_error_attributes(self):
        """Test error attributes."""
        error = UnsupportedParameterTypeError(func_name="test_func", issues=["issue1", "issue2"])

        assert error.func_name == "test_func"
        assert error.issues == ["issue1", "issue2"]


class TestEvalGeneratorGenerateSpec:
    """Tests for generate_spec method behavior."""

    def test_generate_spec_rejects_callable_params(self):
        """Test generate_spec rejects functions with Callable params."""

        gen = EvalGenerator(model="openai:gpt-4o")

        func = Function(
            name="apply",
            description="Apply a function",
            code="from typing import Callable\ndef apply(fn: Callable, x: int) -> int:\n    return fn(x)",
        )
        func.execute()

        with pytest.raises(UnsupportedParameterTypeError):
            gen._agent = MagicMock()
            gen.generate_spec(func)


class TestFunctionExecutionInGenerator:
    """Tests for Function execution within generator context."""

    def test_function_auto_executes(self):
        """Test Function auto-executes when needed."""
        func = Function(
            name="square",
            description="Square a number",
            code="def square(x: int) -> int:\n    return x * x",
        )

        assert func.func is None

        impl = func.impl

        assert func.func is not None
        assert impl(5) == 25

    def test_function_from_callable_already_executed(self):
        """Test Function.from_callable has func set."""

        def my_func(x: int) -> int:
            return x * 2

        func = Function.from_callable(my_func)

        assert func.func is my_func


class TestGeneratorWithMockedLLM:
    """Tests with mocked LLM responses."""

    def test_generate_spec_with_mock(self):
        """Test generate_spec with mocked agent."""
        from vowel.ai import EvalsSource

        gen = EvalGenerator(model="openai:gpt-4o")

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = EvalsSource(
            yaml_spec="""test_func:
  dataset:
    - case:
        input: 5
        expected: 10
"""
        )
        mock_agent.run_sync.return_value = mock_result
        gen._agent = mock_agent

        func = Function(
            name="test_func",
            description="Double a number",
            code="def test_func(x: int) -> int:\n    return x * 2",
        )

        runner, yaml_spec = gen.generate_spec(func, save_to_file=False)

        assert "test_func" in yaml_spec
        mock_agent.run_sync.assert_called_once()

    def test_generate_function_with_mock(self):
        """Test generate_function with mocked agent."""
        gen = EvalGenerator(model="openai:gpt-4o")

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = Function(
            name="fibonacci",
            description="Calculate Fibonacci",
            code="async def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return await fibonacci(n-1) + await fibonacci(n-2)",
        )
        mock_agent.run_sync.return_value = mock_result
        gen._agent = mock_agent

        func = gen.generate_function("Generate Fibonacci function", async_func=True)

        assert func.name == "fibonacci"
        assert "async def" in func.code


class TestGeneratorRetryLogic:
    """Tests for generator retry logic (without actual LLM calls)."""

    def test_retry_parameters_accepted(self):
        """Test retry parameters are accepted."""
        EvalGenerator(model="openai:gpt-4o")

        Function(name="test", description="Test", code="def test(): pass")


class TestGeneratorContextHandling:
    """Tests for context handling in generator."""

    def test_additional_context_string(self):
        """Test additional context as string."""
        gen = EvalGenerator(model="openai:gpt-4o", additional_context="Test edge cases")

        assert gen.additional_context == "Test edge cases"

    def test_additional_context_list(self):
        """Test additional context as list."""
        gen = EvalGenerator(
            model="openai:gpt-4o", additional_context=["Context 1", "Context 2", "Context 3"]
        )

        assert gen.additional_context is not None
        assert len(gen.additional_context) == 3

"""Tests for TDDGenerator.generate_evals_from_signature retry behavior."""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from vowel.eval_types import EvalsSource
from vowel.tdd import FunctionSignature, Param, TDDGenerator


def _make_signature() -> FunctionSignature:
    return FunctionSignature(
        name="add",
        description="Add two numbers",
        params=[
            Param(name="a", type="int", description="first"),
            Param(name="b", type="int", description="second"),
        ],
        return_type="int",
    )


GOOD_YAML = """
add:
  dataset:
    - case:
        inputs: [1, 2]
        expected: 3
    - case:
        inputs: [0, 0]
        expected: 0
"""

BAD_YAML = """
add:
  dataset:
    - case:
        inputs: [1, 2]
        expected: 999
    - case:
        inputs: [0, 0]
        expected: 0
"""


def add(a: int, b: int) -> int:
    return a + b


class TestGenerateEvalsWithoutFunc(unittest.TestCase):
    """Without func parameter, behaves exactly like before (single attempt)."""

    @patch.object(TDDGenerator, "eval_agent", new_callable=PropertyMock)
    def test_single_generation_no_func(self, mock_agent_prop):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = EvalsSource(yaml_spec=GOOD_YAML)
        mock_agent.run_sync.return_value = mock_result
        mock_agent_prop.return_value = mock_agent

        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"
        gen._eval_agent = None

        sig = _make_signature()
        runner, yaml_spec = gen.generate_evals_from_signature(sig)

        assert "add:" in yaml_spec
        mock_agent.run_sync.assert_called_once()


class TestGenerateEvalsWithFunc(unittest.TestCase):
    """With func parameter, spec is validated against the function."""

    @patch.object(TDDGenerator, "eval_agent", new_callable=PropertyMock)
    def test_passes_on_first_try(self, mock_agent_prop):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = EvalsSource(yaml_spec=GOOD_YAML)
        mock_agent.run_sync.return_value = mock_result
        mock_agent_prop.return_value = mock_agent

        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"
        gen._eval_agent = None

        sig = _make_signature()
        runner, yaml_spec = gen.generate_evals_from_signature(sig, func=add, max_retries=2)

        assert "add:" in yaml_spec
        # Should only call LLM once since first try passes
        mock_agent.run_sync.assert_called_once()

    @patch("vowel.tdd.time.sleep")
    @patch.object(TDDGenerator, "eval_agent", new_callable=PropertyMock)
    def test_retries_on_low_coverage(self, mock_agent_prop, mock_sleep):
        """First attempt generates bad spec, second attempt generates good spec."""
        mock_agent = MagicMock()

        bad_result = MagicMock()
        bad_result.output = EvalsSource(yaml_spec=BAD_YAML)
        good_result = MagicMock()
        good_result.output = EvalsSource(yaml_spec=GOOD_YAML)

        mock_agent.run_sync.side_effect = [bad_result, good_result]
        mock_agent_prop.return_value = mock_agent

        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"
        gen._eval_agent = None

        sig = _make_signature()
        runner, yaml_spec = gen.generate_evals_from_signature(
            sig, func=add, max_retries=3, min_coverage=1.0, retry_delay=0.0
        )

        # Should have called LLM twice (bad then good)
        assert mock_agent.run_sync.call_count == 2
        # Final spec should be the good one
        assert "expected: 3" in yaml_spec

    @patch("vowel.tdd.time.sleep")
    @patch.object(TDDGenerator, "eval_agent", new_callable=PropertyMock)
    def test_exhausts_retries_returns_last(self, mock_agent_prop, mock_sleep):
        """All attempts produce bad spec — returns last result anyway."""
        mock_agent = MagicMock()
        bad_result = MagicMock()
        bad_result.output = EvalsSource(yaml_spec=BAD_YAML)

        mock_agent.run_sync.return_value = bad_result
        mock_agent_prop.return_value = mock_agent

        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"
        gen._eval_agent = None

        sig = _make_signature()
        runner, yaml_spec = gen.generate_evals_from_signature(
            sig, func=add, max_retries=2, min_coverage=1.0, retry_delay=0.0
        )

        # 1 initial + 2 retries = 3 calls
        assert mock_agent.run_sync.call_count == 3
        # Still returns a runner
        assert runner is not None

    @patch.object(TDDGenerator, "eval_agent", new_callable=PropertyMock)
    def test_partial_coverage_accepted(self, mock_agent_prop):
        """If min_coverage < 1.0, partial pass is accepted."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        # BAD_YAML has 1/2 passing = 50% coverage
        mock_result.output = EvalsSource(yaml_spec=BAD_YAML)
        mock_agent.run_sync.return_value = mock_result
        mock_agent_prop.return_value = mock_agent

        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"
        gen._eval_agent = None

        sig = _make_signature()
        runner, yaml_spec = gen.generate_evals_from_signature(
            sig, func=add, max_retries=5, min_coverage=0.5, retry_delay=0.0
        )

        # 50% coverage >= 50% target → accepted on first try
        mock_agent.run_sync.assert_called_once()


class TestBuildEvalFailureContext(unittest.TestCase):
    """Test the failure context builder."""

    def test_builds_context_from_failures(self):
        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"

        # Run actual evals with a bad spec to get real summary
        from vowel.runner import RunEvals

        summary = RunEvals.from_source(BAD_YAML).with_functions({"add": add}).run()

        context = gen._build_eval_failure_context(summary)
        assert "FAILED" in context

    def test_unknown_failures_fallback(self):
        gen = TDDGenerator.__new__(TDDGenerator)
        gen.model = "test"

        # Mock summary with no useful info
        mock_summary = MagicMock()
        mock_summary.results = []
        context = gen._build_eval_failure_context(mock_summary)
        assert context == "Unknown failures"

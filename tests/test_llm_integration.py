"""LLM-based integration tests with cassette caching.

These tests use real LLM calls but cache responses for reproducibility.
Run with --update-cassettes to refresh cached responses.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import dotenv
import pytest

dotenv.load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL_NAME", "openrouter:google/gemini-3-flash-preview")

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="No API key available for LLM tests (need OPENROUTER_API_KEY or OPENAI_API_KEY)",
)


class CassetteRecorder:
    """Record and playback LLM responses for deterministic testing."""

    def __init__(self, cassettes_dir: Path, test_name: str):
        self.cassettes_dir = cassettes_dir
        self.test_name = test_name
        self.cassette_file = cassettes_dir / f"{test_name}.json"
        self.recordings: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load existing cassette if available."""
        if self.cassette_file.exists():
            with open(self.cassette_file) as f:
                self.recordings = json.load(f)

    def _save(self):
        """Save cassette to file."""
        self.cassettes_dir.mkdir(exist_ok=True)
        with open(self.cassette_file, "w") as f:
            json.dump(self.recordings, f, indent=2, default=str)

    def _make_key(self, prompt: str, model: str) -> str:
        """Create a cache key from prompt and model."""
        content = f"{model}:{prompt[:200]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def has_recording(self, prompt: str, model: str) -> bool:
        """Check if we have a cached response."""
        key = self._make_key(prompt, model)
        return key in self.recordings

    def get_recording(self, prompt: str, model: str) -> dict | None:
        """Get cached response if available."""
        key = self._make_key(prompt, model)
        return self.recordings.get(key)

    def record(self, prompt: str, model: str, response: dict):
        """Record a response."""
        key = self._make_key(prompt, model)
        self.recordings[key] = {
            "prompt_preview": prompt[:300],
            "model": model,
            "response": response,
        }
        self._save()


@pytest.fixture
def cassette_recorder(cassettes_dir: Path, request) -> CassetteRecorder:
    """Provide cassette recorder for the test."""
    test_name = request.node.name
    return CassetteRecorder(cassettes_dir, test_name)


class TestLLMEvalGeneration:
    """Tests for LLM-based eval generation with cassettes."""

    def test_generate_spec_for_simple_function(self, cassettes_dir: Path):
        """Test generating eval spec for a simple add function."""
        from vowel import EvalGenerator, Function

        cassette = CassetteRecorder(cassettes_dir, "test_generate_spec_simple")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        func = Function(
            name="add_numbers",
            description="Add two integers and return the sum",
            code="def add_numbers(a: int, b: int) -> int:\n    return a + b",
        )

        if cassette.has_recording("generate_spec", model):
            cached = cassette.get_recording("generate_spec", model)
            assert cached is not None
            yaml_spec = cached["response"]["yaml_spec"]

            from vowel import RunEvals

            runner = RunEvals.from_source(yaml_spec).with_functions(
                {"add_numbers": lambda a, b: a + b}
            )
            summary = runner.run()

            assert summary.total_count >= 1
            assert "add_numbers" in yaml_spec
        else:
            runner, yaml_spec = gen.generate_spec(func, save_to_file=False)

            cassette.record(
                "generate_spec", model, {"yaml_spec": yaml_spec, "func_name": func.name}
            )

            summary = runner.with_functions({"add_numbers": lambda a, b: a + b}).run()

            assert summary.total_count >= 1
            assert "add_numbers" in yaml_spec

    def test_generate_spec_for_string_function(self, cassettes_dir: Path):
        """Test generating eval spec for a string processing function."""
        from vowel import EvalGenerator, Function

        cassette = CassetteRecorder(cassettes_dir, "test_generate_spec_string")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        func = Function(
            name="reverse_string",
            description="Reverse a string and return the result",
            code="def reverse_string(s: str) -> str:\n    return s[::-1]",
        )

        if cassette.has_recording("generate_spec", model):
            cached = cassette.get_recording("generate_spec", model)
            assert cached is not None
            yaml_spec = cached["response"]["yaml_spec"]

            from vowel import RunEvals

            runner = RunEvals.from_source(yaml_spec).with_functions(
                {"reverse_string": lambda s: s[::-1]}
            )
            runner.run()

            assert "reverse_string" in yaml_spec
        else:
            runner, yaml_spec = gen.generate_spec(func, save_to_file=False)

            cassette.record(
                "generate_spec", model, {"yaml_spec": yaml_spec, "func_name": func.name}
            )

            runner.with_functions({"reverse_string": lambda s: s[::-1]}).run()

            assert "reverse_string" in yaml_spec


class TestLLMFunctionGeneration:
    """Tests for LLM-based function generation with cassettes."""

    def test_generate_factorial_function(self, cassettes_dir: Path):
        """Test generating a factorial function."""
        from vowel import EvalGenerator

        cassette = CassetteRecorder(cassettes_dir, "test_generate_factorial")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        prompt = "Generate a function that calculates the factorial of a non-negative integer"

        if cassette.has_recording("generate_function", model):
            cached = cassette.get_recording("generate_function", model)
            assert cached is not None

            from vowel import Function

            func = Function(
                name=cached["response"]["name"],
                description=cached["response"]["description"],
                code=cached["response"]["code"],
            )
            func.execute()

            assert func(0) == 1
            assert func(5) == 120
        else:
            func = gen.generate_function(prompt, async_func=False)

            cassette.record(
                "generate_function",
                model,
                {"name": func.name, "description": func.description, "code": func.code},
            )

            func.execute()

            assert func(0) == 1
            assert func(1) == 1
            assert func(5) == 120

    def test_generate_palindrome_checker(self, cassettes_dir: Path):
        """Test generating a palindrome checker function."""
        from vowel import EvalGenerator

        cassette = CassetteRecorder(cassettes_dir, "test_generate_palindrome")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        prompt = "Generate a function that checks if a string is a palindrome (case-insensitive, ignoring spaces)"

        if cassette.has_recording("generate_function", model):
            cached = cassette.get_recording("generate_function", model)
            assert cached is not None

            from vowel import Function

            func = Function(
                name=cached["response"]["name"],
                description=cached["response"]["description"],
                code=cached["response"]["code"],
            )
            func.execute()

            assert func("racecar")
            assert not func("hello")
        else:
            func = gen.generate_function(prompt, async_func=False)

            cassette.record(
                "generate_function",
                model,
                {"name": func.name, "description": func.description, "code": func.code},
            )

            func.execute()

            assert func("racecar")
            assert func("A man a plan a canal Panama".replace(" ", "").lower()) or func(
                "A man a plan a canal Panama"
            )
            assert not func("hello")


class TestLLMGenerateAndRun:
    """Tests for complete generate_and_run flow with cassettes."""

    def test_generate_and_run_complete_flow(self, cassettes_dir: Path):
        """Test complete generate_and_run flow."""
        from vowel import EvalGenerator, Function, GenerationResult

        cassette = CassetteRecorder(cassettes_dir, "test_generate_and_run")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        func = Function(
            name="double",
            description="Double the input number",
            code="def double(x: int) -> int:\n    return x * 2",
        )

        def double_func(x: int) -> int:
            if not isinstance(x, int):
                raise TypeError(f"Expected int, got {type(x).__name__}")
            return x * 2

        if cassette.has_recording("generate_and_run", model):
            cached = cassette.get_recording("generate_and_run", model)
            assert cached is not None

            from vowel import RunEvals

            runner = RunEvals.from_source(cached["response"]["yaml_spec"]).with_functions(
                {"double": double_func}
            )
            summary = runner.run()

            assert summary.total_count >= 1
        else:
            result = gen.generate_and_run(func, auto_retry=False, min_coverage=0.8)

            cassette.record(
                "generate_and_run",
                model,
                {
                    "yaml_spec": result.yaml_spec,
                    "was_healed": result.was_healed,
                    "coverage": result.summary.coverage,
                },
            )

            assert isinstance(result, GenerationResult)
            assert result.yaml_spec is not None
            assert "double" in result.yaml_spec


class TestLLMEdgeCases:
    """Tests for edge cases in LLM integration."""

    def test_generate_spec_complex_types(self, cassettes_dir: Path):
        """Test generating spec for function with complex types."""
        from vowel import EvalGenerator, Function

        cassette = CassetteRecorder(cassettes_dir, "test_complex_types")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        func = Function(
            name="sum_list",
            description="Sum all numbers in a list",
            code="def sum_list(numbers: list[int]) -> int:\n    return sum(numbers)",
        )

        if cassette.has_recording("generate_spec", model):
            cached = cassette.get_recording("generate_spec", model)
            assert cached is not None
            yaml_spec = cached["response"]["yaml_spec"]

            assert "sum_list" in yaml_spec
        else:
            runner, yaml_spec = gen.generate_spec(func, save_to_file=False)

            cassette.record("generate_spec", model, {"yaml_spec": yaml_spec})

            assert "sum_list" in yaml_spec

    def test_generate_spec_with_context(self, cassettes_dir: Path):
        """Test generating spec with additional context."""
        from vowel import EvalGenerator, Function

        cassette = CassetteRecorder(cassettes_dir, "test_with_context")

        model = DEFAULT_MODEL
        gen = EvalGenerator(model=model)

        func = Function(
            name="validate_age",
            description="Validate that age is between 0 and 150",
            code="""def validate_age(age: int) -> bool:
    return 0 <= age <= 150""",
        )

        context = "Focus on edge cases: 0, 150, negative numbers, very large numbers"

        if cassette.has_recording("generate_spec_context", model):
            cached = cassette.get_recording("generate_spec_context", model)
            assert cached is not None
            yaml_spec = cached["response"]["yaml_spec"]

            assert "validate_age" in yaml_spec
        else:
            runner, yaml_spec = gen.generate_spec(
                func, additional_context=context, save_to_file=False
            )

            cassette.record("generate_spec_context", model, {"yaml_spec": yaml_spec})

            assert "validate_age" in yaml_spec


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: mark test as requiring LLM API calls")

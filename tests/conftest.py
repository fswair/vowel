"""Pytest configuration and shared fixtures for vowel tests."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def tests_dir() -> Path:
    """Return the tests directory path."""
    return Path(__file__).parent


@pytest.fixture
def cassettes_dir(tests_dir: Path) -> Path:
    """Return the cassettes directory for LLM response caching."""
    cassette_path = tests_dir / "cassettes"
    cassette_path.mkdir(exist_ok=True)
    return cassette_path


@pytest.fixture
def fixtures_dir(tests_dir: Path) -> Path:
    """Return the fixtures directory path."""
    fixture_path = tests_dir / "fixtures"
    fixture_path.mkdir(exist_ok=True)
    return fixture_path


# ============================================================================
# Sample Functions for Testing
# ============================================================================


@pytest.fixture
def sample_add_function():
    """Sample add function for testing."""

    def add(a: int, b: int) -> int:
        return a + b

    return add


@pytest.fixture
def sample_divide_function():
    """Sample divide function that can raise ZeroDivisionError."""

    def divide(a: float, b: float) -> float:
        return a / b

    return divide


@pytest.fixture
def sample_is_even_function():
    """Sample is_even function for testing."""

    def is_even(n: int) -> bool:
        return n % 2 == 0

    return is_even


@pytest.fixture
def sample_async_function():
    """Sample async function for testing."""

    async def async_double(x: int) -> int:
        return x * 2

    return async_double


@pytest.fixture
def sample_buggy_function():
    """Sample buggy function for healing tests."""

    def is_positive(n: int) -> bool:
        # Bug: should be n > 0, not n >= 0
        return n >= 0

    return is_positive


# ============================================================================
# YAML Fixtures
# ============================================================================


@pytest.fixture
def simple_yaml_spec() -> str:
    """Simple YAML spec for basic testing."""
    return """
add:
  dataset:
    - case:
        inputs: { a: 1, b: 2 }
        expected: 3
    - case:
        inputs: { a: -5, b: 5 }
        expected: 0
"""


@pytest.fixture
def yaml_with_evaluators() -> str:
    """YAML spec with various evaluators."""
    return """
is_even:
  evals:
    Assertion:
      assertion: "output == (input % 2 == 0)"
  dataset:
    - case:
        input: 4
    - case:
        input: 7
    - case:
        input: 0
"""


@pytest.fixture
def yaml_with_type_check() -> str:
    """YAML spec with type checking."""
    return """
divide:
  evals:
    Type:
      type: "float"
      strict: true
  dataset:
    - case:
        inputs: { a: 10, b: 2 }
        expected: 5.0
    - case:
        inputs: { a: 7, b: 2 }
        expected: 3.5
"""


@pytest.fixture
def yaml_with_raises() -> str:
    """YAML spec with exception testing."""
    return """
divide:
  dataset:
    - case:
        inputs: { a: 10, b: 2 }
        expected: 5.0
    - case:
        inputs: { a: 1, b: 0 }
        raises: ZeroDivisionError
"""


@pytest.fixture
def yaml_with_duration() -> str:
    """YAML spec with duration constraint."""
    return """
fast_func:
  evals:
    Duration:
      duration: 1.0
  dataset:
    - case:
        input: 100
        expected: 200
"""


@pytest.fixture
def yaml_with_pattern() -> str:
    """YAML spec with pattern matching."""
    return """
format_email:
  evals:
    Pattern:
      pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
  dataset:
    - case:
        input: "test"
"""


@pytest.fixture
def yaml_with_contains_input() -> str:
    """YAML spec with contains input check."""
    return """
echo:
  evals:
    ContainsInput:
      case_sensitive: false
      as_strings: true
  dataset:
    - case:
        input: "Hello"
"""


# ============================================================================
# LLM Cassette Helper
# ============================================================================


class LLMCassette:
    """Helper class for caching LLM responses."""

    def __init__(self, cassettes_dir: Path):
        self.cassettes_dir = cassettes_dir

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a key."""
        return self.cassettes_dir / f"{cache_key}.json"

    def get(self, prompt: str, model: str) -> dict | None:
        """Get cached response if available."""
        cache_key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def save(self, prompt: str, model: str, response: dict) -> None:
        """Save response to cache."""
        cache_key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(cache_key)

        with open(cache_path, "w") as f:
            json.dump(
                {
                    "prompt": prompt[:500],  # Truncate for readability
                    "model": model,
                    "response": response,
                },
                f,
                indent=2,
            )

    def has(self, prompt: str, model: str) -> bool:
        """Check if response is cached."""
        cache_key = self._get_cache_key(prompt, model)
        return self._get_cache_path(cache_key).exists()


@pytest.fixture
def llm_cassette(cassettes_dir: Path) -> LLMCassette:
    """Provide LLM cassette helper for caching responses."""
    return LLMCassette(cassettes_dir)


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def mock_env_model():
    """Mock environment with MODEL_NAME set."""
    with patch.dict(os.environ, {"MODEL_NAME": "openai:gpt-4o-mini"}):
        yield


@pytest.fixture
def temp_yaml_file(tmp_path: Path, simple_yaml_spec: str) -> Path:
    """Create a temporary YAML file for testing."""
    yaml_file = tmp_path / "test_evals.yml"
    yaml_file.write_text(simple_yaml_spec)
    return yaml_file

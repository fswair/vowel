"""LLM Judge evaluator tests with cassette caching.

These tests specifically test the LLMJudge evaluator functionality.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import dotenv
import pytest

dotenv.load_dotenv()

DEFAULT_JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL", os.getenv("MODEL_NAME", "openrouter:google/gemini-3-flash-preview")
)

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="No API key available for LLM Judge tests (need OPENROUTER_API_KEY or OPENAI_API_KEY)",
)


class LLMJudgeCassette:
    """Cassette recorder specifically for LLM Judge tests."""

    def __init__(self, cassettes_dir: Path, test_name: str):
        self.cassettes_dir = cassettes_dir
        self.test_name = test_name
        self.cassette_file = cassettes_dir / f"llm_judge_{test_name}.json"
        self.recordings: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.cassette_file.exists():
            with open(self.cassette_file) as f:
                self.recordings = json.load(f)

    def _save(self):
        self.cassettes_dir.mkdir(exist_ok=True)
        with open(self.cassette_file, "w") as f:
            json.dump(self.recordings, f, indent=2, default=str)

    def _make_key(self, rubric: str, input_val: Any) -> str:
        content = f"{rubric}:{str(input_val)[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def has_recording(self, rubric: str, input_val: Any) -> bool:
        key = self._make_key(rubric, input_val)
        return key in self.recordings

    def get_recording(self, rubric: str, input_val: Any) -> dict | None:
        key = self._make_key(rubric, input_val)
        return self.recordings.get(key)

    def record(self, rubric: str, input_val: Any, result: dict):
        key = self._make_key(rubric, input_val)
        self.recordings[key] = {
            "rubric": rubric,
            "input_preview": str(input_val)[:200],
            "result": result,
        }
        self._save()


class TestLLMJudgeBasic:
    """Basic LLM Judge evaluator tests."""

    def test_llm_judge_simple_rubric(self, cassettes_dir: Path):
        """Test LLM Judge with simple rubric."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "simple_rubric")

        spec = {
            "greet": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Output should be a polite greeting",
                        "include": ["input", "output"],
                    }
                },
                "dataset": [
                    {"case": {"input": "World"}},
                ],
            }
        }

        def greet(name: str) -> str:
            return f"Hello, {name}! Nice to meet you."

        summary = RunEvals.from_dict(spec).with_functions({"greet": greet}).run()

        cassette.record(
            "Output should be a polite greeting",
            "World",
            {"passed": summary.all_passed, "coverage": summary.coverage},
        )

        assert summary.coverage >= 0.5

    def test_llm_judge_quality_check(self, cassettes_dir: Path):
        """Test LLM Judge for quality checking."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "quality_check")

        spec = {
            "summarize": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Summary should capture the main idea in fewer words than input",
                        "include": ["input", "output"],
                    }
                },
                "dataset": [
                    {
                        "case": {
                            "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
                        }
                    },
                ],
            }
        }

        def summarize(text: str) -> str:
            return text.split(".")[0] + "."

        summary = RunEvals.from_dict(spec).with_functions({"summarize": summarize}).run()

        cassette.record(
            "Summary should capture main idea",
            "The quick brown fox...",
            {"passed": summary.all_passed, "coverage": summary.coverage},
        )

        assert summary.total_count == 1


class TestLLMJudgeWithConfig:
    """Tests for LLM Judge with custom configuration."""

    def test_llm_judge_custom_model(self, cassettes_dir: Path):
        """Test LLM Judge with custom model config."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "custom_model")

        model = DEFAULT_JUDGE_MODEL

        spec = {
            "format_name": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Output should be properly capitalized (first letter uppercase)",
                        "include": ["input", "output"],
                        "config": {"model": model, "temperature": 0.0},
                    }
                },
                "dataset": [
                    {"case": {"input": "john doe"}},
                ],
            }
        }

        def format_name(name: str) -> str:
            return name.title()

        summary = RunEvals.from_dict(spec).with_functions({"format_name": format_name}).run()

        cassette.record(
            "properly capitalized", "john doe", {"passed": summary.all_passed, "model": model}
        )

        assert summary.total_count == 1

    def test_llm_judge_include_expected(self, cassettes_dir: Path):
        """Test LLM Judge including expected output in evaluation."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "include_expected")

        spec = {
            "translate": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Translation should convey the same meaning as the expected output",
                        "include": ["input", "output", "expected_output"],
                    }
                },
                "dataset": [
                    {"case": {"input": "Hello", "expected": "Hola"}},
                ],
            }
        }

        def translate(text: str) -> str:
            translations = {"Hello": "Hola", "Goodbye": "Adiós"}
            return translations.get(text, text)

        summary = RunEvals.from_dict(spec).with_functions({"translate": translate}).run()

        cassette.record("same meaning as expected", "Hello -> Hola", {"passed": summary.all_passed})

        assert summary.total_count == 1


class TestLLMJudgeEdgeCases:
    """Edge case tests for LLM Judge."""

    def test_llm_judge_empty_output(self, cassettes_dir: Path):
        """Test LLM Judge with empty output."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "empty_output")

        spec = {
            "maybe_empty": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Output should not be empty if input is non-empty",
                        "include": ["input", "output"],
                    }
                },
                "dataset": [
                    {"case": {"input": "test"}},
                ],
            }
        }

        def maybe_empty(s: str) -> str:
            return ""

        summary = RunEvals.from_dict(spec).with_functions({"maybe_empty": maybe_empty}).run()

        cassette.record("not empty", "test", {"passed": summary.all_passed})

        assert summary.total_count == 1

    def test_llm_judge_long_input(self, cassettes_dir: Path):
        """Test LLM Judge with longer input."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "long_input")

        long_text = "This is a test. " * 20

        spec = {
            "count_words": {
                "evals": {
                    "LLMJudge": {
                        "rubric": "Output should be a reasonable word count for the input text",
                        "include": ["input", "output"],
                    }
                },
                "dataset": [
                    {"case": {"input": long_text}},
                ],
            }
        }

        def count_words(text: str) -> int:
            return len(text.split())

        summary = RunEvals.from_dict(spec).with_functions({"count_words": count_words}).run()

        cassette.record("reasonable word count", long_text[:100], {"passed": summary.all_passed})

        assert summary.total_count == 1


class TestLLMJudgeCombined:
    """Tests combining LLM Judge with other evaluators."""

    def test_llm_judge_with_type_check(self, cassettes_dir: Path):
        """Test LLM Judge combined with type checking."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "combined_type")

        spec = {
            "analyze": {
                "evals": {
                    "Type": {"type": "str"},
                    "LLMJudge": {
                        "rubric": "Analysis should mention the number of items",
                        "include": ["input", "output"],
                    },
                },
                "dataset": [
                    {"case": {"input": [1, 2, 3, 4, 5]}},
                ],
            }
        }

        def analyze(items: list) -> str:
            return f"The list contains {len(items)} items."

        summary = RunEvals.from_dict(spec).with_functions({"analyze": analyze}).run()

        cassette.record("mention number of items", [1, 2, 3, 4, 5], {"passed": summary.all_passed})

        assert summary.total_count == 1

    def test_llm_judge_with_assertion(self, cassettes_dir: Path):
        """Test LLM Judge combined with assertion."""
        from vowel import RunEvals

        cassette = LLMJudgeCassette(cassettes_dir, "combined_assertion")

        spec = {
            "format_price": {
                "evals": {
                    "Assertion": {"assertion": "'$' in output"},
                    "LLMJudge": {
                        "rubric": "Price should be formatted with dollar sign and two decimal places",
                        "include": ["input", "output"],
                    },
                },
                "dataset": [
                    {"case": {"input": 19.99}},
                ],
            }
        }

        def format_price(amount: float) -> str:
            return f"${amount:.2f}"

        summary = RunEvals.from_dict(spec).with_functions({"format_price": format_price}).run()

        cassette.record("dollar sign and decimals", 19.99, {"passed": summary.all_passed})

        assert summary.total_count == 1

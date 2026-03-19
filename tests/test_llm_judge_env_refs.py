"""Tests for environment variable references in LLM Judge configuration."""

import pytest

from vowel.evals import create_llm_judge


def test_create_llm_judge_resolves_rubric_and_model_env_refs(monkeypatch):
    """Rubric and model support $ENV_VAR style references."""
    monkeypatch.setenv("TEST_JUDGE_MODEL", "openrouter:google/gemini-2.5-flash")
    monkeypatch.setenv("_TEST_JUDGE_RUBRIC", "Output should be concise and accurate")

    judge = create_llm_judge(
        rubric="$_TEST_JUDGE_RUBRIC",
        include=["input"],
        config={"model": "$TEST_JUDGE_MODEL", "temperature": 0.0},
    )

    assert judge.model == "openrouter:google/gemini-2.5-flash"
    assert judge.rubric == "Output should be concise and accurate"


def test_create_llm_judge_raises_when_rubric_env_ref_missing(monkeypatch):
    """Missing rubric env var should raise a clear error."""
    monkeypatch.setenv("TEST_JUDGE_MODEL", "openrouter:google/gemini-2.5-flash")
    monkeypatch.delenv("_MISSING_RUBRIC", raising=False)

    with pytest.raises(ValueError, match="_MISSING_RUBRIC"):
        create_llm_judge(
            rubric="$_MISSING_RUBRIC",
            include=["input"],
            config={"model": "$TEST_JUDGE_MODEL"},
        )

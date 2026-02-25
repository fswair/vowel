"""
Example 02: Evaluators
========================

Demonstrates all built-in evaluators:
- Type       → validate return types (with optional strict mode)
- Assertion  → custom Python expressions
- Duration   → performance constraints
- Pattern    → regex matching on output
- ContainsInput → verify output contains input
- Raises     → exception testing

Run: python -m examples.evaluators.run
"""

from vowel import run_evals

from .functions import (
    calculate_bmi,
    calculate_discount,
    classify_age_group,
    extract_hashtags,
    fibonacci,
    format_phone,
    validate_email,
)


def main():
    print("=" * 60)
    print("Example: Evaluators")
    print("=" * 60)

    summary = run_evals(
        "evaluators/evals.yml",
        functions={
            "validate_email": validate_email,
            "calculate_discount": calculate_discount,
            "format_phone": format_phone,
            "fibonacci": fibonacci,
            "extract_hashtags": extract_hashtags,
            "classify_age_group": classify_age_group,
            "calculate_bmi": calculate_bmi,
        },
    )

    summary.print(include_reasons=True)


if __name__ == "__main__":
    main()

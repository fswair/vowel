"""
Example 01: Basic Usage
========================

Demonstrates the fundamentals of vowel:
- Running evaluations from a YAML file
- Providing custom functions
- Using built-in and stdlib functions
- Reading the EvalSummary result

Run: python -m examples.basic_usage.run
"""

from vowel import run_evals

from .functions import add, factorial, greet, is_even, multiply


def main():
    print("=" * 60)
    print("Example: Basic Usage")
    print("=" * 60)

    # --- Run from YAML file with custom functions ---
    print("\n▸ Running evals from YAML file...")

    summary = run_evals(
        "basic_usage/evals.yml",
        functions={
            "add": add,
            "multiply": multiply,
            "greet": greet,
            "factorial": factorial,
            "is_even": is_even,
        },
    )

    summary.print()

    # --- Run from a Python dict ---
    print("\n▸ Running evals from a Python dict...")

    spec = {
        "add": {
            "dataset": [
                {"case": {"inputs": {"x": 10, "y": 20}, "expected": 30}},
                {"case": {"inputs": {"x": -1, "y": 1}, "expected": 0}},
            ]
        }
    }

    summary = run_evals(spec, functions={"add": add})
    summary.print()

    # --- Inspect the summary ---
    print("\n▸ Summary attributes:")
    print(f"  all_passed:    {summary.all_passed}")
    print(f"  total_count:   {summary.total_count}")
    print(f"  success_count: {summary.success_count}")
    print(f"  coverage:      {summary.coverage:.0%}")


if __name__ == "__main__":
    main()

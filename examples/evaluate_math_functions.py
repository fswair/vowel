"""
Example: Custom math functions with evaluation specs.

This demonstrates how to define your own functions and run evaluations
without needing a functions.py file.
"""

from vowel import run_evals


def square(x: int) -> int:
    """Calculate the square of a number."""
    return x**2


def uppercase(x: str) -> str:
    """Convert string to uppercase."""
    return x.upper() if isinstance(x, str) else str(x).upper()


def filter_positive(lst: list) -> list:
    """Filter positive numbers from a list."""
    return [x for x in lst if x > 0]


eval_specs = {
    "square": {
        "evals": {
            "IsInteger": {"type": "int"},
            "MustBePositive": {"assertion": "output > 0 or input == 0"},
            "CorrectSquare": {"assertion": "input ** 2 == output"},
        },
        "dataset": [
            {"case": {"input": 5, "expected": 25}},
            {"case": {"input": 3, "expected": 9}},
            {"case": {"input": -4, "expected": 16}},
            {"case": {"input": 0, "expected": 0}},
        ],
    },
    "uppercase": {
        "evals": {
            "IsString": {"type": "str"},
            "AllUppercase": {"assertion": "output.isupper()"},
            "ContainsInputString": {
                "contains_input": True,
                "case_sensitive": False,
            },
        },
        "dataset": [
            {"case": {"input": "hello", "expected": "HELLO"}},
            {"case": {"input": "world", "expected": "WORLD"}},
            {"case": {"input": "test", "expected": "TEST"}},
        ],
    },
    "filter_positive": {
        "evals": {
            "IsList": {"type": "list"},
            "AllPositive": {"assertion": "all(x > 0 for x in output)"},
            "SmallerOrEqual": {"assertion": "len(output) <= len(input)"},
        },
        "dataset": [
            {"case": {"input": [1, -2, 3, -4, 5], "expected": [1, 3, 5]}},
            {"case": {"input": [10, 20, 30], "expected": [10, 20, 30]}},
            {"case": {"input": [-5, -10, -15], "expected": []}},
        ],
    },
}

if __name__ == "__main__":
    # Run all evaluations
    summary = run_evals(
        eval_specs,
        functions={
            "square": square,
            "uppercase": uppercase,
            "filter_positive": filter_positive,
        },
    )

    summary.print(include_reasons=True)

    # Run only specific functions
    print("\n" + "=" * 60)
    print("Running only 'square' function")
    print("=" * 60)

    summary = run_evals(
        eval_specs,
        filter_funcs=["square"],
        functions={"square": square},
    )

    summary.print(include_reasons=True)

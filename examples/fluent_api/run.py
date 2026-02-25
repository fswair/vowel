"""
Example: Fluent API (RunEvals)
===============================

Demonstrates the RunEvals fluent API:
- Loading from file, string, and dict
- Chaining .with_functions(), .filter(), .debug()
- Filtering specific functions
- Serializers for custom output comparison
- Coverage threshold checking

Run: python -m examples.fluent_api.run
"""

from vowel import RunEvals


# ── Sample functions ──────────────────────────────────────────
def double(x: int) -> int:
    return x * 2


def triple(x: int) -> int:
    return x * 3


def reverse(text: str) -> str:
    return text[::-1]


def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)


def main():
    print("=" * 60)
    print("Example: Fluent API (RunEvals)")
    print("=" * 60)

    funcs = {
        "double": double,
        "triple": triple,
        "reverse": reverse,
        "fizzbuzz": fizzbuzz,
    }

    # ── 1. From YAML file ────────────────────────────────────
    print("\n▸ From YAML file...")
    summary = (
        RunEvals.from_file("examples/fluent_api/evals.yml").with_functions(funcs).debug().run()
    )
    summary.print()

    # ── 2. From YAML string ──────────────────────────────────
    print("\n▸ From YAML string...")
    yaml_spec = """\
double:
  dataset:
    - case: { input: 7, expected: 14 }
    - case: { input: -3, expected: -6 }
"""
    summary = RunEvals.from_source(yaml_spec).with_functions({"double": double}).run()
    summary.print()

    # ── 3. From Python dict ──────────────────────────────────
    print("\n▸ From Python dict...")
    spec_dict = {
        "triple": {
            "dataset": [
                {"case": {"input": 4, "expected": 12}},
                {"case": {"input": 0, "expected": 0}},
            ]
        }
    }
    summary = RunEvals.from_dict(spec_dict).with_functions({"triple": triple}).run()
    summary.print()

    # ── 4. Filtering specific functions ──────────────────────
    print("\n▸ Filtering only 'fizzbuzz'...")
    summary = (
        RunEvals.from_file("examples/fluent_api/evals.yml")
        .with_functions(funcs)
        .filter(["fizzbuzz"])
        .run()
    )
    summary.print()

    # ── 5. Coverage threshold ────────────────────────────────
    print(f"\n▸ Coverage: {summary.coverage:.0%}")
    print(f"  Meets 90%? {summary.meets_coverage(0.9)}")

    # ── 6. Export for LLM feedback ───────────────────────────
    print("\n▸ JSON export (for LLM feedback):")
    data = summary.to_json()
    print(f"  Keys: {list(data.keys())}")


if __name__ == "__main__":
    main()

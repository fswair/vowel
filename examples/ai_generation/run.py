"""
Example: AI-Powered Generation (EvalGenerator)
================================================

Demonstrates vowel's LLM-powered features:
- Generate a function from a natural language description
- Generate an eval spec for an existing function
- Full pipeline: generate → test → auto-heal

Requires: MODEL_NAME environment variable (e.g. "openai:gpt-4o")

Run: python -m examples.ai_generation.run
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_env() -> bool:
    if not os.getenv("MODEL_NAME"):
        print("Set MODEL_NAME env var to run this example.")
        print("  e.g.  MODEL_NAME=openai:gpt-4o")
        return False
    return True


def demo_generate_function():
    """Generate a Python function from natural language."""
    from vowel import EvalGenerator

    print("\n── Generate a function ─────────────────────────")

    gen = EvalGenerator()

    func = gen.generate_function(
        "Calculate nth fibonacci number using iteration. Handle n=0 and n=1 as base cases.",
        async_func=False,
    )

    print(f"Name: {func.name}")
    print(f"Code:\n{func.code}")
    print(f"\nfib(10) = {func(10)}")
    return func


def demo_generate_spec():
    """Generate an eval spec for an existing function."""
    from vowel import EvalGenerator, Function

    print("\n── Generate eval spec ──────────────────────────")

    def is_palindrome(s: str) -> bool:
        """Check if string is a palindrome (case-insensitive)."""
        cleaned = "".join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    func = Function(
        name="is_palindrome",
        description="Check if string is a palindrome (case-insensitive)",
        code='def is_palindrome(s: str) -> bool:\n    cleaned = "".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]',
        func=is_palindrome,
    )

    gen = EvalGenerator()
    runner, _ = gen.generate_spec(
        func,
        additional_context="Test: empty string, single char, spaces, punctuation",
        save_to_file=False,
    )

    summary = runner.run()
    summary.print()
    return summary


def demo_generate_and_run():
    """Full pipeline: generate function → generate evals → run → heal."""
    from vowel import EvalGenerator

    print("\n── Full pipeline (generate & run) ──────────────")

    gen = EvalGenerator()

    func = gen.generate_function(
        "Calculate factorial of n. Raise ValueError for negative numbers.",
        async_func=False,
    )

    result = gen.generate_and_run(
        func,
        additional_context="Test: 0, 1, 5, 10, negative (should raise)",
        auto_retry=True,
        max_retries=2,
        min_coverage=0.8,
        heal_function=True,
    )

    result.print()

    print(f"\nCoverage : {result.summary.coverage:.0%}")
    print(f"Healed   : {result.was_healed}")


def main():
    print("=" * 60)
    print("Example: AI-Powered Generation")
    print("=" * 60)

    if not check_env():
        sys.exit(1)

    demo_generate_function()
    demo_generate_spec()
    demo_generate_and_run()


if __name__ == "__main__":
    main()

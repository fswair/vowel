"""Test script for EvalGenerator and GenerationResult."""

from vowel import EvalGenerator, GenerationResult


def main():
    generator = EvalGenerator(load_env=True)

    print(f"\nUsing model: {generator.model}")
    print("\n🚀 Step 1: Generate a function from prompt\n")

    func = generator.generate_function(
        prompt="Create a function called 'is_prime' that checks if a number is prime. Return True if prime, False otherwise.",
        async_func=False,
    )

    print(f"Generated: {func.name}")
    func.print()

    print("\n🧪 Step 2: Generate spec and run evals\n")

    result: GenerationResult = generator.generate_and_run(
        func,
        auto_retry=True,
        max_retries=2,
        min_coverage=0.9,
        heal_function=True,
    )

    result.print()

    print("✅ Test completed!\n")


if __name__ == "__main__":
    main()

"""Simple functions to demonstrate basic vowel evaluation."""


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0

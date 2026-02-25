"""Shared functions for CLI eval examples.

These functions are referenced from YAML files via module paths
like `examples.evals.functions.is_palindrome`.

Run any YAML file in this directory with:
    vowel examples/evals/<file>.yml
"""

import json
import re


def is_palindrome(text: str) -> bool:
    """Check if text is a palindrome (case-insensitive, ignoring non-alphanumeric)."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split()) if text.strip() else 0


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension from a filename."""
    parts = filename.rsplit(".", 1)
    return parts[1].lower() if len(parts) > 1 else ""


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def classify_age_group(age: int) -> str:
    """Classify age into child/teenager/adult/senior."""
    if age < 13:
        return "child"
    if age < 18:
        return "teenager"
    if age < 65:
        return "adult"
    return "senior"


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI rounded to 2 decimal places."""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    return round(weight_kg / (height_m**2), 2)


def fibonacci(n: int) -> int:
    """Calculate nth fibonacci number iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text."""
    return re.findall(r"#\w+", text)


def parse_json(text: str) -> dict:
    """Parse JSON string, return empty dict on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def format_phone(number: str) -> str:
    """Format 10-digit phone number to (XXX) XXX-XXXX."""
    digits = "".join(filter(str.isdigit, number))
    if len(digits) != 10:
        raise ValueError("Phone number must have 10 digits")
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def clamp(value: int, low: int, high: int) -> int:
    """Clamp value between low and high."""
    return max(low, min(value, high))

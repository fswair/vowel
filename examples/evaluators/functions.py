"""Functions used to demonstrate vowel evaluators."""

import re


def calculate_discount(price: float, percent: float) -> float:
    """Calculate discounted price. Raises ValueError for invalid percent."""
    if percent < 0 or percent > 100:
        raise ValueError("Discount percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)


def format_phone(number: str) -> str:
    """Format phone number to (XXX) XXX-XXXX."""
    digits = "".join(filter(str.isdigit, number))
    if len(digits) != 10:
        raise ValueError("Phone number must have 10 digits")
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def fibonacci(n: int) -> int:
    """Calculate nth fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def classify_age_group(age: int) -> str:
    """Classify age into groups: child, teenager, adult, senior."""
    if age < 13:
        return "child"
    if age < 18:
        return "teenager"
    if age < 65:
        return "adult"
    return "senior"


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text."""
    return re.findall(r"#\w+", text)


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI rounded to 2 decimal places."""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    return round(weight_kg / (height_m**2), 2)

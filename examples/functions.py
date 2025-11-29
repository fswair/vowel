import asyncio
import json
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


async def calculate_discount(price: float, discount_percent: float) -> float:
    if price < 0:
        raise ValueError("Price must be positive")
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return round(price * (1 - discount_percent / 100), 2)


def format_user_greeting(user: Dict[str, Any]) -> str:
    name = user.get("name", "Guest")
    age = user.get("age")
    if age and age >= 18:
        return f"Welcome, {name}! You have full access."
    return f"Hi {name}! Parental supervision required."


def parse_json_config(config_str: str) -> Dict[str, Any]:
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        return {}


def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#\w+", text)


def classify_age_group(age: int) -> str:
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age < 13:
        return "child"
    elif age < 18:
        return "teenager"
    elif age < 65:
        return "adult"
    else:
        return "senior"


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    return round(weight_kg / (height_m**2), 2)


def is_palindrome(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    return len(text.split())


def get_file_extension(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return parts[1].lower() if len(parts) > 1 else ""


summarizer_agent = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a text summarization expert. Provide concise summaries.",
)


def summarize_text(text: str) -> str:
    async def _run():
        result = await summarizer_agent.run(f"Summarize this text in one sentence: {text}")
        return result.output

    return asyncio.run(_run())


sentiment_agent = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a sentiment analysis expert. Respond with only: positive, negative, or neutral.",
)


def analyze_sentiment(text: str) -> str:
    async def _run():
        result = await sentiment_agent.run(f"What is the sentiment of this text: {text}")
        return result.output.strip().lower()

    return asyncio.run(_run())


translator_agent = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a professional translator. Translate text accurately.",
)


def translate_to_english(text: str, source_lang: str) -> str:
    async def _run():
        result = await translator_agent.run(f"Translate this {source_lang} text to English: {text}")
        return result.output

    return asyncio.run(_run())


code_generator = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a Python code expert. Generate clean, working Python code.",
)


def generate_python_function(description: str) -> str:
    async def _run():
        result = await code_generator.run(
            f"Generate a Python function that {description}. "
            f"Return only the function definition, no explanations."
        )
        return result.output

    return asyncio.run(_run())


math_tutor = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a math tutor. Explain mathematical concepts clearly.",
)


def explain_math_concept(concept: str) -> str:
    async def _run():
        result = await math_tutor.run(
            f"Explain {concept} in simple terms for a beginner. Keep it under 100 words."
        )
        return result.output

    return asyncio.run(_run())


def binary_search(sorted_list: list[int], target: int) -> int:
    """Binary search for target in sorted_list (ascending). Returns index or -1 if not found."""
    low = 0
    high = len(sorted_list) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

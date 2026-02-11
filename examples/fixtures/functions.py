"""Functions and fixtures for the fixtures example.

Fixtures are injected as keyword-only arguments (after *).
Setup functions return the fixture value; teardown cleans up.
"""

import os
import tempfile


# ── Fixture setup / teardown functions ────────────────────────
def create_tmp_file():
    """Setup: create a temporary file, return its path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    return path


def remove_tmp_file(path: str):
    """Teardown: delete the temporary file."""
    if os.path.exists(path):
        os.unlink(path)


def create_db():
    """Setup: return an in-memory user database."""
    print("Database initialized.")
    return {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
    }


def create_config():
    """Setup: return a config dict with a bonus value."""
    return {"bonus": 10}


# ── Functions that consume fixtures via keyword-only args ─────
def write_and_count(content: str, *, tmp: str) -> int:
    """Write content to a temp file and return chars written."""
    with open(tmp, "w") as f:
        return f.write(content)


def count_users(name: str, *, db: dict) -> int:
    """Count users in the fixture database."""
    return len(db.get("users", []))


def add_with_bonus(a: int, b: int, *, config: dict) -> int:
    """Add two numbers plus a bonus from config fixture."""
    return a + b + config["bonus"]

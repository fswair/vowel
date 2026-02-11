"""Test generator fixture support (pytest-style yield fixtures)."""

import tempfile

from vowel import RunEvals


# Generator fixture (pytest-style)
def temp_file_generator():
    """Create a temp file, yield it, then cleanup after."""
    print("  [GEN] Setting up temp file...")
    fd, path = tempfile.mkstemp(suffix=".txt")
    yield path
    # Cleanup after yield
    print("  [GEN] Tearing down temp file...")
    import os

    os.close(fd)
    os.remove(path)
    print("  [GEN] Temp file cleaned up!")


# Tuple API fixture (existing behavior)
def setup_temp_dir():
    print("  [TUPLE] Setting up temp dir...")

    path = tempfile.mkdtemp()
    return path


def teardown_temp_dir(path):
    print("  [TUPLE] Tearing down temp dir...")
    import shutil

    shutil.rmtree(path)
    print("  [TUPLE] Temp dir cleaned up!")


# Normal fixture (no teardown)
def simple_data():
    return {"count": 42}


# Functions using fixtures
def write_to_temp_file(*, temp_file: str, content: str) -> int:
    """Write content to file path and return bytes written."""
    print(f"  [FUNC] Writing to {temp_file}...")
    with open(temp_file, "w") as f:
        return f.write(content)


def check_dir_exists(*, temp_dir: str, should_exist: bool) -> bool:
    """Check if directory exists."""
    import os

    exists = os.path.exists(temp_dir)
    print(f"  [FUNC] Dir exists: {exists}, should_exist: {should_exist}")
    return exists == should_exist


def get_count(*, data) -> int:
    """Extract count from data fixture."""
    print(f"  [FUNC] Getting count from data: {data}")
    return data["count"]


def main():
    print("=" * 60)
    print("TESTING GENERATOR FIXTURES (pytest-style yield)")
    print("=" * 60)

    eval_spec = {
        # Test 1: Generator fixture
        "write_to_temp_file": {
            "fixture": ["temp_file"],
            "dataset": [
                {
                    "case": {
                        "inputs": {"content": "Hello World"},
                        "expected": 11,  # "Hello World" = 11 chars
                    }
                },
                {"case": {"inputs": {"content": "Test"}, "expected": 4}},
            ],
        },
        # Test 2: Tuple API fixture
        "check_dir_exists": {
            "fixture": ["temp_dir"],
            "dataset": [{"case": {"inputs": {"should_exist": True}, "expected": True}}],
        },
    }

    print("\n--- Running tests with generator fixture ---\n")

    summary = (
        RunEvals.from_dict(eval_spec)
        .with_functions(
            {
                "write_to_temp_file": write_to_temp_file,
                "check_dir_exists": check_dir_exists,
            }
        )
        .with_fixtures(
            {
                "temp_file": temp_file_generator,  # Generator fixture (pytest-style)
                "temp_dir": (setup_temp_dir, teardown_temp_dir),  # Tuple API (existing)
            }
        )
        .run()
    )

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    summary.print()

    print("\n" + "=" * 60)
    if summary.all_passed:
        print("✅ ALL TESTS PASSED - Generator fixtures work!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()

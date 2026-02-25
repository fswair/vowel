"""
Example: Fixtures (Dependency Injection)
=========================================

Demonstrates vowel's fixture system:
- Fixtures declared via `fixture: [name]` in YAML
- Injected as keyword-only arguments to functions
- Setup-only fixtures (callable) or setup+teardown (tuple)

Run: python -m examples.fixtures.run
"""

from vowel import RunEvals

from .functions import (
    add_with_bonus,
    count_users,
    create_config,
    create_db,
    create_tmp_file,
    remove_tmp_file,
    write_and_count,
)


def main():
    print("=" * 60)
    print("Example: Fixtures")
    print("=" * 60)

    summary = (
        RunEvals.from_file("fixtures/evals.yml")
        .with_functions(
            {
                "write_and_count": write_and_count,
                "count_users": count_users,
                "add_with_bonus": add_with_bonus,
            }
        )
        .with_fixtures(
            {
                # Setup + teardown (tuple)
                "tmp": (create_tmp_file, remove_tmp_file),
                # Setup only (callable)
                "db": create_db,
                "config": create_config,
            }
        )
        .debug()
        .run()
    )

    summary.print()


if __name__ == "__main__":
    main()

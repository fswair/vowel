# Repository usage and priorities

This document contains concise rules for how agents should inspect and use this project.

## Priority and Workflow

1.  **Read `README.md` first** — It is the primary source for the project's purpose, setup, and quick start.
2.  **Review `docs/` in order** — The documentation contains conceptual decisions and usage examples:
    *   `USERGUIDE.md`: Core concepts and YAML structure.
    *   `API.md`: Programmatic usage.
    *   `YAML_SPEC.md`: Detailed schema for evaluation files.
    *   `FIXTURES.md`: How to use and generate test data.
3.  **Check `examples/`** — Practical examples, eval definitions, and usage scenarios.
4.  **Inspect `db_fixture.yml` and `db.py`** — Example fixtures and data structures used by tests.

## Production Guidance

- **Core Logic:** For production-grade specs or evaluation work, use `skills/vowel-core`.
- **Context:** Refer to `skills/vowel-core/resources/EVAL_SPEC_CONTEXT.md` for production context and guidance.

## Development & Testing

- **Run Tests:** Execute `pytest` to ensure the environment is stable before making changes.
- **Validation:** Use `vowel-schema.json` to validate any new YAML evaluation specs.
- **Consistency:** Keep new evals or specs consistent with existing patterns in `examples/` and `docs/`.

## Quick Tips

- Use the CLI for watching changes: `vowel watch <file>`.
- If you have questions or uncertainty, consult `README.md` and the relevant docs pages.
- Check `TODO` for pending tasks or known issues.

These rules help agents use the project consistently and safely.


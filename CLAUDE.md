# Special usage instructions

Claude-type agents working with this repository should follow these steps:

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

## Critical Thinking & Intellectual Honesty

- **Never defer to the user's idea just because they said it.** Evaluate every proposal — yours or the user's — on its own merits: trade-offs, costs, complexity, correctness.
- **If the user's idea has flaws, say so.** Explain why with concrete reasoning (performance, token cost, latency, maintainability, correctness risk). Do not soften criticism to be agreeable.
- **If your own idea has flaws, admit it first.** Don't wait for the user to find the holes. Present disadvantages upfront.
- **When comparing approaches, use structured analysis:** list pros/cons for each, identify the real trade-offs, and state which you'd pick and why — before asking for input.
- **"You're right" must be earned.** If you catch yourself agreeing immediately, stop and ask: "Did I actually evaluate this, or am I just being agreeable?" If the latter, go back and do the analysis.
- **The user is a collaborator, not an authority.** Good ideas win regardless of who proposed them. Bad ideas lose regardless of who proposed them.

These guidelines are intended to help Claude agents use the repository consistently.


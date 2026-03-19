# CLI Reference

```bash
vowel [OPTIONS] [YAML_FILE]
```

---

## Core Options

| Option | Short | Description |
|--------|-------|-------------|
| `--debug` | | Enable debug mode with stack traces |
| `--dir` | `-d` | Run all YAML files in directory (recursive) |
| `--filter` | `-f` | Only run specific function(s) (comma-separated) |
| `--cov` / `--coverage` | | Required coverage percent (default: 100) |
| `--ci` | | CI mode — exit 1 if coverage not met |
| `--quiet` | `-q` | Minimal output, only show summary |
| `--no-color` | | Disable colored output |
| `--watch` | `-w` | Watch mode: re-run on file changes |
| `--ignore-duration` | | Ignore duration constraints |
| `--verbose` | `-v` | Detailed summary: spec overview, per-case breakdown, pass rate |
| `--hide-report` | | Hide pydantic_evals report output (use with or without `-v`) |

## Information Options

| Option | Description |
|--------|-------------|
| `--list-fixtures` | List fixtures with setup/teardown/scope/usage |
| `--dry-run` | Show test plan without running tests |
| `--fixture-tree` | Show fixture dependency tree |
| `--export-json PATH` | Export results to JSON file |

---

## Examples

```bash
# Run single file
vowel evals.yml

# Run directory
vowel -d ./tests

# Filter functions
vowel evals.yml -f add,multiply,divide

# CI with 90% coverage requirement
vowel evals.yml --ci --cov 90

# Watch mode — re-run on file changes
vowel evals.yml --watch

# Watch with filter
vowel evals.yml -w -f add,multiply

# List fixtures
vowel evals.yml --list-fixtures

# Dry run — show test plan
vowel evals.yml --dry-run

# Show fixture tree
vowel evals.yml --fixture-tree

# Export results
vowel evals.yml --export-json results.json

# Verbose summary — spec overview, per-function table, case breakdown, pass rate
vowel evals.yml -v

# Verbose + hide pydantic_evals report (clean output)
vowel evals.yml -v --hide-report

# Hide report without verbose (still shows Overall Summary panel)
vowel evals.yml --hide-report

# Validate YAML + refresh schema header
vowel schema evals.yml

# Generate schema JSON file (default: vowel-schema.json)
vowel schema --create

# Generate schema JSON at a custom path
vowel schema --create ./schemas/vowel-schema.json

# Show tracked model costs
vowel costs --list
vowel costs --by-generation
vowel costs --by-run
vowel costs --generation <generation_id>
vowel costs --run <run_id>
```

---

## Schema Commands

Use schema commands to validate specs and keep YAML schema headers in sync.

| Command | Description |
|--------|-------------|
| `vowel schema <file>` | Validates YAML and updates the file's schema header safely |
| `vowel schema --create [path]` | Generates `vowel-schema.json` (or writes to custom path) |

---

## Cost Commands

Use cost commands to inspect generation and run cost history.

| Command | Description |
|--------|-------------|
| `vowel costs --list` | List all tracked generations and runs |
| `vowel costs --by-generation` | Aggregate totals by generation id |
| `vowel costs --by-run` | Aggregate totals by run id |
| `vowel costs --generation <id>` | Show detailed rows for one generation |
| `vowel costs --run <id>` | Show detailed rows for one run |

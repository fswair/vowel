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
```

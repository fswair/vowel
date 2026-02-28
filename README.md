# VOWEL

**YAML-based evaluation framework for testing Python functions with AI-powered test generation, function healing and TDD approach.**

vowel makes it easy to define test cases in YAML and run them against your Python functions. It also provides AI-powered generators that can automatically create test specs, generate implementations, and fix buggy functions.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fswair/vowel) [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/fswair/vowel) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/vowel?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/vowel)

## Installation

```bash
pip install vowel

# Or with uv
uv add vowel
```

## Optional Dependencies

Vowel supports several optional dependency groups for enhanced functionality:

| Group         | Install Command                        | Purpose / Extras                |
|---------------|---------------------------------------|---------------------------------|
| **all**       | `pip install vowel[all]`              | All optional features           |
| **dev**       | `pip install vowel[dev]`              | Development & testing tools     |
| **mcp**       | `pip install vowel[mcp]`              | MCP server   |
| **optimization** | `pip install vowel[optimize]`      | Performance optimizations       |
| **monty**     | `pip install vowel[monty]`            | Monty runtime support     |
| **logfire**   | `pip install vowel[logfire]`          | Logfire integration             |

> **Tip:**  
> You can install multiple extras at once, e.g.  
> `pip install vowel[dev,mcp]`
> Recommended: `pip install vowel[all]`

### Development

```bash
git clone https://github.com/fswair/vowel.git
cd vowel
pip install -e ".[all]"
```

---

## Quick Start

> **Note:**  
> For a deeper understanding of how vowel handles fixtures, see the examples in [`db_fixture.yml`](./db_fixture.yml) and [`db.py`](./db.py). These files demonstrate the underlying mechanics of fixture setup and usage.

> **Tip:**  
> To enable YAML schema validation in your editor, place `vowel-schema.json` in your project directory.  
> Then, add the following directive at the top of your YAML file to activate schema support and instructions:
>
> ```yaml
> # yaml-language-server: $schema=<path/to/vowel-schema.json>
> ```
>
> Replace `<path/to/vowel-schema.json>` with the actual path to your schema file.

### 1. Create a YAML spec

```yaml
# evals.yml
add:
  dataset:
    - case:
        inputs: { x: 1, y: 2 }
        expected: 3
    - case:
        inputs: { x: -5, y: 5 }
        expected: 0

divide:
  evals:
    Type:
      type: "float"
  dataset:
    - case:
        inputs: { a: 10, b: 2 }
        expected: 5.0
    - case:
        inputs: { a: 1, b: 0 }
        raises: ZeroDivisionError
```

### 2. Run from CLI

```bash
vowel evals.yml
```

### 3. Or programmatically

```python
from vowel import run_evals

def add(x: int, y: int) -> int:
    return x + y

def divide(a: float, b: float) -> float:
    return a / b

summary = run_evals("evals.yml", functions={"add": add, "divide": divide})
print(f"All passed: {summary.all_passed}")
print(f"Coverage: {summary.coverage * 100:.1f}%")
```

### 4. Or use the fluent API

```python
from vowel import RunEvals

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"add": add, "divide": divide})
    .filter(["add"])
    .debug()
    .run()
)

summary.print()
```

---

## Features

### Evaluators

8 built-in evaluators for flexible testing:

| Evaluator | Purpose |
|-----------|---------|
| **Expected** | Exact value matching |
| **Type** | Return type checking (strict/lenient) |
| **Assertion** | Custom Python expressions (`output > 0`, `output == input * 2`) |
| **Duration** | Performance constraints (function-level & case-level) |
| **Pattern** | Regex validation on output |
| **ContainsInput** | Verify output contains the input |
| **Raises** | Exception class + optional message matching |
| **LLMJudge** | AI-powered rubric evaluation |

```yaml
factorial:
  evals:
    Assertion:
      assertion: "output > 0"
    Type:
      type: "int"
    Duration:
      duration: 1.0
  dataset:
    - case: { input: 0, expected: 1 }
    - case: { input: 5, expected: 120 }
```

> **Full reference:** [docs/EVALUATORS.md](https://github.com/fswair/vowel/blob/main/docs/EVALUATORS.md)

### Fixtures (Dependency Injection)

Inject databases, temp files, caches into functions under test. Three patterns: generator (yield), tuple (setup/teardown), simple (setup only).

```yaml
fixtures:
  db:
    setup: myapp.setup_db
    teardown: myapp.close_db
    scope: module

query_user:
  fixture: [db]
  dataset:
    - case:
        inputs: { user_id: 1 }
        expected: { name: "Alice" }
```

```python
def query_user(user_id: int, *, db: dict) -> dict | None:
    return db["users"].get(user_id)
```

> **Full reference:** [docs/FIXTURES.md](https://github.com/fswair/vowel/blob/main/docs/FIXTURES.md)

### Input Serializers

Transform YAML inputs into Pydantic models, dates, or custom types:

```python
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"get_user": get_user})
    .with_serializer({"get_user": User})      # Schema mode
    .run()
)
```

> **Full reference:** [docs/SERIALIZERS.md](https://github.com/fswair/vowel/blob/main/docs/SERIALIZERS.md)

### AI-Powered Generation

#### EvalGenerator — test existing functions

```python
from vowel import EvalGenerator, Function

generator = EvalGenerator(model="openai:gpt-4o", load_env=True)
func = Function.from_callable(my_function)

result = generator.generate_and_run(func, auto_retry=True, heal_function=True)
print(f"Coverage: {result.summary.coverage * 100:.1f}%")
```

#### TDDGenerator — generate everything from a description

```python
from vowel.tdd import TDDGenerator

generator = TDDGenerator(model="gemini-3-flash-preview", load_env=True)

result = generator.generate_all(
    description="Binary search for target in sorted list. Returns index or -1.",
    name="binary_search"
)

result.print()  # Shows: signature → tests → code → results
```

Step-by-step control:

```python
description = "Calculate factorial of a non-negative integer"
signature = generator.generate_signature(description=description, name="factorial")
runner, yaml_spec = generator.generate_evals_from_signature(signature, description=description)
func = generator.generate_implementation(signature, yaml_spec, description=description)
summary = runner.with_functions({"factorial": func.impl}).run()
```

> **Full reference:** [docs/AI_GENERATION.md](https://github.com/fswair/vowel/blob/main/docs/AI_GENERATION.md)

### MCP Server

Expose vowel's capabilities to AI assistants like Claude Desktop via Model Context Protocol.

> **Setup guide:** [docs/MCP.md](https://github.com/fswair/vowel/blob/main/docs/MCP.md)

---

## CLI

```bash
vowel evals.yml                          # Run single file
vowel -d ./tests                         # Run directory
vowel evals.yml -f add,divide            # Filter functions
vowel evals.yml --ci --cov 90         # CI mode
vowel evals.yml --watch                  # Watch mode
vowel evals.yml --dry-run                # Show plan without running
vowel evals.yml --export-json out.json   # Export results
vowel evals.yml -v                       # Verbose summary
vowel evals.yml -v --hide-report         # Verbose, hide pydantic_evals report
```

> **Full reference:** [docs/CLI.md](https://github.com/fswair/vowel/blob/main/docs/CLI.md)

---

## EvalSummary

```python
summary = run_evals("evals.yml", functions={...})

summary.all_passed       # bool
summary.success_count    # int
summary.failed_count     # int
summary.total_count      # int
summary.coverage         # float (0.0-1.0)
summary.failed_results   # list[EvalResult]

summary.meets_coverage(0.9)    # Check threshold
summary.print()                # Rich formatted output
summary.to_json()              # Export as dict
summary.xml()                  # Export as XML
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [YAML Spec](https://github.com/fswair/vowel/blob/main/docs/YAML_SPEC.md) | Complete YAML format reference |
| [Evaluators](https://github.com/fswair/vowel/blob/main/docs/EVALUATORS.md) | All 8 evaluator types |
| [Fixtures](https://github.com/fswair/vowel/blob/main/docs/FIXTURES.md) | Dependency injection guide |
| [Serializers](https://github.com/fswair/vowel/blob/main/docs/SERIALIZERS.md) | Input serializer patterns |
| [AI Generation](https://github.com/fswair/vowel/blob/main/docs/AI_GENERATION.md) | EvalGenerator & TDDGenerator |
| [CLI](https://github.com/fswair/vowel/blob/main/docs/CLI.md) | Command-line reference |
| [MCP Server](https://github.com/fswair/vowel/blob/main/docs/MCP.md) | AI assistant integration |
| [Troubleshooting](https://github.com/fswair/vowel/blob/main/docs/TROUBLESHOOTING.md) | Common errors & solutions |

---

## License

Apache License 2.0

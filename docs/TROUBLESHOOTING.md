# Troubleshooting

Common errors and their solutions.

---

## Function Not Found

**Error:** `Function 'X' not found in builtins`

**Solution:** Use full module path for non-builtin functions:

```yaml
# Wrong
my_function:
  dataset: ...

# Correct
myapp.utils.my_function:
  dataset: ...
```

Or provide functions programmatically:

```python
from vowel import RunEvals
from myapp.utils import my_function

summary = RunEvals.from_file("evals.yml").with_functions({
    "my_function": my_function
}).run()
```

---

## Fixture Error: Undefined Fixture

**Error:** `Eval 'X' requires undefined fixtures: Y`

**Solution:** Define fixtures in YAML or programmatically:

```yaml
fixtures:
  db:
    setup: my_fixtures.setup_db
    teardown: my_fixtures.teardown_db

my_function:
  fixture:
    - db
```

```python
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"func": func})
    .with_fixtures({"db": (setup_db, teardown_db)})
    .run()
)
```

---

## Type Compatibility Warning

**Error:** Warning about non-serializable types

**Solution:** Use `--debug` flag to see details, or use serializer functions:

```python
summary = RunEvals.from_file("evals.yml").with_serializer({
    "my_func": MyPydanticModel
}).run()
```

See [SERIALIZERS.md](./SERIALIZERS.md) for full serializer documentation.

---

## Assertion Failures

**Error:** Tests fail but function seems correct

**Solutions:**
1. Check assertion syntax — must be valid Python expression
2. Use `--debug` to see full error traces
3. Check variable names: `input`, `output`, `expected`, `duration`

```yaml
# Wrong — uses 'inputs' in assertion for single param
single_param:
  dataset:
    - case:
        input: 5
        assertion: "len(inputs) > 0"  # 'inputs' doesn't exist

# Correct
single_param:
  dataset:
    - case:
        input: 5
        assertion: "output > input"
```

---

## CI Mode Exits with Code 1

**Error:** Tests pass but CI exits with error

**Cause:** Coverage below threshold

**Solution:** Check actual coverage or adjust threshold:

```bash
vowel evals.yml --cov 80

# If coverage is 75%, either fix failing tests or lower threshold
vowel evals.yml --ci --cov 75
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_fixtures.py

# Verbose output
pytest -v

# Show coverage
pytest --cov=vowel --cov-report=html
```

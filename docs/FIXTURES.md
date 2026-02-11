# Fixtures (Dependency Injection)

Complete guide to vowel's fixture system for test setup/teardown and dependency injection.

---

## Overview

Fixtures inject external dependencies (databases, temp files, caches) into functions under test. Functions receive fixture values as **keyword-only arguments** (after `*`).

vowel supports three fixture patterns:

| Pattern | Definition | Cleanup |
|---------|-----------|---------|
| Generator | `yield` value | Code after yield |
| Tuple | `(setup_fn, teardown_fn)` | teardown receives setup's return |
| Simple | Just a callable | No cleanup |

---

## 1. Generator Fixtures (pytest-style yield)

```python
import tempfile, os

def temp_file():
    """Create temp file, yield path, cleanup after."""
    fd, path = tempfile.mkstemp()
    yield path  # ← Injected into function
    # Cleanup runs after test
    os.close(fd)
    os.remove(path)

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"write_file": write_file})
    .with_fixtures({"temp_file": temp_file})
    .run()
)
```

## 2. Tuple API (setup + teardown)

```python
def setup_db():
    """Setup and return connection."""
    return Database.connect()

def teardown_db(conn):
    """Close connection. Receives setup's return value."""
    conn.close()

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"query_user": query_user})
    .with_fixtures({
        "db": (setup_db, teardown_db)
    })
    .run()
)
```

## 3. Simple Fixtures (setup only)

```python
def sample_data():
    """Return static data. No cleanup needed."""
    return {"users": ["alice", "bob"]}

summary = (
    RunEvals.from_file("evals.yml")
    .with_fixtures({"data": sample_data})
    .run()
)
```

---

## Fixture Scopes

Fixtures support three lifecycle scopes (defined in YAML):

| Scope | Behavior |
|-------|----------|
| `function` (default) | Setup/teardown for **each** test case |
| `module` | Setup once per eval spec, teardown after all cases |
| `session` | Setup once per `run_evals()` call, teardown at end |

```yaml
fixtures:
  temp_file:
    setup: my_fixtures.temp_file
    scope: function

  db:
    setup: my_fixtures.setup_db
    teardown: my_fixtures.teardown_db
    scope: module

  cache:
    setup: my_fixtures.setup_cache
    scope: session
```

---

## Using Fixtures in YAML

```yaml
fixtures:
  db:
    setup: fixtures.setup_db
    teardown: fixtures.close_db
    scope: module
    params:              # Optional parameters for setup
      db_name: test_db

  temp_file:
    setup: fixtures.create_temp_file
    scope: function

# Functions reference fixtures by name
query_user:
  fixture:
    - db
  dataset:
    - case:
        inputs: { user_id: 1 }
        expected: { name: "Alice" }

write_to_file:
  fixture:
    - temp_file
  dataset:
    - case:
        inputs: { content: "test" }
        expected: 4
```

---

## Complete Example

```python
from vowel import RunEvals
import tempfile
import os

# Generator fixture (pytest-style)
def temp_file():
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    os.remove(path)

# Tuple API
def setup_database():
    return {"connection": True, "data": {}}

def teardown_database(db):
    db["connection"] = False

# Simple fixture
def config():
    return {"debug": True}

# Functions using fixtures (keyword-only args)
def write_to_file(*, temp_file: str, content: str) -> int:
    with open(temp_file, "w") as f:
        return f.write(content)

def query_db(*, db: dict, key: str):
    return db["data"].get(key)

def get_debug(*, config: dict) -> bool:
    return config["debug"]

# Run with fixtures
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({
        "write_to_file": write_to_file,
        "query_db": query_db,
        "get_debug": get_debug,
    })
    .with_fixtures({
        "temp_file": temp_file,                          # Generator
        "db": (setup_database, teardown_database),       # Tuple
        "config": config,                                # Simple
    })
    .run()
)
```

---

## Requirements

- Generator fixtures: Use `yield` to provide value, cleanup code after yield
- Tuple fixtures: `(setup_func, teardown_func)` — teardown receives setup's return value
- Simple fixtures: Just the function, no teardown needed
- Functions must have matching **keyword-only** parameter names (after `*`)
- Fixture names in YAML `fixture: [...]` must match fixture definitions

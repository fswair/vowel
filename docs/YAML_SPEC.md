# YAML Specification

Complete reference for vowel's YAML evaluation format.

## Basic Structure

```yaml
# Top-level fixture definitions (optional)
fixtures:
  fixture_name:
    setup: module.setup_func      # Import path to setup function
    teardown: module.teardown_func # Import path to teardown (optional)
    scope: function                # function | module | session
    kwargs:                        # Keyword arguments for setup function (optional)
      key: value

# Function evaluation specs
function_name:
  fixture:                         # Fixture dependencies (optional)
    - fixture_name
  evals:                           # Global evaluators (optional)
    EvaluatorName:
      param: value
  dataset:                         # Test cases (required)
    - case:
        input: <value>             # Single parameter
        # OR
        inputs: { a: 1, b: 2 }    # Named parameters (dict)
        # OR
        inputs: [1, 2, 3]         # Positional parameters (list)
        expected: <value>          # Expected output (optional)
        raises: ExceptionType      # Expected exception (optional)
        match: "regex pattern"     # Exception message match (optional)
        duration: 50               # Max ms for this case (optional)
```

---

## Input Types

```yaml
# Single parameter function: func(x)
single_param:
  dataset:
    - case:
        input: 42
        expected: 84

# Multiple positional: func(a, b, c)
multi_positional:
  dataset:
    - case:
        inputs: [1, 2, 3]
        expected: 6

# Named parameters: func(x=1, y=2)
named_params:
  dataset:
    - case:
        inputs: { x: 10, y: 20 }
        expected: 30
```

---

## Function Sources

vowel supports four function sources:

```yaml
# 1. Python builtins
len:
  dataset:
    - case:
        input: [1, 2, 3]
        expected: 3

# 2. Standard library (module.function)
math.sqrt:
  dataset:
    - case:
        input: 16
        expected: 4.0

os.path.join:
  dataset:
    - case:
        inputs: ["/home", "user"]
        expected: "/home/user"

# 3. Your own functions (via functions parameter)
my_function:
  dataset:
    - case:
        input: "hello"
        expected: "HELLO"

# 4. Local module functions (module.function)
# If you have utils.py with uppercase() function:
utils.uppercase:
  dataset:
    - case:
        input: "hello"
        expected: "HELLO"
```

---

## Fixtures in YAML Spec

Fixtures provide dependency injection for functions that need external resources. Define fixtures at the top level, then reference them in function specs via `fixture: [name]`. Functions receive fixture values as **keyword-only arguments** (after `*`).

```yaml
# fixtures_example.yml
fixtures:
  db:
    setup: myapp.fixtures.setup_db
    teardown: myapp.fixtures.close_db
    scope: module          # Created once, shared across all cases
    params:
      db_name: test_db

  cache:
    setup: myapp.fixtures.setup_cache
    scope: session         # Created once per run_evals call

  temp_dir:
    setup: myapp.fixtures.create_temp_dir
    teardown: myapp.fixtures.remove_temp_dir
    scope: function        # Created fresh for each case (default)

# Function depends on 'db' fixture
query_user:
  fixture:
    - db
  dataset:
    - case:
        inputs: { user_id: 1 }
        expected: { name: "Alice", email: "alice@test.com" }
    - case:
        inputs: { user_id: 999 }
        expected: null

# Function depends on multiple fixtures
save_with_cache:
  fixture:
    - db
    - cache
  evals:
    Type:
      type: "bool"
  dataset:
    - case:
        inputs: { key: "user:1", value: "Alice" }
        expected: true
```

Corresponding Python functions must use keyword-only args for fixtures:

```python
def query_user(user_id: int, *, db: dict) -> dict | None:
    return db["users"].get(user_id)

def save_with_cache(key: str, value: str, *, db: dict, cache: dict) -> bool:
    db["data"][key] = value
    cache[key] = value
    return True
```

Run with programmatic fixtures:

```python
from vowel import RunEvals

summary = (
    RunEvals.from_file("fixtures_example.yml")
    .with_functions({"query_user": query_user, "save_with_cache": save_with_cache})
    .with_fixtures({
        "db": (setup_db, close_db),       # Tuple: setup + teardown
        "cache": setup_cache,              # Simple: setup only
        "temp_dir": create_temp_dir,       # Generator: yield-based
    })
    .run()
)
```

**Fixture scopes:**
- `function` (default): Setup/teardown for **each** test case
- `module`: Setup once per eval spec, teardown after all cases
- `session`: Setup once per `run_evals()` call, teardown at end

> See [FIXTURES.md](./FIXTURES.md) for the complete fixture guide including Python API patterns.

---

## Input Serializers in YAML Spec

When your functions accept non-primitive types (Pydantic models, dataclasses, custom objects), YAML inputs need to be deserialized. vowel provides two serializer modes via the programmatic API.

**Schema mode** — automatic conversion from YAML dict to type:

```yaml
# user_evals.yml
get_user_info:
  dataset:
    - case:
        input: { id: 1, name: "Alice", email: "alice@test.com" }
        expected: "User Alice has email alice@test.com"
    - case:
        input: { id: 2, name: "Bob", email: "bob@test.com" }
        expected: "User Bob has email bob@test.com"

process:
  dataset:
    - case:
        inputs:
          user: { id: 1, name: "Alice", email: "alice@test.com" }
          config: { timeout: 30, verbose: true }
        expected: "Alice (timeout=30)"
```

```python
from pydantic import BaseModel
from vowel import RunEvals

class User(BaseModel):
    id: int
    name: str
    email: str

class Config(BaseModel):
    timeout: int
    verbose: bool

def get_user_info(user: User) -> str:
    return f"User {user.name} has email {user.email}"

def process(user: User, config: Config) -> str:
    return f"{user.name} (timeout={config.timeout})"

summary = (
    RunEvals.from_file("user_evals.yml")
    .with_functions({"get_user_info": get_user_info, "process": process})
    .with_serializer({
        "get_user_info": User,                            # Single type
        "process": {"user": User, "config": Config},      # Dict maps param→type
    })
    .run()
)
```

**Serial fn mode** — full control over input transformation:

```python
from datetime import date

def parse_date(data: dict) -> date:
    """Raw input dict → date object."""
    raw = data.get("input") or data.get("inputs")
    return date.fromisoformat(raw)

def get_year(d: date) -> int:
    return d.year

spec = """
get_year:
  dataset:
    - case:
        input: "2026-02-07"
        expected: 2026
    - case:
        input: "2000-01-01"
        expected: 2000
"""

summary = (
    RunEvals.from_source(spec)
    .with_functions({"get_year": get_year})
    .with_serializer(serial_fn={"get_year": parse_date})
    .run()
)
```

**Return types from serial_fn:**
- **Single value** → passed as single argument
- **Tuple** → unpacked as positional arguments
- **Dict** → unpacked as keyword arguments

> See [SERIALIZERS.md](./SERIALIZERS.md) for advanced serializer patterns.

# Input Serializers

Input serializers transform YAML inputs before passing them to your functions. Useful for Pydantic models, custom types, or complex parsing.

---

## Two Modes

### 1. Schema Mode (Automatic)

Pass a Pydantic model or callable — inputs are automatically converted:

```python
from pydantic import BaseModel
from vowel import RunEvals

class User(BaseModel):
    id: int
    name: str
    email: str

def get_user_info(user: User) -> str:
    return f"User {user.name} has email {user.email}"

spec = """
get_user_info:
  dataset:
    - case:
        input: {id: 1, name: "Alice", email: "a@a.com"}
        expected: "User Alice has email a@a.com"
"""

summary = (
    RunEvals.from_source(spec)
    .with_functions({"get_user_info": get_user_info})
    .with_serializer({"get_user_info": User})  # Auto-convert
    .run()
)
```

### 2. Serial Fn Mode (Full Control)

Custom function receives raw input dict, returns anything:

```python
def serialize_user(data: dict) -> User:
    """Extract input/inputs and convert to User."""
    raw = data.get("input") or data.get("inputs")
    if isinstance(raw, list):
        raw = raw[0]
    return User(**raw)

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"get_user_info": get_user_info})
    .with_serializer(serial_fn={"get_user_info": serialize_user})
    .run()
)
```

---

## Advanced Examples

### Multiple Parameters with Different Types

```yaml
process:
  dataset:
    - case:
        inputs:
          user: {id: 1, name: "Alice", email: "a@a.com"}
          config: {timeout: 30, verbose: true}
        expected: "Alice (timeout=30)"
```

```python
class Config(BaseModel):
    timeout: int
    verbose: bool

def process(user: User, config: Config) -> str:
    return f"{user.name} (timeout={config.timeout})"

# Dict schema maps param names to types
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"process": process})
    .with_serializer({"process": {"user": User, "config": Config}})
    .run()
)
```

### Custom Parsing Logic

```python
def custom_serializer(data: dict) -> User:
    """Transform YAML structure to match User model."""
    raw = data.get("input") or data.get("inputs")
    return User(
        id=raw["id"],
        name=f"{raw['first_name']} {raw['last_name']}",
        email=raw["email"]
    )
```

### Callable Serializers

```python
from datetime import date

def parse_date(s: str) -> date:
    return date.fromisoformat(s)

def get_year(d: date) -> int:
    return d.year

summary = (
    RunEvals.from_source("get_year: {dataset: [{case: {input: '2026-01-07', expected: 2026}}]}")
    .with_functions({"get_year": get_year})
    .with_serializer({"get_year": parse_date})
    .run()
)
```

---

## Return Types

Your serializer function can return:

| Return Type | Behavior |
|-------------|----------|
| **Single value** | Passed as single argument |
| **Tuple** | Unpacked as positional arguments |
| **Dict** | Unpacked as keyword arguments |

```python
# Returns tuple for positional args
def serialize_pair(data: dict) -> tuple:
    inputs = data["inputs"]
    return (User(**inputs[0]), User(**inputs[1]))

def compare(u1: User, u2: User) -> bool:
    return u1.id < u2.id
```

# RFC: Fixture-Aware Eval Generation

## Problem Statement

Şu an `EvalGenerator` ve `TDDGenerator` fixture kullanan fonksiyonlar için eval spec üretemiyor.
Fixture'lar keyword-only parametreler olarak tanımlanıyor ama generator bunları bilmiyor.

```python
# Bu fonksiyon için eval spec üretemiyoruz
def get_user(user_id: int, *, db: Database) -> User:
    return db.query(User).get(user_id)
```

---

## Current State

### Fixture System (Working ✅)
- `FixtureDefinition` model: setup, cls, teardown, scope, params
- `FixtureManager`: lifecycle management, scoping, yield fixtures
- `RunEvals.with_fixtures()`: programmatic fixture injection
- YAML `fixtures:` block ve `fixture: [...]` field

### Generator System (No Fixture Support ❌)
- `Function` model: name, description, signature, code - **no fixture metadata**
- `EVAL_SPEC_CONTEXT`: **no fixture documentation**
- `generate_spec()`: **fixture-unaware**
- LLM prompt: **no fixture instructions**

---

## Proposed API Design

### Option A: Extend Existing API (Recommended)

```python
# Function model'e fixture_deps ekle
func = Function(
    name="get_user",
    signature="get_user(user_id: int, *, db: Database) -> User",
    code="...",
    fixture_deps=["db"]  # NEW: explicit fixture dependencies
)

# veya from_callable otomatik çıkarsın
func = Function.from_callable(get_user)  # fixture_deps=["db"] auto-detected

# generate_spec'e fixture config ekle
runner, yaml = generator.generate_spec(
    func,
    fixtures={
        "db": {
            "cls": "myapp.db.Database",
            "kwargs": {"url": "sqlite:///:memory:"},
            "scope": "module"
        }
    }
)
```

### Option B: Separate Method

```python
runner, yaml = generator.generate_with_fixtures(
    func,
    fixture_config={
        "db": FixtureDefinition(cls="myapp.db.Database", scope="module")
    }
)
```

### Option C: Fixture Template System

```python
# Fixture template'leri önceden tanımla
generator = EvalGenerator(
    fixture_templates={
        "database": DatabaseFixtureTemplate(),  # setup/teardown logic
        "api_client": APIClientFixtureTemplate(),
    }
)

# Generator otomatik matching yapar
runner, yaml = generator.generate_spec(func)  # db → database template
```

---

## Implementation Plan

### Phase 1: Function Model Update
```python
class Function(BaseModel):
    name: str
    description: str | None = None
    signature: str | None = None
    code: str
    fixture_deps: list[str] = Field(default_factory=list)  # NEW
    
    @classmethod
    def from_callable(cls, fn: Callable) -> "Function":
        # Extract keyword-only params as fixture deps
        sig = inspect.signature(fn)
        fixture_deps = [
            name for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        return cls(..., fixture_deps=fixture_deps)
```

### Phase 2: EVAL_SPEC_CONTEXT Update
```
## FIXTURES

For functions with external dependencies (database, API clients, etc.):

### Fixture Definition
```yaml
fixtures:
  db:
    cls: myapp.Database          # Class to instantiate
    kwargs: {url: "..."}         # Constructor kwargs
    scope: module                # function|module|session
    teardown: myapp.db.close     # Optional cleanup
```

### Using Fixtures
```yaml
my_function:
  fixture: [db]                  # Inject these fixtures
  dataset:
    - case: {input: 1, expected: {...}}
```
```

### Phase 3: Generator Logic Update
```python
def generate_spec(
    self,
    func: Function,
    fixtures: dict[str, dict | FixtureDefinition] | None = None,
    **kwargs
) -> tuple[RunEvals, str]:
    
    # Build fixture context for LLM
    fixture_context = ""
    if func.fixture_deps:
        fixture_context = f"""
This function has fixture dependencies: {func.fixture_deps}
Generate a `fixtures:` block with appropriate setup.
Use `fixture: {func.fixture_deps}` in the eval spec.
"""
        if fixtures:
            fixture_context += f"\nFixture config provided: {fixtures}"
    
    # Include in prompt
    prompt = f"{EVAL_SPEC_CONTEXT}\n{fixture_context}\n..."
```

### Phase 4: Output Generation
LLM generates:
```yaml
fixtures:
  db:
    cls: myapp.db.Database
    scope: module

get_user:
  fixture: [db]
  dataset:
    - case:
        id: user_exists
        input: 1
        expected: {id: 1, name: "Alice"}
```

---

## Open Questions

1. **Fixture Config Required?**
   - A) User MUST provide fixture config (setup paths)
   - B) LLM can generate placeholder fixtures
   - C) Hybrid: LLM generates structure, user fills paths

2. **Fixture Value in Test Cases?**
   ```yaml
   # Option A: Fixture injected, not in inputs
   input: {user_id: 1}
   
   # Option B: Fixture referenced in inputs
   inputs:
     user_id: 1
     db: $fixture:db
   ```

3. **TDDGenerator Integration?**
   - Signature step: Include fixture params?
   - Evals step: Generate fixture block?
   - Implementation step: Handle fixture injection?

4. **Validation?**
   - Reject functions with *args/**kwargs? (current behavior)
   - Require fixture_deps to be keyword-only?
   - Validate fixture config completeness?

---

## Migration Path

1. **Non-breaking**: `fixtures` param optional, defaults to None
2. **Auto-detection**: `from_callable` extracts fixture_deps automatically
3. **Graceful degradation**: No fixtures → current behavior unchanged

---

## Timeline

- [ ] Phase 1: Function model update (1 day)
- [ ] Phase 2: EVAL_SPEC_CONTEXT update (1 day)  
- [ ] Phase 3: Generator logic (2 days)
- [ ] Phase 4: Testing & docs (1 day)

---

## Related Files

- `src/vowel/eval_types.py` - Function, FixtureDefinition models
- `src/vowel/ai.py` - EvalGenerator, TDDGenerator
- `src/vowel/context.py` - EVAL_SPEC_CONTEXT
- `src/vowel/runner.py` - FixtureManager

## Implementation Note

Before implementing fixture-aware generation, add serializer and fixture examples to EVAL_SPEC_CONTEXT in context.py. This ensures LLM understands these features when generating eval specs.

TODO:
- [ ] Add serializer example (type-based and dict-based)
- [ ] Add fixture example (setup/teardown, cls-based, scope)


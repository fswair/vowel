# Feedback-Guided Exploration

## The Problem: Single-Shot Exploration is Blind

Prior to this change, the CodeMode pipeline ran exploration in a single LLM call:

```
Function source code → LLM (one call) → N snippets → Execute all → Done
```

The LLM never saw execution results during exploration. It generated all snippets based purely on **static reasoning** — reading the source code and inferring what inputs would be interesting. This is "speculation-based exploration."

This works surprisingly well with strong models. In our benchmark, Claude Opus 4.6 produced 44 snippets for `parse_cron` in a single call and achieved 100% coverage with zero refinements. But the approach has structural limitations that no amount of model intelligence can overcome:

### What single-shot exploration misses

**1. Exact error messages**

The LLM reads a `raise ValueError(...)` statement and guesses the error message. But the actual message depends on runtime state — string interpolation, variable values, branch ordering. Example:

```python
# LLM expects:
parse_cron('-1 0 1 1 0')  →  ValueError("minute: -1 out of range 0-59")

# Reality:
parse_cron('-1 0 1 1 0')  →  ValueError("invalid literal for int() with base 10: ''")
```

The minus sign is consumed by the range parser (`-` is the range delimiter), leaving an empty string that fails `int()` conversion. This is a parsing precedence issue that can only be discovered by execution.

**2. Input combination explosions**

For grammar-heavy functions (parsers, validators, DSLs), each syntax element works in isolation, but **combinations** of elements may trigger different code paths. Example from cron parsing:

- `*/15` works (step with wildcard)
- `1-10` works (range)
- `1,5,10` works (comma-separated)
- `1,5-7,*/20` — comma + range + step in one field — was never tried

The LLM tests each primitive but rarely discovers multi-primitive combinations without seeing prior execution results.

**3. Error path ordering**

When a function has multiple validation layers, the order matters:

```python
# Does step validation happen before or after range validation?
parse_cron('0-60/0 * * * *')
# Could be: "Step must be positive" or "invalid range 0-60"
```

Only execution reveals which guard fires first.

## The Solution: Two-Round Evidence-Based Exploration

The new pipeline adds a second exploration round that receives **actual execution results** from Round 1:

```
Round 1: LLM (static reasoning) → 15-30 snippets → Execute
              ↓
         Deterministic cluster summary
              ↓
Round 2: LLM (evidence-based)   →  8-12 snippets → Execute
              ↓
         Combined results → Spec Generation
```

Round 2 sees:
- Every snippet that was tried and its exact output
- A programmatic cluster summary grouping results by behavior class
- An explicit "do not repeat" list

This transforms exploration from **speculation** into **hypothesis refinement under feedback** — the LLM reasons about what it _hasn't_ seen, informed by what it _has_ seen.

## Design Decisions

### Why programmatic clustering (not LLM-based)?

We considered two approaches for building the cluster summary between rounds:

| | Programmatic (chosen) | LLM-based |
|---|---|---|
| Cost | Zero — no LLM call | 1 additional call |
| Determinism | Always produces same output for same input | Non-deterministic |
| Speed | Microseconds | Seconds |
| Depth | Surface-level (type + message prefix) | Semantic understanding |

We chose programmatic clustering because the goal is not "perfect semantic grouping" — it's "sufficient signal to guide Round 2." The Round 2 LLM is intelligent enough to infer gaps from a simple type+message summary. Adding a clustering LLM call would introduce cost and non-determinism without proportional benefit.

### Why exactly 2 rounds (not 3+)?

Three considerations:

1. **Diminishing returns**: Round 1 covers ~80-90% of behavior space through static reasoning. Round 2 targets the remaining gaps. A Round 3 would operate on an already-dense behavior map with very few remaining gaps — the ROI drops sharply.

2. **Reasoning fragmentation**: Strong models like Opus do their best reasoning in large, focused context windows. Splitting reasoning across many small rounds can actually degrade quality. Two rounds is the sweet spot: one large reasoning pass, one targeted refinement.

3. **Cost predictability**: Fixed 2-round means exactly 2 exploration LLM calls. This is predictable and benchmarkable. Variable rounds (3-5) make cost unpredictable and harder to compare across models.

The `exploration_rounds` parameter allows override (`=1` restores old behavior, `=3` for complex domains if needed), but the default of 2 is intentional.

### Why early exit conditions?

Two conditions can terminate exploration before Round 2 completes:

1. **No snippets produced**: If the Round 2 LLM returns an empty plan, it believes Round 1 was already comprehensive. Forcing it to produce snippets would yield duplicates.

2. **No new behavior classes discovered**: After executing Round 2 snippets, we compare behavior keys (`ok:dict`, `err:ValueError:minute: 60 out of range`) between prior and new results. If every new snippet produced a behavior we already had, the exploration space is saturated.

## Implementation Details

### Cluster Summary Format

The `_build_cluster_summary()` method produces a structured text summary:

```markdown
## Observed Behaviour Clusters

### Success clusters
- output type `dict`: 18 cases
- output type `bool`: 3 cases
- output type `list`: 1 case

### Error clusters
- `ValueError` (8 distinct messages):
  - "Expected 5 fields, got 3"
  - "minute: 60 out of range 0-59"
  - "Step must be positive, got -1"
  - ...
- `AttributeError` (2 distinct messages):
  - "'NoneType' object has no attribute 'strip'"
  - "'int' object has no attribute 'strip'"

### Already tried (25 snippets — do NOT repeat these)
- `parse_cron('* * * * *')`
- `parse_cron('5 14 1 6 3')`
- ...
```

This is deterministic, costs zero LLM tokens to produce, and provides exactly the signal Round 2 needs:
- What **output shapes** have been seen (so the LLM can target new ones)
- What **error types and messages** were discovered (so the LLM can find adjacent error paths)
- What **exact code** was already tried (so the LLM won't duplicate)

### Round 2 Prompt Structure

The Round 2 prompt includes:

```
<FunctionCode>   — same source code as Round 1
<Round1Results>  — full execution results (code + output/error for each snippet)
<ClusterSummary> — the programmatic summary above

RULES:
- Do NOT repeat any snippet from the "Already tried" list.
- Produce 8–12 NEW normal snippets targeting uncovered behaviour.
- Produce 3–5 NEW error snippets targeting untried error paths.
```

The snippet count targets (8-12 normal, 3-5 error) are intentionally smaller than Round 1 (15+ normal, 3+ error). Round 2 is surgical, not broad.

### Behavior Key Format

For early exit detection, each result is hashed into a behavior key:

```
Success: "ok:{output_type}"         → "ok:dict", "ok:bool", "ok:list"
Error:   "err:{error_type}:{msg40}" → "err:ValueError:minute: 60 out of range 0-59"
```

The message prefix is truncated to 40 characters — enough to distinguish error paths without being sensitive to minor wording variations.

## Code Changes

All changes are in `src/vowel/codemode.py`. No new files, no new dependencies.

### Modified methods

| Method | Change |
|---|---|
| `explore()` | 2-round loop with early exit; delegates execution to `_execute_plan()` |
| `_get_exploration_plan()` | Unchanged logic, updated docstring and logfire tags |

### New methods

| Method | Purpose |
|---|---|
| `_execute_plan()` | Extracted snippet execution loop (reused by both rounds) |
| `_get_targeted_exploration_plan()` | Round 2 prompt with prior results + cluster summary |
| `_build_cluster_summary()` | Programmatic clustering of results into text summary |
| `_count_new_behaviors()` | Compares behavior keys between prior and new results |

### Backward compatibility

- `explore(func, exploration_rounds=1)` restores exact single-shot behavior
- Default is `exploration_rounds=2` — existing callers get the improvement automatically
- `generate()` calls `explore()` without arguments, so it automatically benefits
- All 478 existing unit tests pass without modification

## Expected Impact

### On strong models (Opus-class)

- Round 1 already produces excellent coverage
- Round 2 adds **combination discovery** and **exact error message alignment**
- Net: ~10-15% more snippets, potentially fewer spec refinement rounds (error messages will match exactly)

### On weaker models (Flash/Lite-class)

- Round 1 produces decent but shallow coverage — misses edge cases
- Round 2 **compensates for weaker static reasoning** by showing actual execution results
- Net: significant quality improvement, likely converting some FAIL scenarios to PASS

### On benchmark discriminability

With Katman 3 (behavioral discovery) added, benchmarks now measure a higher-order capability: **adaptive reasoning under feedback**. This separates models that can merely read code from models that can learn from execution traces — a much more meaningful distinction for agentic coding systems.

## Relationship to the Full Pipeline

The evidence flow through the pipeline is now:

```
Round 1 (speculation)     → snippets → execute → results
                                                    ↓
Round 2 (evidence-based)  → snippets → execute → results
                                                    ↓
                                            all exploration results
                                                    ↓
Spec Generation ← VerifiedExecutionResults + ErrorResults
                                                    ↓
                                              YAML eval spec
                                                    ↓
Validation → RunEvals → coverage check
                                                    ↓
Refinement (if needed) ← failure context
```

Evidence-based reasoning now starts at the **exploration phase** rather than only at spec generation. Since exploration results feed directly into spec generation, any improvement in exploration quality cascades through the entire downstream pipeline.

## Origin

This feature was designed through a three-way analysis between the developer, the implementation agent (GitHub Copilot / Claude Opus 4.6), and ChatGPT. ChatGPT identified the core insight: the pipeline was doing "speculation-based exploration" when it could be doing "evidence-based exploration." The implementation agent confirmed this against the actual codebase, proposed the programmatic clustering approach (Yol A) over LLM-based clustering, and implemented the 2-round design.

The key framing that guided the design:

```
Layer 1: Domain awareness    (from function description)     ✅ already strong
Layer 2: Grammar inference   (from source code)              ✅ already strong
Layer 3: Behavioral discovery (from runtime feedback)        ✅ now added
```

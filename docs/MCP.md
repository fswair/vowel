# MCP Server

vowel includes an MCP (Model Context Protocol) server for AI assistant integration. The server exposes vowel's full capabilities to AI assistants like Claude Desktop.

---

## Available Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | `run_evals_from_file` | Run evaluations from YAML file |
| 2 | `run_evals_from_yaml` | Run evaluations from YAML string |
| 3 | `run_evals_with_fixtures` | Run evaluations with fixture injection |
| 4 | `validate_yaml_spec` | Validate YAML spec without execution |
| 5 | `check_function_compatibility` | Check if function types are YAML-serializable |
| 6 | `generate_function` | Generate function from prompt using AI |
| 7 | `generate_eval_spec` | Generate test spec for function |
| 8 | `generate_and_run` | Generate spec and run with auto-healing |
| 9 | `list_yaml_files` | List YAML files in directory |

---

## Setup

1. Install vowel:
```bash
pip install vowel
```

2. Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "vowel": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/vowel",
        "run",
        "vowel-mcp"
      ]
    }
  }
}
```

3. Restart Claude Desktop

---

## Example Usage

Ask Claude:
- "Run the evaluations in tests.yml"
- "Validate this YAML spec: [paste spec]"
- "Check if my function signature is compatible with vowel"
- "Generate tests for this function"
- "Generate and run tests with auto-healing"

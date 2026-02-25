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

## Quick Taste

To quickly try out the MCP server, add the following code to a script named `agent.py`:

```python
import os
import dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

dotenv.load_dotenv()

MODEL_NAME = os.environ["MODEL_NAME"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

server = MCPServerStdio(
  "vowel-mcp",
  [],
  env={
    "MODEL_NAME": MODEL_NAME,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
  },
)

agent = Agent(MODEL_NAME, toolsets=[server])

server = agent.to_web()
```

To serve the web chat UI, run:

```sh
uvicorn agent:server --port 8080 --reload
```

---

## Observability & Monitoring

To enable observability and monitoring for the MCP server, configure the following environment variables:

```bash
export LOGFIRE_ENABLED=true
export LOGFIRE_TOKEN=your_token
```

This will activate logging and monitoring features via Logfire.

## Setup

1. Install vowel:
```bash
pip install vowel[mcp]
```

2. Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "vowel": {
      "command": "vowel-mcp",
      "args": []
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

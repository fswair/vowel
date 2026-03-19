"""Versioned JSON Schema cache and YAML header helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .utils import EvalsBundle

SCHEMA_CACHE_DIR = Path.home() / ".vowel"


def _schema_version_token(version: str | None = None) -> str:
    if version is None:
        try:
            version = importlib.metadata.version("vowel")
        except importlib.metadata.PackageNotFoundError:
            version = "0.0.0"

    ver = version
    nums = re.findall(r"\d+", ver)
    if not nums:
        return "000"
    return "".join(nums)


def build_yaml_schema_from_bundle() -> dict[str, Any]:
    """Build YAML-file schema directly from runtime models.

    No repository reference file is used. The root shape is forced to match
    vowel's YAML file format:
    - top-level optional `fixtures`
    - top-level optional `serializers`
    - top-level additionalProperties => per-function `Evals`
    """
    bundle_schema = EvalsBundle.model_json_schema(ref_template="#/$defs/{model}")
    defs = bundle_schema.get("$defs", {})
    properties = bundle_schema.get("properties", {})
    fixtures_schema = properties.get(
        "fixtures",
        {
            "type": "object",
            "title": "Fixtures",
        },
    )
    serializers_schema = properties.get(
        "serializers",
        {
            "type": "object",
            "title": "Serializers",
        },
    )

    additional_properties: dict[str, Any]
    if "Evals" in defs:
        # Top-level YAML uses function name as key, so `id` should not be
        # required in each map value even though runtime Evals model has it.
        evals_map_value = deepcopy(defs["Evals"])
        required = evals_map_value.get("required")
        if isinstance(required, list):
            evals_map_value["required"] = [k for k in required if k != "id"]
        evals_map_value["title"] = "Function"
        evals_map_value["description"] = (
            "Function evaluation specification keyed by function import path/name. "
            "Contains fixture dependencies, global evaluators (`evals`), and dataset cases."
        )
        defs["EvalsMapValue"] = evals_map_value
        additional_properties = {"$ref": "#/$defs/EvalsMapValue"}
    else:
        evals_schema = properties.get("evals", {"type": "object"})
        additional_properties = evals_schema.get("additionalProperties", {"type": "object"})

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "fixtures": fixtures_schema,
            "serializers": serializers_schema,
        },
        "additionalProperties": additional_properties,
        "$defs": defs,
    }

    return schema


def ensure_cached_schema(version: str | None = None) -> Path:
    """Ensure the versioned schema file exists and is up to date."""
    schema_data = build_yaml_schema_from_bundle()
    rendered = json.dumps(schema_data, indent=2, ensure_ascii=False) + "\n"

    token = _schema_version_token(version)
    digest = hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:8]
    schema_path = SCHEMA_CACHE_DIR / f"vowel-schema_{token}_{digest}.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    if not schema_path.exists() or schema_path.read_text(encoding="utf-8") != rendered:
        schema_path.write_text(rendered, encoding="utf-8")

    return schema_path


def add_schema_header(yaml_spec: str, schema_path: Path | str) -> str:
    """Prepend YAML language-server schema reference header to YAML content."""
    schema_str = str(schema_path)
    header = f"# yaml-language-server: $schema={schema_str}"

    lines = yaml_spec.splitlines()
    if lines and lines[0].startswith("# yaml-language-server: $schema="):
        lines = lines[1:]
        if lines and lines[0] == "":
            lines = lines[1:]

    body = "\n".join(lines).rstrip("\n")
    return f"{header}\n\n{body}\n"


def materialize_yaml_with_schema_header(yaml_spec: str, version: str | None = None) -> str:
    """Create/refresh versioned schema cache and return header-prefixed YAML."""
    schema_path = ensure_cached_schema(version)
    return add_schema_header(yaml_spec, schema_path)

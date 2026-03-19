"""Tests for generated YAML schema support."""

import json
from pathlib import Path

from vowel.schema import build_yaml_schema_from_bundle, materialize_yaml_with_schema_header


def test_generated_schema_includes_top_level_serializers_property():
    """Top-level `serializers` should be explicitly supported in generated schema."""
    schema = build_yaml_schema_from_bundle()
    properties = schema.get("properties", {})

    assert "fixtures" in properties
    assert "serializers" in properties


def test_generated_schema_keeps_function_additional_properties():
    """Unknown top-level keys must still map to per-function Evals definitions."""
    schema = build_yaml_schema_from_bundle()

    additional = schema.get("additionalProperties", {})
    assert additional == {"$ref": "#/$defs/EvalsMapValue"}


def test_materialized_header_uses_hashed_cache_with_serializers():
    """Schema header should reference a content-addressed cache file that supports serializers."""
    yaml_text = "len:\n  dataset:\n    - case:\n        id: len_basic\n        input: [1]\n        expected: 1\n"
    materialized = materialize_yaml_with_schema_header(yaml_text)
    first_line = materialized.splitlines()[0]

    assert first_line.startswith("# yaml-language-server: $schema=")
    schema_path = Path(first_line.split("$schema=", 1)[1])
    assert schema_path.name.startswith("vowel-schema_")
    assert schema_path.exists()

    schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "serializers" in schema_obj.get("properties", {})

"""Serializer models and helpers for the native YAML serializer example."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):
    """Simple SQL query payload used by example evals."""

    sql: str
    params: list[Any] = Field(default_factory=list)


def query_from_payload(payload: dict[str, Any]) -> Query:
    """serial_fn mode example for YAML-native serializer registry.

    Accepts both:
    - {"input": "SELECT ..."}
    - {"input": {"sql": "SELECT ...", "params": [...]}}
    """

    value = payload.get("input")
    if value is None:
        value = payload.get("inputs")

    if isinstance(value, str):
        return Query(sql=value)

    if isinstance(value, dict):
        sql = value.get("sql")
        params = value.get("params", [])
        if not isinstance(sql, str):
            raise ValueError("Expected 'sql' to be a string in query payload")
        if not isinstance(params, list):
            raise ValueError("Expected 'params' to be a list in query payload")
        return Query(sql=sql, params=params)

    raise ValueError("Unsupported query payload format")

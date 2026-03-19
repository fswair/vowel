"""Functions under test for native serializer + fixture example."""

from __future__ import annotations

from typing import Any

from .defn import Query
from .fixtures import DbConnection


def query_users(query: Query, *, db: DbConnection) -> list[dict[str, Any]]:
    """Schema mode example: input dict -> Query model via serializer schema."""
    return db.execute_query(query)


def query_users_custom(query: Query, *, db: DbConnection) -> list[dict[str, Any]]:
    """serial_fn mode example: raw payload -> Query via custom serializer function."""
    return db.execute_query(query)

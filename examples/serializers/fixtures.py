"""Fixture utilities for the native YAML serializer example."""

from __future__ import annotations

import sqlite3
from typing import Any

from .defn import Query


class DbConnection:
    """Tiny sqlite fixture class used by vowel fixture injection."""

    def __init__(self, db_path: str = ":memory:"):
        # Vowel can execute cases in worker threads; allow sqlite usage across them.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._seed()

    def _seed(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
        )
        cur.execute("DELETE FROM users")
        cur.executemany(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            [("Alice", 28), ("Bob", 34), ("Cara", 41)],
        )
        self.conn.commit()

    def execute_query(self, query: Query) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(query.sql, query.params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

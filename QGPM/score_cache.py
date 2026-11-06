"""SQLite cache of the segment relevance of each item, keyed by (book_id, q_id)."""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

import numpy as np


class ScoreCache:
    """float32 relevance per item; the file name has to identify every input that affects the relevance."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS relevance ("
                          "book_id TEXT NOT NULL, q_id TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY (book_id, q_id))")
        self.conn.commit()

    def get(self, book_id: str, q_id: str) -> Optional[np.ndarray]:
        row = self.conn.execute("SELECT data FROM relevance WHERE book_id=? AND q_id=?", (book_id, q_id)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32).copy()

    def put(self, book_id: str, q_id: str, relevance: np.ndarray) -> None:
        self.conn.execute("INSERT OR REPLACE INTO relevance (book_id, q_id, data) VALUES (?, ?, ?)",
                          (book_id, q_id, np.asarray(relevance, dtype=np.float32).tobytes()))
        self.conn.commit()

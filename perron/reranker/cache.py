"""
Content-Addressable Score Cache for Gemma 4 Evaluation Protocol.

Caches pointwise model scores keyed on (instance_id, candidate_id, prompt_variant).
Ensures candidates shared across arms are evaluated exactly once.
Provides crash-recovery and resumable execution.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PointwiseScoreCache:
    """
    Disk-backed SQLite cache for pointwise LLM scores.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pointwise_scores (
                        instance_id TEXT NOT NULL,
                        candidate_id INTEGER NOT NULL,
                        prompt_variant TEXT NOT NULL,
                        s REAL NOT NULL,
                        format_mass REAL NOT NULL,
                        n_prompt_tokens INTEGER NOT NULL,
                        PRIMARY KEY (instance_id, candidate_id, prompt_variant)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_variant 
                    ON pointwise_scores(prompt_variant, instance_id)
                    """
                )
        finally:
            conn.close()

    def get(
        self, instance_id: str, candidate_id: int, prompt_variant: str
    ) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s, format_mass, n_prompt_tokens
                FROM pointwise_scores
                WHERE instance_id = ? AND candidate_id = ? AND prompt_variant = ?
                """,
                (instance_id, candidate_id, prompt_variant),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "s": row[0],
                    "format_mass": row[1],
                    "n_prompt_tokens": row[2],
                }
        finally:
            conn.close()
        return None

    def put(
        self,
        instance_id: str,
        candidate_id: int,
        prompt_variant: str,
        s: float,
        format_mass: float,
        n_prompt_tokens: int,
    ) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pointwise_scores
                    (instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens),
                )
        finally:
            conn.close()

    def put_batch(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        conn = self._get_connection()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pointwise_scores
                    (instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens)
                    VALUES (:instance_id, :candidate_id, :prompt_variant, :s, :format_mass, :n_prompt_tokens)
                    """,
                    records,
                )
        finally:
            conn.close()

    def get_many(
        self, instance_id: str, candidate_ids: List[int], prompt_variant: str
    ) -> Dict[int, Dict[str, Any]]:
        if not candidate_ids:
            return {}
        placeholders = ",".join("?" for _ in candidate_ids)
        params = [instance_id, prompt_variant] + candidate_ids
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT candidate_id, s, format_mass, n_prompt_tokens
                FROM pointwise_scores
                WHERE instance_id = ? AND prompt_variant = ? AND candidate_id IN ({placeholders})
                """,
                params,
            )
            rows = cursor.fetchall()
            return {
                row[0]: {
                    "s": row[1],
                    "format_mass": row[2],
                    "n_prompt_tokens": row[3],
                }
                for row in rows
            }
        finally:
            conn.close()

    def count(self, prompt_variant: Optional[str] = None) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if prompt_variant:
                cursor.execute(
                    "SELECT COUNT(*) FROM pointwise_scores WHERE prompt_variant = ?",
                    (prompt_variant,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM pointwise_scores")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def dump_all(self, prompt_variant: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if prompt_variant:
                cursor.execute(
                    """
                    SELECT instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens
                    FROM pointwise_scores WHERE prompt_variant = ?
                    """,
                    (prompt_variant,),
                )
            else:
                cursor.execute(
                    """
                    SELECT instance_id, candidate_id, prompt_variant, s, format_mass, n_prompt_tokens
                    FROM pointwise_scores
                    """
                )
            rows = cursor.fetchall()
            return [
                {
                    "instance_id": r[0],
                    "candidate_id": r[1],
                    "prompt_variant": r[2],
                    "s": r[3],
                    "format_mass": r[4],
                    "n_prompt_tokens": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

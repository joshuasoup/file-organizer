from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileRecord:
    path: str
    file_type: str
    size: int
    mtime: float
    content_hash: str | None
    embedding_model: str
    indexed_at: float


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                file_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                content_hash TEXT,
                embedding_model TEXT NOT NULL,
                indexed_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM files")
        self._conn.commit()

    def list_paths(self) -> list[str]:
        cursor = self._conn.execute("SELECT path FROM files")
        return [row["path"] for row in cursor.fetchall()]

    def list_records(self) -> list[FileRecord]:
        cursor = self._conn.execute("SELECT * FROM files")
        rows = cursor.fetchall()
        return [
            FileRecord(
                path=row["path"],
                file_type=row["file_type"],
                size=row["size"],
                mtime=row["mtime"],
                content_hash=row["content_hash"],
                embedding_model=row["embedding_model"],
                indexed_at=row["indexed_at"],
            )
            for row in rows
        ]

    def get_file(self, path: str) -> FileRecord | None:
        cursor = self._conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return FileRecord(
            path=row["path"],
            file_type=row["file_type"],
            size=row["size"],
            mtime=row["mtime"],
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
            indexed_at=row["indexed_at"],
        )

    def upsert_file(
        self,
        path: str,
        file_type: str,
        size: int,
        mtime: float,
        content_hash: str | None,
        embedding_model: str,
        indexed_at: float | None = None,
    ) -> None:
        indexed_at_value = indexed_at or time.time()
        self._conn.execute(
            """
            INSERT INTO files (path, file_type, size, mtime, content_hash, embedding_model, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_type = excluded.file_type,
                size = excluded.size,
                mtime = excluded.mtime,
                content_hash = excluded.content_hash,
                embedding_model = excluded.embedding_model,
                indexed_at = excluded.indexed_at
            """,
            (
                path,
                file_type,
                size,
                mtime,
                content_hash,
                embedding_model,
                indexed_at_value,
            ),
        )
        self._conn.commit()

    def delete_file(self, path: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()

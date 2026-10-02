"""SQLite 存储：图片元数据、向量和已登记的文件夹。"""

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    folder    TEXT NOT NULL,
    size      INTEGER NOT NULL,
    mtime     REAL NOT NULL,
    width     INTEGER,
    height    INTEGER,
    taken_at  REAL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder);
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class ImageRecord:
    id: int
    path: str
    folder: str
    size: int
    mtime: float
    width: int | None
    height: int | None
    taken_at: float | None

    @property
    def date(self) -> float:
        """拍摄时间，没有 EXIF 时退回到文件修改时间。"""
        return self.taken_at if self.taken_at is not None else self.mtime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "name": Path(self.path).name,
            "folder": self.folder,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "date": self.date,
        }


_RECORD_COLUMNS = "id, path, folder, size, mtime, width, height, taken_at"


class Database:
    def __init__(self, path: Path | str):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # meta

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # folders

    def add_folder(self, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (path,))

    def remove_folder(self, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM folders WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM images WHERE folder = ?", (path,))

    def folders(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT path FROM folders ORDER BY path")]

    # images

    def file_states(self, folder: str) -> dict[str, tuple[int, float]]:
        """返回某文件夹下已索引文件的 (size, mtime)，用于增量判断。"""
        rows = self._conn.execute("SELECT path, size, mtime FROM images WHERE folder = ?", (folder,))
        return {path: (size, mtime) for path, size, mtime in rows}

    def upsert_images(self, rows: list[dict]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO images(path, folder, size, mtime, width, height, taken_at, embedding)
                VALUES (:path, :folder, :size, :mtime, :width, :height, :taken_at, :embedding)
                ON CONFLICT(path) DO UPDATE SET
                    folder = excluded.folder, size = excluded.size, mtime = excluded.mtime,
                    width = excluded.width, height = excluded.height,
                    taken_at = excluded.taken_at, embedding = excluded.embedding
                """,
                [{**r, "embedding": np.asarray(r["embedding"], dtype=np.float32).tobytes()} for r in rows],
            )

    def delete_paths(self, paths: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in paths])

    def clear_images(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM images")

    def get(self, image_id: int) -> ImageRecord | None:
        row = self._conn.execute(f"SELECT {_RECORD_COLUMNS} FROM images WHERE id = ?", (image_id,)).fetchone()
        return ImageRecord(*row) if row else None

    def get_many(self, ids: list[int]) -> dict[int, ImageRecord]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(f"SELECT {_RECORD_COLUMNS} FROM images WHERE id IN ({marks})", ids)
        return {r[0]: ImageRecord(*r) for r in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def load_vectors(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """一次性读出全部向量，供内存检索使用：(ids, vectors, folders, dates)。"""
        rows = self._conn.execute("SELECT id, folder, COALESCE(taken_at, mtime), embedding FROM images").fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32), [], np.zeros(0, np.float64)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        folders = [r[1] for r in rows]
        dates = np.fromiter((r[2] for r in rows), dtype=np.float64, count=len(rows))
        vectors = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        return ids, vectors, folders, dates

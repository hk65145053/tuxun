"""内存向量检索。几万张图片的向量只有几十 MB，直接用矩阵乘法做暴力检索即可，毫秒级返回。"""

import threading
from dataclasses import dataclass

import numpy as np

from .db import Database


@dataclass
class Filters:
    folder: str | None = None
    date_from: float | None = None
    date_to: float | None = None


class VectorIndex:
    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.Lock()
        self._dirty = True
        self._ids = np.zeros(0, np.int64)
        self._vectors = np.zeros((0, 0), np.float32)
        self._folders = np.array([], dtype=object)
        self._dates = np.zeros(0, np.float64)

    def invalidate(self) -> None:
        self._dirty = True

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._dirty:
                ids, vectors, folders, dates = self.db.load_vectors()
                self._ids, self._vectors, self._dates = ids, vectors, dates
                self._folders = np.array(folders, dtype=object)
                self._dirty = False

    def _mask(self, filters: Filters | None) -> np.ndarray | None:
        if filters is None:
            return None
        mask = np.ones(len(self._ids), dtype=bool)
        if filters.folder:
            mask &= self._folders == filters.folder
        if filters.date_from is not None:
            mask &= self._dates >= filters.date_from
        if filters.date_to is not None:
            mask &= self._dates <= filters.date_to
        return mask

    def search(
        self, query: np.ndarray, limit: int = 50, offset: int = 0, filters: Filters | None = None,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """按余弦相似度从高到低返回 (图片 id, 分数)。"""
        self._ensure_loaded()
        if len(self._ids) == 0:
            return []
        scores = self._vectors @ query.astype(np.float32)
        mask = self._mask(filters)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        if exclude_id is not None:
            scores = np.where(self._ids == exclude_id, -np.inf, scores)
        k = min(offset + limit, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")][offset:]
        return [(int(self._ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def vector_of(self, image_id: int) -> np.ndarray | None:
        self._ensure_loaded()
        hits = np.nonzero(self._ids == image_id)[0]
        return self._vectors[hits[0]] if len(hits) else None

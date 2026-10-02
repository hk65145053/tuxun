"""内存向量检索。几万张图片的向量只有几十 MB，直接用矩阵乘法做暴力检索即可，毫秒级返回。"""

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .db import Database


@dataclass
class Filters:
    folder: str | None = None
    date_from: float | None = None
    date_to: float | None = None


@dataclass
class _Snapshot:
    ids: np.ndarray
    vectors: np.ndarray
    folders: np.ndarray
    dates: np.ndarray
    floors: dict = field(default_factory=dict)


class VectorIndex:
    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.Lock()
        self._dirty = True
        self._snapshot: _Snapshot | None = None

    def invalidate(self) -> None:
        self._dirty = True

    def _load(self) -> _Snapshot:
        """返回当前向量的快照；同一次查询全程使用同一份快照，不受后台索引刷新影响。"""
        with self._lock:
            if self._dirty or self._snapshot is None:
                ids, vectors, folders, dates = self.db.load_vectors()
                self._snapshot = _Snapshot(ids, vectors, np.array(folders, dtype=object), dates)
                self._dirty = False
            return self._snapshot

    @staticmethod
    def _mask(snap: _Snapshot, filters: Filters | None) -> np.ndarray | None:
        if filters is None:
            return None
        mask = np.ones(len(snap.ids), dtype=bool)
        if filters.folder:
            mask &= snap.folders == filters.folder
        if filters.date_from is not None:
            mask &= snap.dates >= filters.date_from
        if filters.date_to is not None:
            mask &= snap.dates <= filters.date_to
        return mask

    def _floor(self, snap: _Snapshot, references: np.ndarray, rank: int) -> np.ndarray:
        """每张图片上对照词里第 rank 高的分数。同一份快照只算一次。"""
        key = (id(references), rank)
        if key not in snap.floors:
            ref_scores = snap.vectors @ references.astype(np.float32).T
            k = min(rank, ref_scores.shape[1])
            snap.floors[key] = np.partition(ref_scores, -k, axis=1)[:, -k]
        return snap.floors[key]

    def search_combined(
        self,
        queries: dict[str, np.ndarray],
        combine: Callable[[dict[str, np.ndarray]], np.ndarray],
        limit: int = 50,
        offset: int = 0,
        filters: Filters | None = None,
        exclude_id: int | None = None,
        confident: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray] | None = None,
        references: np.ndarray | None = None,
        rank: int = 5,
    ) -> list[tuple[int, float, bool]]:
        """先算出每个查询向量对所有图片的相似度，再用 combine 合成排序分数，返回 (图片 id, 分数, 是否有把握)。

        给了 confident 和 references 时，confident(各词分数, 对照分数线) 为真的图片算“有把握”，
        排在最前面，其余图片跟在后面；两组内部都按分数从高到低。
        """
        snap = self._load()
        if len(snap.ids) == 0:
            return []
        term_scores = {key: snap.vectors @ q.astype(np.float32) for key, q in queries.items()}
        scores = combine(term_scores)
        sure = np.ones(len(scores), dtype=bool)
        if confident is not None and references is not None:
            sure = confident(term_scores, self._floor(snap, references, rank))
        mask = self._mask(snap, filters)
        if exclude_id is not None:
            not_self = snap.ids != exclude_id
            mask = not_self if mask is None else mask & not_self
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        # 分数是余弦相似度或标准化分数，远小于 1e6，加上它就能让“有把握”的一组整体排在前面
        key = scores + np.where(sure, 1e6, 0)
        k = min(offset + limit, len(key))
        if k <= 0:
            return []
        top = np.argpartition(-key, k - 1)[:k]
        top = top[np.argsort(-key[top], kind="stable")][offset:]
        return [(int(snap.ids[i]), float(scores[i]), bool(sure[i])) for i in top if np.isfinite(scores[i])]

    def search(
        self, query: np.ndarray, limit: int = 50, offset: int = 0, filters: Filters | None = None,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float, bool]]:
        """按余弦相似度从高到低返回 (图片 id, 分数, True)。"""
        return self.search_combined({"q": query}, lambda s: s["q"], limit, offset, filters, exclude_id)

    def vector_of(self, image_id: int) -> np.ndarray | None:
        snap = self._load()
        hits = np.nonzero(snap.ids == image_id)[0]
        return snap.vectors[hits[0]] if len(hits) else None

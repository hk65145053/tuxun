"""图库：把数据库、模型、索引任务和检索组合在一起，供网页服务和命令行共用。"""

import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .db import Database, ImageRecord
from .embedder import ChineseClipEmbedder, Embedder
from .indexer import Indexer, IndexProgress
from .query import combine, matches, parse_query, terms_of
from .search import Filters, VectorIndex

log = logging.getLogger(__name__)


class Library:
    def __init__(self, config: Config, embedder_factory: Callable[[], Embedder] | None = None):
        self.config = config
        self.db = Database(config.db_path)
        self.index = VectorIndex(self.db)
        self._embedder_factory = embedder_factory or (lambda: ChineseClipEmbedder(config.model_name))
        self._embedder: Embedder | None = None
        self._embedder_lock = threading.Lock()
        self._references: np.ndarray | None = None
        self._job: threading.Thread | None = None
        self.progress: IndexProgress | None = None
        self.last_error: str | None = None

    @property
    def embedder(self) -> Embedder:
        """模型加载较慢，第一次用到时才加载。"""
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            return self._embedder

    @property
    def model_ready(self) -> bool:
        return self._embedder is not None

    def preload(self) -> None:
        """在后台线程里加载模型，让第一次搜索不用等。"""

        def job() -> None:
            try:
                self.embedder
                self.references()
            except Exception:
                log.exception("模型加载失败")

        threading.Thread(target=job, daemon=True).start()

    def references(self) -> np.ndarray | None:
        """对照词的向量；模型没有提供对照词时返回 None，此时不做相关度过滤。"""
        embedder = self.embedder
        terms = getattr(embedder, "reference_terms", None)
        if not terms:
            return None
        with self._embedder_lock:
            if self._references is None:
                self._references = np.stack([embedder.embed_text(t) for t in terms])
            return self._references

    # 文件夹

    def add_folder(self, path: str) -> str:
        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"文件夹不存在：{folder}")
        for existing in map(Path, self.db.folders()):
            if folder == existing or existing in folder.parents:
                raise ValueError(f"已包含在 {existing} 中")
            if folder in existing.parents:
                raise ValueError(f"{existing} 已登记，请先移除它再添加上级文件夹")
        self.db.add_folder(str(folder))
        return str(folder)

    def remove_folder(self, path: str) -> None:
        self.db.remove_folder(path)
        self.index.invalidate()

    def folders(self) -> list[str]:
        return self.db.folders()

    # 索引

    @property
    def indexing(self) -> bool:
        return self._job is not None and self._job.is_alive()

    def run_index(self, on_progress: Callable[[IndexProgress], None] | None = None) -> IndexProgress:
        def report(p: IndexProgress) -> None:
            self.progress = p
            if on_progress:
                on_progress(p)

        indexer = Indexer(self.db, self.embedder, batch_size=self.config.batch_size)
        try:
            return indexer.index_folders(self.folders(), report)
        finally:
            self.index.invalidate()

    def start_index(self) -> bool:
        """在后台线程中建立索引；已有任务在跑时返回 False。"""
        if self.indexing:
            return False
        self.last_error = None
        self.progress = IndexProgress()

        def job() -> None:
            try:
                self.run_index()
            except Exception as e:
                log.exception("索引失败")
                self.last_error = str(e)

        self._job = threading.Thread(target=job, daemon=True)
        self._job.start()
        return True

    # 检索

    def _records(self, hits: list[tuple[int, float, bool]]) -> list[dict]:
        records = self.db.get_many([i for i, _, _ in hits])
        return [
            {**records[i].to_dict(), "score": round(s, 4), "match": sure} for i, s, sure in hits if i in records
        ]

    def search_text(self, text: str, limit: int = 50, offset: int = 0, filters: Filters | None = None) -> list[dict]:
        """支持多关键词语法，见 query.py。查询为空时抛出 ValueError。

        有把握“图里真有”要找内容的图片（见 query.matches）标 match=True 并排在前面，
        其余图片按相似度跟在后面，标 match=False。
        """
        clauses = parse_query(text)
        vectors = {term: self.embedder.embed_text(term) for term in terms_of(clauses)}
        hits = self.index.search_combined(
            vectors, lambda s: combine(clauses, s), limit, offset, filters,
            confident=lambda s, floor: matches(clauses, s, floor), references=self.references(),
        )
        return self._records(hits)

    def search_similar(
        self, image_id: int, limit: int = 50, offset: int = 0, filters: Filters | None = None
    ) -> list[dict]:
        query = self.index.vector_of(image_id)
        if query is None:
            return []
        return self._records(self.index.search(query, limit, offset, filters, exclude_id=image_id))

    def get(self, image_id: int) -> ImageRecord | None:
        return self.db.get(image_id)

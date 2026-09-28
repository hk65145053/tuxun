"""扫描文件夹并增量建立向量索引：只处理新增或修改过的图片，删除已不存在的记录。"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageOps

from .config import IMAGE_EXTENSIONS
from .db import Database
from .embedder import Embedder

log = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

# 模型输入只有 224 像素，解码时按这个尺寸缩小能大幅加快 JPEG 读取
_DECODE_SIZE = (448, 448)
_EXIF_DATETIME_ORIGINAL = 0x9003
_EXIF_IFD = 0x8769


@dataclass
class IndexProgress:
    total: int = 0
    done: int = 0
    added: int = 0
    removed: int = 0
    failed: list[str] = field(default_factory=list)


@dataclass
class LoadedImage:
    path: str
    size: int
    mtime: float
    image: Image.Image
    width: int
    height: int
    taken_at: float | None


def iter_image_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(dirpath) / name


def _taken_at(img: Image.Image) -> float | None:
    try:
        value = img.getexif().get_ifd(_EXIF_IFD).get(_EXIF_DATETIME_ORIGINAL)
        if value:
            return datetime.strptime(str(value).strip("\x00 "), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return None


def load_image(path: str, size: int, mtime: float) -> LoadedImage:
    with Image.open(path) as img:
        width, height = img.size
        taken_at = _taken_at(img)
        img.draft("RGB", _DECODE_SIZE)
        img = ImageOps.exif_transpose(img).convert("RGB")
    return LoadedImage(path, size, mtime, img, width, height, taken_at)


class Indexer:
    def __init__(self, db: Database, embedder: Embedder, batch_size: int = 32, workers: int = 8):
        self.db = db
        self.embedder = embedder
        self.batch_size = batch_size
        self.workers = workers

    def _check_model(self) -> None:
        """换了模型后旧向量不能再用，需要全部重建。"""
        stored = self.db.get_meta("model")
        if stored != self.embedder.name:
            if stored is not None:
                log.warning("模型从 %s 变为 %s，清空旧索引", stored, self.embedder.name)
                self.db.clear_images()
            self.db.set_meta("model", self.embedder.name)

    def index_folders(
        self, folders: list[str], on_progress: Callable[[IndexProgress], None] | None = None
    ) -> IndexProgress:
        self._check_model()
        progress = IndexProgress()
        todo: list[tuple[str, str, int, float]] = []
        for folder in folders:
            known = self.db.file_states(folder)
            seen: set[str] = set()
            for file in iter_image_files(Path(folder)):
                path = str(file)
                try:
                    st = file.stat()
                except OSError:
                    continue
                seen.add(path)
                if known.get(path) != (st.st_size, st.st_mtime):
                    todo.append((folder, path, st.st_size, st.st_mtime))
            gone = [p for p in known if p not in seen]
            self.db.delete_paths(gone)
            progress.removed += len(gone)

        progress.total = len(todo)
        if on_progress:
            on_progress(progress)

        with ThreadPoolExecutor(self.workers) as pool:
            for start in range(0, len(todo), self.batch_size):
                batch = todo[start : start + self.batch_size]
                futures = [pool.submit(load_image, path, size, mtime) for _, path, size, mtime in batch]
                loaded: list[tuple[str, LoadedImage]] = []
                for (folder, path, _, _), future in zip(batch, futures):
                    try:
                        loaded.append((folder, future.result()))
                    except Exception as e:
                        log.warning("无法读取 %s: %s", path, e)
                        progress.failed.append(path)
                if loaded:
                    vectors = self.embedder.embed_images([item.image for _, item in loaded])
                    self.db.upsert_images(
                        [
                            {
                                "path": item.path,
                                "folder": folder,
                                "size": item.size,
                                "mtime": item.mtime,
                                "width": item.width,
                                "height": item.height,
                                "taken_at": item.taken_at,
                                "embedding": vec,
                            }
                            for (folder, item), vec in zip(loaded, vectors)
                        ]
                    )
                    progress.added += len(loaded)
                progress.done += len(batch)
                if on_progress:
                    on_progress(progress)
        return progress

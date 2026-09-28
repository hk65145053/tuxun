"""本地网页服务。默认只监听 127.0.0.1，只会返回已索引的图片文件。"""

import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from .library import Library
from .search import Filters

STATIC_DIR = Path(__file__).parent / "static"
THUMB_SIZE = (360, 360)


class FolderBody(BaseModel):
    path: str


def _parse_date(value: str | None, end_of_day: bool = False) -> float | None:
    if not value:
        return None
    try:
        day = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, f"日期格式应为 YYYY-MM-DD：{value}")
    if end_of_day:
        day = day.replace(hour=23, minute=59, second=59)
    return day.timestamp()


def create_app(library: Library) -> FastAPI:
    app = FastAPI(title="图寻 Tuxun")

    def filters(folder: str | None, date_from: str | None, date_to: str | None) -> Filters:
        return Filters(folder or None, _parse_date(date_from), _parse_date(date_to, end_of_day=True))

    def record_or_404(image_id: int):
        record = library.get(image_id)
        if record is None or not Path(record.path).is_file():
            raise HTTPException(404, "图片不存在")
        return record

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        p = library.progress
        return {
            "count": library.db.count(),
            "folders": library.folders(),
            "indexing": library.indexing,
            "progress": None if p is None else {
                "total": p.total, "done": p.done, "added": p.added, "removed": p.removed, "failed": len(p.failed),
            },
            "error": library.last_error,
        }

    @app.post("/api/folders")
    def add_folder(body: FolderBody) -> dict:
        try:
            return {"path": library.add_folder(body.path)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/folders")
    def remove_folder(path: str) -> dict:
        library.remove_folder(path)
        return {"ok": True}

    @app.post("/api/index")
    def start_index() -> dict:
        return {"started": library.start_index()}

    @app.get("/api/search")
    def search(
        q: str = Query(..., min_length=1),
        limit: int = Query(60, ge=1, le=500),
        offset: int = Query(0, ge=0),
        folder: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict:
        return {"results": library.search_text(q, limit, offset, filters(folder, date_from, date_to))}

    @app.get("/api/similar/{image_id}")
    def similar(
        image_id: int,
        limit: int = Query(60, ge=1, le=500),
        offset: int = Query(0, ge=0),
        folder: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict:
        record_or_404(image_id)
        return {"results": library.search_similar(image_id, limit, offset, filters(folder, date_from, date_to))}

    @app.get("/image/{image_id}")
    def image(image_id: int) -> FileResponse:
        return FileResponse(record_or_404(image_id).path)

    @app.get("/thumb/{image_id}")
    def thumb(image_id: int) -> Response:
        record = record_or_404(image_id)
        cache = library.config.thumb_dir / f"{image_id}-{int(record.mtime)}.jpg"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(record.path) as img:
                    img.draft("RGB", THUMB_SIZE)
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img.thumbnail(THUMB_SIZE)
                    tmp = cache.with_suffix(f".{threading.get_ident()}.tmp")
                    img.save(tmp, "JPEG", quality=85)
                    tmp.replace(cache)
            except Exception:
                raise HTTPException(500, "无法生成缩略图")
        return FileResponse(cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    return app

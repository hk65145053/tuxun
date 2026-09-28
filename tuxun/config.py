import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class Config:
    home: Path
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32

    @property
    def db_path(self) -> Path:
        return self.home / "tuxun.db"

    @property
    def thumb_dir(self) -> Path:
        return self.home / "thumbs"

    @classmethod
    def load(cls) -> "Config":
        home = Path(os.environ.get("TUXUN_HOME", Path.home() / ".tuxun")).expanduser()
        model = os.environ.get("TUXUN_MODEL", DEFAULT_MODEL)
        home.mkdir(parents=True, exist_ok=True)
        return cls(home=home, model_name=model)

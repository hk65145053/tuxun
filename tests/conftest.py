import numpy as np
import pytest
from PIL import Image

from tuxun.config import Config
from tuxun.library import Library

COLORS = {"红": (255, 0, 0), "绿": (0, 255, 0), "蓝": (0, 0, 255)}


class ColorEmbedder:
    """测试用的假模型：图片向量是平均颜色，文字向量是颜色名对应的 RGB。"""

    name = "fake-color"

    def __init__(self):
        self.image_calls = 0

    def embed_images(self, images):
        self.image_calls += len(images)
        v = np.array([np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(0) for img in images]) + 1
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def embed_text(self, text):
        v = np.array(COLORS[text], dtype=np.float32) + 1
        return v / np.linalg.norm(v)


def make_image(path, color, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def photos(tmp_path):
    root = tmp_path / "photos"
    make_image(root / "red.jpg", (250, 10, 10))
    make_image(root / "sub" / "green.png", (10, 250, 10))
    make_image(root / "blue.jpg", (10, 10, 250))
    (root / "notes.txt").write_text("not an image")
    (root / "broken.jpg").write_bytes(b"not really a jpeg")
    return root


@pytest.fixture
def embedder():
    return ColorEmbedder()


@pytest.fixture
def library(tmp_path, embedder):
    home = tmp_path / "home"
    home.mkdir()
    lib = Library(Config(home=home), embedder_factory=lambda: embedder)
    yield lib
    lib.db.close()

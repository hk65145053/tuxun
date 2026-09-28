"""把图片和文字编码到同一个向量空间。默认使用 Chinese-CLIP，支持中文检索。"""

from typing import Protocol

import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL


class Embedder(Protocol):
    name: str

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """返回形如 (n, dim) 的 L2 归一化 float32 向量。"""

    def embed_text(self, text: str) -> np.ndarray:
        """返回形如 (dim,) 的 L2 归一化 float32 向量。"""


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class ChineseClipEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.name = model_name
        self.device = device
        self._torch = torch
        self.model = ChineseCLIPModel.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)

    def _features(self, output) -> np.ndarray:
        # 新旧版本 transformers 的返回类型不同：可能是张量，也可能是带 pooler_output 的输出对象
        if not isinstance(output, self._torch.Tensor):
            output = output.pooler_output
        return _normalize(output.float().cpu().numpy())

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        with self._torch.no_grad():
            return self._features(self.model.get_image_features(pixel_values=pixel_values))

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], padding=True, truncation=True, max_length=52, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            return self._features(self.model.get_text_features(**inputs))[0]

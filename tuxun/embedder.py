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


# 判断“图里是否真的有这个东西”时用来对照的常见词。模型给出的相似度没有绝对含义，
# 但在同一张图上，真正相关的词应该比大多数常见词得分更高。
REFERENCE_TERMS = (
    "人 一群人 小孩 动物 猫 狗 鸟 花 树 草地 森林 山 海 湖 河 天空 云 太阳 月亮 星空 雪 雨 "
    "城市 街道 建筑 房子 古建筑 寺庙 桥 汽车 自行车 火车 飞机 船 食物 水果 饮料 室内 房间 桌子 "
    "电脑 手机 书 文字 地图 图表 截图 海报 标志 夜景 日落 灯光 沙滩 沙漠 田野 公园 广场 商店 "
    "衣服 运动 照片 风景 艺术 画 卡通"
).split()


class ChineseClipEmbedder:
    reference_terms = REFERENCE_TERMS

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from huggingface_hub import constants, try_to_load_from_cache
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

        # 模型已下载时切到离线模式直接读本地文件。否则 transformers 会联网检查每个文件，
        # 在国内网络下要一两分钟；只传 local_files_only 挡不住全部检查
        was_offline = constants.HF_HUB_OFFLINE
        if isinstance(try_to_load_from_cache(model_name, "config.json"), str):
            constants.HF_HUB_OFFLINE = True
        try:
            model = ChineseCLIPModel.from_pretrained(model_name)
            processor = ChineseCLIPProcessor.from_pretrained(model_name)
        except OSError:
            if was_offline or not constants.HF_HUB_OFFLINE:
                raise
            constants.HF_HUB_OFFLINE = False  # 本地文件不全，联网补齐
            model = ChineseCLIPModel.from_pretrained(model_name)
            processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model = model.to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        self.processor = processor

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

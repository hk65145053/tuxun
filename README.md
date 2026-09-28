# 图寻 Tuxun

用文字搜索本地图片内容的图片管理工具。输入“云”找出有云的照片，输入“月亮”找出拍到月亮的照片，也可以写一句话，比如“海边日落时的小狗”。

- **完全离线**：模型在本机运行，图片不会上传到任何地方
- **中文检索**：使用 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)，把图片和文字编码到同一个向量空间，按相似度排序，不需要事先打标签
- **增量索引**：只处理新增或修改过的图片，删除的图片会自动从索引中移除
- **以图搜图**：在任意一张图片上点“找相似”
- **过滤**：按文件夹、拍摄日期（无 EXIF 时用文件修改时间）筛选

## 安装

需要 Python 3.10 以上。有 NVIDIA 显卡的话，先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装带 CUDA 的 torch，再安装图寻：

```bash
pip install -e .            # 如需读取 iPhone 的 HEIC 照片：pip install -e ".[heic]"
```

第一次使用时会从 Hugging Face 自动下载模型（约 750 MB）。国内网络下载慢的话，可以先设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 使用

```bash
tuxun serve                 # 打开 http://127.0.0.1:8765
```

在网页右上角点“图库”，添加图片文件夹，再点“更新索引”。索引完成后在搜索框输入文字即可。

### 多个关键词

| 写法 | 含义 | 例子 |
| --- | --- | --- |
| 空格 | 并且：每个词都要匹配 | `云 月亮` |
| `\|` | 或者：匹配任意一组 | `猫 \| 狗` |
| `-词` | 排除：去掉明显含有它的图片 | `月亮 -云` |
| `"..."` | 把含空格的内容当成一个词 | `"a dog on the beach"` |

只有一个词或一句话时（例如“夜空中被云遮住一半的月亮”），按整句理解，这通常是效果最好的写法。多个词时每个词单独打分，先在整个图库上标准化，再取“并且”的最低分、“或者”的最高分。模型的分数没有绝对含义，所以这些语法改变的是排序，不是严格过滤。

命令行也能完成同样的事：

```bash
tuxun add ~/Pictures D:/照片
tuxun index
tuxun search 月亮
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TUXUN_HOME` | `~/.tuxun` | 索引数据库和缩略图缓存的位置 |
| `TUXUN_MODEL` | `OFA-Sys/chinese-clip-vit-base-patch16` | 可换成 `chinese-clip-vit-large-patch14` 等更大的模型，效果更好但更慢；换模型后会自动重建索引 |

## 实现

| 模块 | 作用 |
| --- | --- |
| `tuxun/embedder.py` | 加载 Chinese-CLIP，把图片和文字转成归一化向量；有显卡时自动用 GPU 和半精度 |
| `tuxun/indexer.py` | 扫描文件夹，多线程解码图片，按批送进模型，结果写入 SQLite |
| `tuxun/query.py` | 解析多关键词语法，把每个词的分数合成排序分数 |
| `tuxun/search.py` | 把全部向量读进内存做矩阵乘法检索。几万张图片只占几十 MB，单次查询是毫秒级 |
| `tuxun/server.py` | FastAPI 网页服务，默认只监听本机，只返回已索引的图片 |
| `tuxun/static/index.html` | 网页界面 |

## 开发

```bash
pip install -e ".[dev]"
pytest
```

测试使用一个按颜色编码的假模型，不需要下载 Chinese-CLIP。

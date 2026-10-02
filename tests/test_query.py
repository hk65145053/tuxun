from pathlib import Path

import numpy as np
import pytest

from tuxun.query import Clause, combine, matches, parse_query, terms_of
from tests.conftest import COLORS, make_image


def test_parse_query():
    assert parse_query("云") == [Clause(["云"])]
    assert parse_query("  云   月亮 ") == [Clause(["云", "月亮"])]
    assert parse_query("猫|狗 草地") == [Clause(["猫"]), Clause(["狗", "草地"])]
    assert parse_query("月亮 -云") == [Clause(["月亮"], ["云"])]
    assert parse_query('"a dog on the beach" -"a cat"') == [Clause(["a dog on the beach"], ["a cat"])]
    assert parse_query("雨-天") == [Clause(["雨-天"])]
    assert terms_of(parse_query("猫 | 猫 -狗")) == ["猫", "狗"]
    for empty in ["", "   ", "|", "- | -", '""']:
        with pytest.raises(ValueError):
            parse_query(empty)


def test_single_term_keeps_raw_similarity():
    scores = np.array([0.1, 0.3, 0.2], dtype=np.float32)
    assert combine(parse_query("云"), {"云": scores}) is scores


def test_and_uses_weakest_term_after_normalizing():
    # “云”的原始分数整体偏高，不标准化的话取最小值会只看“月亮”
    scores = {
        "云": np.array([0.90, 0.95, 0.80, 0.82], dtype=np.float32),
        "月亮": np.array([0.10, 0.30, 0.31, 0.12], dtype=np.float32),
    }
    ranked = np.argsort(-combine(parse_query("云 月亮"), scores))
    assert ranked[0] == 1


def names(results):
    return [Path(r["path"]).name for r in results]


@pytest.fixture
def colors(library, tmp_path):
    root = tmp_path / "colors"
    for name, color in {
        "red": (250, 10, 10), "green": (10, 250, 10), "blue": (10, 10, 250),
        "yellow": (250, 250, 10), "cyan": (10, 250, 250),
    }.items():
        make_image(root / f"{name}.png", color)
    library.add_folder(str(root))
    library.run_index()
    return library


def test_and_or_exclude_end_to_end(colors):
    assert names(colors.search_text("红 绿", limit=1)) == ["yellow.png"]
    assert set(names(colors.search_text("红 | 蓝", limit=2))) == {"red.png", "blue.png"}
    # 黄色和青色都含绿，与“绿”的相似度相同；排除红色后青色应排在黄色前面
    assert names(colors.search_text("绿 -红", limit=3)) == ["green.png", "cyan.png", "yellow.png"]


def test_matches_requires_beating_reference_floor():
    floor = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    scores = {
        "云": np.array([0.6, 0.6, 0.4], dtype=np.float32),
        "月亮": np.array([0.6, 0.4, 0.6], dtype=np.float32),
    }
    assert matches(parse_query("云"), scores, floor).tolist() == [True, True, False]
    assert matches(parse_query("云 月亮"), scores, floor).tolist() == [True, False, False]
    assert matches(parse_query("云 | 月亮"), scores, floor).tolist() == [True, True, True]
    assert matches(parse_query("-云"), scores, floor).tolist() == [True, True, True]


def test_confident_matches_come_first(colors, embedder):
    embedder.reference_terms = list(COLORS)
    results = colors.search_text("红", limit=10)
    assert len(results) == 5
    # 有把握的一组排在前面，其余图片跟在后面；青色图片里没有红色，不算有把握
    flags = [r["match"] for r in results]
    assert flags == sorted(flags, reverse=True) and flags[0] and not flags[-1]
    assert names(results)[0] == "red.png" and names(results)[-1] == "cyan.png"
    # 分页时两组的顺序保持不变
    assert names(colors.search_text("红", limit=2, offset=1)) == names(results)[1:3]

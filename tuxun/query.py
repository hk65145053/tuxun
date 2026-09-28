"""多关键词查询语法。

- 空格表示“并且”：云 月亮
- 竖线表示“或者”：猫 | 狗
- 减号表示“排除”：月亮 -云
- 双引号把含空格的内容当成一个词："a dog on the beach"
- 只有一个词时按整句话理解，和不用语法时完全一样
"""

import re
from dataclasses import dataclass, field

import numpy as np

_TOKEN = re.compile(r'(-?)"([^"]*)"|(\S+)')


@dataclass
class Clause:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


def parse_query(text: str) -> list[Clause]:
    """解析成若干个“或者”分支，每个分支内是“并且”关系。没有有效词时抛出 ValueError。"""
    clauses: list[Clause] = []
    for part in text.split("|"):
        clause = Clause()
        for match in _TOKEN.finditer(part):
            negated, quoted, bare = match.groups()
            if quoted is not None:
                term = quoted.strip()
            else:
                negated = "-" if bare.startswith("-") else ""
                term = bare.lstrip("-")
            if term:
                (clause.exclude if negated else clause.include).append(term)
        if clause.include or clause.exclude:
            clauses.append(clause)
    if not clauses:
        raise ValueError("搜索内容为空")
    return clauses


def terms_of(clauses: list[Clause]) -> list[str]:
    seen: dict[str, None] = {}
    for clause in clauses:
        for term in clause.include + clause.exclude:
            seen.setdefault(term)
    return list(seen)


def is_simple(clauses: list[Clause]) -> bool:
    return len(clauses) == 1 and not clauses[0].exclude and len(clauses[0].include) == 1


def combine(clauses: list[Clause], term_scores: dict[str, np.ndarray]) -> np.ndarray:
    """把每个词的相似度合成一个排序分数。

    不同词的原始分数基线不同（有的词和所有图片都比较像），直接比较会偏向某个词，
    所以先在整个图库上做标准化，再取“并且”的最小值、“或者”的最大值，
    排除词只惩罚明显比平均水平更像它的图片。
    """
    if is_simple(clauses):
        return term_scores[clauses[0].include[0]]

    z = {}
    for term, s in term_scores.items():
        std = s.std()
        z[term] = (s - s.mean()) / std if std > 0 else np.zeros_like(s)

    n = len(next(iter(term_scores.values())))
    result = np.full(n, -np.inf, dtype=np.float32)
    for clause in clauses:
        score = np.zeros(n, dtype=np.float32)
        if clause.include:
            score = np.min([z[t] for t in clause.include], axis=0)
        if clause.exclude:
            score = score - np.clip(np.max([z[t] for t in clause.exclude], axis=0), 0, None)
        result = np.maximum(result, score)
    return result

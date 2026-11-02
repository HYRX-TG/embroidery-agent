"""BM25 关键词检索（jieba 中文分词）。"""
from __future__ import annotations

import math
from collections import Counter

import jieba


def tokenize(text: str) -> list[str]:
    """中文分词 + 小写归一。"""
    return [token.lower() for token in jieba.lcut(text) if token.strip()]


class BM25Index:
    """标准 BM25 实现，零外部依赖（仅 jieba）。"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_token_counts: list[Counter] = [Counter(tokenize(doc)) for doc in documents]
        self.doc_lengths = [sum(c.values()) for c in self.doc_token_counts]
        self.avg_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for counter in self.doc_token_counts:
            self.document_frequency.update(counter.keys())
        self.doc_count = len(documents)

    def _idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """返回 query 相对每个文档的 BM25 分数列表。"""
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * self.doc_count
        scores = [0.0] * self.doc_count
        for term in set(query_terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, counter in enumerate(self.doc_token_counts):
                tf = counter.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_length)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(i, score) for i, score in ranked[:top_k] if score > 0]

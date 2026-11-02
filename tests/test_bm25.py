"""BM25 检索单元测试：纯本地、无网络依赖（CI 可用）。"""
from embroidery_agent.retrieval.bm25 import BM25Index


def test_bm25_ranks_relevant_first():
    texts = [
        "苏绣以精细雅洁著称，是四大名绣之首",
        "打籽绣每针绕线成小籽点，立体感强",
        "手工绣由绣娘一针一线绣制，工期长成本高",
        "刺绣保养要防潮防蛀避免暴晒",
    ]
    index = BM25Index(texts)
    hits = index.search("苏绣的特点", top_k=2)
    assert hits and hits[0][0] == 0


def test_bm25_empty_corpus():
    index = BM25Index([])
    assert index.search("苏绣", top_k=3) == []


def test_bm25_top_k_bounds():
    texts = ["一条", "两条", "三条"]
    index = BM25Index(texts)
    hits = index.search("一条", top_k=10)
    assert len(hits) <= 3

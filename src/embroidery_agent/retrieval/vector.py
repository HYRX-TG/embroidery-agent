"""向量检索：FAISS 索引 + Ollama/bge embedding。

设计要点（参考 private-gpt 的解耦思路）：
- Embedding 通过 OpenAI 兼容接口调用 Ollama（/v1/embeddings），
  后续换成任何 OpenAI 兼容 embedding 服务零改动。
- 向量库用 FAISS（faiss-cpu 即可，8GB 显存方案 GPU 不必参与向量索引）。
"""
from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


class EmbeddingClient:
    """通过 OpenAI 兼容接口调用本地 embedding 模型（Ollama）。"""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _call(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                # OpenAI 兼容格式
                return [item["embedding"] for item in data["data"]]
        except Exception:
            pass
        # Ollama 原生 /api/embed（batch 格式）兜底
        url = f"{self.base_url.replace('/v1', '')}/api/embed"
        payload = {"model": self.model, "input": texts}
        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["embeddings"]

    def encode(self, text: str) -> list[float]:
        return self._call([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        return np.asarray(self._call(texts), dtype="float32")


class FaissVectorStore:
    """FAISS 内嵌向量库（内存 + 磁盘持久化）。"""

    def __init__(self, dimension: int):
        if not FAISS_AVAILABLE:
            raise RuntimeError("需要安装 faiss-cpu：pip install 'embroidery-agent[retrieval]'")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        self.ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        faiss.normalize_L2(vectors)  # 内积 + L2 归一化 = 余弦相似度
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        if self.size == 0:
            return []
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(top_k, self.size))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.ids[int(idx)], float(score)))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        (path.with_suffix(".ids.txt")).write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"索引不存在：{path}")
        index = faiss.read_index(str(path))
        ids = (path.with_suffix(".ids.txt")).read_text(encoding="utf-8").splitlines()
        store = cls(index.d)
        store.index = index
        store.ids = ids
        return store

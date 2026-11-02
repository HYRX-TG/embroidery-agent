"""检索管线：文档加载 → 切分 → 建库/检索 → BM25+向量 混合 → Rerank 精排 → 权限过滤。

权限核心：检索前按角色过滤 chunk 的 visibility（public 对客户可见，
public+internal 仅员工可见），这是双角色最硬的权限边界。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core.schema import KnowledgeChunk, Role
from .bm25 import BM25Index
from .reranker import Reranker
from .vector import EmbeddingClient, FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    chunk: KnowledgeChunk
    score: float
    sources: list[str] = field(default_factory=list)


def split_text(text: str, chunk_size: int = 300, overlap: int = 40) -> list[str]:
    """按段落优先、固定长度兜底的结构化切块。"""
    text = re.sub(r"\r\n", "\n", text).strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) <= chunk_size:
            buffer += paragraph + "\n"
            continue
        if buffer.strip():
            chunks.append(buffer.strip())
        # 超长段落内部按长度切
        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size - overlap:]
        buffer = paragraph + "\n"
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def load_kb_documents(kb_dir: Path) -> list[KnowledgeChunk]:
    """读取知识库目录下所有 .md/.txt，生成切块。文件名前缀决定 category/visibility。"""
    chunks: list[KnowledgeChunk] = []
    if not kb_dir.exists():
        return chunks
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue
        category = path.stem
        visibility = "internal" if "internal" in path.parts else "public"
        text = path.read_text(encoding="utf-8")
        for i, chunk_text in enumerate(split_text(text)):
            doc_id = hashlib.md5(f"{path.name}:{i}".encode("utf-8")).hexdigest()[:12]
            chunks.append(
                KnowledgeChunk(
                    id=doc_id,
                    text=chunk_text,
                    source=path.name,
                    category=category,
                    visibility=visibility,
                    metadata={"chunk_index": i},
                )
            )
    return chunks


def reciprocal_rank_fusion(results: dict[str, list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for source, items in results.items():
        for i, (doc_id, _) in enumerate(sorted(items, key=lambda x: x[1], reverse=True)):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return fused


class KnowledgeBase:
    """构建与加载知识库索引。"""

    def __init__(
        self,
        public_path: str | Path,
        internal_path: str | Path,
        index_path: str | Path,
        embedding: EmbeddingClient,
        chunk_size: int = 300,
        chunk_overlap: int = 40,
    ):
        self.public_path = Path(public_path)
        self.internal_path = Path(internal_path)
        self.index_path = Path(index_path)
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: dict[str, KnowledgeChunk] = {}
        self.bm25: BM25Index | None = None
        self.vector_store: FaissVectorStore | None = None
        self._dimension = 0

    def build(self) -> dict:
        """构建索引：读取文档 → 切块 → BM25 → FAISS。"""
        public_chunks = load_kb_documents(self.public_path)
        internal_chunks = load_kb_documents(self.internal_path)
        all_chunks = public_chunks + internal_chunks
        self.chunks = {chunk.id: chunk for chunk in all_chunks}

        texts = [chunk.text for chunk in all_chunks]
        self.bm25 = BM25Index(texts)

        if texts:
            try:
                vectors = self.embedding.encode_batch(texts)
                self._dimension = vectors.shape[1]
                self.vector_store = FaissVectorStore(self._dimension)
                self.vector_store.add([chunk.id for chunk in all_chunks], vectors)
                self.vector_store.save(self.index_path / "index.faiss")
            except Exception as exc:
                # Embedding 服务（Ollama）不可用时降级为仅 BM25 检索，
                # 保证无模型环境下仍可构建知识库并演示检索链路。
                self.vector_store = None
                self._dimension = 0
                self._build_warning = f"向量索引构建失败，已降级为仅 BM25：{exc}"
        else:
            self.vector_store = None

        # 记录源文件清单（mtime+size），load 时据此判断是否需要重建
        self._write_manifest()

        return {
            "public_chunks": len(public_chunks),
            "internal_chunks": len(internal_chunks),
            "total_chunks": len(all_chunks),
            "dimension": self._dimension,
            "vector_ok": self.vector_store is not None,
            "warning": getattr(self, "_build_warning", None),
            "index_path": str(self.index_path),
        }

    # ------------------------------------------------------------------
    # 源文件变更检测（manifest 清单：模型 + 每个源文件 mtime/size）
    # ------------------------------------------------------------------
    def _source_files(self) -> list[Path]:
        files: list[Path] = []
        for d in (self.public_path, self.internal_path):
            if not d.exists():
                continue
            files.extend(
                p for p in sorted(d.rglob("*"))
                if p.is_file() and p.suffix.lower() in (".md", ".txt")
            )
        return files

    def _manifest(self) -> dict:
        return {
            "model": self.embedding.model,
            "files": {
                str(p.resolve()): [p.stat().st_mtime_ns, p.stat().st_size]
                for p in self._source_files()
            },
        }

    def _manifest_path(self) -> Path:
        return self.index_path / "manifest.json"

    def _write_manifest(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self._manifest_path()).write_text(
            json.dumps(self._manifest(), ensure_ascii=False), encoding="utf-8"
        )

    def _kb_stale(self) -> bool:
        """知识库源文件或 embedding 模型变更 → 需要重建索引。

        判定条件：索引缺失、manifest 缺失、或任一源文件的 mtime/size 与清单不一致。
        """
        if not (self.index_path / "index.faiss").exists():
            return True
        try:
            saved = json.loads(self._manifest_path().read_text(encoding="utf-8"))
        except Exception:
            return True
        return saved != self._manifest()

    def load(self) -> dict:
        """加载已构建索引；源文件变更时自动重建（rebuilt=True）。"""
        if self._kb_stale():
            stats = self.build()
            return {**stats, "rebuilt": True, "reason": "kb_stale"}
        if self.bm25 is None:
            public_chunks = load_kb_documents(self.public_path)
            internal_chunks = load_kb_documents(self.internal_path)
            all_chunks = public_chunks + internal_chunks
            self.chunks = {chunk.id: chunk for chunk in all_chunks}
            self.bm25 = BM25Index([chunk.text for chunk in all_chunks])
        if self.vector_store is None:
            try:
                self.vector_store = FaissVectorStore.load(self.index_path / "index.faiss")
            except FileNotFoundError:
                self.vector_store = None
        return {
            "total_chunks": len(self.chunks),
            "vector_loaded": self.vector_store is not None,
            "rebuilt": False,
        }


class HybridRetriever:
    """混合检索：BM25 + 向量 → RRF 融合 → Rerank 精排 → 权限过滤。"""

    def __init__(
        self,
        kb: KnowledgeBase,
        reranker: Reranker | None = None,
        recall_k: int = 20,
        top_k: int = 5,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_rerank: bool = True,
    ):
        self.kb = kb
        self.reranker = reranker
        self.recall_k = recall_k
        self.top_k = top_k
        self.use_bm25 = use_bm25
        self.use_vector = use_vector
        self.use_rerank = use_rerank

    def search(self, query: str, role: Role = Role.CUSTOMER) -> list[RetrievalHit]:
        kb = self.kb
        if kb.bm25 is None or not kb.chunks:
            return []

        allowed = {"public"} if role == Role.CUSTOMER else {"public", "internal"}
        id_to_index = {chunk.id: i for i, chunk in enumerate(kb.chunks.values()) if chunk.visibility in allowed}
        index_to_id = {v: k for k, v in id_to_index.items()}

        results: dict[str, list[tuple[str, float]]] = {}
        if self.use_bm25 and kb.bm25 is not None:
            bm25_hits = kb.bm25.search(query, self.recall_k)
            results["bm25"] = [
                (index_to_id[i], score) for i, score in bm25_hits if i in index_to_id
            ]

        if self.use_vector and kb.vector_store is not None:
            try:
                query_vector = kb.embedding.encode(query)
                vector_hits = kb.vector_store.search(query_vector, self.recall_k)
                results["vector"] = [
                    (doc_id, score) for doc_id, score in vector_hits if doc_id in id_to_index
                ]
            except Exception as exc:
                logger.warning("vector_search_failed error=%s（本次查询降级为仅 BM25）", exc)
                results.setdefault("vector", [])

        if not results:
            return []

        fused = reciprocal_rank_fusion(results)
        ranked_ids = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: self.recall_k]
        candidates = [
            (doc_id, kb.chunks[doc_id], fused[doc_id])
            for doc_id, _ in ranked_ids
            if doc_id in kb.chunks
        ]

        if self.use_rerank and self.reranker is not None and self.reranker.is_ready():
            reranked = self.reranker.rerank(
                query,
                [(doc_id, chunk.text, score) for doc_id, chunk, score in candidates],
                top_k=self.top_k,
            )
            selected = [(doc_id, kb.chunks[doc_id], score) for doc_id, _, score in reranked]
        else:
            selected = [(doc_id, chunk, score) for doc_id, chunk, score in candidates[: self.top_k]]

        return [
            RetrievalHit(
                chunk=chunk,
                score=score,
                sources=sorted({source for source, items in results.items() if doc_id in {d for d, _ in items}}),
            )
            for doc_id, chunk, score in selected
        ]

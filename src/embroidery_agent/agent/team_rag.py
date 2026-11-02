"""团队外挂 RAG 的向量实现：本地知识目录 → bge-m3 向量 → FAISS 检索。

实现 rag.py 的 RAGProvider 协议，作为 team.rag_provider 的默认推荐值：

    "rag_provider": "embroidery_agent.agent.team_rag.VectorRAGProvider"

特性：
- 数据源：data/knowledge（原有公开知识库）+ data/team_kb（外部数据集，
  由 scripts/fetch_team_kb.py 从开源数据集摄取，含来源声明）；
- 索引缓存：data/index/team_rag/ 下持久化 FAISS 索引 + chunk 明文 +
  manifest（源文件 mtime 清单）；源文件未变则直接加载，变了自动重建；
- 优雅降级：Ollama embedding 不可用时返回空结果（专家自动退化为无 RAG），
  不阻塞业务。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from ..core.settings import get_settings
from ..retrieval.vector import EmbeddingClient, FaissVectorStore
from .rag import Doc

logger = logging.getLogger("embroidery_agent.team_rag")

CHUNK_TARGET_CHARS = 400
CHUNK_MAX_CHARS = 600
EMBED_BATCH_SIZE = 64  # Ollama 单次 embedding 批次上限（过大其内部 tokenizer 会拒绝）


class VectorRAGProvider:
    name = "vector_kb"

    def __init__(
        self,
        kb_dirs: list[str | Path] | None = None,
        index_dir: str | Path | None = None,
        embedding: EmbeddingClient | None = None,
    ):
        settings = get_settings()
        config = settings.config
        llm_cfg = config["llm"]
        team_cfg = config.get("team", {})

        self.kb_dirs = [
            settings.resolve_path(str(p))
            for p in (kb_dirs or team_cfg.get("rag_kb_dirs")
                      or ["data/knowledge", "data/team_kb"])
        ]
        self.index_dir = settings.resolve_path(
            str(index_dir or team_cfg.get("rag_index_dir") or "data/index/team_rag")
        )
        self.embedding = embedding or EmbeddingClient(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["embedding_model"],
            timeout=llm_cfg.get("timeout_seconds", 120),
        )
        self.min_score = float(team_cfg.get("rag_min_score", 0.35))
        self._store: FaissVectorStore | None = None
        self._chunks: dict[str, Doc] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 文档装载与切块
    # ------------------------------------------------------------------
    def _source_files(self) -> list[Path]:
        files: list[Path] = []
        for d in self.kb_dirs:
            if not d.exists():
                continue
            files.extend(
                p for p in sorted(d.rglob("*"))
                if p.is_file() and p.suffix.lower() in (".md", ".txt")
            )
        return files

    @staticmethod
    def _chunk_file(path: Path) -> list[Doc]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # 按空行分段，携带最近的标题作为上下文
        heading = ""
        paragraphs: list[str] = []
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                heading = block.splitlines()[0].lstrip("# ").strip()
                continue
            paragraphs.append((heading, block))

        chunks: list[Doc] = []
        buf, buf_heading, size = "", "", 0
        for para_heading, block in paragraphs:
            if buf and (size + len(block) > CHUNK_TARGET_CHARS
                        or para_heading != buf_heading):
                chunks.append(Doc(
                    text=buf.strip()[:CHUNK_MAX_CHARS],
                    source=path.name,
                    category=buf_heading or "通用",
                ))
                buf, size = "", 0
            buf_heading = para_heading
            buf += ("\n" if buf else "") + block
            size += len(block)
        if buf.strip():
            chunks.append(Doc(
                text=buf.strip()[:CHUNK_MAX_CHARS],
                source=path.name,
                category=buf_heading or "通用",
            ))
        return chunks

    # ------------------------------------------------------------------
    # 索引构建与缓存
    # ------------------------------------------------------------------
    def _manifest(self) -> dict:
        return {
            "model": self.embedding.model,
            "files": {
                str(p): [p.stat().st_mtime_ns, p.stat().st_size]
                for p in self._source_files()
            },
        }

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _cache_valid(self) -> bool:
        try:
            saved = json.loads(self._manifest_path().read_text(encoding="utf-8"))
            return saved == self._manifest()
        except Exception:
            return False

    def _encode_all(self, texts: list[str]):
        """分批编码，规避 Ollama 大批量 400。"""
        import numpy as np

        parts = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            parts.append(self.embedding.encode_batch(batch))
        return np.vstack(parts)

    def build(self) -> dict:
        """全量构建索引（可由脚本或运维手动调用）。"""
        files = self._source_files()
        docs: list[Doc] = []
        for p in files:
            docs.extend(self._chunk_file(p))
        if not docs:
            logger.warning("RAG 知识目录为空: %s", [str(d) for d in self.kb_dirs])
            self._store, self._chunks = None, {}
            return {"chunks": 0, "files": 0}

        vectors = self._encode_all([d.text for d in docs])
        store = FaissVectorStore(dimension=vectors.shape[1])
        ids = [f"c{i}" for i in range(len(docs))]
        store.add(ids, vectors)
        store.save(self.index_dir / "team_rag.faiss")
        (self.index_dir / "chunks.json").write_text(
            json.dumps(
                [{"id": i, "text": d.text, "source": d.source, "category": d.category}
                 for i, d in zip(ids, docs)],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._manifest_path().write_text(
            json.dumps(self._manifest(), ensure_ascii=False), encoding="utf-8"
        )
        self._store, self._chunks = store, dict(zip(ids, docs))
        logger.info("RAG 索引构建完成: %d chunks / %d files", len(docs), len(files))
        return {"chunks": len(docs), "files": len(files)}

    def _ensure_index(self) -> None:
        if self._store is not None:
            return
        with self._lock:
            if self._store is not None:
                return
            if self._cache_valid():
                try:
                    store = FaissVectorStore.load(self.index_dir / "team_rag.faiss")
                    raw = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
                    self._chunks = {r["id"]: Doc(
                        text=r["text"], source=r["source"], category=r["category"]
                    ) for r in raw}
                    self._store = store
                    return
                except Exception:
                    logger.warning("RAG 缓存索引加载失败，触发重建", exc_info=True)
            try:
                self.build()
            except Exception:
                logger.error("RAG 索引构建失败（embedding 服务不可用？）", exc_info=True)
                self._store, self._chunks = None, {}

    # ------------------------------------------------------------------
    # RAGProvider 协议
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> list[Doc]:
        self._ensure_index()
        if self._store is None or not self._chunks:
            return []
        try:
            qvec = self.embedding.encode(query)
        except Exception:
            logger.warning("RAG 查询向量化失败", exc_info=True)
            return []
        hits = self._store.search(qvec, top_k=top_k)
        results = []
        for chunk_id, score in hits:
            doc = self._chunks.get(chunk_id)
            if doc is None or score < self.min_score:
                continue
            results.append(Doc(
                text=doc.text, source=doc.source,
                category=doc.category, score=round(float(score), 4),
            ))
        return results

    def stats(self) -> dict:
        self._ensure_index()
        files = self._source_files()
        return {
            "chunks": len(self._chunks),
            "files": len(files),
            "index_ready": self._store is not None,
            "kb_dirs": [str(d) for d in self.kb_dirs],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider = VectorRAGProvider()
    print(json.dumps(provider.build(), ensure_ascii=False))

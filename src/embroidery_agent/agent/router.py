"""角色路由：组装 Agent、Retriever、LLM，作为 API 层的单一入口。"""
from __future__ import annotations

from ..core.settings import Settings, get_settings
from ..generation.llm import LLMClient
from ..retrieval.pipeline import HybridRetriever, KnowledgeBase
from ..retrieval.reranker import Reranker
from ..retrieval.vector import EmbeddingClient
from .customer import CustomerAgent
from .staff import StaffAgent
from .team import TeamOrchestrator
from .ticket import TicketStore


class AgentRuntime:
    """持有全局单例：知识库 / 检索器 / LLM / 客户与员工 Agent / 工单存储。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        config = self.settings.config

        kb_cfg = config["kb"]
        llm_cfg = config["llm"]
        retrieval_cfg = config["retrieval"]
        storage_cfg = config["storage"]

        self.embedding = EmbeddingClient(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["embedding_model"],
            timeout=llm_cfg.get("timeout_seconds", 120),
        )
        self.llm = LLMClient(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["chat_model"],
            timeout=llm_cfg.get("timeout_seconds", 120),
        )
        self.kb = KnowledgeBase(
            public_path=self.settings.resolve_path(kb_cfg["public_path"]),
            internal_path=self.settings.resolve_path(kb_cfg["internal_path"]),
            index_path=self.settings.resolve_path(kb_cfg["index_path"]),
            embedding=self.embedding,
            chunk_size=kb_cfg.get("chunk_size", 300),
            chunk_overlap=kb_cfg.get("chunk_overlap", 40),
        )

        rerank_cfg = config.get("rerank", {})
        reranker = None
        if rerank_cfg.get("enabled", False):
            reranker = Reranker(
                model_name=rerank_cfg.get("model_name", "BAAI/bge-reranker-base"),
                device=rerank_cfg.get("device", "cpu"),
            )

        self.retriever = HybridRetriever(
            kb=self.kb,
            reranker=reranker,
            recall_k=retrieval_cfg.get("recall_k", 20),
            top_k=retrieval_cfg.get("top_k", 5),
            use_bm25=retrieval_cfg.get("use_bm25", True),
            use_vector=retrieval_cfg.get("use_vector", True),
            use_rerank=retrieval_cfg.get("use_rerank", True),
        )

        self.tickets = TicketStore(self.settings.resolve_path(storage_cfg["db_path"]))
        self.customer = CustomerAgent(
            self.retriever, self.llm, self.tickets,
            db_path=self.settings.resolve_path(storage_cfg["db_path"]),
        )
        self.staff = StaffAgent(self.retriever, self.llm)
        # 多 Agent 团队：外挂 RAG 由 TeamOrchestrator 按 team.rag_provider 配置自行加载
        self.team = TeamOrchestrator(self.llm, self.settings)

    def build_kb(self) -> dict:
        return self.kb.build()

    def load_kb(self) -> dict:
        return self.kb.load()

    def status(self) -> dict:
        return {
            "llm_ready": self.llm.is_ready(),
            "llm_model": self.llm.model,
            "embedding_model": self.embedding.model,
            "kb_chunks": len(self.kb.chunks),
            "kb_public": sum(1 for c in self.kb.chunks.values() if c.visibility == "public"),
            "kb_internal": sum(1 for c in self.kb.chunks.values() if c.visibility == "internal"),
            "vector_index": self.kb.vector_store is not None,
            "rerank_ready": bool(
                self.retriever.reranker and self.retriever.reranker.is_ready()
            ),
            "team_agents": [a["id"] for a in self.team.list_agents()],
            "team_rag": getattr(self.team.rag, "name", "") or "",
        }


def build_runtime(settings: Settings | None = None) -> AgentRuntime:
    return AgentRuntime(settings)

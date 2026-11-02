"""外挂 RAG 接口（预留）：多 Agent 团队统一的知识检索入口。

当前为留空状态（NullRAGProvider 恒返回空结果），后期接入企业私有知识库时：

1. 实现本模块的 RAGProvider 协议，例如：

    class MyVectorRAG:
        name = "my_vector_rag"
        def retrieve(self, query: str, top_k: int = 5) -> list[Doc]:
            # 调用你的向量库 / ES / 数据库，返回 Doc 列表
            ...

2. 在 configs/default.json 的 team.rag_provider 中填入导入路径::

        "rag_provider": "my_company.rag.MyVectorRAG"

   或每个 Agent 单独指定（team.agents.<agent_id>.rag_provider）。

3. 重启服务即生效；检索结果会自动拼入各专家 Agent 的回答上下文，
   并要求模型标注来源编号。

注意：现有 agent/ 的双角色知识库（HybridRetriever）不受影响，
团队层的 RAG 是独立、可插拔的外挂通道。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Doc:
    """检索返回的文档片段。"""
    text: str
    source: str = ""
    category: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class RAGProvider(Protocol):
    """知识检索协议：所有外挂知识源实现此接口即可接入团队。"""

    name: str

    def retrieve(self, query: str, top_k: int = 5) -> list[Doc]:
        ...


class NullRAGProvider:
    """空实现：外挂 RAG 尚未导入时的默认提供方。"""

    name = "null"

    def retrieve(self, query: str, top_k: int = 5) -> list[Doc]:
        return []


def load_rag_provider(dotted_path: str) -> RAGProvider | None:
    """按导入路径实例化 RAGProvider，如 "my_pkg.rag.MyVectorRAG"。"""
    dotted_path = (dotted_path or "").strip()
    if not dotted_path:
        return None
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"rag_provider 配置无效: {dotted_path}")
    cls = getattr(importlib.import_module(module_path), class_name)
    provider = cls()
    if not callable(getattr(provider, "retrieve", None)):
        raise TypeError(f"{dotted_path} 未实现 RAGProvider 协议（缺少 retrieve 方法）")
    return provider

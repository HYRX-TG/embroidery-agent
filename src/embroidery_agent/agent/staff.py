"""员工会话处理器：行业答疑（含内部库）+ 对话复盘 + 文案生成。

权限：员工检索可同时使用 public + internal 知识库（双角色权限第二层）。
"""
from __future__ import annotations

from typing import Iterable

from ..core.schema import Intent, Role
from ..generation.answer import build_extractive_fallback, staff_messages, stream_grounded_answer
from ..generation.llm import LLMChunk, LLMClient
from ..generation.prompts import STAFF_SYSTEM_PROMPT
from ..nlp.guard import guard_query
from ..nlp.intent import IntentClassifier
from ..nlp.ner import CraftNER
from ..retrieval.pipeline import HybridRetriever


class StaffAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        ner: CraftNER | None = None,
        classifier: IntentClassifier | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.ner = ner or CraftNER()
        self.classifier = classifier or IntentClassifier(self.ner)

    def process(self, query: str) -> Iterable[LLMChunk]:
        # 提示词注入拦截：员工端同样防护（防止诱导输出内部成本/底价）
        refusal = guard_query(query, role="staff")
        if refusal:
            yield LLMChunk(refusal, kind="fallback")
            yield LLMChunk("", done=True)
            return

        result = self.classifier.classify(query, Role.STAFF.value)
        hits = self.retriever.search(query, role=Role.STAFF)

        if result.intent in (Intent.STAFF_ANALYSIS, Intent.STAFF_COPYWRITING):
            # 复盘/文案类：直接交给 LLM（内部库内容已注入上下文）
            messages = staff_messages(query, hits)
            fallback = (
                "该任务需要本地 LLM 支持（Ollama）。请先启动 Ollama 并加载模型后重试。"
                if not hits
                else build_extractive_fallback(query, hits)
            )
            yield from stream_grounded_answer(self.llm, messages, fallback_text=fallback)
            return

        # 默认：行业知识答疑（含内部库）
        messages = staff_messages(query, hits)
        fallback = build_extractive_fallback(query, hits)
        yield from stream_grounded_answer(self.llm, messages, fallback_text=fallback)

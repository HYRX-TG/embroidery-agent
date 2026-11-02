"""TeamOrchestrator：多 Agent 团队协调器。

流程：用户请求 → 意图路由 → 派发给专家 Agent → 流式返回。

生产化设计：
- 路由顺序：关键词正则（强领域信号，零延迟）→ LLM JSON 路由（带 LRU 缓存）
  → 默认客服，任何输入都有出口；关键词命中时完全不调用 LLM；
- 会话持久化：SQLite（复用工单库文件），重启/多 worker 不丢上下文；
- 结构化日志：路由决策来源、接单专家、耗时可观测（logs 通过 logging 输出）。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from ..generation.llm import LLMChunk, LLMClient
from .experts import EXPERT_CLASSES, BaseExpert
from .prompts_team import TEAM_ROUTER_PROMPT
from .rag import RAGProvider, load_rag_provider

logger = logging.getLogger("embroidery_agent.team")

# 关键词路由（优先于 LLM，命中即省一次推理调用）。顺序即优先级。
KEYWORD_ROUTES: list[tuple[str, str]] = [
    (r"运费|快递|物流|发货|承运|首重|续重|体积重|到付|发往|发.{0,3}(华东|华南|华北|西南|西北|东北)|kg|千克|公斤", "logistics"),
    (r"报价|核算成本|单价|折扣|成本价|毛利|报个价|多少钱|什么价|怎么卖", "quotation"),
    (r"排程|交期|工期|产能|工序|排产|赶工|多少天能|要多久|几天能|能不能加急", "production"),
    (r"质检|疵品|瑕疵|返工|跳线|色差|断线|起皱|抽丝|开线", "quality"),
    (r"工单|统计|报表|趋势|积压|数据看板|做了多少单", "analytics"),
]

ROUTE_CACHE_SIZE = 256


class TeamOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        settings: object,
        rag: RAGProvider | None = None,
    ):
        self.llm = llm
        self.settings = settings
        config = settings.config  # type: ignore[attr-defined]
        team_cfg = config.get("team", {})

        # 外挂 RAG：配置填导入路径则动态加载；留空为 None（各专家自动跳过检索）
        self.rag = rag
        if self.rag is None:
            provider_path = (team_cfg.get("rag_provider") or "").strip()
            if provider_path:
                self.rag = load_rag_provider(provider_path)
                logger.info("外挂 RAG 已加载: %s", provider_path)

        enabled = team_cfg.get("enabled") or list(EXPERT_CLASSES.keys())
        self.experts: dict[str, BaseExpert] = {}
        for agent_id in enabled:
            cls = EXPERT_CLASSES.get(agent_id)
            if cls is None:
                logger.warning("配置了未知专家，已忽略: %s", agent_id)
                continue
            if agent_id == "analytics":
                db_path = settings.resolve_path(config["storage"]["db_path"])  # type: ignore[attr-defined]
                self.experts[agent_id] = cls(self.llm, self.rag, str(db_path))
            else:
                self.experts[agent_id] = cls(self.llm, self.rag)
        self.router_temperature = float(team_cfg.get("router_temperature", 0.1))

        # 会话持久化：复用工单 SQLite（新表，互不影响）
        self.session_db = Path(
            settings.resolve_path(config["storage"]["db_path"])  # type: ignore[attr-defined]
        )
        self._init_session_table()
        self._route_cache: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------
    # 会话持久化
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.session_db), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_session_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS team_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_team_messages_session "
                "ON team_messages(session_id, id)"
            )

    def _history(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM team_messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def _remember(self, session_id: str, role: str, content: str,
                  agent_id: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO team_messages (session_id, role, content, agent_id) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content[:8000], agent_id),
            )

    # ------------------------------------------------------------------
    # 信息
    # ------------------------------------------------------------------
    def list_agents(self) -> list[dict]:
        return [
            {"id": e.spec.agent_id, "name": e.spec.name, "description": e.spec.description}
            for e in self.experts.values()
        ]

    # ------------------------------------------------------------------
    # 路由：关键词优先 → LLM（缓存）→ 默认
    # ------------------------------------------------------------------
    def route(self, query: str) -> str:
        agent_id = self._route_by_keywords(query)
        source = "keyword"
        if agent_id is None:
            agent_id = self._route_by_llm(query)
            source = "llm" if agent_id else "default"
        if agent_id not in self.experts:
            agent_id = next(iter(self.experts))
        logger.info("路由[%s] %r -> %s", source, query[:40], agent_id)
        return agent_id

    def _route_by_llm(self, query: str) -> str | None:
        key = re.sub(r"\s+", "", query)[:64]
        if key in self._route_cache:
            self._route_cache.move_to_end(key)
            return self._route_cache[key]
        if not self.llm.is_ready():
            return None
        agents_desc = "\n".join(
            f"- {e.spec.agent_id}: {e.spec.description}" for e in self.experts.values()
        )
        prompt = TEAM_ROUTER_PROMPT.replace("{agents}", agents_desc)
        agent_id = None
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": f"{prompt}\n\n用户请求：{query}"}],
                temperature=self.router_temperature,
                response_format="json_object",
            )
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                candidate = data.get("agent", "")
                if candidate in self.experts:
                    agent_id = candidate
        except Exception:
            logger.debug("LLM 路由失败", exc_info=True)
        if agent_id is not None:
            self._route_cache[key] = agent_id
            if len(self._route_cache) > ROUTE_CACHE_SIZE:
                self._route_cache.popitem(last=False)
        return agent_id

    @staticmethod
    def _route_by_keywords(query: str) -> str | None:
        for pattern, agent_id in KEYWORD_ROUTES:
            if re.search(pattern, query):
                return agent_id
        return None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def process(self, session_id: str, query: str,
                agent_id: str = "") -> tuple[str, Iterable[LLMChunk]]:
        """返回 (agent_id, 流式响应)。"""
        start = time.monotonic()
        if agent_id in self.experts:
            selected = agent_id
            logger.info("路由[manual] %r -> %s", query[:40], selected)
        else:
            selected = self.route(query)
        expert = self.experts[selected]
        history = self._history(session_id)
        self._remember(session_id, "user", query, selected)

        def stream() -> Iterable[LLMChunk]:
            collected: list[str] = []
            try:
                for chunk in expert.answer(query, history):
                    if chunk.kind in ("content", "fallback") and chunk.text:
                        collected.append(chunk.text)
                    yield chunk
            finally:
                self._remember(session_id, "assistant", "".join(collected), selected)
                logger.info("专家 %s 处理完成，耗时 %.2fs",
                            selected, time.monotonic() - start)

        return selected, stream()

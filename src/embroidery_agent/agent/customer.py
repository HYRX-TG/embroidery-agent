"""客户会话处理器：意图路由 + 需求追问循环 + 工单自动创建。

核心流程（对应系统提示词的第 2/3 节）：
1. 意图分类
2. KNOWLEDGE → RAG 检索 → 流式回答
3. CUSTOM_REQUIREMENT → 需求表单填充 → 缺字段则单条追问 → 齐全则输出表单+方案+工单
4. PATTERN_MEANING / ORDER_SERVICE / COMPLAINT / SMALL_TALK → 对应话术
5. 门店专属信息（价格/工期/库存等）→ 固定兜底话术 + 自动工单
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..core.schema import (
    Intent,
    Message,
    RequirementForm,
    Role,
    Ticket,
)
from ..generation.answer import (
    build_extractive_fallback,
    create_ticket_from_requirement,
    customer_messages,
    extract_requirement_form,
    stream_grounded_answer,
)
from ..generation.llm import LLMChunk, LLMClient
from ..generation.prompts import (
    COMPLAINT_REPLY,
    FALLBACK_TICKET_MESSAGE,
    SMALL_TALK_REPLY,
)
from ..nlp.guard import guard_query
from ..nlp.intent import IntentClassifier
from ..nlp.ner import CraftNER
from ..retrieval.pipeline import HybridRetriever, RetrievalHit
from .ticket import TicketStore

# 需求追问的固定顺序：中文标签 → 追问话术（missing_fields 返回中文标签）
QUESTION_FLOW: list[tuple[str, str]] = [
    ("图案主题", "请问您想要绣什么图案或主题呢？比如牡丹、龙凤呈祥、山水、生肖等，或者根据用途来推荐~"),
    ("尺寸规格", "请问您需要的绣品尺寸大概是多少呢？比如画框 60×80cm，或者摆件的高度~"),
    ("工艺类型", "您倾向于哪种工艺类型呢？手工绣（细腻耗时）、半手工（性价比高）还是机绣（工期短）？"),
    ("使用用途", "这幅绣品的用途是什么呢？比如送礼、家装装饰、婚庆还是企业定制？"),
    ("定制数量", "请问需要几幅呢？一件还是多件？"),
    ("预算区间", "方便告知您的预算范围吗？我可以帮您匹配更合适的方案~"),
    ("交付时限", "您希望什么时候交付呢？有具体的时间要求吗？"),
    ("特殊要求", "还有其他特殊要求吗？比如特定颜色、装裱方式、图案细节等~"),
]

PRICE_SPECIAL_TERMS = ("价格", "多少钱", "报价", "价位", "优惠", "折扣", "成本", "底价", "工期", "发货时间", "多久到")

# 客户确认回环：明确认可词（短句命中才视为确认，长句仍算修改意见）
CONFIRM_KEYWORDS = ("确认", "没问题", "对的", "提交", "可以了", "就这样", "没错", "同意", "ok", "ok的", "确认无误")


@dataclass
class CustomerSession:
    """单个客户会话的状态（SQLite 持久化，重启/多 worker 不丢）。"""
    session_id: str = ""
    conversation: list[Message] = field(default_factory=list)
    requirement: RequirementForm = field(default_factory=RequirementForm)
    ticket: Ticket | None = None
    collecting: bool = False  # 处于定制需求收集模式（追问循环中）
    confirming: bool = False  # 表单已生成，等待客户确认/修改（确认回环）
    _store: "CustomerAgent | None" = field(default=None, repr=False, compare=False)

    def append(self, role: str, content: str) -> None:
        self.conversation.append(Message(role=role, content=content))
        if self._store is not None:
            self._store._append_message(self, role, content)


class CustomerAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        tickets: TicketStore,
        ner: CraftNER | None = None,
        classifier: IntentClassifier | None = None,
        db_path: str | Path | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.tickets = tickets
        self.ner = ner or CraftNER()
        self.classifier = classifier or IntentClassifier(self.ner)
        self.sessions: dict[str, CustomerSession] = {}
        self.db_path = Path(db_path) if db_path else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_session_tables()

    # ------------------------------------------------------------------
    # 会话持久化（SQLite：customer_sessions + customer_messages）
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_session_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_sessions (
                    session_id TEXT PRIMARY KEY,
                    requirement TEXT NOT NULL DEFAULT '{}',
                    collecting INTEGER NOT NULL DEFAULT 0,
                    confirming INTEGER NOT NULL DEFAULT 0,
                    ticket_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            # 旧库迁移：补 confirming 列（已存在则忽略）
            try:
                conn.execute(
                    "ALTER TABLE customer_sessions ADD COLUMN confirming INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_customer_messages_session "
                "ON customer_messages(session_id, id)"
            )

    def _load_session(self, session_id: str) -> CustomerSession | None:
        if self.db_path is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT requirement, collecting, confirming, ticket_id FROM customer_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            messages = conn.execute(
                "SELECT role, content FROM customer_messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        session = CustomerSession(_store=self, session_id=session_id)
        session.conversation = [Message(role=r["role"], content=r["content"]) for r in messages]
        try:
            session.requirement = RequirementForm(**json.loads(row["requirement"] or "{}"))
        except (TypeError, ValueError):
            session.requirement = RequirementForm()
        session.collecting = bool(row["collecting"])
        session.confirming = bool(row["confirming"])
        if row["ticket_id"]:
            session.ticket = self.tickets.get(row["ticket_id"])
        return session

    def _save_session(self, session: CustomerSession) -> None:
        if self.db_path is None:
            return
        from ..core.schema import now_iso
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO customer_sessions (session_id, requirement, collecting, confirming, ticket_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    requirement = excluded.requirement,
                    collecting = excluded.collecting,
                    confirming = excluded.confirming,
                    ticket_id = excluded.ticket_id,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    json.dumps(session.requirement.to_dict(), ensure_ascii=False),
                    1 if session.collecting else 0,
                    1 if session.confirming else 0,
                    session.ticket.id if session.ticket else "",
                    now_iso(),
                ),
            )

    def _append_message(self, session: CustomerSession, role: str, content: str) -> None:
        if self.db_path is None:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO customer_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session.session_id, role, content[:8000]),
            )
            conn.execute(
                """
                INSERT INTO customer_sessions (session_id, requirement, collecting, ticket_id, updated_at)
                VALUES (?, '{}', 0, '', datetime('now'))
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session.session_id,),
            )

    def list_messages(self, session_id: str) -> list[dict]:
        """返回某会话的历史消息（供前端回显）；无会话/未启用持久化时返回空。"""
        if self.db_path is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM customer_messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]

    def get_session(self, session_id: str) -> CustomerSession:
        if session_id not in self.sessions:
            loaded = self._load_session(session_id) if self.db_path is not None else None
            session = loaded if loaded is not None else CustomerSession(_store=self, session_id=session_id)
            self.sessions[session_id] = session
        return self.sessions[session_id]

    def _retrieve(self, query: str) -> list[RetrievalHit]:
        return self.retriever.search(query, role=Role.CUSTOMER)

    def _hit_special_business(self, query: str) -> bool:
        """命中门店专属业务词（价格/工期/库存等）→ 强制转人工，不检索应答。"""
        return any(term in query for term in PRICE_SPECIAL_TERMS)

    def process(self, session_id: str, query: str) -> Iterable[LLMChunk]:
        # 提示词注入拦截：命中即礼貌拒绝，不进入任何业务流程（审计日志在 guard 内）
        refusal = guard_query(query, role="customer")
        if refusal:
            yield LLMChunk(refusal, kind="fallback")
            yield LLMChunk("", done=True)
            return

        session = self.get_session(session_id)
        session.append("user", query)

        # 门店专属业务信息（价格/工期等）→ 兜底话术 + 工单（不经过 RAG）
        if self._hit_special_business(query):
            yield LLMChunk(FALLBACK_TICKET_MESSAGE, kind="fallback")
            yield LLMChunk("", done=True)
            session.append("assistant", FALLBACK_TICKET_MESSAGE)
            self._ensure_ticket(session, note="客户咨询门店专属业务信息（价格/工期等），需人工跟进")
            return

        # 需求收集模式：回复优先并入表单；跑题判定由 _handle_requirement
        # 依据「本轮是否新填字段」处理（有新信息=表单回答，无新信息才放行知识问答）
        if session.collecting and not self._exit_requirement(query):
            yield from self._handle_requirement(session, query)
            return

        # 确认回环：表单已生成，等待客户"确认"提交或"说修改点"
        if session.confirming and not self._exit_requirement(query):
            yield from self._handle_confirmation(session, query)
            return

        result = self.classifier.classify(query, Role.CUSTOMER.value)

        if result.intent == Intent.CUSTOM_REQUIREMENT:
            yield from self._handle_requirement(session, query)
            return

        if result.intent == Intent.PATTERN_MEANING:
            yield from self._stream_knowledge(session, query)
            return

        if result.intent == Intent.COMPLAINT:
            yield LLMChunk(COMPLAINT_REPLY, kind="content")
            yield LLMChunk("", done=True)
            session.append("assistant", COMPLAINT_REPLY)
            return

        if result.intent == Intent.ORDER_SERVICE:
            # 订单/售后：仅输出行业通用规则；门店专属规则已在 _hit_special_business 拦截
            yield from self._stream_knowledge(session, query)
            return

        if result.intent == Intent.SMALL_TALK:
            # 问候/寒暄也由本地模型自然回应；仅在模型不可用时用固定话术兜底
            yield from self._stream_knowledge(session, query, fallback_text=SMALL_TALK_REPLY)
            return

        # 默认知识咨询
        yield from self._stream_knowledge(session, query)

    @staticmethod
    def _exit_requirement(query: str) -> bool:
        """客户明确退出定制流程（不用了/先看看/随便问问）。"""
        return any(term in query for term in ("不用了", "先看看", "随便问问", "只是问问", "不定了", "算了", "以后再说"))

    def _handle_requirement(self, session: CustomerSession, query: str) -> Iterable[LLMChunk]:
        session.collecting = True
        # 先做表单抽取，用「本轮是否新填了字段」区分表单回答与跑题提问：
        # 有新信息 = 客户在回答追问；无新信息 = 可能是顺带的知识提问
        conversation_text = "\n".join(m.content for m in session.conversation[-8:])
        before = {k for k in session.requirement.__dataclass_fields__ if getattr(session.requirement, k).strip()}
        session.requirement = extract_requirement_form(
            self.llm, conversation_text, current=session.requirement
        )
        after = {k for k in session.requirement.__dataclass_fields__ if getattr(session.requirement, k).strip()}
        new_info = after - before
        self._save_session(session)  # 收集进度落库：中途断线/重启可续

        missing = session.requirement.missing_fields()
        if missing:
            if not new_info:
                # 本轮没有提供表单信息 → 跑题判定：明确的知识/售后问题先放行回答
                result = self.classifier.classify(query, Role.CUSTOMER.value)
                off_topic = (
                    result.intent in (Intent.KNOWLEDGE, Intent.PATTERN_MEANING, Intent.ORDER_SERVICE)
                    and result.confidence != "low"
                )
                if off_topic:
                    yield from self._stream_knowledge(session, query, preserve_collecting=True)
                    return
            # missing 返回中文标签（图案主题/尺寸规格…），QUESTION_FLOW 已对齐
            next_label = missing[0]
            question = next((q for label, q in QUESTION_FLOW if label == next_label), None)
            if question is None:
                # 剩余缺失字段不在追问列表（如数量/核心偏好）→ 按顺序兜底提问
                question = f"请问您还有别的信息想补充吗？比如{next_label}~"
            yield LLMChunk(question, kind="content")
            yield LLMChunk("", done=True)
            session.append("assistant", question)
            return

        # 信息齐全 → 展示表单，进入「客户确认回环」（客户认可后才转人工建单）
        response = self._render_form(session.requirement, first_show=True)
        yield LLMChunk(response, kind="content")
        yield LLMChunk("", done=True)
        session.append("assistant", response)
        session.collecting = False
        session.confirming = True
        self._save_session(session)

    def _render_form(self, requirement: RequirementForm, first_show: bool) -> str:
        """渲染结构化需求表单 + 确认引导（首次展示与修改后复用同一份渲染）。"""
        form = requirement.to_dict()
        form_text = "\n".join(f"- {label}：{form[key]}" for key, label in [
            ("pattern_theme", "图案主题"), ("size", "尺寸规格"), ("craft_type", "工艺类型"),
            ("quantity", "定制数量"), ("usage", "使用用途"), ("budget_range", "预算区间"),
            ("deadline", "交付时限"), ("special_requirements", "特殊要求"),
            ("core_preferences", "核心偏好"),
        ] if form[key])
        opening = (
            "太棒了！我已为您整理好定制需求，请您过目：\n\n"
            "【结构化需求表单】\n"
            if first_show
            else "好的，已按您的要求更新，最新表单如下：\n\n【结构化需求表单】\n"
        )
        return (
            f"{opening}"
            f"{form_text}\n\n"
            "【适配方案建议】\n"
            "根据您的用途与预算，建议优先考虑手工绣（细节更细腻、收藏价值高）；"
            "如需控制成本或缩短工期，可考虑半手工/机绣方案。具体报价与工期"
            "将由刺绣顾问为您精准核算。\n\n"
            "【请您确认】\n"
            "内容无误请回复「确认」，我将提交给专属刺绣顾问为您核验；"
            "如需调整，请直接告诉我要修改的地方~"
        )

    def _handle_confirmation(self, session: CustomerSession, query: str) -> Iterable[LLMChunk]:
        """确认回环：客户「确认」→ 提交转人工；其他输入视为修改意见 → 更新表单重新确认。"""
        text = query.strip()
        if any(k in text.lower() for k in CONFIRM_KEYWORDS) and len(text) <= 12:
            session.confirming = False
            self._save_session(session)
            self._ensure_ticket(
                session,
                note="客户已确认需求表单，待人工核验后走流程",
                summary=f"定制需求（客户已确认）：{session.requirement.pattern_theme or '未指定主题'}",
            )
            response = (
                "收到您的确认！需求已提交，专属刺绣顾问将尽快为您核验并核算报价，"
                "请您耐心等待~ 如后续需要调整，随时告诉我。"
            )
            yield LLMChunk(response, kind="content")
            yield LLMChunk("", done=True)
            session.append("assistant", response)
            return

        # 视为修改意见：LLM 抽取合并（空字段保留旧值），更新后重新确认
        conversation_text = "\n".join(m.content for m in session.conversation[-8:])
        session.requirement = extract_requirement_form(
            self.llm, conversation_text, current=session.requirement
        )
        missing = session.requirement.missing_fields()
        if missing:
            # 修改把某字段改空了 → 回到收集模式继续追问
            session.confirming = False
            session.collecting = True
            self._save_session(session)
            yield from self._handle_requirement(session, query)
            return
        response = self._render_form(session.requirement, first_show=False)
        yield LLMChunk(response, kind="content")
        yield LLMChunk("", done=True)
        session.append("assistant", response)
        self._save_session(session)

    def _stream_knowledge(
        self,
        session: CustomerSession,
        query: str,
        fallback_text: str | None = None,
        preserve_collecting: bool = False,
    ) -> Iterable[LLMChunk]:
        hits = self._retrieve(query)
        fallback = fallback_text or build_extractive_fallback(query, hits)
        messages = customer_messages(query, hits, session.conversation[:-1])
        collected: list[str] = []
        for chunk in stream_grounded_answer(self.llm, messages, fallback_text=fallback):
            if chunk.text:
                collected.append(chunk.text)
            yield chunk
        # 落库真实回答，保证多轮对话上下文完整（不再写占位符）
        answer_text = "".join(collected).strip()
        session.append("assistant", answer_text or fallback)
        # 知识咨询结束即退出需求收集模式（客户若继续定制会重新触发）；
        # 收集模式中的"跑题放行"保留收集状态，答完继续收集
        if not preserve_collecting:
            session.collecting = False
        self._save_session(session)

    def _ensure_ticket(self, session: CustomerSession, note: str, summary: str = "") -> None:
        if session.ticket is None:
            session.ticket = create_ticket_from_requirement(
                self.tickets,
                session.requirement,
                session.conversation,
                summary=summary,
            )
        session.ticket.note = note
        # 备注变更真实落库并刷新 updated_at，便于人工按时间排序跟进
        updated = self.tickets.update_note(session.ticket.id, note)
        if updated is not None:
            session.ticket = updated
        self._save_session(session)  # ticket_id 与 collecting 状态落库

"""回答生成：RAG 上下文拼装 + 流式输出 + 抽取式兜底。

抽取式兜底：LLM 不可用（Ollama 未启动/模型缺失）时，直接引用知识库切片回答，
保证"本地模型没配好也能演示检索链路"，参考 medical_retrieval_plus 的 fallback 设计。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from ..agent.ticket import TicketStore
from ..core.schema import Message, RequirementForm, Ticket, TicketStatus
from ..nlp.ner import CraftNER
from ..retrieval.pipeline import RetrievalHit
from .llm import LLMChunk, LLMClient
from .prompts import (
    CUSTOMER_SYSTEM_PROMPT,
    FALLBACK_TICKET_MESSAGE,
    REQUIREMENT_FORM_PROMPT,
    STAFF_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def build_rag_context(hits: list[RetrievalHit], max_chars: int = 2500) -> str:
    """把检索结果拼成带编号的上下文（供 LLM 引用）。"""
    parts: list[str] = []
    total = 0
    for i, hit in enumerate(hits, 1):
        snippet = hit.chunk.text
        if total + len(snippet) > max_chars:
            snippet = snippet[: max_chars - total] + "…"
        parts.append(f"[{i}]（来源：{hit.chunk.source}）{snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def build_extractive_fallback(query: str, hits: list[RetrievalHit]) -> str:
    """无 LLM 时的抽取式回答：直接给出知识库原文 + 来源。"""
    if not hits:
        return (
            "目前知识库中没有与您问题直接相关的资料。"
            "我已为您记录需求，稍后由刺绣顾问为您精准解答。\n"
            f"{FALLBACK_TICKET_MESSAGE}"
        )
    top = hits[0]
    excerpt = re.sub(r"\s+", " ", top.chunk.text).strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:400] + "……"
    return (
        "根据知识库资料，为您整理如下：\n\n"
        f"{excerpt} [1]\n\n"
        f"参考来源：[1]《{top.chunk.source}》（{top.chunk.category}）\n\n"
        "以上为行业通用知识参考，具体门店价格、工期等专属信息，"
        "将由刺绣顾问为您精准解答。"
    )


def customer_messages(query: str, hits: list[RetrievalHit], history: list[Message]) -> list[dict]:
    """拼装客户会话 messages（system + 历史 + RAG 上下文 + 当前问题）。"""
    context = build_rag_context(hits)
    messages: list[dict] = [{"role": "system", "content": CUSTOMER_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    if hits:
        messages.append(
            {
                "role": "system",
                "content": (
                    "以下是知识库检索到的参考资料，回答时优先基于这些内容，"
                    "并在结尾标注来源序号：\n" + context
                ),
            }
        )
    messages.append({"role": "user", "content": query})
    return messages


def staff_messages(query: str, hits: list[RetrievalHit]) -> list[dict]:
    context = build_rag_context(hits)
    messages: list[dict] = [{"role": "system", "content": STAFF_SYSTEM_PROMPT}]
    if hits:
        messages.append(
            {"role": "system", "content": "参考资料：\n" + context}
        )
    messages.append({"role": "user", "content": query})
    return messages


def apply_rule_extraction(text: str, form: RequirementForm) -> RequirementForm:
    """确定性规则抽取（第一优先级）：高频表达直接命中，零 LLM、零延迟、可测试。

    命中即写入（确定性优先于 LLM 结果）；未命中的字段留给 LLM 补。
    """
    import re as _re

    text = text.strip()
    if not text:
        return form

    # 尺寸：60乘80 / 60×80 / 60x80 / 60*80
    m = _re.search(r"(\d{2,4})\s*[乘x×*]\s*(\d{1,4})", text)
    if m:
        form.size = f"{m.group(1)}×{m.group(2)}cm"

    # 工艺类型
    craft = CraftNER().extract_craft_type(text)
    if craft:
        form.craft_type = craft

    # 数量：3幅 / 两件 / 5个
    m = _re.search(r"([一二两三四五六七八九十\d]{1,3})\s*(幅|件|个|对)", text)
    if m:
        form.quantity = m.group(0)

    # 预算：预算三千 / 预算改成五千 / 预算3000 / 价位两千左右
    m = _re.search(
        r"(?:预算|价位|改成|改为|换成)[^，。;；]{0,4}?([一二两三四五六七八九\d]{1,4})(千|万|块|元)?",
        text,
    ) or _re.search(r"([一二两三四五六七八九\d]{1,4})(千|万)(左右)?(以内|以内吧|以内要)?", text)
    if m and any(ch.isdigit() or ch in "一二两三四五六七八九" for ch in m.group(1)):
        cleaned = _re.sub(r"^(预算|价位)", "", m.group(0).replace("改成", "").replace("改为", "").replace("换成", ""))
        form.budget_range = cleaned

    # 用途/场景
    if _re.search(r"(挂|放|装饰)(在)?(客厅|卧室|书房|玄关|办公室|大堂|会议室)", text) or "家装" in text:
        form.usage = form.usage or "家装"
    if _re.search(r"送(给)?(我)?(妈|妈咪|妈妈|爸|爸|长辈|姥姥|奶奶|爷爷|外公|外婆|领导|客户|老师|朋友|闺蜜|同事)", text) or "送长辈" in text or "送人" in text:
        form.usage = "送礼"
    if any(k in text for k in ("婚庆", "结婚", "婚礼", "陪嫁")):
        form.usage = "婚庆"
    if any(k in text for k in ("企业", "公司", "开业", "员工福利", "伴手礼")):
        form.usage = form.usage or "企业/商务"

    # 交付时限：一个月内 / 下个月底 / 30天
    m = _re.search(r"(一个月内|两个月内|半个月内|\d+天内|下个月?\S{0,4}之前?|年底前|月底前)", text)
    if m:
        form.deadline = m.group(1)

    # 图案主题：纹样实体词命中即取（"牡丹题材吧"→牡丹）
    if not form.pattern_theme or "题材" in text or "图案" in text or "改成" in text or "换成" in text:
        entities = CraftNER().extract(text)
        for e in entities:
            if e.type == "纹样":
                form.pattern_theme = e.text
                break

    # 修改类语句中"特殊要求"兜底：含"要/想加/别"的补充描述且不在以上字段时
    return form


def extract_requirement_form(
    llm: LLMClient,
    conversation_text: str,
    current: RequirementForm | None = None,
) -> RequirementForm:
    """从对话中结构化提取需求表单：规则先行，LLM 只补规则漏掉的字段。

    合并策略：
    1. 规则抽取（apply_rule_extraction）命中即写入——确定性优先；
    2. LLM 抽取只填充规则未命中的**空字段**，不覆盖已有值——防止模型抖动冲掉已确认信息。
    """
    form = current or RequirementForm()

    # 第一优先级：确定性规则（高频表达：尺寸/工艺/数量/预算/用途/时限/主题）
    form = apply_rule_extraction(conversation_text.splitlines()[-1] if conversation_text else "", form)

    if not llm.is_ready():
        logger.warning(
            "form_extraction_skipped reason=llm_not_ready model=%s detail=%s",
            llm.model, llm.missing_reason(),
        )
        return form

    # 第二优先级：LLM 补空字段
    prompt = REQUIREMENT_FORM_PROMPT.replace("{conversation}", conversation_text[-3000:])
    try:
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format="json_object",
        )
        data = json.loads(raw)
        for field_name in form.__dataclass_fields__:
            if getattr(form, field_name).strip():
                continue  # 已有值（规则命中或先前收集）不覆盖
            value = data.get(field_name)
            if isinstance(value, str) and value.strip():
                setattr(form, field_name, value.strip())
            elif value is not None and not isinstance(value, str):
                setattr(form, field_name, str(value))
    except Exception as exc:
        logger.warning("form_extraction_llm_failed error=%s", exc)
    return form


def create_ticket_from_requirement(
    store: TicketStore,
    requirement: RequirementForm,
    conversation: list[Message],
    summary: str = "",
) -> Ticket:
    ticket = Ticket(
        role="customer",
        customer_summary=summary or "客户定制需求（知识库无覆盖，需人工跟进）",
        conversation=[m.to_dict() for m in conversation],
        requirement=requirement,
        status=TicketStatus.PENDING.value,
        note="需人工跟进：客户定制需求待顾问确认",
    )
    return store.create(ticket)


def stream_grounded_answer(
    llm: LLMClient,
    messages: list[dict],
    fallback_text: str | None = None,
) -> Iterable[LLMChunk]:
    """流式生成回答；LLM 不可用或生成失败时降级为 fallback 文本（并记录告警）。"""
    if not llm.is_ready():
        logger.warning(
            "llm_fallback reason=llm_not_ready model=%s detail=%s",
            llm.model, llm.missing_reason(),
        )
        yield LLMChunk(fallback_text or "模型服务未就绪，请先启动 Ollama。", kind="fallback")
        yield LLMChunk("", done=True)
        return
    try:
        for chunk in llm.stream_chat(messages):
            yield chunk
    except Exception as exc:
        logger.warning(
            "llm_fallback reason=stream_error model=%s error=%s", llm.model, exc
        )
        yield LLMChunk(fallback_text or "生成失败，已切换为知识库直接引用。", kind="fallback")
        yield LLMChunk("", done=True)

"""专家 Agent 实现：客服 / 物流 / 报价 / 排程 / 质检 / 数据。

结构（生产可用三段式）：
1. 参数提取：LLM 以 JSON 模式从用户请求提取业务参数（只理解，不算数）；
2. 确定性核算：business_tools 纯函数完成运费/成本/排期/统计计算；
3. 流式解释：LLM 基于核算结果生成面向用户的说明（严禁改动数字）。

LLM 不可用时整体降级：直接返回核算原始结果或补参提示，保证业务闭环不中断。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from ..generation.llm import LLMChunk, LLMClient
from . import business_tools
from .prompts_team import (
    ANALYTICS_PARAMS_PROMPT,
    ANALYTICS_PROMPT,
    CUSTOMER_SERVICE_PROMPT,
    LOGISTICS_MISSING,
    LOGISTICS_PARAMS_PROMPT,
    LOGISTICS_PROMPT,
    PRODUCTION_MISSING,
    PRODUCTION_PARAMS_PROMPT,
    PRODUCTION_PROMPT,
    QUALITY_PROMPT,
    QUOTATION_MISSING,
    QUOTATION_PARAMS_PROMPT,
    QUOTATION_PROMPT,
)
from .prompt_store import load_prompt
from .rag import Doc, RAGProvider


@dataclass
class ExpertSpec:
    """专家元信息（路由器与前端列表共用）。"""
    agent_id: str
    name: str
    description: str


def _rag_block(docs: list[Doc]) -> str:
    if not docs:
        return ""
    parts = [
        f"[{i}]（来源：{d.source or '外挂知识库'}）{d.text[:500]}"
        for i, d in enumerate(docs, 1)
    ]
    return "\n\n以下是外挂知识库检索到的参考资料，回答时优先基于资料并标注 [编号]：\n" + "\n\n".join(parts)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class BaseExpert:
    spec: ExpertSpec = None  # type: ignore[assignment]
    system_prompt: str = ""

    def __init__(self, llm: LLMClient, rag: RAGProvider | None = None):
        self.llm = llm
        self.rag = rag

    def _system(self) -> str:
        """系统提示词：优先热读 prompts/team/{id}.md 的 system 节，
        文件缺失回退类内常量。"""
        return load_prompt(self.spec.agent_id, default=self.system_prompt)

    def build_messages(self, query: str, history: list[dict], tool_result: dict | None = None) -> list[dict]:
        system = self._system()
        if tool_result is not None:
            system = system.replace(
                "{tool_result}", json.dumps(tool_result, ensure_ascii=False, indent=2)
            )
        if "{tool_result}" in system:  # 未填充的占位符兜底清理
            system = system.replace("{tool_result}", "{}")
        messages: list[dict] = [{"role": "system", "content": system}]

        docs = []
        if self.rag is not None:
            try:
                docs = self.rag.retrieve(query, top_k=5) or []
            except Exception:
                docs = []
        if docs:
            messages.append({"role": "system", "content": _rag_block(docs)})

        for msg in history[-6:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": query})
        return messages

    def answer(self, query: str, history: list[dict]) -> Iterable[LLMChunk]:
        """流式回答。"""
        messages = self.build_messages(query, history)
        fallback = None
        if self.rag is None:
            fallback = "模型服务未就绪。请启动 Ollama 后重试。"
        try:
            yield from self.llm.stream_chat(messages)
        except Exception:
            yield LLMChunk(fallback or "生成失败，请稍后重试。", kind="fallback")
        yield LLMChunk("", done=True)


class ToolExpert(BaseExpert):
    """带确定性工具的专家：提取参数 → 核算 → 解释。"""

    params_prompt: str = ""
    tool_fn: Callable[..., dict] = None  # type: ignore[assignment]
    required: tuple[str, ...] = ()
    missing_message: str = ""

    def _params_prompt(self) -> str:
        return load_prompt(self.spec.agent_id, default=self.params_prompt, section="params")

    def _missing_message(self) -> str:
        return load_prompt(self.spec.agent_id, default=self.missing_message, section="missing")

    def _extract_params(self, query: str, history: list[dict] | None = None) -> dict:
        """LLM 提取参数。history 传入最近对话，使省略句（如"只要5幅呢"）
        能从上下文继承尺寸/工艺/装裱等已陈述参数。"""
        if not self.llm.is_ready():
            return {}
        prompt = self._params_prompt().replace("{query}", query)
        # 附上最近对话上下文，明确"用户可能省略上文已提过的信息"
        if history:
            recent = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')[:200]}"
                for m in history[-4:]
            )
            prompt = (f"以下是与该用户的近期对话，提取参数时可参考其中已陈述的信息"
                      f"（当前请求可能省略了上文提到的尺寸/工艺/数量等）：\n{recent}\n\n"
                      f"{prompt}")
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format="json_object",
            )
            return _extract_json(raw)
        except Exception:
            return {}

    def _fill_defaults(self, params: dict) -> dict:
        return params

    def _regex_extract(self, query: str) -> dict:
        """确定性参数提取（正则）。子类按需覆写；命中即省一次 LLM 调用。"""
        return {}

    def handle(self, query: str, history: list[dict]) -> Iterable[LLMChunk]:
        params = self._regex_extract(query)
        missing = [k for k in self.required if params.get(k) in (None, "", "null")]
        # 正则没集齐必填项才调 LLM 补齐（带历史上下文，支持省略句），不覆盖正则结果
        if missing and self.llm.is_ready():
            for key, value in self._extract_params(query, history).items():
                if params.get(key) in (None, "", "null"):
                    params[key] = value
        params = self._fill_defaults(params)
        missing = [k for k in self.required if params.get(k) in (None, "", "null")]
        if missing:
            yield LLMChunk(self._missing_message(), kind="content")
            yield LLMChunk("", done=True)
            return
        try:
            result = self.tool_fn(**params)
        except (ValueError, TypeError) as exc:
            yield LLMChunk(f"核算失败：{exc}\n\n{self._missing_message()}", kind="content")
            yield LLMChunk("", done=True)
            return
        except Exception as exc:  # 工具层异常统一兜底
            yield LLMChunk(f"核算出错：{exc}", kind="fallback")
            yield LLMChunk("", done=True)
            return
        messages = self.build_messages(query, history, tool_result=result)
        try:
            yield from self.llm.stream_chat(messages)
        except Exception:
            yield LLMChunk(
                "模型服务未就绪，以下是原始核算结果：\n"
                + json.dumps(result, ensure_ascii=False, indent=2),
                kind="fallback",
            )
        yield LLMChunk("", done=True)

    def answer(self, query: str, history: list[dict]) -> Iterable[LLMChunk]:
        return self.handle(query, history)


class CustomerServiceExpert(BaseExpert):
    spec = ExpertSpec("customer_service", "智能客服助手",
                      "刺绣行业知识问答（工艺/针法/原材料/养护/选购）、文化背景、售前答疑、话术生成。凡涉及知识咨询都选我")
    system_prompt = CUSTOMER_SERVICE_PROMPT


class LogisticsExpert(ToolExpert):
    spec = ExpertSpec("logistics", "物流核算", "运费试算、承运商比选、时效评估、包装建议")
    system_prompt = LOGISTICS_PROMPT
    params_prompt = LOGISTICS_PARAMS_PROMPT
    tool_fn = staticmethod(business_tools.shipping_quote)
    required = ("weight_kg", "dest_zone")
    missing_message = LOGISTICS_MISSING

    # 数字（含小数，克自动转千克）；单位后仅排除英文字母数字（中文语境无词边界）
    WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|千克|公斤|斤|克|g)(?![a-zA-Z0-9])")
    DIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)")
    ZONES = ("华南", "华东", "华北", "西南", "西北", "东北")
    CARRIERS = {"顺丰": "shunfeng", "德邦": "debang", "韵达": "yunda"}
    # 比较连词：出现则视为多家比价，不应锁定某一家承运商
    COMPARE_RE = re.compile(r"和|与|还是|或者|对比|哪个|谁|哪家|vs")

    @staticmethod
    def _regex_extract(query: str) -> dict:
        params: dict = {}
        m = LogisticsExpert.WEIGHT_RE.search(query)
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            params["weight_kg"] = value / 2 if unit == "斤" else (value / 1000 if unit == "克" else value)
        zone = next((z for z in LogisticsExpert.ZONES if z in query), None)
        if zone:
            params["dest_zone"] = zone
        dim = LogisticsExpert.DIM_RE.search(query)
        if dim:
            params["length_cm"], params["width_cm"], params["height_cm"] = (
                float(dim.group(1)), float(dim.group(2)), float(dim.group(3)))
        # 提及多家承运商或在比价语境 → 不锁定 express，输出全列表比选
        mentioned = [k for k in LogisticsExpert.CARRIERS if k in query]
        if mentioned and len(mentioned) == 1 and not LogisticsExpert.COMPARE_RE.search(query):
            params["express"] = LogisticsExpert.CARRIERS[mentioned[0]]
        return params

    @staticmethod
    def _fill_defaults(params: dict) -> dict:
        for key in ("express", "length_cm", "width_cm", "height_cm"):
            if params.get(key) in ("", "null", None):
                params[key] = 0.0 if key != "express" else ""
        return params


class QuotationExpert(ToolExpert):
    spec = ExpertSpec("quotation", "订单报价核算", "按工艺尺寸数量估算工时成本与报价区间（内部）")
    system_prompt = QUOTATION_PROMPT
    params_prompt = QUOTATION_PARAMS_PROMPT
    tool_fn = staticmethod(business_tools.order_quote)
    required = ("width_cm", "height_cm", "craft")
    missing_message = QUOTATION_MISSING

    DIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)")
    QTY_RE = re.compile(r"(\d+)\s*(?:幅|件|套|个|批)")
    CRAFT_MAP = {"手工绣": "hand", "手工": "hand", "半手工": "semi",
                 "半手工绣": "semi", "机绣": "machine", "机器绣": "machine"}

    @staticmethod
    def _regex_extract(query: str) -> dict:
        params: dict = {}
        dim = QuotationExpert.DIM_RE.search(query)
        if dim:
            params["width_cm"], params["height_cm"] = float(dim.group(1)), float(dim.group(2))
        qty = QuotationExpert.QTY_RE.search(query)
        if qty:
            params["quantity"] = int(qty.group(1))
        craft = next((k for k in QuotationExpert.CRAFT_MAP if k in query), None)
        if craft:
            params["craft"] = QuotationExpert.CRAFT_MAP[craft]
        if "装裱" in query:
            params["framing"] = True
        return params

    @staticmethod
    def _fill_defaults(params: dict) -> dict:
        craft = params.get("craft")
        mapping = {"手工": "hand", "半手工": "semi", "机绣": "machine", "机器": "machine"}
        if craft in mapping:
            params["craft"] = mapping[craft]
        if params.get("quantity") in (None, "", "null"):
            params["quantity"] = 1
        if params.get("framing") in (None, "", "null"):
            params["framing"] = False
        return params


class ProductionExpert(ToolExpert):
    spec = ExpertSpec("production", "生产排程", "订单工序拆解、工时估算、交期评估、产能建议")
    system_prompt = PRODUCTION_PROMPT
    params_prompt = PRODUCTION_PARAMS_PROMPT
    tool_fn = staticmethod(business_tools.schedule_estimate)
    required = ("quantity",)
    missing_message = PRODUCTION_MISSING

    QTY_RE = re.compile(r"(\d+)\s*(?:幅|件|套|个|批|单)")
    DIM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)")

    @staticmethod
    def _regex_extract(query: str) -> dict:
        params: dict = {}
        qty = ProductionExpert.QTY_RE.search(query)
        if qty:
            params["quantity"] = int(qty.group(1))
        craft = next((k for k in QuotationExpert.CRAFT_MAP if k in query), None)
        if craft:
            params["craft"] = QuotationExpert.CRAFT_MAP[craft]
        dim = ProductionExpert.DIM_RE.search(query)
        if dim:
            params["width_cm"], params["height_cm"] = float(dim.group(1)), float(dim.group(2))
        return params

    @staticmethod
    def _fill_defaults(params: dict) -> dict:
        craft = params.get("craft")
        mapping = {"手工": "hand", "半手工": "semi", "机绣": "machine", "机器": "machine"}
        if craft in mapping:
            params["craft"] = mapping[craft]
        if params.get("craft") in (None, "", "null"):
            params["craft"] = "machine"
        # 未给尺寸时按默认 40×60cm 估算（不能填 0，否则绣制工时被算成 0）
        for key, default in (("width_cm", 40.0), ("height_cm", 60.0)):
            if params.get(key) in (None, "", "null", 0, 0.0):
                params[key] = default
        if params.get("parallel_workers") in (None, "", "null"):
            params["parallel_workers"] = 0
        return params


class QualityExpert(BaseExpert):
    spec = ExpertSpec("quality", "质检助手",
                      "只诊断已成品的疵点问题（跳线/色差/起皱/抽丝/断线等）并给返工建议。原材料好坏、工艺知识、养护咨询一律归 customer_service，不要选我")
    system_prompt = QUALITY_PROMPT


class AnalyticsExpert(ToolExpert):
    spec = ExpertSpec("analytics", "经营数据助手", "工单量、状态分布、趋势等经营统计问答")
    system_prompt = ANALYTICS_PROMPT
    params_prompt = ANALYTICS_PARAMS_PROMPT
    required: tuple[str, ...] = ()
    missing_message = ""

    def __init__(self, llm: LLMClient, rag: RAGProvider | None, db_path: str):
        super().__init__(llm, rag)
        self.db_path = db_path

    def handle(self, query: str, history: list[dict]) -> Iterable[LLMChunk]:
        days = 30
        params = self._extract_params(query)
        try:
            days = int(params.get("days") or 30)
        except (TypeError, ValueError):
            days = 30
        try:
            result = business_tools.ticket_stats(self.db_path, days=days)
        except Exception as exc:
            yield LLMChunk(f"统计数据读取失败：{exc}", kind="fallback")
            yield LLMChunk("", done=True)
            return
        messages = self.build_messages(query, history, tool_result=result)
        try:
            yield from self.llm.stream_chat(messages)
        except Exception:
            yield LLMChunk(
                "模型服务未就绪，以下是原始统计结果：\n"
                + json.dumps(result, ensure_ascii=False, indent=2),
                kind="fallback",
            )
        yield LLMChunk("", done=True)


EXPERT_CLASSES = {
    "customer_service": CustomerServiceExpert,
    "logistics": LogisticsExpert,
    "quotation": QuotationExpert,
    "production": ProductionExpert,
    "quality": QualityExpert,
    "analytics": AnalyticsExpert,
}

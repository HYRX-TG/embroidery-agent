"""核心数据模型：角色、需求表单、工单、消息、知识条目。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Role(str, Enum):
    """双角色：客户（对外）与员工（对内）。"""
    CUSTOMER = "customer"
    STAFF = "staff"


class Intent(str, Enum):
    """意图分类，用于 Agent 路由。"""
    KNOWLEDGE = "knowledge"      # 行业知识咨询（走 RAG）
    CUSTOM_REQUIREMENT = "custom_requirement"  # 定制需求（走追问循环 + 表单）
    PATTERN_MEANING = "pattern_meaning"  # 纹样寓意
    ORDER_SERVICE = "order_service"  # 订单/售后咨询（行业通用规则，门店规则转人工）
    COMPLAINT = "complaint"      # 异议/投诉
    STAFF_ANALYSIS = "staff_analysis"  # B端：对话复盘
    STAFF_COPYWRITING = "staff_copywriting"  # B端：文案生成
    SMALL_TALK = "small_talk"    # 闲聊
    UNCLEAR = "unclear"          # 无法判断


class TicketStatus(str, Enum):
    PENDING = "pending"          # 待人工跟进
    QUOTING = "quoting"          # 报价中
    CONFIRMED = "confirmed"      # 已确认
    PRODUCING = "producing"      # 生产中
    SHIPPED = "shipped"          # 已发货
    CLOSED = "closed"            # 已关闭


@dataclass
class RequirementForm:
    """定制需求结构化表单（客户需求信息齐全后输出）。"""
    pattern_theme: str = ""       # 图案主题
    size: str = ""                # 尺寸规格
    craft_type: str = ""          # 工艺类型（手工/半手工/机绣）
    quantity: str = ""            # 定制数量
    usage: str = ""               # 使用用途
    budget_range: str = ""        # 预算区间
    deadline: str = ""            # 交付时限
    special_requirements: str = ""  # 特殊要求
    core_preferences: str = ""    # 客户核心偏好

    def missing_fields(self) -> list[str]:
        fields = [
            ("pattern_theme", "图案主题"), ("size", "尺寸规格"),
            ("craft_type", "工艺类型"), ("usage", "使用用途"),
            ("quantity", "定制数量"), ("budget_range", "预算区间"),
            ("deadline", "交付时限"),
        ]
        return [label for key, label in fields if not getattr(self, key).strip()]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ticket:
    """需求工单：知识库无覆盖 / 复杂问题 / 需人工确认时自动创建。"""
    id: str = ""
    role: str = Role.CUSTOMER.value
    customer_summary: str = ""          # 需求摘要
    conversation: list[dict] = field(default_factory=list)  # 关联对话记录
    requirement: Optional[RequirementForm] = None
    status: str = TicketStatus.PENDING.value
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    note: str = ""                      # 需人工跟进备注

    def to_dict(self) -> dict:
        data = asdict(self)
        data["requirement"] = self.requirement.to_dict() if self.requirement else None
        return data


@dataclass
class KnowledgeChunk:
    """知识库切块。"""
    id: str
    text: str
    source: str            # 来源文件
    category: str          # 工艺/针法/品类/保养/纹样寓意/内部
    visibility: str        # public / internal
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "visibility": self.visibility,
            "metadata": self.metadata,
        }


@dataclass
class Message:
    role: str            # user / assistant
    content: str
    kind: str = "content"  # content / reasoning / fallback
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(value: str) -> str:
    """清洗文本：去空白、控制字符。"""
    return re.sub(r"\s+", " ", (value or "").replace("\x00", " ")).strip()


def json_dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)

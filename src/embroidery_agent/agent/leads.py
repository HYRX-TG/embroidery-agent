# -*- coding: utf-8 -*-
"""客户意向分级 Agent（确定性规则，不耗 LLM）。

给工单自动打"高/中/低"意向标签，帮客服排跟进优先级：
- 高：需求已收齐（图案+尺寸）可进入报价，或出现成交信号词；
- 中：有明确定制意向 / 问过价格工期 / 需求收集了一半；
- 低：纯知识咨询（留存线索，暂不跟进）。

规则可审计、可回归：每个判定都有明确 reason。
"""
from __future__ import annotations

import re
from typing import Any

from ..core.schema import RequirementForm

# 成交硬信号：出现即视为高意向
HARD_SIGNALS = [
    "下单", "定金", "汇款", "付款", "成交",
    "现在就要", "先定", "就定了", "确定要", "马上安排",
]
# 意向信号：问价/工期/定制动作
INTENT_SIGNALS = [
    "多少钱", "价格", "报价", "价位", "工期", "要多久", "多久能",
    "定制", "定做", "想绣", "想做", "能做", "能绣", "买个",
    "想买", "要买", "绣幅", "绣朵", "送人", "送长辈",
]
# 明确成交否定（防止"现在就要吗？"这种反问被误判）；只排除问句形态
NEG_SIGNALS = ["？", "?", "吗"]


class LeadScorer:
    name = "lead_score"

    def score(self, requirement: RequirementForm | None, conversation_text: str) -> dict[str, Any]:
        """返回 {level: high|medium|low, reasons: [..], score: 0-100}。"""
        text = conversation_text or ""
        reasons: list[str] = []

        # 1. 成交硬信号（跳过疑问句）
        if not any(k in text for k in NEG_SIGNALS):
            for sig in HARD_SIGNALS:
                if sig in text:
                    reasons.append(f"成交信号「{sig}」")
                    return {"level": "high", "reasons": reasons, "score": 95}

        # 2. 需求完整性：图案+尺寸都齐 → 可报价
        req = requirement or RequirementForm()
        has_theme = bool(req.pattern_theme.strip())
        has_size = bool(req.size.strip())
        if has_theme and has_size:
            reasons.append("需求已收齐（图案+尺寸），可进入报价")
            return {"level": "high", "reasons": reasons, "score": 90}

        # 3. 意向信号（问价/工期/定制）
        hits = [s for s in INTENT_SIGNALS if s in text]
        if hits:
            reasons.append("意向信号：" + "/".join(hits[:3]))
        if has_theme or has_size:
            reasons.append("需求已收集部分字段")
        if hits or has_theme or has_size:
            level = "medium" if (hits or has_theme or has_size) else "low"
            return {"level": level, "reasons": reasons, "score": 60 if hits else 45}

        # 4. 兜底：纯知识咨询
        return {"level": "low", "reasons": ["纯知识咨询，暂不跟进"], "score": 15}

"""提示词注入防御：识别常见越狱/指令覆盖话术，礼貌拒绝并记录。

设计原则：
- 规则层拦截明显的注入模板（零依赖、可解释、可测试）；
- system prompt 中同时写入防御条款（软防线）；
- 命中注入不执行、不回答原问题，统一用国风话术化解，并记入日志供审计。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 注入模式（正则，大小写不敏感）
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"忽略(你?之前的?|以上|上面的?|所有)?(系统)?(指令|提示|设定|规则)", re.IGNORECASE),
    re.compile(r"无视(之前的?|上面|以上|所有)?(系统)?(指令|提示|设定|规则)", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|rules)", re.IGNORECASE),
    re.compile(r"(你|你实际上?)?(现在)?是?(一个)?(普通|无限制|不受限[制]?|没有限制|不受任何限制)(的)?(AI|机器人|助手|模型)", re.IGNORECASE),
    re.compile(r"越狱|jailbreak|DAN模式|开发者模式|上帝模式", re.IGNORECASE),
    re.compile(r"(输出|打印|显示|告诉我)你?的?(系统提示|系统指令|system\s*prompt|初始指令|设定词)", re.IGNORECASE),
    re.compile(r"进入(开发者|调试|维护|内部)模式", re.IGNORECASE),
    re.compile(r"假装你(是|可以)|pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"(透露|泄露|说出|告诉我|说下|说一下|给我看).{0,6}(内部|本店)?(成本|底价|报价表|价格表)", re.IGNORECASE),
]

# 拒绝话术（国风、不生硬、不解释技术细节）
REFUSAL_MESSAGE = (
    "抱歉呀，我只能为您解答刺绣工艺与定制相关的问题~ "
    "如果您有定制需求或想了解刺绣知识，随时告诉我，我会认真为您解答的。"
)


def detect_injection(text: str) -> bool:
    """检测输入是否命中注入模板。"""
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def guard_query(text: str, role: str = "customer") -> str | None:
    """命中注入时返回拒绝话术，否则返回 None。命中会记录审计日志。"""
    if detect_injection(text):
        logger.warning(
            "prompt_injection_blocked role=%s text=%r", role, text[:120]
        )
        return REFUSAL_MESSAGE
    return None

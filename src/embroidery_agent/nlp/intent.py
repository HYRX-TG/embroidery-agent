"""意图识别：基于规则关键词，判断客户/员工请求的意图类型。

规则先行（零依赖、可解释、离线），LLM 路由作为后续增强选项。

判定优先级（C 端客户）：
    异议投诉 > 订单售后 > 价格转人工 > 定制需求 > 纹样寓意 > 知识咨询 > 闲聊
B 端员工：复盘/文案 > 内部知识（工期/运费/报价/投诉等业务词一律视为知识咨询）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.schema import Intent
from .ner import CraftNER


@dataclass
class IntentResult:
    intent: Intent
    confidence: str = "medium"  # high / medium / low
    matched_signals: list[str] = field(default_factory=list)


# 定制需求触发词：只保留明确的"要做东西"动作词。
# 场景词（送礼/家装/婚庆/乔迁等）不再单独触发——客户可能只是问"送什么好"，
# 场景 + 动作/尺寸同时出现才算定制。
CUSTOM_SIGNALS = [
    "定制", "定做", "订做", "想绣", "想做", "打算做",
    "帮我做", "帮我绣", "做一幅", "来一幅", "要一幅", "来一对",
    "专属定制", "企业定制", "定制礼品",
]
# 弱定制信号：不带"绣/定制"等强词也能表达定制意向的口语问法
# （"能做清明上河图吗" "绣朵牡丹送长辈好吗" "买个绣品送人"）
WEAK_CUSTOM_SIGNALS = [
    "能做", "能绣", "可以绣", "能不能绣", "绣得了",
    "买个", "买一幅", "买幅", "买一个", "想买",
    "绣幅", "绣一个", "绣朵", "绣一幅", "做幅", "做一幅图",
    "送长辈", "送人", "送老人", "送妈妈", "送朋友", "送客户",
    "当礼物", "做礼物", "送礼物", "送礼",
]
# 定制类知识问句（命中则视为知识咨询，而非定制意向）：
# "你们能做哪些绣品"是知识，"能做清明上河图吗"是定制意向
CUSTOM_NEG_SIGNALS = ["哪些", "种类", "样式", "都是啥"]
# 弱定制词的咨询性排除（仅抑制弱信号词，不影响"定制/定做"等强词）：
# "刺绣围巾披肩适合送人吗""能绣外国油画风格的作品吗"是知识咨询
WEAK_NEG_SIGNALS = ["适合", "风格", "外国"]
# 尺寸/规格触发词（真正的规格表达；"装裱/画框"是服务话题，不表示定制意图）
SIZE_SIGNALS = ["尺寸", "多大", "几乘几", "规格", "长宽", "厘米", "cm", "乘", "×"]
# 纹样寓意触发词（"题材/绣什么"须排除知识介绍类问句，见 PATTERN_NEG）
PATTERN_SIGNALS = ["寓意", "象征", "什么含义", "代表什么", "兆头", "吉祥", "纹样", "题材", "绣什么"]
# 知识介绍类问句：命中即视为知识咨询，而非纹样寓意（如"湘绣擅长绣什么"）
PATTERN_NEG = ["擅长", "适合绣", "有哪些", "一般", "分类"]
# 订单/售后（含时效与库存，命中即转人工或走行业通用规则）
ORDER_SIGNALS = [
    "发货", "物流", "快递", "运费", "退换", "退货", "退款", "售后", "保修",
    "修复", "多久到", "工期", "发货时间", "要多久", "现货", "库存", "有没有货",
]
# 报价相关（禁止报价，转人工）
PRICE_SIGNALS = ["多少钱", "多钱", "价格", "报价", "价位", "贵不贵", "成本", "优惠", "折扣", "底价", "便宜"]
# 投诉/异议
COMPLAINT_SIGNALS = ["太贵", "贵了", "色差", "不满意", "投诉", "质量差", "太差", "失望", "和图片不符", "有色差", "褪色", "起毛"]
# 知识性问句模板：命中则优先按知识处理，避免"什么是定制肖像绣"被当成定制
KNOWLEDGE_ASK_RE = re.compile(r"什么是|是什么|啥是|是啥|有什么区别|啥区别")
# 员工侧
STAFF_ANALYSIS_SIGNALS = ["复盘", "分析这段对话", "总结客户", "提炼", "成交概率", "跟进建议", "客户意向"]
STAFF_COPYWRITING_SIGNALS = ["文案", "口播", "朋友圈", "介绍", "宣传", "详情页", "话术", "脚本", "合作方案"]
# 真正的问候/寒暄（只有这些才走固定闲聊话术，其余一律交给模型+知识库）
GREETING_SIGNALS = [
    "你好", "您好", "hi", "hello", "在吗", "在么", "你是谁", "介绍一下你",
    "谢谢", "感谢", "再见", "拜拜", "辛苦了", "早上好", "晚上好", "下午好",
]


class IntentClassifier:
    def __init__(self, ner: CraftNER | None = None):
        self.ner = ner or CraftNER()

    def classify(self, text: str, role: str) -> IntentResult:
        """判断意图。

        - C 端：价格/工期/库存等门店专属信息一律转人工（order_service）；
        - B 端（员工）：同一批信号词（工期/运费/报价/投诉）代表"内部知识咨询"，
          不再误判为客户订单或投诉。
        """
        text = text.strip()
        if not text:
            return IntentResult(Intent.UNCLEAR, "low")

        # ---- B 端员工：业务词一律视为内部知识 ----
        if role == "staff":
            if any(k in text for k in STAFF_ANALYSIS_SIGNALS):
                return IntentResult(Intent.STAFF_ANALYSIS, "high", ["员工复盘关键词"])
            if any(k in text for k in STAFF_COPYWRITING_SIGNALS):
                return IntentResult(Intent.STAFF_COPYWRITING, "high", ["文案关键词"])
            entities = self.ner.extract(text)
            if entities:
                return IntentResult(Intent.KNOWLEDGE, "medium", [f"实体:{e.text}" for e in entities[:3]])
            if any(k in text for k in ORDER_SIGNALS) or any(k in text for k in PRICE_SIGNALS) or any(k in text for k in CUSTOM_SIGNALS):
                return IntentResult(Intent.KNOWLEDGE, "medium", ["员工业务咨询"])
            if any(k in text for k in COMPLAINT_SIGNALS):
                return IntentResult(Intent.KNOWLEDGE, "medium", ["员工售后流程咨询"])
            return IntentResult(Intent.KNOWLEDGE, "low", [])

        # ---- C 端客户 ----
        is_definition_question = bool(KNOWLEDGE_ASK_RE.search(text))

        # 异议/投诉（优先级最高，先安抚再处理）
        if any(k in text for k in COMPLAINT_SIGNALS):
            return IntentResult(Intent.COMPLAINT, "high", ["异议投诉关键词"])

        # 订单/售后（含时效、库存；命中即转人工或行业通用规则）
        if any(k in text for k in ORDER_SIGNALS):
            return IntentResult(Intent.ORDER_SERVICE, "high", ["订单售后关键词"])

        # 报价（门店专属信息，一律转人工）
        if any(k in text for k in PRICE_SIGNALS):
            return IntentResult(Intent.ORDER_SERVICE, "high", ["价格咨询，转人工"])

        # 定制 vs 纹样寓意交叉："绣朵牡丹送长辈好吗"是定制意向，
        # "牡丹送长辈有什么寓意"是纹样寓意；寓意问优先于弱定制词，强定制词优先于寓意问
        strong_custom = any(k in text for k in CUSTOM_SIGNALS) or any(k in text for k in SIZE_SIGNALS)
        weak_custom = any(k in text for k in WEAK_CUSTOM_SIGNALS)
        pattern_meaning = any(k in text for k in PATTERN_SIGNALS) and not any(k in text for k in PATTERN_NEG)
        custom_neg = any(k in text for k in CUSTOM_NEG_SIGNALS)
        weak_neg = any(k in text for k in WEAK_NEG_SIGNALS)

        if not is_definition_question:
            if pattern_meaning and not strong_custom:
                return IntentResult(Intent.PATTERN_MEANING, "medium", ["纹样寓意关键词"])
            if strong_custom and not custom_neg:
                return IntentResult(Intent.CUSTOM_REQUIREMENT, "high", ["定制需求关键词"])
            if weak_custom and not custom_neg and not weak_neg:
                return IntentResult(Intent.CUSTOM_REQUIREMENT, "high", ["定制需求关键词"])

        if pattern_meaning:
            return IntentResult(Intent.PATTERN_MEANING, "medium", ["纹样寓意关键词"])

        # 命中了工艺实体（针法/品类/纹样等）→ 知识咨询
        entities = self.ner.extract(text)
        if entities:
            return IntentResult(Intent.KNOWLEDGE, "medium", [f"实体:{e.text}" for e in entities[:3]])

        # 纯问候/寒暄（短句且命中问候词）→ 闲聊
        if len(text) <= 12 and any(k in text.lower() for k in GREETING_SIGNALS):
            return IntentResult(Intent.SMALL_TALK, "medium", ["问候语"])

        # 兜底：未命中任何规则的问题一律走「知识咨询」——由检索 + 本地模型自主回答，
        # 绝不能退化成固定话术（那是"预设"而非"模型推理"）。
        return IntentResult(Intent.KNOWLEDGE, "low", [])

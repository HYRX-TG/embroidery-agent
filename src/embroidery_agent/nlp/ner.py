"""工艺实体识别：基于词典 + 正则，识别针法/品类/纹样/工艺术语。

与 medical_retrieval_plus 的规则 NER 同思路——词典驱动、零依赖、可扩展。
词典后续可替换/扩充为私有知识库提炼的词表。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    text: str
    type: str


# ---- 刺绣行业公开术语词典（示例，可扩充） ----
DEFAULT_LEXICON: dict[str, list[str]] = {
    "针法": [
        "平针", "齐针", "散针", "套针", "戗针", "滚针", "打籽绣", "盘金绣",
        "抢针", "掺针", "乱针", "十字绣", "回针", "长短针", "轮廓绣", "缎面绣",
        "法国结", "锁链绣", "链式绣", "莨绣", "盘银绣", "锁针",
    ],
    "品类": [
        "苏绣", "湘绣", "粤绣", "蜀绣", "顾绣", "京绣", "手工绣", "机绣",
        "半手工绣", "绣品挂画", "服饰刺绣", "摆件", "双面绣", "单面绣",
        "定制礼品", "团扇", "屏风", "荷包", "香囊", "绣花鞋", "刺绣抱枕",
        "刺绣围巾", "婚嫁绣品", "刺绣壁挂", "十字绣成品",
    ],
    "纹样": [
        "牡丹", "荷花", "梅花", "兰花", "竹子", "菊花", "山水", "仙鹤",
        "龙凤呈祥", "喜鹊", "鸳鸯", "锦鲤", "麒麟", "凤凰", "龙纹",
        "福禄寿", "蝙蝠", "松鹤延年", "石榴", "葫芦", "祥云", "如意",
        "百鸟朝凤", "孔雀", "蝴蝶", "金鱼", "梅兰竹菊", "年年有余",
    ],
    "工艺流程": [
        "打版", "配线", "绷框", "上绷", "勾稿", "劈丝", "穿针", "绣制",
        "装裱", "绷布", "拓稿", "描样", "染线", "劈丝线", "底稿",
    ],
    "材料": [
        "真丝", "蚕丝线", "丝线", "棉线", "羊毛线", "绣布", "真丝面料",
        "绸缎", "亚麻", "雪纺", "丝绒", "底布", "绣绷", "绣花针",
    ],
    "工具": ["绣绷", "绣花针", "剪刀", "顶针", "绷架", "针线盒", "穿线器"],
}

# 品类归属正则（把"手工绣XX""机绣XX"归入对应工艺类型）
CRAFT_TYPE_KEYWORDS = {
    "手工": "手工绣",
    "机绣": "机绣",
    "半手工": "半手工绣",
}


class CraftNER:
    """刺绣行业规则 NER。"""

    def __init__(self, lexicon: dict[str, list[str]] | None = None):
        self.lexicon = lexicon or DEFAULT_LEXICON
        self._compiled: dict[str, list[re.Pattern]] = {}
        for entity_type, terms in self.lexicon.items():
            self._compiled[entity_type] = [
                re.compile(re.escape(term)) for term in terms
            ]

    def extract(self, text: str) -> list[Entity]:
        """抽取文本中的工艺实体，保持出现顺序。"""
        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()
        for entity_type, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    match = pattern.search(text)
                    key = (match.group(0), entity_type)
                    if key not in seen:
                        seen.add(key)
                        entities.append(Entity(match.group(0), entity_type))
        return entities

    def extract_craft_type(self, text: str) -> str:
        """判断客户提到的工艺类型：手工绣/半手工绣/机绣。"""
        for keyword, craft in CRAFT_TYPE_KEYWORDS.items():
            if keyword in text:
                return craft
        return ""


def format_entities(entities: list[Entity]) -> str:
    return ", ".join(f"{e.text}/{e.type}" for e in entities)

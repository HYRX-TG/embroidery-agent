"""客户意向分级 Agent 单元测试：确定性规则回归。"""
from embroidery_agent.agent.leads import LeadScorer
from embroidery_agent.core.schema import RequirementForm


def _score(text: str = "", theme: str = "", size: str = "") -> dict:
    req = RequirementForm(pattern_theme=theme, size=size)
    return LeadScorer().score(req, text)


def test_full_requirement_is_high():
    r = _score(text="我想定制牡丹绣品，60乘80，麻烦给个价", theme="牡丹", size="60×80cm")
    assert r["level"] == "high"
    assert r["score"] >= 80


def test_hard_signal_is_high():
    r = _score(text="这个定金怎么付，我准备下单了")
    assert r["level"] == "high"
    assert "成交信号" in "".join(r["reasons"])


def test_intent_only_is_medium():
    r = _score(text="定制的绣品多少钱")
    assert r["level"] == "medium"


def test_half_requirement_is_medium():
    r = _score(text="想要牡丹的", theme="牡丹")
    assert r["level"] == "medium"


def test_pure_knowledge_is_low():
    r = _score(text="苏绣和湘绣有什么区别")
    assert r["level"] == "low"


def test_question_about_buying_not_hard():
    # "现在就要吗？"是反问，不能误判为成交
    r = _score(text="你们现在就要定金吗？")
    assert r["level"] != "high"

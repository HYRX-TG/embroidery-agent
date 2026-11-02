"""会话落库单元测试：跨实例（模拟重启/多 worker）恢复。"""
import tempfile
from pathlib import Path

from embroidery_agent.agent.customer import CustomerAgent
from embroidery_agent.agent.ticket import TicketStore
from embroidery_agent.generation.llm import LLMClient


def _new_agent(db: Path, tickets: TicketStore) -> CustomerAgent:
    return CustomerAgent(
        retriever=None,
        llm=LLMClient("http://127.0.0.1:11434/v1", "unused"),
        tickets=tickets,
        db_path=db,
    )


def test_session_survives_restart():
    db = Path(tempfile.mkdtemp()) / "sessions.db"
    tickets = TicketStore(db)
    agent_a = _new_agent(db, tickets)
    s = agent_a.get_session("sess-1")
    s.append("user", "我想定制一幅牡丹挂画")
    s.append("assistant", "请问尺寸多少呢？")
    s.requirement.pattern_theme = "牡丹"
    s.collecting = True
    agent_a._save_session(s)

    # 模拟重启：全新实例从同一 DB 恢复
    agent_b = _new_agent(db, tickets)
    s2 = agent_b.get_session("sess-1")
    assert [m.content for m in s2.conversation] == ["我想定制一幅牡丹挂画", "请问尺寸多少呢？"]
    assert s2.requirement.pattern_theme == "牡丹"
    assert s2.collecting is True


def test_session_continues_across_instances():
    db = Path(tempfile.mkdtemp()) / "sessions.db"
    tickets = TicketStore(db)
    agent_a = _new_agent(db, tickets)
    s = agent_a.get_session("sess-2")
    s.append("user", "60×80cm")
    agent_a._save_session(s)

    agent_b = _new_agent(db, tickets)
    s2 = agent_b.get_session("sess-2")
    s2.append("user", "手工绣")
    agent_b._save_session(s2)

    agent_c = _new_agent(db, tickets)
    s3 = agent_c.get_session("sess-2")
    assert [m.content for m in s3.conversation] == ["60×80cm", "手工绣"]


def test_new_session_is_empty():
    db = Path(tempfile.mkdtemp()) / "sessions.db"
    tickets = TicketStore(db)
    agent = _new_agent(db, tickets)
    s = agent.get_session("brand-new")
    assert s.conversation == []
    assert s.collecting is False

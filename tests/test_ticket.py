"""工单存储单元测试：重点回归 updated_at 修复。"""
import tempfile
from pathlib import Path

from embroidery_agent.agent.ticket import TicketStore
from embroidery_agent.core.schema import RequirementForm, Ticket, TicketStatus, now_iso


def _store() -> TicketStore:
    return TicketStore(Path(tempfile.mkdtemp()) / "tickets.db")


def _make_ticket(tid: str, theme: str = "牡丹") -> Ticket:
    return Ticket(
        id=tid,
        role="customer",
        customer_summary="",
        requirement=RequirementForm(pattern_theme=theme),
        conversation=[],
        status=TicketStatus.PENDING,
        note="",
        created_at=now_iso(),
        updated_at=now_iso(),
    )


def test_create_ticket_has_valid_dates():
    store = _store()
    ticket = store.create(_make_ticket("t-1"))
    assert ticket.created_at
    assert ticket.updated_at
    assert ticket.updated_at >= ticket.created_at


def test_update_status_refreshes_updated_at():
    store = _store()
    ticket = store.create(_make_ticket("t-2"))
    before = ticket.updated_at
    updated = store.update_status(ticket.id, TicketStatus.QUOTING)
    assert updated is not None
    # 修复回归：updated_at 必须是时间戳字符串，而不是状态值
    assert updated.updated_at != TicketStatus.QUOTING.value
    assert updated.status == TicketStatus.QUOTING.value
    assert updated.updated_at >= before


def test_append_message_updates_conversation():
    store = _store()
    ticket = store.create(_make_ticket("t-3"))
    store.append_message(ticket.id, {"role": "assistant", "content": "您好"})
    after = store.get(ticket.id)
    assert after is not None
    assert len(after.conversation) == 1
    assert after.conversation[0]["content"] == "您好"
    # 时间戳为秒级精度，同秒内更新可能相同；断言格式合法即可
    assert len(after.updated_at) >= 19


def test_update_note_persists():
    store = _store()
    ticket = store.create(_make_ticket("t-4"))
    updated = store.update_note(ticket.id, "待顾问确认报价")
    assert updated is not None
    assert updated.note == "待顾问确认报价"
    reloaded = store.get(ticket.id)
    assert reloaded is not None
    assert reloaded.note == "待顾问确认报价"


def test_time_format_iso():
    assert len(now_iso()) >= 19

"""工单存储：SQLite 持久化，本地零依赖。

工单状态机：pending → quoting → confirmed → producing → shipped → closed。
对话记录自动附到工单，便于人工跟进。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from ..core.schema import RequirementForm, Ticket, TicketStatus, now_iso


class TicketStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    customer_summary TEXT NOT NULL DEFAULT '',
                    requirement TEXT,
                    conversation TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)"
            )

    def create(self, ticket: Ticket) -> Ticket:
        if not ticket.id:
            ticket.id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tickets (id, role, customer_summary, requirement, conversation, status, note, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket.id,
                    ticket.role,
                    ticket.customer_summary,
                    json.dumps(ticket.requirement.to_dict(), ensure_ascii=False) if ticket.requirement else None,
                    json.dumps(ticket.conversation, ensure_ascii=False),
                    ticket.status,
                    ticket.note,
                    ticket.created_at,
                    ticket.updated_at,
                ),
            )
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_ticket(row)

    def append_message(self, ticket_id: str, message: dict) -> None:
        ticket = self.get(ticket_id)
        if ticket is None:
            return
        ticket.conversation.append(message)
        ticket.updated_at = now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE tickets SET conversation = ?, updated_at = ? WHERE id = ?",
                (json.dumps(ticket.conversation, ensure_ascii=False), ticket.updated_at, ticket_id),
            )

    def list_all(self, limit: int = 50) -> list[Ticket]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_ticket(row) for row in rows]

    def update_note(self, ticket_id: str, note: str) -> Ticket | None:
        """更新工单跟进备注并刷新 updated_at（人工跟进时按时间排序）。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE tickets SET note = ?, updated_at = ? WHERE id = ?",
                (note, now_iso(), ticket_id),
            )
        return self.get(ticket_id)

    def update_status(self, ticket_id: str, status: TicketStatus) -> Ticket | None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now_iso(), ticket_id),
            )
        return self.get(ticket_id)

    @staticmethod
    def _row_to_ticket(row: sqlite3.Row) -> Ticket:
        requirement_raw = row["requirement"]
        requirement = (
            RequirementForm(**json.loads(requirement_raw))
            if requirement_raw
            else None
        )
        return Ticket(
            id=row["id"],
            role=row["role"],
            customer_summary=row["customer_summary"],
            requirement=requirement,
            conversation=json.loads(row["conversation"]),
            status=row["status"],
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

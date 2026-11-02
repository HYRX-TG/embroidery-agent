# -*- coding: utf-8 -*-
"""渠道接入层：把知绣坊客服接到微信/直播等外部平台。

设计要点
- 每个平台一个 ChannelAdapter，负责 ①GET 验签 ②解析入站消息 ③异步推送回复；
- 平台回调一律先返回空串（微信要求 5 秒内应答），回答通过客服消息异步推送；
- openid↔session 绑定与消息幂等去重落在 SQLite（与业务同库）；
- 未配置平台推送凭证（app_id/secret）时进入"本地模式"：回复写入待推送表，
  便于本地全链路测试，不真正外发。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    channel: str
    external_id: str      # 平台侧用户标识（openid / 主播uid）
    content: str
    msg_id: str = ""      # 平台消息ID（用于幂等）
    raw: dict = field(default_factory=dict)


class ChannelStore:
    """渠道侧持久化：绑定、幂等、待推送回复。与业务库同文件。"""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS channel_bindings (
                    channel TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (channel, external_id)
                );
                CREATE TABLE IF NOT EXISTS channel_msg_ids (
                    channel TEXT NOT NULL,
                    msg_id TEXT NOT NULL,
                    processed_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (channel, msg_id)
                );
                CREATE TABLE IF NOT EXISTS channel_pending_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    pushed_at TEXT
                );
                """
            )

    def get_or_create_session(self, channel: str, external_id: str) -> str:
        """openid ↔ 会话绑定：同一用户在平台侧始终是同一会话。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM channel_bindings WHERE channel=? AND external_id=?",
                (channel, external_id),
            ).fetchone()
            if row:
                return row["session_id"]
            session_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO channel_bindings (channel, external_id, session_id) VALUES (?, ?, ?)",
                (channel, external_id, session_id),
            )
            return session_id

    def is_duplicate(self, channel: str, msg_id: str) -> bool:
        if not msg_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM channel_msg_ids WHERE channel=? AND msg_id=?",
                (channel, msg_id),
            ).fetchone()
            return row is not None

    def mark_processed(self, channel: str, msg_id: str) -> None:
        if not msg_id:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO channel_msg_ids (channel, msg_id) VALUES (?, ?)",
                (channel, msg_id),
            )

    def queue_reply(self, channel: str, external_id: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO channel_pending_replies (channel, external_id, content) VALUES (?, ?, ?)",
                (channel, external_id, content[:4000]),
            )

    def pending_replies(self, channel: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT * FROM channel_pending_replies"
        args: list = []
        if channel:
            sql += " WHERE channel=?"
            args.append(channel)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def mark_pushed(self, reply_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE channel_pending_replies SET status='pushed', pushed_at=datetime('now') WHERE id=?",
                (reply_id,),
            )


class ChannelAdapter:
    """平台适配基类：子类实现验签/解析/推送。"""

    name = "base"

    def __init__(self, config: dict, store: ChannelStore):
        self.config = config or {}
        self.store = store

    def verify(self, params: dict) -> str | None:
        """GET 验签：平台接入时回调的校验请求。返回原样 echostr 或 None。"""
        raise NotImplementedError

    def parse_incoming(self, body: bytes) -> IncomingMessage | None:
        """解析平台入站消息（XML/JSON）→ 内部消息对象。"""
        raise NotImplementedError

    def push_reply(self, external_id: str, content: str) -> bool:
        """异步推送回复到平台；本地模式返回 False（回复已入待推送表）。"""
        return False

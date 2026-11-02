# -*- coding: utf-8 -*-
"""渠道注册与 webhook 分发入口。"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..agent.customer import CustomerAgent
from .base import ChannelAdapter, ChannelStore
from .wechat import WeChatAdapter

LOGGER = logging.getLogger(__name__)

# 平台名 → 适配器类
_ADAPTERS: dict[str, type[ChannelAdapter]] = {
    "wechat": WeChatAdapter,
}


class ChannelHub:
    """渠道中枢：按配置创建适配器，处理平台回调并驱动客服回答。"""

    def __init__(
        self,
        channels_cfg: dict[str, dict],
        customer: CustomerAgent,
        db_path: Path | str,
        max_workers: int = 4,
    ):
        self.customer = customer
        self.store = ChannelStore(db_path)
        self.adapters: dict[str, ChannelAdapter] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="channel"
        )
        for name, cfg in (channels_cfg or {}).items():
            cls = _ADAPTERS.get(name)
            if not cls:
                LOGGER.warning("未知渠道 %s，跳过", name)
                continue
            if not cfg.get("enabled", True):
                continue
            self.adapters[name] = cls(cfg, self.store)
            LOGGER.info("渠道 %s 已加载", name)

    def adapter(self, name: str) -> ChannelAdapter | None:
        return self.adapters.get(name)

    def handle_webhook(self, channel: str, params: dict, body: bytes) -> str:
        """处理平台回调：GET 验签返回 echostr；POST 异步应答并立即返回空串。"""
        adapter = self.adapter(channel)
        if adapter is None:
            raise KeyError(channel)

        if body:
            msg = adapter.parse_incoming(body)
            if msg is None:
                return ""
            if not msg.content:
                return ""
            if self.store.is_duplicate(channel, msg.msg_id):
                LOGGER.info("渠道 %s 重复消息 %s，跳过", channel, msg.msg_id)
                return ""
            self.store.mark_processed(channel, msg.msg_id)
            self._executor.submit(self._answer, adapter, msg)
            return ""
        return adapter.verify(params) or ""

    def _answer(self, adapter: ChannelAdapter, msg) -> None:
        """后台处理一条平台消息：绑定会话 → 生成回答 → 推送/入待推送表。"""
        try:
            session_id = self.store.get_or_create_session(msg.channel, msg.external_id)
            collected: list[str] = []
            for chunk in self.customer.process(session_id, msg.content):
                if getattr(chunk, "kind", "") in ("content", "fallback") and chunk.text:
                    collected.append(chunk.text)
            answer = "".join(collected).strip()
            if not answer:
                return
            pushed = adapter.push_reply(msg.external_id, answer)
            self.store.queue_reply(msg.channel, msg.external_id, answer)
            LOGGER.info(
                "渠道 %s 用户 %s 已应答（%d 字，推送=%s）",
                msg.channel, msg.external_id, len(answer), pushed,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("渠道 %s 应答失败: %s", msg.channel, exc, exc_info=True)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

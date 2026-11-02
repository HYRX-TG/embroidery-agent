# -*- coding: utf-8 -*-
"""微信公众号/订阅号渠道适配器。

接入形态
- 公众号后台「服务器配置」→ URL 填 {公网}/api/v1/channels/wechat/webhook，Token 填配置值；
- GET：平台验签（SHA1(token, timestamp, nonce) 排序拼接）；
- POST：接收文本消息 XML → 解析 → 幂等 → 绑定会话 → 异步回复；
- 回复：公众号被动回复限 5 秒（本地 7B 模型必然超时），因此一律走
  「客服消息」接口（48 小时窗口）异步推送；未配置 app_id/secret 时进入
  本地模式：回复写入待推送表，由运营侧或测试脚本轮询外发。
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from .base import ChannelAdapter, ChannelStore, IncomingMessage

LOGGER = logging.getLogger(__name__)


class WeChatAdapter(ChannelAdapter):
    name = "wechat"

    def __init__(self, config: dict, store: ChannelStore):
        super().__init__(config, store)
        self.token = str(config.get("token", ""))
        self.app_id = str(config.get("app_id", ""))
        self.app_secret = str(config.get("app_secret", ""))
        self.verify_signature = bool(config.get("verify_signature", True))
        self._token_cache: dict[str, Any] = {"token": "", "expires_at": 0}
        self._lock = threading.Lock()

    # ---- 微信接入校验 ----
    def verify(self, params: dict) -> str | None:
        signature = params.get("signature", "")
        timestamp = params.get("timestamp", "")
        nonce = params.get("nonce", "")
        echostr = params.get("echostr", "")
        if not self.verify_signature:
            return echostr or None
        if not (signature and timestamp and nonce and echostr):
            return None
        if self._sha1(self.token, timestamp, nonce) == signature:
            return echostr
        return None

    @staticmethod
    def _sha1(token: str, timestamp: str, nonce: str) -> str:
        parts = "".join(sorted([token, timestamp, nonce]))
        return hashlib.sha1(parts.encode("utf-8")).hexdigest()

    # ---- 入站解析 ----
    def parse_incoming(self, body: bytes) -> IncomingMessage | None:
        """解析微信文本消息 XML（容忍换行/缩进/CDATA）。"""
        try:
            root = ET.fromstring(body.decode("utf-8", errors="replace"))
        except ET.ParseError:
            return None

        def g(tag: str) -> str:
            el = root.find(tag)
            return (el.text or "").strip() if el is not None else ""

        if g("MsgType") != "text":
            return None
        content = g("Content")
        if not content:
            return None
        return IncomingMessage(
            channel=self.name,
            external_id=g("FromUserName"),
            content=content,
            msg_id=g("MsgId"),
            raw={"to": g("ToUserName"), "create_time": g("CreateTime")},
        )

    # ---- 异步回复 ----
    def push_reply(self, external_id: str, content: str) -> bool:
        """客服消息推送。未配置凭证时返回 False（回复已入待推送表）。"""
        if not (self.app_id and self.app_secret):
            return False
        token = self._access_token()
        if not token:
            LOGGER.warning("微信 access_token 获取失败，回复保留在待推送表")
            return False
        try:
            resp = requests.post(
                f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}",
                json={
                    "touser": external_id,
                    "msgtype": "text",
                    "text": {"content": content[:4000]},
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("errcode") == 0:
                return True
            LOGGER.warning("微信客服消息发送失败: %s", data)
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("微信客服消息发送异常: %s", exc)
            return False

    def _access_token(self) -> str:
        now = time.time()
        with self._lock:
            if self._token_cache["token"] and self._token_cache["expires_at"] > now + 60:
                return self._token_cache["token"]
            try:
                resp = requests.get(
                    "https://api.weixin.qq.com/cgi-bin/token",
                    params={
                        "grant_type": "client_credential",
                        "appid": self.app_id,
                        "secret": self.app_secret,
                    },
                    timeout=10,
                )
                data = resp.json()
                token = data.get("access_token", "")
                if token:
                    self._token_cache = {
                        "token": token,
                        "expires_at": now + int(data.get("expires_in", 7200)),
                    }
                    return token
                LOGGER.warning("获取微信 access_token 失败: %s", data)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("获取微信 access_token 异常: %s", exc)
            return ""

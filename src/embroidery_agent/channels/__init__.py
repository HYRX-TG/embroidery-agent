# -*- coding: utf-8 -*-
"""渠道接入层：把知绣坊客服接到微信/直播等外部平台。"""
from .base import ChannelAdapter, ChannelStore, IncomingMessage
from .hub import ChannelHub
from .wechat import WeChatAdapter

__all__ = ["ChannelAdapter", "ChannelStore", "IncomingMessage", "ChannelHub", "WeChatAdapter"]

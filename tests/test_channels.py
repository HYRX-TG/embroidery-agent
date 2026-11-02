"""渠道接入层单元测试：XML 解析 / 验签 / 绑定 / 幂等 / 本地模式。"""
import hashlib
import tempfile
from pathlib import Path

from embroidery_agent.channels.base import ChannelStore
from embroidery_agent.channels.wechat import WeChatAdapter

WX_MSG = """<xml>
  <ToUserName><![CDATA[gh_xxx]]></ToUserName>
  <FromUserName><![CDATA[openid_abc123]]></FromUserName>
  <CreateTime>1348831860</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[苏绣有什么特点]]></Content>
  <MsgId>1234567890123456</MsgId>
</xml>"""


def _store() -> ChannelStore:
    return ChannelStore(Path(tempfile.mkdtemp()) / "ch.db")


def _adapter(store: ChannelStore, verify: bool = False) -> WeChatAdapter:
    return WeChatAdapter(
        {"token": "test-token", "verify_signature": verify, "app_id": "", "app_secret": ""},
        store,
    )


def test_parse_wechat_text_message():
    msg = _adapter(_store()).parse_incoming(WX_MSG.encode("utf-8"))
    assert msg is not None
    assert msg.external_id == "openid_abc123"
    assert msg.content == "苏绣有什么特点"
    assert msg.msg_id == "1234567890123456"


def test_parse_non_text_message_returns_none():
    xml = WX_MSG.replace("text", "image")
    assert _adapter(_store()).parse_incoming(xml.encode("utf-8")) is None


def test_parse_empty_content_returns_none():
    xml = WX_MSG.replace("苏绣有什么特点", "")
    assert _adapter(_store()).parse_incoming(xml.encode("utf-8")) is None


def test_parse_garbage_returns_none():
    assert _adapter(_store()).parse_incoming(b"not xml at all") is None


def test_verify_signature_ok():
    ts, nonce = "1726000000", "nonce123"
    signature = hashlib.sha1(
        "".join(sorted(["test-token", ts, nonce])).encode("utf-8")
    ).hexdigest()
    reply = _adapter(_store(), verify=True).verify(
        {"signature": signature, "timestamp": ts, "nonce": nonce, "echostr": "echo-ok"}
    )
    assert reply == "echo-ok"


def test_verify_signature_bad():
    assert _adapter(_store(), verify=True).verify(
        {"signature": "deadbeef", "timestamp": "1", "nonce": "2", "echostr": "x"}
    ) is None


def test_verify_skipped_in_local_mode():
    assert _adapter(_store(), verify=False).verify({"echostr": "local-echo"}) == "local-echo"


def test_binding_same_user_same_session():
    store = _store()
    s1 = store.get_or_create_session("wechat", "openid_a")
    s2 = store.get_or_create_session("wechat", "openid_a")
    s3 = store.get_or_create_session("wechat", "openid_b")
    assert s1 == s2
    assert s1 != s3


def test_msg_id_dedup():
    store = _store()
    assert not store.is_duplicate("wechat", "111")
    store.mark_processed("wechat", "111")
    assert store.is_duplicate("wechat", "111")


def test_push_reply_local_mode_false():
    # 未配置 app_id/secret → 本地模式，不真正推送
    store = _store()
    assert _adapter(store).push_reply("openid_a", "你好") is False


def test_queue_and_list_replies():
    store = _store()
    store.queue_reply("wechat", "openid_a", "第一条回答")
    store.queue_reply("wechat", "openid_a", "第二条回答")
    rows = store.pending_replies(channel="wechat")
    assert len(rows) == 2
    assert rows[0]["content"] == "第二条回答"
    assert rows[0]["status"] == "pending"

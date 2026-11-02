import os
# -*- coding: utf-8 -*-
"""渠道端到端冒烟：模拟微信服务器回调，验证 验签→解析→绑定→回答→待推送 全链路。"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
DB = Path("D:/embroidery_agent/data/embroidery.db")
TOKEN = os.environ.get("ZHIXIU_SMOKE_TOKEN", "")  # 本地联调用，勿写死真实 token

passed = failed = 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print("[PASS] " + name + ((" | " + extra) if extra else ""))
    else:
        failed += 1
        print("[FAIL] " + name + ((" | " + extra) if extra else ""))


def wx_xml(to, frm, content, msgid):
    return f"""<xml>
  <ToUserName><![CDATA[{to}]]></ToUserName>
  <FromUserName><![CDATA[{frm}]]></FromUserName>
  <CreateTime>1726000000</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[{content}]]></Content>
  <MsgId>{msgid}</MsgId>
</xml>"""


def pending_count():
    conn = sqlite3.connect(DB)
    n = conn.execute("SELECT COUNT(*) FROM channel_pending_replies").fetchone()[0]
    conn.close()
    return n


# 1. GET 验签（verify_signature=false，应直接回显 echostr）
r = requests.get(BASE + "/api/v1/channels/wechat/webhook", params={"echostr": "echo-hello"}, timeout=10)
check("GET 验签返回 echostr", r.status_code == 200 and r.text == "echo-hello", f"status={r.status_code} text={r.text!r}")

# 2. 未知渠道 404
r = requests.post(BASE + "/api/v1/channels/douyin/webhook", data=b"<xml/>", timeout=10)
check("未知渠道 404", r.status_code == 404, str(r.status_code))

# 3. 首次消息：问知识问题，立即返回空串（微信要求 5 秒应答）
msgid1 = "9000000000000001"
xml1 = wx_xml("gh_xxx", "openid_wx_user_001", "苏绣和湘绣有什么区别", msgid1)
t0 = time.time()
r = requests.post(BASE + "/api/v1/channels/wechat/webhook", data=xml1.encode("utf-8"), timeout=15)
latency = time.time() - t0
check("POST 消息立即返回空串", r.status_code == 200 and r.text == "", f"status={r.status_code} 响应耗时={latency:.2f}s")
check("响应在 5 秒内（微信硬限制）", latency < 5.0, f"{latency:.2f}s")

# 4. 轮询待推送表（LLM 生成需要几秒）
deadline = time.time() + 90
reply_found = False
while time.time() < deadline:
    rows = pending_count()
    if rows >= 1:
        reply_found = True
        break
    time.sleep(2)
check("回答已生成并入待推送表", reply_found, f"pending={pending_count()}")

# 5. 重复消息（同 MsgId）→ 幂等不重复处理
before = pending_count()
r = requests.post(BASE + "/api/v1/channels/wechat/webhook", data=xml1.encode("utf-8"), timeout=15)
time.sleep(3)
after = pending_count()
check("同 MsgId 重复消息不重复应答", r.status_code == 200 and after == before, f"before={before} after={after}")

# 6. 第二个用户 → 独立会话（绑定验证）
msgid2 = "9000000000000002"
xml2 = wx_xml("gh_xxx", "openid_wx_user_002", "牡丹纹样有什么寓意", msgid2)
r = requests.post(BASE + "/api/v1/channels/wechat/webhook", data=xml2.encode("utf-8"), timeout=15)
deadline = time.time() + 90
found2 = False
while time.time() < deadline:
    if pending_count() >= before + 1:
        found2 = True
        break
    time.sleep(2)
check("第二个用户独立会话已应答", found2, f"pending={pending_count()}")

# 7. 绑定表：两个 openid 对应两个不同 session
conn = sqlite3.connect(DB)
rows = conn.execute(
    "SELECT external_id, session_id FROM channel_bindings ORDER BY external_id"
).fetchall()
conn.close()
check("openid 绑定 2 个独立会话", len(rows) == 2, str(rows))
if len(rows) == 2:
    check("会话 ID 不同", rows[0][1] != rows[1][1])

# 8. 回答内容抽查
conn = sqlite3.connect(DB)
latest = conn.execute(
    "SELECT content FROM channel_pending_replies ORDER BY id DESC LIMIT 1"
).fetchone()
conn.close()
check("回答含知识内容", latest is not None and len(latest[0]) > 20, (latest[0][:60] + "…") if latest else "")

print("")
print(f"结果: {passed} 通过, {failed} 失败")

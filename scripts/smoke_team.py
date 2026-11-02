"""多 Agent 团队冒烟测试（不依赖 LLM）：

    cd D:\\embroidery_agent
    $env:PYTHONPATH = "src"
    python scripts/smoke_team.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PASSED = 0
FAILED = 0


def staff_passcode() -> str:
    """从配置读取员工口令（与运行时一致，不硬编码）。"""
    cfg = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    return str(cfg.get("staff", {}).get("passcode", "") or "")


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def project_rag_ready() -> bool:
    """探测本地 embedding 服务是否可达（决定是否跑真实 RAG 检索）。"""
    try:
        import httpx
        return httpx.get("http://127.0.0.1:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def main() -> int:
    print("== 1. 模块导入 ==")
    try:
        from embroidery_agent.agent import business_tools
        from embroidery_agent.agent.rag import NullRAGProvider, load_rag_provider
        from embroidery_agent.agent.experts import EXPERT_CLASSES
        from embroidery_agent.agent.team import TeamOrchestrator
        from embroidery_agent.agent.prompt_store import load_prompt
        from embroidery_agent.agent import team_rag
        from embroidery_agent.api.server import create_app
        check("全部团队模块可导入", True)
    except Exception as exc:
        check("全部团队模块可导入", False, str(exc))
        return 1

    print("== 1.5 正则参数提取 ==")
    from embroidery_agent.agent.experts import LogisticsExpert, QuotationExpert, ProductionExpert
    p = LogisticsExpert._regex_extract("2.5kg发华东，用韵达，尺寸50x40x30cm")
    check("物流正则: 重量", abs(p.get("weight_kg", 0) - 2.5) < 1e-9, str(p))
    check("物流正则: 分区", p.get("dest_zone") == "华东", str(p))
    check("物流正则: 承运商", p.get("express") == "yunda", str(p))
    check("物流正则: 尺寸", p.get("length_cm") == 50 and p.get("height_cm") == 30, str(p))
    p2 = LogisticsExpert._regex_extract("3斤到东北")
    check("物流正则: 斤转kg", abs(p2.get("weight_kg", 0) - 1.5) < 1e-9, str(p2))
    p3 = QuotationExpert._regex_extract("30x40cm手工绣12幅含装裱报个价")
    check("报价正则: 尺寸/工艺/数量/装裱",
          p3.get("width_cm") == 30 and p3.get("craft") == "hand"
          and p3.get("quantity") == 12 and p3.get("framing") is True, str(p3))
    p4 = ProductionExpert._regex_extract("100幅机绣订单多久能交")
    check("排程正则: 数量+工艺", p4.get("quantity") == 100 and p4.get("craft") == "machine", str(p4))
    p5 = LogisticsExpert._regex_extract("帮我算运费")
    check("物流正则: 无参不误伤", p5 == {}, str(p5))

    print("== 2. 物流运费试算 ==")
    q = business_tools.shipping_quote(weight_kg=2.5, dest_zone="华东")
    check("返回多个承运商比选", len(q["quotes"]) == 3, str(q))
    check("推荐为费用最低者", q["recommended"] == q["quotes"][0]["carrier"])
    # 2.5kg，首重1kg → 续重1.5kg；韵达 8 + 1.5*1.5 = 10.25
    yunda = next(x for x in q["quotes"] if x["carrier_key"] == "yunda")
    check("韵达 2.5kg 华东 = 10.25 元", abs(yunda["estimated_fee"] - 10.25) < 1e-6, str(yunda))
    # 体积重：50×40×40/6000 = 13.33 > 2.5
    q2 = business_tools.shipping_quote(weight_kg=2.5, dest_zone="华北",
                                       length_cm=50, width_cm=40, height_cm=40)
    check("体积重参与计费重", abs(q2["chargeable_weight_kg"] - 13.33) < 0.01, str(q2["chargeable_weight_kg"]))
    try:
        business_tools.shipping_quote(weight_kg=1, dest_zone="火星")
        check("非法分区报错", False)
    except ValueError:
        check("非法分区报错", True)

    print("== 3. 订单报价核算 ==")
    r = business_tools.order_quote(width_cm=30, height_cm=40, craft="hand", quantity=12)
    check("单价区间低于总价区间比例", r["total_price_range"][0] > r["unit_price_range"][0] * 12 * 0.99)
    check("12件触发95折", r["discount"] == 0.95)
    r2 = business_tools.order_quote(width_cm=30, height_cm=40, craft="machine", quantity=1, framing=True)
    check("装裱费计入成本", r2["unit_cost"] > business_tools.order_quote(30, 40, "machine", 1)["unit_cost"])
    try:
        business_tools.order_quote(width_cm=-5, height_cm=40)
        check("非法尺寸报错", False)
    except ValueError:
        check("非法尺寸报错", True)

    print("== 4. 生产排程估算 ==")
    s = business_tools.schedule_estimate(quantity=10, craft="machine")
    check("四道工序齐全", [p["name"] for p in s["processes"]] == ["打版", "绣制", "后整", "质检"])
    check("估算天数为正", s["estimated_days"] > 0)
    s2 = business_tools.schedule_estimate(quantity=10, craft="hand")
    check("手工绣总工时大于机绣", s2["total_hours"] > s["total_hours"])

    print("== 5. 工单统计（只读） ==")
    db = PROJECT_ROOT / "data" / "embroidery.db"
    if db.exists():
        st = business_tools.ticket_stats(db, days=30)
        check("统计返回结构完整", "total" in st and "by_status" in st)
    else:
        print("  [SKIP] 工单库不存在，跳过")

    print("== 6. RAG 外挂接口 ==")
    check("NullRAG 返回空", NullRAGProvider().retrieve("测试") == [])
    check("空配置加载返回 None", load_rag_provider("") is None)
    try:
        load_rag_provider("badpath")
        check("无效路径报错", False)
    except (ValueError, ModuleNotFoundError, AttributeError):
        check("无效路径报错", True)

    print("== 6.5 提示词外置热更新 ==")
    sys_prompt = load_prompt("logistics", default="fallback")
    check("logistics.md system 节热读", "物流核算专员" in sys_prompt)
    default = load_prompt("不存在的专家", default="内置兜底", section="params")
    check("缺文件回退内置", default == "内置兜底")
    params_prompt = load_prompt("quotation", default="", section="params")
    check("quotation.md params 节热读", "width_cm" in params_prompt)

    print("== 6.6 团队 RAG 检索 ==")
    if project_rag_ready():
        hits = None
        try:
            from embroidery_agent.agent.team_rag import VectorRAGProvider
            provider = VectorRAGProvider()
            hits = provider.retrieve("苏绣有什么特点", top_k=3)
            check("RAG 检索返回命中", bool(hits), str(hits)[:150])
            check("命中带来源与分数", bool(hits) and all(h.source and h.score for h in hits))
        except Exception as exc:
            check("RAG 检索返回命中", False, str(exc))
    else:
        print("  [SKIP] 本地 embedding 服务不可用，跳过 RAG 检索")

    print("== 7. FastAPI 路由注册 ==")
    from fastapi.testclient import TestClient  # noqa
    app = create_app()
    routes = {r.path for r in app.routes}
    for p in ("/team", "/api/v1/team/agents", "/api/v1/team/chat",
              "/api/v1/tools/logistics", "/api/v1/tools/quotation", "/api/v1/tools/schedule"):
        check(f"端点 {p} 已注册", p in routes)
    with TestClient(app) as client:
        passcode = staff_passcode()
        staff_headers = {"x-staff-access": passcode}

        r_agents_noauth = client.get("/api/v1/team/agents")
        check("团队专家列表无口令 403", r_agents_noauth.status_code == 403, r_agents_noauth.text[:120])
        agents = client.get("/api/v1/team/agents", headers=staff_headers).json()["agents"]
        check("专家列表 6 人", len(agents) == 6, str(agents))
        check("含物流与报价", {a["id"] for a in agents} >= {"logistics", "quotation"})
        # 无 LLM 时团队对话：路由应走关键词兜底，工具专家应输出补参提示
        resp = client.post("/api/v1/team/chat", json={"content": "帮我算运费", "session_id": "t1"},
                           headers=staff_headers)
        check("团队对话端点 200", resp.status_code == 200, resp.text[:200])
        check("SSE 含 agent 事件", "event: agent" in resp.text)
        check("关键词路由命中物流", '"id": "logistics"' in resp.text or '"id":"logistics"' in resp.text)
        # 团队对话为员工工具，无口令应 403
        resp_noauth = client.post("/api/v1/team/chat", json={"content": "帮我算运费", "session_id": "t2"})
        check("团队对话无口令 403", resp_noauth.status_code == 403, resp_noauth.text[:200])

        # 员工会话与内部接口鉴权（回归：防内部知识泄露）
        r_staff_noauth = client.post("/api/v1/staff/chat", json={"content": "手工绣工期标准"})
        check("员工会话无口令 403", r_staff_noauth.status_code == 403, r_staff_noauth.text[:200])
        r_search_staff = client.get("/search", params={"q": "内部标准", "role": "staff"})
        check("search(role=staff) 无口令 403", r_search_staff.status_code == 403, r_search_staff.text[:200])
        r_search_cust = client.get("/search", params={"q": "苏绣特点", "role": "customer"})
        check("search(role=customer) 可访问", r_search_cust.status_code == 200, r_search_cust.text[:120])
        r_kb_build = client.post("/api/v1/kb/build")
        check("重建索引无口令 403", r_kb_build.status_code == 403, r_kb_build.text[:200])
        r_status_noauth = client.get("/api/v1/status")
        check("系统状态无口令 403", r_status_noauth.status_code == 403, r_status_noauth.text[:200])
        r_metrics_noauth = client.get("/api/v1/metrics")
        check("监控指标无口令 403", r_metrics_noauth.status_code == 403, r_metrics_noauth.text[:200])
        r_metrics = client.get("/api/v1/metrics", headers=staff_headers)
        check("监控指标带口令 200", r_metrics.status_code == 200, r_metrics.text[:200])
        check("监控含请求计数", '"requests"' in r_metrics.text)
        # 员工会话带口令应可用（SSE 200）
        r_staff_ok = client.post("/api/v1/staff/chat", json={"content": "苏绣有什么特点"}, headers=staff_headers)
        check("员工会话带口令 200", r_staff_ok.status_code == 200, r_staff_ok.text[:120])

        # 工具直调（带口令）
        r3 = client.post("/api/v1/tools/logistics",
                         json={"weight_kg": 2, "dest_zone": "华南"},
                         headers=staff_headers)
        check("运费直调 200", r3.status_code == 200, r3.text[:200])
        r4 = client.post("/api/v1/tools/logistics", json={"weight_kg": 2, "dest_zone": "华南"})
        check("无口令 403", r4.status_code == 403)
        r5 = client.post("/api/v1/tools/schedule", json={"quantity": 5},
                         headers=staff_headers)
        check("排程直调 200", r5.status_code == 200, r5.text[:200])

    print(f"\n结果: {PASSED} 通过, {FAILED} 失败")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

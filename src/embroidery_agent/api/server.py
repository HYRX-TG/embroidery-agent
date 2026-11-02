"""FastAPI 服务：双角色（客户/员工）+ SSE 流式 + 工单查询 + 多 Agent 团队。

端点：
- GET  /health                        健康检查
- GET  /api/v1/status                 运行时状态（LLM/知识库/索引）
- GET  /search?q=&role=               检索（返回原始命中，便于调试）
- POST /api/v1/customer/chat          客户流式对话（SSE）
- POST /api/v1/customer/history       历史会话回显
- POST /api/v1/staff/chat             员工流式对话（SSE）
- GET  /api/v1/tickets                工单列表
- POST /api/v1/kb/build               重建知识库索引（需要时手动触发）
- GET  /team                          团队工作台页面
- GET  /api/v1/team/agents            专家 Agent 列表
- POST /api/v1/team/chat              多 Agent 团队对话（自动路由，SSE）
- POST /api/v1/tools/logistics        运费试算（确定性工具，不经 LLM）
- POST /api/v1/tools/quotation        报价核算（确定性工具，不经 LLM）
- POST /api/v1/tools/schedule         排程估算（确定性工具，不经 LLM）
- GET/POST /api/v1/channels/{ch}/webhook   外部平台消息回调（微信等）
- GET  /api/v1/metrics                进程监控指标（员工口令）
"""
from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..channels.hub import ChannelHub
from ..agent.leads import LeadScorer
from .rate_limit import SlidingWindowLimiter
from pydantic import BaseModel, Field

from ..agent.router import AgentRuntime, build_runtime
from ..agent import business_tools
from ..core.schema import Role, json_dumps
from ..generation.llm import LLMChunk

# 团队/路由结构化日志（root 无 handler 时才生效，不覆盖 uvicorn 自身日志）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    session_id: str = ""
    agent: str = ""  # 指定专家 id；留空 = 自动路由


class LogisticsToolRequest(BaseModel):
    weight_kg: float = Field(gt=0)
    dest_zone: str
    express: str = ""
    length_cm: float = 0
    width_cm: float = 0
    height_cm: float = 0


class QuotationToolRequest(BaseModel):
    width_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    craft: str = "machine"
    quantity: int = Field(default=1, ge=1)
    framing: bool = False


class ScheduleToolRequest(BaseModel):
    quantity: int = Field(ge=1)
    craft: str = "machine"
    width_cm: float = 40
    height_cm: float = 60
    parallel_workers: int = 0


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def staff_passcode(runtime: AgentRuntime) -> str:
    """员工口令：环境变量 EMBROIDERY_STAFF_PASSCODE 优先，其次 configs/default.json 的 staff.passcode。

    生产部署必须用环境变量注入，避免明文口令落盘。
    """
    configured = str(runtime.settings.config.get("staff", {}).get("passcode", "") or "")
    return os.environ.get("EMBROIDERY_STAFF_PASSCODE", "") or configured


def _setup_log_file() -> None:
    """日志落盘到项目根 logs/app.log（同时保留控制台输出）。"""
    try:
        # server.py 位于 src/embroidery_agent/api/server.py → 项目根 = parents[2].parent
        log_dir = (Path(__file__).resolve().parents[2]).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
    except OSError:
        # 日志目录不可写时静默降级为仅控制台，不影响服务运行
        pass


def require_staff(runtime: AgentRuntime, x_staff_access: str | None) -> None:
    """统一员工鉴权：口令缺失或错误一律 403。

    口令来源优先级：环境变量 EMBROIDERY_STAFF_PASSCODE > 配置 staff.passcode > 配置 staff.passcode_sha256。
    前两者用 hmac.compare_digest 常量时间比较；哈希配置比对输入的 SHA-256，避免配置文件明文落盘。
    """
    env_passcode = os.environ.get("EMBROIDERY_STAFF_PASSCODE", "")
    config = runtime.settings.config.get("staff", {})
    configured = str(config.get("passcode", "") or "")
    configured_sha256 = str(config.get("passcode_sha256", "") or "")
    provided = x_staff_access or ""

    ok = False
    if env_passcode:
        ok = hmac.compare_digest(provided, env_passcode)
    elif configured:
        ok = hmac.compare_digest(provided, configured)
    elif configured_sha256:
        digest = hashlib.sha256(provided.encode("utf-8")).hexdigest()
        ok = hmac.compare_digest(digest, configured_sha256.lower())

    if (env_passcode or configured or configured_sha256) and not ok:
        raise HTTPException(status_code=403, detail="无权限访问内部信息")


def chunk_stream(chunks) -> AsyncIterator[str]:
    async def generator():
        for chunk in chunks:
            if chunk.kind in ("content", "fallback"):
                yield sse("delta", {"text": chunk.text, "kind": chunk.kind})
            elif chunk.kind == "reasoning":
                yield sse("reasoning_delta", {"text": chunk.text})
            if chunk.done:
                yield sse("done", {})
    return generator()


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    _setup_log_file()
    rt = runtime or build_runtime()
    app = FastAPI(title="知绣坊 · 刺绣企业本地智能客服")

    # 外部平台渠道（微信等）：回调由平台签名鉴权，openid 绑定会话
    hub = ChannelHub(
        channels_cfg=rt.settings.config.get("channels", {}),
        customer=rt.customer,
        db_path=rt.customer.db_path,
    )

    @app.on_event("shutdown")
    def _shutdown_hub():
        hub.close()

    # 轻量进程监控：请求/错误计数（/api/v1/metrics 员工鉴权可查）
    proc_metrics = {"requests": 0, "errors": 0, "started_at": __import__("time").time()}

    # 速率限制（单进程滑动窗口；配置见 default.json 的 rate_limit 段）
    rl_cfg = rt.settings.config.get("rate_limit", {})
    limiter = SlidingWindowLimiter(
        limits={
            "chat": int(rl_cfg.get("chat_per_minute", 10)),
            "search": int(rl_cfg.get("search_per_minute", 30)),
        }
        if rl_cfg.get("enabled", True)
        else {},
    )

    def check_rate(request: Request, bucket: str) -> None:
        """按客户端 IP 限流；超限抛 429。"""
        ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(bucket, ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"请求太频繁啦，请约 {retry_after} 秒后再试~",
                headers={"Retry-After": str(retry_after)},
            )

    @app.middleware("http")
    async def count_requests(request, call_next):
        proc_metrics["requests"] += 1
        try:
            return await call_next(request)
        except Exception:
            proc_metrics["errors"] += 1
            raise

    @app.get("/api/v1/metrics")
    def metrics(x_staff_access: str | None = Header(default=None)):
        """进程监控指标（运行时长/请求数/错误数），内部信息需员工口令。"""
        require_staff(rt, x_staff_access)
        import time
        proc_metrics["uptime_seconds"] = round(time.time() - proc_metrics["started_at"], 1)
        return proc_metrics

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/status")
    def status(x_staff_access: str | None = Header(default=None)):
        """系统状态（模型/知识库/索引）为内部信息，员工口令校验。"""
        require_staff(rt, x_staff_access)
        try:
            rt.load_kb()
        except Exception:
            pass
        return rt.status()

    @app.get("/api/v1/tickets")
    def tickets(x_staff_access: str | None = Header(default=None)):
        """工单列表为内部信息，员工口令校验；每条附加客户意向分级（确定性规则）。"""
        require_staff(rt, x_staff_access)
        try:
            items = rt.tickets.list_all(50)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"工单读取失败：{exc}") from exc
        scorer = LeadScorer()
        result = []
        for t in items:
            d = t.to_dict()
            conv_text = " ".join(
                m.get("content", "") for m in (d.get("conversation") or [])
            )
            d["lead"] = scorer.score(t.requirement, conv_text)
            result.append(d)
        return {"tickets": result}

    @app.api_route("/api/v1/channels/{channel}/webhook", methods=["GET", "POST"])
    async def channel_webhook(channel: str, request: Request):
        """外部平台消息回调：GET 验签（echostr），POST 收消息异步应答。

        平台侧鉴权靠渠道签名；回复走客服消息异步推送，未配置凭证时入待推送表。
        """
        body = await request.body()
        try:
            reply = hub.handle_webhook(channel, dict(request.query_params), body)
        except KeyError:
            raise HTTPException(status_code=404, detail="未知渠道")
        return PlainTextResponse(reply)

    @app.get("/api/v1/customer/history")
    def customer_history(session_id: str = ""):
        """历史会话回显：按 session_id 返回已落库的消息（session_id 即顾客会话凭证）。"""
        if not session_id or len(session_id) > 64:
            return {"messages": []}
        return {"messages": rt.customer.list_messages(session_id)}

    @app.get("/search")
    def search(q: str, role: str = "customer", x_staff_access: str | None = Header(default=None), request: Request = None):
        """检索调试接口。role=staff 会返回内部知识库内容，必须员工口令校验。"""
        if role == "staff":
            require_staff(rt, x_staff_access)
        if request is not None:
            check_rate(request, "search")
        role_enum = Role.CUSTOMER if role != "staff" else Role.STAFF
        try:
            rt.load_kb()
        except Exception:
            pass
        hits = rt.retriever.search(q, role=role_enum)
        return {
            "query": q,
            "role": role_enum.value,
            "hits": [h.chunk.to_dict() | {"score": round(h.score, 4), "sources": h.sources} for h in hits],
        }

    @app.post("/api/v1/customer/chat")
    def customer_chat(payload: ChatRequest, request: Request):
        session_id = payload.session_id or uuid.uuid4().hex[:12]
        # HTTP header 仅支持 latin-1，非 ASCII 的 session_id 会使响应头编码失败
        try:
            session_id.encode("ascii")
        except UnicodeEncodeError:
            session_id = uuid.uuid4().hex[:12]
        check_rate(request, "chat")
        try:
            rt.load_kb()
        except Exception:
            pass
        if not rt.kb.chunks:
            raise HTTPException(status_code=503, detail="知识库未构建，请先执行 python scripts/build_kb.py")
        chunks = rt.customer.process(session_id, payload.content)
        return StreamingResponse(
            chunk_stream(chunks),
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )

    @app.post("/api/v1/staff/chat")
    def staff_chat(payload: ChatRequest, x_staff_access: str | None = Header(default=None), request: Request = None):
        """员工会话涉及内部知识库，必须员工口令校验。"""
        require_staff(rt, x_staff_access)
        if request is not None:
            check_rate(request, "chat")
        try:
            rt.load_kb()
        except Exception:
            pass
        if not rt.kb.chunks:
            raise HTTPException(status_code=503, detail="知识库未构建，请先执行 python scripts/build_kb.py")
        chunks = rt.staff.process(payload.content)
        return StreamingResponse(chunk_stream(chunks), media_type="text/event-stream")

    @app.post("/api/v1/kb/build")
    def kb_build(x_staff_access: str | None = Header(default=None)):
        """重建知识库索引为运维操作，必须员工口令校验。"""
        require_staff(rt, x_staff_access)
        try:
            stats = rt.build_kb()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"知识库构建失败：{exc}") from exc
        return {"ok": True, "stats": stats}

    # ---------------- 多 Agent 团队 ----------------
    def _require_staff(x_staff_access: str | None) -> None:
        """工具直调与内部信息一致，均需员工口令。"""
        require_staff(rt, x_staff_access)

    @app.get("/team")
    def team_page():
        return FileResponse(WEB_DIR / "team.html")

    @app.get("/api/v1/team/agents")
    def team_agents(x_staff_access: str | None = Header(default=None)):
        require_staff(rt, x_staff_access)
        return {"agents": rt.team.list_agents()}

    @app.get("/api/v1/team/status")
    def team_status(x_staff_access: str | None = Header(default=None)):
        """团队运行时状态（含 RAG 索引），内部信息需员工口令。"""
        require_staff(rt, x_staff_access)
        rag_stats = {}
        try:
            rag = getattr(rt.team, "rag", None)
            if rag is not None and hasattr(rag, "stats"):
                rag_stats = rag.stats()
        except Exception:
            rag_stats = {"error": "RAG 状态读取失败"}
        return {"agents": rt.team.list_agents(), "rag": rag_stats}

    @app.post("/api/v1/team/chat")
    def team_chat(payload: ChatRequest, x_staff_access: str | None = Header(default=None), request: Request = None):
        """团队专家含内部核算（报价/排程/经营数据），必须员工口令校验。"""
        require_staff(rt, x_staff_access)
        if request is not None:
            check_rate(request, "chat")
        session_id = payload.session_id or uuid.uuid4().hex[:12]
        try:
            agent_id, chunks = rt.team.process(
                session_id, payload.content, agent_id=payload.agent
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"团队调度失败：{exc}") from exc

        def generator():
            # 首个事件告知前端本次接单的专家（前端可展示徽标）
            try:
                agent = next(a for a in rt.team.list_agents() if a["id"] == agent_id)
                yield sse("agent", {"id": agent_id, "name": agent["name"], "session_id": session_id})
            except StopIteration:
                yield sse("agent", {"id": agent_id, "name": agent_id, "session_id": session_id})
            try:
                for chunk in chunks:
                    if chunk.kind in ("content", "fallback"):
                        yield sse("delta", {"text": chunk.text, "kind": chunk.kind})
                    elif chunk.kind == "reasoning":
                        yield sse("reasoning_delta", {"text": chunk.text})
                    if chunk.done:
                        yield sse("done", {})
                        return
            except Exception as exc:  # 流中途异常：向客户端明确报错而非悬挂
                yield sse("error", {"message": f"生成中断：{exc}"})
                yield sse("done", {})

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )

    @app.post("/api/v1/tools/logistics")
    def tool_logistics(payload: LogisticsToolRequest, x_staff_access: str | None = Header(default=None)):
        _require_staff(x_staff_access)
        try:
            return business_tools.shipping_quote(**payload.model_dump())
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/tools/quotation")
    def tool_quotation(payload: QuotationToolRequest, x_staff_access: str | None = Header(default=None)):
        _require_staff(x_staff_access)
        try:
            return business_tools.order_quote(**payload.model_dump())
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/tools/schedule")
    def tool_schedule(payload: ScheduleToolRequest, x_staff_access: str | None = Header(default=None)):
        _require_staff(x_staff_access)
        try:
            return business_tools.schedule_estimate(**payload.model_dump())
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()

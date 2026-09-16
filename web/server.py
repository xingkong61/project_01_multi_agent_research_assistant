"""
web/server.py — 多智能体研究助手 · FastAPI 服务（L2 部署）

对外提供：
  GET  /                古典风格前端（静态托管 web/static）
  GET  /api/health      健康检查（供 Docker/Nginx 探活，无需鉴权）
  POST /api/research    提交研究问题，SSE 流式返回实时轨迹与最终报告

复用根目录 main.stream_research() 这一流式核心，与 CLI/Streamlit 共享同一套图逻辑。
安全：HTTP Basic Auth（WEB_USERNAME/WEB_PASSWORD）+ 按 IP 令牌桶限流。
长任务在 L2 仍为同步流式（依赖 Nginx proxy_read_timeout 放宽），异步队列留待 L3。

启动：
    uvicorn web.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# 允许以 `uvicorn web.server:app` 从项目根运行；也兼容直接 python web/server.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import preflight_check, stream_research
from web.auth import auth_configured, get_limiter, verify_credentials

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Multi-Agent Research Assistant", version="2.1")

security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)):
    """未配置口令时放行（本地开发）；配置后强制 Basic Auth。"""
    if not auth_configured():
        return None
    ok = credentials is not None and verify_credentials(
        credentials.username.decode("utf-8", "ignore") if isinstance(credentials.username, bytes) else credentials.username,
        credentials.password.decode("utf-8", "ignore") if isinstance(credentials.password, bytes) else credentials.password,
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="需要访问口令",
            headers={"WWW-Authenticate": 'Basic realm="Research"'},
        )
    return credentials


class ResearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    model: str | None = None
    max_iterations: int = Field(default=4, ge=1, le=6)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='5' fill='#7c2d12'/>"
    "<text x='16' y='23' font-size='18' text-anchor='middle' "
    "font-family='Georgia,serif' fill='#f4ecd8'>研</text></svg>"
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # 内联 SVG 图标，消除浏览器默认 favicon 请求的 404 噪音
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
async def index():
    path = STATIC_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="前端资源缺失")
    return FileResponse(path)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/research", dependencies=[Depends(require_auth)])
async def research(req: ResearchRequest, request: Request):
    # 限流：每客户端 IP 令牌桶
    if not get_limiter().allow(_client_ip(request)):
        return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)

    # 预检：LLM 配置缺失快速失败（不进入流）
    problems = preflight_check(req.model, quiet=True)
    hard = [p for p in problems if not p.startswith("[提示]")]
    if hard:
        return JSONResponse({"detail": "服务端配置不完整：" + "；".join(hard)}, status_code=500)

    # 把阻塞式的生成器放到线程池，避免卡住事件循环
    iterator = stream_research(req.query.strip(), req.model, req.max_iterations)

    async def event_stream():
        loop = asyncio.get_event_loop()
        it = iter(iterator)
        try:
            while True:
                nxt = await loop.run_in_executor(None, lambda: next(it, None))
                if nxt is None:
                    break
                kind, payload = nxt
                if kind == "trace":
                    yield _sse("trace", {"line": payload})
                elif kind == "status":
                    yield _sse("status", payload)
                elif kind == "done":
                    report = payload.get("report_markdown", "")
                    findings = payload.get("findings", {})
                    n_sources = sum(len(f.get("sources", [])) for f in findings.values())
                    yield _sse("result", {
                        "report": report,
                        "tasks": len(payload.get("tasks", [])),
                        "iterations": payload.get("iteration", 0),
                        "sources": n_sources,
                        "errors": payload.get("errors", []),
                    })
        except Exception as exc:  # noqa: BLE001 — 流内异常必须以事件告知前端
            from agents import describe_exc
            yield _sse("error", {"detail": describe_exc(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 对该响应的缓冲，保证实时推送
        },
    )


# 静态资源（css/js）挂载在最后，避免拦截上面的 API 路由
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

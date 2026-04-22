"""
Web 后端 - FastAPI 服务
提供聊天 API 和记忆树浏览接口，配合前端网页使用
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from yggdrasil.agent_loop import AgentLoop, AgentEvent
from yggdrasil.memory_engine import MemoryEngine
from yggdrasil.database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────────────

DB_PATH = os.environ.get("YGGDRASIL_DB_PATH", str(Path(__file__).parent / "yggdrasil.db"))
WORKSPACE_ROOT = os.environ.get("YGGDRASIL_WORKSPACE", str(Path.home() / "yggdrasil_workspace"))
MODEL_NAME = os.environ.get("YGGDRASIL_MODEL", "deepseek-chat")
KEYS_PATH = os.environ.get(
    "PORTABLE_LLM_KEYS_PATH",
    str(Path(__file__).resolve().parent.parent / "portable_llm" / "keys.yaml"),
)

app = FastAPI(title="Yggdrasil 世界树", version="0.1.0")

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 全局 Agent 实例
agent: AgentLoop | None = None


def get_agent() -> AgentLoop:
    global agent
    if agent is None:
        agent = AgentLoop(
            model_name=MODEL_NAME,
            keys_path=KEYS_PATH,
            db_path=DB_PATH,
            workspace_root=WORKSPACE_ROOT,
        )
    return agent


# ── REST API ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model: str | None = None


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    events: list[dict]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步聊天接口 (用于简单调用)"""
    a = get_agent()
    if req.model and req.model != a.client.model_name:
        a.client = __import__("portable_llm", fromlist=["UnifiedLLMClient"]).UnifiedLLMClient(
            model_name=req.model,
            keys_path=KEYS_PATH,
        )

    events = []
    a.on_event(lambda e: events.append({"type": e.type, "data": e.data}))

    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(None, a.run, req.message)

    a._event_callbacks.clear()
    return ChatResponse(
        response=response_text,
        tokens_used=a.total_tokens_used,
        events=events,
    )


@app.get("/api/memory")
async def memory_tree(node_name: str | None = None):
    """获取记忆树节点信息"""
    a = get_agent()
    return a.get_memory_tree(node_name)


@app.get("/api/memory/full_tree")
async def full_tree():
    """获取完整记忆树结构 (递归)"""
    a = get_agent()

    def build_tree(name: str, depth: int = 0) -> dict:
        if depth > 8:
            return {"name": name, "children": []}
        try:
            ctx = a.engine.retrieve_context(name)
            return {
                "name": ctx["node"]["name"],
                "content_preview": ctx["node"]["content"][:100],
                "weight": ctx["node"]["current_weight"],
                "k_value": ctx["node"]["k_value"],
                "children": [build_tree(c["name"], depth + 1) for c in ctx["children"]],
                "edges": ctx["outgoing_edges"],
            }
        except Exception:
            return {"name": name, "children": []}

    return build_tree("root")


@app.get("/api/state")
async def agent_state():
    """获取 Agent 当前状态"""
    a = get_agent()
    return a.get_state()


@app.post("/api/reboot")
async def reboot_agent():
    """强制重启 Agent 上下文"""
    a = get_agent()
    a._tool_reboot_context()
    return {"status": "rebooted"}


# ── WebSocket (实时事件流) ──────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """WebSocket 聊天接口 - 实时推送 Agent 事件"""
    await ws.accept()
    a = get_agent()

    async def send_event(event: AgentEvent):
        try:
            await ws.send_json({
                "type": event.type,
                "data": event.data,
                "timestamp": event.timestamp,
            })
        except Exception:
            pass

    # 因为 agent_loop.run 是同步的，需要桥接到 async
    loop = asyncio.get_event_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def sync_event_handler(event: AgentEvent):
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    a._event_callbacks.clear()
    a.on_event(sync_event_handler)

    try:
        while True:
            # 等待用户消息
            data = await ws.receive_json()
            user_msg = data.get("message", "")
            if not user_msg:
                continue

            # 在线程中运行 agent loop
            agent_task = loop.run_in_executor(None, a.run, user_msg)

            # 同时转发事件
            done = False
            while not done:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    await send_event(event)
                except asyncio.TimeoutError:
                    if agent_task.done():
                        done = True

            # 发送最终结果
            final = await agent_task

            # 清空剩余事件
            while not event_queue.empty():
                event = event_queue.get_nowait()
                await send_event(event)

            await ws.send_json({
                "type": "final_response",
                "data": {"text": final, "tokens": a.total_tokens_used},
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        a._event_callbacks.clear()


# ── 前端页面 ────────────────────────────────────────────

@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Yggdrasil 世界树</h1><p>前端文件未找到</p>")


# 挂载静态资源
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn
    port = int(os.environ.get("YGGDRASIL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

"""
Web 后端 - FastAPI 服务 (多对话版)
支持多对话、持久化、后台执行、实时事件流
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from yggdrasil.conversation_manager import ConversationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────────────

DATA_DIR = os.environ.get("YGGDRASIL_DATA_DIR", str(Path(__file__).parent / "data"))
WORKSPACE_ROOT = os.environ.get("YGGDRASIL_WORKSPACE", str(Path.home() / "yggdrasil_workspace"))
DEFAULT_MODEL = os.environ.get("YGGDRASIL_MODEL", "deepseek-chat")
KEYS_PATH = os.environ.get(
    "PORTABLE_LLM_KEYS_PATH",
    str(Path(__file__).resolve().parent.parent / "portable_llm" / "keys.yaml"),
)

# 全局对话管理器
manager: ConversationManager | None = None


def get_manager() -> ConversationManager:
    global manager
    if manager is None:
        manager = ConversationManager(
            data_dir=DATA_DIR,
            keys_path=KEYS_PATH,
            workspace_root=WORKSPACE_ROOT,
        )
    return manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if manager:
        await manager.shutdown()


app = FastAPI(title="Yggdrasil 世界树", version="0.4.0", lifespan=lifespan)

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ── 对话管理 REST API ──────────────────────────────────

class CreateConversationRequest(BaseModel):
    model: str = "deepseek-chat"
    title: str | None = None
    config: dict | None = None  # 可选覆盖: {"warn_ratio": 0.1, "reboot_ratio": 0.2}


class SendMessageRequest(BaseModel):
    message: str
    model: str | None = None
    images: list[str] | None = None  # base64 data URL 列表（多模态图片输入）


@app.post("/api/conversations")
async def create_conversation(req: CreateConversationRequest):
    m = get_manager()
    info = m.create_conversation(model=req.model, title=req.title, config=req.config)
    return {"id": info.id, "title": info.title, "model": info.model, "created_at": info.created_at}


@app.get("/api/conversations")
async def list_conversations():
    m = get_manager()
    convs = m.list_conversations()
    return [
        {"id": c.id, "title": c.title, "model": c.model, "created_at": c.created_at, "updated_at": c.updated_at, "status": c.status}
        for c in convs
    ]


@app.get("/api/conversations/{conv_id}")
async def get_conversation_detail(conv_id: str):
    m = get_manager()
    info = m.get_conversation(conv_id)
    if not info:
        raise HTTPException(404, "对话不存在")
    return {"id": info.id, "title": info.title, "model": info.model, "created_at": info.created_at, "updated_at": info.updated_at, "status": info.status}


@app.patch("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, req: dict):
    m = get_manager()
    m.update_conversation(conv_id, **req)
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    m = get_manager()
    m.delete_conversation(conv_id)
    return {"ok": True}


# ── 聊天 API ──────────────────────────────────────────

@app.post("/api/conversations/{conv_id}/messages")
async def send_message(conv_id: str, req: SendMessageRequest):
    m = get_manager()
    info = m.get_conversation(conv_id)
    if not info:
        raise HTTPException(404, "对话不存在")

    model = req.model or info.model
    status = await m.send_message(conv_id, req.message, model, images=req.images)

    if status == "agent_busy":
        raise HTTPException(409, "Agent 正在执行中，请等待完成")

    return {"status": status, "conversation_id": conv_id}


# ── SSE 流式聊天 API ──────────────────────────────────

@app.post("/api/conversations/{conv_id}/messages/stream")
async def send_message_stream(conv_id: str, req: SendMessageRequest):
    """SSE 流式返回 Agent 响应。每个事件为 'data: {JSON}\n\n' 格式。"""
    m = get_manager()
    info = m.get_conversation(conv_id)
    if not info:
        raise HTTPException(404, "对话不存在")

    model = req.model or info.model

    async def event_generator():
        try:
            async for event in m.send_message_stream(conv_id, req.message, model, images=req.images):
                payload = json.dumps(event, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"
        except Exception as e:
            error_payload = json.dumps({"event": "error", "data": {"error": str(e)}}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, limit: int = 200):
    m = get_manager()
    history = m.get_chat_history(conv_id, limit)
    return history


@app.get("/api/conversations/{conv_id}/status")
async def get_conv_status(conv_id: str):
    m = get_manager()
    return m.get_status(conv_id)


@app.get("/api/conversations/{conv_id}/events")
async def get_events(conv_id: str, since: int = 0):
    m = get_manager()
    events = m.get_event_log(conv_id, since_line=since)
    return {"events": events, "total": len(events)}


# ── 记忆树 API ────────────────────────────────────────

@app.get("/api/conversations/{conv_id}/memory")
async def memory_node(conv_id: str, node_name: str | None = None):
    m = get_manager()
    return m.get_memory_tree(conv_id, node_name)


@app.get("/api/conversations/{conv_id}/memory/full_tree")
async def full_tree(conv_id: str):
    m = get_manager()
    return m.get_full_tree(conv_id)


# ── WebSocket (实时事件流) ──────────────────────────────

@app.websocket("/ws/chat/{conv_id}")
async def websocket_chat(ws: WebSocket, conv_id: str):
    await ws.accept()
    m = get_manager()

    info = m.get_conversation(conv_id)
    if not info:
        await ws.send_json({"type": "error", "data": {"error": "对话不存在"}})
        await ws.close()
        return

    event_queue = m.connect_ws(conv_id)
    if not event_queue:
        await ws.send_json({"type": "error", "data": {"error": "无法连接对话"}})
        await ws.close()
        return

    # 发送当前状态
    status = m.get_status(conv_id)
    await ws.send_json({"type": "status", "data": status})

    async def forward_events():
        try:
            while True:
                evt = await event_queue.get()
                try:
                    await ws.send_json(evt)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    forward_task = asyncio.create_task(forward_events())

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                user_msg = data.get("message", "")
                model = data.get("model") or info.model
                images = data.get("images")  # 可选: base64 data URL 列表
                if not user_msg:
                    continue
                status = await m.send_message(conv_id, user_msg, model, images=images)
                if status == "agent_busy":
                    await ws.send_json({"type": "error", "data": {"error": "Agent 正在执行中"}})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conv_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {conv_id}: {e}")
    finally:
        forward_task.cancel()
        m.disconnect_ws(conv_id, event_queue)


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

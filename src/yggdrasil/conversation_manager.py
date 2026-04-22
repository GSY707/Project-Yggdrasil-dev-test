"""
对话管理器 - 管理多对话生命周期
每个对话拥有独立的记忆树 DB、聊天历史、事件日志。
Agent 运行为服务端任务，WebSocket 断开不影响执行。
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .async_agent_loop import AgentLoop, AgentEvent

logger = logging.getLogger(__name__)

# ── 元数据库 Schema ──────────────────────────────────

META_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '新对话',
    model       TEXT NOT NULL DEFAULT 'deepseek-chat',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'idle'
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    created_at      REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON chat_messages(conversation_id, created_at);
"""


@dataclass
class ConversationInfo:
    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    status: str  # idle | running | completed | error


@dataclass
class ActiveConversation:
    """正在运行的对话实例"""
    info: ConversationInfo
    agent: AgentLoop
    task: asyncio.Task | None = None
    event_buffer: list[dict] = field(default_factory=list)
    ws_connections: list = field(default_factory=list)  # 活跃的 WebSocket
    start_time: float | None = None
    total_tokens: int = 0


class ConversationManager:
    """
    管理多个对话实例。
    - 对话元数据和聊天历史存在 meta.db
    - 每对话独立记忆 DB: data/{conv_id}/memory.db
    - 事件日志: data/{conv_id}/events.jsonl
    - Agent 任务与 WebSocket 解耦
    """

    def __init__(
        self,
        data_dir: str,
        keys_path: str | None = None,
        workspace_root: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.keys_path = keys_path
        self.workspace_root = workspace_root

        # 元数据库
        self._meta_db_path = str(self.data_dir / "meta.db")
        self._ensure_meta_schema()

        # 活跃对话
        self._active: dict[str, ActiveConversation] = {}

    # ── 元数据库操作 ───────────────────────────────────

    def _meta_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._meta_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _meta_tx(self):
        conn = self._meta_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_meta_schema(self):
        with self._meta_tx() as conn:
            conn.executescript(META_SCHEMA)
            # 增量迁移：多模态支持 - 添加 images_json 列
            try:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN images_json TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在

    # ── 对话 CRUD ──────────────────────────────────────

    def create_conversation(self, model: str = "deepseek-chat", title: str | None = None, config: dict | None = None) -> ConversationInfo:
        conv_id = uuid.uuid4().hex[:12]
        now = time.time()
        title = title or "新对话"

        with self._meta_tx() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, model, created_at, updated_at, status) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, title, model, now, now, "idle"),
            )

        # 创建对话数据目录
        conv_dir = self.data_dir / conv_id
        conv_dir.mkdir(exist_ok=True)

        # 保存 config 到文件用于后续 agent 创建
        if config:
            import json as _json
            (conv_dir / "agent_config.json").write_text(_json.dumps(config), encoding="utf-8")

        return ConversationInfo(id=conv_id, title=title, model=model, created_at=now, updated_at=now, status="idle")

    def list_conversations(self) -> list[ConversationInfo]:
        with self._meta_tx() as conn:
            rows = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
        result = []
        for r in rows:
            info = ConversationInfo(**dict(r))
            # 如果有活跃任务在跑，用实时状态
            if r["id"] in self._active and self._active[r["id"]].task and not self._active[r["id"]].task.done():
                info.status = "running"
            result.append(info)
        return result

    def get_conversation(self, conv_id: str) -> ConversationInfo | None:
        with self._meta_tx() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not row:
            return None
        info = ConversationInfo(**dict(row))
        if conv_id in self._active and self._active[conv_id].task and not self._active[conv_id].task.done():
            info.status = "running"
        return info

    def update_conversation(self, conv_id: str, **fields):
        allowed = {"title", "model", "status"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [conv_id]
        with self._meta_tx() as conn:
            conn.execute(f"UPDATE conversations SET {set_clause} WHERE id = ?", values)

    def delete_conversation(self, conv_id: str) -> bool:
        # 停止活跃任务
        if conv_id in self._active:
            ac = self._active.pop(conv_id)
            if ac.task and not ac.task.done():
                ac.task.cancel()

        with self._meta_tx() as conn:
            affected = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)).rowcount

        # 删除数据目录
        conv_dir = self.data_dir / conv_id
        if conv_dir.exists():
            import shutil
            shutil.rmtree(conv_dir, ignore_errors=True)

        return affected > 0

    # ── 聊天历史持久化 ──────────────────────────────────

    def save_message(self, conv_id: str, role: str, content: str, tool_calls: list | None = None, images: list[str] | None = None):
        """保存聊天消息。images 为 base64 data URL 列表，会保存到磁盘并记录路径。"""
        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None

        # 保存图片到磁盘，记录相对路径
        images_json = None
        if images:
            from . import workspace_tools
            images_dir = str(self.data_dir / conv_id / "images")
            saved_paths = []
            for img_data_url in images:
                try:
                    fpath = workspace_tools.save_uploaded_image(img_data_url, images_dir)
                    saved_paths.append(fpath)
                except Exception as e:
                    logger.warning(f"Failed to save image for {conv_id}: {e}")
            if saved_paths:
                images_json = json.dumps(saved_paths, ensure_ascii=False)

        with self._meta_tx() as conn:
            conn.execute(
                "INSERT INTO chat_messages (conversation_id, role, content, tool_calls_json, images_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, role, content or "", tool_calls_json, images_json, time.time()),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conv_id))

    def get_chat_history(self, conv_id: str, limit: int = 200) -> list[dict]:
        with self._meta_tx() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls_json, images_json, created_at FROM chat_messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
                (conv_id, limit),
            ).fetchall()
        result = []
        for r in rows:
            msg = {"role": r["role"], "content": r["content"], "timestamp": r["created_at"]}
            if r["tool_calls_json"]:
                msg["tool_calls"] = json.loads(r["tool_calls_json"])
            if r["images_json"]:
                msg["images"] = json.loads(r["images_json"])
            result.append(msg)
        return result

    # ── 事件日志 ────────────────────────────────────────

    def _log_event(self, conv_id: str, event: dict):
        log_path = self.data_dir / conv_id / "events.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event for {conv_id}: {e}")

    def get_event_log(self, conv_id: str, since_line: int = 0) -> list[dict]:
        """读取事件日志，可指定从第几行开始"""
        log_path = self.data_dir / conv_id / "events.jsonl"
        if not log_path.exists():
            return []
        events = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= since_line and line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read event log for {conv_id}: {e}")
        return events

    # ── Agent 生命周期 ──────────────────────────────────

    def _get_or_create_agent(self, conv_id: str, model: str) -> ActiveConversation:
        if conv_id in self._active:
            ac = self._active[conv_id]
            # 模型切换
            if model != ac.agent.client.model_name:
                from portable_llm import UnifiedLLMClient
                from .async_agent_loop import get_model_profile
                ac.agent.client = UnifiedLLMClient(model_name=model, keys_path=self.keys_path)
                ac.agent.model_name = model
                ac.agent.model_profile = get_model_profile(model)
                ac.agent.token_warn_threshold = int(ac.agent.model_profile["context_length"] * ac.agent.model_profile["warn_ratio"])
                ac.agent.token_reboot_threshold = int(ac.agent.model_profile["context_length"] * ac.agent.model_profile["reboot_ratio"])
                # 重置 token 计数器和警告（不同模型的阈值不同）
                ac.agent.total_input_tokens = 0
                ac.agent.total_output_tokens = 0
                ac.agent.current_context_tokens = 0
                ac.agent._token_warning_injected = False
                ac.info.model = model
                self.update_conversation(conv_id, model=model)
            return ac

        # 创建新 Agent 实例
        conv_dir = self.data_dir / conv_id
        conv_dir.mkdir(exist_ok=True)
        memory_db_path = str(conv_dir / "memory.db")

        # 读取可能的 config 覆盖 (如 warn_ratio, reboot_ratio)
        agent_config = {}
        config_path = conv_dir / "agent_config.json"
        if config_path.exists():
            try:
                agent_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        agent = AgentLoop(
            model_name=model,
            config=agent_config,
            keys_path=self.keys_path,
            db_path=memory_db_path,
            workspace_root=self.workspace_root,
        )

        info = self.get_conversation(conv_id)
        if not info:
            info = ConversationInfo(id=conv_id, title="新对话", model=model, created_at=time.time(), updated_at=time.time(), status="idle")

        ac = ActiveConversation(info=info, agent=agent)
        self._active[conv_id] = ac

        # 注册事件回调
        def event_handler(event: AgentEvent):
            evt_dict = {"type": event.type, "data": event.data, "timestamp": event.timestamp}
            # 日志落盘
            self._log_event(conv_id, evt_dict)
            # 缓冲事件
            ac.event_buffer.append(evt_dict)
            # 实时推送给连接的 WebSocket
            self._broadcast_to_ws(ac, evt_dict)
            # 注意：消息持久化由 send_message() 统一处理，不在此重复保存

        agent.on_event(event_handler)
        return ac

    def _broadcast_to_ws(self, ac: ActiveConversation, evt_dict: dict):
        """向所有连接的 WebSocket 广播事件 (线程安全)"""
        dead = []
        for ws_queue in ac.ws_connections:
            try:
                # _emit 可能从 to_thread 工作线程调用，asyncio.Queue 不是线程安全的
                # 使用 call_soon_threadsafe 安全地放入队列
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(ws_queue.put_nowait, evt_dict)
            except RuntimeError:
                # 已在事件循环线程中
                try:
                    ws_queue.put_nowait(evt_dict)
                except Exception:
                    dead.append(ws_queue)
            except Exception:
                dead.append(ws_queue)
        for d in dead:
            if d in ac.ws_connections:
                ac.ws_connections.remove(d)

    async def send_message(self, conv_id: str, message: str, model: str = "deepseek-chat", images: list[str] | None = None) -> str:
        """
        发送消息到对话。Agent 任务在服务端运行。
        返回 task 状态。如果已有任务在跑，拒绝新消息。
        images: 可选的 base64 data URL 列表（多模态图片输入）
        """
        ac = self._get_or_create_agent(conv_id, model)

        if ac.task and not ac.task.done():
            return "agent_busy"

        # 持久化用户消息（含图片）
        self.save_message(conv_id, "user", message, images=images)

        # 首条消息自动设置标题
        if ac.info.title == "新对话":
            auto_title = message[:30].replace("\n", " ")
            if len(message) > 30:
                auto_title += "..."
            self.update_conversation(conv_id, title=auto_title)
            ac.info.title = auto_title

        ac.event_buffer.clear()
        ac.start_time = time.time()

        self.update_conversation(conv_id, status="running")

        async def _run():
            try:
                result = await ac.agent.run(message, images=images)
                ac.total_tokens = ac.agent.total_input_tokens + ac.agent.total_output_tokens
                elapsed = time.time() - ac.start_time if ac.start_time else 0
                # 持久化 assistant 回复
                if result:
                    self.save_message(conv_id, "assistant", result)
                # 发送完成事件
                final_evt = {
                    "type": "final_response",
                    "data": {"text": result, "tokens": ac.total_tokens, "elapsed": round(elapsed, 1)},
                    "timestamp": time.time(),
                }
                ac.event_buffer.append(final_evt)
                self._log_event(conv_id, final_evt)
                self._broadcast_to_ws(ac, final_evt)
                self.update_conversation(conv_id, status="idle")
                return result
            except Exception as e:
                logger.error(f"Agent error in {conv_id}: {e}", exc_info=True)
                error_evt = {"type": "error", "data": {"error": str(e)}, "timestamp": time.time()}
                ac.event_buffer.append(error_evt)
                self._log_event(conv_id, error_evt)
                self._broadcast_to_ws(ac, error_evt)
                self.update_conversation(conv_id, status="error")
                return f"Error: {e}"

        ac.task = asyncio.create_task(_run())
        return "started"

    async def send_message_stream(self, conv_id: str, message: str, model: str = "deepseek-chat", images: list[str] | None = None):
        """
        流式发送消息。async generator，yield SSE 事件字典。
        与 send_message 不同：不创建后台 task，调用者直接消费流。
        """
        ac = self._get_or_create_agent(conv_id, model)

        if ac.task and not ac.task.done():
            yield {"event": "error", "data": {"error": "Agent 正在执行中，请等待完成"}}
            return

        self.save_message(conv_id, "user", message, images=images)

        if ac.info.title == "新对话":
            auto_title = message[:30].replace("\n", " ")
            if len(message) > 30:
                auto_title += "..."
            self.update_conversation(conv_id, title=auto_title)
            ac.info.title = auto_title

        ac.event_buffer.clear()
        ac.start_time = time.time()
        self.update_conversation(conv_id, status="running")

        final_text = ""
        try:
            async for event in ac.agent.run_stream(message, images=images):
                evt_type = event.get("event", "")
                # Capture final text
                if evt_type == "done":
                    final_text = event.get("data", {}).get("text", "")
                # Log and broadcast
                evt_dict = {"type": evt_type, "data": event.get("data"), "timestamp": time.time()}
                self._log_event(conv_id, evt_dict)
                self._broadcast_to_ws(ac, evt_dict)
                yield event
        except Exception as e:
            logger.error(f"Stream error in {conv_id}: {e}", exc_info=True)
            yield {"event": "error", "data": {"error": str(e)}}
        finally:
            if final_text:
                self.save_message(conv_id, "assistant", final_text)
            self.update_conversation(conv_id, status="idle")

    def connect_ws(self, conv_id: str) -> asyncio.Queue | None:
        """WebSocket 连接到对话，返回事件队列"""
        if conv_id not in self._active:
            # 尝试懒加载
            info = self.get_conversation(conv_id)
            if not info:
                return None
            self._get_or_create_agent(conv_id, info.model)

        ac = self._active[conv_id]
        queue: asyncio.Queue = asyncio.Queue()
        ac.ws_connections.append(queue)
        return queue

    def disconnect_ws(self, conv_id: str, queue: asyncio.Queue):
        """WebSocket 断开"""
        if conv_id in self._active:
            ac = self._active[conv_id]
            if queue in ac.ws_connections:
                ac.ws_connections.remove(queue)

    def get_status(self, conv_id: str) -> dict:
        """获取对话运行状态"""
        if conv_id in self._active:
            ac = self._active[conv_id]
            is_running = ac.task is not None and not ac.task.done()
            elapsed = (time.time() - ac.start_time) if ac.start_time and is_running else None
            return {
                "is_running": is_running,
                "context_tokens": ac.agent.current_context_tokens,
                "total_input": ac.agent.total_input_tokens,
                "total_output": ac.agent.total_output_tokens,
                "total_cost": ac.agent._calculate_cost(),
                "elapsed": round(elapsed, 1) if elapsed else None,
                "model": ac.agent.client.model_name,
                "message_count": len(ac.agent.messages),
                "reboot_count": ac.agent.reboot_count,
                "buffered_events": len(ac.event_buffer),
            }
        return {"is_running": False, "total_tokens": 0}

    def get_memory_tree(self, conv_id: str, node_name: str | None = None) -> dict:
        info = self.get_conversation(conv_id)
        if not info:
            return {"error": "对话不存在"}
        ac = self._get_or_create_agent(conv_id, info.model)
        return ac.agent.get_memory_tree(node_name)

    def get_full_tree(self, conv_id: str) -> dict:
        info = self.get_conversation(conv_id)
        if not info:
            return {"error": "对话不存在"}
        ac = self._get_or_create_agent(conv_id, info.model)
        return self._build_tree(ac.agent, "root")

    def _build_tree(self, agent: AgentLoop, name: str, depth: int = 0) -> dict:
        if depth > 8:
            return {"name": name, "children": []}
        try:
            ctx = agent.engine.retrieve_context(name)
            return {
                "name": ctx["node"]["name"],
                "content_preview": ctx["node"]["content"][:100],
                "weight": ctx["node"]["current_weight"],
                "k_value": ctx["node"]["k_value"],
                "children": [self._build_tree(agent, c["name"], depth + 1) for c in ctx["children"]],
                "edges": ctx["outgoing_edges"],
            }
        except Exception:
            return {"name": name, "children": []}

    async def shutdown(self):
        """关闭所有活跃对话"""
        for conv_id, ac in self._active.items():
            if ac.task and not ac.task.done():
                ac.task.cancel()
            await ac.agent.shutdown()
        self._active.clear()

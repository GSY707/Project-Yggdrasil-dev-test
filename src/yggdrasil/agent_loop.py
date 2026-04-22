"""
Agent 主循环引擎
驱动 LLM 调用工具 → 解析 tool_calls → 执行本地工具 → 反馈结果 → 循环
"""

import json
import sys
import os
import logging
import time
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field

# portable_llm 在项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from portable_llm import UnifiedLLMClient, LLMResponse

from .prompts import BOOTLOADER_PROMPT, COMPRESS_CONTEXT_PROMPT
from .memory_engine import MemoryEngine
from .database import Database
from . import workspace_tools

logger = logging.getLogger(__name__)


# ── 工具定义 (OpenAI function calling format) ─────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": "猴子爬树：检索节点的内容、关联边、子节点列表。输入为空时返回根节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {
                        "type": "string",
                        "description": "节点名称。为空时返回根节点。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "只返回权重最高的 k 个子节点",
                    },
                    "top_p": {
                        "type": "number",
                        "description": "只返回累积权重占比达 p 的子节点 (0-1)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_memory_node",
            "description": "创建记忆节点。k_value 控制遗忘速度：0=永久，越大衰减越快。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "节点名称(唯一)"},
                    "content": {"type": "string", "description": "节点内容"},
                    "parent_name": {"type": "string", "description": "父节点名称，默认挂在 root 下"},
                    "k_value": {"type": "number", "description": "衰减系数：0=永久, 0.01=缓慢, 0.1=快速, ≥1.0=阅后即焚"},
                    "initial_score": {"type": "number", "description": "初始重要性 (默认1.0)"},
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "label": {"type": "string"},
                            },
                        },
                        "description": "关联出边 [{target, label}]",
                    },
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory_node",
            "description": "更新记忆节点 (Git 式：自动保存历史快照)。只传需要改的字段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string", "description": "要修改的节点名称"},
                    "new_name": {"type": "string", "description": "新名称"},
                    "new_parent_name": {"type": "string", "description": "新父节点名称(移动节点)"},
                    "new_content": {"type": "string", "description": "新内容"},
                    "new_k_value": {"type": "number", "description": "新衰减系数"},
                    "append_content": {"type": "boolean", "description": "true=追加内容而非覆盖"},
                },
                "required": ["node_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hard_delete",
            "description": "物理删除节点及子树。不可逆。用于切除幻觉或废弃节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string", "description": "要删除的节点名称"},
                },
                "required": ["node_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_history",
            "description": "查询节点的历史版本快照（时间漫游）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {"type": "string", "description": "节点名称"},
                },
                "required": ["node_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_nodes",
            "description": "在两个节点之间建立关联边。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string", "description": "源节点名称"},
                    "target_name": {"type": "string", "description": "目标节点名称"},
                    "label": {"type": "string", "description": "关系标签，如 RELATED_TO, IS_SAME_AS, DEPENDS_ON"},
                },
                "required": ["source_name", "target_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径(工作区相对路径或绝对路径)"},
                    "start_line": {"type": "integer", "description": "起始行号(1-based)"},
                    "end_line": {"type": "integer", "description": "结束行号(1-based, 含)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "在工作区文件中搜索文本或正则表达式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索文本或正则表达式"},
                    "path": {"type": "string", "description": "限定搜索的子目录"},
                    "max_results": {"type": "integer", "description": "最大返回结果数(默认30)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出工作区目录内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径(留空列出工作区根目录)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件到工作区。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径(工作区相对路径)"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reboot_context",
            "description": "主动重启上下文：清空对话历史，从根节点重新开始。调用前请确保已将工作进度保存到 execution_state。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_context",
            "description": "压缩当前上下文而非完全重启。声明未来目标，系统保留相关信息、裁剪无关内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "future_intent": {"type": "string", "description": "接下来你打算做什么（越具体越好）"},
                },
                "required": ["future_intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_memory",
            "description": "触发碎片记忆整理：遍历记忆树叶子节点，自动发现关联并建边。适合空闲时执行。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class AgentEvent:
    """Agent 循环中的事件，用于发送给前端"""
    type: str  # "thinking" | "tool_call" | "tool_result" | "message" | "reboot" | "error" | "status"
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class AgentLoop:
    """
    世界树 Agent 主循环引擎。
    驱动 LLM → 工具调用 → 结果反馈 → 继续推理 的完整闭环。
    """

    MAX_TOOL_ROUNDS = 30  # 单次用户消息最多工具调用轮次
    TOKEN_WARNING_THRESHOLD = 20000  # Token 预警线

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        config: dict | None = None,
        keys_path: str | None = None,
        db_path: str | None = None,
        workspace_root: str | None = None,
    ):
        # LLM 客户端
        self.client = UnifiedLLMClient(
            model_name=model_name,
            config=config or {},
            keys_path=keys_path,
        )

        # 记忆引擎
        db = Database(db_path) if db_path else Database()
        self.engine = MemoryEngine(db)
        self.engine.initialize_root()

        # 工作区
        if workspace_root:
            workspace_tools.WORKSPACE_ROOT = workspace_root

        # 对话状态
        self.messages: list[dict] = []
        self.total_tokens_used = 0
        self.is_running = False

        # 事件回调 (前端订阅)
        self._event_callbacks: list[Callable[[AgentEvent], None]] = []

        # 工具映射
        self._tool_handlers = {
            "retrieve_context": self._tool_retrieve_context,
            "create_memory_node": self._tool_create_memory_node,
            "update_memory_node": self._tool_update_memory_node,
            "hard_delete": self._tool_hard_delete,
            "node_history": self._tool_node_history,
            "link_nodes": self._tool_link_nodes,
            "read_file": self._tool_read_file,
            "search_files": self._tool_search_files,
            "list_directory": self._tool_list_directory,
            "write_file": self._tool_write_file,
            "reboot_context": self._tool_reboot_context,
            "compress_context": self._tool_compress_context,
            "consolidate_memory": self._tool_consolidate_memory,
        }

    def on_event(self, callback: Callable[[AgentEvent], None]):
        """注册事件回调"""
        self._event_callbacks.append(callback)

    def _emit(self, event_type: str, data: Any = None):
        evt = AgentEvent(type=event_type, data=data)
        for cb in self._event_callbacks:
            try:
                cb(evt)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
        return evt

    # ── 主循环 ──────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """
        处理一条用户消息，驱动完整的 Agent 循环。
        返回 Agent 的最终文本响应。
        """
        self.is_running = True
        self._emit("status", {"state": "running"})

        # 如果是首次对话，注入 System Prompt
        if not self.messages:
            self.messages.append({"role": "system", "content": BOOTLOADER_PROMPT})

        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})
        self._emit("message", {"role": "user", "content": user_message})

        final_text = ""
        try:
            final_text = self._agent_loop()
        except Exception as e:
            error_msg = f"Agent 循环异常: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit("error", {"error": error_msg})
            final_text = f"抱歉，我遇到了一个错误: {e}"
        finally:
            self.is_running = False
            self._emit("status", {"state": "idle"})

        return final_text

    def _agent_loop(self) -> str:
        """核心循环：调用 LLM → 处理 tool_calls → 反馈结果 → 重复"""
        for round_idx in range(self.MAX_TOOL_ROUNDS):
            # 调用 LLM
            response = self.client.call_with_tools(
                messages=self.messages,
                temperature=0.3,
                tools=TOOL_DEFINITIONS,
            )

            # 累计 token
            self.total_tokens_used += response.usage.total_tokens
            self._emit("status", {"tokens_used": self.total_tokens_used})

            # Token 预警检查
            if self.total_tokens_used > self.TOKEN_WARNING_THRESHOLD:
                self._inject_token_warning()

            # 没有 tool_calls → LLM 产出了最终回复
            if not response.tool_calls:
                text = response.text or ""
                self.messages.append({"role": "assistant", "content": text})
                self._emit("message", {"role": "assistant", "content": text})
                return text

            # 有 tool_calls → 逐个执行
            # 先把 assistant 的 tool_calls 消息加入历史
            # OpenAI 格式要求: content 可为 null, tool_calls 必须有 type 字段
            normalized_tool_calls = []
            for tc in response.tool_calls:
                ntc = {
                    "id": tc.get("id", f"call_{tc['function']['name']}"),
                    "type": "function",
                    "function": tc["function"],
                }
                normalized_tool_calls.append(ntc)

            assistant_msg = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": normalized_tool_calls,
            }
            self.messages.append(assistant_msg)

            if response.text:
                self._emit("thinking", {"content": response.text})

            for tc in normalized_tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}

                self._emit("tool_call", {"name": func_name, "arguments": args, "id": tc.get("id", "")})

                # 执行工具
                result = self._execute_tool(func_name, args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                # 截断过长的工具结果
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "\n...[结果已截断，共 " + str(len(result_str)) + " 字符]"

                self._emit("tool_result", {"name": func_name, "result": result_str[:2000], "id": tc.get("id", "")})

                # 添加 tool 结果到消息历史
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{func_name}"),
                    "content": result_str,
                })

            # 检查是否触发了重启
            if self._check_reboot_flag():
                return "[上下文已重启，正在从根节点恢复...]"

        return "达到最大工具调用轮次限制，请缩小任务范围。"

    def _inject_token_warning(self):
        """注入 Token 预警消息"""
        warning = (
            "⚠️ 警告：你的上下文 Token 已超过预警线。认知能力面临下降风险。\n"
            "请立即：\n"
            "1. 将当前工作进度存入 execution_state 节点\n"
            "2. 考虑调用 reboot_context() 重启，或 compress_context() 压缩上下文"
        )
        self.messages.append({"role": "system", "content": warning})
        self._emit("status", {"warning": "token_threshold_exceeded"})

    # ── 工具执行分发 ────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> dict:
        handler = self._tool_handlers.get(name)
        if not handler:
            return {"error": f"未知工具 '{name}'"}
        try:
            return handler(**args)
        except Exception as e:
            logger.error(f"工具 {name} 执行错误: {e}", exc_info=True)
            return {"error": str(e)}

    # ── 记忆工具实现 ────────────────────────────────────

    def _tool_retrieve_context(self, node_name: str | None = None, top_k: int | None = None, top_p: float | None = None) -> dict:
        return self.engine.retrieve_context(node_name=node_name, top_k=top_k, top_p=top_p)

    def _tool_create_memory_node(self, name: str, content: str, parent_name: str | None = None, k_value: float = 0.0, initial_score: float = 1.0, edges: list | None = None) -> dict:
        return self.engine.create_memory_node(
            name=name, content=content,
            parent_name=parent_name or "root",
            k_value=k_value, initial_score=initial_score,
            edges=edges,
        )

    def _tool_update_memory_node(self, node_name: str, **kwargs) -> dict:
        return self.engine.update(node_name, **kwargs)

    def _tool_hard_delete(self, node_name: str) -> dict:
        return self.engine.hard_delete(node_name)

    def _tool_node_history(self, node_name: str) -> list:
        return self.engine.node_history(node_name)

    def _tool_link_nodes(self, source_name: str, target_name: str, label: str = "RELATED_TO") -> dict:
        src = self.engine.db.get_node_by_name(source_name)
        tgt = self.engine.db.get_node_by_name(target_name)
        if not src:
            return {"error": f"源节点 '{source_name}' 不存在"}
        if not tgt:
            return {"error": f"目标节点 '{target_name}' 不存在"}
        self.engine.db.create_edge(src["id"], tgt["id"], label)
        return {"linked": f"{source_name} --{label}--> {target_name}"}

    # ── 工作区工具实现 ──────────────────────────────────

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        return workspace_tools.read_file(path, start_line, end_line)

    def _tool_search_files(self, query: str, path: str | None = None, max_results: int = 30) -> dict:
        return workspace_tools.search_files(query, path, max_results)

    def _tool_list_directory(self, path: str | None = None) -> dict:
        return workspace_tools.list_directory(path)

    def _tool_write_file(self, path: str, content: str) -> dict:
        return workspace_tools.write_file(path, content)

    # ── 上下文管理工具 ──────────────────────────────────

    _reboot_flag = False

    def _tool_reboot_context(self) -> dict:
        """清空对话历史，从根节点重启"""
        self._reboot_flag = True
        self.messages = []
        self.total_tokens_used = 0
        self._emit("reboot", {"reason": "agent_requested"})
        return {"status": "context_rebooted", "message": "上下文已清空。下次调用将从根节点苏醒。"}

    def _check_reboot_flag(self) -> bool:
        if self._reboot_flag:
            self._reboot_flag = False
            return True
        return False

    def _tool_compress_context(self, future_intent: str) -> dict:
        """调用 Sub-Agent 压缩当前上下文"""
        self._emit("status", {"state": "compressing"})

        # 构造压缩 prompt
        conversation_text = "\n".join(
            f"[{m['role']}] {m.get('content', '')[:500]}"
            for m in self.messages[-30:]  # 最近30条
        )

        exec_state = ""
        try:
            ctx = self.engine.retrieve_context("execution_state")
            exec_state = ctx["node"]["content"]
        except Exception:
            pass

        compress_input = (
            f"## future_intent\n{future_intent}\n\n"
            f"## current_execution_state\n{exec_state}\n\n"
            f"## conversation_history (最近)\n{conversation_text}"
        )

        # 用同一个 LLM 做压缩 (Sub-Agent 角色)
        compress_response = self.client.generate(
            system_prompt=COMPRESS_CONTEXT_PROMPT,
            user_prompt=compress_input,
            temperature=0.1,
        )

        compressed = compress_response.text

        # 将压缩结果写入 execution_state
        try:
            self.engine.update("execution_state", new_content=compressed)
        except Exception as e:
            logger.error(f"Failed to update execution_state: {e}")

        # 用压缩后的上下文替换臃肿的历史
        self.messages = [
            {"role": "system", "content": BOOTLOADER_PROMPT},
            {"role": "system", "content": f"[上下文压缩摘要]\n{compressed}"},
        ]
        self.total_tokens_used = 0

        self._emit("status", {"state": "compressed", "intent": future_intent})
        return {
            "status": "context_compressed",
            "future_intent": future_intent,
            "summary_length": len(compressed),
        }

    # ── 碎节点整理 ──────────────────────────────────────

    def _tool_consolidate_memory(self) -> dict:
        """遍历记忆树叶子节点，用 LLM 发现潜在关联"""
        self._emit("status", {"state": "consolidating"})

        # 收集所有叶子节点 (无子节点的节点)
        all_leaves = self._collect_leaves("root", depth=0, max_depth=5)
        if len(all_leaves) < 2:
            return {"message": "叶子节点不足，无需整理", "leaves": len(all_leaves)}

        # 构造叶子节点摘要列表
        leaf_summaries = "\n".join(
            f"- [{l['name']}] (parent={l.get('parent', '?')}): {l['content'][:100]}"
            for l in all_leaves[:50]  # 限制数量
        )

        prompt = (
            "以下是记忆树中的叶子节点列表。请分析它们之间的潜在关联关系。\n"
            "只输出你确信的关联，格式为 JSON 数组：\n"
            '[{"source": "节点A名", "target": "节点B名", "label": "关系类型", "reason": "简短原因"}]\n'
            "关系类型可选: IS_SAME_AS, DEPENDS_ON, RELATED_TO, CONTRADICTS\n"
            "如果没有发现明确关联，返回空数组 []\n\n"
            f"叶子节点:\n{leaf_summaries}"
        )

        response = self.client.generate(
            system_prompt="你是一个知识图谱关联分析专家。只输出 JSON，不要其他内容。",
            user_prompt=prompt,
            temperature=0.1,
        )

        # 解析并建立关联
        links_created = []
        try:
            # 从回复中提取 JSON
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            suggestions = json.loads(text)
            for s in suggestions:
                src = self.engine.db.get_node_by_name(s["source"])
                tgt = self.engine.db.get_node_by_name(s["target"])
                if src and tgt:
                    self.engine.db.create_edge(src["id"], tgt["id"], s.get("label", "RELATED_TO"))
                    links_created.append(f"{s['source']} --{s.get('label', 'RELATED_TO')}--> {s['target']}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse consolidation response: {e}")

        self._emit("status", {"state": "idle"})
        return {
            "leaves_analyzed": len(all_leaves),
            "links_created": links_created,
            "total_new_links": len(links_created),
        }

    def _collect_leaves(self, node_name: str, depth: int, max_depth: int) -> list[dict]:
        """递归收集叶子节点"""
        if depth > max_depth:
            return []
        try:
            ctx = self.engine.retrieve_context(node_name)
        except Exception:
            return []

        if not ctx["children"]:
            return [{"name": node_name, "content": ctx["node"]["content"], "parent": str(ctx["node"].get("parent_id", ""))}]

        leaves = []
        for child in ctx["children"]:
            leaves.extend(self._collect_leaves(child["name"], depth + 1, max_depth))
        return leaves

    # ── 对话状态接口 ────────────────────────────────────

    def get_state(self) -> dict:
        """获取当前 Agent 状态 (供前端展示)"""
        return {
            "is_running": self.is_running,
            "total_tokens": self.total_tokens_used,
            "message_count": len(self.messages),
            "model": self.client.model_name,
        }

    def get_memory_tree(self, node_name: str | None = None) -> dict:
        """获取记忆树结构 (供前端展示)"""
        try:
            return self.engine.retrieve_context(node_name)
        except Exception as e:
            return {"error": str(e)}

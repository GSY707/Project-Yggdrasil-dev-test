"""
Agent 主循环引擎 (异步版)
驱动 LLM 调用工具 → 解析 tool_calls → 执行本地工具 → 反馈结果 → 循环
LLM 调用通过 asyncio.to_thread 桥接同步 portable_llm
"""

import asyncio
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

from .prompts import get_bootloader_prompt, COMPRESS_CONTEXT_PROMPT
from .memory_engine import MemoryEngine
from .database import Database
from . import workspace_tools
from . import bootstrap as bootstrap_engine
from .web_search import web_search, fetch_webpage
from .task_queue import TaskQueue, TaskStatus

logger = logging.getLogger(__name__)


# ── 模型参数配置表 ────────────────────────────────────

MODEL_PROFILES = {
    # model_name_prefix → {context_length, max_output, warn_ratio, reboot_ratio, cost_per_m_input, cost_per_m_output, cost_per_m_cached}
    "deepseek-chat": {
        "context_length": 128_000,
        "max_output": 8_000,
        "warn_ratio": 0.25,      # 1/4 上下文时预警
        "reboot_ratio": 0.50,    # 1/2 上下文时强制重启
        "cost_input": 2.0,       # 元/M tokens
        "cost_output": 3.0,
        "cost_cached": 0.2,
    },
    "deepseek-reasoner": {
        "context_length": 128_000,
        "max_output": 64_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 2.0,
        "cost_output": 3.0,
        "cost_cached": 0.2,
    },
    "longcat-flash-lite": {
        "context_length": 320_000,
        "max_output": 320_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,     # 50M 免费 tokens
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "longcat-flash-omni": {
        "context_length": 8_000,
        "max_output": 8_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,     # 与其他 flash 共享 5M 免费
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "longcat-flash-chat": {
        "context_length": 256_000,
        "max_output": 256_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "longcat-flash-thinking": {
        "context_length": 256_000,
        "max_output": 256_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "google/gemma-4-31b-it:free": {
        "context_length": 262_144,
        "max_output": 32_800,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "context_length": 262_144,
        "max_output": 262_144,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "z-ai/glm-4.5-air:free": {
        "context_length": 131_072,
        "max_output": 96_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "minimax/minimax-m2.5:free": {
        "context_length": 196_608,
        "max_output": 196_608,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
    "arcee-ai/trinity-large-preview:free": {
        "context_length": 131_000,
        "max_output": 131_000,
        "warn_ratio": 0.25,
        "reboot_ratio": 0.50,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cached": 0.0,
    },
}

# 默认配置（保守值）
DEFAULT_PROFILE = {
    "context_length": 32_000,
    "max_output": 4_000,
    "warn_ratio": 0.25,
    "reboot_ratio": 0.50,
    "cost_input": 0.0,
    "cost_output": 0.0,
    "cost_cached": 0.0,
}


def get_model_profile(model_name: str) -> dict:
    """按模型名称匹配配置，支持前缀模糊匹配"""
    if model_name in MODEL_PROFILES:
        return MODEL_PROFILES[model_name]
    # 前缀匹配
    for prefix, profile in MODEL_PROFILES.items():
        if model_name.startswith(prefix):
            return profile
    return DEFAULT_PROFILE


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
            "description": "创建记忆节点。必须指定 parent_name（分类目录节点），禁止直接挂 root。先创建分类目录再创建内容节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "节点名称(唯一)"},
                    "content": {"type": "string", "description": "节点内容"},
                    "parent_name": {"type": "string", "description": "父节点名称（必填！内容节点必须挂在分类目录下，不要挂 root）"},
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
            "name": "append_file",
            "description": "追加内容到文件末尾。文件不存在时自动创建。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径(工作区相对路径)"},
                    "content": {"type": "string", "description": "要追加的内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除工作区中的文件或空目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件或空目录路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "在工作区目录下执行 shell 命令。支持 python/node/git/pytest 等白名单命令。用于运行脚本、测试、查看输出。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令 (如 'python test.py', 'pytest -v')"},
                    "timeout": {"type": "integer", "description": "超时秒数(默认30, 最长120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "查看工作区中的图片文件，返回 base64 data URL（可直接嵌入多模态对话）。支持 png/jpg/gif/webp/bmp。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "图片文件路径(工作区相对路径)"},
                    "max_size_mb": {"type": "number", "description": "最大文件大小限制(MB, 默认5)"},
                },
                "required": ["path"],
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
    # ── 记忆搜索工具 ──
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "模糊搜索记忆树节点（按名称和内容匹配）。当你不确定目标节点的位置时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "integer", "description": "最大返回数(默认20)"},
                },
                "required": ["query"],
            },
        },
    },
    # ── 网络搜索工具 ──
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网。使用 DuckDuckGo 搜索引擎。返回标题、URL、摘要列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "最大返回结果数(默认8)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "抓取网页内容并提取正文文本。用于深入阅读搜索结果中的页面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "网页 URL (必须是 http 或 https)"},
                    "max_chars": {"type": "integer", "description": "最大返回字符数(默认8000)"},
                },
                "required": ["url"],
            },
        },
    },
    # ── 后台任务工具 ──
    {
        "type": "function",
        "function": {
            "name": "run_background_task",
            "description": "将耗时操作放到后台执行（如碎片整理、长文本消化）。调用后立即返回 task_id，不阻塞主循环。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["consolidate_memory", "digest_text"],
                        "description": "任务类型：consolidate_memory=碎片整理, digest_text=长文本消化",
                    },
                    "params": {
                        "type": "object",
                        "description": "任务参数。digest_text 需要 {text, node_name}",
                    },
                },
                "required": ["task_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_task",
            "description": "检查后台任务的状态和结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "列出所有后台任务及其状态。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_human",
            "description": "向人类求助。当你连续多次失败、需要授权、或遇到超出能力范围的问题时使用。调用后当前任务暂停，等待人类回复后继续。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "你要问人类的具体问题（描述清楚上下文和你已尝试的方法）"},
                    "context": {"type": "string", "description": "相关背景信息（已尝试的方案、错误日志等）"},
                    "urgency": {"type": "string", "enum": ["low", "medium", "high"], "description": "紧急程度"},
                },
                "required": ["question"],
            },
        },
    },
    # ── 自举进化工具 ──
    {
        "type": "function",
        "function": {
            "name": "inspect_source",
            "description": "自省：查看自身项目源代码结构（模块列表、文件大小、行数）。用于了解自己的代码架构。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "读取自身项目的源代码文件。路径相对于项目根目录（如 'src/yggdrasil/database.py'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于项目根目录的源文件路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_evolution",
            "description": "提交自举进化提案：修改自身源代码。系统会在沙箱中测试、由评审法官评估，通过后自动合并。连续3次被驳回会触发人类求助。",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "本次修改的简要描述"},
                    "target_file": {"type": "string", "description": "要修改的源文件路径（相对于项目根目录，如 'src/yggdrasil/tool_router.py'）"},
                    "modified_content": {"type": "string", "description": "修改后的完整文件内容"},
                    "reason": {"type": "string", "description": "修改原因（什么问题促使你要改这个文件？）"},
                    "test_commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "自定义测试命令列表（可选，默认跑 import 检查 + pytest）",
                    },
                },
                "required": ["description", "target_file", "modified_content", "reason"],
            },
        },
    },
]


@dataclass
class AgentEvent:
    """Agent 循环中的事件，用于发送给前端"""
    type: str  # "thinking" | "tool_call" | "tool_result" | "message" | "reboot" | "error" | "status" | "task_update"
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class AgentLoop:
    """
    世界树 Agent 主循环引擎 (异步版)。
    驱动 LLM → 工具调用 → 结果反馈 → 继续推理 的完整闭环。
    LLM 调用和网络 I/O 通过 asyncio.to_thread 实现非阻塞。
    后台任务通过 TaskQueue 并发执行。
    """

    MAX_TOOL_ROUNDS = 200
    MAX_AUTO_CONTINUES = 50  # reboot/compress 后最多自动续接次数

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        config: dict | None = None,
        keys_path: str | None = None,
        db_path: str | None = None,
        workspace_root: str | None = None,
    ):
        # 模型配置
        self.model_name = model_name
        self.model_profile = get_model_profile(model_name)

        # 支持通过 config 临时覆盖阈值 (方便压力测试)
        config = config or {}
        if "warn_ratio" in config:
            self.model_profile = {**self.model_profile, "warn_ratio": config["warn_ratio"]}
        if "reboot_ratio" in config:
            self.model_profile = {**self.model_profile, "reboot_ratio": config["reboot_ratio"]}

        self.token_warn_threshold = int(self.model_profile["context_length"] * self.model_profile["warn_ratio"])
        self.token_reboot_threshold = int(self.model_profile["context_length"] * self.model_profile["reboot_ratio"])

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

        # 后台任务队列
        self.task_queue = TaskQueue(max_workers=4)
        self.task_queue.on_task_event(self._on_task_event)

        # 对话状态
        self.messages: list[dict] = []
        self.total_input_tokens = 0      # 累计输入 tokens (费用计算)
        self.total_output_tokens = 0     # 累计输出 tokens (费用计算)
        self.current_context_tokens = 0  # 最近一次 prompt_tokens (当前上下文大小)
        self.is_running = False
        self._token_warning_injected = False
        self._current_user_message: str = ""  # 跨重启保持的原始用户消息
        self.reboot_count: int = 0  # 累计重启次数

        # 循环检测状态
        self._recent_tool_calls: list[str] = []  # 最近工具调用名称序列
        self._loop_detect_window = 10  # 检测窗口大小
        self._loop_detect_threshold = 6  # 窗口内同工具出现次数阈值

        # 事件回调 (前端订阅)
        self._event_callbacks: list[Callable[[AgentEvent], None]] = []

        # 工具映射 — 同步工具
        self._sync_tool_handlers = {
            "retrieve_context": self._tool_retrieve_context,
            "create_memory_node": self._tool_create_memory_node,
            "update_memory_node": self._tool_update_memory_node,
            "hard_delete": self._tool_hard_delete,
            "node_history": self._tool_node_history,
            "link_nodes": self._tool_link_nodes,
            "search_memory": self._tool_search_memory,
            "read_file": self._tool_read_file,
            "search_files": self._tool_search_files,
            "list_directory": self._tool_list_directory,
            "write_file": self._tool_write_file,
            "append_file": self._tool_append_file,
            "delete_file": self._tool_delete_file,
            "execute_command": self._tool_execute_command,
            "view_image": self._tool_view_image,
            "reboot_context": self._tool_reboot_context,
            "run_background_task": self._tool_run_background_task,
            "check_task": self._tool_check_task,
            "list_tasks": self._tool_list_tasks,
            "ask_human": self._tool_ask_human,
            "inspect_source": self._tool_inspect_source,
            "read_source": self._tool_read_source,
        }

        # 工具映射 — 需要在线程中执行的 I/O 密集工具 (网络/LLM)
        self._io_tool_handlers = {
            "web_search": self._tool_web_search,
            "fetch_webpage": self._tool_fetch_webpage,
            "compress_context": self._tool_compress_context,
            "consolidate_memory": self._tool_consolidate_memory,
            "propose_evolution": self._tool_propose_evolution,
        }

        # 自举状态
        self._evolution_rejection_count = 0
        self._MAX_EVOLUTION_REJECTIONS = 3

    # ── 工具调用安全标准化 ──────────────────────────────

    def _normalize_tool_call(self, tc: dict) -> dict | None:
        """安全标准化一个 tool_call 字典。返回 None 表示该调用无效应跳过。"""
        try:
            func = tc.get("function") or {}
            name = func.get("name") or ""
            arguments = func.get("arguments") or "{}"
            if not name:
                logger.warning(f"Skipping tool_call with empty name: {tc}")
                return None
            return {
                "id": tc.get("id") or f"call_{name}_{id(tc)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        except Exception as e:
            logger.error(f"Failed to normalize tool_call: {tc}, error: {e}")
            return None

    # 工具参数类型期望表
    _TOOL_TYPE_HINTS: dict[str, dict[str, type]] = {
        "retrieve_context": {"top_k": int, "top_p": float},
        "create_memory_node": {"k_value": float, "initial_score": float},
        "update_memory_node": {"new_k_value": float, "append_content": bool},
        "read_file": {"start_line": int, "end_line": int},
        "search_files": {"max_results": int},
        "web_search": {"max_results": int},
        "fetch_webpage": {"max_chars": int},
        "execute_command": {"timeout": int},
        "view_image": {"max_size_mb": float},
        "search_memory": {"limit": int},
    }

    # 常见参数名别名映射 (LLM 高频犯的参数名错误)
    _TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
        "update_memory_node": {"content": "new_content", "name": "node_name"},
        "retrieve_context": {"name": "node_name"},
        "hard_delete": {"name": "node_name"},
        "node_history": {"name": "node_name"},
        "write_file": {"file_content": "content", "text": "content", "data": "content"},
    }

    def _coerce_tool_args(self, func_name: str, args: dict) -> dict:
        """尝试修正 LLM 常见的参数类型和参数名错误。"""
        # 1. 参数名别名映射
        aliases = self._TOOL_ARG_ALIASES.get(func_name, {})
        coerced = {}
        for k, v in args.items():
            # 如果 k 是别名且目标键名不在 args 中，则替换
            mapped_k = aliases.get(k, k) if k in aliases and aliases[k] not in args else k
            coerced[mapped_k] = v

        # 2. 类型修正
        hints = self._TOOL_TYPE_HINTS.get(func_name, {})
        for k, v in coerced.items():
            expected_type = hints.get(k)
            if expected_type and not isinstance(v, expected_type):
                try:
                    if expected_type == bool and isinstance(v, str):
                        coerced[k] = v.lower() in ("true", "1", "yes")
                    elif expected_type == bool and isinstance(v, (int, float)):
                        coerced[k] = bool(v)
                    else:
                        coerced[k] = expected_type(v)
                except (ValueError, TypeError):
                    pass  # 保持原值，让下游自然报错

        return coerced

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

    def _on_task_event(self, event_type: str, data: dict):
        """TaskQueue 事件回调 → 转发到前端"""
        self._emit("task_update", data)

    # ── 苏醒上下文预注入 ──────────────────────────────

    def _build_awakening_context(self) -> str:
        """读取记忆树核心节点，构建苏醒上下文文本"""
        parts = []

        # 1. 根节点概览
        try:
            root_ctx = self.engine.retrieve_context()
            children_list = "\n".join(
                f"  - {c['name']}: {c['content_preview'][:100]}"
                for c in root_ctx["children"]
            )
            parts.append(f"### 记忆树根节点\n{root_ctx['node']['content']}\n子节点:\n{children_list}")
        except Exception as e:
            parts.append(f"### 记忆树根节点\n[读取失败: {e}]")

        # 2. 身份
        try:
            identity_ctx = self.engine.retrieve_context("identity")
            content = identity_ctx["node"]["content"]
            children_info = ""
            if identity_ctx["children"]:
                children_info = "\n子节点:\n" + "\n".join(
                    f"  - {c['name']}" for c in identity_ctx["children"]
                )
            parts.append(f"### identity\n{content}{children_info}")
        except Exception:
            pass

        # 3. 行为准则
        try:
            rules_ctx = self.engine.retrieve_context("operational_rules")
            parts.append(f"### operational_rules\n{rules_ctx['node']['content']}")
        except Exception:
            pass

        # 4. 执行状态
        try:
            exec_ctx = self.engine.retrieve_context("execution_state")
            content = exec_ctx["node"]["content"]
            children_info = ""
            if exec_ctx["children"]:
                children_info = "\n子节点:\n" + "\n".join(
                    f"  - {c['name']}: {c['content_preview'][:100]}"
                    for c in exec_ctx["children"]
                )
            parts.append(f"### execution_state\n{content}{children_info}")
        except Exception:
            pass

        return "\n\n".join(parts) if parts else ""

    def _build_initial_messages(self) -> list[dict]:
        """构建初始消息列表：系统提示词 + 苏醒上下文"""
        messages = [{"role": "system", "content": get_bootloader_prompt()}]

        awakening = self._build_awakening_context()
        if awakening:
            messages.append({
                "role": "system",
                "content": f"[苏醒上下文 - 系统自动加载]\n\n{awakening}",
            })

        return messages

    # ── 主循环 (异步) ──────────────────────────────────

    async def run(self, user_message: str, images: list[str] | None = None) -> str:
        """
        处理一条用户消息，驱动完整的 Agent 循环。
        支持 reboot/compress 后自动续接 —— Agent 会自主从 execution_state 恢复并继续工作。
        返回 Agent 的最终文本响应。

        参数:
            user_message: 用户文本消息
            images: 可选的 base64 data URL 列表（多模态图片输入）
        """
        self.is_running = True
        self._emit("status", {"state": "running"})

        # 首次对话注入 System Prompt + 苏醒上下文
        if not self.messages:
            self.messages = self._build_initial_messages()

        # 保存原始用户消息，并写入记忆树以保证跨重启不丢失
        self._current_user_message = user_message
        try:
            self.engine.update(
                "execution_state",
                new_content=f"[用户请求] {user_message}\n[状态] 刚接收，待处理",
            )
        except Exception as e:
            logger.warning(f"Failed to save user message to execution_state: {e}")

        # 构建消息内容：纯文本或多模态
        if images:
            content = [{"type": "text", "text": user_message}]
            for img_url in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })
            self.messages.append({"role": "user", "content": content})
            self._emit("message", {"role": "user", "content": user_message, "images": len(images)})
        else:
            self.messages.append({"role": "user", "content": user_message})
            self._emit("message", {"role": "user", "content": user_message})

        final_text = ""
        auto_continue_count = 0

        try:
            while True:
                result = await self._agent_loop()

                # 检查是否是 reboot/compress 触发的自动续接
                if self._auto_continue_flag and auto_continue_count < self.MAX_AUTO_CONTINUES:
                    self._auto_continue_flag = False
                    auto_continue_count += 1
                    self._emit("status", {"state": "auto_continuing", "round": auto_continue_count})
                    logger.info(f"Auto-continue #{auto_continue_count} after reboot/compress")

                    # 注入续接消息：包含原始用户请求 + 记忆上下文已预加载
                    continue_msg = (
                        f"[系统自动续接] 上下文已重启，记忆上下文已重新加载。\n"
                        f"用户的原始请求：{self._current_user_message}\n"
                        f"根据 execution_state 中的进度继续推进。"
                        f"如果任务已全部完成，必须向用户发送最终回复（用中文）。"
                    )
                    self.messages.append({"role": "user", "content": continue_msg})
                    continue
                else:
                    final_text = result
                    break

        except Exception as e:
            error_msg = f"Agent 循环异常: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit("error", {"error": error_msg})
            final_text = f"抱歉，我遇到了一个错误: {e}"
        finally:
            self.is_running = False
            self._auto_continue_flag = False
            self._emit("status", {"state": "idle"})

        return final_text

    _auto_continue_flag = False

    async def run_stream(self, user_message: str, images: list[str] | None = None):
        """
        流式处理用户消息。async generator，yield SSE 事件字典：
        - {"event": "content_delta", "data": {"content": "..."}}
        - {"event": "tool_call", "data": {"name": ..., "arguments": ...}}
        - {"event": "tool_result", "data": {"name": ..., "result": ...}}
        - {"event": "status", "data": {...}}
        - {"event": "done", "data": {"text": "...", "usage": {...}}}
        - {"event": "error", "data": {"error": "..."}}
        """
        self.is_running = True
        self._emit("status", {"state": "running"})

        if not self.messages:
            self.messages = self._build_initial_messages()

        self._current_user_message = user_message
        try:
            self.engine.update(
                "execution_state",
                new_content=f"[用户请求] {user_message}\n[状态] 刚接收，待处理",
            )
        except Exception as e:
            logger.warning(f"Failed to save user message to execution_state: {e}")

        if images:
            content = [{"type": "text", "text": user_message}]
            for img_url in images:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "user", "content": user_message})

        yield {"event": "status", "data": {"state": "running"}}

        auto_continue_count = 0
        final_text = ""

        try:
            while True:
                async for event in self._agent_loop_stream():
                    if event.get("event") == "_loop_result":
                        final_text = event["data"]["text"]
                    else:
                        yield event

                if self._auto_continue_flag and auto_continue_count < self.MAX_AUTO_CONTINUES:
                    self._auto_continue_flag = False
                    auto_continue_count += 1
                    yield {"event": "status", "data": {"state": "auto_continuing", "round": auto_continue_count}}
                    continue_msg = (
                        f"[系统自动续接] 上下文已重启，记忆上下文已重新加载。\n"
                        f"用户的原始请求：{self._current_user_message}\n"
                        f"根据 execution_state 中的进度继续推进。"
                        f"如果任务已全部完成，必须向用户发送最终回复（用中文）。"
                    )
                    self.messages.append({"role": "user", "content": continue_msg})
                    continue
                else:
                    break

        except Exception as e:
            error_msg = f"Agent 循环异常: {e}"
            logger.error(error_msg, exc_info=True)
            yield {"event": "error", "data": {"error": error_msg}}
            final_text = f"抱歉，我遇到了一个错误: {e}"
        finally:
            self.is_running = False
            self._auto_continue_flag = False

        yield {"event": "done", "data": {"text": final_text, "total_input": self.total_input_tokens, "total_output": self.total_output_tokens, "total_cost": self._calculate_cost()}}

    async def _agent_loop_stream(self):
        """流式核心循环：yield SSE 事件，最后 yield _loop_result"""
        for round_idx in range(self.MAX_TOOL_ROUNDS):
            per_turn_max = min(self.model_profile.get("max_output", 16_000), 16_000)

            # 流式调用 LLM
            collected_content = ""
            collected_tool_calls = []
            final_usage = None

            def _run_stream():
                return list(self.client.call_with_tools_stream(
                    messages=self.messages,
                    temperature=0.3,
                    tools=TOOL_DEFINITIONS,
                    max_tokens=per_turn_max,
                ))

            try:
                stream_events = await asyncio.to_thread(_run_stream)
            except Exception as api_error:
                logger.error(f"LLM stream API call failed at round {round_idx}: {api_error}", exc_info=True)
                yield {"event": "error", "data": {"error": f"LLM API 调用失败: {api_error}", "round": round_idx}}
                if round_idx >= 2:
                    yield {"event": "_loop_result", "data": {"text": f"LLM API 连续调用失败: {api_error}"}}
                    return
                await asyncio.sleep(min(2 ** round_idx, 8))
                continue

            for evt in stream_events:
                evt_type = evt.get("type", "")
                if evt_type == "content_delta":
                    collected_content += evt["content"]
                    yield {"event": "content_delta", "data": {"content": evt["content"]}}
                elif evt_type == "tool_call_delta":
                    yield {"event": "tool_call_delta", "data": evt}
                elif evt_type == "done":
                    final_usage = evt.get("usage")
                    resp = evt.get("response")
                    if resp:
                        collected_content = resp.text
                        collected_tool_calls = resp.tool_calls

            # Build LLMResponse from streamed data
            if final_usage is None:
                from portable_llm import LLMUsage
                final_usage = LLMUsage()

            response = LLMResponse(
                text=collected_content,
                usage=final_usage,
                tool_calls=collected_tool_calls,
                raw_message=None,
            )

            # Token 统计
            usage = response.usage
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.current_context_tokens = usage.prompt_tokens

            yield {"event": "status", "data": {
                "context_tokens": self.current_context_tokens,
                "output_tokens": usage.completion_tokens,
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total_cost": self._calculate_cost(),
                "warn_at": self.token_warn_threshold,
                "reboot_at": self.token_reboot_threshold,
            }}

            # Token warnings / forced reboot (same logic as _agent_loop)
            if self.current_context_tokens > self.token_warn_threshold and not self._token_warning_injected:
                self._inject_token_warning()
                self._token_warning_injected = True

            if self.current_context_tokens > self.token_reboot_threshold:
                logger.warning(f"Context tokens {self.current_context_tokens} exceeded reboot threshold, forcing reboot")
                try:
                    progress_note = f"[系统自动存档] 上下文达到 {self.current_context_tokens} prompt_tokens，触发强制重启。"
                    self.engine.update("execution_state", new_content=progress_note, append_content=True)
                except Exception as e:
                    logger.error(f"Auto-save execution_state failed: {e}")
                self._reboot_flag = False
                self._auto_continue_flag = True
                self.messages = self._build_initial_messages()
                self.current_context_tokens = 0
                self._token_warning_injected = False
                self._recent_tool_calls.clear()
                self.reboot_count += 1
                yield {"event": "reboot", "data": {"reason": "forced_token_limit", "reboot_count": self.reboot_count}}
                yield {"event": "_loop_result", "data": {"text": "[上下文已强制重启]"}}
                return

            # 没有 tool_calls → 最终回复
            if not response.tool_calls:
                text = response.text or ""
                self.messages.append({"role": "assistant", "content": text})
                self._emit("message", {"role": "assistant", "content": text})
                yield {"event": "_loop_result", "data": {"text": text}}
                return

            # 有 tool_calls → 安全标准化 + 执行
            normalized_tool_calls = []
            for tc in response.tool_calls:
                ntc = self._normalize_tool_call(tc)
                if ntc is not None:
                    normalized_tool_calls.append(ntc)

            # 如果所有 tool_calls 都无效，当作纯文本回复 + 注入格式提示
            if not normalized_tool_calls:
                text = response.text or "[LLM 返回了无效的工具调用格式]"
                self.messages.append({"role": "assistant", "content": text})
                self._emit("message", {"role": "assistant", "content": text})
                self.messages.append({"role": "system", "content": "⚠️ 你上一次的工具调用格式无效（函数名为空或结构残缺）。请使用正确的工具名称和参数格式重试。"})
                continue

            assistant_msg = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": normalized_tool_calls,
            }
            self.messages.append(assistant_msg)

            if response.text:
                self._emit("thinking", {"content": response.text})

            # 执行工具
            tool_results = await self._execute_tools_concurrent(normalized_tool_calls)

            for tc, result_str in zip(normalized_tool_calls, tool_results):
                func_name = tc["function"]["name"]
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{func_name}"),
                    "content": result_str,
                })
                yield {"event": "tool_result", "data": {"name": func_name, "result": result_str[:2000]}}

            # 循环检测 (same as _agent_loop)
            for tc in normalized_tool_calls:
                self._recent_tool_calls.append(tc["function"]["name"])
            self._recent_tool_calls = self._recent_tool_calls[-self._loop_detect_window:]
            if len(self._recent_tool_calls) >= self._loop_detect_window:
                from collections import Counter
                counter = Counter(self._recent_tool_calls)
                most_common_name, most_common_count = counter.most_common(1)[0]
                if most_common_count >= self._loop_detect_threshold:
                    loop_warning = (
                        f"⚠️ 循环检测警告：最近 {self._loop_detect_window} 次工具调用中，"
                        f"'{most_common_name}' 被调用了 {most_common_count} 次。\n"
                        f"你可能陷入了无效循环。请停止重复同一操作。"
                    )
                    self.messages.append({"role": "system", "content": loop_warning})
                    yield {"event": "status", "data": {"warning": "loop_detected", "tool": most_common_name}}
                    if most_common_count >= self._loop_detect_threshold + 2:
                        yield {"event": "_loop_result", "data": {"text": f"[系统强制终止] 检测到工具 '{most_common_name}' 无限循环调用。"}}
                        return

            if self._check_reboot_flag():
                if not self.messages:
                    self.messages = self._build_initial_messages()
                yield {"event": "_loop_result", "data": {"text": "[上下文已重启]"}}
                return

            if self._check_ask_human_flag():
                yield {"event": "_loop_result", "data": {"text": f"[等待人类回复] {self._ask_human_question}"}}
                return

        yield {"event": "_loop_result", "data": {"text": "达到最大工具调用轮次限制，请缩小任务范围。"}}

    async def _agent_loop(self) -> str:
        """核心循环：调用 LLM → 处理 tool_calls → 反馈结果 → 重复"""
        for round_idx in range(self.MAX_TOOL_ROUNDS):
            # LLM 调用 — 在线程中执行 (non-blocking)
            # max_tokens per turn: 用 max_output 但上限 16K (单次 API 调用无需更多)
            per_turn_max = min(self.model_profile.get("max_output", 16_000), 16_000)
            try:
                response: LLMResponse = await asyncio.to_thread(
                    self.client.call_with_tools,
                    messages=self.messages,
                    temperature=0.3,
                    tools=TOOL_DEFINITIONS,
                    max_tokens=per_turn_max,
                )
            except Exception as api_error:
                logger.error(f"LLM API call failed at round {round_idx}: {api_error}", exc_info=True)
                self._emit("error", {"error": f"LLM API 调用失败: {api_error}", "round": round_idx})
                if round_idx >= 2:  # 连续失败则放弃
                    return f"LLM API 连续调用失败，无法继续。最后错误: {api_error}"
                await asyncio.sleep(min(2 ** round_idx, 8))
                continue

            # Token 统计 (区分 input/output)
            usage = response.usage
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.current_context_tokens = usage.prompt_tokens  # 上下文窗口用量 = 最新 prompt_tokens
            self._emit("status", {
                "context_tokens": self.current_context_tokens,
                "output_tokens": usage.completion_tokens,
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total_cost": self._calculate_cost(),
                "warn_at": self.token_warn_threshold,
                "reboot_at": self.token_reboot_threshold,
            })

            # 动态阈值：当前上下文到达 warn_ratio 时预警
            if self.current_context_tokens > self.token_warn_threshold and not self._token_warning_injected:
                self._inject_token_warning()
                self._token_warning_injected = True

            # 动态阈值：当前上下文到达 reboot_ratio 时强制存档+重启
            if self.current_context_tokens > self.token_reboot_threshold:
                logger.warning(f"Context tokens {self.current_context_tokens} exceeded reboot threshold {self.token_reboot_threshold}, forcing reboot")
                self._emit("status", {"warning": "forced_reboot", "context_tokens": self.current_context_tokens})
                # 直接执行重启，不依赖 Agent 遵循指令
                try:
                    # 尝试保存当前进度到 execution_state
                    progress_note = f"[系统自动存档] 上下文达到 {self.current_context_tokens} prompt_tokens，触发强制重启。累计费用: ¥{self._calculate_cost():.4f}"
                    self.engine.update("execution_state", new_content=progress_note, append_content=True)
                except Exception as e:
                    logger.error(f"Auto-save execution_state failed: {e}")
                # 直接触发重启并清理标志 (不重置累计 tokens，保留费用统计)
                self._reboot_flag = False
                self._auto_continue_flag = True
                self.messages = self._build_initial_messages()
                self.current_context_tokens = 0
                self._token_warning_injected = False
                self._recent_tool_calls.clear()  # 重置循环检测
                self.reboot_count += 1
                self._emit("reboot", {"reason": "forced_token_limit", "reboot_count": self.reboot_count})
                logger.info(f"Forced reboot #{self.reboot_count} at {self.current_context_tokens} tokens")
                return "[上下文已强制重启]"

            # 没有 tool_calls → 最终回复
            if not response.tool_calls:
                text = response.text or ""
                self.messages.append({"role": "assistant", "content": text})
                self._emit("message", {"role": "assistant", "content": text})
                return text

            # 有 tool_calls → 安全标准化 + 执行
            normalized_tool_calls = []
            for tc in response.tool_calls:
                ntc = self._normalize_tool_call(tc)
                if ntc is not None:
                    normalized_tool_calls.append(ntc)

            # 如果所有 tool_calls 都无效，当作纯文本回复 + 注入格式提示
            if not normalized_tool_calls:
                text = response.text or "[LLM 返回了无效的工具调用格式]"
                self.messages.append({"role": "assistant", "content": text})
                self._emit("message", {"role": "assistant", "content": text})
                self.messages.append({"role": "system", "content": "⚠️ 你上一次的工具调用格式无效（函数名为空或结构残缺）。请使用正确的工具名称和参数格式重试。"})
                continue

            assistant_msg = {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": normalized_tool_calls,
            }
            self.messages.append(assistant_msg)

            if response.text:
                self._emit("thinking", {"content": response.text})

            # ── 并发执行工具调用 ──
            # 将同一轮的多个 tool_calls 分组：可并发的放一起执行
            tool_results = await self._execute_tools_concurrent(normalized_tool_calls)

            for tc, result_str in zip(normalized_tool_calls, tool_results):
                func_name = tc["function"]["name"]
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{func_name}"),
                    "content": result_str,
                })

            # ── 循环检测 ──
            for tc in normalized_tool_calls:
                self._recent_tool_calls.append(tc["function"]["name"])
            # 保留最近 N 次调用
            self._recent_tool_calls = self._recent_tool_calls[-self._loop_detect_window:]
            if len(self._recent_tool_calls) >= self._loop_detect_window:
                from collections import Counter
                counter = Counter(self._recent_tool_calls)
                most_common_name, most_common_count = counter.most_common(1)[0]
                if most_common_count >= self._loop_detect_threshold:
                    loop_warning = (
                        f"⚠️ 循环检测警告：最近 {self._loop_detect_window} 次工具调用中，"
                        f"'{most_common_name}' 被调用了 {most_common_count} 次。\n"
                        f"你可能陷入了无效循环。请立即：\n"
                        f"1. 停止重复同一操作\n"
                        f"2. 换一种方法达成目标（如果缺少所需工具，直接告诉用户）\n"
                        f"3. 如任务已完成，向用户发送最终回复\n"
                        f"如果你继续循环，系统将强制终止。"
                    )
                    self.messages.append({"role": "system", "content": loop_warning})
                    self._emit("status", {"warning": "loop_detected", "tool": most_common_name, "count": most_common_count})
                    logger.warning(f"Loop detected: {most_common_name} called {most_common_count}/{self._loop_detect_window} times")
                    # 如果连续超过阈值+2 还没停下来，强制返回
                    if most_common_count >= self._loop_detect_threshold + 2:
                        logger.error(f"Forced termination: {most_common_name} loop exceeded {self._loop_detect_threshold + 2}")
                        self._emit("status", {"warning": "loop_force_terminated"})
                        return f"[系统强制终止] 检测到工具 '{most_common_name}' 无限循环调用，已自动停止。请检查任务逻辑或补充所需工具。"

            # 检查重启标志
            if self._check_reboot_flag():
                # 注入 system prompt + 苏醒上下文为下一轮 auto-continue 做准备
                if not self.messages:
                    self.messages = self._build_initial_messages()
                return "[上下文已重启]"

            # 检查求助标志
            if self._check_ask_human_flag():
                return f"[等待人类回复] {self._ask_human_question}"

        return "达到最大工具调用轮次限制，请缩小任务范围。"

    async def _execute_tools_concurrent(self, tool_calls: list[dict]) -> list[str]:
        """
        并发执行多个工具调用。
        - 同步工具直接执行
        - I/O 密集工具在线程中执行
        - 状态修改工具 (reboot, compress) 在其他工具之后单独执行
        """
        STATE_TOOLS = {"reboot_context", "compress_context"}

        # 分离：普通工具 vs 状态修改工具
        normal_tcs = []
        deferred_tcs = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            if name in STATE_TOOLS:
                deferred_tcs.append(tc)
            else:
                normal_tcs.append(tc)

        async def _exec_one(tc: dict) -> str:
            func_name = tc.get("function", {}).get("name", "unknown")
            raw_args = tc.get("function", {}).get("arguments", "{}")

            # JSON 解析 — 失败时返回明确错误而非静默空字典
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError) as e:
                error_msg = f"参数 JSON 解析失败: {e}。原始参数: {str(raw_args)[:200]}"
                logger.warning(f"Tool {func_name}: {error_msg}")
                result_str = json.dumps({"error": error_msg, "hint": "请检查 arguments 是否为合法 JSON 字符串"}, ensure_ascii=False)
                self._emit("tool_call", {"name": func_name, "arguments": {}, "id": tc.get("id", "")})
                self._emit("tool_result", {"name": func_name, "result": result_str, "id": tc.get("id", "")})
                return result_str

            # 参数类型和别名自动修正
            args = self._coerce_tool_args(func_name, args)

            self._emit("tool_call", {"name": func_name, "arguments": args, "id": tc.get("id", "")})

            result = await self._execute_tool(func_name, args)
            result_str = json.dumps(result, ensure_ascii=False, default=str)

            if len(result_str) > 8000:
                result_str = result_str[:8000] + f"\n...[结果已截断，共 {len(result_str)} 字符]"

            self._emit("tool_result", {"name": func_name, "result": result_str[:2000], "id": tc.get("id", "")})
            return result_str

        # 先并发执行普通工具
        results_map = {}
        if normal_tcs:
            tasks = [_exec_one(tc) for tc in normal_tcs]
            normal_results = await asyncio.gather(*tasks, return_exceptions=True)
            for tc, r in zip(normal_tcs, normal_results):
                tid = tc.get("id", tc["function"]["name"])
                if isinstance(r, Exception):
                    results_map[tid] = json.dumps({"error": str(r)}, ensure_ascii=False)
                else:
                    results_map[tid] = r

        # 再顺序执行状态修改工具
        for tc in deferred_tcs:
            tid = tc.get("id", tc["function"]["name"])
            try:
                r = await _exec_one(tc)
                results_map[tid] = r
            except Exception as e:
                results_map[tid] = json.dumps({"error": str(e)}, ensure_ascii=False)

        # 按原始顺序返回
        final = []
        for tc in tool_calls:
            tid = tc.get("id", tc["function"]["name"])
            final.append(results_map.get(tid, json.dumps({"error": "unknown"}, ensure_ascii=False)))
        return final

    def _calculate_cost(self) -> float:
        """计算累计费用（元）"""
        p = self.model_profile
        return (self.total_input_tokens * p["cost_input"] + self.total_output_tokens * p["cost_output"]) / 1_000_000

    def _inject_token_warning(self):
        pct = self.current_context_tokens / self.model_profile["context_length"] * 100
        warning = (
            f"🚨 上下文已达 {pct:.0f}%（{self.current_context_tokens}/{self.model_profile['context_length']} tokens）\n"
            f"强制重启线: {self.token_reboot_threshold} tokens ({self.model_profile['reboot_ratio']*100:.0f}%)\n\n"
            f"**你的注意力和推理能力正在显著下降。**\n"
            f"你必须立即执行以下步骤（不要跳过任何一步）：\n"
            f"1. 用 update_memory_node 将当前工作进度完整保存到 execution_state（包含：已完成的步骤、下一步计划、关键上下文）\n"
            f"2. 调用 reboot_context() 主动重启上下文\n\n"
            f"**不要忽略此警告继续工作。** 继续下去你会产生更多错误，且接近强制重启线时系统会自动清空上下文，届时未保存的进度将全部丢失。"
        )
        self.messages.append({"role": "system", "content": warning})
        self._emit("status", {"warning": "token_threshold_exceeded", "pct": round(pct, 1)})

    # ── 工具执行分发 (异步) ─────────────────────────────

    async def _execute_tool(self, name: str, args: dict) -> dict:
        # 同步工具
        handler = self._sync_tool_handlers.get(name)
        if handler:
            try:
                return handler(**args)
            except Exception as e:
                logger.error(f"工具 {name} 执行错误: {e}", exc_info=True)
                return {"error": str(e)}

        # I/O 密集工具 — 在线程中执行
        io_handler = self._io_tool_handlers.get(name)
        if io_handler:
            try:
                return await asyncio.to_thread(io_handler, **args)
            except Exception as e:
                logger.error(f"工具 {name} 执行错误: {e}", exc_info=True)
                return {"error": str(e)}

        return {"error": f"未知工具 '{name}'"}

    # ── 记忆工具实现 ────────────────────────────────────

    def _tool_retrieve_context(self, node_name: str | None = None, top_k: int | None = None, top_p: float | None = None) -> dict:
        return self.engine.retrieve_context(node_name=node_name, top_k=top_k, top_p=top_p)

    def _tool_create_memory_node(self, name: str, content: str, parent_name: str | None = None, k_value: float = 0.0, initial_score: float = 1.0, edges: list | None = None) -> dict:
        effective_parent = parent_name or "root"

        # 防护：阻止直接将内容节点挂在 root 下（系统保留节点除外）
        SYSTEM_NODES = {"identity", "global_map", "execution_state",
                        "operational_rules"}
        if effective_parent == "root" and name not in SYSTEM_NODES:
            root = self.engine.db.get_root_node()
            if root:
                children = self.engine.db.get_children(root["id"])
                non_system = [c for c in children if c["name"] not in SYSTEM_NODES]
                if len(non_system) >= 2:
                    child_names = [c["name"] for c in non_system[:10]]
                    return {
                        "error": f"禁止直接挂在 root 下。root 已有非系统子节点: {child_names}。"
                                 f"请将新节点挂到已有的分类目录下，或先创建一个分类目录(如 projects, knowledge)再把内容挂进去。"
                                 f"root 子节点只能是分类目录，不能是具体内容节点。"
                    }

        try:
            return self.engine.create_memory_node(
                name=name, content=content,
                parent_name=effective_parent,
                k_value=k_value, initial_score=initial_score,
                edges=edges,
            )
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                # 同名节点已存在：检查是否同父目录，若是则自动合并
                existing = self.engine.db.get_node_by_name(name)
                if existing:
                    parent_node = self.engine.db.get_node_by_name(effective_parent)
                    parent_id = parent_node["id"] if parent_node else None
                    if existing.get("parent_id") == parent_id:
                        # 同父目录：自动追加合并
                        result = self.engine.update(name, new_content=content, append_content=True)
                        result["_warning"] = (
                            f"节点 '{name}' 已存在于同一父目录下，已自动追加内容。"
                            f"下次请直接使用 update_memory_node(node_name='{name}', new_content=..., append_content=true) 来更新已有节点。"
                        )
                        return result
                    else:
                        return {
                            "error": f"节点 '{name}' 已存在（位于其他父目录下）。请选择不同的名称，或使用 update_memory_node 更新已有节点。"
                        }
            raise

    def _tool_update_memory_node(self, node_name: str, **kwargs) -> dict:
        # 容错已移至 _coerce_tool_args 统一处理
        # 保留此处作为最后防线
        if "content" in kwargs and "new_content" not in kwargs:
            kwargs["new_content"] = kwargs.pop("content")
        if "name" in kwargs and "new_name" not in kwargs and kwargs["name"] != node_name:
            kwargs["new_name"] = kwargs.pop("name")
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

    def _tool_search_memory(self, query: str, limit: int = 20) -> list:
        return self.engine.search_memory(query, limit)

    # ── 工作区工具实现 ──────────────────────────────────

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        return workspace_tools.read_file(path, start_line, end_line)

    def _tool_search_files(self, query: str, path: str | None = None, max_results: int = 30) -> dict:
        return workspace_tools.search_files(query, path, max_results)

    def _tool_list_directory(self, path: str | None = None) -> dict:
        return workspace_tools.list_directory(path)

    def _tool_write_file(self, path: str, content: str = None, **kwargs) -> dict:
        # 容错已部分移至 _coerce_tool_args，此处保留最后防线
        if content is None:
            for alt_key in ("file_content", "text", "data", "file_data"):
                if alt_key in kwargs:
                    content = kwargs[alt_key]
                    break
        if content is None:
            return {"error": "缺少 content 参数。请使用 write_file(path=..., content=...)"}
        return workspace_tools.write_file(path, content)

    def _tool_append_file(self, path: str, content: str) -> dict:
        return workspace_tools.append_file(path, content)

    def _tool_delete_file(self, path: str) -> dict:
        return workspace_tools.delete_file(path)

    def _tool_execute_command(self, command: str, timeout: int = 30) -> dict:
        return workspace_tools.execute_command(command, timeout)

    def _tool_view_image(self, path: str, max_size_mb: float = 5.0) -> dict:
        return workspace_tools.view_image(path, max_size_mb)

    # ── 网络搜索工具 ───────────────────────────────────

    def _tool_web_search(self, query: str, max_results: int = 8) -> dict:
        return web_search(query, max_results)

    def _tool_fetch_webpage(self, url: str, max_chars: int = 8000) -> dict:
        return fetch_webpage(url, max_chars)

    # ── 上下文管理工具 ──────────────────────────────────

    _reboot_flag = False

    def _tool_reboot_context(self) -> dict:
        self._reboot_flag = True
        self._auto_continue_flag = True  # 触发自动续接
        self.messages = []
        self.current_context_tokens = 0
        self._token_warning_injected = False  # 重置 token 警告标记
        self._recent_tool_calls.clear()  # 重置循环检测
        self.reboot_count += 1
        self._emit("reboot", {"reason": "agent_requested", "reboot_count": self.reboot_count})
        logger.info(f"Agent-requested reboot #{self.reboot_count}")
        return {"status": "context_rebooted", "message": "上下文已清空。系统将自动驱动苏醒协议恢复你的工作进度。"}

    def _check_reboot_flag(self) -> bool:
        if self._reboot_flag:
            self._reboot_flag = False
            return True
        return False

    # ── 求助工具 ───────────────────────────────────────

    _ask_human_flag = False
    _ask_human_question: str = ""

    def _tool_ask_human(self, question: str, context: str = "", urgency: str = "medium") -> dict:
        """向人类求助，暂停当前循环等待回复。"""
        self._ask_human_flag = True
        self._ask_human_question = question
        self._emit("ask_human", {
            "question": question,
            "context": context,
            "urgency": urgency,
        })
        logger.info(f"Agent requesting human help (urgency={urgency}): {question[:100]}")
        return {
            "status": "waiting_for_human",
            "message": "已将问题发送给人类。请等待人类回复。当前任务循环将暂停。",
        }

    def _check_ask_human_flag(self) -> bool:
        if self._ask_human_flag:
            self._ask_human_flag = False
            return True
        return False

    def _tool_compress_context(self, future_intent: str) -> dict:
        """调用 Sub-Agent 压缩当前上下文"""
        self._emit("status", {"state": "compressing"})

        conversation_text = "\n".join(
            f"[{m['role']}] {m.get('content', '')[:500]}"
            for m in self.messages[-30:]
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

        compress_response = self.client.generate(
            system_prompt=COMPRESS_CONTEXT_PROMPT,
            user_prompt=compress_input,
            temperature=0.1,
        )

        compressed = compress_response.text

        try:
            self.engine.update("execution_state", new_content=compressed)
        except Exception as e:
            logger.error(f"Failed to update execution_state: {e}")

        self.messages = self._build_initial_messages()
        self.messages.append({"role": "system", "content": f"[上下文压缩摘要]\n{compressed}"})
        self.current_context_tokens = 0
        # 压缩后不需要 reboot_flag，但需要续接
        # （compress 不触发 reboot，LLM 下一轮直接继续）

        self._emit("status", {"state": "compressed", "intent": future_intent})
        return {
            "status": "context_compressed",
            "future_intent": future_intent,
            "summary_length": len(compressed),
        }

    # ── 碎节点整理 ──────────────────────────────────────

    def _tool_consolidate_memory(self) -> dict:
        self._emit("status", {"state": "consolidating"})

        all_leaves = self._collect_leaves("root", depth=0, max_depth=5)
        if len(all_leaves) < 2:
            return {"message": "叶子节点不足，无需整理", "leaves": len(all_leaves)}

        leaf_summaries = "\n".join(
            f"- [{l['name']}] (parent={l.get('parent', '?')}): {l['content'][:100]}"
            for l in all_leaves[:50]
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

        links_created = []
        try:
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

    # ── 后台任务工具 ────────────────────────────────────

    def _tool_run_background_task(self, task_type: str, params: dict | None = None) -> dict:
        """启动一个后台任务"""
        params = params or {}

        if task_type == "consolidate_memory":
            task_id = self.task_queue.submit_sync(
                "碎片记忆整理",
                self._tool_consolidate_memory,
            )
            return {"task_id": task_id, "task_type": task_type, "status": "submitted"}

        elif task_type == "digest_text":
            text = params.get("text", "")
            node_name = params.get("node_name", "digest_result")
            if not text:
                return {"error": "digest_text 需要 params.text"}
            task_id = self.task_queue.submit_sync(
                f"长文本消化: {node_name}",
                self._bg_digest_text,
                text,
                node_name,
            )
            return {"task_id": task_id, "task_type": task_type, "status": "submitted"}

        else:
            return {"error": f"未知任务类型: {task_type}，可选: consolidate_memory, digest_text"}

    def _tool_check_task(self, task_id: str) -> dict:
        record = self.task_queue.get_task(task_id)
        if not record:
            return {"error": f"任务 {task_id} 不存在"}
        result_preview = None
        if record.result is not None:
            preview = json.dumps(record.result, ensure_ascii=False, default=str)
            result_preview = preview[:2000]
        return {
            "task_id": record.id,
            "name": record.name,
            "status": record.status.value,
            "progress": record.progress,
            "elapsed": record.elapsed,
            "error": record.error,
            "result": result_preview,
        }

    def _tool_list_tasks(self) -> dict:
        tasks = self.task_queue.list_tasks()
        return {
            "total": len(tasks),
            "tasks": [
                {
                    "task_id": t.id,
                    "name": t.name,
                    "status": t.status.value,
                    "elapsed": t.elapsed,
                }
                for t in tasks[:20]
            ],
        }

    def _bg_digest_text(self, text: str, node_name: str) -> dict:
        """后台任务：使用 LLM 消化长文本并存入记忆树"""
        # 分块摘要
        chunk_size = 4000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []

        for i, chunk in enumerate(chunks):
            resp = self.client.generate(
                system_prompt="你是文本摘要专家。将以下文本段落压缩为关键要点，保留核心事实和技术细节。",
                user_prompt=f"[第 {i + 1}/{len(chunks)} 段]\n{chunk}",
                temperature=0.1,
            )
            summaries.append(resp.text)

        # 如果有多段，做最终汇总
        if len(summaries) > 1:
            combined = "\n\n---\n\n".join(summaries)
            final_resp = self.client.generate(
                system_prompt="你是文本摘要专家。将以下多段摘要合并为一份连贯的综合摘要。",
                user_prompt=combined,
                temperature=0.1,
            )
            final_summary = final_resp.text
        else:
            final_summary = summaries[0] if summaries else "无内容"

        # 写入记忆树
        result = self.engine.create_memory_node(
            name=node_name,
            content=final_summary,
            parent_name="root",
            k_value=0.01,
        )

        return {"node_name": node_name, "summary_length": len(final_summary), "chunks_processed": len(chunks), **result}

    # ── 自举进化工具 ────────────────────────────────────

    def _tool_inspect_source(self) -> dict:
        """返回自身项目的源代码结构"""
        return bootstrap_engine.get_source_layout()

    def _tool_read_source(self, path: str) -> dict:
        """读取自身项目的源代码文件"""
        try:
            content = bootstrap_engine.read_source_file(path)
            lines = content.count("\n") + 1
            return {"path": path, "lines": lines, "content": content}
        except Exception as e:
            return {"error": str(e)}

    def _tool_propose_evolution(
        self,
        description: str,
        target_file: str,
        modified_content: str,
        reason: str,
        test_commands: list[str] | None = None,
    ) -> dict:
        """提交自举进化提案：沙箱测试 → 评审 → 合并/驳回"""
        import uuid

        self._emit("status", {"state": "evolving", "target": target_file})

        # 读取原始文件内容
        try:
            original_content = bootstrap_engine.read_source_file(target_file)
        except FileNotFoundError:
            original_content = ""  # 新建文件

        # 构建提案
        proposal = bootstrap_engine.EvolutionProposal(
            id=uuid.uuid4().hex[:8],
            description=description,
            target_file=target_file,
            original_content=original_content,
            modified_content=modified_content,
            reason=reason,
        )

        self._emit("evolution", {
            "phase": "proposed",
            "id": proposal.id,
            "description": description,
            "target_file": target_file,
        })

        # 执行完整流程：测试 → 评审
        proposal = bootstrap_engine.execute_evolution(
            proposal,
            self.client,
            test_commands=test_commands,
            max_rejections=self._MAX_EVOLUTION_REJECTIONS,
        )

        self._emit("evolution", {
            "phase": proposal.status.value,
            "id": proposal.id,
            "test_success": proposal.test_results.get("success", False),
            "verdict": proposal.evaluation.get("verdict", "unknown"),
            "scores": proposal.evaluation.get("scores", {}),
        })

        # 处理结果
        if proposal.status == bootstrap_engine.EvolutionStatus.APPROVED:
            # 合并到源码
            merged = bootstrap_engine.apply_approved_proposal(proposal)
            if merged:
                self._emit("evolution", {"phase": "merged", "id": proposal.id, "target_file": target_file})
                # 记录到记忆树
                try:
                    self.engine.create_memory_node(
                        name=f"evolution_{proposal.id}",
                        content=f"[自举进化] {description}\n文件: {target_file}\n原因: {reason}\n评估: {json.dumps(proposal.evaluation, ensure_ascii=False)}",
                        parent_name="execution_state",
                        k_value=0.05,
                    )
                except Exception:
                    pass

                return {
                    "status": "merged",
                    "proposal_id": proposal.id,
                    "message": f"进化提案已通过评审并合并到 {target_file}",
                    "evaluation": proposal.evaluation,
                }
            else:
                return {
                    "status": "merge_failed",
                    "proposal_id": proposal.id,
                    "message": "评审通过但合并失败，已自动回滚",
                }

        elif proposal.status == bootstrap_engine.EvolutionStatus.REJECTED:
            self._evolution_rejection_count += 1

            # 连续驳回超过阈值 → 触发人类求助
            if self._evolution_rejection_count >= self._MAX_EVOLUTION_REJECTIONS:
                self._evolution_rejection_count = 0
                return {
                    "status": "rejected_escalate",
                    "proposal_id": proposal.id,
                    "rejection_count": proposal.rejection_count,
                    "message": f"进化提案已连续被驳回 {self._MAX_EVOLUTION_REJECTIONS} 次。建议调用 ask_human 求助。",
                    "evaluation": proposal.evaluation,
                    "test_results": {
                        "success": proposal.test_results.get("success"),
                        "stdout": proposal.test_results.get("stdout", "")[:2000],
                        "stderr": proposal.test_results.get("stderr", "")[:1000],
                    },
                }
            else:
                return {
                    "status": "rejected",
                    "proposal_id": proposal.id,
                    "rejection_count": self._evolution_rejection_count,
                    "message": f"进化提案被驳回（第 {self._evolution_rejection_count}/{self._MAX_EVOLUTION_REJECTIONS} 次）。请根据反馈修改后重新提交。",
                    "evaluation": proposal.evaluation,
                    "test_results": {
                        "success": proposal.test_results.get("success"),
                        "stdout": proposal.test_results.get("stdout", "")[:2000],
                        "stderr": proposal.test_results.get("stderr", "")[:1000],
                    },
                }

        else:
            return {
                "status": proposal.status.value,
                "proposal_id": proposal.id,
                "message": "提案处于意外状态",
            }

    # ── 对话状态接口 ────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "is_running": self.is_running,
            "context_tokens": self.current_context_tokens,
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
            "total_cost": self._calculate_cost(),
            "message_count": len(self.messages),
            "model": self.client.model_name,
            "background_tasks": len(self.task_queue.list_tasks(TaskStatus.RUNNING)),
        }

    def get_memory_tree(self, node_name: str | None = None) -> dict:
        try:
            return self.engine.retrieve_context(node_name)
        except Exception as e:
            return {"error": str(e)}

    async def shutdown(self):
        """优雅关闭"""
        await self.task_queue.shutdown()

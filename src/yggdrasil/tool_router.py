"""
泛型工具路由 - Dynamic Tool Retrieval & Execution
实现蓝图中的 generic_tool_executor：
  LLM 从记忆树查阅工具说明书 → 通过此路由执行任意注册工具
"""

import json
import importlib
import traceback
from typing import Any, Callable


class ToolRouter:
    """
    泛型工具分发器。
    所有业务工具注册到此路由器中，LLM 通过 generic_tool_executor
    传入 target_tool_name + tool_arguments_json 来调用。
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: dict[str, dict] = {}

    def register(
        self,
        name: str,
        handler: Callable,
        schema: dict | None = None,
    ):
        """
        注册一个工具。

        参数:
            name: 工具名称 (LLM 在说明书中看到的名字)
            handler: 实际执行函数，接收 **kwargs
            schema: 可选的参数 JSON Schema (用于错误提示)
        """
        self._tools[name] = handler
        if schema:
            self._tool_schemas[name] = schema

    def execute(self, target_tool_name: str, tool_arguments_json: str) -> dict:
        """
        泛型工具执行入口。

        参数:
            target_tool_name: 目标工具名称
            tool_arguments_json: JSON 字符串格式的参数

        返回:
            {"success": bool, "result": Any, "error": str | None}
        """
        # 检查工具是否存在
        if target_tool_name not in self._tools:
            available = list(self._tools.keys())
            return {
                "success": False,
                "result": None,
                "error": f"工具 '{target_tool_name}' 未注册。可用工具: {available}",
            }

        # 解析参数 JSON
        try:
            args = json.loads(tool_arguments_json) if tool_arguments_json.strip() else {}
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "result": None,
                "error": (
                    f"参数 JSON 解析失败: {e}\n"
                    f"你传入的原始字符串: {tool_arguments_json!r}\n"
                    f"请检查 JSON 格式是否正确，确保使用双引号、没有多余逗号。"
                ),
            }

        if not isinstance(args, dict):
            return {
                "success": False,
                "result": None,
                "error": "参数必须是一个 JSON 对象 (dict)，而非数组或标量。",
            }

        # 执行工具
        handler = self._tools[target_tool_name]
        try:
            result = handler(**args)
            return {
                "success": True,
                "result": result,
                "error": None,
            }
        except TypeError as e:
            # 参数不匹配 — 提供详细反馈让 LLM 自我纠正
            schema = self._tool_schemas.get(target_tool_name, {})
            return {
                "success": False,
                "result": None,
                "error": (
                    f"工具 '{target_tool_name}' 参数错误: {e}\n"
                    f"工具参数规范: {json.dumps(schema, ensure_ascii=False)}\n"
                    f"请阅读记忆树中的工具说明书，修正参数后重试。"
                ),
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "success": False,
                "result": None,
                "error": (
                    f"工具 '{target_tool_name}' 执行异常: {e}\n"
                    f"堆栈: {tb}\n"
                    f"如果连续两次失败，请停止重试，查阅源码或调用 ask_human 求助。"
                ),
            }

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

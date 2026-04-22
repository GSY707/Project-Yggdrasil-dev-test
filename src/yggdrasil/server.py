"""
MCP Server 入口 - 世界树认知记忆引擎
暴露给 LLM 的工具接口：
  1. create_memory_node   - 创建记忆节点
  2. retrieve_context     - 猴子爬树检索 (explore)
  3. update               - Git 式更新节点
  4. hard_delete          - 物理删除节点
  5. node_history         - 版本历史回溯
  6. generic_tool_executor - 泛型工具路由
"""

import json
import os
from mcp.server.fastmcp import FastMCP
from .memory_engine import MemoryEngine
from .tool_router import ToolRouter
from .database import Database

# ── 初始化 ─────────────────────────────────────────────

db = Database()
engine = MemoryEngine(db)
router = ToolRouter()

mcp = FastMCP(
    "yggdrasil",
    instructions=(
        "你正在使用世界树(Yggdrasil)认知记忆系统。\n"
        "【第一法则】醒来后必须先调用 retrieve_context() (无参数) 阅读根节点，了解「我是谁、我在哪、我要干什么」。\n"
        "【检索法则】不要猜测记忆内容，必须通过 retrieve_context 逐层爬树获取信息。\n"
        "【报错自愈】如果工具连续两次报错，停止重试，去查阅说明书或历史记录。\n"
        "【记忆衰减】创建节点时合理设置 k_value: 0=永久钢印，越大衰减越快。\n"
    ),
)

# 启动时初始化根节点
engine.initialize_root()


# ── MCP 工具定义 ────────────────────────────────────────

@mcp.tool()
def create_memory_node(
    name: str,
    content: str,
    parent_name: str | None = None,
    k_value: float = 0.0,
    initial_score: float = 1.0,
    edges: str = "[]",
) -> str:
    """创建一个新的记忆节点。

    Args:
        name: 节点名称 (必须唯一)
        content: 节点内容文本。不重要的内容可以用文件绝对路径作为指针。
        parent_name: 父节点名称。为空时挂在根节点下。
        k_value: 衰减系数。0=永久记忆(核心钢印), 0.01=缓慢衰减, 0.1=快速衰减, 1.0+=阅后即焚
        initial_score: 初始重要性分数 (默认1.0)
        edges: 关联出边的 JSON 数组，格式: [{"target": "目标节点名", "label": "关系类型"}]
    """
    try:
        edges_list = json.loads(edges) if edges else []
        if parent_name is None:
            parent_name = "root"
        result = engine.create_memory_node(
            name=name,
            content=content,
            parent_name=parent_name,
            k_value=k_value,
            initial_score=initial_score,
            edges=edges_list,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def retrieve_context(
    node_name: str | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """猴子爬树：检索指定节点的全部上下文（内容、关联边、子节点列表）。

    这是你的核心检索工具。从根节点开始逐层下钻，直到找到目标信息。
    输入为空时返回根节点。

    Args:
        node_name: 要检索的节点名称。为空时返回根节点（起点）。
        top_k: 只返回权重最高的 k 个子节点
        top_p: 只返回累积权重占比达 p 的子节点 (0.0~1.0)
    """
    try:
        result = engine.retrieve_context(
            node_name=node_name,
            top_k=top_k,
            top_p=top_p,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def update(
    node_name: str,
    new_name: str | None = None,
    new_parent_name: str | None = None,
    new_content: str | None = None,
    new_k_value: float | None = None,
    new_initial_score: float | None = None,
    new_edges: str | None = None,
    append_content: bool = False,
) -> str:
    """更新一个记忆节点（Git 式：自动保留历史快照）。

    支持修改名称、父节点、内容、衰减系数、边。未传入的参数保持不变。

    Args:
        node_name: 要修改的节点名称
        new_name: 新名称 (不传=不改)
        new_parent_name: 新父节点名称，用于移动节点 (不传=不改)
        new_content: 新内容 (不传=不改)
        new_k_value: 新衰减系数 (不传=不改)
        new_initial_score: 新初始分数 (不传=不改)
        new_edges: 新的出边 JSON 数组 (不传=不改, 传入时替换全部出边)
        append_content: 为 true 时 new_content 追加到现有内容末尾，而非覆盖
    """
    try:
        edges_list = json.loads(new_edges) if new_edges else None
        result = engine.update(
            node_name=node_name,
            new_name=new_name,
            new_parent_name=new_parent_name,
            new_content=new_content,
            new_k_value=new_k_value,
            new_initial_score=new_initial_score,
            new_edges=edges_list,
            append_content=append_content,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def hard_delete(node_name: str) -> str:
    """物理删除节点及其全部子树和关联边。

    用于切除严重幻觉、废弃规则或死链节点。不可逆操作。
    不允许删除根节点。

    Args:
        node_name: 要删除的节点名称
    """
    try:
        result = engine.hard_delete(node_name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def node_history(node_name: str) -> str:
    """查询节点的历史版本快照（时间漫游工具）。

    当你发现当前信息可能有误、或需要回溯之前的方案时使用。

    Args:
        node_name: 要查询历史的节点名称
    """
    try:
        result = engine.node_history(node_name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def generic_tool_executor(
    target_tool_name: str,
    tool_arguments_json: str = "{}",
) -> str:
    """泛型工具执行器 — 通过记忆树中查阅到的工具说明书来调用任意注册工具。

    使用步骤:
    1. 先在记忆树中搜索目标工具的说明书节点
    2. 阅读说明书了解参数格式
    3. 通过本工具传入工具名和参数 JSON 执行

    如果连续两次调用失败，请停止重试，主动查阅源码或调用 ask_human 求助。

    Args:
        target_tool_name: 目标工具名称（从记忆树说明书中获取）
        tool_arguments_json: 传递给目标工具的参数，必须是合法 JSON 字符串
    """
    result = router.execute(target_tool_name, tool_arguments_json)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── 启动入口 ────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
